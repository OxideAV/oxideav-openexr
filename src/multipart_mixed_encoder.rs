//! Multi-part EXR encoder + reader for **mixed** part types — a single
//! file carrying any combination of `type="scanlineimage"` (flat
//! scanline) and `type="tiledimage"` (flat tiled, ONE_LEVEL) parts.
//!
//! Round-232 surface. Until now every multi-part entry point
//! ([`crate::encode_exr_multipart`] / [`crate::parse_exr_multipart`] for
//! flat scanline, [`crate::encode_exr_multipart_tiled`] /
//! [`crate::parse_exr_multipart_tiled`] for flat tiled ONE_LEVEL, and the
//! deep cousins) required every part to share the same type. The format
//! itself does not enforce that — a multi-part file is simply a sequence
//! of independent parts each declaring its own `type`, `chunkCount`, and
//! tile-ness (via the `tiles[tiledesc]` attribute). This module exposes
//! the mixed flat-scanline + flat-tiled case (the two non-deep part types
//! sharing the same version-field bits) behind a single encode + parse
//! pair.
//!
//! Binary layout (mixed flat scanline + flat tiled, version-field bit
//! 0x1000 set; `non_image` 0x800 NOT set, `single_tile` 0x200 NOT set —
//! per-part `type` carries the discrimination):
//!
//! ```text
//! magic(4) | version(4 with multipart=0x1000)
//! | header_0 ... NUL | header_1 ... NUL | NUL          (extra NUL = end-of-headers)
//! | offset_table_0(chunkCount_0×u64) | offset_table_1(...) | ...
//! | chunks: each starts with i32 part_number, then either
//!     scanline: i32 Y | i32 size | payload[size]                   (= 12 + size B)
//!     tiled:    i32 tx | i32 ty | i32 lvlx | i32 lvly | i32 size | payload[size]  (= 24 + size B)
//!   The reader dispatches the chunk-body shape via the part's declared
//!   `type` attribute, exactly as a homogeneous reader would.
//! ```
//!
//! ONE_LEVEL only for tiled parts (multi-level tiled parts in a mixed
//! file is a followup; pure multi-level multi-part files keep using
//! [`crate::parse_exr_multipart_tiled_multilevel`]). Compression NONE /
//! ZIP / ZIPS / RLE per part. Per-part payload layouts are identical
//! to the homogeneous writers — only the dispatch on `type` is new.
//!
//! Companion reader: [`parse_exr_multipart_mixed`]. Self-roundtrips at
//! every supported compression on layouts mixing scanline and tiled
//! parts in arbitrary order.

use crate::decoder::{
    apply_zip_interleave, apply_zip_predictor, extract_required, find_chunk_count, find_part_type,
    scatter_tile_into_planes, subsampled_dim,
};
use crate::error::{ExrError, Result};
use crate::header::{encode_attribute_value, parse_multipart_headers, VersionField};
use crate::image::{ExrImage, ExrPlane};
use crate::tiled::tiledesc_from_attribute;
use crate::types::{
    Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType, EXR_MAGIC,
};

/// One part for [`encode_exr_multipart_mixed`]. The two variants carry
/// the same field-set as [`crate::MultipartScanlinePart`] and
/// [`crate::MultipartTiledPart`] respectively; the writer dispatches
/// on the variant to emit the matching per-part header + chunk shape.
pub enum MultipartMixedPart<'a> {
    /// Flat scanline part (`type="scanlineimage"`). Honours every
    /// rule the single-typed scanline writer does.
    Scanline {
        /// Unique non-empty part name.
        name: String,
        /// dataWindow / displayWindow dimensions (both > 0).
        width: u32,
        height: u32,
        /// Channels in alphabetical order; sub-sampling allowed.
        channels: Vec<Channel>,
        /// One per-channel f32 slice (sub-sampled length per the
        /// channel's `x_sampling` / `y_sampling`).
        planes: Vec<&'a [f32]>,
        compression: Compression,
    },
    /// Flat tiled part (`type="tiledimage"`, ONE_LEVEL + ROUND_DOWN).
    /// Channels must use 1×1 sampling (tiled files require this).
    Tiled {
        /// Unique non-empty part name.
        name: String,
        /// dataWindow / displayWindow dimensions (both > 0).
        width: u32,
        height: u32,
        /// Tile size (both > 0). Edge tiles store only their valid
        /// pixel rectangle.
        tile_x: u32,
        tile_y: u32,
        /// Channels in alphabetical order (1×1 sampling).
        channels: Vec<Channel>,
        /// One `width * height` f32 slice per channel.
        planes: Vec<&'a [f32]>,
        compression: Compression,
    },
}

impl MultipartMixedPart<'_> {
    fn name(&self) -> &str {
        match self {
            Self::Scanline { name, .. } | Self::Tiled { name, .. } => name,
        }
    }
    fn compression(&self) -> Compression {
        match self {
            Self::Scanline { compression, .. } | Self::Tiled { compression, .. } => *compression,
        }
    }
}

/// One image surfaced by [`parse_exr_multipart_mixed`]. Variants mirror
/// the per-part `type` attribute; both wrap a fully-decoded [`ExrImage`].
#[derive(Debug, Clone)]
pub enum MultipartMixedImage {
    Scanline(ExrImage),
    Tiled(ExrImage),
}

impl MultipartMixedImage {
    /// Borrow the underlying decoded image regardless of part type.
    pub fn image(&self) -> &ExrImage {
        match self {
            Self::Scanline(img) | Self::Tiled(img) => img,
        }
    }
    /// Consume and return the underlying decoded image.
    pub fn into_image(self) -> ExrImage {
        match self {
            Self::Scanline(img) | Self::Tiled(img) => img,
        }
    }
    /// True for flat scanline parts (`type="scanlineimage"`).
    pub fn is_scanline(&self) -> bool {
        matches!(self, Self::Scanline(_))
    }
    /// True for flat tiled parts (`type="tiledimage"`).
    pub fn is_tiled(&self) -> bool {
        matches!(self, Self::Tiled(_))
    }
}

/// Encode a multi-part EXR file whose parts may freely mix
/// `type="scanlineimage"` and `type="tiledimage"`. Validation, attribute
/// layout, and chunk-body emission per part mirror the homogeneous
/// writers exactly; this entry only adds the dispatch on
/// [`MultipartMixedPart`] variant.
///
/// Self-roundtrips through [`parse_exr_multipart_mixed`].
pub fn encode_exr_multipart_mixed(parts: &[MultipartMixedPart]) -> Result<Vec<u8>> {
    if parts.is_empty() {
        return Err(ExrError::invalid(
            "encode_exr_multipart_mixed: at least one part required".to_string(),
        ));
    }

    // ---- Per-part validation (shape rules identical to homogeneous writers). ----
    for (i, p) in parts.iter().enumerate() {
        let name = p.name();
        if name.is_empty() {
            return Err(ExrError::invalid(format!(
                "mixed multi-part part {i}: empty name"
            )));
        }
        for (j, other) in parts.iter().enumerate() {
            if j != i && other.name() == name {
                return Err(ExrError::invalid(format!(
                    "duplicate mixed multi-part name '{name}' (parts {i} and {j})"
                )));
            }
        }
        if !matches!(
            p.compression(),
            Compression::None | Compression::Zip | Compression::Zips | Compression::Rle
        ) {
            return Err(ExrError::unsupported(format!(
                "mixed multi-part part '{name}': compression {:?} \
                 (encoder supports NONE/ZIP/ZIPS/RLE)",
                p.compression()
            )));
        }
        match p {
            MultipartMixedPart::Scanline {
                width,
                height,
                channels,
                planes,
                ..
            } => {
                if *width == 0 || *height == 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part part '{name}': dataWindow {width}×{height} must be > 0"
                    )));
                }
                if channels.len() != planes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part part '{name}': channels.len()={} != planes.len()={}",
                        channels.len(),
                        planes.len()
                    )));
                }
                for win in channels.windows(2) {
                    if win[0].name >= win[1].name {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part part '{name}': channels not alphabetical: \
                             '{}' >= '{}'",
                            win[0].name, win[1].name
                        )));
                    }
                }
                for (ch, plane) in channels.iter().zip(planes.iter()) {
                    if ch.x_sampling <= 0 || ch.y_sampling <= 0 {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part part '{name}': channel '{}' \
                             x_sampling={} y_sampling={} (must be positive)",
                            ch.name, ch.x_sampling, ch.y_sampling
                        )));
                    }
                    let pw = subsampled_dim(*width, ch.x_sampling as u32) as usize;
                    let ph = subsampled_dim(*height, ch.y_sampling as u32) as usize;
                    let need = pw * ph;
                    if plane.len() != need {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part part '{name}': channel '{}' plane length {} \
                             != subsampled {pw}×{ph} = {need}",
                            ch.name,
                            plane.len()
                        )));
                    }
                }
            }
            MultipartMixedPart::Tiled {
                width,
                height,
                tile_x,
                tile_y,
                channels,
                planes,
                ..
            } => {
                if *width == 0 || *height == 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part part '{name}': dataWindow {width}×{height} must be > 0"
                    )));
                }
                if *tile_x == 0 || *tile_y == 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part part '{name}': tile size {tile_x}×{tile_y} \
                         must both be > 0"
                    )));
                }
                if channels.len() != planes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part part '{name}': channels.len()={} != planes.len()={}",
                        channels.len(),
                        planes.len()
                    )));
                }
                for win in channels.windows(2) {
                    if win[0].name >= win[1].name {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part part '{name}': channels not alphabetical: \
                             '{}' >= '{}'",
                            win[0].name, win[1].name
                        )));
                    }
                }
                for (ch, plane) in channels.iter().zip(planes.iter()) {
                    if ch.x_sampling != 1 || ch.y_sampling != 1 {
                        return Err(ExrError::unsupported(format!(
                            "mixed multi-part part '{name}': sub-sampled channel '{}' \
                             (tiled parts require 1×1 sampling)",
                            ch.name
                        )));
                    }
                    let need = (*width as usize) * (*height as usize);
                    if plane.len() != need {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part part '{name}': channel '{}' plane length {} \
                             != width*height = {need}",
                            ch.name,
                            plane.len()
                        )));
                    }
                }
            }
        }
    }

    // ---- Per-part chunk counts. ----
    let mut chunk_counts: Vec<u32> = Vec::with_capacity(parts.len());
    for p in parts {
        let cc = match p {
            MultipartMixedPart::Scanline {
                height,
                compression,
                ..
            } => {
                let block_h = compression.scanlines_per_block();
                height.div_ceil(block_h)
            }
            MultipartMixedPart::Tiled {
                width,
                height,
                tile_x,
                tile_y,
                ..
            } => width.div_ceil(*tile_x) * height.div_ceil(*tile_y),
        };
        chunk_counts.push(cc);
    }

    // ---- Per-part header byte blocks. ----
    let mut header_byte_blocks: Vec<Vec<u8>> = Vec::with_capacity(parts.len());
    for (i, p) in parts.iter().enumerate() {
        let attrs = match p {
            MultipartMixedPart::Scanline {
                name,
                width,
                height,
                channels,
                compression,
                ..
            } => build_scanline_part_attrs(
                name,
                *width,
                *height,
                channels,
                *compression,
                chunk_counts[i] as i32,
            ),
            MultipartMixedPart::Tiled {
                name,
                width,
                height,
                tile_x,
                tile_y,
                channels,
                compression,
                ..
            } => build_tiled_part_attrs(
                name,
                *width,
                *height,
                *tile_x,
                *tile_y,
                channels,
                *compression,
                chunk_counts[i] as i32,
            ),
        };
        header_byte_blocks.push(encode_part_header_attributes(&attrs));
    }

    // ---- Stitch magic + version + headers + double-NUL. ----
    // multipart (0x1000) only — `non_image` is for deep parts; `single_tile`
    // is never set on multi-part files (the per-part `tiles[tiledesc]`
    // attribute + `type` string carry the tile-ness signal).
    let version = VersionField::from_u32(2 | 0x1000);
    let mut out: Vec<u8> = Vec::with_capacity(2048);
    out.extend_from_slice(&EXR_MAGIC.to_le_bytes());
    out.extend_from_slice(&version.to_u32().to_le_bytes());
    for hb in &header_byte_blocks {
        out.extend_from_slice(hb);
        out.push(0); // per-part header terminator
    }
    out.push(0); // double-NUL = end-of-all-headers

    // ---- Build per-part chunk payloads. ----
    // Two records per part keep flat-scanline blocks and flat-tile
    // payloads in distinct shapes so the offset / emission pass can
    // dispatch per part.
    enum ChunkPayload {
        Scanline { y: u32, payload: Vec<u8> },
        Tile { tx: u32, ty: u32, payload: Vec<u8> },
    }
    let mut all_chunks: Vec<(u32, ChunkPayload)> = Vec::new();
    for (part_idx, p) in parts.iter().enumerate() {
        match p {
            MultipartMixedPart::Scanline {
                width,
                height,
                channels,
                planes,
                compression,
                ..
            } => {
                let block_h = compression.scanlines_per_block();
                let cc = chunk_counts[part_idx] as usize;
                for block_idx in 0..cc {
                    let row0 = block_idx as u32 * block_h;
                    let lines_in_block = (height - row0).min(block_h) as usize;
                    let mut raw: Vec<u8> = Vec::new();
                    for line in 0..lines_in_block {
                        let y = row0 as usize + line;
                        for (ch_idx, ch) in channels.iter().enumerate() {
                            let ys = ch.y_sampling as u32;
                            if (y as u32) % ys != 0 {
                                continue;
                            }
                            let xs = ch.x_sampling as u32;
                            let pw = subsampled_dim(*width, xs) as usize;
                            let plane_y = y / ys as usize;
                            let plane = planes[ch_idx];
                            for x in 0..pw {
                                let v = plane[plane_y * pw + x];
                                push_pixel(&mut raw, v, ch.pixel_type);
                            }
                        }
                    }
                    let payload = compress_block(raw, *compression)?;
                    all_chunks.push((part_idx as u32, ChunkPayload::Scanline { y: row0, payload }));
                }
            }
            MultipartMixedPart::Tiled {
                width,
                height,
                tile_x,
                tile_y,
                channels,
                planes,
                compression,
                ..
            } => {
                let txc = width.div_ceil(*tile_x);
                let tyc = height.div_ceil(*tile_y);
                for ty in 0..tyc {
                    for tx in 0..txc {
                        let x0 = tx * tile_x;
                        let y0 = ty * tile_y;
                        let x1 = (x0 + tile_x).min(*width);
                        let y1 = (y0 + tile_y).min(*height);
                        let tw = (x1 - x0) as usize;
                        let th = (y1 - y0) as usize;
                        let mut raw: Vec<u8> = Vec::new();
                        for line in 0..th {
                            let dst_y = y0 as usize + line;
                            for (ch_idx, ch) in channels.iter().enumerate() {
                                let plane = planes[ch_idx];
                                for xx in 0..tw {
                                    let dst_x = x0 as usize + xx;
                                    let v = plane[dst_y * (*width as usize) + dst_x];
                                    push_pixel(&mut raw, v, ch.pixel_type);
                                }
                            }
                        }
                        let payload = compress_block(raw, *compression)?;
                        all_chunks.push((part_idx as u32, ChunkPayload::Tile { tx, ty, payload }));
                    }
                }
            }
        }
    }

    // ---- Compute absolute chunk offsets after concatenated offset tables. ----
    let header_bytes_so_far = out.len();
    let total_chunks: usize = chunk_counts.iter().map(|&c| c as usize).sum();
    let offset_table_bytes = total_chunks * 8;
    let chunks_start = header_bytes_so_far + offset_table_bytes;

    let mut per_part_table: Vec<Vec<u64>> = vec![Vec::new(); parts.len()];
    let mut running = chunks_start;
    for (pi, payload) in &all_chunks {
        per_part_table[*pi as usize].push(running as u64);
        // Scanline chunk on disk: i32 part + i32 Y + i32 size = 12 B + payload
        // Tile chunk on disk:     i32 part + i32 tx + i32 ty + i32 lvlx + i32 lvly + i32 size = 24 B + payload
        running += match payload {
            ChunkPayload::Scanline { payload, .. } => 12 + payload.len(),
            ChunkPayload::Tile { payload, .. } => 24 + payload.len(),
        };
    }

    // Concatenated offset tables, part-order.
    for table in &per_part_table {
        for &o in table {
            out.extend_from_slice(&o.to_le_bytes());
        }
    }

    // Emit chunks in the same flat order they were built.
    for (part_idx, payload) in all_chunks {
        out.extend_from_slice(&(part_idx as i32).to_le_bytes());
        match payload {
            ChunkPayload::Scanline { y, payload } => {
                out.extend_from_slice(&(y as i32).to_le_bytes());
                out.extend_from_slice(&(payload.len() as i32).to_le_bytes());
                out.extend_from_slice(&payload);
            }
            ChunkPayload::Tile { tx, ty, payload } => {
                out.extend_from_slice(&(tx as i32).to_le_bytes());
                out.extend_from_slice(&(ty as i32).to_le_bytes());
                out.extend_from_slice(&0i32.to_le_bytes()); // lvlx
                out.extend_from_slice(&0i32.to_le_bytes()); // lvly
                out.extend_from_slice(&(payload.len() as i32).to_le_bytes());
                out.extend_from_slice(&payload);
            }
        }
    }

    Ok(out)
}

/// Parse a multi-part EXR whose parts may freely mix
/// `type="scanlineimage"` and `type="tiledimage"` (ONE_LEVEL).
///
/// Companion to [`encode_exr_multipart_mixed`]. Deep parts
/// (`type="deepscanline"` / `type="deeptile"`) and multi-level tiled
/// parts (`tiles[tiledesc].level_mode != 0`) are rejected with pointers
/// at the dedicated entries — call [`crate::parse_exr_deep_multipart`],
/// [`crate::parse_exr_multipart_deep_tiled`], or
/// [`crate::parse_exr_multipart_tiled_multilevel`] for those shapes.
///
/// Like the other multi-part readers we walk chunks by linear scan
/// rather than offset-table lookup so that zero-filled tables produced
/// by some reference flows still decode correctly.
pub fn parse_exr_multipart_mixed(bytes: &[u8]) -> Result<Vec<MultipartMixedImage>> {
    let parts = parse_multipart_headers(bytes)?;
    if parts.is_empty() {
        return Err(ExrError::invalid(
            "mixed multi-part file has no parts".to_string(),
        ));
    }
    if parts[0].version.non_image {
        return Err(ExrError::invalid(
            "parse_exr_multipart_mixed called on a deep (non_image bit set) file \
             — call parse_exr_deep_multipart() / parse_exr_multipart_deep_tiled() \
             for deep parts"
                .to_string(),
        ));
    }

    // Classify each part by its declared `type` attribute. Reject deep
    // types and multi-level tiled parts up front.
    #[derive(Clone, Copy, Debug)]
    enum PartKind {
        Scanline,
        Tiled,
    }
    let mut part_kinds: Vec<PartKind> = Vec::with_capacity(parts.len());
    for (i, part) in parts.iter().enumerate() {
        let part_type = find_part_type(&part.attributes).ok_or_else(|| {
            ExrError::invalid(format!(
                "mixed multi-part: part {i} missing required 'type' attribute"
            ))
        })?;
        match part_type.as_str() {
            "scanlineimage" => part_kinds.push(PartKind::Scanline),
            "tiledimage" => {
                let tdesc_attr = part
                    .attributes
                    .iter()
                    .find(|a| a.name == "tiles")
                    .ok_or_else(|| {
                        ExrError::invalid(format!(
                            "mixed multi-part tiled part {i} missing required 'tiles' attribute"
                        ))
                    })?;
                let tdesc = tiledesc_from_attribute(&tdesc_attr.value)?;
                if tdesc.level_mode != 0 {
                    return Err(ExrError::unsupported(format!(
                        "mixed multi-part part {i}: tiledesc level_mode={} \
                         (parse_exr_multipart_mixed only handles ONE_LEVEL tiled \
                         parts — call parse_exr_multipart_tiled_multilevel() for \
                         multi-level tiled multi-part files)",
                        tdesc.level_mode
                    )));
                }
                part_kinds.push(PartKind::Tiled);
            }
            "deepscanline" | "deeptile" => {
                return Err(ExrError::unsupported(format!(
                    "mixed multi-part part {i} type='{part_type}' \
                     — deep parts are not handled by parse_exr_multipart_mixed; \
                     call parse_exr_deep_multipart() or \
                     parse_exr_multipart_deep_tiled() instead"
                )));
            }
            other => {
                return Err(ExrError::unsupported(format!(
                    "mixed multi-part part {i} type='{other}' \
                     (only 'scanlineimage' and 'tiledimage' supported)"
                )));
            }
        }
    }

    // Per-part chunkCount (mandatory in multi-part files).
    let mut chunk_counts: Vec<usize> = Vec::with_capacity(parts.len());
    for (i, part) in parts.iter().enumerate() {
        let cc = find_chunk_count(&part.attributes).ok_or_else(|| {
            ExrError::invalid(format!(
                "mixed multi-part part {i} missing required chunkCount attribute"
            ))
        })?;
        chunk_counts.push(cc);
    }

    // Per-part decode state.
    struct PartState {
        kind: PartKind,
        req: crate::decoder::RequiredAttrs,
        sorted_channels: Vec<Channel>,
        planes: Vec<ExrPlane>,
        tile_x: u32,
        tile_y: u32,
        tx_count: u32,
        ty_count: u32,
    }
    let mut state: Vec<PartState> = Vec::with_capacity(parts.len());
    for (part_idx, part) in parts.iter().enumerate() {
        let req = extract_required(&part.attributes)?;
        if !matches!(
            req.compression,
            Compression::None | Compression::Zip | Compression::Zips | Compression::Rle
        ) {
            return Err(ExrError::unsupported(format!(
                "mixed multi-part part {part_idx}: compression {:?} not yet implemented",
                req.compression
            )));
        }
        let width = req.data_window.width();
        let height = req.data_window.height();
        if width == 0 || height == 0 {
            return Err(ExrError::invalid(format!(
                "mixed multi-part part {part_idx}: dataWindow {width}×{height} must be > 0"
            )));
        }
        let mut sorted_channels = req.channels.clone();
        sorted_channels.sort_by(|a, b| a.name.cmp(&b.name));
        let (tile_x, tile_y, tx_count, ty_count) = match part_kinds[part_idx] {
            PartKind::Scanline => (0u32, 0u32, 0u32, 0u32),
            PartKind::Tiled => {
                for ch in &sorted_channels {
                    if ch.x_sampling != 1 || ch.y_sampling != 1 {
                        return Err(ExrError::unsupported(format!(
                            "mixed multi-part tiled part {part_idx}: sub-sampled channel \
                             '{}' (tiled parts require 1×1 sampling)",
                            ch.name
                        )));
                    }
                }
                let tdesc_attr = part
                    .attributes
                    .iter()
                    .find(|a| a.name == "tiles")
                    .expect("validated above");
                let tdesc = tiledesc_from_attribute(&tdesc_attr.value)?;
                if tdesc.x_size == 0 || tdesc.y_size == 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part tiled part {part_idx}: tile size {}×{} \
                         must both be > 0",
                        tdesc.x_size, tdesc.y_size
                    )));
                }
                let txc = width.div_ceil(tdesc.x_size);
                let tyc = height.div_ceil(tdesc.y_size);
                let expected = (txc as usize) * (tyc as usize);
                if chunk_counts[part_idx] != expected {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part tiled part {part_idx}: chunkCount={} but \
                         tile grid {txc}×{tyc} expects {expected}",
                        chunk_counts[part_idx]
                    )));
                }
                (tdesc.x_size, tdesc.y_size, txc, tyc)
            }
        };
        let planes: Vec<ExrPlane> = sorted_channels
            .iter()
            .map(|c| {
                let pw = subsampled_dim(width, c.x_sampling as u32) as usize;
                let ph = subsampled_dim(height, c.y_sampling as u32) as usize;
                ExrPlane {
                    name: c.name.clone(),
                    samples: vec![0.0; pw * ph],
                }
            })
            .collect();
        state.push(PartState {
            kind: part_kinds[part_idx],
            req,
            sorted_channels,
            planes,
            tile_x,
            tile_y,
            tx_count,
            ty_count,
        });
    }

    // Linear chunk scan — dispatch chunk shape on the part's declared kind.
    let total_chunks: usize = chunk_counts.iter().sum();
    let tables_start = parts.last().unwrap().end_offset;
    let chunk_scan_start = tables_start + total_chunks * 8;
    if chunk_scan_start > bytes.len() {
        return Err(ExrError::invalid(format!(
            "mixed multi-part offset tables run past EOF (need {}, have {})",
            chunk_scan_start,
            bytes.len()
        )));
    }
    let mut scan_pos = chunk_scan_start;
    for _chunk_global_idx in 0..total_chunks {
        if scan_pos + 4 > bytes.len() {
            return Err(ExrError::invalid(format!(
                "mixed multi-part: unexpected EOF at chunk scan position {scan_pos}"
            )));
        }
        let part_num = i32::from_le_bytes(bytes[scan_pos..scan_pos + 4].try_into().unwrap());
        if part_num < 0 || part_num as usize >= parts.len() {
            return Err(ExrError::invalid(format!(
                "mixed multi-part chunk at {scan_pos}: part_number={part_num} out of \
                 range 0..{}",
                parts.len()
            )));
        }
        let part_idx = part_num as usize;
        let ps = &mut state[part_idx];
        match ps.kind {
            PartKind::Scanline => {
                if scan_pos + 12 > bytes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part scanline chunk at {scan_pos}: header runs past EOF"
                    )));
                }
                let y_coord =
                    i32::from_le_bytes(bytes[scan_pos + 4..scan_pos + 8].try_into().unwrap());
                let payload_size =
                    i32::from_le_bytes(bytes[scan_pos + 8..scan_pos + 12].try_into().unwrap());
                if payload_size < 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part scanline chunk at {scan_pos}: \
                         negative payload size {payload_size}"
                    )));
                }
                let pl_start = scan_pos + 12;
                let pl_end = pl_start + payload_size as usize;
                if pl_end > bytes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part scanline chunk at {scan_pos}: payload runs past EOF"
                    )));
                }
                let width = ps.req.data_window.width();
                let height = ps.req.data_window.height();
                let row_in_image = (y_coord - ps.req.data_window.y_min) as i64;
                if row_in_image < 0 || row_in_image as u32 >= height {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part scanline part {part_idx} chunk Y={y_coord} \
                         outside dataWindow"
                    )));
                }
                let block_y0 = row_in_image as u32;
                let block_h = ps.req.compression.scanlines_per_block();
                let lines_in_block = ((height - block_y0).min(block_h)) as usize;
                let uncompressed_size: usize = ps
                    .sorted_channels
                    .iter()
                    .map(|ch| {
                        let ys = ch.y_sampling as u32;
                        let lines = (0..lines_in_block as u32)
                            .filter(|&l| (block_y0 + l) % ys == 0)
                            .count();
                        let xs = ch.x_sampling as u32;
                        let pw = subsampled_dim(width, xs) as usize;
                        lines * ch.pixel_type.bytes_per_sample() * pw
                    })
                    .sum();
                let payload = &bytes[pl_start..pl_end];
                let uncompressed =
                    decompress_block(payload, uncompressed_size, ps.req.compression)?;
                scatter_scanline_block_into_planes(
                    &uncompressed,
                    &ps.sorted_channels,
                    &mut ps.planes,
                    width,
                    block_y0,
                    lines_in_block,
                )?;
                scan_pos = pl_end;
            }
            PartKind::Tiled => {
                if scan_pos + 24 > bytes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part tiled chunk at {scan_pos}: header runs past EOF"
                    )));
                }
                let h_tx =
                    i32::from_le_bytes(bytes[scan_pos + 4..scan_pos + 8].try_into().unwrap());
                let h_ty =
                    i32::from_le_bytes(bytes[scan_pos + 8..scan_pos + 12].try_into().unwrap());
                let lvl_x =
                    i32::from_le_bytes(bytes[scan_pos + 12..scan_pos + 16].try_into().unwrap());
                let lvl_y =
                    i32::from_le_bytes(bytes[scan_pos + 16..scan_pos + 20].try_into().unwrap());
                let payload_size =
                    i32::from_le_bytes(bytes[scan_pos + 20..scan_pos + 24].try_into().unwrap());
                if payload_size < 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part tiled chunk at {scan_pos}: \
                         negative payload size {payload_size}"
                    )));
                }
                if lvl_x != 0 || lvl_y != 0 {
                    return Err(ExrError::unsupported(format!(
                        "mixed multi-part tiled chunk at {scan_pos}: lvlx={lvl_x} \
                         lvly={lvl_y} (parse_exr_multipart_mixed is ONE_LEVEL only)"
                    )));
                }
                let pl_start = scan_pos + 24;
                let pl_end = pl_start + payload_size as usize;
                if pl_end > bytes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part tiled chunk at {scan_pos}: payload runs past EOF"
                    )));
                }
                let width = ps.req.data_window.width();
                let height = ps.req.data_window.height();
                let tx = h_tx as u32;
                let ty = h_ty as u32;
                if tx >= ps.tx_count || ty >= ps.ty_count {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part tiled chunk at {scan_pos}: tile ({tx},{ty}) \
                         out of grid {}×{}",
                        ps.tx_count, ps.ty_count
                    )));
                }
                let x0 = tx * ps.tile_x;
                let y0 = ty * ps.tile_y;
                let x1 = (x0 + ps.tile_x).min(width);
                let y1 = (y0 + ps.tile_y).min(height);
                let tw = (x1 - x0) as usize;
                let th = (y1 - y0) as usize;
                let payload = &bytes[pl_start..pl_end];
                scatter_tile_into_planes(
                    payload,
                    &ps.sorted_channels,
                    &mut ps.planes,
                    width,
                    x0,
                    y0,
                    tw,
                    th,
                    ps.req.compression,
                    (ty * ps.tx_count + tx) as usize,
                )?;
                scan_pos = pl_end;
                let _ = height;
            }
        }
    }

    // Assemble per-part outputs.
    let mut images: Vec<MultipartMixedImage> = Vec::with_capacity(parts.len());
    for (part_idx, part) in parts.iter().enumerate() {
        let PartState {
            kind,
            req,
            sorted_channels,
            planes,
            ..
        } = state.remove(0);
        let img = ExrImage {
            data_window: req.data_window,
            display_window: req.display_window,
            line_order: req.line_order,
            compression: req.compression,
            pixel_aspect_ratio: req.pixel_aspect_ratio,
            screen_window_center: req.screen_window_center,
            screen_window_width: req.screen_window_width,
            channels: sorted_channels,
            planes,
            attributes: part.attributes.clone(),
        };
        let _ = part_idx;
        images.push(match kind {
            PartKind::Scanline => MultipartMixedImage::Scanline(img),
            PartKind::Tiled => MultipartMixedImage::Tiled(img),
        });
    }
    Ok(images)
}

// ---------------- Helpers ----------------

fn push_pixel(raw: &mut Vec<u8>, v: f32, pixel_type: PixelType) {
    match pixel_type {
        PixelType::Half => raw.extend_from_slice(&crate::half::f32_to_half(v).to_le_bytes()),
        PixelType::Float => raw.extend_from_slice(&v.to_le_bytes()),
        PixelType::Uint => {
            let u = if v.is_nan() || v < 0.0 {
                0u32
            } else if v >= (u32::MAX as f32) {
                u32::MAX
            } else {
                (v + 0.5) as u32
            };
            raw.extend_from_slice(&u.to_le_bytes());
        }
    }
}

fn compress_block(raw: Vec<u8>, compression: Compression) -> Result<Vec<u8>> {
    Ok(match compression {
        Compression::None => raw,
        Compression::Zip | Compression::Zips => {
            let mut interleaved = vec![0u8; raw.len()];
            apply_zip_interleave(&raw, &mut interleaved);
            apply_zip_predictor(&mut interleaved);
            let compressed = crate::encoder::zlib_deflate_pub(&interleaved)?;
            if compressed.len() >= raw.len() {
                raw
            } else {
                compressed
            }
        }
        Compression::Rle => {
            let mut interleaved = vec![0u8; raw.len()];
            apply_zip_interleave(&raw, &mut interleaved);
            apply_zip_predictor(&mut interleaved);
            let compressed = crate::rle::rle_compress(&interleaved);
            if compressed.len() >= raw.len() {
                raw
            } else {
                compressed
            }
        }
        _ => unreachable!("filtered above"),
    })
}

/// Decompress one scanline block (mirrors the helper used inside
/// [`crate::parse_exr_multipart`]; the chunk-body decompress + reverse
/// preprocessing pipeline is shared across every flat-scanline reader).
fn decompress_block(
    payload: &[u8],
    uncompressed_size: usize,
    compression: Compression,
) -> Result<Vec<u8>> {
    if matches!(compression, Compression::None) {
        if payload.len() != uncompressed_size {
            return Err(ExrError::invalid(format!(
                "mixed multi-part scanline NONE block: size mismatch (have {}, want {uncompressed_size})",
                payload.len()
            )));
        }
        return Ok(payload.to_vec());
    }
    if payload.len() == uncompressed_size {
        // Encoder may have stored raw bytes when compression didn't shrink.
        return Ok(payload.to_vec());
    }
    let raw = match compression {
        Compression::Zip | Compression::Zips => {
            crate::decoder::zlib_inflate_pub(payload, uncompressed_size)?
        }
        Compression::Rle => crate::rle::rle_decompress(payload, uncompressed_size)?,
        _ => unreachable!("filtered above"),
    };
    if raw.len() != uncompressed_size {
        return Err(ExrError::invalid(format!(
            "mixed multi-part scanline block size mismatch after decode \
             (have {}, want {uncompressed_size})",
            raw.len()
        )));
    }
    Ok(crate::decoder::undo_zip_pipeline_pub(raw))
}

/// Scatter one decompressed scanline-block byte stream into the per-channel
/// f32 planes. Mirrors the private helper inside `decoder.rs`; replicated
/// here so the mixed reader can keep both chunk-body branches local.
fn scatter_scanline_block_into_planes(
    uncompressed: &[u8],
    sorted_channels: &[Channel],
    planes: &mut [ExrPlane],
    width: u32,
    block_y0: u32,
    lines_in_block: usize,
) -> Result<()> {
    let mut p = 0usize;
    for line in 0..lines_in_block {
        let dst_y = block_y0 as usize + line;
        for (ch_idx, ch) in sorted_channels.iter().enumerate() {
            let ys = ch.y_sampling as u32;
            if (dst_y as u32) % ys != 0 {
                continue;
            }
            let xs = ch.x_sampling as u32;
            let pw = subsampled_dim(width, xs) as usize;
            let dst_y_sub = dst_y / ys as usize;
            let plane = &mut planes[ch_idx].samples;
            for x in 0..pw {
                let v = match ch.pixel_type {
                    PixelType::Half => {
                        let bits = u16::from_le_bytes(uncompressed[p..p + 2].try_into().unwrap());
                        crate::half::half_to_f32(bits)
                    }
                    PixelType::Float => {
                        f32::from_le_bytes(uncompressed[p..p + 4].try_into().unwrap())
                    }
                    PixelType::Uint => {
                        let bits = u32::from_le_bytes(uncompressed[p..p + 4].try_into().unwrap());
                        bits as f32
                    }
                };
                plane[dst_y_sub * pw + x] = v;
                p += ch.pixel_type.bytes_per_sample();
            }
        }
    }
    if p != uncompressed.len() {
        return Err(ExrError::invalid(format!(
            "mixed multi-part scanline block consumed {p} of {} payload bytes",
            uncompressed.len()
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_scanline_part_attrs(
    name: &str,
    width: u32,
    height: u32,
    channels: &[Channel],
    compression: Compression,
    chunk_count: i32,
) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (width - 1) as i32,
        y_max: (height - 1) as i32,
    };
    vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(channels.to_vec()),
        },
        Attribute {
            name: "chunkCount".to_string(),
            value: AttributeValue::Other {
                type_name: "int".to_string(),
                data: chunk_count.to_le_bytes().to_vec(),
            },
        },
        Attribute {
            name: "compression".to_string(),
            value: AttributeValue::Compression(compression),
        },
        Attribute {
            name: "dataWindow".to_string(),
            value: AttributeValue::Box2i(win),
        },
        Attribute {
            name: "displayWindow".to_string(),
            value: AttributeValue::Box2i(win),
        },
        Attribute {
            name: "lineOrder".to_string(),
            value: AttributeValue::LineOrder(LineOrder::IncreasingY),
        },
        Attribute {
            name: "name".to_string(),
            value: AttributeValue::Other {
                type_name: "string".to_string(),
                data: name.as_bytes().to_vec(),
            },
        },
        Attribute {
            name: "pixelAspectRatio".to_string(),
            value: AttributeValue::Float(1.0),
        },
        Attribute {
            name: "screenWindowCenter".to_string(),
            value: AttributeValue::V2f(0.0, 0.0),
        },
        Attribute {
            name: "screenWindowWidth".to_string(),
            value: AttributeValue::Float(1.0),
        },
        Attribute {
            name: "type".to_string(),
            value: AttributeValue::Other {
                type_name: "string".to_string(),
                data: b"scanlineimage".to_vec(),
            },
        },
    ]
}

#[allow(clippy::too_many_arguments)]
fn build_tiled_part_attrs(
    name: &str,
    width: u32,
    height: u32,
    tile_x: u32,
    tile_y: u32,
    channels: &[Channel],
    compression: Compression,
    chunk_count: i32,
) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (width - 1) as i32,
        y_max: (height - 1) as i32,
    };
    let mut tiledesc = Vec::with_capacity(9);
    tiledesc.extend_from_slice(&tile_x.to_le_bytes());
    tiledesc.extend_from_slice(&tile_y.to_le_bytes());
    tiledesc.push(0x00); // ONE_LEVEL + ROUND_DOWN

    vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(channels.to_vec()),
        },
        Attribute {
            name: "chunkCount".to_string(),
            value: AttributeValue::Other {
                type_name: "int".to_string(),
                data: chunk_count.to_le_bytes().to_vec(),
            },
        },
        Attribute {
            name: "compression".to_string(),
            value: AttributeValue::Compression(compression),
        },
        Attribute {
            name: "dataWindow".to_string(),
            value: AttributeValue::Box2i(win),
        },
        Attribute {
            name: "displayWindow".to_string(),
            value: AttributeValue::Box2i(win),
        },
        Attribute {
            name: "lineOrder".to_string(),
            value: AttributeValue::LineOrder(LineOrder::IncreasingY),
        },
        Attribute {
            name: "name".to_string(),
            value: AttributeValue::Other {
                type_name: "string".to_string(),
                data: name.as_bytes().to_vec(),
            },
        },
        Attribute {
            name: "pixelAspectRatio".to_string(),
            value: AttributeValue::Float(1.0),
        },
        Attribute {
            name: "screenWindowCenter".to_string(),
            value: AttributeValue::V2f(0.0, 0.0),
        },
        Attribute {
            name: "screenWindowWidth".to_string(),
            value: AttributeValue::Float(1.0),
        },
        Attribute {
            name: "tiles".to_string(),
            value: AttributeValue::Other {
                type_name: "tiledesc".to_string(),
                data: tiledesc,
            },
        },
        Attribute {
            name: "type".to_string(),
            value: AttributeValue::Other {
                type_name: "string".to_string(),
                data: b"tiledimage".to_vec(),
            },
        },
    ]
}

fn encode_part_header_attributes(attrs: &[Attribute]) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    for a in attrs {
        out.extend_from_slice(a.name.as_bytes());
        out.push(0);
        let (type_name, payload) = encode_attribute_value(&a.value);
        out.extend_from_slice(type_name.as_bytes());
        out.push(0);
        out.extend_from_slice(&(payload.len() as i32).to_le_bytes());
        out.extend_from_slice(&payload);
    }
    out
}
