//! Multi-part flat (non-deep) **multi-level** tiled EXR encoder.
//!
//! Composes the round-78 single-part [`MIPMAP_LEVELS`] encoder
//! ([`crate::encode_exr_tiled_mipmap`]) with the round-192 multi-part
//! flat-tiled envelope ([`crate::encode_exr_multipart_tiled`]). Each
//! part is an independent `type="tiledimage"` MIPMAP pyramid in the
//! ROUND_DOWN tile layout; the file as a whole sets the multipart
//! (0x1000) version-field bit only, with per-part `tiles[tiledesc,
//! level_mode=1]` carrying the multi-level signal.
//!
//! Binary layout:
//!
//! ```text
//! magic(4) | version(4 with multipart=0x1000)
//! | header_0 ... NUL | header_1 ... NUL | NUL          (extra NUL = end-of-headers)
//! | offset_table_0(chunkCount_0×u64) | offset_table_1(...) | ...
//! | chunks: each starts with i32 part_number,
//!           then i32 tx | i32 ty | i32 lvlx | i32 lvly | i32 size | payload[size].
//! ```
//!
//! Per the OpenEXR Technical Introduction (MIPMAP iteration order for
//! INCREASING_Y), each part's tile chunks visit the diagonal of
//! `(lvlx, lvly)` pairs only — levels `0..N-1` with `lvlx == lvly ==
//! level` — and within each level tiles are emitted ty-outer tx-inner
//! (row-major INCREASING_Y). Parts are concatenated in part-order. Each
//! chunk is prefixed by `i32 part_number`, matching the round-192
//! multi-part tiled chunk shape (24 bytes of chunk header).
//!
//! Per-part `chunkCount` equals the sum over the part's pyramid levels
//! of `ceil(level_w / tile_x) * ceil(level_h / tile_y)`.
//!
//! ROUND_DOWN only (the OpenEXR default). RIPMAP_LEVELS in multi-part
//! form is a followup. Compression NONE / ZIP / ZIPS / RLE supported
//! (each tile is independently compressed and falls back to raw if
//! compression doesn't shrink it, mirroring the rest of the crate).
//!
//! Companion reader: [`crate::parse_exr_multipart_tiled_multilevel`].

use crate::decoder::{apply_zip_interleave, apply_zip_predictor, mipmap_level_dim};
use crate::error::{ExrError, Result};
use crate::header::{encode_attribute_value, VersionField};
use crate::mipmap_encoder::{mipmap_level_count_round_down, MipmapLevel};
use crate::types::{
    Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType, EXR_MAGIC,
};

/// One multi-level flat tiled part for
/// [`encode_exr_multipart_tiled_mipmap`]. Carries the part name, tile
/// geometry, channels, and a full ROUND_DOWN MIPMAP pyramid (one
/// [`MipmapLevel`] per level, in level-index order; pyramid length must
/// equal `mipmap_level_count_round_down(level0_w, level0_h)`).
///
/// All channels must use 1×1 sampling (spec requirement for tiled files).
/// Channels must be in alphabetical order and match each pyramid level's
/// plane order.
pub struct MultipartMipmapTiledPart {
    /// Unique non-empty part name.
    pub name: String,
    /// Tile dimensions (both > 0). Edge tiles at every level are stored
    /// at their valid pixel size only.
    pub tile_x: u32,
    pub tile_y: u32,
    /// Channels in alphabetical order. 1×1 sampling required.
    pub channels: Vec<Channel>,
    /// Full ROUND_DOWN mipmap pyramid: pyramid[0] is the full-resolution
    /// level, pyramid[l] has dimensions
    /// `mipmap_level_dim(pyramid[0].width / .height, l, false)`. Each
    /// level's `planes.len()` must equal `channels.len()` and each plane
    /// must be `width * height` long.
    pub pyramid: Vec<MipmapLevel>,
    /// Per-part compression (applied uniformly across the pyramid, per
    /// the single `compression` header attribute).
    pub compression: Compression,
}

/// Encode a multi-part flat (non-deep) multi-level tiled EXR file. Each
/// part is independently validated (unique non-empty name; channels
/// alphabetical with 1×1 sampling; pyramid length and per-level dims
/// match the ROUND_DOWN spec; plane lengths match `level_w * level_h`;
/// compression in {NONE, ZIP, ZIPS, RLE}; tile sizes > 0).
///
/// Tile chunks are emitted in the spec's iteration order per part —
/// levels 0..N-1, ty-outer tx-inner (INCREASING_Y row-major) within each
/// level — then concatenated across parts. Each chunk on disk is
/// `i32 part_number, i32 tx, i32 ty, i32 lvlx, i32 lvly, i32 size,
/// payload[size]` (24 bytes of chunk header).
///
/// Self-roundtrips through [`crate::parse_exr_multipart_tiled_multilevel`].
pub fn encode_exr_multipart_tiled_mipmap(parts: &[MultipartMipmapTiledPart]) -> Result<Vec<u8>> {
    if parts.is_empty() {
        return Err(ExrError::invalid(
            "encode_exr_multipart_tiled_mipmap: at least one part required".to_string(),
        ));
    }
    // ---- Validate every part up front. ----
    for (i, p) in parts.iter().enumerate() {
        if p.name.is_empty() {
            return Err(ExrError::invalid(format!(
                "multi-part mipmap tiled part {i}: empty name"
            )));
        }
        for (j, other) in parts.iter().enumerate() {
            if j != i && other.name == p.name {
                return Err(ExrError::invalid(format!(
                    "duplicate multi-part mipmap tiled part name '{}' (parts {i} and {j})",
                    p.name
                )));
            }
        }
        if p.tile_x == 0 || p.tile_y == 0 {
            return Err(ExrError::invalid(format!(
                "multi-part mipmap tiled part '{}': tile size {}×{} must both be > 0",
                p.name, p.tile_x, p.tile_y
            )));
        }
        if !matches!(
            p.compression,
            Compression::None | Compression::Zip | Compression::Zips | Compression::Rle
        ) {
            return Err(ExrError::unsupported(format!(
                "multi-part mipmap tiled part '{}': compression {:?} \
                 (encoder supports NONE/ZIP/ZIPS/RLE)",
                p.name, p.compression
            )));
        }
        if p.pyramid.is_empty() {
            return Err(ExrError::invalid(format!(
                "multi-part mipmap tiled part '{}': pyramid is empty",
                p.name
            )));
        }
        for win in p.channels.windows(2) {
            if win[0].name >= win[1].name {
                return Err(ExrError::invalid(format!(
                    "multi-part mipmap tiled part '{}': channels not alphabetical: '{}' >= '{}'",
                    p.name, win[0].name, win[1].name
                )));
            }
        }
        for ch in &p.channels {
            if ch.x_sampling != 1 || ch.y_sampling != 1 {
                return Err(ExrError::unsupported(format!(
                    "multi-part mipmap tiled part '{}': sub-sampled channel '{}' \
                     (tiled files require 1×1 sampling)",
                    p.name, ch.name
                )));
            }
        }
        // Validate pyramid: length matches ROUND_DOWN count, per-level
        // dims match spec, per-channel plane length matches level dims.
        let width = p.pyramid[0].width;
        let height = p.pyramid[0].height;
        if width == 0 || height == 0 {
            return Err(ExrError::invalid(format!(
                "multi-part mipmap tiled part '{}': level-0 {}×{} must be > 0",
                p.name, width, height
            )));
        }
        let want_levels = mipmap_level_count_round_down(width, height);
        if p.pyramid.len() as u32 != want_levels {
            return Err(ExrError::invalid(format!(
                "multi-part mipmap tiled part '{}': pyramid has {} levels, expected \
                 {want_levels} for {width}×{height} ROUND_DOWN",
                p.name,
                p.pyramid.len()
            )));
        }
        for (l, lvl) in p.pyramid.iter().enumerate() {
            let want_w = mipmap_level_dim(width, l as u32, false);
            let want_h = mipmap_level_dim(height, l as u32, false);
            if lvl.width != want_w || lvl.height != want_h {
                return Err(ExrError::invalid(format!(
                    "multi-part mipmap tiled part '{}': level {l} is {}×{}, spec requires \
                     {want_w}×{want_h} (ROUND_DOWN)",
                    p.name, lvl.width, lvl.height
                )));
            }
            if lvl.planes.len() != p.channels.len() {
                return Err(ExrError::invalid(format!(
                    "multi-part mipmap tiled part '{}': level {l} has {} planes but {} \
                     channels declared",
                    p.name,
                    lvl.planes.len(),
                    p.channels.len()
                )));
            }
            let need = (lvl.width as usize) * (lvl.height as usize);
            for (ch, plane) in p.channels.iter().zip(lvl.planes.iter()) {
                if plane.len() != need {
                    return Err(ExrError::invalid(format!(
                        "multi-part mipmap tiled part '{}': level {l} channel '{}' plane \
                         length {} != {}*{} = {need}",
                        p.name,
                        ch.name,
                        plane.len(),
                        lvl.width,
                        lvl.height
                    )));
                }
            }
        }
    }

    // ---- Per-part chunk counts (sum over levels of tx_count * ty_count). ----
    let mut chunk_counts: Vec<u32> = Vec::with_capacity(parts.len());
    for p in parts {
        let mut cc: u32 = 0;
        for lvl in &p.pyramid {
            cc += lvl.width.div_ceil(p.tile_x) * lvl.height.div_ceil(p.tile_y);
        }
        chunk_counts.push(cc);
    }

    // ---- Build per-part header byte blocks. ----
    let mut header_byte_blocks: Vec<Vec<u8>> = Vec::with_capacity(parts.len());
    for (i, p) in parts.iter().enumerate() {
        let attrs = build_mipmap_tiled_part_attrs(p, chunk_counts[i] as i32);
        header_byte_blocks.push(encode_part_header_attributes(&attrs));
    }

    // ---- Stitch magic + version + headers + double-NUL terminator. ----
    // multipart bit (0x1000) only. Per-part `tiles[tiledesc,
    // level_mode=1]` + `type="tiledimage"` carry the multi-level
    // tile-ness signal — the `single_tile` (0x200) bit is NOT set, as in
    // the round-192 multi-part flat-tiled writer.
    let version = VersionField::from_u32(2 | 0x1000);
    let mut out: Vec<u8> = Vec::with_capacity(4096);
    out.extend_from_slice(&EXR_MAGIC.to_le_bytes());
    out.extend_from_slice(&version.to_u32().to_le_bytes());
    for hb in &header_byte_blocks {
        out.extend_from_slice(hb);
        out.push(0); // per-part header terminator
    }
    out.push(0); // double-NUL = end-of-all-headers

    // ---- Build per-tile payloads (levels outer, ty-outer tx-inner within level). ----
    struct TilePayload {
        part_idx: u32,
        tx: u32,
        ty: u32,
        lvl: u32,
        payload: Vec<u8>,
    }
    let mut all_tiles: Vec<TilePayload> = Vec::new();
    for (part_idx, p) in parts.iter().enumerate() {
        for (l, lvl) in p.pyramid.iter().enumerate() {
            let lvl_idx = l as u32;
            let txc = lvl.width.div_ceil(p.tile_x);
            let tyc = lvl.height.div_ceil(p.tile_y);
            for ty in 0..tyc {
                for tx in 0..txc {
                    let x0 = tx * p.tile_x;
                    let y0 = ty * p.tile_y;
                    let x1 = (x0 + p.tile_x).min(lvl.width);
                    let y1 = (y0 + p.tile_y).min(lvl.height);
                    let tw = (x1 - x0) as usize;
                    let th = (y1 - y0) as usize;
                    let bpp: usize = p
                        .channels
                        .iter()
                        .map(|c| c.pixel_type.bytes_per_sample())
                        .sum();
                    let mut raw: Vec<u8> = Vec::with_capacity(tw * th * bpp);
                    for line in 0..th {
                        let dst_y = y0 as usize + line;
                        for (ch_idx, ch) in p.channels.iter().enumerate() {
                            let plane = &lvl.planes[ch_idx];
                            for xx in 0..tw {
                                let dst_x = x0 as usize + xx;
                                let v = plane[dst_y * lvl.width as usize + dst_x];
                                match ch.pixel_type {
                                    PixelType::Half => raw.extend_from_slice(
                                        &crate::half::f32_to_half(v).to_le_bytes(),
                                    ),
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
                        }
                    }
                    let payload = compress_tile_payload(raw, p.compression)?;
                    all_tiles.push(TilePayload {
                        part_idx: part_idx as u32,
                        tx,
                        ty,
                        lvl: lvl_idx,
                        payload,
                    });
                }
            }
        }
    }

    // ---- Compute absolute offsets after the concatenated offset tables. ----
    let header_bytes_so_far = out.len();
    let total_chunks: usize = chunk_counts.iter().map(|&c| c as usize).sum();
    let offset_table_bytes = total_chunks * 8;
    let chunks_start = header_bytes_so_far + offset_table_bytes;

    // Per-chunk header on disk: i32 part_number + i32 tx + i32 ty +
    // i32 lvlx + i32 lvly + i32 size = 24 bytes.
    let mut per_part_table: Vec<Vec<u64>> = vec![Vec::new(); parts.len()];
    let mut running = chunks_start;
    for c in &all_tiles {
        per_part_table[c.part_idx as usize].push(running as u64);
        running += 24 + c.payload.len();
    }

    // Emit concatenated offset tables (part 0, part 1, ...).
    for table in &per_part_table {
        for &o in table {
            out.extend_from_slice(&o.to_le_bytes());
        }
    }

    // Emit chunks in the same order they were built. MIPMAP convention:
    // lvlx == lvly == level index (the diagonal of the (lvlx, lvly) grid).
    for c in all_tiles {
        out.extend_from_slice(&(c.part_idx as i32).to_le_bytes());
        out.extend_from_slice(&(c.tx as i32).to_le_bytes());
        out.extend_from_slice(&(c.ty as i32).to_le_bytes());
        out.extend_from_slice(&(c.lvl as i32).to_le_bytes()); // lvlx
        out.extend_from_slice(&(c.lvl as i32).to_le_bytes()); // lvly
        out.extend_from_slice(&(c.payload.len() as i32).to_le_bytes());
        out.extend_from_slice(&c.payload);
    }

    Ok(out)
}

/// Per-part attribute set for a multi-part multi-level flat tiled part:
/// standard required attributes + `name` + `tiles[tiledesc]` (MIPMAP_LEVELS +
/// ROUND_DOWN) + `type[string="tiledimage"]` + `chunkCount`.
fn build_mipmap_tiled_part_attrs(
    part: &MultipartMipmapTiledPart,
    chunk_count: i32,
) -> Vec<Attribute> {
    let width = part.pyramid[0].width;
    let height = part.pyramid[0].height;
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (width - 1) as i32,
        y_max: (height - 1) as i32,
    };
    // tiledesc payload: u32 xSize | u32 ySize | u8 mode (low nibble =
    // level mode, high nibble = round mode). MIPMAP_LEVELS + ROUND_DOWN
    // = 0x01.
    let mut tiledesc = Vec::with_capacity(9);
    tiledesc.extend_from_slice(&part.tile_x.to_le_bytes());
    tiledesc.extend_from_slice(&part.tile_y.to_le_bytes());
    tiledesc.push(0x01);

    vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(part.channels.clone()),
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
            value: AttributeValue::Compression(part.compression),
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
                data: part.name.as_bytes().to_vec(),
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
                // EXR `string` type stores raw bytes — no NUL terminator
                // (length is in the size field).
                type_name: "string".to_string(),
                data: b"tiledimage".to_vec(),
            },
        },
    ]
}

/// Encode one part's attribute table (without the trailing per-part NUL
/// terminator — caller appends).
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

/// Compress one tile's raw byte stream per the spec's "store raw if
/// compression doesn't shrink it" rule. Identical to the
/// [`crate::multipart_tiled_encoder`] helper of the same name — kept
/// private to this module to avoid leaking a wider cross-module
/// dependency.
fn compress_tile_payload(raw: Vec<u8>, compression: Compression) -> Result<Vec<u8>> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::parse_exr_multipart_tiled_multilevel;
    use crate::mipmap_encoder::build_box_filter_pyramid;

    fn make_planes(w: u32, h: u32, salt: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let pixels = (w as usize) * (h as usize);
        let mut a = Vec::with_capacity(pixels);
        let mut b = Vec::with_capacity(pixels);
        let mut g = Vec::with_capacity(pixels);
        let mut r = Vec::with_capacity(pixels);
        for y in 0..h {
            for x in 0..w {
                r.push((x as f32) / (w as f32) + salt);
                g.push((y as f32) / (h as f32));
                b.push(((x ^ y) as f32) * 0.01);
                a.push(1.0);
            }
        }
        (a, b, g, r)
    }

    fn rgba_channels() -> Vec<Channel> {
        vec![
            Channel {
                name: "A".to_string(),
                pixel_type: PixelType::Float,
                p_linear: false,
                x_sampling: 1,
                y_sampling: 1,
            },
            Channel {
                name: "B".to_string(),
                pixel_type: PixelType::Float,
                p_linear: false,
                x_sampling: 1,
                y_sampling: 1,
            },
            Channel {
                name: "G".to_string(),
                pixel_type: PixelType::Float,
                p_linear: false,
                x_sampling: 1,
                y_sampling: 1,
            },
            Channel {
                name: "R".to_string(),
                pixel_type: PixelType::Float,
                p_linear: false,
                x_sampling: 1,
                y_sampling: 1,
            },
        ]
    }

    fn build_part(
        name: &str,
        w: u32,
        h: u32,
        salt: f32,
        comp: Compression,
        tile: u32,
    ) -> MultipartMipmapTiledPart {
        let (a, b, g, r) = make_planes(w, h, salt);
        let pyramid = build_box_filter_pyramid(w, h, &[a, b, g, r]);
        MultipartMipmapTiledPart {
            name: name.to_string(),
            tile_x: tile,
            tile_y: tile,
            channels: rgba_channels(),
            pyramid,
            compression: comp,
        }
    }

    #[test]
    fn mipmap_multipart_two_parts_none_self_roundtrip() {
        let p0 = build_part("p0", 16, 16, 0.0, Compression::None, 8);
        let p1 = build_part("p1", 32, 16, 0.5, Compression::None, 8);
        // Take a snapshot of every level for each part for comparison.
        let p0_levels: Vec<MipmapLevel> = p0.pyramid.clone();
        let p1_levels: Vec<MipmapLevel> = p1.pyramid.clone();
        let parts = vec![p0, p1];
        let bytes = encode_exr_multipart_tiled_mipmap(&parts).unwrap();
        let decoded = parse_exr_multipart_tiled_multilevel(&bytes).unwrap();
        assert_eq!(decoded.len(), 2);
        let expected = [&p0_levels, &p1_levels];
        for (part_img, exp_pyr) in decoded.iter().zip(expected.iter()) {
            assert_eq!(part_img.levels.len(), exp_pyr.len());
            for (got, want) in part_img.levels.iter().zip(exp_pyr.iter()) {
                assert_eq!(got.width, want.width);
                assert_eq!(got.height, want.height);
                for (gp, wp) in got.planes.iter().zip(want.planes.iter()) {
                    assert_eq!(&gp.samples, wp);
                }
            }
        }
    }

    #[test]
    fn mipmap_multipart_three_parts_mixed_compression() {
        let p0 = build_part("none", 12, 9, 0.0, Compression::None, 4);
        let p1 = build_part("zip", 8, 8, 0.25, Compression::Zip, 4);
        let p2 = build_part("rle", 16, 16, 0.75, Compression::Rle, 8);
        let p0_levels = p0.pyramid.clone();
        let p1_levels = p1.pyramid.clone();
        let p2_levels = p2.pyramid.clone();
        let parts = vec![p0, p1, p2];
        let bytes = encode_exr_multipart_tiled_mipmap(&parts).unwrap();
        let decoded = parse_exr_multipart_tiled_multilevel(&bytes).unwrap();
        assert_eq!(decoded.len(), 3);
        let expected = [&p0_levels, &p1_levels, &p2_levels];
        for (part_img, exp_pyr) in decoded.iter().zip(expected.iter()) {
            assert_eq!(part_img.levels.len(), exp_pyr.len());
            for (got, want) in part_img.levels.iter().zip(exp_pyr.iter()) {
                assert_eq!(got.width, want.width);
                assert_eq!(got.height, want.height);
                assert_eq!(got.level_x, got.level_y);
                for (gp, wp) in got.planes.iter().zip(want.planes.iter()) {
                    assert_eq!(&gp.samples, wp);
                }
            }
        }
    }

    #[test]
    fn mipmap_multipart_zips_edge_tiles_self_roundtrip() {
        // Non-power-of-two so ROUND_DOWN produces non-square edge tiles
        // at lower levels (13×9 -> 6×4 -> 3×2 -> 1×1 = 4 levels).
        let p = build_part("edge", 13, 9, 0.0, Compression::Zips, 4);
        let p_levels = p.pyramid.clone();
        let bytes = encode_exr_multipart_tiled_mipmap(&[p]).unwrap();
        let decoded = parse_exr_multipart_tiled_multilevel(&bytes).unwrap();
        assert_eq!(decoded.len(), 1);
        let part_img = &decoded[0];
        assert_eq!(part_img.levels.len(), p_levels.len());
        for (got, want) in part_img.levels.iter().zip(p_levels.iter()) {
            assert_eq!(got.width, want.width);
            assert_eq!(got.height, want.height);
            for (gp, wp) in got.planes.iter().zip(want.planes.iter()) {
                assert_eq!(&gp.samples, wp);
            }
        }
    }

    #[test]
    fn mipmap_multipart_version_field_bits() {
        // multipart bit (0x1000) set; single_tile (0x200) MUST NOT be set
        // — per-part `tiles[tiledesc, level_mode=1]` + `type="tiledimage"`
        // carry the multi-level tile-ness signal in the multi-part
        // discipline.
        let p = build_part("p", 8, 8, 0.0, Compression::None, 4);
        let bytes = encode_exr_multipart_tiled_mipmap(&[p]).unwrap();
        assert_eq!(&bytes[0..4], &EXR_MAGIC.to_le_bytes()[..]);
        let ver = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(ver & 0x1000, 0x1000, "multipart bit must be set");
        assert_eq!(ver & 0x200, 0, "single_tile bit must NOT be set");
        assert_eq!(ver & 0x800, 0, "non_image bit must NOT be set (flat tiled)");
    }

    #[test]
    fn mipmap_multipart_rejects_empty_parts() {
        let err = encode_exr_multipart_tiled_mipmap(&[]).unwrap_err();
        assert!(format!("{err}").contains("at least one part"));
    }

    #[test]
    fn mipmap_multipart_rejects_duplicate_names() {
        let p0 = build_part("dup", 8, 8, 0.0, Compression::None, 4);
        let p1 = build_part("dup", 8, 8, 0.5, Compression::None, 4);
        let err = encode_exr_multipart_tiled_mipmap(&[p0, p1]).unwrap_err();
        assert!(format!("{err}").contains("duplicate"));
    }

    #[test]
    fn mipmap_multipart_rejects_bad_pyramid_length() {
        // Truncate the pyramid to wrong length.
        let mut p = build_part("trunc", 16, 16, 0.0, Compression::None, 8);
        p.pyramid.truncate(1);
        let err = encode_exr_multipart_tiled_mipmap(&[p]).unwrap_err();
        assert!(format!("{err}").contains("expected"));
    }

    #[test]
    fn mipmap_multipart_rejects_unsupported_compression() {
        let mut p = build_part("piz", 8, 8, 0.0, Compression::None, 4);
        p.compression = Compression::Piz;
        let err = encode_exr_multipart_tiled_mipmap(&[p]).unwrap_err();
        assert!(format!("{err}").contains("NONE/ZIP/ZIPS/RLE"));
    }

    #[test]
    fn mipmap_multipart_rejects_subsampled_channels() {
        let mut p = build_part("sub", 8, 8, 0.0, Compression::None, 4);
        p.channels[0].x_sampling = 2;
        let err = encode_exr_multipart_tiled_mipmap(&[p]).unwrap_err();
        assert!(format!("{err}").contains("sub-sampled"));
    }
}
