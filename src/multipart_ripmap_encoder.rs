//! Multi-part flat (non-deep) **RIPMAP_LEVELS** tiled EXR encoder.
//!
//! Composes the round-124 single-part RIPMAP_LEVELS encoder
//! ([`crate::encode_exr_tiled_ripmap`]) with the round-192 multi-part
//! flat-tiled envelope ([`crate::encode_exr_multipart_tiled`]). Each
//! part is an independent `type="tiledimage"` RIPMAP grid in the
//! ROUND_DOWN tile layout; the file as a whole sets the multipart
//! (0x1000) version-field bit only, with per-part `tiles[tiledesc,
//! level_mode=2]` carrying the 2-D-reduction-grid signal.
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
//! Within each part, RIPMAP cells are visited in `lvly`-outer
//! `lvlx`-inner order (matching the single-part RIPMAP writer and the
//! decoder's `compute_total_tiles` RIPMAP branch), and within each
//! `(lvlx, lvly)` cell tiles are emitted INCREASING_Y row-major
//! (ty-outer tx-inner). Parts are concatenated in part-order. Each
//! chunk is prefixed by `i32 part_number`, matching the round-192
//! multi-part tiled chunk shape (24 bytes of chunk header).
//!
//! Per-part `chunkCount` equals the sum over the part's `(nx * ny)` grid
//! cells of `ceil(cell_w / tile_x) * ceil(cell_h / tile_y)`.
//!
//! ROUND_DOWN only (the OpenEXR default). Compression NONE / ZIP / ZIPS
//! / RLE supported per part (each tile independently compressed,
//! falling back to raw if compression doesn't shrink it, mirroring the
//! rest of the crate).
//!
//! Companion reader: [`crate::parse_exr_multipart_tiled_multilevel`].

use crate::decoder::{apply_zip_interleave, apply_zip_predictor, mipmap_level_dim};
use crate::error::{ExrError, Result};
use crate::header::{encode_attribute_value, VersionField};
use crate::mipmap_encoder::{ripmap_level_counts_round_down, RipmapPyramid};
use crate::types::{
    Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType, EXR_MAGIC,
};

/// One RIPMAP_LEVELS flat tiled part for
/// [`encode_exr_multipart_tiled_ripmap`]. Carries the part name, tile
/// geometry, channels, and a full ROUND_DOWN 2-D ripmap pyramid.
///
/// All channels must use 1×1 sampling (spec requirement for tiled files).
/// Channels must be in alphabetical order and match each ripmap cell's
/// plane order. The pyramid grid is indexed `grid[lvly][lvlx]` and must
/// be `ripmap_level_counts_round_down(level0_w, level0_h)` shaped.
pub struct MultipartRipmapTiledPart {
    /// Unique non-empty part name.
    pub name: String,
    /// Tile dimensions (both > 0). Edge tiles at every grid cell are
    /// stored at their valid pixel size only.
    pub tile_x: u32,
    pub tile_y: u32,
    /// Channels in alphabetical order. 1×1 sampling required.
    pub channels: Vec<Channel>,
    /// Full ROUND_DOWN ripmap pyramid (2-D grid). Cell `(lvlx, lvly)`
    /// has dimensions `(mipmap_level_dim(w, lvlx), mipmap_level_dim(h,
    /// lvly))`. `grid.len() == ny` and every row's length equals `nx`,
    /// per `ripmap_level_counts_round_down(width, height)`.
    pub pyramid: RipmapPyramid,
    /// Per-part compression (applied uniformly across the grid, per the
    /// single `compression` header attribute).
    pub compression: Compression,
}

/// Encode a multi-part flat (non-deep) RIPMAP_LEVELS tiled EXR file.
/// Each part is independently validated (unique non-empty name; channels
/// alphabetical with 1×1 sampling; grid shape and per-cell dims match
/// the ROUND_DOWN spec; plane lengths match `cell_w * cell_h`;
/// compression in {NONE, ZIP, ZIPS, RLE}; tile sizes > 0).
///
/// Tile chunks are emitted in the spec's iteration order per part —
/// `lvly`-outer `lvlx`-inner across the grid, then ty-outer tx-inner
/// (INCREASING_Y row-major) within each cell — then concatenated across
/// parts. Each chunk on disk is `i32 part_number, i32 tx, i32 ty, i32
/// lvlx, i32 lvly, i32 size, payload[size]` (24 bytes of chunk header).
///
/// Self-roundtrips through [`crate::parse_exr_multipart_tiled_multilevel`].
pub fn encode_exr_multipart_tiled_ripmap(parts: &[MultipartRipmapTiledPart]) -> Result<Vec<u8>> {
    if parts.is_empty() {
        return Err(ExrError::invalid(
            "encode_exr_multipart_tiled_ripmap: at least one part required".to_string(),
        ));
    }
    // ---- Validate every part up front. ----
    for (i, p) in parts.iter().enumerate() {
        if p.name.is_empty() {
            return Err(ExrError::invalid(format!(
                "multi-part ripmap tiled part {i}: empty name"
            )));
        }
        for (j, other) in parts.iter().enumerate() {
            if j != i && other.name == p.name {
                return Err(ExrError::invalid(format!(
                    "duplicate multi-part ripmap tiled part name '{}' (parts {i} and {j})",
                    p.name
                )));
            }
        }
        if p.tile_x == 0 || p.tile_y == 0 {
            return Err(ExrError::invalid(format!(
                "multi-part ripmap tiled part '{}': tile size {}×{} must both be > 0",
                p.name, p.tile_x, p.tile_y
            )));
        }
        if !matches!(
            p.compression,
            Compression::None | Compression::Zip | Compression::Zips | Compression::Rle
        ) {
            return Err(ExrError::unsupported(format!(
                "multi-part ripmap tiled part '{}': compression {:?} \
                 (encoder supports NONE/ZIP/ZIPS/RLE)",
                p.name, p.compression
            )));
        }
        if p.pyramid.grid.is_empty() || p.pyramid.grid[0].is_empty() {
            return Err(ExrError::invalid(format!(
                "multi-part ripmap tiled part '{}': pyramid grid must have at least one cell",
                p.name
            )));
        }
        for win in p.channels.windows(2) {
            if win[0].name >= win[1].name {
                return Err(ExrError::invalid(format!(
                    "multi-part ripmap tiled part '{}': channels not alphabetical: '{}' >= '{}'",
                    p.name, win[0].name, win[1].name
                )));
            }
        }
        for ch in &p.channels {
            if ch.x_sampling != 1 || ch.y_sampling != 1 {
                return Err(ExrError::unsupported(format!(
                    "multi-part ripmap tiled part '{}': sub-sampled channel '{}' \
                     (tiled files require 1×1 sampling)",
                    p.name, ch.name
                )));
            }
        }
        // Validate grid shape + per-cell dims + per-channel plane length.
        let width = p.pyramid.grid[0][0].width;
        let height = p.pyramid.grid[0][0].height;
        if width == 0 || height == 0 {
            return Err(ExrError::invalid(format!(
                "multi-part ripmap tiled part '{}': level-(0,0) {}×{} must be > 0",
                p.name, width, height
            )));
        }
        let (nx, ny) = ripmap_level_counts_round_down(width, height);
        if p.pyramid.grid.len() as u32 != ny {
            return Err(ExrError::invalid(format!(
                "multi-part ripmap tiled part '{}': grid has {} y-levels, expected \
                 {ny} for height {height} ROUND_DOWN",
                p.name,
                p.pyramid.grid.len()
            )));
        }
        for (lvly, row) in p.pyramid.grid.iter().enumerate() {
            if row.len() as u32 != nx {
                return Err(ExrError::invalid(format!(
                    "multi-part ripmap tiled part '{}': grid row {lvly} has {} x-levels, \
                     expected {nx} for width {width} ROUND_DOWN",
                    p.name,
                    row.len()
                )));
            }
            for (lvlx, cell) in row.iter().enumerate() {
                let want_w = mipmap_level_dim(width, lvlx as u32, false);
                let want_h = mipmap_level_dim(height, lvly as u32, false);
                if cell.width != want_w || cell.height != want_h {
                    return Err(ExrError::invalid(format!(
                        "multi-part ripmap tiled part '{}': cell ({lvlx},{lvly}) is {}×{}, \
                         spec requires {want_w}×{want_h} (ROUND_DOWN)",
                        p.name, cell.width, cell.height
                    )));
                }
                if cell.planes.len() != p.channels.len() {
                    return Err(ExrError::invalid(format!(
                        "multi-part ripmap tiled part '{}': cell ({lvlx},{lvly}) has {} planes \
                         but {} channels declared",
                        p.name,
                        cell.planes.len(),
                        p.channels.len()
                    )));
                }
                let need = (cell.width as usize) * (cell.height as usize);
                for (ch, plane) in p.channels.iter().zip(cell.planes.iter()) {
                    if plane.len() != need {
                        return Err(ExrError::invalid(format!(
                            "multi-part ripmap tiled part '{}': cell ({lvlx},{lvly}) channel \
                             '{}' plane length {} != {}*{} = {need}",
                            p.name,
                            ch.name,
                            plane.len(),
                            cell.width,
                            cell.height
                        )));
                    }
                }
            }
        }
    }

    // ---- Per-part chunk counts (sum over grid cells of tx_count * ty_count). ----
    let mut chunk_counts: Vec<u32> = Vec::with_capacity(parts.len());
    for p in parts {
        let mut cc: u32 = 0;
        for row in &p.pyramid.grid {
            for cell in row {
                cc += cell.width.div_ceil(p.tile_x) * cell.height.div_ceil(p.tile_y);
            }
        }
        chunk_counts.push(cc);
    }

    // ---- Build per-part header byte blocks. ----
    let mut header_byte_blocks: Vec<Vec<u8>> = Vec::with_capacity(parts.len());
    for (i, p) in parts.iter().enumerate() {
        let attrs = build_ripmap_tiled_part_attrs(p, chunk_counts[i] as i32);
        header_byte_blocks.push(encode_part_header_attributes(&attrs));
    }

    // ---- Stitch magic + version + headers + double-NUL terminator. ----
    // multipart bit (0x1000) only. Per-part `tiles[tiledesc,
    // level_mode=2]` + `type="tiledimage"` carry the multi-level
    // tile-ness signal — the `single_tile` (0x200) bit is NOT set, as in
    // the round-192 multi-part flat-tiled writer and round-196 MIPMAP
    // multi-part writer.
    let version = VersionField::from_u32(2 | 0x1000);
    let mut out: Vec<u8> = Vec::with_capacity(4096);
    out.extend_from_slice(&EXR_MAGIC.to_le_bytes());
    out.extend_from_slice(&version.to_u32().to_le_bytes());
    for hb in &header_byte_blocks {
        out.extend_from_slice(hb);
        out.push(0); // per-part header terminator
    }
    out.push(0); // double-NUL = end-of-all-headers

    // ---- Build per-tile payloads (lvly-outer lvlx-inner, then ty-outer tx-inner). ----
    struct TilePayload {
        part_idx: u32,
        tx: u32,
        ty: u32,
        lvlx: u32,
        lvly: u32,
        payload: Vec<u8>,
    }
    let mut all_tiles: Vec<TilePayload> = Vec::new();
    for (part_idx, p) in parts.iter().enumerate() {
        for (lvly, row) in p.pyramid.grid.iter().enumerate() {
            for (lvlx, cell) in row.iter().enumerate() {
                let txc = cell.width.div_ceil(p.tile_x);
                let tyc = cell.height.div_ceil(p.tile_y);
                for ty in 0..tyc {
                    for tx in 0..txc {
                        let x0 = tx * p.tile_x;
                        let y0 = ty * p.tile_y;
                        let x1 = (x0 + p.tile_x).min(cell.width);
                        let y1 = (y0 + p.tile_y).min(cell.height);
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
                                let plane = &cell.planes[ch_idx];
                                for xx in 0..tw {
                                    let dst_x = x0 as usize + xx;
                                    let v = plane[dst_y * cell.width as usize + dst_x];
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
                            lvlx: lvlx as u32,
                            lvly: lvly as u32,
                            payload,
                        });
                    }
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

    // Emit chunks in the same order they were built. RIPMAP convention:
    // lvlx and lvly are independent.
    for c in all_tiles {
        out.extend_from_slice(&(c.part_idx as i32).to_le_bytes());
        out.extend_from_slice(&(c.tx as i32).to_le_bytes());
        out.extend_from_slice(&(c.ty as i32).to_le_bytes());
        out.extend_from_slice(&(c.lvlx as i32).to_le_bytes());
        out.extend_from_slice(&(c.lvly as i32).to_le_bytes());
        out.extend_from_slice(&(c.payload.len() as i32).to_le_bytes());
        out.extend_from_slice(&c.payload);
    }

    Ok(out)
}

/// Per-part attribute set for a multi-part ripmap flat tiled part:
/// standard required attributes + `name` + `tiles[tiledesc]` (RIPMAP_LEVELS +
/// ROUND_DOWN) + `type[string="tiledimage"]` + `chunkCount`.
fn build_ripmap_tiled_part_attrs(
    part: &MultipartRipmapTiledPart,
    chunk_count: i32,
) -> Vec<Attribute> {
    let width = part.pyramid.grid[0][0].width;
    let height = part.pyramid.grid[0][0].height;
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (width - 1) as i32,
        y_max: (height - 1) as i32,
    };
    // tiledesc payload: u32 xSize | u32 ySize | u8 mode (low nibble =
    // level mode, high nibble = round mode). RIPMAP_LEVELS + ROUND_DOWN
    // = 0x02.
    let mut tiledesc = Vec::with_capacity(9);
    tiledesc.extend_from_slice(&part.tile_x.to_le_bytes());
    tiledesc.extend_from_slice(&part.tile_y.to_le_bytes());
    tiledesc.push(0x02);

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
/// [`crate::multipart_mipmap_encoder`] and
/// [`crate::multipart_tiled_encoder`] helpers of the same name — kept
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
    use crate::mipmap_encoder::build_box_filter_ripmap;

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
    ) -> MultipartRipmapTiledPart {
        let (a, b, g, r) = make_planes(w, h, salt);
        let pyramid = build_box_filter_ripmap(w, h, &[a, b, g, r]);
        MultipartRipmapTiledPart {
            name: name.to_string(),
            tile_x: tile,
            tile_y: tile,
            channels: rgba_channels(),
            pyramid,
            compression: comp,
        }
    }

    fn snapshot(p: &MultipartRipmapTiledPart) -> Vec<Vec<Vec<Vec<f32>>>> {
        p.pyramid
            .grid
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| cell.planes.clone())
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn assert_decoded_matches(
        decoded_part: &crate::decoder::MultilevelTiledPart,
        expected_grid: &[Vec<Vec<Vec<f32>>>],
        width: u32,
        height: u32,
    ) {
        // For RIPMAP the decoder returns the levels in lvly-outer
        // lvlx-inner order — same as the encoder's emission.
        let (nx, ny) = ripmap_level_counts_round_down(width, height);
        assert_eq!(decoded_part.levels.len() as u32, nx * ny);
        let mut k = 0usize;
        for (ly, row) in expected_grid.iter().enumerate() {
            for (lx, want_cell) in row.iter().enumerate() {
                let lvl = &decoded_part.levels[k];
                assert_eq!(lvl.level_x as usize, lx);
                assert_eq!(lvl.level_y as usize, ly);
                assert_eq!(lvl.planes.len(), want_cell.len());
                for (gp, wp) in lvl.planes.iter().zip(want_cell.iter()) {
                    assert_eq!(&gp.samples, wp);
                }
                k += 1;
            }
        }
    }

    #[test]
    fn ripmap_multipart_two_parts_none_self_roundtrip() {
        // 16×16 → nx=ny=5; 32×16 → nx=6, ny=5 (independent x/y reductions).
        let p0 = build_part("p0", 16, 16, 0.0, Compression::None, 8);
        let p1 = build_part("p1", 32, 16, 0.5, Compression::None, 8);
        let g0 = snapshot(&p0);
        let g1 = snapshot(&p1);
        let parts = vec![p0, p1];
        let bytes = encode_exr_multipart_tiled_ripmap(&parts).unwrap();
        let decoded = parse_exr_multipart_tiled_multilevel(&bytes).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_decoded_matches(&decoded[0], &g0, 16, 16);
        assert_decoded_matches(&decoded[1], &g1, 32, 16);
    }

    #[test]
    fn ripmap_multipart_three_parts_mixed_compression() {
        let p0 = build_part("none", 12, 9, 0.0, Compression::None, 4);
        let p1 = build_part("zip", 8, 8, 0.25, Compression::Zip, 4);
        let p2 = build_part("rle", 16, 16, 0.75, Compression::Rle, 8);
        let g0 = snapshot(&p0);
        let g1 = snapshot(&p1);
        let g2 = snapshot(&p2);
        let parts = vec![p0, p1, p2];
        let bytes = encode_exr_multipart_tiled_ripmap(&parts).unwrap();
        let decoded = parse_exr_multipart_tiled_multilevel(&bytes).unwrap();
        assert_eq!(decoded.len(), 3);
        assert_decoded_matches(&decoded[0], &g0, 12, 9);
        assert_decoded_matches(&decoded[1], &g1, 8, 8);
        assert_decoded_matches(&decoded[2], &g2, 16, 16);
    }

    #[test]
    fn ripmap_multipart_zips_edge_tiles_self_roundtrip() {
        // Non-power-of-two so ROUND_DOWN produces non-square edge tiles
        // at lower levels (13×9 grid is nx=4 ny=4).
        let p = build_part("edge", 13, 9, 0.0, Compression::Zips, 4);
        let g = snapshot(&p);
        let bytes = encode_exr_multipart_tiled_ripmap(&[p]).unwrap();
        let decoded = parse_exr_multipart_tiled_multilevel(&bytes).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_decoded_matches(&decoded[0], &g, 13, 9);
    }

    #[test]
    fn ripmap_multipart_version_field_bits() {
        // multipart bit (0x1000) set; single_tile (0x200) MUST NOT be set
        // — per-part `tiles[tiledesc, level_mode=2]` + `type="tiledimage"`
        // carry the multi-level tile-ness signal in the multi-part
        // discipline.
        let p = build_part("p", 8, 8, 0.0, Compression::None, 4);
        let bytes = encode_exr_multipart_tiled_ripmap(&[p]).unwrap();
        assert_eq!(&bytes[0..4], &EXR_MAGIC.to_le_bytes()[..]);
        let ver = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(ver & 0x1000, 0x1000, "multipart bit must be set");
        assert_eq!(ver & 0x200, 0, "single_tile bit must NOT be set");
        assert_eq!(ver & 0x800, 0, "non_image bit must NOT be set (flat tiled)");
    }

    #[test]
    fn ripmap_multipart_rejects_empty_parts() {
        let err = encode_exr_multipart_tiled_ripmap(&[]).unwrap_err();
        assert!(format!("{err}").contains("at least one part"));
    }

    #[test]
    fn ripmap_multipart_rejects_duplicate_names() {
        let p0 = build_part("dup", 8, 8, 0.0, Compression::None, 4);
        let p1 = build_part("dup", 8, 8, 0.5, Compression::None, 4);
        let err = encode_exr_multipart_tiled_ripmap(&[p0, p1]).unwrap_err();
        assert!(format!("{err}").contains("duplicate"));
    }

    #[test]
    fn ripmap_multipart_rejects_bad_grid_shape() {
        // Truncate one grid row to wrong x-length.
        let mut p = build_part("trunc", 16, 16, 0.0, Compression::None, 8);
        p.pyramid.grid[0].truncate(1);
        let err = encode_exr_multipart_tiled_ripmap(&[p]).unwrap_err();
        assert!(format!("{err}").contains("x-levels"));
    }

    #[test]
    fn ripmap_multipart_rejects_unsupported_compression() {
        let mut p = build_part("piz", 8, 8, 0.0, Compression::None, 4);
        p.compression = Compression::Piz;
        let err = encode_exr_multipart_tiled_ripmap(&[p]).unwrap_err();
        assert!(format!("{err}").contains("NONE/ZIP/ZIPS/RLE"));
    }

    #[test]
    fn ripmap_multipart_rejects_subsampled_channels() {
        let mut p = build_part("sub", 8, 8, 0.0, Compression::None, 4);
        p.channels[0].x_sampling = 2;
        let err = encode_exr_multipart_tiled_ripmap(&[p]).unwrap_err();
        assert!(format!("{err}").contains("sub-sampled"));
    }

    #[test]
    fn ripmap_multipart_routes_through_multipart_tiled_one_level_redirect() {
        // A RIPMAP_LEVELS multi-part file should be rejected by the
        // ONE_LEVEL multi-part tiled reader with the multilevel pointer
        // message — same routing behaviour as the MIPMAP multi-part case.
        let p = build_part("rip", 8, 8, 0.0, Compression::None, 4);
        let bytes = encode_exr_multipart_tiled_ripmap(&[p]).unwrap();
        let err = crate::parse_exr_multipart_tiled(&bytes).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("parse_exr_multipart_tiled_multilevel"),
            "expected redirect message, got: {msg}"
        );
    }
}
