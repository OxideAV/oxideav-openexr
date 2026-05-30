//! Multi-part flat (non-deep) tiled EXR encoder.
//!
//! A multi-part tiled file carries one or more independent
//! `type="tiledimage"` parts in a single .exr stream. By analogy with
//! the multi-part deep-tiled writer (round 181), this writer does NOT
//! set the `single_tile` (0x200) version-field bit — the per-part
//! `tiles[tiledesc]` attribute + `type="tiledimage"` string carry the
//! tile-ness signal. Only the `multipart` (0x1000) bit is set.
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
//! ONE_LEVEL only. Per-tile payload layout matches single-part flat
//! tiled exactly (row-major within the tile, channels in alphabetical
//! order). Edge tiles store only the valid pixel rectangle.
//!
//! Compression: NONE / ZIP / ZIPS / RLE. Each tile is independently
//! compressed (the spec's "store raw if compression doesn't shrink it"
//! rule applies per-tile, as in the single-part tile encoder).
//!
//! Companion reader: [`crate::parse_exr_multipart_tiled`].

use crate::decoder::{apply_zip_interleave, apply_zip_predictor};
use crate::error::{ExrError, Result};
use crate::header::{encode_attribute_value, VersionField};
use crate::types::{
    Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType, EXR_MAGIC,
};

/// One flat tiled part for [`encode_exr_multipart_tiled`]. Mirrors the
/// shape of [`crate::MultipartScanlinePart`] plus the tile-size fields
/// from the single-part tiled writer.
pub struct MultipartTiledPart<'a> {
    /// Unique non-empty part name.
    pub name: String,
    /// Width / height of the data window (display window matches it).
    pub width: u32,
    pub height: u32,
    /// Tile dimensions (both > 0). Edge tiles are stored at their valid
    /// pixel size only.
    pub tile_x: u32,
    pub tile_y: u32,
    /// Channels in alphabetical order. All must use 1×1 sampling
    /// (tiled files require this).
    pub channels: Vec<Channel>,
    /// One `width * height` f32 slice per channel, in the same
    /// alphabetical order as `channels`. UINT channels store the f32
    /// value rounded to nearest u32.
    pub planes: Vec<&'a [f32]>,
    pub compression: Compression,
}

/// Encode a multi-part flat (non-deep) tiled EXR file. Each part is
/// validated independently (unique non-empty name; channels alphabetical
/// with 1×1 sampling; plane lengths match `width * height`; compression
/// in {NONE, ZIP, ZIPS, RLE}; tile sizes > 0). Tile chunks are emitted
/// ty-outer tx-inner within each part (INCREASING_Y row-major), and
/// part chunks are concatenated in part-order.
///
/// Self-roundtrips through [`crate::parse_exr_multipart_tiled`].
pub fn encode_exr_multipart_tiled(parts: &[MultipartTiledPart]) -> Result<Vec<u8>> {
    if parts.is_empty() {
        return Err(ExrError::invalid(
            "encode_exr_multipart_tiled: at least one part required".to_string(),
        ));
    }
    // ---- Validate every part up front. ----
    for (i, p) in parts.iter().enumerate() {
        if p.name.is_empty() {
            return Err(ExrError::invalid(format!(
                "multi-part tiled part {i}: empty name"
            )));
        }
        for (j, other) in parts.iter().enumerate() {
            if j != i && other.name == p.name {
                return Err(ExrError::invalid(format!(
                    "duplicate multi-part tiled part name '{}' (parts {i} and {j})",
                    p.name
                )));
            }
        }
        if p.width == 0 || p.height == 0 {
            return Err(ExrError::invalid(format!(
                "multi-part tiled part '{}': dataWindow {}×{} must be > 0",
                p.name, p.width, p.height
            )));
        }
        if p.tile_x == 0 || p.tile_y == 0 {
            return Err(ExrError::invalid(format!(
                "multi-part tiled part '{}': tile size {}×{} must both be > 0",
                p.name, p.tile_x, p.tile_y
            )));
        }
        if p.channels.len() != p.planes.len() {
            return Err(ExrError::invalid(format!(
                "multi-part tiled part '{}': channels.len()={} != planes.len()={}",
                p.name,
                p.channels.len(),
                p.planes.len()
            )));
        }
        for win in p.channels.windows(2) {
            if win[0].name >= win[1].name {
                return Err(ExrError::invalid(format!(
                    "multi-part tiled part '{}': channels not alphabetical: '{}' >= '{}'",
                    p.name, win[0].name, win[1].name
                )));
            }
        }
        for (ch, plane) in p.channels.iter().zip(p.planes.iter()) {
            if ch.x_sampling != 1 || ch.y_sampling != 1 {
                return Err(ExrError::unsupported(format!(
                    "multi-part tiled part '{}': sub-sampled channel '{}' \
                     (tiled files require 1×1 sampling)",
                    p.name, ch.name
                )));
            }
            let need = (p.width as usize) * (p.height as usize);
            if plane.len() != need {
                return Err(ExrError::invalid(format!(
                    "multi-part tiled part '{}': channel '{}' plane length {} != \
                     width*height = {need}",
                    p.name,
                    ch.name,
                    plane.len()
                )));
            }
        }
        if !matches!(
            p.compression,
            Compression::None | Compression::Zip | Compression::Zips | Compression::Rle
        ) {
            return Err(ExrError::unsupported(format!(
                "multi-part tiled part '{}': compression {:?} \
                 (encoder supports NONE/ZIP/ZIPS/RLE)",
                p.name, p.compression
            )));
        }
    }

    // ---- Per-part chunk counts (tx_count * ty_count). ----
    let mut chunk_counts: Vec<u32> = Vec::with_capacity(parts.len());
    let mut tx_counts: Vec<u32> = Vec::with_capacity(parts.len());
    let mut ty_counts: Vec<u32> = Vec::with_capacity(parts.len());
    for p in parts {
        let txc = p.width.div_ceil(p.tile_x);
        let tyc = p.height.div_ceil(p.tile_y);
        chunk_counts.push(txc * tyc);
        tx_counts.push(txc);
        ty_counts.push(tyc);
    }

    // ---- Build per-part header byte blocks. ----
    let mut header_byte_blocks: Vec<Vec<u8>> = Vec::with_capacity(parts.len());
    for (i, p) in parts.iter().enumerate() {
        let attrs = build_tiled_part_attrs(p, chunk_counts[i] as i32);
        header_byte_blocks.push(encode_part_header_attributes(&attrs));
    }

    // ---- Stitch magic + version + headers + double-NUL terminator. ----
    // multipart bit (0x1000) only; the per-part `tiles[tiledesc]`
    // attribute + `type="tiledimage"` carry the tile-ness signal, just
    // like the multi-part deep-tiled writer.
    let version = VersionField::from_u32(2 | 0x1000);
    let mut out: Vec<u8> = Vec::with_capacity(2048);
    out.extend_from_slice(&EXR_MAGIC.to_le_bytes());
    out.extend_from_slice(&version.to_u32().to_le_bytes());
    for hb in &header_byte_blocks {
        out.extend_from_slice(hb);
        out.push(0); // per-part header terminator
    }
    out.push(0); // double-NUL = end-of-all-headers

    // ---- Build per-tile payloads (ty-outer, tx-inner). ----
    struct TilePayload {
        part_idx: u32,
        tx: u32,
        ty: u32,
        payload: Vec<u8>,
    }
    let mut all_tiles: Vec<TilePayload> = Vec::new();
    for (part_idx, p) in parts.iter().enumerate() {
        let txc = tx_counts[part_idx];
        let tyc = ty_counts[part_idx];
        for ty in 0..tyc {
            for tx in 0..txc {
                let x0 = tx * p.tile_x;
                let y0 = ty * p.tile_y;
                let x1 = (x0 + p.tile_x).min(p.width);
                let y1 = (y0 + p.tile_y).min(p.height);
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
                        let plane = p.planes[ch_idx];
                        for xx in 0..tw {
                            let dst_x = x0 as usize + xx;
                            let v = plane[dst_y * p.width as usize + dst_x];
                            match ch.pixel_type {
                                PixelType::Half => raw
                                    .extend_from_slice(&crate::half::f32_to_half(v).to_le_bytes()),
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
                    payload,
                });
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

    // Emit chunks in the same order they were built.
    for c in all_tiles {
        out.extend_from_slice(&(c.part_idx as i32).to_le_bytes());
        out.extend_from_slice(&(c.tx as i32).to_le_bytes());
        out.extend_from_slice(&(c.ty as i32).to_le_bytes());
        out.extend_from_slice(&0i32.to_le_bytes()); // lvlx
        out.extend_from_slice(&0i32.to_le_bytes()); // lvly
        out.extend_from_slice(&(c.payload.len() as i32).to_le_bytes());
        out.extend_from_slice(&c.payload);
    }

    Ok(out)
}

/// Per-part attribute set for a multi-part flat tiled part: standard
/// required attributes + `name` + `tiles[tiledesc]` (ONE_LEVEL +
/// ROUND_DOWN) + `type[string="tiledimage"]` + `chunkCount`.
fn build_tiled_part_attrs(part: &MultipartTiledPart, chunk_count: i32) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (part.width - 1) as i32,
        y_max: (part.height - 1) as i32,
    };
    // tiledesc payload: u32 xSize | u32 ySize | u8 mode (low nibble =
    // level mode, high nibble = round mode). ONE_LEVEL + ROUND_DOWN = 0x00.
    let mut tiledesc = Vec::with_capacity(9);
    tiledesc.extend_from_slice(&part.tile_x.to_le_bytes());
    tiledesc.extend_from_slice(&part.tile_y.to_le_bytes());
    tiledesc.push(0x00);

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
/// compression doesn't shrink it" rule.
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
    use crate::parse_exr_multipart_tiled;

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

    #[test]
    fn multipart_tiled_two_parts_none_self_roundtrip() {
        let w = 16;
        let h = 16;
        let (a0, b0, g0, r0) = make_planes(w, h, 0.0);
        let (a1, b1, g1, r1) = make_planes(w, h, 0.5);
        let parts = vec![
            MultipartTiledPart {
                name: "p0".to_string(),
                width: w,
                height: h,
                tile_x: 8,
                tile_y: 8,
                channels: rgba_channels(),
                planes: vec![&a0, &b0, &g0, &r0],
                compression: Compression::None,
            },
            MultipartTiledPart {
                name: "p1".to_string(),
                width: w,
                height: h,
                tile_x: 8,
                tile_y: 8,
                channels: rgba_channels(),
                planes: vec![&a1, &b1, &g1, &r1],
                compression: Compression::None,
            },
        ];
        let bytes = encode_exr_multipart_tiled(&parts).unwrap();
        let imgs = parse_exr_multipart_tiled(&bytes).unwrap();
        assert_eq!(imgs.len(), 2);
        let sources = [(&a0, &b0, &g0, &r0), (&a1, &b1, &g1, &r1)];
        for (img, (sa, sb, sg, sr)) in imgs.iter().zip(sources.iter()) {
            assert_eq!(img.width(), w);
            assert_eq!(img.height(), h);
            assert_eq!(&img.planes[0].samples, *sa);
            assert_eq!(&img.planes[1].samples, *sb);
            assert_eq!(&img.planes[2].samples, *sg);
            assert_eq!(&img.planes[3].samples, *sr);
        }
    }

    #[test]
    fn multipart_tiled_three_parts_mixed_compression() {
        let w = 12;
        let h = 9;
        let (a0, b0, g0, r0) = make_planes(w, h, 0.0);
        let (a1, b1, g1, r1) = make_planes(w, h, 0.25);
        let (a2, b2, g2, r2) = make_planes(w, h, 0.75);
        let parts = vec![
            MultipartTiledPart {
                name: "none".to_string(),
                width: w,
                height: h,
                tile_x: 4,
                tile_y: 4,
                channels: rgba_channels(),
                planes: vec![&a0, &b0, &g0, &r0],
                compression: Compression::None,
            },
            MultipartTiledPart {
                name: "zips".to_string(),
                width: w,
                height: h,
                tile_x: 4,
                tile_y: 4,
                channels: rgba_channels(),
                planes: vec![&a1, &b1, &g1, &r1],
                compression: Compression::Zips,
            },
            MultipartTiledPart {
                name: "rle".to_string(),
                width: w,
                height: h,
                tile_x: 4,
                tile_y: 4,
                channels: rgba_channels(),
                planes: vec![&a2, &b2, &g2, &r2],
                compression: Compression::Rle,
            },
        ];
        let bytes = encode_exr_multipart_tiled(&parts).unwrap();
        let imgs = parse_exr_multipart_tiled(&bytes).unwrap();
        assert_eq!(imgs.len(), 3);
        let sources = [
            (&a0, &b0, &g0, &r0),
            (&a1, &b1, &g1, &r1),
            (&a2, &b2, &g2, &r2),
        ];
        for (img, (sa, sb, sg, sr)) in imgs.iter().zip(sources.iter()) {
            assert_eq!(img.width(), w);
            assert_eq!(img.height(), h);
            assert_eq!(&img.planes[0].samples, *sa);
            assert_eq!(&img.planes[1].samples, *sb);
            assert_eq!(&img.planes[2].samples, *sg);
            assert_eq!(&img.planes[3].samples, *sr);
        }
    }

    #[test]
    fn multipart_tiled_zip_edge_tiles_self_roundtrip() {
        // 13×9 with 4×3 tiles: 4×3 = 12 tiles, right column + bottom row
        // are edge tiles smaller than 4×3.
        let w = 13;
        let h = 9;
        let (a, b, g, r) = make_planes(w, h, 0.0);
        let parts = vec![MultipartTiledPart {
            name: "edge".to_string(),
            width: w,
            height: h,
            tile_x: 4,
            tile_y: 3,
            channels: rgba_channels(),
            planes: vec![&a, &b, &g, &r],
            compression: Compression::Zip,
        }];
        let bytes = encode_exr_multipart_tiled(&parts).unwrap();
        let imgs = parse_exr_multipart_tiled(&bytes).unwrap();
        assert_eq!(imgs.len(), 1);
        let img = &imgs[0];
        assert_eq!(img.width(), w);
        assert_eq!(img.height(), h);
        assert_eq!(&img.planes[0].samples, &a);
        assert_eq!(&img.planes[1].samples, &b);
        assert_eq!(&img.planes[2].samples, &g);
        assert_eq!(&img.planes[3].samples, &r);
    }

    #[test]
    fn multipart_tiled_rejects_empty_parts() {
        let r = encode_exr_multipart_tiled(&[]);
        assert!(r.is_err());
    }

    #[test]
    fn multipart_tiled_rejects_duplicate_names() {
        let w = 4;
        let h = 4;
        let (a, b, g, r) = make_planes(w, h, 0.0);
        let parts = vec![
            MultipartTiledPart {
                name: "dup".to_string(),
                width: w,
                height: h,
                tile_x: 2,
                tile_y: 2,
                channels: rgba_channels(),
                planes: vec![&a, &b, &g, &r],
                compression: Compression::None,
            },
            MultipartTiledPart {
                name: "dup".to_string(),
                width: w,
                height: h,
                tile_x: 2,
                tile_y: 2,
                channels: rgba_channels(),
                planes: vec![&a, &b, &g, &r],
                compression: Compression::None,
            },
        ];
        let err = encode_exr_multipart_tiled(&parts).unwrap_err();
        assert!(format!("{err}").contains("duplicate"));
    }

    #[test]
    fn multipart_tiled_rejects_subsampled_channels() {
        let w = 4;
        let h = 4;
        let plane: Vec<f32> = vec![0.0; (w * h) as usize];
        let parts = vec![MultipartTiledPart {
            name: "sub".to_string(),
            width: w,
            height: h,
            tile_x: 2,
            tile_y: 2,
            channels: vec![Channel {
                name: "Y".to_string(),
                pixel_type: PixelType::Float,
                p_linear: false,
                x_sampling: 2,
                y_sampling: 2,
            }],
            planes: vec![&plane],
            compression: Compression::None,
        }];
        let err = encode_exr_multipart_tiled(&parts).unwrap_err();
        assert!(format!("{err}").contains("sub-sampled"));
    }

    #[test]
    fn parse_exr_multipart_rejects_tiledimage_parts() {
        // Build a 1-part tiled multipart file and verify the scanline
        // multipart reader points at the new entry rather than mis-parsing.
        let w = 4;
        let h = 4;
        let (a, b, g, r) = make_planes(w, h, 0.0);
        let parts = vec![MultipartTiledPart {
            name: "tile".to_string(),
            width: w,
            height: h,
            tile_x: 2,
            tile_y: 2,
            channels: rgba_channels(),
            planes: vec![&a, &b, &g, &r],
            compression: Compression::None,
        }];
        let bytes = encode_exr_multipart_tiled(&parts).unwrap();
        let err = crate::parse_exr_multipart(&bytes).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("parse_exr_multipart_tiled"),
            "expected pointer to parse_exr_multipart_tiled, got: {msg}"
        );
    }
}
