//! MIPMAP_LEVELS tiled-output EXR encoder (single-part).
//!
//! Layout (single-part tiled file with the `single_tile` version bit set
//! and `tiledesc.level_mode == 1`):
//!
//! ```text
//! magic(4) | version(4 with single_tile bit set)
//! header attributes (channels, compression, dataWindow, displayWindow,
//!   lineOrder, pixelAspectRatio, screenWindowCenter, screenWindowWidth,
//!   tiles[tiledesc, mode=0x01 = MIPMAP+ROUND_DOWN], chunkCount[int],
//!   type[string="tiledimage"])
//! NUL terminator
//! tile offset table: chunkCount * u64 LE absolute byte offsets, with
//!   chunkCount = sum over levels 0..N-1 of
//!   ceil(lw/tw)*ceil(lh/th), where (lw,lh) = mipmap_level_dim(...,level).
//! tile chunks (each: tx i32 | ty i32 | lvlx i32 | lvly i32 | size i32 |
//!   payload[size]) — for MIPMAP, lvlx == lvly == level.
//! ```
//!
//! Per the openexr.com Technical Introduction (§Tile offset table
//! ordering for INCREASING_Y line order), MIPMAP_LEVELS visits the
//! diagonal of (lvlx,lvly) pairs only: levels 0..N-1 in ascending order,
//! and within each level tiles are laid out row-major INCREASING_Y
//! (ty outer, tx inner).
//!
//! Caller supplies a pyramid: one plane per channel **per level**. The
//! crate does NOT itself filter / downsample from level 0 — that is the
//! caller's responsibility (different applications want different
//! filters, and the spec deliberately does not mandate one). The
//! `encode_exr_tiled_rgba_float_mipmap_box_filter` convenience helper
//! provides a basic 2×2 box-filter pyramid for callers who don't need
//! to control filtering.

use crate::decoder::{apply_zip_interleave, apply_zip_predictor};
use crate::error::{ExrError, Result};
use crate::header::{encode_header, VersionField};
use crate::types::{Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType};

/// One level of a mipmap pyramid: explicit width/height plus one f32 plane
/// per channel (in the same alphabetical order as the file's channel list).
/// Plane lengths must equal `level_w * level_h`.
#[derive(Debug, Clone)]
pub struct MipmapLevel {
    pub width: u32,
    pub height: u32,
    /// One `width * height` f32 plane per channel. Channels are in the
    /// same alphabetical order as the file-level `channels` list.
    pub planes: Vec<Vec<f32>>,
}

/// Number of mipmap levels for ROUND_DOWN given `max(w,h)`. Matches
/// `crate::decoder::mipmap_level_count(max(w,h), false)`.
pub fn mipmap_level_count_round_down(width: u32, height: u32) -> u32 {
    crate::decoder::mipmap_level_count(width.max(height), false)
}

/// Encode an RGBA-float MIPMAP_LEVELS tiled EXR using a 2×2 box-filter
/// pyramid generated from the caller's level-0 image.
///
/// `samples` is `width * height * 4` long, in `R, G, B, A` pixel order.
/// Uses ROUND_DOWN rounding (the OpenEXR default).
pub fn encode_exr_tiled_rgba_float_mipmap_box_filter(
    width: u32,
    height: u32,
    samples: &[f32],
    compression: Compression,
    tile_x: u32,
    tile_y: u32,
) -> Result<Vec<u8>> {
    let need = (width as usize) * (height as usize) * 4;
    if samples.len() != need {
        return Err(ExrError::invalid(format!(
            "samples length {} != width({width})*height({height})*4 = {need}",
            samples.len()
        )));
    }
    // Synthesise four per-channel planes from interleaved RGBA, in
    // alphabetical order: A, B, G, R (matching the on-disk layout).
    let pixels = (width as usize) * (height as usize);
    let mut a = Vec::with_capacity(pixels);
    let mut b = Vec::with_capacity(pixels);
    let mut g = Vec::with_capacity(pixels);
    let mut r = Vec::with_capacity(pixels);
    for px in 0..pixels {
        r.push(samples[px * 4]);
        g.push(samples[px * 4 + 1]);
        b.push(samples[px * 4 + 2]);
        a.push(samples[px * 4 + 3]);
    }

    let channels = vec![
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
    ];

    let pyramid = build_box_filter_pyramid(width, height, &[a, b, g, r]);

    encode_exr_tiled_mipmap(&channels, &pyramid, compression, tile_x, tile_y)
}

/// Build a ROUND_DOWN mipmap pyramid by 2×2 box filtering each channel
/// plane independently. Level 0 is the input; level `l` has dimensions
/// `mipmap_level_dim(full, l, false)`. Returns one `MipmapLevel` per level.
///
/// At odd-dimension levels the 2×2 box is taken from the top-left
/// `2*new × 2*new` pixels (i.e. last odd pixel is dropped). This matches
/// the ROUND_DOWN convention.
pub fn build_box_filter_pyramid(
    width: u32,
    height: u32,
    level0_planes: &[Vec<f32>],
) -> Vec<MipmapLevel> {
    let n_levels = mipmap_level_count_round_down(width, height);
    let mut out: Vec<MipmapLevel> = Vec::with_capacity(n_levels as usize);
    // Push level 0 verbatim.
    out.push(MipmapLevel {
        width,
        height,
        planes: level0_planes.to_vec(),
    });
    for l in 1..n_levels {
        let prev = &out[(l - 1) as usize];
        let lw = (prev.width / 2).max(1);
        let lh = (prev.height / 2).max(1);
        let mut planes: Vec<Vec<f32>> = Vec::with_capacity(prev.planes.len());
        for src in &prev.planes {
            let mut dst = vec![0.0_f32; (lw * lh) as usize];
            for y in 0..lh {
                for x in 0..lw {
                    // 2×2 box: floor(prev/2) pixels from upper-left.
                    let sx = (x * 2) as usize;
                    let sy = (y * 2) as usize;
                    let pw = prev.width as usize;
                    let sx1 = (sx + 1).min(prev.width as usize - 1);
                    let sy1 = (sy + 1).min(prev.height as usize - 1);
                    let v00 = src[sy * pw + sx];
                    let v01 = src[sy * pw + sx1];
                    let v10 = src[sy1 * pw + sx];
                    let v11 = src[sy1 * pw + sx1];
                    dst[(y * lw + x) as usize] = 0.25 * (v00 + v01 + v10 + v11);
                }
            }
            planes.push(dst);
        }
        out.push(MipmapLevel {
            width: lw,
            height: lh,
            planes,
        });
    }
    out
}

/// General MIPMAP_LEVELS tiled encoder. Writes a single-part tiled EXR
/// where the offset table walks levels `0..N-1` in ascending order and
/// within each level emits tile chunks in INCREASING_Y row-major order.
///
/// All channels MUST have `x_sampling == 1 && y_sampling == 1` (the
/// OpenEXR file format requires this for tiled files).
///
/// The pyramid length must equal
/// `mipmap_level_count_round_down(level0_w, level0_h)`. Level `l`'s
/// `width`/`height` must equal `mipmap_level_dim(level0_w, l,
/// round_up=false)` and `mipmap_level_dim(level0_h, l, false)`
/// respectively, and each plane length must equal `level_w * level_h`.
///
/// All levels use the same compression mode (per the file's single
/// `compression` attribute). Encoder supports NONE / ZIP / ZIPS / RLE.
pub fn encode_exr_tiled_mipmap(
    channels: &[Channel],
    pyramid: &[MipmapLevel],
    compression: Compression,
    tile_x: u32,
    tile_y: u32,
) -> Result<Vec<u8>> {
    if pyramid.is_empty() {
        return Err(ExrError::invalid(
            "mipmap pyramid must have at least one level".to_string(),
        ));
    }
    if tile_x == 0 || tile_y == 0 {
        return Err(ExrError::invalid(format!(
            "tile size {tile_x}×{tile_y} must both be > 0"
        )));
    }
    if !matches!(
        compression,
        Compression::None | Compression::Zip | Compression::Zips | Compression::Rle
    ) {
        return Err(ExrError::unsupported(format!(
            "compression {compression:?} (mipmap tiled encoder supports NONE + ZIP + ZIPS + RLE)"
        )));
    }
    for ch in channels {
        if ch.x_sampling != 1 || ch.y_sampling != 1 {
            return Err(ExrError::unsupported(format!(
                "channel '{}' sampling != 1×1 in tiled encode (spec requires 1×1 in tiled files)",
                ch.name
            )));
        }
    }
    for win in channels.windows(2) {
        if win[0].name >= win[1].name {
            return Err(ExrError::invalid(format!(
                "channels not in alphabetical order: '{}' >= '{}'",
                win[0].name, win[1].name
            )));
        }
    }

    let width = pyramid[0].width;
    let height = pyramid[0].height;
    let expected_levels = mipmap_level_count_round_down(width, height);
    if pyramid.len() as u32 != expected_levels {
        return Err(ExrError::invalid(format!(
            "mipmap pyramid has {} levels, expected {expected_levels} for {width}×{height} ROUND_DOWN",
            pyramid.len()
        )));
    }
    for (l, lvl) in pyramid.iter().enumerate() {
        let want_w = crate::decoder::mipmap_level_dim(width, l as u32, false);
        let want_h = crate::decoder::mipmap_level_dim(height, l as u32, false);
        if lvl.width != want_w || lvl.height != want_h {
            return Err(ExrError::invalid(format!(
                "mipmap level {l} is {}×{} but spec requires {want_w}×{want_h} (ROUND_DOWN)",
                lvl.width, lvl.height
            )));
        }
        if lvl.planes.len() != channels.len() {
            return Err(ExrError::invalid(format!(
                "mipmap level {l} has {} planes but {} channels declared",
                lvl.planes.len(),
                channels.len()
            )));
        }
        for (ch, p) in channels.iter().zip(lvl.planes.iter()) {
            let need = (lvl.width as usize) * (lvl.height as usize);
            if p.len() != need {
                return Err(ExrError::invalid(format!(
                    "mipmap level {l} channel '{}' plane length {} != {}*{} = {need}",
                    ch.name,
                    p.len(),
                    lvl.width,
                    lvl.height
                )));
            }
        }
    }

    // Compute chunk count = sum over levels of tile-grid size.
    let mut chunk_count: u32 = 0;
    for lvl in pyramid {
        chunk_count += lvl.width.div_ceil(tile_x) * lvl.height.div_ceil(tile_y);
    }

    let attrs = build_tiled_mipmap_attributes(
        channels,
        width,
        height,
        compression,
        tile_x,
        tile_y,
        chunk_count,
    );

    let version = VersionField::from_u32(2 | 0x200);
    let header_bytes = encode_header(version, &attrs);

    // Build per-tile payloads. Iteration order: levels 0..N-1, within each
    // level ty outer (0..ty_count), tx inner (0..tx_count).
    #[allow(clippy::type_complexity)]
    let mut tile_chunks: Vec<(u32, u32, u32, u32, Vec<u8>)> =
        Vec::with_capacity(chunk_count as usize);
    for (l, lvl) in pyramid.iter().enumerate() {
        let lvl_idx = l as u32;
        let tx_count = lvl.width.div_ceil(tile_x);
        let ty_count = lvl.height.div_ceil(tile_y);
        for ty in 0..ty_count {
            for tx in 0..tx_count {
                let x0 = tx * tile_x;
                let y0 = ty * tile_y;
                let x1 = (x0 + tile_x).min(lvl.width);
                let y1 = (y0 + tile_y).min(lvl.height);
                let tw = (x1 - x0) as usize;
                let th = (y1 - y0) as usize;
                let bpp: usize = channels
                    .iter()
                    .map(|c| c.pixel_type.bytes_per_sample())
                    .sum();
                let mut raw = Vec::with_capacity(tw * th * bpp);
                for line in 0..th {
                    let dst_y = y0 as usize + line;
                    for (ch_idx, ch) in channels.iter().enumerate() {
                        let plane = &lvl.planes[ch_idx];
                        for xx in 0..tw {
                            let dst_x = x0 as usize + xx;
                            let v = plane[dst_y * lvl.width as usize + dst_x];
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
                let payload = compress_tile_payload(raw, compression)?;
                // For MIPMAP_LEVELS the chunk header carries lvlx == lvly
                // == level index (the diagonal of the (lvlx, lvly) grid).
                tile_chunks.push((tx, ty, lvl_idx, lvl_idx, payload));
            }
        }
    }

    // Compute absolute byte offsets for each tile chunk.
    let offset_table_size = (chunk_count as usize) * 8;
    let chunks_start = header_bytes.len() + offset_table_size;
    let mut tile_offsets: Vec<u64> = Vec::with_capacity(chunk_count as usize);
    {
        let mut running = chunks_start;
        for (_, _, _, _, p) in &tile_chunks {
            tile_offsets.push(running as u64);
            running += 20 + p.len();
        }
    }
    let total_size = tile_offsets
        .last()
        .map(|&o| o as usize)
        .unwrap_or(chunks_start)
        + tile_chunks
            .last()
            .map(|(_, _, _, _, p)| 20 + p.len())
            .unwrap_or(0);

    let mut out = Vec::with_capacity(total_size);
    out.extend_from_slice(&header_bytes);
    for &off in &tile_offsets {
        out.extend_from_slice(&off.to_le_bytes());
    }
    for (tx, ty, lx, ly, p) in tile_chunks {
        out.extend_from_slice(&(tx as i32).to_le_bytes());
        out.extend_from_slice(&(ty as i32).to_le_bytes());
        out.extend_from_slice(&(lx as i32).to_le_bytes());
        out.extend_from_slice(&(ly as i32).to_le_bytes());
        out.extend_from_slice(&(p.len() as i32).to_le_bytes());
        out.extend_from_slice(&p);
    }
    Ok(out)
}

fn build_tiled_mipmap_attributes(
    channels: &[Channel],
    width: u32,
    height: u32,
    compression: Compression,
    tile_x: u32,
    tile_y: u32,
    chunk_count: u32,
) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (width - 1) as i32,
        y_max: (height - 1) as i32,
    };
    // tiledesc: u32 xSize | u32 ySize | u8 mode. mode = (round_mode << 4)
    // | level_mode; MIPMAP_LEVELS = 1, ROUND_DOWN = 0.
    let mut tiledesc = Vec::with_capacity(9);
    tiledesc.extend_from_slice(&tile_x.to_le_bytes());
    tiledesc.extend_from_slice(&tile_y.to_le_bytes());
    tiledesc.push(0x01);

    vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(channels.to_vec()),
        },
        Attribute {
            name: "chunkCount".to_string(),
            value: AttributeValue::Other {
                type_name: "int".to_string(),
                data: (chunk_count as i32).to_le_bytes().to_vec(),
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
    use crate::parse_exr;

    fn make_image(w: u32, h: u32) -> Vec<f32> {
        let mut s = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                s.push(x as f32 / w as f32);
                s.push(y as f32 / h as f32);
                s.push(((x ^ y) as f32) * 0.01);
                s.push(1.0);
            }
        }
        s
    }

    fn assert_planes_match_rgba(img: &crate::ExrImage, source_rgba: &[f32]) {
        let w = img.width() as usize;
        let h = img.height() as usize;
        let a = &img.planes[0].samples;
        let b = &img.planes[1].samples;
        let g = &img.planes[2].samples;
        let r = &img.planes[3].samples;
        for y in 0..h {
            for x in 0..w {
                let off = y * w + x;
                assert!(
                    (r[off] - source_rgba[off * 4]).abs() < 1e-6,
                    "R mismatch ({x},{y})"
                );
                assert!(
                    (g[off] - source_rgba[off * 4 + 1]).abs() < 1e-6,
                    "G mismatch ({x},{y})"
                );
                assert!(
                    (b[off] - source_rgba[off * 4 + 2]).abs() < 1e-6,
                    "B mismatch ({x},{y})"
                );
                assert!(
                    (a[off] - source_rgba[off * 4 + 3]).abs() < 1e-6,
                    "A mismatch ({x},{y})"
                );
            }
        }
    }

    #[test]
    fn mipmap_none_self_roundtrip_level0() {
        // 32×32 with 16×16 tiles. Pyramid has 6 levels (32,16,8,4,2,1).
        // Self-roundtrip should recover the level-0 RGBA samples.
        let w = 32;
        let h = 32;
        let samples = make_image(w, h);
        let bytes = encode_exr_tiled_rgba_float_mipmap_box_filter(
            w,
            h,
            &samples,
            Compression::None,
            16,
            16,
        )
        .unwrap();
        let img = parse_exr(&bytes).unwrap();
        assert_eq!(img.width(), w);
        assert_eq!(img.height(), h);
        assert_planes_match_rgba(&img, &samples);
    }

    #[test]
    fn mipmap_zip_self_roundtrip_level0() {
        let w = 32;
        let h = 32;
        let samples = make_image(w, h);
        let bytes =
            encode_exr_tiled_rgba_float_mipmap_box_filter(w, h, &samples, Compression::Zip, 16, 16)
                .unwrap();
        let img = parse_exr(&bytes).unwrap();
        assert_planes_match_rgba(&img, &samples);
    }

    #[test]
    fn mipmap_rle_self_roundtrip_constant() {
        let w = 32;
        let h = 32;
        let samples = vec![0.25_f32; (w * h * 4) as usize];
        let bytes =
            encode_exr_tiled_rgba_float_mipmap_box_filter(w, h, &samples, Compression::Rle, 16, 16)
                .unwrap();
        let img = parse_exr(&bytes).unwrap();
        assert_planes_match_rgba(&img, &samples);
    }

    #[test]
    fn mipmap_zips_self_roundtrip_non_square() {
        let w = 16;
        let h = 8;
        let samples = make_image(w, h);
        let bytes =
            encode_exr_tiled_rgba_float_mipmap_box_filter(w, h, &samples, Compression::Zips, 8, 8)
                .unwrap();
        let img = parse_exr(&bytes).unwrap();
        assert_planes_match_rgba(&img, &samples);
    }

    #[test]
    fn mipmap_chunk_count_matches_pyramid_sum() {
        // 32×32 with 16×16 tiles, ROUND_DOWN:
        // Level 0: 32×32 -> 2×2 = 4 tiles
        // Level 1: 16×16 -> 1×1 = 1 tile
        // Level 2: 8×8   -> 1×1 = 1 tile
        // Level 3: 4×4   -> 1×1 = 1 tile
        // Level 4: 2×2   -> 1×1 = 1 tile
        // Level 5: 1×1   -> 1×1 = 1 tile
        // Total: 9.
        let w = 32;
        let h = 32;
        let samples = make_image(w, h);
        let bytes = encode_exr_tiled_rgba_float_mipmap_box_filter(
            w,
            h,
            &samples,
            Compression::None,
            16,
            16,
        )
        .unwrap();

        let header = crate::header::parse_header(&bytes).unwrap();
        let chunk_count_attr = header
            .attributes
            .iter()
            .find(|a| a.name == "chunkCount")
            .expect("encoder must emit chunkCount");
        match &chunk_count_attr.value {
            AttributeValue::Other { type_name, data } => {
                assert_eq!(type_name, "int");
                let v = i32::from_le_bytes(data[..4].try_into().unwrap());
                assert_eq!(v, 9, "chunk count should match pyramid total");
            }
            _ => panic!("chunkCount should be Other(int)"),
        }

        // Also verify tiledesc level_mode = 1 (MIPMAP_LEVELS).
        let tiles_attr = header
            .attributes
            .iter()
            .find(|a| a.name == "tiles")
            .expect("encoder must emit tiles");
        match &tiles_attr.value {
            AttributeValue::Other { type_name, data } => {
                assert_eq!(type_name, "tiledesc");
                assert_eq!(data.len(), 9);
                assert_eq!(data[8] & 0x0F, 1, "level_mode should be MIPMAP_LEVELS");
                assert_eq!((data[8] >> 4) & 0x0F, 0, "round_mode should be ROUND_DOWN");
            }
            _ => panic!("tiles should be Other(tiledesc)"),
        }
    }

    #[test]
    fn mipmap_rejects_wrong_pyramid_length() {
        let w = 32;
        let h = 32;
        // Only 3 levels for a 32×32 image (which actually needs 6).
        let bad_pyramid = vec![
            MipmapLevel {
                width: 32,
                height: 32,
                planes: vec![vec![0.0; 32 * 32]; 1],
            },
            MipmapLevel {
                width: 16,
                height: 16,
                planes: vec![vec![0.0; 16 * 16]; 1],
            },
            MipmapLevel {
                width: 8,
                height: 8,
                planes: vec![vec![0.0; 8 * 8]; 1],
            },
        ];
        let chs = vec![Channel {
            name: "Y".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        }];
        let _ = (w, h);
        let r = encode_exr_tiled_mipmap(&chs, &bad_pyramid, Compression::None, 16, 16);
        assert!(r.is_err(), "should reject short pyramid");
    }

    #[test]
    fn build_box_filter_pyramid_sizes_correct() {
        let w = 16u32;
        let h = 16u32;
        // single constant plane of 1.0 → every level stays at 1.0
        let plane: Vec<f32> = vec![1.0; (w * h) as usize];
        let pyr = build_box_filter_pyramid(w, h, &[plane]);
        // 16 → 5 levels (16,8,4,2,1)
        assert_eq!(pyr.len(), 5);
        assert_eq!((pyr[0].width, pyr[0].height), (16, 16));
        assert_eq!((pyr[1].width, pyr[1].height), (8, 8));
        assert_eq!((pyr[2].width, pyr[2].height), (4, 4));
        assert_eq!((pyr[3].width, pyr[3].height), (2, 2));
        assert_eq!((pyr[4].width, pyr[4].height), (1, 1));
        // Constant input → constant output at every level
        for lvl in &pyr {
            for &v in &lvl.planes[0] {
                assert!(
                    (v - 1.0).abs() < 1e-9,
                    "box-filter constant should stay 1.0"
                );
            }
        }
    }
}
