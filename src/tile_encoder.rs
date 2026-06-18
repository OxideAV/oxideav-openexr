//! Tiled-output EXR encoder (single-part, ONE_LEVEL).
//!
//! Layout (single-part tiled file, version-field bit 0x200 set):
//!
//! ```text
//! magic(4) | version(4 with single_tile bit set)
//! header attributes (channels, compression, dataWindow, displayWindow,
//!   lineOrder, pixelAspectRatio, screenWindowCenter, screenWindowWidth,
//!   tiles[tiledesc], chunkCount[int], type[string="tiledimage"])
//! NUL terminator
//! tile offset table: chunkCount * u64 LE absolute byte offsets
//! tile chunks (each: tx i32 | ty i32 | lvlx i32 | lvly i32 | size i32 | payload[size])
//! ```
//!
//! ONE_LEVEL only — chunkCount equals `ceil(width / tileX) *
//! ceil(height / tileY)`. `lvlx` and `lvly` are always 0 in the tile
//! chunk header for ONE_LEVEL.
//!
//! Per-tile payload: row-major within the tile, channels in
//! alphabetical order (matching the on-disk scanline layout, just
//! restricted to the tile's pixel rectangle). Edge tiles store only the
//! valid pixel rectangle (i.e. last column/row tiles may be smaller
//! than `tileX × tileY`).
//!
//! Compression rules carry over from the scanline path: NONE / ZIP /
//! ZIPS / RLE supported. Each tile payload is independently compressed,
//! then stored either compressed or raw — whichever is smaller, exactly
//! as the spec mandates.
//!
//! Sub-sampled channels are NOT supported in tiled encode: the
//! reference EXR docs say tiled files MUST have all channels at 1×1
//! sampling. (The decoder accepts sub-sampled scanline files, so this
//! restriction is encoder-only.)

use crate::decoder::{apply_zip_interleave, apply_zip_predictor};
use crate::error::{ExrError, Result};
use crate::header::{encode_header, VersionField};
use crate::types::{Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType};

/// Build the standard 4-channel RGBA-float tiled-file attribute set
/// (NONE / ZIP / ZIPS / RLE). The chunk count must be added separately
/// because it depends on the tile grid.
fn rgba_float_tiled_attributes(
    width: u32,
    height: u32,
    compression: Compression,
    tile_x: u32,
    tile_y: u32,
    chunk_count: u32,
) -> Vec<Attribute> {
    let chs = vec![
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
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (width - 1) as i32,
        y_max: (height - 1) as i32,
    };
    // tiledesc payload: u32 xSize | u32 ySize | u8 mode (level mode in low
    // 4 bits, round mode in high 4 bits). ONE_LEVEL + ROUND_DOWN = 0x00.
    let mut tiledesc = Vec::with_capacity(9);
    tiledesc.extend_from_slice(&tile_x.to_le_bytes());
    tiledesc.extend_from_slice(&tile_y.to_le_bytes());
    tiledesc.push(0x00);

    vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(chs),
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
                // EXR `string` type stores raw bytes — no NUL terminator
                // (length is in the size field).
                data: b"tiledimage".to_vec(),
            },
        },
    ]
}

/// Encode an RGBA-float tiled EXR (single-part, ONE_LEVEL) with the
/// requested compression and tile size. `samples` is `width * height *
/// 4` long, in `R, G, B, A` pixel order.
///
/// Companion to [`crate::encode_exr_scanline_rgba_float_with`].
/// Round-trips bit-exactly through [`crate::parse_exr`].
pub fn encode_exr_tiled_rgba_float_with(
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
    if tile_x == 0 || tile_y == 0 {
        return Err(ExrError::invalid(format!(
            "tile size {tile_x}×{tile_y} must both be > 0"
        )));
    }
    if !matches!(
        compression,
        Compression::None
            | Compression::Zip
            | Compression::Zips
            | Compression::Rle
            | Compression::Pxr24
            | Compression::B44
            | Compression::B44a
    ) {
        return Err(ExrError::unsupported(format!(
            "compression {compression:?} (tiled encoder supports \
             NONE + ZIP + ZIPS + RLE + PXR24 + B44 + B44A)"
        )));
    }

    // Reshape interleaved RGBA into per-channel planes in alphabetical
    // order: A, B, G, R.
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

    let chs = vec![
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

    let planes_f32: Vec<&[f32]> = vec![&a, &b, &g, &r];

    encode_exr_tiled(
        width,
        height,
        &chs,
        &planes_f32,
        compression,
        tile_x,
        tile_y,
    )
}

/// General-purpose tiled encoder. Writes a single-part ONE_LEVEL tiled
/// EXR where each plane carries one `width × height` `f32` slice in
/// alphabetical channel order. UINT channels store the f32 value
/// rounded to nearest u32.
///
/// All channels MUST have `x_sampling == 1 && y_sampling == 1` (the
/// OpenEXR file format requires this for tiled files).
pub fn encode_exr_tiled(
    width: u32,
    height: u32,
    channels: &[Channel],
    planes: &[&[f32]],
    compression: Compression,
    tile_x: u32,
    tile_y: u32,
) -> Result<Vec<u8>> {
    if channels.len() != planes.len() {
        return Err(ExrError::invalid(format!(
            "channels.len()={} != planes.len()={}",
            channels.len(),
            planes.len()
        )));
    }
    for (ch, p) in channels.iter().zip(planes.iter()) {
        if ch.x_sampling != 1 || ch.y_sampling != 1 {
            return Err(ExrError::unsupported(format!(
                "channel '{}' sampling != 1×1 in tiled encode (spec requires 1×1 in tiled files)",
                ch.name
            )));
        }
        let need = (width as usize) * (height as usize);
        if p.len() != need {
            return Err(ExrError::invalid(format!(
                "channel '{}' plane length {} != width*height = {need}",
                ch.name,
                p.len()
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
    if tile_x == 0 || tile_y == 0 {
        return Err(ExrError::invalid(format!(
            "tile size {tile_x}×{tile_y} must both be > 0"
        )));
    }
    if !matches!(
        compression,
        Compression::None
            | Compression::Zip
            | Compression::Zips
            | Compression::Rle
            | Compression::Pxr24
            | Compression::B44
            | Compression::B44a
    ) {
        return Err(ExrError::unsupported(format!(
            "compression {compression:?} (tiled encoder supports \
             NONE + ZIP + ZIPS + RLE + PXR24 + B44 + B44A)"
        )));
    }

    // Tile grid (ONE_LEVEL): tx_count × ty_count tiles total.
    let tx_count = width.div_ceil(tile_x);
    let ty_count = height.div_ceil(tile_y);
    let chunk_count = tx_count * ty_count;

    let attrs = build_tiled_attributes(
        channels,
        width,
        height,
        compression,
        tile_x,
        tile_y,
        chunk_count,
    );

    // Set the version-field single_tile bit (0x200) and format version 2.
    let version = VersionField::from_u32(2 | 0x200);
    let header_bytes = encode_header(version, &attrs);

    // Build per-tile payloads in INCREASING_Y row-major order: ty
    // outer, tx inner.
    let mut tile_payloads: Vec<(u32, u32, Vec<u8>)> = Vec::with_capacity(chunk_count as usize);
    for ty in 0..ty_count {
        for tx in 0..tx_count {
            let x0 = tx * tile_x;
            let y0 = ty * tile_y;
            let x1 = (x0 + tile_x).min(width);
            let y1 = (y0 + tile_y).min(height);
            let tw = (x1 - x0) as usize;
            let th = (y1 - y0) as usize;

            // Build the raw byte stream for this tile: row-major, then
            // channel-alphabetical within each row.
            let bpp: usize = channels
                .iter()
                .map(|c| c.pixel_type.bytes_per_sample())
                .sum();
            let mut raw = Vec::with_capacity(tw * th * bpp);
            for line in 0..th {
                let dst_y = y0 as usize + line;
                for (ch_idx, ch) in channels.iter().enumerate() {
                    let plane = planes[ch_idx];
                    for xx in 0..tw {
                        let dst_x = x0 as usize + xx;
                        let v = plane[dst_y * width as usize + dst_x];
                        match ch.pixel_type {
                            PixelType::Half => {
                                raw.extend_from_slice(&crate::half::f32_to_half(v).to_le_bytes())
                            }
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
            // PXR24 / B44 reorganise the tile into byte-plane / per-channel
            // forms (a layout distinct from the native `raw` stream), so they
            // consume tile-local f32 sub-planes rather than `raw`. Gather one
            // `tw × th` sub-plane per channel (1×1 sampling for tiled files).
            let payload = match compression {
                Compression::Pxr24 | Compression::B44 | Compression::B44a => {
                    let mut sub_planes: Vec<Vec<f32>> = Vec::with_capacity(channels.len());
                    for plane in planes.iter() {
                        let mut sp = Vec::with_capacity(tw * th);
                        for line in 0..th {
                            let dst_y = y0 as usize + line;
                            for xx in 0..tw {
                                let dst_x = x0 as usize + xx;
                                sp.push(plane[dst_y * width as usize + dst_x]);
                            }
                        }
                        sub_planes.push(sp);
                    }
                    let refs: Vec<&[f32]> = sub_planes.iter().map(|p| p.as_slice()).collect();
                    compress_tile_payload_reorg(raw, channels, &refs, tw as u32, th, compression)?
                }
                _ => compress_tile_payload(raw, compression)?,
            };
            tile_payloads.push((tx, ty, payload));
        }
    }

    // Compute absolute byte offsets for each tile chunk.
    let offset_table_size = (chunk_count as usize) * 8;
    let chunks_start = header_bytes.len() + offset_table_size;
    let mut tile_offsets: Vec<u64> = Vec::with_capacity(chunk_count as usize);
    {
        let mut running = chunks_start;
        for (_tx, _ty, p) in &tile_payloads {
            tile_offsets.push(running as u64);
            running += 20 + p.len(); // 4×i32 coords + i32 size + payload
        }
    }
    let total_size = tile_offsets
        .last()
        .map(|&o| o as usize)
        .unwrap_or(chunks_start)
        + tile_payloads
            .last()
            .map(|(_, _, p)| 20 + p.len())
            .unwrap_or(0);

    let mut out = Vec::with_capacity(total_size);
    out.extend_from_slice(&header_bytes);
    for &off in &tile_offsets {
        out.extend_from_slice(&off.to_le_bytes());
    }
    for (tx, ty, p) in tile_payloads {
        out.extend_from_slice(&(tx as i32).to_le_bytes());
        out.extend_from_slice(&(ty as i32).to_le_bytes());
        out.extend_from_slice(&0i32.to_le_bytes()); // lvlx
        out.extend_from_slice(&0i32.to_le_bytes()); // lvly
        out.extend_from_slice(&(p.len() as i32).to_le_bytes());
        out.extend_from_slice(&p);
    }
    Ok(out)
}

/// Build the full attribute list for a tiled file, including channels
/// taken from the provided list (so callers using non-RGBA channel sets
/// can still encode tiled).
fn build_tiled_attributes(
    channels: &[Channel],
    width: u32,
    height: u32,
    compression: Compression,
    tile_x: u32,
    tile_y: u32,
    chunk_count: u32,
) -> Vec<Attribute> {
    // Reuse the canonical attribute builder for non-channel attrs, then
    // splice in the user-provided channel list.
    let mut attrs =
        rgba_float_tiled_attributes(width, height, compression, tile_x, tile_y, chunk_count);
    if let Some(ch_attr) = attrs.iter_mut().find(|a| a.name == "channels") {
        ch_attr.value = AttributeValue::Channels(channels.to_vec());
    }
    attrs
}

/// Compress one tile's raw byte stream according to the file's
/// compression mode (or return raw bytes if compression doesn't shrink
/// it — the spec mandates this).
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

/// Compress one tile under PXR24 / B44 / B44A. Unlike [`compress_tile_payload`]
/// (which works on the interleaved native `raw` stream), these schemes
/// reorganise the tile into byte-plane / per-channel forms and so consume the
/// tile-local f32 sub-planes (`sub_planes[i]` is channel `i`'s `tw × th`
/// row-major slice). The tile is treated as a single self-contained block:
/// width = `tw`, origin row 0, `th` rows (tiled files use 1×1 sampling).
///
/// `raw` is the interleaved native tile stream, used for the shared
/// raw-fallback decision (store the raw bytes when the reorganised payload
/// would not be smaller — observer-spec §0).
fn compress_tile_payload_reorg(
    raw: Vec<u8>,
    channels: &[Channel],
    sub_planes: &[&[f32]],
    tw: u32,
    th: usize,
    compression: Compression,
) -> Result<Vec<u8>> {
    Ok(match compression {
        Compression::Pxr24 => {
            crate::encoder::build_pxr24_block_payload(channels, sub_planes, tw, 0, th)?
        }
        Compression::B44 | Compression::B44a => {
            let flat = matches!(compression, Compression::B44a);
            let packed =
                crate::encoder::build_b44_block_payload(channels, sub_planes, tw, 0, th, flat);
            if packed.len() >= raw.len() {
                raw
            } else {
                packed
            }
        }
        _ => unreachable!("caller dispatches only PXR24 / B44 / B44A here"),
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
                assert_eq!(r[off], source_rgba[off * 4], "R mismatch ({x},{y})");
                assert_eq!(g[off], source_rgba[off * 4 + 1], "G mismatch ({x},{y})");
                assert_eq!(b[off], source_rgba[off * 4 + 2], "B mismatch ({x},{y})");
                assert_eq!(a[off], source_rgba[off * 4 + 3], "A mismatch ({x},{y})");
            }
        }
    }

    #[test]
    fn tiled_none_self_roundtrip_8x8_in_16x16() {
        let w = 16;
        let h = 16;
        let samples = make_image(w, h);
        let bytes =
            encode_exr_tiled_rgba_float_with(w, h, &samples, Compression::None, 8, 8).unwrap();
        let img = parse_exr(&bytes).unwrap();
        assert_eq!(img.width(), w);
        assert_eq!(img.height(), h);
        assert_planes_match_rgba(&img, &samples);
    }

    #[test]
    fn tiled_zip_self_roundtrip_8x8_in_16x16() {
        let w = 16;
        let h = 16;
        let samples = make_image(w, h);
        let bytes =
            encode_exr_tiled_rgba_float_with(w, h, &samples, Compression::Zip, 8, 8).unwrap();
        let img = parse_exr(&bytes).unwrap();
        assert_planes_match_rgba(&img, &samples);
    }

    #[test]
    fn tiled_zips_self_roundtrip_4x4_in_12x9_edge_tiles() {
        // 12×9 with 4×4 tiles: 3×3 grid; right column is full, bottom
        // row is partial (height=9, tile_y=4 → tiles 0..2 contain y=0..3
        // and y=4..7, tile 2 contains only y=8 (1 row).
        let w = 12;
        let h = 9;
        let samples = make_image(w, h);
        let bytes =
            encode_exr_tiled_rgba_float_with(w, h, &samples, Compression::Zips, 4, 4).unwrap();
        let img = parse_exr(&bytes).unwrap();
        assert_planes_match_rgba(&img, &samples);
    }

    #[test]
    fn tiled_rle_self_roundtrip_constant_image() {
        // RLE-friendly constant image: 32×32 with 16×16 tiles.
        let w = 32;
        let h = 32;
        let samples = vec![0.5_f32; (w * h * 4) as usize];
        let bytes =
            encode_exr_tiled_rgba_float_with(w, h, &samples, Compression::Rle, 16, 16).unwrap();
        let img = parse_exr(&bytes).unwrap();
        assert_planes_match_rgba(&img, &samples);
    }

    #[test]
    fn tiled_chunk_count_matches_grid() {
        // 32×16 with 8×8 tiles: 4×2 = 8 chunks.
        let w = 32;
        let h = 16;
        let samples = make_image(w, h);
        let bytes =
            encode_exr_tiled_rgba_float_with(w, h, &samples, Compression::None, 8, 8).unwrap();

        // Inspect the chunkCount attribute back out via parse_header.
        let header = crate::header::parse_header(&bytes).unwrap();
        let chunk_count_attr = header
            .attributes
            .iter()
            .find(|a| a.name == "chunkCount")
            .expect("encoder must emit chunkCount");
        match &chunk_count_attr.value {
            AttributeValue::Int(v) => assert_eq!(*v, 8),
            AttributeValue::Other { type_name, data } if type_name == "int" => {
                let v = i32::from_le_bytes(data[..4].try_into().unwrap());
                assert_eq!(v, 8);
            }
            other => {
                panic!("chunkCount should decode as Int (or legacy Other(int)); got {other:?}")
            }
        }
    }

    /// Mirror of the PXR24 24-bit FLOAT reduction (observer-spec §1.1) so a
    /// FLOAT-channel roundtrip can be asserted exactly against the lossy
    /// quantisation the scheme is defined to apply.
    fn pxr24_reduce(f: f32) -> f32 {
        let bits = f.to_bits();
        let s = bits & 0x8000_0000;
        let e = bits & 0x7f80_0000;
        let m = bits & 0x007f_ffff;
        let i: u32 = if e == 0x7f80_0000 {
            if m == 0 {
                e >> 8
            } else {
                let mm = m >> 8;
                (e >> 8) | mm | u32::from(mm == 0)
            }
        } else {
            let r = ((e | m) + (m & 0x80)) >> 8;
            if r >= 0x7f8000 {
                (e | m) >> 8
            } else {
                r
            }
        };
        let code24 = (s >> 8) | i;
        f32::from_bits((code24 & 0x00ff_ffff) << 8)
    }

    #[test]
    fn tiled_pxr24_float_roundtrip_8x8_in_16x16() {
        // PXR24 is lossy only for FLOAT (24-bit reduction); each decoded
        // sample must equal the spec reduction of the source value.
        let w = 16;
        let h = 16;
        let samples = make_image(w, h);
        let bytes =
            encode_exr_tiled_rgba_float_with(w, h, &samples, Compression::Pxr24, 8, 8).unwrap();
        let img = parse_exr(&bytes).unwrap();
        assert_eq!(img.width(), w);
        assert_eq!(img.height(), h);
        let (wi, hi) = (w as usize, h as usize);
        let a = &img.planes[0].samples;
        let b = &img.planes[1].samples;
        let g = &img.planes[2].samples;
        let r = &img.planes[3].samples;
        for y in 0..hi {
            for x in 0..wi {
                let off = y * wi + x;
                assert_eq!(r[off], pxr24_reduce(samples[off * 4]), "R ({x},{y})");
                assert_eq!(g[off], pxr24_reduce(samples[off * 4 + 1]), "G ({x},{y})");
                assert_eq!(b[off], pxr24_reduce(samples[off * 4 + 2]), "B ({x},{y})");
                assert_eq!(a[off], pxr24_reduce(samples[off * 4 + 3]), "A ({x},{y})");
            }
        }
    }

    #[test]
    fn tiled_pxr24_edge_tiles_roundtrip_4x4_in_12x9() {
        // Partial edge tiles (right column / bottom row smaller than the
        // tile size) must still reduce exactly per the PXR24 spec.
        let w = 12;
        let h = 9;
        let samples = make_image(w, h);
        let bytes =
            encode_exr_tiled_rgba_float_with(w, h, &samples, Compression::Pxr24, 4, 4).unwrap();
        let img = parse_exr(&bytes).unwrap();
        let (wi, hi) = (w as usize, h as usize);
        let r = &img.planes[3].samples;
        for y in 0..hi {
            for x in 0..wi {
                let off = y * wi + x;
                assert_eq!(r[off], pxr24_reduce(samples[off * 4]), "R ({x},{y})");
            }
        }
    }

    /// Build a single-HALF-channel tiled file (channel "Y") so the B44 HALF
    /// block packing is actually exercised (FLOAT/UINT channels are copied
    /// raw under B44).
    fn half_channel() -> Vec<Channel> {
        vec![Channel {
            name: "Y".to_string(),
            pixel_type: PixelType::Half,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        }]
    }

    #[test]
    fn tiled_b44_half_roundtrip_is_fixed_point() {
        // B44 HALF packing is lossy; encoding the decoded values again must
        // reproduce the same decoded image (a fixed point), which proves the
        // tile decode path inverts the tile encode path bit-for-bit.
        let w = 16;
        let h = 16;
        let chs = half_channel();
        let mut plane = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                plane.push(((x as f32) * 0.1 + (y as f32) * 0.05).sin() * 4.0);
            }
        }
        let bytes = encode_exr_tiled(w, h, &chs, &[&plane], Compression::B44, 8, 8).unwrap();
        let img = parse_exr(&bytes).unwrap();
        let decoded: Vec<f32> = img.planes[0].samples.clone();

        // Re-encode the decoded samples and decode again.
        let bytes2 = encode_exr_tiled(w, h, &chs, &[&decoded], Compression::B44, 8, 8).unwrap();
        let img2 = parse_exr(&bytes2).unwrap();
        assert_eq!(
            decoded, img2.planes[0].samples,
            "B44 tile decode must be a fixed point of the tile encode"
        );
    }

    #[test]
    fn tiled_b44a_flat_block_roundtrips_constant() {
        // A constant HALF image collapses to B44A 3-byte flat blocks; the
        // decode must recover the exact constant for every pixel.
        let w = 16;
        let h = 16;
        let chs = half_channel();
        // HALF-representable constant (0.5 is exact in binary16).
        let plane = vec![0.5_f32; (w * h) as usize];
        let bytes = encode_exr_tiled(w, h, &chs, &[&plane], Compression::B44a, 8, 8).unwrap();
        let img = parse_exr(&bytes).unwrap();
        for (i, &v) in img.planes[0].samples.iter().enumerate() {
            assert_eq!(v, 0.5, "B44A flat block pixel {i}");
        }
    }

    #[test]
    fn tiled_b44_edge_tiles_roundtrip_4x4_in_10x7() {
        // Partial edge tiles with the 4×4 B44 block edge-replication: the
        // decoded image must be a fixed point under re-encode.
        let w = 10;
        let h = 7;
        let chs = half_channel();
        let mut plane = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                plane.push((x as f32) * 0.3 - (y as f32) * 0.2);
            }
        }
        let bytes = encode_exr_tiled(w, h, &chs, &[&plane], Compression::B44, 4, 4).unwrap();
        let img = parse_exr(&bytes).unwrap();
        let decoded: Vec<f32> = img.planes[0].samples.clone();
        let bytes2 = encode_exr_tiled(w, h, &chs, &[&decoded], Compression::B44, 4, 4).unwrap();
        let img2 = parse_exr(&bytes2).unwrap();
        assert_eq!(decoded, img2.planes[0].samples, "B44 edge-tile fixed point");
    }

    #[test]
    fn tiled_rejects_subsampled_channels() {
        let w = 8;
        let h = 8;
        let chs = vec![Channel {
            name: "Y".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 2,
            y_sampling: 2,
        }];
        let plane: Vec<f32> = vec![0.0; (w * h) as usize];
        let r = encode_exr_tiled(w, h, &chs, &[&plane], Compression::None, 4, 4);
        assert!(r.is_err());
    }
}
