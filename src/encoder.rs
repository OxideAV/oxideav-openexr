//! Scanline EXR encoder. Round-1 supports a fixed-shape RGBA float
//! pipeline ([`encode_exr_scanline_rgba_float`]) plus a more general
//! per-channel writer ([`encode_exr_scanline`]) used by the round-trip
//! tests.
//!
//! Layout we emit:
//!
//! * magic + version (format-version 2, no flag bits).
//! * header attributes: channels, compression, dataWindow,
//!   displayWindow, lineOrder, pixelAspectRatio, screenWindowCenter,
//!   screenWindowWidth.
//! * NUL header terminator.
//! * line offset table (`num_blocks` u64 LE entries).
//! * scanline blocks, each `Y(i32) | size(i32) | payload(size bytes)`.
//!
//! For NO_COMPRESSION the payload is exactly the row-major,
//! channel-alphabetical, sample-flat byte stream the spec describes.
//! For ZIP we prepend the spec-mandated interleave + predictor
//! transforms and run the result through zlib. If the compressed-plus-
//! transformed buffer would be larger than the raw payload, the spec
//! says we MUST emit the raw bytes — that case is handled below.
//!
//! Sub-sampled channels (`xSampling != 1` or `ySampling != 1`): for each
//! image scanline `y`, only channels with `y % ySampling == 0`
//! contribute samples to the block payload (per the OpenEXR spec),
//! and each contributing channel writes `subsampled_dim(width, xSampling)`
//! samples. The per-channel plane the caller supplies must already be
//! sized to its sub-sampled dimensions (matches the decoder's `ExrPlane`
//! layout).

use crate::decoder::subsampled_dim;
use crate::error::{ExrError, Result};
use crate::header::{encode_header, VersionField};
use crate::types::{Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType};

/// Reduce a binary32 word to its PXR24 24-bit code (observer-spec §1.1):
/// round the mantissa to 15 bits and discard the low byte of the 32-bit
/// word. The returned value occupies the low 24 bits of the `u32`.
///
/// This is the exact inverse-target of [`crate::decoder`]'s
/// `pxr24_code_to_f32_bits`: the decoder places this 24-bit code back
/// into the top 24 bits of a binary32, so a sample round-trips to
/// `f32::from_bits(code << 8)`.
fn pxr24_f32_to_code24(bits: u32) -> u32 {
    let s = bits & 0x8000_0000;
    let e = bits & 0x7f80_0000;
    let m = bits & 0x007f_ffff;
    if e == 0x7f80_0000 {
        // inf / NaN: keep the exponent (all ones), carry the top 15
        // mantissa bits. A NaN whose top 15 mantissa bits are all zero
        // would collapse to infinity, so force at least one bit set.
        if m == 0 {
            e >> 8
        } else {
            let mm = m >> 8;
            (e >> 8) | mm | u32::from(mm == 0)
        }
    } else {
        // Finite: add the round bit (bit 7 of the mantissa) into (e|m)
        // and shift right by 8. If rounding carries the exponent past
        // the maximum, redo by truncation instead.
        let mut i = ((e | m) + (m & 0x80)) >> 8;
        if i >= 0x7f_8000 {
            i = (e | m) >> 8;
        }
        (s >> 8) | i
    }
}

/// Build the standard 4-channel RGBA float header attribute set.
fn rgba_float_attributes(width: u32, height: u32, compression: Compression) -> Vec<Attribute> {
    // chlist must be alphabetical: A, B, G, R.
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
    vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(chs),
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
    ]
}

/// Reduced on-the-wire byte width of a channel under PXR24: FLOAT is
/// carried as 3 bytes (24-bit reduction), HALF as 2, UINT as 4.
/// Mirrors the decoder's `pxr24_channel_bytes`.
fn pxr24_channel_bytes(pt: PixelType) -> usize {
    match pt {
        PixelType::Float => 3,
        PixelType::Half => 2,
        PixelType::Uint => 4,
    }
}

/// Build the PXR24 reorganised (byte-plane + horizontal-delta) stream for
/// one scanline block, then zlib-deflate it (observer-spec §1.2/§1.3).
///
/// Visiting order matches the decoder exactly: rows top-to-bottom, then
/// channels in sorted order, then byte planes most-significant first,
/// plane-major across the row. Each plane byte is the horizontal delta of
/// the sample's integer code against the previous sample of the same
/// channel on the same row (prediction resets to 0 per channel/row).
/// FLOAT codes are the 24-bit reduction; HALF/UINT codes are the raw
/// little-endian sample value widened to `u32`.
///
/// Returns the zlib stream, or — per the universal raw-fallback rule —
/// the reorganised bytes themselves when deflate did not shrink them. The
/// caller distinguishes the two by comparing the returned length against
/// the uncompressed (native) block size, exactly as the decoder does.
fn build_pxr24_block_payload(
    channels: &[Channel],
    planes: &[&[f32]],
    width: u32,
    block_y0: u32,
    lines_in_block: usize,
) -> Result<Vec<u8>> {
    // Size the reorganised stream: sum over present rows/channels of
    // (reduced bytes per sample) * (sub-width).
    let mut reorg_size = 0usize;
    for line in 0..lines_in_block as u32 {
        let dst_y = block_y0 + line;
        for ch in channels {
            let ys = ch.y_sampling as u32;
            if dst_y % ys != 0 {
                continue;
            }
            let pw = subsampled_dim(width, ch.x_sampling as u32) as usize;
            reorg_size += pxr24_channel_bytes(ch.pixel_type) * pw;
        }
    }

    let mut reorg = vec![0u8; reorg_size];
    let mut wp = 0usize;
    for line in 0..lines_in_block as u32 {
        let dst_y = block_y0 + line;
        for (ch_idx, ch) in channels.iter().enumerate() {
            let ys = ch.y_sampling as u32;
            if dst_y % ys != 0 {
                continue;
            }
            let xs = ch.x_sampling as u32;
            let pw = subsampled_dim(width, xs) as usize;
            let nbytes = pxr24_channel_bytes(ch.pixel_type);
            let plane_y = (dst_y / ys) as usize;
            let plane = planes[ch_idx];

            // Emit one byte-plane per reduced byte, most-significant
            // first, plane-major. `prev` is the previous sample's code on
            // this channel/row (resets to 0 at the row start), and the
            // delta is taken modulo 2^32 (wrapping), matching the
            // decoder's running prefix-sum.
            let mut prev: u32 = 0;
            for x in 0..pw {
                let v = plane[plane_y * pw + x];
                let code = match ch.pixel_type {
                    PixelType::Float => pxr24_f32_to_code24(v.to_bits()),
                    PixelType::Half => u32::from(crate::half::f32_to_half(v)),
                    PixelType::Uint => {
                        if v.is_nan() || v < 0.0 {
                            0u32
                        } else if v >= (u32::MAX as f32) {
                            u32::MAX
                        } else {
                            (v + 0.5) as u32
                        }
                    }
                };
                let diff = code.wrapping_sub(prev);
                prev = code;
                // Most-significant reduced byte first. For FLOAT the code
                // is 24-bit so bytes are diff>>16, diff>>8, diff; HALF is
                // diff>>8, diff; UINT is diff>>24..diff.
                for b in 0..nbytes {
                    let shift = 8 * (nbytes - 1 - b);
                    reorg[wp + b * pw + x] = (diff >> shift) as u8;
                }
            }
            wp += nbytes * pw;
        }
    }
    debug_assert_eq!(wp, reorg_size);

    let compressed = zlib_deflate(&reorg)?;
    // Universal raw-fallback (observer-spec §0): if deflate did not
    // shrink the reorganised stream, store it uncompressed. The decoder
    // detects this by `payload.len() == reorg_size`.
    Ok(if compressed.len() >= reorg.len() {
        reorg
    } else {
        compressed
    })
}

/// Encode a width × height RGBA-float scanline EXR with the requested
/// compression. `samples` is `width * height * 4` long, in `R, G, B, A`
/// pixel order.
///
/// This is the primary public encode entry point — `parse_exr` of the
/// returned bytes round-trips back to the input pixels (modulo the
/// HALF<->FLOAT precision ladder if you ever change the channel
/// declarations to HALF).
pub fn encode_exr_scanline_rgba_float(width: u32, height: u32, samples: &[f32]) -> Result<Vec<u8>> {
    encode_exr_scanline_rgba_float_with(width, height, samples, Compression::Zip)
}

/// Same as [`encode_exr_scanline_rgba_float`] but with an explicit
/// compression mode (round 1 supports NO_COMPRESSION + ZIP).
pub fn encode_exr_scanline_rgba_float_with(
    width: u32,
    height: u32,
    samples: &[f32],
    compression: Compression,
) -> Result<Vec<u8>> {
    let need = (width as usize) * (height as usize) * 4;
    if samples.len() != need {
        return Err(ExrError::invalid(format!(
            "samples length {} != width({width})*height({height})*4 = {need}",
            samples.len()
        )));
    }
    if !matches!(
        compression,
        Compression::None
            | Compression::Zip
            | Compression::Zips
            | Compression::Rle
            | Compression::Pxr24
    ) {
        return Err(ExrError::unsupported(format!(
            "compression {compression:?} (encoder supports NONE + ZIP + ZIPS + RLE + PXR24; PIZ/B44 read-only or deferred)"
        )));
    }

    // Reshape to a per-channel plane vector matching the alphabetical
    // chlist order A, B, G, R. The input is interleaved RGBA so the
    // per-pixel offsets are R=0, G=1, B=2, A=3.
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

    let chs = match &rgba_float_attributes(width, height, compression)[0].value {
        AttributeValue::Channels(c) => c.clone(),
        _ => unreachable!(),
    };
    let planes_f32: Vec<&[f32]> = vec![&a, &b, &g, &r];
    encode_exr_scanline(
        width,
        height,
        &chs,
        &planes_f32,
        compression,
        rgba_float_attributes(width, height, compression),
    )
}

/// General-purpose scanline encoder. `planes` must contain one
/// `width * height` `f32` slice per channel in the same alphabetical
/// order as `channels`. UINT channels store the f32 value rounded to
/// nearest u32 (clamped to `[0, u32::MAX as f32]`).
pub fn encode_exr_scanline(
    width: u32,
    height: u32,
    channels: &[Channel],
    planes: &[&[f32]],
    compression: Compression,
    attributes: Vec<Attribute>,
) -> Result<Vec<u8>> {
    if channels.len() != planes.len() {
        return Err(ExrError::invalid(format!(
            "channels.len()={} != planes.len()={}",
            channels.len(),
            planes.len()
        )));
    }
    for (ch, p) in channels.iter().zip(planes.iter()) {
        if ch.x_sampling <= 0 || ch.y_sampling <= 0 {
            return Err(ExrError::invalid(format!(
                "channel '{}' x_sampling={} y_sampling={} (must be positive)",
                ch.name, ch.x_sampling, ch.y_sampling
            )));
        }
        let pw = subsampled_dim(width, ch.x_sampling as u32) as usize;
        let ph = subsampled_dim(height, ch.y_sampling as u32) as usize;
        let need = pw * ph;
        if p.len() != need {
            return Err(ExrError::invalid(format!(
                "channel '{}' plane length {} != subsampled width*height = {pw}*{ph} = {need}",
                ch.name,
                p.len()
            )));
        }
    }

    // Verify channels are sorted alphabetically (file layout requires
    // alphabetical pixel data order; we keep the chlist matching).
    for win in channels.windows(2) {
        if win[0].name >= win[1].name {
            return Err(ExrError::invalid(format!(
                "channels not in alphabetical order: '{}' >= '{}'",
                win[0].name, win[1].name
            )));
        }
    }

    if !matches!(
        compression,
        Compression::None
            | Compression::Zip
            | Compression::Zips
            | Compression::Rle
            | Compression::Pxr24
    ) {
        return Err(ExrError::unsupported(format!(
            "compression {compression:?} (encoder supports NONE + ZIP + ZIPS + RLE + PXR24)"
        )));
    }

    let block_h = compression.scanlines_per_block();
    let num_blocks = height.div_ceil(block_h) as usize;

    // Emit the header.
    let header_bytes = encode_header(VersionField::from_u32(2), &attributes);

    // Build each block's payload (raw uncompressed bytes), then
    // compress if requested. With sub-sampling, the per-line bytes
    // depend on which channels' y_sampling divide the image row: only
    // those channels contribute samples, and each contributes its
    // sub-sampled-width count.
    let mut block_payloads: Vec<Vec<u8>> = Vec::with_capacity(num_blocks);
    for block_idx in 0..num_blocks {
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
                let pw = subsampled_dim(width, xs) as usize;
                let plane_y = y / ys as usize;
                let plane = planes[ch_idx];
                for x in 0..pw {
                    let v = plane[plane_y * pw + x];
                    match ch.pixel_type {
                        PixelType::Half => {
                            raw.extend_from_slice(&crate::half::f32_to_half(v).to_le_bytes())
                        }
                        PixelType::Float => raw.extend_from_slice(&v.to_le_bytes()),
                        PixelType::Uint => {
                            // Round to nearest, clamp to u32 range, then
                            // emit as little-endian u32. NaN and negatives
                            // both map to 0 (collapse the two clauses to
                            // satisfy clippy::if_same_then_else).
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
        let payload = match compression {
            Compression::None => raw,
            Compression::Zip | Compression::Zips => {
                let mut interleaved = vec![0u8; raw.len()];
                crate::decoder::apply_zip_interleave(&raw, &mut interleaved);
                crate::decoder::apply_zip_predictor(&mut interleaved);
                let compressed = zlib_deflate(&interleaved)?;
                if compressed.len() >= raw.len() {
                    raw // Spec rule: store whichever is smaller.
                } else {
                    compressed
                }
            }
            Compression::Rle => {
                let mut interleaved = vec![0u8; raw.len()];
                crate::decoder::apply_zip_interleave(&raw, &mut interleaved);
                crate::decoder::apply_zip_predictor(&mut interleaved);
                let compressed = crate::rle::rle_compress(&interleaved);
                if compressed.len() >= raw.len() {
                    raw // Spec: store whichever is smaller.
                } else {
                    compressed
                }
            }
            Compression::Pxr24 => {
                // PXR24 reorganises the FLOAT-reduced / HALF / UINT codes
                // into byte-plane + horizontal-delta form straight from
                // the planes (a different layout from the native `raw`
                // stream), then zlib-deflates with a raw fallback.
                build_pxr24_block_payload(channels, planes, width, row0, lines_in_block)?
            }
            _ => unreachable!("filtered above"),
        };
        block_payloads.push(payload);
    }

    // Build the final byte stream:
    //   header | offset table | block headers + payloads
    // Offsets are absolute byte positions into the file, so we have to
    // compute the offset of each block first.
    let offset_table_size = num_blocks * 8;
    let mut block_offsets = Vec::with_capacity(num_blocks);
    let mut running = header_bytes.len() + offset_table_size;
    for p in &block_payloads {
        block_offsets.push(running as u64);
        running += 8 + p.len(); // 8 bytes for Y(i32) + size(i32)
    }

    let mut out = Vec::with_capacity(running);
    out.extend_from_slice(&header_bytes);
    for &off in &block_offsets {
        out.extend_from_slice(&off.to_le_bytes());
    }
    for (block_idx, p) in block_payloads.iter().enumerate() {
        let y = block_idx as u32 * block_h;
        out.extend_from_slice(&(y as i32).to_le_bytes());
        out.extend_from_slice(&(p.len() as i32).to_le_bytes());
        out.extend_from_slice(p);
    }

    Ok(out)
}

fn zlib_deflate(data: &[u8]) -> Result<Vec<u8>> {
    zlib_deflate_pub(data)
}

/// Public-in-crate alias of [`zlib_deflate`] for use by sibling encoder
/// modules (tile_encoder, multipart_encoder).
pub(crate) fn zlib_deflate_pub(data: &[u8]) -> Result<Vec<u8>> {
    use flate2::write::ZlibEncoder;
    use flate2::Compression as FlateLevel;
    use std::io::Write;

    let mut enc = ZlibEncoder::new(Vec::new(), FlateLevel::default());
    enc.write_all(data)
        .map_err(|e| ExrError::invalid(format!("zlib deflate failed: {e}")))?;
    enc.finish()
        .map_err(|e| ExrError::invalid(format!("zlib finish failed: {e}")))
}
