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

use crate::error::{ExrError, Result};
use crate::header::{encode_header, VersionField};
use crate::types::{Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType};

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
    if !matches!(compression, Compression::None | Compression::Zip) {
        return Err(ExrError::unsupported(format!(
            "compression {compression:?} (round-1 supports NO_COMPRESSION + ZIP)"
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
/// order as `channels`.
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
        if ch.x_sampling != 1 || ch.y_sampling != 1 {
            return Err(ExrError::unsupported(format!(
                "channel '{}' x_sampling={} y_sampling={} (round-2 followup)",
                ch.name, ch.x_sampling, ch.y_sampling
            )));
        }
        if ch.pixel_type == PixelType::Uint {
            return Err(ExrError::unsupported(format!(
                "channel '{}' pixelType=UINT (round-2 followup)",
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

    let bpp: usize = channels
        .iter()
        .map(|c| c.pixel_type.bytes_per_sample())
        .sum();
    let row_bytes = bpp * width as usize;
    let block_h = compression.scanlines_per_block();
    let num_blocks = height.div_ceil(block_h) as usize;

    // Emit the header.
    let header_bytes = encode_header(VersionField::from_u32(2), &attributes);

    // Build each block's payload (raw uncompressed bytes), then
    // compress if requested.
    let mut block_payloads: Vec<Vec<u8>> = Vec::with_capacity(num_blocks);
    for block_idx in 0..num_blocks {
        let row0 = block_idx as u32 * block_h;
        let lines_in_block = (height - row0).min(block_h) as usize;
        let mut raw = Vec::with_capacity(lines_in_block * row_bytes);
        for line in 0..lines_in_block {
            let y = row0 as usize + line;
            for (ch_idx, ch) in channels.iter().enumerate() {
                let plane = planes[ch_idx];
                for x in 0..width as usize {
                    let v = plane[y * width as usize + x];
                    match ch.pixel_type {
                        PixelType::Half => {
                            raw.extend_from_slice(&crate::half::f32_to_half(v).to_le_bytes())
                        }
                        PixelType::Float => raw.extend_from_slice(&v.to_le_bytes()),
                        PixelType::Uint => unreachable!("filtered above"),
                    }
                }
            }
        }
        let payload = match compression {
            Compression::None => raw,
            Compression::Zip => {
                let mut interleaved = vec![0u8; raw.len()];
                crate::decoder::apply_zip_interleave(&raw, &mut interleaved);
                crate::decoder::apply_zip_predictor(&mut interleaved);
                let compressed = zlib_deflate(&interleaved)?;
                if compressed.len() >= raw.len() {
                    // Spec rule: store whichever is smaller.
                    raw
                } else {
                    compressed
                }
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
    use flate2::write::ZlibEncoder;
    use flate2::Compression as FlateLevel;
    use std::io::Write;

    let mut enc = ZlibEncoder::new(Vec::new(), FlateLevel::default());
    enc.write_all(data)
        .map_err(|e| ExrError::invalid(format!("zlib deflate failed: {e}")))?;
    enc.finish()
        .map_err(|e| ExrError::invalid(format!("zlib finish failed: {e}")))
}
