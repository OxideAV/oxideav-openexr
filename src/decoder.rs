//! Top-level scanline EXR decoder.
//!
//! Walks the header (via [`crate::header::parse_header`]), reads the
//! line offset table, then iterates the scanline blocks: per block the
//! file stores `Y(i32) | size(i32) | payload(size bytes)`. Payload is
//! either uncompressed channel-then-row-major samples
//! (NO_COMPRESSION) or zlib-deflated bytes that, after decompression,
//! match the uncompressed layout — except that ZIP-family compression
//! pre-applies two reversible transforms documented at openexr.com:
//!
//!   1. interleave: low-byte half / high-byte half are concatenated so
//!      similar magnitudes in adjacent samples sit next to each other.
//!   2. predictor: each byte adds the previous one (mod 256), so most
//!      values become small deltas that deflate compresses well.
//!
//! Both transforms are byte-level and trivially invertible. The
//! [`apply_zip_unpredictor`] / [`apply_zip_uninterleave`] helpers below
//! match them; the encoder side does the inverse pair.

use crate::error::{ExrError, Result};
use crate::header::parse_header;
use crate::image::{ExrImage, ExrPlane};
use crate::types::{Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType};

/// Inverse of the ZIP predictor pass: each byte after the first adds
/// the previous one, modulo 256. After this pass byte i holds
/// `original[i]`.
pub fn apply_zip_unpredictor(buf: &mut [u8]) {
    for i in 1..buf.len() {
        buf[i] = buf[i].wrapping_add(buf[i - 1]);
    }
}

/// Forward ZIP predictor: each byte stores `original[i] - original[i-1]`
/// (modulo 256). The first byte is unchanged.
pub fn apply_zip_predictor(buf: &mut [u8]) {
    // Walk right-to-left so we read original values before overwriting.
    for i in (1..buf.len()).rev() {
        buf[i] = buf[i].wrapping_sub(buf[i - 1]);
    }
}

/// Inverse of the ZIP byte-half interleave: source has the first
/// `(N+1)/2` bytes from even-indexed positions in the output and the
/// remaining bytes from odd-indexed positions.
pub fn apply_zip_uninterleave(src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    let n = dst.len();
    let half = n.div_ceil(2);
    let mut s_even = 0;
    let mut s_odd = half;
    let mut i = 0;
    while i < n {
        dst[i] = src[s_even];
        s_even += 1;
        i += 1;
        if i < n {
            dst[i] = src[s_odd];
            s_odd += 1;
            i += 1;
        }
    }
}

/// Forward ZIP byte-half interleave (inverse of [`apply_zip_uninterleave`]).
pub fn apply_zip_interleave(src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    let n = src.len();
    let half = n.div_ceil(2);
    let mut d_even = 0;
    let mut d_odd = half;
    let mut i = 0;
    while i < n {
        dst[d_even] = src[i];
        d_even += 1;
        i += 1;
        if i < n {
            dst[d_odd] = src[i];
            d_odd += 1;
            i += 1;
        }
    }
}

/// Locate the first attribute with the given name (case-sensitive).
fn find_attribute<'a>(attrs: &'a [Attribute], name: &str) -> Option<&'a AttributeValue> {
    attrs.iter().find(|a| a.name == name).map(|a| &a.value)
}

/// Pull every required attribute out of the parsed header into a
/// shape that's easier to consume downstream.
struct RequiredAttrs {
    channels: Vec<Channel>,
    compression: Compression,
    data_window: Box2i,
    display_window: Box2i,
    line_order: LineOrder,
    pixel_aspect_ratio: f32,
    screen_window_center: (f32, f32),
    screen_window_width: f32,
}

fn extract_required(attrs: &[Attribute]) -> Result<RequiredAttrs> {
    let channels = match find_attribute(attrs, "channels") {
        Some(AttributeValue::Channels(c)) => c.clone(),
        _ => {
            return Err(ExrError::invalid(
                "missing required attribute 'channels'".to_string(),
            ))
        }
    };
    let compression = match find_attribute(attrs, "compression") {
        Some(AttributeValue::Compression(c)) => *c,
        _ => {
            return Err(ExrError::invalid(
                "missing required attribute 'compression'".to_string(),
            ))
        }
    };
    let data_window = match find_attribute(attrs, "dataWindow") {
        Some(AttributeValue::Box2i(b)) => *b,
        _ => {
            return Err(ExrError::invalid(
                "missing required attribute 'dataWindow'".to_string(),
            ))
        }
    };
    let display_window = match find_attribute(attrs, "displayWindow") {
        Some(AttributeValue::Box2i(b)) => *b,
        _ => {
            return Err(ExrError::invalid(
                "missing required attribute 'displayWindow'".to_string(),
            ))
        }
    };
    let line_order = match find_attribute(attrs, "lineOrder") {
        Some(AttributeValue::LineOrder(l)) => *l,
        _ => {
            return Err(ExrError::invalid(
                "missing required attribute 'lineOrder'".to_string(),
            ))
        }
    };
    let pixel_aspect_ratio = match find_attribute(attrs, "pixelAspectRatio") {
        Some(AttributeValue::Float(f)) => *f,
        _ => 1.0, // tolerate omission with a sane default
    };
    let screen_window_center = match find_attribute(attrs, "screenWindowCenter") {
        Some(AttributeValue::V2f(x, y)) => (*x, *y),
        _ => (0.0, 0.0),
    };
    let screen_window_width = match find_attribute(attrs, "screenWindowWidth") {
        Some(AttributeValue::Float(f)) => *f,
        _ => 1.0,
    };
    Ok(RequiredAttrs {
        channels,
        compression,
        data_window,
        display_window,
        line_order,
        pixel_aspect_ratio,
        screen_window_center,
        screen_window_width,
    })
}

/// Parse a single-part scanline EXR file from a byte slice.
pub fn parse_exr(bytes: &[u8]) -> Result<ExrImage> {
    let header = parse_header(bytes)?;
    let req = extract_required(&header.attributes)?;

    // Round-1 channel limitation: every channel must be 1×1 sampled. The
    // sub-sampling math (used by chroma sub-sampled deep YUV files) is
    // a round-2 followup.
    for ch in &req.channels {
        if ch.x_sampling != 1 || ch.y_sampling != 1 {
            return Err(ExrError::unsupported(format!(
                "channel '{}' uses xSampling={} ySampling={} (sub-sampled channels are a round-2 followup)",
                ch.name, ch.x_sampling, ch.y_sampling
            )));
        }
        if ch.pixel_type == PixelType::Uint {
            return Err(ExrError::unsupported(format!(
                "channel '{}' uses pixelType=UINT (round-2 followup; HALF + FLOAT only)",
                ch.name
            )));
        }
    }

    // Round-1 compression limitation.
    if !matches!(req.compression, Compression::None | Compression::Zip) {
        return Err(ExrError::unsupported(format!(
            "compression {:?} (round-1 supports NO_COMPRESSION + ZIP only)",
            req.compression
        )));
    }

    let width = req.data_window.width();
    let height = req.data_window.height();
    if width == 0 || height == 0 {
        return Err(ExrError::invalid(format!(
            "dataWindow width={width} height={height} — must both be > 0"
        )));
    }

    // Channels are stored alphabetically within each scanline block; the
    // chlist itself is also alphabetical in a well-formed file but not
    // strictly required to be so. Sort our local copy to match the
    // pixel-data layout.
    let mut sorted_channels = req.channels.clone();
    sorted_channels.sort_by(|a, b| a.name.cmp(&b.name));

    // Bytes per pixel (sum of sample sizes of all channels — assumes 1×1
    // sampling, enforced above).
    let bpp: usize = sorted_channels
        .iter()
        .map(|c| c.pixel_type.bytes_per_sample())
        .sum();
    let row_bytes = bpp * width as usize;

    // Per-block scanline count: depends on compression.
    let block_h = req.compression.scanlines_per_block();

    // Number of blocks = ceil(height / block_h).
    let num_blocks = height.div_ceil(block_h) as usize;

    // Read the offset table: `num_blocks` u64 LE entries directly
    // following the header NUL.
    let mut pos = header.end_offset;
    if pos + num_blocks * 8 > bytes.len() {
        return Err(ExrError::invalid(format!(
            "line offset table runs past EOF: need {} bytes at offset {pos}, file size {}",
            num_blocks * 8,
            bytes.len()
        )));
    }
    let mut offsets = Vec::with_capacity(num_blocks);
    for _ in 0..num_blocks {
        let off = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
        offsets.push(off as usize);
        pos += 8;
    }

    // Allocate per-channel f32 planes.
    let mut planes: Vec<ExrPlane> = sorted_channels
        .iter()
        .map(|c| ExrPlane {
            name: c.name.clone(),
            samples: vec![0.0; (width * height) as usize],
        })
        .collect();

    // Decode each block.
    for (block_idx, &block_off) in offsets.iter().enumerate() {
        if block_off + 8 > bytes.len() {
            return Err(ExrError::invalid(format!(
                "block {block_idx} offset {block_off} past EOF",
            )));
        }
        let y_coord = i32::from_le_bytes(bytes[block_off..block_off + 4].try_into().unwrap());
        let payload_size =
            i32::from_le_bytes(bytes[block_off + 4..block_off + 8].try_into().unwrap());
        if payload_size < 0 {
            return Err(ExrError::invalid(format!(
                "block {block_idx} negative size {payload_size}"
            )));
        }
        let payload_start = block_off + 8;
        let payload_end = payload_start + payload_size as usize;
        if payload_end > bytes.len() {
            return Err(ExrError::invalid(format!(
                "block {block_idx} payload runs past EOF (start {payload_start}, size {payload_size})"
            )));
        }
        let payload = &bytes[payload_start..payload_end];

        // Where in the data window does this block start?
        let row_in_image = (y_coord - req.data_window.y_min) as i64;
        if row_in_image < 0 || row_in_image as u32 >= height {
            return Err(ExrError::invalid(format!(
                "block {block_idx} Y={y_coord} outside dataWindow y_min={}, height={}",
                req.data_window.y_min, height
            )));
        }
        let lines_in_block = ((height - row_in_image as u32).min(block_h)) as usize;
        let uncompressed_size = lines_in_block * row_bytes;

        // Get the uncompressed bytes (decompress if needed).
        let uncompressed: Vec<u8> = match req.compression {
            Compression::None => {
                if payload.len() != uncompressed_size {
                    return Err(ExrError::invalid(format!(
                        "block {block_idx} uncompressed size mismatch: have {} want {}",
                        payload.len(),
                        uncompressed_size
                    )));
                }
                payload.to_vec()
            }
            Compression::Zip => {
                // ZIP rule: the file stores whichever is smaller, the
                // raw uncompressed bytes or the zlib-compressed +
                // pre-transformed bytes. So if payload.len() ==
                // uncompressed_size we MUST treat it as raw.
                if payload.len() == uncompressed_size {
                    payload.to_vec()
                } else {
                    let inflated = zlib_inflate(payload, uncompressed_size)?;
                    if inflated.len() != uncompressed_size {
                        return Err(ExrError::invalid(format!(
                            "block {block_idx} inflate produced {} bytes, expected {}",
                            inflated.len(),
                            uncompressed_size
                        )));
                    }
                    // Reverse the predictor pass first (it ran AFTER the
                    // interleave on encode, so it runs FIRST on decode).
                    let mut predicted = inflated;
                    apply_zip_unpredictor(&mut predicted);
                    let mut out = vec![0u8; uncompressed_size];
                    apply_zip_uninterleave(&predicted, &mut out);
                    out
                }
            }
            _ => unreachable!("filtered above"),
        };

        // Now scatter the uncompressed payload into the f32 planes.
        // Layout: row-major across `lines_in_block`; per row, channels
        // alphabetical; per channel, `width` samples; no padding.
        let mut p = 0usize;
        for line in 0..lines_in_block {
            let dst_y = row_in_image as usize + line;
            for (ch_idx, ch) in sorted_channels.iter().enumerate() {
                let ss = ch.pixel_type.bytes_per_sample();
                let plane = &mut planes[ch_idx].samples;
                for x in 0..width as usize {
                    let v = match ch.pixel_type {
                        PixelType::Half => {
                            let bits =
                                u16::from_le_bytes(uncompressed[p..p + 2].try_into().unwrap());
                            crate::half::half_to_f32(bits)
                        }
                        PixelType::Float => {
                            f32::from_le_bytes(uncompressed[p..p + 4].try_into().unwrap())
                        }
                        PixelType::Uint => unreachable!("filtered above"),
                    };
                    plane[dst_y * width as usize + x] = v;
                    p += ss;
                }
            }
        }
        if p != uncompressed.len() {
            return Err(ExrError::invalid(format!(
                "block {block_idx} consumed {p} of {} payload bytes",
                uncompressed.len()
            )));
        }
    }

    Ok(ExrImage {
        data_window: req.data_window,
        display_window: req.display_window,
        line_order: req.line_order,
        compression: req.compression,
        pixel_aspect_ratio: req.pixel_aspect_ratio,
        screen_window_center: req.screen_window_center,
        screen_window_width: req.screen_window_width,
        channels: sorted_channels,
        planes,
        attributes: header.attributes,
    })
}

/// zlib-decompress `data` into a buffer at most `expected_size` bytes
/// long. We use `flate2`'s `ZlibDecoder` (pure-Rust `miniz_oxide`
/// backend per Cargo.toml).
fn zlib_inflate(data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    use flate2::read::ZlibDecoder;
    use std::io::Read;

    let mut out = Vec::with_capacity(expected_size);
    let mut dec = ZlibDecoder::new(data);
    dec.read_to_end(&mut out)
        .map_err(|e| ExrError::invalid(format!("zlib inflate failed: {e}")))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predictor_roundtrip() {
        let original: Vec<u8> = (0..255).step_by(7).collect();
        let mut buf = original.clone();
        apply_zip_predictor(&mut buf);
        apply_zip_unpredictor(&mut buf);
        assert_eq!(buf, original);
    }

    #[test]
    fn interleave_roundtrip_even() {
        let src: Vec<u8> = (0..16).collect();
        let mut mid = vec![0u8; 16];
        apply_zip_interleave(&src, &mut mid);
        let mut back = vec![0u8; 16];
        apply_zip_uninterleave(&mid, &mut back);
        assert_eq!(src, back);
    }

    #[test]
    fn interleave_roundtrip_odd() {
        let src: Vec<u8> = (0..15).collect();
        let mut mid = vec![0u8; 15];
        apply_zip_interleave(&src, &mut mid);
        let mut back = vec![0u8; 15];
        apply_zip_uninterleave(&mid, &mut back);
        assert_eq!(src, back);
    }
}
