//! Top-level EXR decoder (scanline + tiled).
//!
//! Walks the header (via [`crate::header::parse_header`]), reads the
//! offset table, then iterates the data chunks. For scanline files
//! each chunk is `Y(i32) | size(i32) | payload(size bytes)`. For tiled
//! files each chunk is `tx(i32) | ty(i32) | lvlx(i32) | lvly(i32) |
//! size(i32) | payload(size bytes)`.
//!
//! Compression coverage (round 2): NONE, ZIP, ZIPS, RLE.
//!
//! ZIP-family compression pre-applies two reversible transforms
//! documented at openexr.com:
//!   1. interleave: low-byte half / high-byte half are concatenated so
//!      similar magnitudes in adjacent samples sit next to each other.
//!   2. predictor: each byte adds the previous one (mod 256), so most
//!      values become small deltas that compress well.
//!
//! Both transforms are byte-level and trivially invertible — see
//! [`apply_zip_unpredictor`] / [`apply_zip_uninterleave`] (the encoder
//! side does the inverse pair).

use crate::error::{ExrError, Result};
use crate::header::parse_header;
use crate::image::{ExrImage, ExrPlane};
use crate::rle::rle_decompress;
use crate::tiled::tiledesc_from_attribute;
use crate::types::{Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType};

/// Inverse of the ZIP predictor pass per the openexr.com spec.
///
/// Spec encoder formula: `out[i] = (raw[i] - prev_raw + 128 + 256) & 0xFF`
/// (the `+128` recenters the typical-small delta on byte 0x80, which
/// helps the entropy coder).
///
/// Spec decoder inverse: `raw[i] = (in[i] + prev_raw - 128) & 0xFF`,
/// where `prev_raw` is the just-recovered byte at position i-1.
pub fn apply_zip_unpredictor(buf: &mut [u8]) {
    if buf.is_empty() {
        return;
    }
    let mut prev = buf[0];
    for slot in buf.iter_mut().skip(1) {
        let v = ((*slot as u32 + prev as u32).wrapping_sub(128) & 0xFF) as u8;
        *slot = v;
        prev = v;
    }
}

/// Forward ZIP predictor (inverse of [`apply_zip_unpredictor`]):
/// `out[i] = (raw[i] - prev_raw + 128) & 0xFF`. The first byte is
/// unchanged.
pub fn apply_zip_predictor(buf: &mut [u8]) {
    // Walk left-to-right over the original values, snapshotting prev
    // before overwriting. (Right-to-left also works but tracking
    // prev avoids re-reading already-overwritten cells.)
    if buf.is_empty() {
        return;
    }
    let mut prev = buf[0];
    for slot in buf.iter_mut().skip(1) {
        let cur = *slot;
        let d = ((cur as u32 + 128).wrapping_sub(prev as u32) & 0xFF) as u8;
        *slot = d;
        prev = cur;
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

/// Compute the sub-sampled dimension for a given sampling factor: the
/// channel holds `dim.div_ceil(sampling)` samples along that axis.
pub fn subsampled_dim(dim: u32, sampling: u32) -> u32 {
    if sampling <= 1 {
        return dim;
    }
    dim.div_ceil(sampling)
}

/// Locate the first attribute with the given name (case-sensitive).
fn find_attribute<'a>(attrs: &'a [Attribute], name: &str) -> Option<&'a AttributeValue> {
    attrs.iter().find(|a| a.name == name).map(|a| &a.value)
}

/// Pull every required attribute out of the parsed header.
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
        _ => 1.0,
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

/// Reverse the ZIP-family preprocessing pipeline (uninterleave +
/// unpredict). Operates on `payload` in place after copying.
fn undo_zip_pipeline(raw: Vec<u8>) -> Vec<u8> {
    let mut predicted = raw;
    apply_zip_unpredictor(&mut predicted);
    let mut out = vec![0u8; predicted.len()];
    apply_zip_uninterleave(&predicted, &mut out);
    out
}

/// Decode a ZIP / ZIPS payload (same algorithm; the per-block scanline
/// count differs but that's handled outside).
fn decode_zip_payload(payload: &[u8], uncompressed_size: usize) -> Result<Vec<u8>> {
    if payload.len() == uncompressed_size {
        // Spec: encoder may emit raw bytes if zlib doesn't shrink.
        return Ok(payload.to_vec());
    }
    let inflated = zlib_inflate(payload, uncompressed_size)?;
    if inflated.len() != uncompressed_size {
        return Err(ExrError::invalid(format!(
            "ZIP inflate produced {} bytes, expected {uncompressed_size}",
            inflated.len()
        )));
    }
    Ok(undo_zip_pipeline(inflated))
}

/// Decode an RLE payload (RLE → predictor → interleave inverse).
fn decode_rle_payload(payload: &[u8], uncompressed_size: usize) -> Result<Vec<u8>> {
    if payload.len() == uncompressed_size {
        return Ok(payload.to_vec());
    }
    let raw = rle_decompress(payload, uncompressed_size)?;
    Ok(undo_zip_pipeline(raw))
}

/// Decode one byte-flat scanline-block payload into the per-channel
/// f32 planes. `block_y0` is the top row of the block within the data
/// window; `lines_in_block` is the number of image rows the block
/// covers. Sub-sampled channels skip image rows that aren't divisible
/// by their `y_sampling` factor.
#[allow(clippy::too_many_arguments)]
fn scatter_block_into_planes(
    uncompressed: &[u8],
    sorted_channels: &[Channel],
    planes: &mut [ExrPlane],
    width: u32,
    height: u32,
    block_y0: u32,
    lines_in_block: usize,
) -> Result<()> {
    let _ = height;
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
                        // UINT decodes as a u32 → f32 view. Bit-exact
                        // recovery up to 2^24; beyond that the f32
                        // mantissa starts rounding. UINT producers are
                        // typically integer ID/depth maps that fit in
                        // 24 bits, so this matches the openexr.com
                        // reference's "as float" behaviour.
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
            "block consumed {p} of {} payload bytes",
            uncompressed.len()
        )));
    }
    Ok(())
}

/// Parse a single-part EXR file (scanline OR tiled) from a byte slice.
pub fn parse_exr(bytes: &[u8]) -> Result<ExrImage> {
    let header = parse_header(bytes)?;
    let req = extract_required(&header.attributes)?;

    if !matches!(
        req.compression,
        Compression::None | Compression::Zip | Compression::Zips | Compression::Rle
    ) {
        return Err(ExrError::unsupported(format!(
            "compression {:?} (round-2 supports NONE + ZIP + ZIPS + RLE; PIZ/B44/B44A/DWAA/DWAB/Pxr24 deferred)",
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

    let mut sorted_channels = req.channels.clone();
    sorted_channels.sort_by(|a, b| a.name.cmp(&b.name));

    for ch in &sorted_channels {
        if ch.x_sampling <= 0 || ch.y_sampling <= 0 {
            return Err(ExrError::invalid(format!(
                "channel '{}' has non-positive sampling factor x={} y={}",
                ch.name, ch.x_sampling, ch.y_sampling
            )));
        }
    }

    if header.version.single_tile {
        // Tiled file path. Sub-sampled tiles are uncommon and kept out
        // of round 2.
        if sorted_channels
            .iter()
            .any(|c| c.x_sampling != 1 || c.y_sampling != 1)
        {
            return Err(ExrError::unsupported(
                "tiled + sub-sampled channels (round-3 followup)".to_string(),
            ));
        }
        return parse_tiled(bytes, &header, &req, &sorted_channels);
    }

    // Per-block scanline count: depends on compression.
    let block_h = req.compression.scanlines_per_block();
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

    // Allocate per-channel f32 planes at each channel's sub-sampled size.
    let mut planes: Vec<ExrPlane> = sorted_channels
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

    for (block_idx, &block_off) in offsets.iter().enumerate() {
        if block_off + 8 > bytes.len() {
            return Err(ExrError::invalid(format!(
                "block {block_idx} offset {block_off} past EOF"
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

        let row_in_image = (y_coord - req.data_window.y_min) as i64;
        if row_in_image < 0 || row_in_image as u32 >= height {
            return Err(ExrError::invalid(format!(
                "block {block_idx} Y={y_coord} outside dataWindow y_min={}, height={}",
                req.data_window.y_min, height
            )));
        }
        let lines_in_block = ((height - row_in_image as u32).min(block_h)) as usize;

        // Per-block uncompressed size = sum over channels of
        // (effective per-channel rows in this block) * (sub-sampled
        // width) * (bytes per sample).
        let block_y0 = row_in_image as u32;
        let uncompressed_size: usize = sorted_channels
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
            Compression::Zip | Compression::Zips => decode_zip_payload(payload, uncompressed_size)?,
            Compression::Rle => decode_rle_payload(payload, uncompressed_size)?,
            _ => unreachable!("filtered above"),
        };

        scatter_block_into_planes(
            &uncompressed,
            &sorted_channels,
            &mut planes,
            width,
            height,
            block_y0,
            lines_in_block,
        )?;
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

/// Decode a tiled single-part EXR file. ONE_LEVEL only.
fn parse_tiled(
    bytes: &[u8],
    header: &crate::header::ParsedHeader,
    req: &RequiredAttrs,
    sorted_channels: &[Channel],
) -> Result<ExrImage> {
    let tdesc_attr = header
        .attributes
        .iter()
        .find(|a| a.name == "tiles")
        .ok_or_else(|| {
            ExrError::invalid(
                "tiled file missing required `tiles` attribute (tiledesc)".to_string(),
            )
        })?;
    let tdesc = tiledesc_from_attribute(&tdesc_attr.value)?;
    if tdesc.level_mode != 0 {
        return Err(ExrError::unsupported(format!(
            "tiled level mode {} (only ONE_LEVEL supported in round 2; mip/rip-map deferred)",
            tdesc.level_mode
        )));
    }
    if tdesc.x_size == 0 || tdesc.y_size == 0 {
        return Err(ExrError::invalid(format!(
            "tiledesc x_size={} y_size={} — both must be > 0",
            tdesc.x_size, tdesc.y_size
        )));
    }

    let width = req.data_window.width();
    let height = req.data_window.height();
    let tiles_x = width.div_ceil(tdesc.x_size) as usize;
    let tiles_y = height.div_ceil(tdesc.y_size) as usize;
    let num_tiles = tiles_x * tiles_y;

    let mut pos = header.end_offset;
    if pos + num_tiles * 8 > bytes.len() {
        return Err(ExrError::invalid(format!(
            "tile offset table runs past EOF (need {} bytes at {pos}, file size {})",
            num_tiles * 8,
            bytes.len()
        )));
    }
    let mut offsets = Vec::with_capacity(num_tiles);
    for _ in 0..num_tiles {
        offsets.push(u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize);
        pos += 8;
    }

    let mut planes: Vec<ExrPlane> = sorted_channels
        .iter()
        .map(|c| ExrPlane {
            name: c.name.clone(),
            samples: vec![0.0; (width * height) as usize],
        })
        .collect();
    let bpp: usize = sorted_channels
        .iter()
        .map(|c| c.pixel_type.bytes_per_sample())
        .sum();

    for (tile_idx, &tile_off) in offsets.iter().enumerate() {
        let tx = tile_idx % tiles_x;
        let ty = tile_idx / tiles_x;
        if tile_off + 20 > bytes.len() {
            return Err(ExrError::invalid(format!(
                "tile {tile_idx} offset {tile_off} past EOF"
            )));
        }
        let h_tx = i32::from_le_bytes(bytes[tile_off..tile_off + 4].try_into().unwrap());
        let h_ty = i32::from_le_bytes(bytes[tile_off + 4..tile_off + 8].try_into().unwrap());
        let lvl_x = i32::from_le_bytes(bytes[tile_off + 8..tile_off + 12].try_into().unwrap());
        let lvl_y = i32::from_le_bytes(bytes[tile_off + 12..tile_off + 16].try_into().unwrap());
        let payload_size =
            i32::from_le_bytes(bytes[tile_off + 16..tile_off + 20].try_into().unwrap());
        if lvl_x != 0 || lvl_y != 0 {
            return Err(ExrError::unsupported(format!(
                "tile {tile_idx} at level ({lvl_x},{lvl_y}) — multi-level deferred"
            )));
        }
        if h_tx as usize != tx || h_ty as usize != ty {
            return Err(ExrError::invalid(format!(
                "tile header coords ({h_tx},{h_ty}) do not match table position ({tx},{ty})",
            )));
        }
        if payload_size < 0 {
            return Err(ExrError::invalid(format!(
                "tile {tile_idx} negative payload size {payload_size}"
            )));
        }
        let pl_start = tile_off + 20;
        let pl_end = pl_start + payload_size as usize;
        if pl_end > bytes.len() {
            return Err(ExrError::invalid(format!(
                "tile {tile_idx} payload runs past EOF"
            )));
        }
        let payload = &bytes[pl_start..pl_end];

        let x0 = (tx as u32) * tdesc.x_size;
        let y0 = (ty as u32) * tdesc.y_size;
        let x1 = (x0 + tdesc.x_size).min(width);
        let y1 = (y0 + tdesc.y_size).min(height);
        let tw = (x1 - x0) as usize;
        let th = (y1 - y0) as usize;
        let uncompressed_size = tw * th * bpp;

        let uncompressed: Vec<u8> = match req.compression {
            Compression::None => {
                if payload.len() != uncompressed_size {
                    return Err(ExrError::invalid(format!(
                        "tile {tile_idx} uncompressed size mismatch: have {} want {}",
                        payload.len(),
                        uncompressed_size
                    )));
                }
                payload.to_vec()
            }
            Compression::Zip | Compression::Zips => decode_zip_payload(payload, uncompressed_size)?,
            Compression::Rle => decode_rle_payload(payload, uncompressed_size)?,
            _ => unreachable!("filtered above"),
        };

        let mut p = 0usize;
        for line in 0..th {
            let dst_y = y0 as usize + line;
            for (ch_idx, ch) in sorted_channels.iter().enumerate() {
                let plane = &mut planes[ch_idx].samples;
                for x in 0..tw {
                    let dst_x = x0 as usize + x;
                    let v = match ch.pixel_type {
                        PixelType::Half => {
                            let bits =
                                u16::from_le_bytes(uncompressed[p..p + 2].try_into().unwrap());
                            crate::half::half_to_f32(bits)
                        }
                        PixelType::Float => {
                            f32::from_le_bytes(uncompressed[p..p + 4].try_into().unwrap())
                        }
                        PixelType::Uint => {
                            let bits =
                                u32::from_le_bytes(uncompressed[p..p + 4].try_into().unwrap());
                            bits as f32
                        }
                    };
                    plane[dst_y * width as usize + dst_x] = v;
                    p += ch.pixel_type.bytes_per_sample();
                }
            }
        }
        if p != uncompressed.len() {
            return Err(ExrError::invalid(format!(
                "tile {tile_idx} consumed {p} of {} payload bytes",
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
        channels: sorted_channels.to_vec(),
        planes,
        attributes: header.attributes.clone(),
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

    #[test]
    fn subsampled_dim_basics() {
        assert_eq!(subsampled_dim(10, 1), 10);
        assert_eq!(subsampled_dim(10, 2), 5);
        assert_eq!(subsampled_dim(11, 2), 6);
        assert_eq!(subsampled_dim(0, 4), 0);
    }
}
