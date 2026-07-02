//! Top-level EXR decoder (scanline + tiled, single-part + multi-part).
//!
//! Walks the header (via [`crate::header::parse_header`]), reads the
//! offset table, then iterates the data chunks. For scanline files
//! each chunk is `Y(i32) | size(i32) | payload(size bytes)`. For tiled
//! files each chunk is `tx(i32) | ty(i32) | lvlx(i32) | lvly(i32) |
//! size(i32) | payload(size bytes)`.
//!
//! Multi-part files (version-field bit 12 set) contain multiple
//! sequential headers (each NUL-terminated), a double-NUL end marker,
//! then concatenated per-part offset tables, then chunks prefixed with
//! a 4-byte part number. Use [`parse_exr_multipart`] to parse them.
//!
//! Compression coverage: NONE, ZIP, ZIPS, RLE, PXR24, and B44 / B44A
//! (PXR24 + B44/B44A decode for single-part scanline images — see
//! [`decode_pxr24_payload`] and [`crate::b44`]). PIZ / DWAA / DWAB:
//! header-parsed and rejected on parse with a clear unsupported message.
//!
//! ZIP-family compression pre-applies two reversible transforms
//! documented in the OpenEXR file-format spec:
//!   1. interleave: low-byte half / high-byte half are concatenated so
//!      similar magnitudes in adjacent samples sit next to each other.
//!   2. predictor: each byte adds the previous one (mod 256), so most
//!      values become small deltas that compress well.
//!
//! Both transforms are byte-level and trivially invertible — see
//! [`apply_zip_unpredictor`] / [`apply_zip_uninterleave`] (the encoder
//! side does the inverse pair).
//!
//! Tiled multi-level files (MIPMAP_LEVELS / RIPMAP_LEVELS) are now
//! supported in read mode: the full-resolution level (lvlx=0, lvly=0)
//! is decoded into `ExrImage`; higher-resolution levels are skipped.
//! Level-dimension formulas (ROUND_DOWN / ROUND_UP) follow the
//! OpenEXR spec §2.2.

use crate::error::{ExrError, Result};
use crate::header::{parse_header, parse_multipart_headers};
use crate::image::{ExrImage, ExrPlane};
use crate::rle::rle_decompress;
use crate::tiled::tiledesc_from_attribute;
use crate::types::{Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType};

/// Inverse of the ZIP predictor pass per the OpenEXR spec.
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
pub(crate) struct RequiredAttrs {
    pub(crate) channels: Vec<Channel>,
    pub(crate) compression: Compression,
    pub(crate) data_window: Box2i,
    pub(crate) display_window: Box2i,
    pub(crate) line_order: LineOrder,
    pub(crate) pixel_aspect_ratio: f32,
    pub(crate) screen_window_center: (f32, f32),
    pub(crate) screen_window_width: f32,
}

pub(crate) fn extract_required(attrs: &[Attribute]) -> Result<RequiredAttrs> {
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
pub(crate) fn undo_zip_pipeline_pub(raw: Vec<u8>) -> Vec<u8> {
    undo_zip_pipeline(raw)
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

/// Describes, for one PXR24 scanline block, which channel rows are
/// present and how wide each row is. The PXR24 inflated stream is laid
/// out line-major then channel-major then byte-plane-major, so the
/// decoder must replay the exact same visiting order to reconstruct the
/// standard uncompressed little-endian sample stream.
pub(crate) struct Pxr24RowSpec<'a> {
    pub(crate) sorted_channels: &'a [Channel],
    pub(crate) width: u32,
    pub(crate) block_y0: u32,
    pub(crate) lines_in_block: usize,
}

/// Reduced on-the-wire byte width of a channel under PXR24: FLOAT is
/// carried as 3 bytes (24-bit reduction), HALF as 2, UINT as 4.
fn pxr24_channel_bytes(pt: PixelType) -> usize {
    match pt {
        PixelType::Float => 3,
        PixelType::Half => 2,
        PixelType::Uint => 4,
    }
}

/// Reverse the PXR24 24-bit float reduction: place the recovered 24-bit
/// code into the top 24 bits of a binary32 word (the dropped low byte is
/// implicitly zero). See observer-spec §1.1/§1.3.
fn pxr24_code_to_f32_bits(code24: u32) -> u32 {
    (code24 & 0x00ff_ffff) << 8
}

/// Decode a PXR24 scanline-block payload into the standard uncompressed
/// little-endian sample byte stream consumed by
/// [`scatter_block_into_planes`].
///
/// PXR24 (observer-spec §1): zlib-inflate the payload, then for each
/// image row walk the channels in sorted order; each channel emits one
/// byte plane per reduced byte (FLOAT=3, HALF=2, UINT=4) holding the
/// most-significant byte first, plane-major across the row. Each plane
/// byte is a horizontal delta against the previous sample of the same
/// channel on the same row (prediction resets to 0 per channel/row).
/// The decoder reassembles each sample's integer code by prefix-summing
/// the reassembled per-sample delta, then writes the sample back in the
/// channel's native pixel type (FLOAT reconstructed from the 24-bit
/// code, HALF/UINT verbatim) as little-endian bytes.
pub(crate) fn decode_pxr24_payload(
    payload: &[u8],
    spec: &Pxr24RowSpec,
    uncompressed_size: usize,
) -> Result<Vec<u8>> {
    // Reorganised (delta+plane) byte-stream size for this block: sum over
    // present rows/channels of (reduced bytes per sample) * (sub-width).
    let mut reorg_size = 0usize;
    for line in 0..spec.lines_in_block as u32 {
        let dst_y = spec.block_y0 + line;
        for ch in spec.sorted_channels {
            let ys = ch.y_sampling as u32;
            if dst_y % ys != 0 {
                continue;
            }
            let pw = subsampled_dim(spec.width, ch.x_sampling as u32) as usize;
            reorg_size += pxr24_channel_bytes(ch.pixel_type) * pw;
        }
    }

    // Raw-fallback: an encoder that couldn't shrink the chunk stores the
    // reorganised stream uncompressed (compressed length == reorg size).
    let reorg: Vec<u8> = if payload.len() == reorg_size {
        payload.to_vec()
    } else {
        let inflated = zlib_inflate(payload, reorg_size)?;
        if inflated.len() != reorg_size {
            return Err(ExrError::invalid(format!(
                "PXR24 inflate produced {} bytes, expected {reorg_size}",
                inflated.len()
            )));
        }
        inflated
    };

    let mut out = vec![0u8; uncompressed_size];
    let mut rp = 0usize; // read cursor into the reorganised stream
    let mut wp = 0usize; // write cursor into the native sample stream
    for line in 0..spec.lines_in_block as u32 {
        let dst_y = spec.block_y0 + line;
        for ch in spec.sorted_channels {
            let ys = ch.y_sampling as u32;
            if dst_y % ys != 0 {
                continue;
            }
            let pw = subsampled_dim(spec.width, ch.x_sampling as u32) as usize;
            let nbytes = pxr24_channel_bytes(ch.pixel_type);
            // Read this channel/row's planes (most-significant first),
            // prefix-summing each sample's reassembled delta into a
            // running 32-bit code.
            let planes = &reorg[rp..rp + nbytes * pw];
            rp += nbytes * pw;
            let mut acc: u32 = 0;
            for x in 0..pw {
                let mut diff: u32 = 0;
                for b in 0..nbytes {
                    diff = (diff << 8) | u32::from(planes[b * pw + x]);
                }
                acc = acc.wrapping_add(diff);
                match ch.pixel_type {
                    PixelType::Float => {
                        let bits = pxr24_code_to_f32_bits(acc);
                        out[wp..wp + 4].copy_from_slice(&bits.to_le_bytes());
                        wp += 4;
                    }
                    PixelType::Half => {
                        let bits = (acc & 0xffff) as u16;
                        out[wp..wp + 2].copy_from_slice(&bits.to_le_bytes());
                        wp += 2;
                    }
                    PixelType::Uint => {
                        out[wp..wp + 4].copy_from_slice(&acc.to_le_bytes());
                        wp += 4;
                    }
                }
            }
        }
    }
    if wp != uncompressed_size {
        return Err(ExrError::invalid(format!(
            "PXR24 produced {wp} of {uncompressed_size} native bytes"
        )));
    }
    Ok(out)
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
                        // 24 bits, so this matches the reference
                        // encoder's "as float" behaviour.
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

/// Decode one B44 / B44A scanline-chunk payload and scatter it directly
/// into the per-channel image planes (observer-spec §2).
///
/// Unlike the ZIP / RLE / PXR24 path, B44 regroups the chunk into
/// per-channel contiguous planes rather than the interleaved scanline
/// stream, so it bypasses [`scatter_block_into_planes`] and writes each
/// recovered sample straight into the target plane. `uncompressed_size`
/// is the interleaved-stream size used to detect the shared raw fallback
/// (compressed length == uncompressed length ⇒ payload is the raw
/// interleaved bytes and no B44 transform applies).
#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_b44_block_into_planes(
    payload: &[u8],
    sorted_channels: &[Channel],
    planes: &mut [ExrPlane],
    width: u32,
    block_y0: u32,
    lines_in_block: usize,
    uncompressed_size: usize,
) -> Result<()> {
    // Shared raw fallback: an encoder that couldn't shrink the chunk
    // stores the interleaved uncompressed stream verbatim.
    if payload.len() == uncompressed_size {
        return scatter_block_into_planes(
            payload,
            sorted_channels,
            planes,
            width,
            0,
            block_y0,
            lines_in_block,
        );
    }

    // Per-channel sub-sampled extents within this chunk: width is the
    // channel's sub-sampled image width; height is the count of image rows
    // in the chunk that survive the channel's vertical subsampling.
    let extents: Vec<crate::b44::B44ChannelExtent> = sorted_channels
        .iter()
        .map(|ch| {
            let ys = ch.y_sampling as u32;
            let ph = (0..lines_in_block as u32)
                .filter(|&l| (block_y0 + l) % ys == 0)
                .count();
            let pw = subsampled_dim(width, ch.x_sampling as u32) as usize;
            crate::b44::B44ChannelExtent { pw, ph }
        })
        .collect();

    let decoded = crate::b44::decode_b44_chunk(payload, sorted_channels, &extents)?;

    for (ch_idx, ch) in sorted_channels.iter().enumerate() {
        let ys = ch.y_sampling as u32;
        let xs = ch.x_sampling as u32;
        let pw = subsampled_dim(width, xs) as usize;
        let plane = &mut planes[ch_idx].samples;
        // Map the chunk-local row index (0..ph) back to the channel's
        // sub-sampled image row. The chunk's present rows are exactly the
        // image rows in [block_y0, block_y0+lines_in_block) divisible by
        // y_sampling; their sub-sampled indices are contiguous.
        let mut chunk_row = 0usize;
        for l in 0..lines_in_block as u32 {
            let dst_y = block_y0 + l;
            if dst_y % ys != 0 {
                continue;
            }
            let dst_y_sub = (dst_y / ys) as usize;
            match &decoded[ch_idx] {
                crate::b44::B44Plane::Half(codes) => {
                    let row = &codes[chunk_row * pw..chunk_row * pw + pw];
                    for (x, &code) in row.iter().enumerate() {
                        plane[dst_y_sub * pw + x] = crate::half::half_to_f32(code);
                    }
                }
                crate::b44::B44Plane::Raw(bytes) => {
                    let bps = ch.pixel_type.bytes_per_sample();
                    let base = chunk_row * pw * bps;
                    for x in 0..pw {
                        let off = base + x * bps;
                        let v = match ch.pixel_type {
                            PixelType::Float => {
                                f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
                            }
                            PixelType::Uint => {
                                u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as f32
                            }
                            PixelType::Half => unreachable!("HALF handled as B44Plane::Half"),
                        };
                        plane[dst_y_sub * pw + x] = v;
                    }
                }
            }
            chunk_row += 1;
        }
    }
    Ok(())
}

/// Parse a single-part EXR file (scanline OR tiled) from a byte slice.
///
/// For multi-part files use [`parse_exr_multipart`] instead; this
/// function returns an error if the multi-part bit is set.
pub fn parse_exr(bytes: &[u8]) -> Result<ExrImage> {
    let header = parse_header(bytes)?;
    // parse_header already rejects multipart; also reject here explicitly
    // so callers get a clear message pointing at parse_exr_multipart.
    if header.version.multipart {
        return Err(ExrError::unsupported(
            "multi-part EXR: use parse_exr_multipart()".to_string(),
        ));
    }
    let req = extract_required(&header.attributes)?;

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
        // `block_off` is an untrusted u64 from the offset table cast to
        // usize; a hostile value near usize::MAX would overflow the
        // `block_off + 8` bounds check itself (debug panic / release wrap
        // into an out-of-bounds slice). Add with overflow detection.
        let block_hdr_end = block_off
            .checked_add(8)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| {
                ExrError::invalid(format!("block {block_idx} offset {block_off} past EOF"))
            })?;
        let y_coord = i32::from_le_bytes(bytes[block_off..block_off + 4].try_into().unwrap());
        let payload_size =
            i32::from_le_bytes(bytes[block_off + 4..block_hdr_end].try_into().unwrap());
        if payload_size < 0 {
            return Err(ExrError::invalid(format!(
                "block {block_idx} negative size {payload_size}"
            )));
        }
        let payload_start = block_hdr_end;
        let payload_end = payload_start
            .checked_add(payload_size as usize)
            .ok_or_else(|| {
                ExrError::invalid(format!(
                    "block {block_idx} payload size {payload_size} overflows address space"
                ))
            })?;
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

        // B44 / B44A do not produce the standard interleaved scanline
        // stream; they decode into per-channel planes and scatter
        // directly. The shared raw fallback (compressed == uncompressed)
        // is handled inside the B44 branch.
        if matches!(req.compression, Compression::B44 | Compression::B44a) {
            scatter_b44_block_into_planes(
                payload,
                &sorted_channels,
                &mut planes,
                width,
                block_y0,
                lines_in_block,
                uncompressed_size,
            )?;
            continue;
        }

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
            Compression::Pxr24 => decode_pxr24_payload(
                payload,
                &Pxr24RowSpec {
                    sorted_channels: &sorted_channels,
                    width,
                    block_y0,
                    lines_in_block,
                },
                uncompressed_size,
            )?,
            other => {
                return Err(ExrError::unsupported(format!(
                    "scanline compression {other:?} not yet implemented"
                )))
            }
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

/// Compute the number of mipmap levels for one dimension with the given
/// rounding mode. `round_up=false` ≡ ROUND_DOWN (spec default), `true`
/// ≡ ROUND_UP.
///
/// Formula: keep halving (floor or ceil) until the dimension reaches 1.
pub fn mipmap_level_count(mut dim: u32, round_up: bool) -> u32 {
    let mut n = 1u32;
    while dim > 1 {
        dim = if round_up { dim.div_ceil(2) } else { dim / 2 };
        n += 1;
    }
    n
}

/// Return the width/height of mipmap level `level` (0 = full res).
pub fn mipmap_level_dim(full_dim: u32, level: u32, round_up: bool) -> u32 {
    let mut d = full_dim;
    for _ in 0..level {
        if d <= 1 {
            return 1;
        }
        d = if round_up { d.div_ceil(2) } else { d / 2 };
    }
    d.max(1)
}

/// Decode a one-level tile payload into the per-channel f32 planes.
/// `x0,y0` are the top-left pixel coordinates of the tile within the
/// full-resolution image (level 0,0). `tw,th` are the valid pixel
/// dimensions of this tile (may be smaller than tdesc tile size at edges).
#[allow(clippy::too_many_arguments)]
pub(crate) fn scatter_tile_into_planes(
    payload: &[u8],
    sorted_channels: &[Channel],
    planes: &mut [ExrPlane],
    width: u32,
    x0: u32,
    y0: u32,
    tw: usize,
    th: usize,
    compression: Compression,
    tile_idx: usize,
) -> Result<()> {
    let bpp: usize = sorted_channels
        .iter()
        .map(|c| c.pixel_type.bytes_per_sample())
        .sum();
    let uncompressed_size = tw * th * bpp;

    // B44 / B44A regroup the tile into per-channel planes rather than the
    // interleaved native stream, so they scatter directly into the tile
    // rectangle (observer-spec §2). Tiled files are constrained to 1×1
    // sampling (enforced by the tiled parsers), so a tile is a self-contained
    // `tw × th` block with no vertical-subsampling row gaps.
    if matches!(compression, Compression::B44 | Compression::B44a) {
        return scatter_b44_tile_into_planes(
            payload,
            sorted_channels,
            planes,
            width,
            x0,
            y0,
            tw,
            th,
            uncompressed_size,
        );
    }

    let uncompressed: Vec<u8> = match compression {
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
        Compression::Pxr24 => decode_pxr24_payload(
            payload,
            // A tile is a single self-contained block: full width = `tw`,
            // block origin row 0, `th` rows, all present (1×1 sampling).
            &Pxr24RowSpec {
                sorted_channels,
                width: tw as u32,
                block_y0: 0,
                lines_in_block: th,
            },
            uncompressed_size,
        )?,
        _ => {
            return Err(ExrError::unsupported(format!(
                "compression {compression:?} not yet implemented for tiled files (tile {tile_idx})"
            )))
        }
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
    Ok(())
}

/// Decode one B44 / B44A tile payload and scatter it into the per-channel
/// image planes (observer-spec §2). The tile is a self-contained
/// `tw × th` block at pixel offset `(x0, y0)`; tiled files use 1×1
/// sampling so there is no vertical-subsampling row gap.
///
/// `uncompressed_size` is the interleaved-native-stream size of the tile,
/// used to detect the shared raw fallback (compressed length ==
/// uncompressed length ⇒ payload is the raw interleaved bytes and no B44
/// transform applies).
#[allow(clippy::too_many_arguments)]
fn scatter_b44_tile_into_planes(
    payload: &[u8],
    sorted_channels: &[Channel],
    planes: &mut [ExrPlane],
    width: u32,
    x0: u32,
    y0: u32,
    tw: usize,
    th: usize,
    uncompressed_size: usize,
) -> Result<()> {
    // Shared raw fallback: an encoder that couldn't shrink the tile stores
    // the interleaved uncompressed stream verbatim (channel-interleaved,
    // row-major over the tile rectangle).
    if payload.len() == uncompressed_size {
        let mut p = 0usize;
        for line in 0..th {
            let dst_y = y0 as usize + line;
            for (ch_idx, ch) in sorted_channels.iter().enumerate() {
                let plane = &mut planes[ch_idx].samples;
                for x in 0..tw {
                    let dst_x = x0 as usize + x;
                    let v = match ch.pixel_type {
                        PixelType::Half => {
                            let bits = u16::from_le_bytes(payload[p..p + 2].try_into().unwrap());
                            crate::half::half_to_f32(bits)
                        }
                        PixelType::Float => {
                            f32::from_le_bytes(payload[p..p + 4].try_into().unwrap())
                        }
                        PixelType::Uint => {
                            let bits = u32::from_le_bytes(payload[p..p + 4].try_into().unwrap());
                            bits as f32
                        }
                    };
                    plane[dst_y * width as usize + dst_x] = v;
                    p += ch.pixel_type.bytes_per_sample();
                }
            }
        }
        return Ok(());
    }

    // Each channel's tile-local plane is the full tile rectangle (1×1
    // sampling).
    let extents: Vec<crate::b44::B44ChannelExtent> = sorted_channels
        .iter()
        .map(|_| crate::b44::B44ChannelExtent { pw: tw, ph: th })
        .collect();

    let decoded = crate::b44::decode_b44_chunk(payload, sorted_channels, &extents)?;

    for (ch_idx, ch) in sorted_channels.iter().enumerate() {
        let plane = &mut planes[ch_idx].samples;
        match &decoded[ch_idx] {
            crate::b44::B44Plane::Half(codes) => {
                for ty in 0..th {
                    let dst_y = y0 as usize + ty;
                    let row = &codes[ty * tw..ty * tw + tw];
                    for (tx, &code) in row.iter().enumerate() {
                        let dst_x = x0 as usize + tx;
                        plane[dst_y * width as usize + dst_x] = crate::half::half_to_f32(code);
                    }
                }
            }
            crate::b44::B44Plane::Raw(bytes) => {
                let bps = ch.pixel_type.bytes_per_sample();
                for ty in 0..th {
                    let dst_y = y0 as usize + ty;
                    for tx in 0..tw {
                        let off = (ty * tw + tx) * bps;
                        let v = match ch.pixel_type {
                            PixelType::Float => {
                                f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap())
                            }
                            PixelType::Uint => {
                                u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as f32
                            }
                            PixelType::Half => unreachable!("HALF handled as B44Plane::Half"),
                        };
                        let dst_x = x0 as usize + tx;
                        plane[dst_y * width as usize + dst_x] = v;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Decode a tiled single-part EXR file.
///
/// Supports ONE_LEVEL, MIPMAP_LEVELS, and RIPMAP_LEVELS. For multi-level
/// files, only the full-resolution level (lvlx=0, lvly=0) is decoded into
/// the returned `ExrImage`; higher-resolution reduction levels are skipped
/// after being read (so the offset table is consumed correctly).
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
    if tdesc.level_mode > 2 {
        return Err(ExrError::invalid(format!(
            "tiledesc level_mode {} unknown (expected 0=ONE_LEVEL, 1=MIPMAP, 2=RIPMAP)",
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
    let round_up = tdesc.round_mode != 0;

    // Compute the total number of tiles across all levels.
    // ONE_LEVEL (0): single level with ceil(w/tw) * ceil(h/th) tiles.
    // MIPMAP (1): levels 0..N-1 where N = mipmap_level_count(max(w,h)).
    //             Within each level l: tiles_x = ceil(lw/tw), tiles_y = ceil(lh/th).
    //             lw[l] = mipmap_level_dim(w, l, round_up), same for h.
    // RIPMAP (2): all combinations of x-level (0..Nx-1) and y-level (0..Ny-1).
    //             Tile (lvlx, lvly) has lw=level_dim(w,lvlx), lh=level_dim(h,lvly).
    //             Table order: lvlx inner loop, lvly outer loop.
    let total_tiles = compute_total_tiles(
        tdesc.level_mode,
        width,
        height,
        tdesc.x_size,
        tdesc.y_size,
        round_up,
    );

    let mut pos = header.end_offset;
    if pos + total_tiles * 8 > bytes.len() {
        return Err(ExrError::invalid(format!(
            "tile offset table runs past EOF (need {} bytes at {pos}, file size {})",
            total_tiles * 8,
            bytes.len()
        )));
    }
    // Read all offsets into a flat Vec. We'll look up by sequential index.
    let mut all_offsets: Vec<usize> = Vec::with_capacity(total_tiles);
    for _ in 0..total_tiles {
        all_offsets.push(u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize);
        pos += 8;
    }

    let mut planes: Vec<ExrPlane> = sorted_channels
        .iter()
        .map(|c| ExrPlane {
            name: c.name.clone(),
            samples: vec![0.0; (width * height) as usize],
        })
        .collect();

    // Iterate all tiles. For multi-level files we only scatter tiles with
    // lvlx=0 and lvly=0 (the full-resolution level) into the planes.
    for (tile_idx, &tile_off) in all_offsets.iter().enumerate() {
        // `tile_off` is an untrusted u64 offset-table entry cast to usize;
        // guard the 20-byte tile-header bounds check against usize overflow.
        let tile_hdr_end = tile_off
            .checked_add(20)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| {
                ExrError::invalid(format!("tile {tile_idx} offset {tile_off} past EOF"))
            })?;
        let h_tx = i32::from_le_bytes(bytes[tile_off..tile_off + 4].try_into().unwrap());
        let h_ty = i32::from_le_bytes(bytes[tile_off + 4..tile_off + 8].try_into().unwrap());
        let lvl_x = i32::from_le_bytes(bytes[tile_off + 8..tile_off + 12].try_into().unwrap());
        let lvl_y = i32::from_le_bytes(bytes[tile_off + 12..tile_off + 16].try_into().unwrap());
        let payload_size =
            i32::from_le_bytes(bytes[tile_off + 16..tile_hdr_end].try_into().unwrap());
        if payload_size < 0 {
            return Err(ExrError::invalid(format!(
                "tile {tile_idx} negative payload size {payload_size}"
            )));
        }
        let pl_start = tile_hdr_end;
        let pl_end = pl_start.checked_add(payload_size as usize).ok_or_else(|| {
            ExrError::invalid(format!(
                "tile {tile_idx} payload size {payload_size} overflows address space"
            ))
        })?;
        if pl_end > bytes.len() {
            return Err(ExrError::invalid(format!(
                "tile {tile_idx} payload runs past EOF"
            )));
        }

        // Only decode the full-resolution level.
        if lvl_x != 0 || lvl_y != 0 {
            continue;
        }

        let tx = h_tx as u32;
        let ty = h_ty as u32;
        let payload = &bytes[pl_start..pl_end];

        // `h_tx`/`h_ty` come straight off the wire; hostile values must
        // not underflow the edge-tile clip below.
        let txc = width.div_ceil(tdesc.x_size);
        let tyc = height.div_ceil(tdesc.y_size);
        if h_tx < 0 || h_ty < 0 || tx >= txc || ty >= tyc {
            return Err(ExrError::invalid(format!(
                "tile {tile_idx} at ({h_tx},{h_ty}) outside tile grid {txc}×{tyc}"
            )));
        }

        let x0 = tx * tdesc.x_size;
        let y0 = ty * tdesc.y_size;
        let x1 = (x0 + tdesc.x_size).min(width);
        let y1 = (y0 + tdesc.y_size).min(height);
        let tw = (x1 - x0) as usize;
        let th = (y1 - y0) as usize;

        scatter_tile_into_planes(
            payload,
            sorted_channels,
            &mut planes,
            width,
            x0,
            y0,
            tw,
            th,
            req.compression,
            tile_idx,
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
        channels: sorted_channels.to_vec(),
        planes,
        attributes: header.attributes.clone(),
    })
}

/// Total number of tile chunks in the offset table for a tiled file.
///
/// * ONE_LEVEL (mode=0): `ceil(w/tw) * ceil(h/th)`.
/// * MIPMAP (mode=1): sum over levels 0..N-1 of
///   `ceil(lw/tw) * ceil(lh/th)`.
/// * RIPMAP (mode=2): sum over (lvlx, lvly) pairs (all combinations of
///   x-levels and y-levels) of `ceil(lw/tw) * ceil(lh/th)`.
fn compute_total_tiles(
    level_mode: u8,
    width: u32,
    height: u32,
    tile_w: u32,
    tile_h: u32,
    round_up: bool,
) -> usize {
    match level_mode {
        0 => {
            // ONE_LEVEL
            let tx = width.div_ceil(tile_w) as usize;
            let ty = height.div_ceil(tile_h) as usize;
            tx * ty
        }
        1 => {
            // MIPMAP: levels governed by the larger dimension.
            let n = mipmap_level_count(width.max(height), round_up);
            let mut total = 0usize;
            for l in 0..n {
                let lw = mipmap_level_dim(width, l, round_up);
                let lh = mipmap_level_dim(height, l, round_up);
                total += lw.div_ceil(tile_w) as usize * lh.div_ceil(tile_h) as usize;
            }
            total
        }
        2 => {
            // RIPMAP: independent x-levels and y-levels.
            let nx = mipmap_level_count(width, round_up);
            let ny = mipmap_level_count(height, round_up);
            let mut total = 0usize;
            for ly in 0..ny {
                let lh = mipmap_level_dim(height, ly, round_up);
                for lx in 0..nx {
                    let lw = mipmap_level_dim(width, lx, round_up);
                    total += lw.div_ceil(tile_w) as usize * lh.div_ceil(tile_h) as usize;
                }
            }
            total
        }
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Multi-level tiled READ (full MIPMAP / RIPMAP pyramid)
// ---------------------------------------------------------------------------

/// One decoded pyramid level of a tiled EXR file. Carries the per-level
/// width / height (which may be smaller than the file's full-resolution
/// dataWindow for any level beyond `(0, 0)`) plus one `f32` plane per
/// channel sized to that level's dimensions.
///
/// For MIPMAP files `level_x == level_y` for every entry; for RIPMAP
/// files the two axes are independent (see [`MultilevelTiledImage`] for
/// the per-axis level count).
///
/// Sample layout matches [`ExrImage::planes`]: row-major, alphabetical
/// channel order, length `width * height` per channel.
#[derive(Debug, Clone)]
pub struct TiledLevel {
    /// Mipmap x-level index (0 = full resolution).
    pub level_x: u32,
    /// Mipmap y-level index (0 = full resolution). Equal to `level_x`
    /// for MIPMAP_LEVELS files.
    pub level_y: u32,
    /// Width of this level (per the spec's level-dim formula).
    pub width: u32,
    /// Height of this level.
    pub height: u32,
    /// Per-channel f32 sample plane (`width * height` long), in the
    /// same alphabetical channel order as `channels`.
    pub planes: Vec<ExrPlane>,
}

/// Result of [`parse_exr_tiled_multilevel`]. Carries header-level
/// metadata once plus every decoded pyramid level.
#[derive(Debug, Clone)]
pub struct MultilevelTiledImage {
    /// `0 = ONE_LEVEL`, `1 = MIPMAP_LEVELS`, `2 = RIPMAP_LEVELS`.
    pub level_mode: u8,
    /// `0 = ROUND_DOWN`, `1 = ROUND_UP`.
    pub round_mode: u8,
    /// Tile width in pixels (from the `tiles` attribute).
    pub tile_x: u32,
    /// Tile height in pixels.
    pub tile_y: u32,
    /// Full-resolution data window (level 0,0).
    pub data_window: Box2i,
    /// Full-resolution display window.
    pub display_window: Box2i,
    /// Channel list (alphabetical-sorted, matching each level's plane order).
    pub channels: Vec<Channel>,
    /// Compression mode declared by the file (already applied to each
    /// `levels[*].planes` payload — callers see decoded f32 samples).
    pub compression: Compression,
    /// Decoded pyramid levels in the spec's iteration order:
    /// * ONE_LEVEL: a single entry at `(0, 0)`.
    /// * MIPMAP_LEVELS: levels `0..N-1`, `level_x == level_y == n`.
    /// * RIPMAP_LEVELS: `level_y` outer, `level_x` inner — i.e.
    ///   `(0,0), (1,0), ..., (Nx-1, 0), (0, 1), (1, 1), ...`.
    pub levels: Vec<TiledLevel>,
}

/// Parse a tiled single-part EXR file and return every decoded mipmap /
/// ripmap level. Companion to [`parse_exr`], which is unchanged and
/// continues to return only the full-resolution level.
///
/// Supports ONE_LEVEL, MIPMAP_LEVELS, and RIPMAP_LEVELS files at
/// compression NONE / ZIP / ZIPS / RLE. Channels must be at 1×1
/// sampling (spec requirement for tiled files).
///
/// Round-trips bit-exactly against [`crate::encode_exr_tiled_mipmap`]
/// and [`crate::encode_exr_tiled_ripmap`]: the per-level pyramid
/// supplied to the encoder is recovered sample-for-sample by this
/// function.
pub fn parse_exr_tiled_multilevel(bytes: &[u8]) -> Result<MultilevelTiledImage> {
    let header = parse_header(bytes)?;
    if header.version.multipart {
        return Err(ExrError::unsupported(
            "multi-part EXR: parse_exr_tiled_multilevel is single-part only".to_string(),
        ));
    }
    if !header.version.single_tile {
        return Err(ExrError::invalid(
            "parse_exr_tiled_multilevel: file is not tiled (single_tile bit not set)".to_string(),
        ));
    }
    let req = extract_required(&header.attributes)?;

    let mut sorted_channels = req.channels.clone();
    sorted_channels.sort_by(|a, b| a.name.cmp(&b.name));
    for ch in &sorted_channels {
        if ch.x_sampling != 1 || ch.y_sampling != 1 {
            return Err(ExrError::unsupported(format!(
                "tiled multilevel: channel '{}' sampling {}×{} (spec requires 1×1 for tiled files)",
                ch.name, ch.x_sampling, ch.y_sampling
            )));
        }
    }
    if !matches!(
        req.compression,
        Compression::None
            | Compression::Zip
            | Compression::Zips
            | Compression::Rle
            | Compression::Pxr24
            | Compression::B44
            | Compression::B44a
    ) {
        return Err(ExrError::unsupported(format!(
            "tiled multilevel: compression {:?} not yet implemented",
            req.compression
        )));
    }

    let tdesc_attr = header
        .attributes
        .iter()
        .find(|a| a.name == "tiles")
        .ok_or_else(|| {
            ExrError::invalid("tiled file missing required `tiles` attribute".to_string())
        })?;
    let tdesc = tiledesc_from_attribute(&tdesc_attr.value)?;
    if tdesc.level_mode > 2 {
        return Err(ExrError::invalid(format!(
            "tiledesc level_mode {} unknown (expected 0/1/2)",
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
    if width == 0 || height == 0 {
        return Err(ExrError::invalid(format!(
            "tiled multilevel: dataWindow {width}×{height} must be > 0"
        )));
    }
    let round_up = tdesc.round_mode != 0;

    // Enumerate the levels we expect (and their per-level dims) in the
    // spec's iteration order. This drives both the per-level plane
    // allocation and the dispatch table for incoming tile chunks.
    let mut levels: Vec<TiledLevel> = match tdesc.level_mode {
        0 => vec![TiledLevel {
            level_x: 0,
            level_y: 0,
            width,
            height,
            planes: alloc_planes(&sorted_channels, width, height),
        }],
        1 => {
            let n = mipmap_level_count(width.max(height), round_up);
            (0..n)
                .map(|l| {
                    let lw = mipmap_level_dim(width, l, round_up);
                    let lh = mipmap_level_dim(height, l, round_up);
                    TiledLevel {
                        level_x: l,
                        level_y: l,
                        width: lw,
                        height: lh,
                        planes: alloc_planes(&sorted_channels, lw, lh),
                    }
                })
                .collect()
        }
        2 => {
            let nx = mipmap_level_count(width, round_up);
            let ny = mipmap_level_count(height, round_up);
            let mut v = Vec::with_capacity((nx * ny) as usize);
            for ly in 0..ny {
                let lh = mipmap_level_dim(height, ly, round_up);
                for lx in 0..nx {
                    let lw = mipmap_level_dim(width, lx, round_up);
                    v.push(TiledLevel {
                        level_x: lx,
                        level_y: ly,
                        width: lw,
                        height: lh,
                        planes: alloc_planes(&sorted_channels, lw, lh),
                    });
                }
            }
            v
        }
        _ => unreachable!("checked above"),
    };

    // Walk the tile offset table (one u64 per chunk; total_tiles == sum
    // over levels of tiles_x*tiles_y).
    let total_tiles = compute_total_tiles(
        tdesc.level_mode,
        width,
        height,
        tdesc.x_size,
        tdesc.y_size,
        round_up,
    );
    let mut pos = header.end_offset;
    if pos + total_tiles * 8 > bytes.len() {
        return Err(ExrError::invalid(format!(
            "tile offset table runs past EOF (need {} bytes at {pos}, file size {})",
            total_tiles * 8,
            bytes.len()
        )));
    }
    let mut offsets: Vec<usize> = Vec::with_capacity(total_tiles);
    for _ in 0..total_tiles {
        offsets.push(u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize);
        pos += 8;
    }

    // For each tile chunk: read (tx, ty, lvlx, lvly, size, payload),
    // locate the matching level slot by (lvlx, lvly), then scatter into
    // that level's planes using the level's dims as the row stride.
    for (tile_idx, &tile_off) in offsets.iter().enumerate() {
        // `tile_off` is an untrusted u64 offset-table entry cast to usize;
        // guard the 20-byte tile-header bounds check against usize overflow.
        let tile_hdr_end = tile_off
            .checked_add(20)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| {
                ExrError::invalid(format!("tile {tile_idx} offset {tile_off} past EOF"))
            })?;
        let tx = i32::from_le_bytes(bytes[tile_off..tile_off + 4].try_into().unwrap());
        let ty = i32::from_le_bytes(bytes[tile_off + 4..tile_off + 8].try_into().unwrap());
        let lvl_x = i32::from_le_bytes(bytes[tile_off + 8..tile_off + 12].try_into().unwrap());
        let lvl_y = i32::from_le_bytes(bytes[tile_off + 12..tile_off + 16].try_into().unwrap());
        let payload_size =
            i32::from_le_bytes(bytes[tile_off + 16..tile_hdr_end].try_into().unwrap());
        if payload_size < 0 || tx < 0 || ty < 0 || lvl_x < 0 || lvl_y < 0 {
            return Err(ExrError::invalid(format!(
                "tile {tile_idx} bad header: tx={tx} ty={ty} lvlx={lvl_x} lvly={lvl_y} size={payload_size}"
            )));
        }
        let pl_start = tile_hdr_end;
        let pl_end = pl_start.checked_add(payload_size as usize).ok_or_else(|| {
            ExrError::invalid(format!(
                "tile {tile_idx} payload size {payload_size} overflows address space"
            ))
        })?;
        if pl_end > bytes.len() {
            return Err(ExrError::invalid(format!(
                "tile {tile_idx} payload runs past EOF"
            )));
        }
        let payload = &bytes[pl_start..pl_end];

        // Find the level slot by (lvlx, lvly). For ONE_LEVEL there's
        // only one entry and lvl_x/lvl_y must both be 0; for MIPMAP
        // lvl_x == lvl_y; for RIPMAP they're independent.
        let level = levels
            .iter_mut()
            .find(|l| l.level_x as i32 == lvl_x && l.level_y as i32 == lvl_y)
            .ok_or_else(|| {
                ExrError::invalid(format!(
                    "tile {tile_idx} carries unknown level ({lvl_x}, {lvl_y})"
                ))
            })?;

        // `tx`/`ty` come straight off the wire; reject negatives before
        // the `as u32` multiply (a hostile index must not overflow it),
        // then bound against this level's dimensions.
        let x0 = u32::try_from(tx)
            .ok()
            .and_then(|v| v.checked_mul(tdesc.x_size));
        let y0 = u32::try_from(ty)
            .ok()
            .and_then(|v| v.checked_mul(tdesc.y_size));
        let (Some(x0), Some(y0)) = (x0, y0) else {
            return Err(ExrError::invalid(format!(
                "tile {tile_idx} carries hostile tile index ({tx},{ty})"
            )));
        };
        if x0 >= level.width || y0 >= level.height {
            return Err(ExrError::invalid(format!(
                "tile {tile_idx} at ({tx},{ty}) outside level ({lvl_x},{lvl_y}) dims {}×{}",
                level.width, level.height
            )));
        }
        let x1 = (x0 + tdesc.x_size).min(level.width);
        let y1 = (y0 + tdesc.y_size).min(level.height);
        let tw = (x1 - x0) as usize;
        let th = (y1 - y0) as usize;

        scatter_tile_into_planes(
            payload,
            &sorted_channels,
            &mut level.planes,
            level.width,
            x0,
            y0,
            tw,
            th,
            req.compression,
            tile_idx,
        )?;
    }

    Ok(MultilevelTiledImage {
        level_mode: tdesc.level_mode,
        round_mode: tdesc.round_mode,
        tile_x: tdesc.x_size,
        tile_y: tdesc.y_size,
        data_window: req.data_window,
        display_window: req.display_window,
        channels: sorted_channels,
        compression: req.compression,
        levels,
    })
}

/// Allocate one zero-initialised `f32` plane per channel at the given
/// dimensions. Used by [`parse_exr_tiled_multilevel`] to size per-level
/// pixel storage.
fn alloc_planes(channels: &[Channel], width: u32, height: u32) -> Vec<ExrPlane> {
    channels
        .iter()
        .map(|c| ExrPlane {
            name: c.name.clone(),
            samples: vec![0.0; (width as usize) * (height as usize)],
        })
        .collect()
}

/// Parse a multi-part EXR file and return one `ExrImage` per part.
///
/// Multi-part files are identified by version-field bit 12 being set.
/// The binary layout is:
///
/// ```text
/// magic(4) | version(4)
/// | header_0 ... NUL | header_1 ... NUL | NUL   (extra NUL = end)
/// | offset_table_0(chunkCount×8) | offset_table_1(chunkCount×8) | ...
/// | chunks (each: i32 part_number | i32 Y | i32 size | payload[size])
/// ```
///
/// The `chunkCount` attribute is mandatory in multi-part files.
///
/// **Offset table robustness**: some EXR encoders (including the reference
/// `exrmultipart -combine`) emit zero-filled offset table entries for
/// parts beyond the first. We therefore decode chunks by sequential scan
/// rather than index-lookup, which handles both fully-populated and
/// zero-padded tables correctly.
///
/// Only `scanlineimage` part type is supported in this round.
pub fn parse_exr_multipart(bytes: &[u8]) -> Result<Vec<ExrImage>> {
    let parts = parse_multipart_headers(bytes)?;
    if parts.is_empty() {
        return Err(ExrError::invalid(
            "multi-part file has no parts".to_string(),
        ));
    }
    // Reject deep parts here — multi-part deep files (non_image bit set)
    // have one or more `type=deepscanline` parts which must be parsed
    // through parse_exr_deep_multipart instead. We don't allow a hybrid
    // walk because a deep chunk's record layout is different
    // (`i32 part | i32 Y | 3*u64 sizes | table | data` vs
    // `i32 part | i32 Y | i32 size | payload`).
    for (i, part) in parts.iter().enumerate() {
        if let Some(t) = find_part_type(&part.attributes) {
            if t == "deepscanline" || t == "deeptile" {
                return Err(ExrError::unsupported(format!(
                    "multi-part part {i} has type='{t}' — call \
                     parse_exr_deep_multipart() for deep multi-part files"
                )));
            }
            if t == "tiledimage" {
                return Err(ExrError::unsupported(format!(
                    "multi-part part {i} has type='tiledimage' — call \
                     parse_exr_multipart_tiled() for multi-part flat tiled files"
                )));
            }
        }
    }

    // Collect chunkCount per part (mandatory in multi-part files).
    let mut chunk_counts: Vec<usize> = Vec::with_capacity(parts.len());
    for (i, part) in parts.iter().enumerate() {
        let cc = find_chunk_count(&part.attributes).ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part part {i} missing required chunkCount attribute"
            ))
        })?;
        chunk_counts.push(cc);
    }

    // Skip past the offset tables (may be zero-filled; we scan linearly).
    // All parts' tables are stored consecutively; each table has
    // `chunkCount` u64 entries.
    let total_chunks: usize = chunk_counts.iter().sum();
    let tables_start = parts.last().unwrap().end_offset;
    let chunk_scan_start = tables_start + total_chunks * 8;
    if chunk_scan_start > bytes.len() {
        return Err(ExrError::invalid(format!(
            "multi-part offset tables run past EOF (need {}, have {})",
            chunk_scan_start,
            bytes.len()
        )));
    }

    // Prepare per-part state.
    let mut part_reqs: Vec<RequiredAttrs> = Vec::with_capacity(parts.len());
    let mut part_sorted_channels: Vec<Vec<Channel>> = Vec::with_capacity(parts.len());
    let mut planes_list: Vec<Vec<ExrPlane>> = Vec::with_capacity(parts.len());

    for (part_idx, part) in parts.iter().enumerate() {
        let req = extract_required(&part.attributes)?;
        if !matches!(
            req.compression,
            Compression::None
                | Compression::Zip
                | Compression::Zips
                | Compression::Rle
                | Compression::Pxr24
                | Compression::B44
                | Compression::B44a
        ) {
            return Err(ExrError::unsupported(format!(
                "multi-part part {part_idx}: compression {:?} not yet implemented",
                req.compression
            )));
        }
        let width = req.data_window.width();
        let height = req.data_window.height();
        if width == 0 || height == 0 {
            return Err(ExrError::invalid(format!(
                "multi-part part {part_idx}: dataWindow {width}×{height} must be > 0"
            )));
        }

        let mut sorted_channels = req.channels.clone();
        sorted_channels.sort_by(|a, b| a.name.cmp(&b.name));

        // A sampling factor of zero off the wire would divide by zero
        // in the plane-shape and chunk-size math below.
        for ch in &sorted_channels {
            if ch.x_sampling <= 0 || ch.y_sampling <= 0 {
                return Err(ExrError::invalid(format!(
                    "multi-part part {part_idx}: channel '{}' has non-positive sampling \
                     factor x={} y={}",
                    ch.name, ch.x_sampling, ch.y_sampling
                )));
            }
        }

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

        part_reqs.push(req);
        part_sorted_channels.push(sorted_channels);
        planes_list.push(planes);
    }

    // Sequential chunk scan: each chunk starts with i32 part_number,
    // followed by i32 Y, i32 size, payload.
    let mut scan_pos = chunk_scan_start;
    for _chunk_global_idx in 0..total_chunks {
        if scan_pos + 12 > bytes.len() {
            return Err(ExrError::invalid(format!(
                "multi-part: unexpected EOF at chunk scan position {scan_pos}"
            )));
        }
        let part_num = i32::from_le_bytes(bytes[scan_pos..scan_pos + 4].try_into().unwrap());
        let y_coord = i32::from_le_bytes(bytes[scan_pos + 4..scan_pos + 8].try_into().unwrap());
        let payload_size =
            i32::from_le_bytes(bytes[scan_pos + 8..scan_pos + 12].try_into().unwrap());

        if part_num < 0 || part_num as usize >= parts.len() {
            return Err(ExrError::invalid(format!(
                "multi-part chunk at {scan_pos}: part_number={part_num} out of range 0..{}",
                parts.len()
            )));
        }
        if payload_size < 0 {
            return Err(ExrError::invalid(format!(
                "multi-part chunk at {scan_pos}: negative payload size {payload_size}"
            )));
        }
        let pl_start = scan_pos + 12;
        let pl_end = pl_start + payload_size as usize;
        if pl_end > bytes.len() {
            return Err(ExrError::invalid(format!(
                "multi-part chunk at {scan_pos}: payload runs past EOF"
            )));
        }
        let payload = &bytes[pl_start..pl_end];

        let part_idx = part_num as usize;
        let req = &part_reqs[part_idx];
        let sorted_channels = &part_sorted_channels[part_idx];
        let width = req.data_window.width();
        let height = req.data_window.height();

        let row_in_image = (y_coord - req.data_window.y_min) as i64;
        if row_in_image < 0 || row_in_image as u32 >= height {
            return Err(ExrError::invalid(format!(
                "multi-part part {part_idx} chunk Y={y_coord} outside dataWindow"
            )));
        }
        let block_y0 = row_in_image as u32;
        let block_h = req.compression.scanlines_per_block();
        let lines_in_block = ((height - block_y0).min(block_h)) as usize;

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

        // B44 / B44A regroup the chunk into per-channel planes and scatter
        // directly (the shared raw fallback is handled inside the branch).
        if matches!(req.compression, Compression::B44 | Compression::B44a) {
            scatter_b44_block_into_planes(
                payload,
                sorted_channels,
                &mut planes_list[part_idx],
                width,
                block_y0,
                lines_in_block,
                uncompressed_size,
            )?;
            scan_pos = pl_end;
            continue;
        }

        let uncompressed: Vec<u8> = match req.compression {
            Compression::None => {
                if payload.len() != uncompressed_size {
                    return Err(ExrError::invalid(format!(
                        "multi-part part {part_idx}: size mismatch: have {} want {}",
                        payload.len(),
                        uncompressed_size
                    )));
                }
                payload.to_vec()
            }
            Compression::Zip | Compression::Zips => decode_zip_payload(payload, uncompressed_size)?,
            Compression::Rle => decode_rle_payload(payload, uncompressed_size)?,
            Compression::Pxr24 => decode_pxr24_payload(
                payload,
                &Pxr24RowSpec {
                    sorted_channels,
                    width,
                    block_y0,
                    lines_in_block,
                },
                uncompressed_size,
            )?,
            _ => unreachable!("filtered above"),
        };

        scatter_block_into_planes(
            &uncompressed,
            sorted_channels,
            &mut planes_list[part_idx],
            width,
            height,
            block_y0,
            lines_in_block,
        )?;

        scan_pos = pl_end;
    }

    // Assemble output images, consuming the planes_list in order.
    let mut images: Vec<ExrImage> = Vec::with_capacity(parts.len());
    let mut planes_iter = planes_list.into_iter();
    for (part_idx, part) in parts.iter().enumerate() {
        let req = &part_reqs[part_idx];
        images.push(ExrImage {
            data_window: req.data_window,
            display_window: req.display_window,
            line_order: req.line_order,
            compression: req.compression,
            pixel_aspect_ratio: req.pixel_aspect_ratio,
            screen_window_center: req.screen_window_center,
            screen_window_width: req.screen_window_width,
            channels: part_sorted_channels[part_idx].clone(),
            planes: planes_iter.next().unwrap(),
            attributes: part.attributes.clone(),
        });
    }

    Ok(images)
}

/// Parse a multi-part flat (non-deep) tiled EXR file. Every part must
/// carry `type = "tiledimage"` plus the standard tiled per-part required
/// attributes (`name`, `chunkCount`, `dataWindow`, `displayWindow`,
/// `channels`, `compression`, `lineOrder`, `pixelAspectRatio`,
/// `screenWindowCenter`, `screenWindowWidth`, `tiles[tiledesc]` ONE_LEVEL).
///
/// Layout (multi-part flat tiled, version-field bit 0x1000 set; the
/// `single_tile` 0x200 bit is NOT set — per-part `type="tiledimage"` +
/// the `tiles[tiledesc]` attribute are the tile-ness discriminators,
/// mirroring the multi-part deep-tiled discipline):
///
/// ```text
/// magic(4) | version(4 with multipart=0x1000)
/// | header_0 ... NUL | header_1 ... NUL | NUL          (extra NUL = end-of-headers)
/// | offset_table_0(chunkCount_0×u64) | offset_table_1(...) | ...
/// | chunks: each starts with i32 part_number,
///           then i32 tx | i32 ty | i32 lvlx | i32 lvly | i32 size | payload[size].
/// ```
///
/// Per-tile payload layout is identical to single-part flat tiled:
/// row-major within the tile, channels in alphabetical order, edge
/// tiles store only the valid pixel rectangle. ONE_LEVEL only.
/// Compression NONE / ZIP / ZIPS / RLE supported.
///
/// **Offset table robustness**: like the scanline multi-part reader,
/// we decode chunks by linear scan rather than index lookup so that
/// zero-filled offset tables (some reference flows emit them) still
/// decode correctly.
///
/// Companion to [`crate::encode_exr_multipart_tiled`].
pub fn parse_exr_multipart_tiled(bytes: &[u8]) -> Result<Vec<ExrImage>> {
    let parts = parse_multipart_headers(bytes)?;
    if parts.is_empty() {
        return Err(ExrError::invalid(
            "multi-part tiled file has no parts".to_string(),
        ));
    }
    // Every part must carry `type = "tiledimage"`. (Deep tiled / deep
    // scanline / flat scanline have their own entry points.)
    for (i, part) in parts.iter().enumerate() {
        let part_type = find_part_type(&part.attributes).ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part tiled: part {i} missing required 'type' attribute"
            ))
        })?;
        if part_type != "tiledimage" {
            return Err(ExrError::unsupported(format!(
                "multi-part tiled: part {i} type='{part_type}' \
                 (only 'tiledimage' supported — 'scanlineimage' routes through \
                 parse_exr_multipart, deep types route through parse_exr_deep_multipart \
                 / parse_exr_multipart_deep_tiled)"
            )));
        }
    }
    if parts[0].version.non_image {
        return Err(ExrError::invalid(
            "parse_exr_multipart_tiled called on a deep (non_image bit set) file \
             — use parse_exr_multipart_deep_tiled() instead"
                .to_string(),
        ));
    }

    // Collect chunkCount per part (mandatory in multi-part files).
    let mut chunk_counts: Vec<usize> = Vec::with_capacity(parts.len());
    for (i, part) in parts.iter().enumerate() {
        let cc = find_chunk_count(&part.attributes).ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part tiled part {i} missing required chunkCount attribute"
            ))
        })?;
        chunk_counts.push(cc);
    }

    // Per-part state: required attrs, tile geometry, sorted channels, planes.
    struct PartState {
        req: RequiredAttrs,
        sorted_channels: Vec<Channel>,
        tile_x: u32,
        tile_y: u32,
        tx_count: u32,
        #[allow(dead_code)]
        ty_count: u32,
        planes: Vec<ExrPlane>,
    }

    let mut state: Vec<PartState> = Vec::with_capacity(parts.len());
    for (part_idx, part) in parts.iter().enumerate() {
        let req = extract_required(&part.attributes)?;
        if !matches!(
            req.compression,
            Compression::None
                | Compression::Zip
                | Compression::Zips
                | Compression::Rle
                | Compression::Pxr24
                | Compression::B44
                | Compression::B44a
        ) {
            return Err(ExrError::unsupported(format!(
                "multi-part tiled part {part_idx}: compression {:?} not yet implemented",
                req.compression
            )));
        }
        let width = req.data_window.width();
        let height = req.data_window.height();
        if width == 0 || height == 0 {
            return Err(ExrError::invalid(format!(
                "multi-part tiled part {part_idx}: dataWindow {width}×{height} must be > 0"
            )));
        }
        let mut sorted_channels = req.channels.clone();
        sorted_channels.sort_by(|a, b| a.name.cmp(&b.name));
        for ch in &sorted_channels {
            if ch.x_sampling != 1 || ch.y_sampling != 1 {
                return Err(ExrError::unsupported(format!(
                    "multi-part tiled part {part_idx}: sub-sampled channel '{}' \
                     (tiled files require 1×1 sampling)",
                    ch.name
                )));
            }
        }
        let tdesc_attr = part
            .attributes
            .iter()
            .find(|a| a.name == "tiles")
            .ok_or_else(|| {
                ExrError::invalid(format!(
                    "multi-part tiled part {part_idx} missing required 'tiles' attribute"
                ))
            })?;
        let tdesc = tiledesc_from_attribute(&tdesc_attr.value)?;
        if tdesc.level_mode == 1 || tdesc.level_mode == 2 {
            return Err(ExrError::unsupported(format!(
                "multi-part tiled part {part_idx}: tiledesc level_mode={} (MIPMAP_LEVELS \
                 or RIPMAP_LEVELS) — call parse_exr_multipart_tiled_multilevel() for \
                 multi-level multi-part tiled files",
                tdesc.level_mode
            )));
        }
        if tdesc.level_mode != 0 {
            return Err(ExrError::unsupported(format!(
                "multi-part tiled part {part_idx}: tiledesc level_mode={} \
                 (only ONE_LEVEL + MIPMAP_LEVELS + RIPMAP_LEVELS supported)",
                tdesc.level_mode
            )));
        }
        if tdesc.x_size == 0 || tdesc.y_size == 0 {
            return Err(ExrError::invalid(format!(
                "multi-part tiled part {part_idx}: tile size {}×{} must both be > 0",
                tdesc.x_size, tdesc.y_size
            )));
        }
        let tx_count = width.div_ceil(tdesc.x_size);
        let ty_count = height.div_ceil(tdesc.y_size);
        let expected = (tx_count as usize) * (ty_count as usize);
        if chunk_counts[part_idx] != expected {
            return Err(ExrError::invalid(format!(
                "multi-part tiled part {part_idx}: chunkCount={} but tile grid \
                 {tx_count}×{ty_count} expects {expected}",
                chunk_counts[part_idx]
            )));
        }

        let planes: Vec<ExrPlane> = sorted_channels
            .iter()
            .map(|c| ExrPlane {
                name: c.name.clone(),
                samples: vec![0.0; (width as usize) * (height as usize)],
            })
            .collect();
        state.push(PartState {
            req,
            sorted_channels,
            tile_x: tdesc.x_size,
            tile_y: tdesc.y_size,
            tx_count,
            ty_count,
            planes,
        });
    }

    // Skip past all concatenated offset tables.
    let total_chunks: usize = chunk_counts.iter().sum();
    let tables_start = parts.last().unwrap().end_offset;
    let chunk_scan_start = tables_start + total_chunks * 8;
    if chunk_scan_start > bytes.len() {
        return Err(ExrError::invalid(format!(
            "multi-part tiled offset tables run past EOF (need {}, have {})",
            chunk_scan_start,
            bytes.len()
        )));
    }

    // Linear chunk scan: each chunk starts with i32 part_number, then
    // i32 tx, i32 ty, i32 lvlx, i32 lvly, i32 size, payload[size].
    let mut scan_pos = chunk_scan_start;
    for _chunk_global_idx in 0..total_chunks {
        if scan_pos + 24 > bytes.len() {
            return Err(ExrError::invalid(format!(
                "multi-part tiled: unexpected EOF at chunk scan position {scan_pos}"
            )));
        }
        let part_num = i32::from_le_bytes(bytes[scan_pos..scan_pos + 4].try_into().unwrap());
        let h_tx = i32::from_le_bytes(bytes[scan_pos + 4..scan_pos + 8].try_into().unwrap());
        let h_ty = i32::from_le_bytes(bytes[scan_pos + 8..scan_pos + 12].try_into().unwrap());
        let lvl_x = i32::from_le_bytes(bytes[scan_pos + 12..scan_pos + 16].try_into().unwrap());
        let lvl_y = i32::from_le_bytes(bytes[scan_pos + 16..scan_pos + 20].try_into().unwrap());
        let payload_size =
            i32::from_le_bytes(bytes[scan_pos + 20..scan_pos + 24].try_into().unwrap());

        if part_num < 0 || part_num as usize >= parts.len() {
            return Err(ExrError::invalid(format!(
                "multi-part tiled chunk at {scan_pos}: part_number={part_num} out of range 0..{}",
                parts.len()
            )));
        }
        if payload_size < 0 {
            return Err(ExrError::invalid(format!(
                "multi-part tiled chunk at {scan_pos}: negative payload size {payload_size}"
            )));
        }
        if lvl_x != 0 || lvl_y != 0 {
            return Err(ExrError::unsupported(format!(
                "multi-part tiled chunk at {scan_pos}: lvlx={lvl_x} lvly={lvl_y} \
                 (parse_exr_multipart_tiled is ONE_LEVEL only — call \
                 parse_exr_multipart_tiled_multilevel() for MIPMAP/RIPMAP)"
            )));
        }
        let pl_start = scan_pos + 24;
        let pl_end = pl_start + payload_size as usize;
        if pl_end > bytes.len() {
            return Err(ExrError::invalid(format!(
                "multi-part tiled chunk at {scan_pos}: payload runs past EOF"
            )));
        }
        let part_idx = part_num as usize;
        let ps = &mut state[part_idx];
        let width = ps.req.data_window.width();
        let height = ps.req.data_window.height();
        let tx = h_tx as u32;
        let ty = h_ty as u32;
        if tx >= ps.tx_count || ty >= ps.ty_count {
            return Err(ExrError::invalid(format!(
                "multi-part tiled chunk at {scan_pos}: tile ({tx},{ty}) out of grid \
                 {}×{}",
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
    }

    // Assemble per-part ExrImage outputs.
    let mut images: Vec<ExrImage> = Vec::with_capacity(parts.len());
    for (part_idx, part) in parts.iter().enumerate() {
        let PartState {
            req,
            sorted_channels,
            planes,
            ..
        } = state.remove(0);
        let _ = part_idx;
        images.push(ExrImage {
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
        });
    }
    Ok(images)
}

/// One multi-level part decoded by
/// [`parse_exr_multipart_tiled_multilevel`]. Carries per-part metadata
/// once plus every decoded pyramid level (in the spec's iteration
/// order — MIPMAP_LEVELS produces levels `0..N-1` along the `(l, l)`
/// diagonal; RIPMAP_LEVELS produces the full 2-D grid in `lvly`-outer
/// `lvlx`-inner order).
#[derive(Debug, Clone)]
pub struct MultilevelTiledPart {
    /// `0 = ONE_LEVEL`, `1 = MIPMAP_LEVELS`, `2 = RIPMAP_LEVELS`.
    /// MIPMAP comes from [`encode_exr_multipart_tiled_mipmap`]; RIPMAP
    /// comes from [`crate::encode_exr_multipart_tiled_ripmap`].
    pub level_mode: u8,
    /// `0 = ROUND_DOWN`, `1 = ROUND_UP`. The encoder emits ROUND_DOWN.
    pub round_mode: u8,
    /// Tile dimensions from the part's `tiles[tiledesc]` attribute.
    pub tile_x: u32,
    pub tile_y: u32,
    /// Full-resolution data window (level 0,0).
    pub data_window: Box2i,
    pub display_window: Box2i,
    /// Channel list (alphabetical, matching each level's plane order).
    pub channels: Vec<Channel>,
    /// Compression declared by the part (already applied to each
    /// level's planes — callers see decoded f32 samples).
    pub compression: Compression,
    /// Decoded pyramid levels in spec iteration order.
    pub levels: Vec<TiledLevel>,
    /// The full part attribute list (for callers that need access to
    /// optional attributes like `name`).
    pub attributes: Vec<Attribute>,
}

/// Parse a multi-part **multi-level** flat (non-deep) tiled EXR file
/// and return one [`MultilevelTiledPart`] per part. Companion to
/// [`crate::encode_exr_multipart_tiled_mipmap`].
///
/// Accepts every part shape [`parse_exr_multipart_tiled`] does (each
/// part `type="tiledimage"`, `tiles[tiledesc]` attribute, the standard
/// required attributes) **plus** parts whose tiledesc declares
/// `level_mode == 1` (MIPMAP_LEVELS) or `level_mode == 2`
/// (RIPMAP_LEVELS). ONE_LEVEL parts (`level_mode == 0`) surface as a
/// single-entry `levels` vector for uniform handling alongside
/// multi-level parts. Compression NONE / ZIP / ZIPS / RLE.
///
/// Layout (multi-part flat tiled, version-field bit 0x1000 set):
///
/// ```text
/// magic(4) | version(4 with multipart=0x1000)
/// | header_0 ... NUL | header_1 ... NUL | NUL          (extra NUL = end-of-headers)
/// | offset_table_0(chunkCount_0×u64) | offset_table_1(...) | ...
/// | chunks: each starts with i32 part_number,
///           then i32 tx | i32 ty | i32 lvlx | i32 lvly | i32 size | payload[size].
/// ```
///
/// Per-tile payload layout matches the single-part flat-tiled and
/// single-part MIPMAP encoders: row-major within the tile, channels in
/// alphabetical order, edge tiles store only the valid pixel rectangle.
///
/// **Offset table robustness**: chunks are decoded by linear scan (not
/// index lookup), matching the round-192 ONE_LEVEL multi-part reader,
/// so zero-filled offset tables still decode correctly.
///
/// For RIPMAP_LEVELS parts, the chunks visit `(lvlx, lvly)` cells in
/// `lvly`-outer / `lvlx`-inner order with INCREASING_Y row-major within
/// each cell (the encoder companion writes them in the same order).
pub fn parse_exr_multipart_tiled_multilevel(bytes: &[u8]) -> Result<Vec<MultilevelTiledPart>> {
    let parts = parse_multipart_headers(bytes)?;
    if parts.is_empty() {
        return Err(ExrError::invalid(
            "multi-part tiled multilevel file has no parts".to_string(),
        ));
    }
    for (i, part) in parts.iter().enumerate() {
        let part_type = find_part_type(&part.attributes).ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part tiled multilevel: part {i} missing required 'type' attribute"
            ))
        })?;
        if part_type != "tiledimage" {
            return Err(ExrError::unsupported(format!(
                "multi-part tiled multilevel: part {i} type='{part_type}' \
                 (only 'tiledimage' supported)"
            )));
        }
    }
    if parts[0].version.non_image {
        return Err(ExrError::invalid(
            "parse_exr_multipart_tiled_multilevel called on a deep (non_image bit set) \
             file — use parse_exr_multipart_deep_tiled() instead"
                .to_string(),
        ));
    }

    let mut chunk_counts: Vec<usize> = Vec::with_capacity(parts.len());
    for (i, part) in parts.iter().enumerate() {
        let cc = find_chunk_count(&part.attributes).ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part tiled multilevel part {i} missing required chunkCount attribute"
            ))
        })?;
        chunk_counts.push(cc);
    }

    // Per-part state: required attrs, tile geometry, sorted channels,
    // tiledesc, allocated per-level planes.
    struct PartState {
        req: RequiredAttrs,
        sorted_channels: Vec<Channel>,
        tile_x: u32,
        tile_y: u32,
        level_mode: u8,
        round_mode: u8,
        levels: Vec<TiledLevel>,
    }

    let mut state: Vec<PartState> = Vec::with_capacity(parts.len());
    for (part_idx, part) in parts.iter().enumerate() {
        let req = extract_required(&part.attributes)?;
        if !matches!(
            req.compression,
            Compression::None
                | Compression::Zip
                | Compression::Zips
                | Compression::Rle
                | Compression::Pxr24
                | Compression::B44
                | Compression::B44a
        ) {
            return Err(ExrError::unsupported(format!(
                "multi-part tiled multilevel part {part_idx}: compression {:?} not yet implemented",
                req.compression
            )));
        }
        let width = req.data_window.width();
        let height = req.data_window.height();
        if width == 0 || height == 0 {
            return Err(ExrError::invalid(format!(
                "multi-part tiled multilevel part {part_idx}: dataWindow \
                 {width}×{height} must be > 0"
            )));
        }
        let mut sorted_channels = req.channels.clone();
        sorted_channels.sort_by(|a, b| a.name.cmp(&b.name));
        for ch in &sorted_channels {
            if ch.x_sampling != 1 || ch.y_sampling != 1 {
                return Err(ExrError::unsupported(format!(
                    "multi-part tiled multilevel part {part_idx}: sub-sampled channel '{}' \
                     (tiled files require 1×1 sampling)",
                    ch.name
                )));
            }
        }
        let tdesc_attr = part
            .attributes
            .iter()
            .find(|a| a.name == "tiles")
            .ok_or_else(|| {
                ExrError::invalid(format!(
                    "multi-part tiled multilevel part {part_idx} missing required \
                     'tiles' attribute"
                ))
            })?;
        let tdesc = tiledesc_from_attribute(&tdesc_attr.value)?;
        if tdesc.x_size == 0 || tdesc.y_size == 0 {
            return Err(ExrError::invalid(format!(
                "multi-part tiled multilevel part {part_idx}: tile size {}×{} must \
                 both be > 0",
                tdesc.x_size, tdesc.y_size
            )));
        }
        if tdesc.level_mode > 2 {
            return Err(ExrError::invalid(format!(
                "multi-part tiled multilevel part {part_idx}: tiledesc level_mode={} \
                 unknown (expected 0/1/2)",
                tdesc.level_mode
            )));
        }
        let round_up = tdesc.round_mode != 0;

        // Enumerate the levels we expect for this part in the spec's
        // iteration order, then allocate per-level planes.
        // ONE_LEVEL: single (0,0) entry.
        // MIPMAP_LEVELS: diagonal levels 0..N-1 with level_x == level_y.
        // RIPMAP_LEVELS: full 2-D grid in lvly-outer / lvlx-inner order.
        let levels: Vec<TiledLevel> = match tdesc.level_mode {
            0 => vec![TiledLevel {
                level_x: 0,
                level_y: 0,
                width,
                height,
                planes: alloc_planes(&sorted_channels, width, height),
            }],
            1 => {
                let n = mipmap_level_count(width.max(height), round_up);
                (0..n)
                    .map(|l| {
                        let lw = mipmap_level_dim(width, l, round_up);
                        let lh = mipmap_level_dim(height, l, round_up);
                        TiledLevel {
                            level_x: l,
                            level_y: l,
                            width: lw,
                            height: lh,
                            planes: alloc_planes(&sorted_channels, lw, lh),
                        }
                    })
                    .collect()
            }
            2 => {
                let nx = mipmap_level_count(width, round_up);
                let ny = mipmap_level_count(height, round_up);
                let mut v = Vec::with_capacity((nx * ny) as usize);
                for ly in 0..ny {
                    let lh = mipmap_level_dim(height, ly, round_up);
                    for lx in 0..nx {
                        let lw = mipmap_level_dim(width, lx, round_up);
                        v.push(TiledLevel {
                            level_x: lx,
                            level_y: ly,
                            width: lw,
                            height: lh,
                            planes: alloc_planes(&sorted_channels, lw, lh),
                        });
                    }
                }
                v
            }
            _ => unreachable!("checked above"),
        };

        // Validate chunkCount = sum over levels of tile-grid size.
        let expected: usize = levels
            .iter()
            .map(|l| {
                l.width.div_ceil(tdesc.x_size) as usize * l.height.div_ceil(tdesc.y_size) as usize
            })
            .sum();
        if chunk_counts[part_idx] != expected {
            return Err(ExrError::invalid(format!(
                "multi-part tiled multilevel part {part_idx}: chunkCount={} but \
                 multi-level grid expects {expected}",
                chunk_counts[part_idx]
            )));
        }

        state.push(PartState {
            req,
            sorted_channels,
            tile_x: tdesc.x_size,
            tile_y: tdesc.y_size,
            level_mode: tdesc.level_mode,
            round_mode: tdesc.round_mode,
            levels,
        });
    }

    // Skip past all concatenated offset tables.
    let total_chunks: usize = chunk_counts.iter().sum();
    let tables_start = parts.last().unwrap().end_offset;
    let chunk_scan_start = tables_start + total_chunks * 8;
    if chunk_scan_start > bytes.len() {
        return Err(ExrError::invalid(format!(
            "multi-part tiled multilevel offset tables run past EOF (need {}, have {})",
            chunk_scan_start,
            bytes.len()
        )));
    }

    // Linear chunk scan.
    let mut scan_pos = chunk_scan_start;
    for _chunk_global_idx in 0..total_chunks {
        if scan_pos + 24 > bytes.len() {
            return Err(ExrError::invalid(format!(
                "multi-part tiled multilevel: unexpected EOF at chunk scan position {scan_pos}"
            )));
        }
        let part_num = i32::from_le_bytes(bytes[scan_pos..scan_pos + 4].try_into().unwrap());
        let h_tx = i32::from_le_bytes(bytes[scan_pos + 4..scan_pos + 8].try_into().unwrap());
        let h_ty = i32::from_le_bytes(bytes[scan_pos + 8..scan_pos + 12].try_into().unwrap());
        let lvl_x = i32::from_le_bytes(bytes[scan_pos + 12..scan_pos + 16].try_into().unwrap());
        let lvl_y = i32::from_le_bytes(bytes[scan_pos + 16..scan_pos + 20].try_into().unwrap());
        let payload_size =
            i32::from_le_bytes(bytes[scan_pos + 20..scan_pos + 24].try_into().unwrap());

        if part_num < 0 || part_num as usize >= parts.len() {
            return Err(ExrError::invalid(format!(
                "multi-part tiled multilevel chunk at {scan_pos}: part_number={part_num} \
                 out of range 0..{}",
                parts.len()
            )));
        }
        if payload_size < 0 || h_tx < 0 || h_ty < 0 || lvl_x < 0 || lvl_y < 0 {
            return Err(ExrError::invalid(format!(
                "multi-part tiled multilevel chunk at {scan_pos}: bad header tx={h_tx} \
                 ty={h_ty} lvlx={lvl_x} lvly={lvl_y} size={payload_size}"
            )));
        }
        let pl_start = scan_pos + 24;
        let pl_end = pl_start + payload_size as usize;
        if pl_end > bytes.len() {
            return Err(ExrError::invalid(format!(
                "multi-part tiled multilevel chunk at {scan_pos}: payload runs past EOF"
            )));
        }
        let part_idx = part_num as usize;
        let ps = &mut state[part_idx];

        // Locate the matching level slot by (lvlx, lvly).
        let level = ps
            .levels
            .iter_mut()
            .find(|l| l.level_x as i32 == lvl_x && l.level_y as i32 == lvl_y)
            .ok_or_else(|| {
                ExrError::invalid(format!(
                    "multi-part tiled multilevel chunk at {scan_pos}: unknown level \
                     ({lvl_x},{lvl_y}) on part {part_idx}"
                ))
            })?;
        let tx = h_tx as u32;
        let ty = h_ty as u32;
        let x0 = tx * ps.tile_x;
        let y0 = ty * ps.tile_y;
        if x0 >= level.width || y0 >= level.height {
            return Err(ExrError::invalid(format!(
                "multi-part tiled multilevel chunk at {scan_pos}: tile ({tx},{ty}) outside \
                 level ({lvl_x},{lvl_y}) dims {}×{}",
                level.width, level.height
            )));
        }
        let x1 = (x0 + ps.tile_x).min(level.width);
        let y1 = (y0 + ps.tile_y).min(level.height);
        let tw = (x1 - x0) as usize;
        let th = (y1 - y0) as usize;
        let payload = &bytes[pl_start..pl_end];
        scatter_tile_into_planes(
            payload,
            &ps.sorted_channels,
            &mut level.planes,
            level.width,
            x0,
            y0,
            tw,
            th,
            ps.req.compression,
            0, // tile_idx is only used for diagnostics
        )?;
        scan_pos = pl_end;
    }

    // Assemble outputs in part order.
    let mut out: Vec<MultilevelTiledPart> = Vec::with_capacity(parts.len());
    for (part_idx, part) in parts.iter().enumerate() {
        let PartState {
            req,
            sorted_channels,
            tile_x,
            tile_y,
            level_mode,
            round_mode,
            levels,
            ..
        } = state.remove(0);
        let _ = part_idx;
        out.push(MultilevelTiledPart {
            level_mode,
            round_mode,
            tile_x,
            tile_y,
            data_window: req.data_window,
            display_window: req.display_window,
            channels: sorted_channels,
            compression: req.compression,
            levels,
            attributes: part.attributes.clone(),
        });
    }
    Ok(out)
}

/// Find the `type` (string) attribute in a part's attribute list. Used
/// to discriminate `scanlineimage` from `tiledimage` / `deepscanline`
/// / `deeptile` in multi-part files.
///
/// Accepts both the typed [`AttributeValue::String`] variant (produced
/// by `parse_attribute_value`) and the legacy
/// `AttributeValue::Other { type_name: "string", data }` shape that
/// older encoder modules in this crate still emit. The two are
/// equivalent on disk; only the in-memory representation differs.
pub(crate) fn find_part_type(attrs: &[Attribute]) -> Option<String> {
    for a in attrs {
        if a.name == "type" {
            match &a.value {
                AttributeValue::String(s) => return Some(s.clone()),
                AttributeValue::Other { type_name, data } if type_name == "string" => {
                    return Some(String::from_utf8_lossy(data).to_string());
                }
                _ => {}
            }
        }
    }
    None
}

/// Find the `chunkCount` attribute in a part's attribute list.
///
/// Accepts both [`AttributeValue::Int`] and the legacy
/// `AttributeValue::Other { type_name: "int", data }` (4 bytes).
pub(crate) fn find_chunk_count(attrs: &[Attribute]) -> Option<usize> {
    for a in attrs {
        if a.name == "chunkCount" {
            match &a.value {
                AttributeValue::Int(v) if *v >= 0 => return Some(*v as usize),
                AttributeValue::Other { type_name, data }
                    if type_name == "int" && data.len() == 4 =>
                {
                    let v = i32::from_le_bytes(data[..4].try_into().unwrap());
                    if v >= 0 {
                        return Some(v as usize);
                    }
                }
                _ => {}
            }
        }
    }
    None
}

/// Public-`crate` wrapper around [`zlib_inflate`] so sibling modules
/// (e.g. the mixed multi-part reader) can reuse the same `flate2`-backed
/// decompressor without duplicating the inflate plumbing.
pub(crate) fn zlib_inflate_pub(data: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    zlib_inflate(data, expected_size)
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
