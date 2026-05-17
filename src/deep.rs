//! Deep-data scanline EXR reader (and self-roundtrip writer).
//!
//! Round-73 scaffold for the "deep" branch of the openexr.com spec. A
//! deep image stores an arbitrary number of samples at every pixel
//! position (typical use: list of (z, opacity, RGBA) per pixel for
//! Z-compositing of semi-transparent volumetric layers).
//!
//! On-disk layout, derived directly from the openexr.com File Layout
//! page §Deep scanline part:
//!
//! ```text
//! header: required attributes PLUS
//!   - type = "deepscanline" (string)
//!   - version = 1 (int)
//!   - chunkCount = ceil(height / scanlines_per_block) (int)
//!   - maxSamplesPerPixel = max-over-pixels samples count (int)
//! version-field bit 11 set (non_image / deep).
//! Single-part deep files do NOT set the multipart bit (12);
//! multipart-deep is a followup.
//!
//! After the header (single-NUL terminated), an offset table with one
//! u64 per chunk, then chunks. Each deep scanline chunk:
//!
//!   y_coord                              : i32
//!   packed_pixel_offset_table_size       : u64
//!   packed_sample_data_size              : u64
//!   unpacked_sample_data_size            : u64
//!   compressed pixel offset table        : packed_pixel_offset_table_size bytes
//!   compressed sample data               : packed_sample_data_size bytes
//! ```
//!
//! The pixel offset table is a list of `i32` (one per *column* in the
//! data window) holding the *cumulative* number of samples from
//! column 0 through this column inclusive. After decompression the
//! sample data is laid out non-interleaved: all of channel 0's samples
//! first (in pixel scan order), then channel 1's, and so on.
//!
//! Compression: the openexr.com spec page lists NONE / RLE / ZIPS / ZIP
//! as the permitted set for deep data, but empirical testing against
//! the reference `exrinfo` binary shows it rejects ZIP_COMPRESSION
//! (16-scanline blocks) with `EXR_ERR_INVALID_ATTR: Invalid compression
//! for deep data`. The spec is ambiguous here; we follow the reference
//! and accept only NONE / RLE / ZIPS (single-scanline blocks) on both
//! encode and decode. PIZ is forbidden in the spec text.
//!
//! The deep payload paths reuse the existing
//! [`crate::decoder::apply_zip_unpredictor`] /
//! [`crate::decoder::apply_zip_uninterleave`] pipeline because the
//! openexr.com spec applies the same byte-level predictor/interleave
//! transforms to deep ZIP/ZIPS/RLE as to flat ZIP/ZIPS/RLE. The
//! pixel-offset-table chunk is compressed independently using the same
//! algorithm.

use crate::decoder::{
    apply_zip_interleave, apply_zip_predictor, apply_zip_uninterleave, apply_zip_unpredictor,
};
use crate::error::{ExrError, Result};
use crate::header::{encode_attribute_value, parse_header, VersionField};
use crate::rle::{rle_compress, rle_decompress};
use crate::types::{
    Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType, EXR_MAGIC,
};

/// One deep-scanline image, returned by [`parse_exr_deep_scanline`].
///
/// `samples` is a flat row-major-of-pixels-then-channels structure: for
/// pixel index `p = y * width + x`, the `samples_per_pixel[p]` samples
/// for channel `c` start at byte offset given by walking the offsets
/// table. To avoid forcing a particular per-pixel container shape on
/// callers, we instead expose:
///
/// * `samples_per_pixel`: one `u32` per pixel (`width * height` long).
/// * `channels`: the channel list (alphabetical order).
/// * `channel_samples`: one `Vec<f32>` per channel, length =
///   `samples_per_pixel.iter().sum::<u32>() as usize`, laid out in
///   pixel-scan order; pixel `p`'s slice spans
///   `[prefix_sum[p]..prefix_sum[p] + samples_per_pixel[p]]`.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepExrImage {
    pub data_window: Box2i,
    pub display_window: Box2i,
    pub line_order: LineOrder,
    pub compression: Compression,
    pub channels: Vec<Channel>,
    /// `width * height` long.
    pub samples_per_pixel: Vec<u32>,
    /// One `Vec<f32>` per channel; total length per channel equals the
    /// sum of `samples_per_pixel` (every channel carries one value per
    /// sample). UINT channels store the u32 value reinterpreted as f32
    /// (matching the flat-EXR `Uint` convention in
    /// [`crate::image::ExrPlane`]).
    pub channel_samples: Vec<Vec<f32>>,
    pub attributes: Vec<Attribute>,
}

impl DeepExrImage {
    pub fn width(&self) -> u32 {
        self.data_window.width()
    }
    pub fn height(&self) -> u32 {
        self.data_window.height()
    }
    /// Total number of samples across all pixels.
    pub fn total_samples(&self) -> u64 {
        self.samples_per_pixel.iter().map(|&n| n as u64).sum()
    }
}

/// Helper: cumulative-prefix-sum of a samples-per-pixel slice. Output
/// length == input length; entry `i` is the running sum *through pixel i*
/// (NOT exclusive — matches the openexr.com offset-table convention).
fn cumulative_inclusive(spp: &[u32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(spp.len());
    let mut acc: i64 = 0;
    for &n in spp {
        acc += n as i64;
        out.push(acc as i32);
    }
    out
}

/// Convert a list of column-indexed cumulative offsets back to per-pixel
/// sample counts. Inverse of [`cumulative_inclusive`] for one row.
fn per_pixel_from_cumulative(cumulative: &[i32]) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(cumulative.len());
    let mut prev: i32 = 0;
    for &c in cumulative {
        if c < prev {
            return Err(ExrError::invalid(format!(
                "deep offset table not monotonic: {prev} -> {c}"
            )));
        }
        out.push((c - prev) as u32);
        prev = c;
    }
    Ok(out)
}

/// Apply the same compress-byte pipeline used for flat scanlines:
/// interleave + predictor + (zlib | rle). Inverse of
/// [`decompress_buffer`].
fn compress_buffer(raw: &[u8], compression: Compression) -> Result<Vec<u8>> {
    Ok(match compression {
        Compression::None => raw.to_vec(),
        Compression::Zips => {
            let mut interleaved = vec![0u8; raw.len()];
            apply_zip_interleave(raw, &mut interleaved);
            apply_zip_predictor(&mut interleaved);
            let compressed = crate::encoder::zlib_deflate_pub(&interleaved)?;
            if compressed.len() >= raw.len() {
                raw.to_vec()
            } else {
                compressed
            }
        }
        Compression::Rle => {
            let mut interleaved = vec![0u8; raw.len()];
            apply_zip_interleave(raw, &mut interleaved);
            apply_zip_predictor(&mut interleaved);
            let compressed = rle_compress(&interleaved);
            if compressed.len() >= raw.len() {
                raw.to_vec()
            } else {
                compressed
            }
        }
        _ => {
            return Err(ExrError::unsupported(format!(
                "compression {compression:?} not permitted for deep data \
                 (openexr.com reference accepts only NONE/RLE/ZIPS — even \
                 though the spec page also lists ZIP, exrinfo rejects \
                 ZIP-compressed deep files with EXR_ERR_INVALID_ATTR)"
            )))
        }
    })
}

/// Inverse of [`compress_buffer`]. `unpacked_size` is the expected size
/// of the recovered uncompressed buffer.
fn decompress_buffer(
    payload: &[u8],
    unpacked_size: usize,
    compression: Compression,
) -> Result<Vec<u8>> {
    if compression == Compression::None {
        if payload.len() != unpacked_size {
            return Err(ExrError::invalid(format!(
                "deep NONE block size mismatch: have {} want {unpacked_size}",
                payload.len()
            )));
        }
        return Ok(payload.to_vec());
    }
    // Stored-uncompressed escape: if packed_size == unpacked_size the
    // encoder kept the bytes raw (per the openexr.com "store whichever
    // is smaller" rule).
    if payload.len() == unpacked_size {
        return Ok(payload.to_vec());
    }
    let inflated = match compression {
        Compression::Zips => {
            use flate2::read::ZlibDecoder;
            use std::io::Read;
            let mut out = Vec::with_capacity(unpacked_size);
            let mut dec = ZlibDecoder::new(payload);
            dec.read_to_end(&mut out)
                .map_err(|e| ExrError::invalid(format!("deep zlib inflate failed: {e}")))?;
            out
        }
        Compression::Rle => rle_decompress(payload, unpacked_size)?,
        _ => unreachable!("filtered above"),
    };
    if inflated.len() != unpacked_size {
        return Err(ExrError::invalid(format!(
            "deep inflate produced {} bytes, expected {unpacked_size}",
            inflated.len()
        )));
    }
    // Reverse predictor + interleave (matches flat-EXR pipeline).
    let mut predicted = inflated;
    apply_zip_unpredictor(&mut predicted);
    let mut out = vec![0u8; predicted.len()];
    apply_zip_uninterleave(&predicted, &mut out);
    Ok(out)
}

/// Parse a single-part deep scanline EXR. Multi-part deep is a followup.
pub fn parse_exr_deep_scanline(bytes: &[u8]) -> Result<DeepExrImage> {
    // parse_header rejects non_image; we need our own header walker so
    // we can accept bit 11. Re-use the byte layout by relaxing the
    // version check in a thin wrapper.
    let header = parse_header_allow_deep(bytes)?;
    if header.version.multipart {
        return Err(ExrError::unsupported(
            "multi-part deep EXR (use a future parse_exr_deep_multipart)".to_string(),
        ));
    }
    if !header.version.non_image {
        return Err(ExrError::invalid(
            "parse_exr_deep_scanline called on a flat (non-deep) file".to_string(),
        ));
    }
    // Verify type attribute.
    let part_type = find_string_attr(&header.attributes, "type").ok_or_else(|| {
        ExrError::invalid("deep file missing required 'type' attribute".to_string())
    })?;
    if part_type != "deepscanline" {
        return Err(ExrError::unsupported(format!(
            "deep file type='{part_type}' (only 'deepscanline' supported in this round)"
        )));
    }
    // Required: chunkCount, dataWindow, displayWindow, channels,
    // compression, lineOrder.
    let chunk_count = find_int_attr(&header.attributes, "chunkCount").ok_or_else(|| {
        ExrError::invalid("deep file missing required 'chunkCount' attribute".to_string())
    })? as usize;
    let data_window = find_box2i(&header.attributes, "dataWindow").ok_or_else(|| {
        ExrError::invalid("deep file missing required 'dataWindow' attribute".to_string())
    })?;
    let display_window = find_box2i(&header.attributes, "displayWindow").unwrap_or(data_window);
    let line_order =
        find_line_order(&header.attributes, "lineOrder").unwrap_or(LineOrder::IncreasingY);
    let compression = find_compression(&header.attributes, "compression").ok_or_else(|| {
        ExrError::invalid("deep file missing required 'compression' attribute".to_string())
    })?;
    if !matches!(
        compression,
        Compression::None | Compression::Rle | Compression::Zips
    ) {
        return Err(ExrError::invalid(format!(
            "deep file uses compression {compression:?} (openexr.com reference \
             accepts only NONE/RLE/ZIPS for deep — ZIP is listed in the spec \
             page but exrinfo rejects it)"
        )));
    }
    let channels = find_channels(&header.attributes).ok_or_else(|| {
        ExrError::invalid("deep file missing required 'channels' attribute".to_string())
    })?;
    let mut sorted_channels = channels.clone();
    sorted_channels.sort_by(|a, b| a.name.cmp(&b.name));
    for ch in &sorted_channels {
        if ch.x_sampling != 1 || ch.y_sampling != 1 {
            return Err(ExrError::unsupported(format!(
                "deep + sub-sampled channels (channel '{}')",
                ch.name
            )));
        }
    }

    let width = data_window.width();
    let height = data_window.height();
    if width == 0 || height == 0 {
        return Err(ExrError::invalid(format!(
            "deep dataWindow {width}×{height} must be > 0"
        )));
    }
    let block_h = compression.scanlines_per_block();
    let expected_chunks = height.div_ceil(block_h) as usize;
    if chunk_count != expected_chunks {
        return Err(ExrError::invalid(format!(
            "deep chunkCount={chunk_count} disagrees with height/block_h math ({expected_chunks})"
        )));
    }

    // Offset table.
    let mut pos = header.end_offset;
    if pos + chunk_count * 8 > bytes.len() {
        return Err(ExrError::invalid(
            "deep offset table runs past EOF".to_string(),
        ));
    }
    let mut offsets = Vec::with_capacity(chunk_count);
    for _ in 0..chunk_count {
        let off = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
        offsets.push(off);
        pos += 8;
    }

    // We accumulate per-row sample counts and per-channel sample values
    // for the entire image, in pixel-scan order.
    let mut samples_per_pixel: Vec<u32> = vec![0; (width as usize) * (height as usize)];
    // Pre-allocate per-channel sample storage. We don't know the total
    // yet, so push as we go.
    let mut channel_samples: Vec<Vec<f32>> =
        (0..sorted_channels.len()).map(|_| Vec::new()).collect();

    for (block_idx, &block_off) in offsets.iter().enumerate() {
        if block_off + 4 + 8 + 8 + 8 > bytes.len() {
            return Err(ExrError::invalid(format!(
                "deep block {block_idx} header past EOF"
            )));
        }
        let y_coord = i32::from_le_bytes(bytes[block_off..block_off + 4].try_into().unwrap());
        let packed_table =
            u64::from_le_bytes(bytes[block_off + 4..block_off + 12].try_into().unwrap()) as usize;
        let packed_data =
            u64::from_le_bytes(bytes[block_off + 12..block_off + 20].try_into().unwrap()) as usize;
        let unpacked_data =
            u64::from_le_bytes(bytes[block_off + 20..block_off + 28].try_into().unwrap()) as usize;
        let table_start = block_off + 28;
        let table_end = table_start + packed_table;
        let data_start = table_end;
        let data_end = data_start + packed_data;
        if data_end > bytes.len() {
            return Err(ExrError::invalid(format!(
                "deep block {block_idx} payload past EOF"
            )));
        }

        let row_in_image = (y_coord - data_window.y_min) as i64;
        if row_in_image < 0 || row_in_image as u32 >= height {
            return Err(ExrError::invalid(format!(
                "deep block {block_idx} Y={y_coord} outside dataWindow"
            )));
        }
        let block_y0 = row_in_image as u32;
        let rows_in_block = ((height - block_y0).min(block_h)) as usize;
        let entries_in_table = rows_in_block * width as usize;
        let unpacked_table_size = entries_in_table * 4;

        // Decompress the offset table.
        let table_bytes = decompress_buffer(
            &bytes[table_start..table_end],
            unpacked_table_size,
            compression,
        )?;
        let mut cumulative_flat: Vec<i32> = Vec::with_capacity(entries_in_table);
        for chunk in table_bytes.chunks_exact(4) {
            cumulative_flat.push(i32::from_le_bytes(chunk.try_into().unwrap()));
        }
        if cumulative_flat.len() != entries_in_table {
            return Err(ExrError::invalid(format!(
                "deep block {block_idx} offset-table size mismatch ({} != {entries_in_table})",
                cumulative_flat.len()
            )));
        }

        // Per-row, derive per-pixel samples from the row's cumulative
        // slice. The spec says the table is per-column-in-dataWindow,
        // so each row has `width` entries; restart at row boundaries.
        let mut block_samples_total: u64 = 0;
        for r in 0..rows_in_block {
            let row_slice = &cumulative_flat[r * width as usize..(r + 1) * width as usize];
            let per_pixel = per_pixel_from_cumulative(row_slice)?;
            let dst_row = block_y0 as usize + r;
            let dst_base = dst_row * width as usize;
            for (i, &n) in per_pixel.iter().enumerate() {
                samples_per_pixel[dst_base + i] = n;
                block_samples_total += n as u64;
            }
        }

        // Decompress sample data (one big buffer, channels concatenated
        // non-interleaved).
        let block_bpp: usize = sorted_channels
            .iter()
            .map(|c| c.pixel_type.bytes_per_sample())
            .sum();
        let expected_unpacked = block_samples_total as usize * block_bpp;
        if expected_unpacked != unpacked_data {
            return Err(ExrError::invalid(format!(
                "deep block {block_idx}: derived unpacked_data={expected_unpacked} \
                 disagrees with header unpacked_data={unpacked_data}"
            )));
        }
        let sample_bytes =
            decompress_buffer(&bytes[data_start..data_end], unpacked_data, compression)?;
        // Scatter channel-major: channel 0's bytes first, then channel 1's, ...
        let mut p = 0usize;
        for (ch_idx, ch) in sorted_channels.iter().enumerate() {
            let bps = ch.pixel_type.bytes_per_sample();
            let need = block_samples_total as usize * bps;
            if p + need > sample_bytes.len() {
                return Err(ExrError::invalid(format!(
                    "deep block {block_idx}: channel {} bytes past payload end",
                    ch.name
                )));
            }
            for s in 0..(block_samples_total as usize) {
                let off = p + s * bps;
                let v = match ch.pixel_type {
                    PixelType::Half => crate::half::half_to_f32(u16::from_le_bytes(
                        sample_bytes[off..off + 2].try_into().unwrap(),
                    )),
                    PixelType::Float => {
                        f32::from_le_bytes(sample_bytes[off..off + 4].try_into().unwrap())
                    }
                    PixelType::Uint => {
                        let bits =
                            u32::from_le_bytes(sample_bytes[off..off + 4].try_into().unwrap());
                        bits as f32
                    }
                };
                channel_samples[ch_idx].push(v);
            }
            p += need;
        }
        if p != sample_bytes.len() {
            return Err(ExrError::invalid(format!(
                "deep block {block_idx}: consumed {p} of {} payload bytes",
                sample_bytes.len()
            )));
        }
    }

    Ok(DeepExrImage {
        data_window,
        display_window,
        line_order,
        compression,
        channels: sorted_channels,
        samples_per_pixel,
        channel_samples,
        attributes: header.attributes,
    })
}

/// Same as [`parse_header`] but tolerates the deep / non-image version
/// bit (which [`parse_header`] rejects).
fn parse_header_allow_deep(bytes: &[u8]) -> Result<crate::header::ParsedHeader> {
    // Quick check first: if the non_image bit isn't set, just delegate
    // (avoids duplicating header walker code).
    if bytes.len() >= 8 {
        let v = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let vf = VersionField::from_u32(v);
        if !vf.non_image {
            return parse_header(bytes);
        }
    }
    // Non-image bit is set. Temporarily clear it, parse, then restore.
    let mut tmp = bytes.to_vec();
    if tmp.len() < 8 {
        return parse_header(&tmp); // will fail with magic/version error
    }
    let v = u32::from_le_bytes(tmp[4..8].try_into().unwrap());
    let cleared = v & !0x800;
    tmp[4..8].copy_from_slice(&cleared.to_le_bytes());
    let mut parsed = parse_header(&tmp)?;
    // Re-apply the bit on the returned VersionField so callers see the
    // real flags.
    let real_vf = VersionField::from_u32(v);
    parsed.version = real_vf;
    Ok(parsed)
}

fn find_string_attr(attrs: &[Attribute], name: &str) -> Option<String> {
    for a in attrs {
        if a.name == name {
            if let AttributeValue::Other { type_name, data } = &a.value {
                if type_name == "string" {
                    return Some(String::from_utf8_lossy(data).to_string());
                }
            }
        }
    }
    None
}

fn find_int_attr(attrs: &[Attribute], name: &str) -> Option<i32> {
    for a in attrs {
        if a.name == name {
            if let AttributeValue::Other { type_name, data } = &a.value {
                if type_name == "int" && data.len() == 4 {
                    return Some(i32::from_le_bytes(data[..4].try_into().unwrap()));
                }
            }
        }
    }
    None
}

fn find_box2i(attrs: &[Attribute], name: &str) -> Option<Box2i> {
    for a in attrs {
        if a.name == name {
            if let AttributeValue::Box2i(b) = &a.value {
                return Some(*b);
            }
        }
    }
    None
}

fn find_line_order(attrs: &[Attribute], name: &str) -> Option<LineOrder> {
    for a in attrs {
        if a.name == name {
            if let AttributeValue::LineOrder(l) = &a.value {
                return Some(*l);
            }
        }
    }
    None
}

fn find_compression(attrs: &[Attribute], name: &str) -> Option<Compression> {
    for a in attrs {
        if a.name == name {
            if let AttributeValue::Compression(c) = &a.value {
                return Some(*c);
            }
        }
    }
    None
}

fn find_channels(attrs: &[Attribute]) -> Option<Vec<Channel>> {
    for a in attrs {
        if a.name == "channels" {
            if let AttributeValue::Channels(c) = &a.value {
                return Some(c.clone());
            }
        }
    }
    None
}

// ---------------------------------------------------------------------
// Writer: emits the self-roundtrip-validated layout described above.
// Not exposed in the README capability matrix yet — it's an internal
// scaffold for testing [`parse_exr_deep_scanline`] until we wire up a
// real deep encoder.
// ---------------------------------------------------------------------

/// Input descriptor for [`encode_exr_deep_scanline`].
pub struct DeepScanlineInput<'a> {
    pub width: u32,
    pub height: u32,
    /// Channels in alphabetical order. (Sub-sampled channels are not
    /// supported in this round.)
    pub channels: Vec<Channel>,
    /// One u32 per pixel (`width * height` long) — how many samples
    /// this pixel carries.
    pub samples_per_pixel: &'a [u32],
    /// One Vec<f32> per channel, each `samples_per_pixel.iter().sum()`
    /// long, in pixel-scan order.
    pub channel_samples: Vec<&'a [f32]>,
    pub compression: Compression,
}

/// Self-roundtrip-tested deep scanline writer. Emits a single-part
/// deep file the reader [`parse_exr_deep_scanline`] can decode bit-
/// exactly back to the input.
pub fn encode_exr_deep_scanline(input: &DeepScanlineInput) -> Result<Vec<u8>> {
    if !matches!(
        input.compression,
        Compression::None | Compression::Rle | Compression::Zips
    ) {
        return Err(ExrError::unsupported(format!(
            "deep encode compression {:?} (openexr.com reference accepts only \
             NONE/RLE/ZIPS for deep — ZIP is listed in the spec page but \
             exrinfo rejects it with EXR_ERR_INVALID_ATTR)",
            input.compression
        )));
    }
    let pixels = (input.width as usize) * (input.height as usize);
    if input.samples_per_pixel.len() != pixels {
        return Err(ExrError::invalid(format!(
            "samples_per_pixel len {} != width*height = {pixels}",
            input.samples_per_pixel.len()
        )));
    }
    if input.channels.len() != input.channel_samples.len() {
        return Err(ExrError::invalid(format!(
            "channels.len()={} != channel_samples.len()={}",
            input.channels.len(),
            input.channel_samples.len()
        )));
    }
    let total_samples: u64 = input.samples_per_pixel.iter().map(|&n| n as u64).sum();
    for (ch, slc) in input.channels.iter().zip(input.channel_samples.iter()) {
        if ch.x_sampling != 1 || ch.y_sampling != 1 {
            return Err(ExrError::unsupported(format!(
                "deep encode + sub-sampled channel '{}'",
                ch.name
            )));
        }
        if slc.len() != total_samples as usize {
            return Err(ExrError::invalid(format!(
                "channel '{}' sample slice len {} != total_samples {total_samples}",
                ch.name,
                slc.len()
            )));
        }
    }
    // Channels must be alphabetical.
    for win in input.channels.windows(2) {
        if win[0].name >= win[1].name {
            return Err(ExrError::invalid(format!(
                "deep channels not alphabetical: '{}' >= '{}'",
                win[0].name, win[1].name
            )));
        }
    }

    let block_h = input.compression.scanlines_per_block();
    let chunk_count = input.height.div_ceil(block_h) as usize;
    let max_samples = input.samples_per_pixel.iter().copied().max().unwrap_or(0) as i32;

    // Header attributes.
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (input.width - 1) as i32,
        y_max: (input.height - 1) as i32,
    };
    let attrs = vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(input.channels.clone()),
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
            value: AttributeValue::Compression(input.compression),
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
            name: "maxSamplesPerPixel".to_string(),
            value: AttributeValue::Other {
                type_name: "int".to_string(),
                data: max_samples.to_le_bytes().to_vec(),
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
            name: "type".to_string(),
            value: AttributeValue::Other {
                type_name: "string".to_string(),
                data: b"deepscanline".to_vec(),
            },
        },
        Attribute {
            name: "version".to_string(),
            value: AttributeValue::Other {
                type_name: "int".to_string(),
                data: 1i32.to_le_bytes().to_vec(),
            },
        },
    ];

    // Encode header bytes (magic + version with non_image bit set +
    // attribute table + NUL).
    let version = VersionField::from_u32(2 | 0x800);
    let mut header_bytes = Vec::with_capacity(512);
    header_bytes.extend_from_slice(&EXR_MAGIC.to_le_bytes());
    header_bytes.extend_from_slice(&version.to_u32().to_le_bytes());
    for a in &attrs {
        header_bytes.extend_from_slice(a.name.as_bytes());
        header_bytes.push(0);
        let (type_name, payload) = encode_attribute_value(&a.value);
        header_bytes.extend_from_slice(type_name.as_bytes());
        header_bytes.push(0);
        header_bytes.extend_from_slice(&(payload.len() as i32).to_le_bytes());
        header_bytes.extend_from_slice(&payload);
    }
    header_bytes.push(0); // header terminator

    // Build per-chunk payloads (still raw, before assembly).
    struct ChunkBlob {
        y: i32,
        packed_table: Vec<u8>,
        packed_data: Vec<u8>,
        unpacked_data_len: u64,
    }
    let mut chunks: Vec<ChunkBlob> = Vec::with_capacity(chunk_count);

    // Pre-compute cumulative-from-start so we can quickly slice each
    // channel's samples for a given row range.
    let row_sample_starts: Vec<u64> = {
        // cumulative-EXCLUSIVE samples up to (and not including) row r.
        let mut v = Vec::with_capacity(input.height as usize + 1);
        v.push(0u64);
        for r in 0..input.height as usize {
            let row_base = r * input.width as usize;
            let row_sum: u64 = input.samples_per_pixel[row_base..row_base + input.width as usize]
                .iter()
                .map(|&n| n as u64)
                .sum();
            let last = *v.last().unwrap();
            v.push(last + row_sum);
        }
        v
    };

    for block_idx in 0..chunk_count {
        let row0 = (block_idx as u32) * block_h;
        let rows_in_block = (input.height - row0).min(block_h) as usize;
        let entries = rows_in_block * input.width as usize;
        // Build per-row cumulative offsets.
        let mut table_bytes = Vec::with_capacity(entries * 4);
        let mut block_samples: u64 = 0;
        for r in 0..rows_in_block {
            let dst_row = row0 as usize + r;
            let row_slice = &input.samples_per_pixel
                [dst_row * input.width as usize..(dst_row + 1) * input.width as usize];
            let cumulative = cumulative_inclusive(row_slice);
            for c in cumulative {
                table_bytes.extend_from_slice(&c.to_le_bytes());
            }
            block_samples += row_slice.iter().map(|&n| n as u64).sum::<u64>();
        }
        // Slice each channel's samples for this block (channels stored
        // non-interleaved: ch0 all samples, then ch1, ...).
        let block_sample_start = row_sample_starts[row0 as usize] as usize;
        let block_sample_end = row_sample_starts[row0 as usize + rows_in_block] as usize;
        let block_sample_count = block_sample_end - block_sample_start;
        let bpp_total: usize = input
            .channels
            .iter()
            .map(|c| c.pixel_type.bytes_per_sample())
            .sum();
        let mut sample_bytes: Vec<u8> = Vec::with_capacity(block_sample_count * bpp_total);
        for (ch_idx, ch) in input.channels.iter().enumerate() {
            let slc = &input.channel_samples[ch_idx][block_sample_start..block_sample_end];
            for &v in slc {
                match ch.pixel_type {
                    PixelType::Half => {
                        sample_bytes.extend_from_slice(&crate::half::f32_to_half(v).to_le_bytes())
                    }
                    PixelType::Float => sample_bytes.extend_from_slice(&v.to_le_bytes()),
                    PixelType::Uint => {
                        let u = if v.is_nan() || v < 0.0 {
                            0u32
                        } else if v >= (u32::MAX as f32) {
                            u32::MAX
                        } else {
                            (v + 0.5) as u32
                        };
                        sample_bytes.extend_from_slice(&u.to_le_bytes());
                    }
                }
            }
        }
        let packed_table = compress_buffer(&table_bytes, input.compression)?;
        let packed_data = compress_buffer(&sample_bytes, input.compression)?;
        let _ = block_samples; // (we used row_sample_starts for sample counts)
        chunks.push(ChunkBlob {
            y: row0 as i32,
            packed_table,
            packed_data,
            unpacked_data_len: sample_bytes.len() as u64,
        });
    }

    // Compute absolute offsets after header + offset table.
    let offset_table_bytes = chunk_count * 8;
    let chunks_start = header_bytes.len() + offset_table_bytes;
    let mut absolute_offsets: Vec<u64> = Vec::with_capacity(chunk_count);
    let mut running = chunks_start;
    for c in &chunks {
        absolute_offsets.push(running as u64);
        // header: i32 y + 3*u64 = 28 bytes, then table + data
        running += 28 + c.packed_table.len() + c.packed_data.len();
    }

    let mut out = Vec::with_capacity(running);
    out.extend_from_slice(&header_bytes);
    for &o in &absolute_offsets {
        out.extend_from_slice(&o.to_le_bytes());
    }
    for c in &chunks {
        out.extend_from_slice(&c.y.to_le_bytes());
        out.extend_from_slice(&(c.packed_table.len() as u64).to_le_bytes());
        out.extend_from_slice(&(c.packed_data.len() as u64).to_le_bytes());
        out.extend_from_slice(&c.unpacked_data_len.to_le_bytes());
        out.extend_from_slice(&c.packed_table);
        out.extend_from_slice(&c.packed_data);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_channels_rgba_half() -> Vec<Channel> {
        // Alphabetical: A, B, G, R.
        ["A", "B", "G", "R"]
            .iter()
            .map(|n| Channel {
                name: n.to_string(),
                pixel_type: PixelType::Half,
                p_linear: false,
                x_sampling: 1,
                y_sampling: 1,
            })
            .collect()
    }

    /// FLOAT channels for tests that need bit-exact round-trip (HALF
    /// quantization makes `0.05_f32` round to `0.049987793_f32`).
    fn mk_channels_rgba_float() -> Vec<Channel> {
        ["A", "B", "G", "R"]
            .iter()
            .map(|n| Channel {
                name: n.to_string(),
                pixel_type: PixelType::Float,
                p_linear: false,
                x_sampling: 1,
                y_sampling: 1,
            })
            .collect()
    }

    fn synthetic_deep(w: u32, h: u32) -> (Vec<u32>, [Vec<f32>; 4]) {
        // Deterministic per-pixel sample counts: 0..3 mod 4.
        let pixels = (w * h) as usize;
        let mut spp = Vec::with_capacity(pixels);
        for i in 0..pixels {
            spp.push((i as u32) % 4);
        }
        let total: usize = spp.iter().sum::<u32>() as usize;
        let mut a = Vec::with_capacity(total);
        let mut b = Vec::with_capacity(total);
        let mut g = Vec::with_capacity(total);
        let mut r = Vec::with_capacity(total);
        let mut s = 0usize;
        for &n in &spp {
            for k in 0..n as usize {
                let t = (s + k) as f32;
                r.push((t * 0.125).fract());
                g.push((t * 0.25).fract());
                b.push((t * 0.5).fract());
                a.push(if k == 0 { 1.0 } else { 0.5 });
            }
            s += n as usize;
        }
        (spp, [a, b, g, r])
    }

    #[test]
    fn deep_scanline_roundtrip_none() {
        let (spp, planes) = synthetic_deep(8, 4);
        let input = DeepScanlineInput {
            width: 8,
            height: 4,
            channels: mk_channels_rgba_half(),
            samples_per_pixel: &spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::None,
        };
        let bytes = encode_exr_deep_scanline(&input).unwrap();
        let img = parse_exr_deep_scanline(&bytes).unwrap();
        assert_eq!(img.samples_per_pixel, spp);
        for (got, want) in img.channel_samples.iter().zip(planes.iter()) {
            // HALF roundtrip exact on these synthetic values.
            assert_eq!(got, want);
        }
    }

    #[test]
    fn deep_scanline_roundtrip_zips() {
        let (spp, planes) = synthetic_deep(16, 6);
        let input = DeepScanlineInput {
            width: 16,
            height: 6,
            channels: mk_channels_rgba_half(),
            samples_per_pixel: &spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::Zips,
        };
        let bytes = encode_exr_deep_scanline(&input).unwrap();
        let img = parse_exr_deep_scanline(&bytes).unwrap();
        assert_eq!(img.samples_per_pixel, spp);
        for (got, want) in img.channel_samples.iter().zip(planes.iter()) {
            assert_eq!(got, want);
        }
    }

    #[test]
    fn deep_scanline_roundtrip_zips_multiline() {
        // 20 lines with ZIPS (1-line blocks) exercises a non-trivial
        // chunk count (20 chunks) and lets us check multi-row offset
        // tables in aggregate, even though each block is only one row.
        let (spp, planes) = synthetic_deep(12, 20);
        let input = DeepScanlineInput {
            width: 12,
            height: 20,
            channels: mk_channels_rgba_half(),
            samples_per_pixel: &spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::Zips,
        };
        let bytes = encode_exr_deep_scanline(&input).unwrap();
        let img = parse_exr_deep_scanline(&bytes).unwrap();
        assert_eq!(img.width(), 12);
        assert_eq!(img.height(), 20);
        assert_eq!(img.samples_per_pixel, spp);
        for (got, want) in img.channel_samples.iter().zip(planes.iter()) {
            assert_eq!(got, want);
        }
    }

    #[test]
    fn deep_scanline_rejects_zip_compression() {
        // ZIP (16-line block) is rejected by the openexr.com reference
        // for deep data even though the spec page lists it as permitted
        // — match that behaviour so we never write a file exrinfo will
        // refuse with EXR_ERR_INVALID_ATTR.
        let (spp, planes) = synthetic_deep(8, 4);
        let input = DeepScanlineInput {
            width: 8,
            height: 4,
            channels: mk_channels_rgba_half(),
            samples_per_pixel: &spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::Zip,
        };
        let r = encode_exr_deep_scanline(&input);
        assert!(r.is_err(), "deep encoder must reject ZIP compression");
    }

    #[test]
    fn deep_scanline_roundtrip_rle() {
        let (spp, planes) = synthetic_deep(10, 3);
        let input = DeepScanlineInput {
            width: 10,
            height: 3,
            channels: mk_channels_rgba_half(),
            samples_per_pixel: &spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::Rle,
        };
        let bytes = encode_exr_deep_scanline(&input).unwrap();
        let img = parse_exr_deep_scanline(&bytes).unwrap();
        assert_eq!(img.samples_per_pixel, spp);
        for (got, want) in img.channel_samples.iter().zip(planes.iter()) {
            assert_eq!(got, want);
        }
    }

    #[test]
    fn deep_scanline_rejects_flat_file() {
        // Hand-build a flat (non-deep) file header and ensure the
        // deep parser refuses.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&EXR_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes()); // version=2, no flags
        bytes.push(0); // empty header terminator
        let r = parse_exr_deep_scanline(&bytes);
        assert!(r.is_err());
    }

    #[test]
    fn deep_scanline_rejects_piz() {
        let (spp, planes) = synthetic_deep(2, 2);
        let input = DeepScanlineInput {
            width: 2,
            height: 2,
            channels: mk_channels_rgba_half(),
            samples_per_pixel: &spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::Piz,
        };
        let r = encode_exr_deep_scanline(&input);
        assert!(r.is_err());
    }

    #[test]
    fn deep_scanline_all_zero_samples() {
        // Every pixel carries 0 samples (degenerate but spec-legal).
        let w = 4u32;
        let h = 2u32;
        let spp = vec![0u32; (w * h) as usize];
        let empty: Vec<f32> = Vec::new();
        let input = DeepScanlineInput {
            width: w,
            height: h,
            channels: mk_channels_rgba_float(),
            samples_per_pixel: &spp,
            channel_samples: vec![&empty, &empty, &empty, &empty],
            compression: Compression::Zips,
        };
        let bytes = encode_exr_deep_scanline(&input).unwrap();
        let img = parse_exr_deep_scanline(&bytes).unwrap();
        assert_eq!(img.samples_per_pixel, spp);
        for got in &img.channel_samples {
            assert!(got.is_empty());
        }
        assert_eq!(img.total_samples(), 0);
    }
}
