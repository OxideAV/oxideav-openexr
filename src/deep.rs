//! Deep-data scanline EXR reader (and self-roundtrip writer).
//!
//! Round-73 scaffold for the "deep" branch of the OpenEXR spec. A
//! deep image stores an arbitrary number of samples at every pixel
//! position (typical use: list of (z, opacity, RGBA) per pixel for
//! Z-compositing of semi-transparent volumetric layers).
//!
//! On-disk layout, derived directly from the OpenEXR File Layout
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
//! Compression: the OpenEXR spec page lists NONE / RLE / ZIPS / ZIP
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
//! OpenEXR spec applies the same byte-level predictor/interleave
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
/// (NOT exclusive — matches the OpenEXR offset-table convention).
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
                 (reference encoder accepts only NONE/RLE/ZIPS — even \
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
    // encoder kept the bytes raw (per the OpenEXR "store whichever
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

/// One part of a multi-part deep file. Same shape as [`DeepExrImage`]
/// plus a `name` slot identifying the part (the `name` attribute is
/// mandatory on every multi-part header per the OpenEXR spec).
///
/// Returned by [`parse_exr_deep_multipart`].
#[derive(Debug, Clone, PartialEq)]
pub struct DeepScanlinePart {
    pub name: String,
    pub data_window: Box2i,
    pub display_window: Box2i,
    pub line_order: LineOrder,
    pub compression: Compression,
    pub channels: Vec<Channel>,
    /// `width * height` long.
    pub samples_per_pixel: Vec<u32>,
    /// One `Vec<f32>` per channel; total length per channel equals the
    /// sum of `samples_per_pixel`. UINT stored as the u32 bits cast to
    /// f32 (matching the flat-EXR convention).
    pub channel_samples: Vec<Vec<f32>>,
    pub attributes: Vec<Attribute>,
}

impl DeepScanlinePart {
    pub fn width(&self) -> u32 {
        self.data_window.width()
    }
    pub fn height(&self) -> u32 {
        self.data_window.height()
    }
    pub fn total_samples(&self) -> u64 {
        self.samples_per_pixel.iter().map(|&n| n as u64).sum()
    }
}

/// Parse a multi-part deep EXR file (version-field bits 0x1800
/// = multipart + non_image).
///
/// Each part must carry `type = "deepscanline"` plus the standard
/// per-part required attributes (`name`, `chunkCount`, `dataWindow`,
/// `displayWindow`, `channels`, `compression`, `lineOrder`,
/// `pixelAspectRatio`, `screenWindowCenter`, `screenWindowWidth`) and
/// the deep-specific `version` (always 1) + `maxSamplesPerPixel`.
///
/// The on-disk layout is a straight extension of flat multi-part: the
/// per-part headers double-NUL-terminate, followed by per-part offset
/// tables (each `chunkCount` × u64), followed by interleaved chunks
/// each prefixed with `i32 part_number` then the standard deep chunk
/// record `i32 Y + u64 packed_table + u64 packed_data + u64
/// unpacked_data + packed_table_bytes + packed_sample_bytes`.
///
/// Like [`crate::parse_exr_multipart`], we walk chunks linearly rather
/// than via the offset table, because the reference
/// `exrmultipart -combine` emits zero-filled offset tables for parts
/// beyond the first.
///
/// Compression: NONE / RLE / ZIPS only (matching
/// [`parse_exr_deep_scanline`] and the reference `exrinfo`'s rejection
/// of deep ZIP).
pub fn parse_exr_deep_multipart(bytes: &[u8]) -> Result<Vec<DeepScanlinePart>> {
    let parts = parse_multipart_headers_allow_deep(bytes)?;
    if parts.is_empty() {
        return Err(ExrError::invalid(
            "multi-part deep file has no parts".to_string(),
        ));
    }
    // Every part must be deepscanline. (deeptile is a followup.)
    for (i, part) in parts.iter().enumerate() {
        let part_type = find_string_attr(&part.attributes, "type").ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part deep: part {i} missing required 'type' attribute"
            ))
        })?;
        if part_type != "deepscanline" {
            return Err(ExrError::unsupported(format!(
                "multi-part deep: part {i} type='{part_type}' \
                 (only 'deepscanline' supported — 'deeptile' is a followup, \
                 'scanlineimage'/'tiledimage' would route through \
                 parse_exr_multipart)"
            )));
        }
    }
    if !parts[0].version.non_image {
        return Err(ExrError::invalid(
            "parse_exr_deep_multipart called on a multi-part file without the \
             non_image (deep) version bit set"
                .to_string(),
        ));
    }
    if !parts[0].version.multipart {
        return Err(ExrError::invalid(
            "parse_exr_deep_multipart called on a non-multipart EXR".to_string(),
        ));
    }

    // Per-part metadata.
    struct PartState {
        name: String,
        data_window: Box2i,
        display_window: Box2i,
        line_order: LineOrder,
        compression: Compression,
        channels: Vec<Channel>,
        attributes: Vec<Attribute>,
        chunk_count: usize,
        width: u32,
        height: u32,
        samples_per_pixel: Vec<u32>,
        channel_samples: Vec<Vec<f32>>,
    }

    let mut state: Vec<PartState> = Vec::with_capacity(parts.len());
    let mut chunk_counts: Vec<usize> = Vec::with_capacity(parts.len());

    for (i, part) in parts.iter().enumerate() {
        let name = find_string_attr(&part.attributes, "name").ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part deep part {i} missing required 'name' attribute"
            ))
        })?;
        let chunk_count = crate::decoder::find_chunk_count(&part.attributes).ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part deep part {i} ('{name}') missing required 'chunkCount' attribute"
            ))
        })?;
        let data_window = find_box2i(&part.attributes, "dataWindow").ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part deep part {i} ('{name}') missing required 'dataWindow' attribute"
            ))
        })?;
        let display_window = find_box2i(&part.attributes, "displayWindow").unwrap_or(data_window);
        let line_order =
            find_line_order(&part.attributes, "lineOrder").unwrap_or(LineOrder::IncreasingY);
        let compression = find_compression(&part.attributes, "compression").ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part deep part {i} ('{name}') missing required 'compression' attribute"
            ))
        })?;
        if !matches!(
            compression,
            Compression::None | Compression::Rle | Compression::Zips
        ) {
            return Err(ExrError::invalid(format!(
                "multi-part deep part {i} ('{name}') uses compression \
                 {compression:?} (reference encoder accepts only \
                 NONE/RLE/ZIPS for deep)"
            )));
        }
        let channels = find_channels(&part.attributes).ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part deep part {i} ('{name}') missing required 'channels' attribute"
            ))
        })?;
        let mut sorted_channels = channels.clone();
        sorted_channels.sort_by(|a, b| a.name.cmp(&b.name));
        for ch in &sorted_channels {
            if ch.x_sampling != 1 || ch.y_sampling != 1 {
                return Err(ExrError::unsupported(format!(
                    "multi-part deep part {i} ('{name}'): sub-sampled channel '{}'",
                    ch.name
                )));
            }
        }
        let width = data_window.width();
        let height = data_window.height();
        if width == 0 || height == 0 {
            return Err(ExrError::invalid(format!(
                "multi-part deep part {i} ('{name}'): dataWindow {width}×{height} must be > 0"
            )));
        }
        let block_h = compression.scanlines_per_block();
        let expected_chunks = height.div_ceil(block_h) as usize;
        if chunk_count != expected_chunks {
            return Err(ExrError::invalid(format!(
                "multi-part deep part {i} ('{name}'): chunkCount={chunk_count} \
                 disagrees with height/block_h math ({expected_chunks})"
            )));
        }
        chunk_counts.push(chunk_count);
        let pixels = (width as usize) * (height as usize);
        state.push(PartState {
            name,
            data_window,
            display_window,
            line_order,
            compression,
            channels: sorted_channels.clone(),
            attributes: part.attributes.clone(),
            chunk_count,
            width,
            height,
            samples_per_pixel: vec![0u32; pixels],
            channel_samples: (0..sorted_channels.len()).map(|_| Vec::new()).collect(),
        });
    }

    // Skip past the offset tables (may be zero-filled).
    let total_chunks: usize = chunk_counts.iter().sum();
    let tables_start = parts.last().unwrap().end_offset;
    let chunk_scan_start = tables_start + total_chunks * 8;
    if chunk_scan_start > bytes.len() {
        return Err(ExrError::invalid(format!(
            "multi-part deep offset tables run past EOF (need {chunk_scan_start}, have {})",
            bytes.len()
        )));
    }

    // Linear scan: each chunk is `i32 part_number + i32 Y + 3*u64
    // sizes + packed_table + packed_data`.
    let mut scan_pos = chunk_scan_start;
    for _ in 0..total_chunks {
        if scan_pos + 4 + 4 + 24 > bytes.len() {
            return Err(ExrError::invalid(format!(
                "multi-part deep: unexpected EOF at chunk scan position {scan_pos}"
            )));
        }
        let part_num = i32::from_le_bytes(bytes[scan_pos..scan_pos + 4].try_into().unwrap());
        let y_coord = i32::from_le_bytes(bytes[scan_pos + 4..scan_pos + 8].try_into().unwrap());
        let packed_table =
            u64::from_le_bytes(bytes[scan_pos + 8..scan_pos + 16].try_into().unwrap()) as usize;
        let packed_data =
            u64::from_le_bytes(bytes[scan_pos + 16..scan_pos + 24].try_into().unwrap()) as usize;
        let unpacked_data =
            u64::from_le_bytes(bytes[scan_pos + 24..scan_pos + 32].try_into().unwrap()) as usize;
        if part_num < 0 || part_num as usize >= state.len() {
            return Err(ExrError::invalid(format!(
                "multi-part deep chunk at {scan_pos}: part_number={part_num} out of range 0..{}",
                state.len()
            )));
        }
        let table_start = scan_pos + 32;
        let table_end = table_start + packed_table;
        let data_start = table_end;
        let data_end = data_start + packed_data;
        if data_end > bytes.len() {
            return Err(ExrError::invalid(format!(
                "multi-part deep chunk at {scan_pos}: payload runs past EOF"
            )));
        }

        let part_idx = part_num as usize;
        let ps = &mut state[part_idx];
        let width = ps.width;
        let height = ps.height;
        let compression = ps.compression;
        let row_in_image = (y_coord - ps.data_window.y_min) as i64;
        if row_in_image < 0 || row_in_image as u32 >= height {
            return Err(ExrError::invalid(format!(
                "multi-part deep part {part_idx} ('{}'): chunk Y={y_coord} outside dataWindow",
                ps.name
            )));
        }
        let block_y0 = row_in_image as u32;
        let block_h = compression.scanlines_per_block();
        let rows_in_block = ((height - block_y0).min(block_h)) as usize;
        let entries_in_table = rows_in_block * width as usize;
        let unpacked_table_size = entries_in_table * 4;

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
                "multi-part deep part {part_idx} ('{}'): offset-table size mismatch ({} != {entries_in_table})",
                ps.name,
                cumulative_flat.len()
            )));
        }

        let mut block_samples_total: u64 = 0;
        for r in 0..rows_in_block {
            let row_slice = &cumulative_flat[r * width as usize..(r + 1) * width as usize];
            let per_pixel = per_pixel_from_cumulative(row_slice)?;
            let dst_row = block_y0 as usize + r;
            let dst_base = dst_row * width as usize;
            for (i, &n) in per_pixel.iter().enumerate() {
                ps.samples_per_pixel[dst_base + i] = n;
                block_samples_total += n as u64;
            }
        }

        let block_bpp: usize = ps
            .channels
            .iter()
            .map(|c| c.pixel_type.bytes_per_sample())
            .sum();
        let expected_unpacked = block_samples_total as usize * block_bpp;
        if expected_unpacked != unpacked_data {
            return Err(ExrError::invalid(format!(
                "multi-part deep part {part_idx} ('{}'): derived unpacked_data={expected_unpacked} \
                 disagrees with header unpacked_data={unpacked_data}",
                ps.name
            )));
        }
        let sample_bytes =
            decompress_buffer(&bytes[data_start..data_end], unpacked_data, compression)?;
        let mut p = 0usize;
        // Snapshot channel types/names so we can borrow channel_samples mutably below.
        let channel_types: Vec<(PixelType, String)> = ps
            .channels
            .iter()
            .map(|c| (c.pixel_type, c.name.clone()))
            .collect();
        for (ch_idx, (pixel_type, ch_name)) in channel_types.iter().enumerate() {
            let bps = pixel_type.bytes_per_sample();
            let need = block_samples_total as usize * bps;
            if p + need > sample_bytes.len() {
                return Err(ExrError::invalid(format!(
                    "multi-part deep part {part_idx} ('{}'): channel '{ch_name}' bytes past payload end",
                    ps.name
                )));
            }
            for s in 0..(block_samples_total as usize) {
                let off = p + s * bps;
                let v = match pixel_type {
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
                ps.channel_samples[ch_idx].push(v);
            }
            p += need;
        }
        if p != sample_bytes.len() {
            return Err(ExrError::invalid(format!(
                "multi-part deep part {part_idx} ('{}'): consumed {p} of {} payload bytes",
                ps.name,
                sample_bytes.len()
            )));
        }

        scan_pos = data_end;
        let _ = ps.chunk_count; // reserved for future bounds checks
    }

    Ok(state
        .into_iter()
        .map(|ps| DeepScanlinePart {
            name: ps.name,
            data_window: ps.data_window,
            display_window: ps.display_window,
            line_order: ps.line_order,
            compression: ps.compression,
            channels: ps.channels,
            samples_per_pixel: ps.samples_per_pixel,
            channel_samples: ps.channel_samples,
            attributes: ps.attributes,
        })
        .collect())
}

/// Variant of [`crate::header::parse_multipart_headers`] that — like
/// [`parse_header_allow_deep`] for single-part — tolerates the
/// `non_image` bit (which `parse_multipart_headers` no longer rejects
/// directly, but we still need this wrapper for the explicit deep walk).
///
/// Today this is just a thin pass-through; the historical name is
/// retained for symmetry with the single-part deep parser.
fn parse_multipart_headers_allow_deep(bytes: &[u8]) -> Result<Vec<crate::header::ParsedHeader>> {
    crate::header::parse_multipart_headers(bytes)
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
            "deep file uses compression {compression:?} (reference encoder \
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
            "deep encode compression {:?} (reference encoder accepts only \
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

// ---------------------------------------------------------------------
// Multi-part deep scanline WRITE (round 127).
//
// On-disk layout (mirror of what `parse_exr_deep_multipart` consumes):
//
//   magic(4) | version(4 with bits 0x1800 set = multipart + non_image)
//   per-part header_0 ... NUL | header_1 ... NUL | NUL  (double-NUL EOH)
//   concatenated offset tables (per-part `chunkCount × u64`)
//   chunks each prefixed with `i32 part_number`, then the standard deep
//   chunk record:
//     `i32 Y, u64 packed_table, u64 packed_data, u64 unpacked_data,
//      table_bytes, data_bytes`
//
// Required per-part attributes (the reader's
// `parse_exr_deep_multipart` enforces these):
//   channels, chunkCount, compression, dataWindow, displayWindow,
//   lineOrder, maxSamplesPerPixel, name, pixelAspectRatio,
//   screenWindowCenter, screenWindowWidth, type = "deepscanline",
//   version = 1.
//
// Compression: NONE / RLE / ZIPS (matching the single-part deep
// encoder; deep ZIP rejected by the reference `exrinfo`).
// ---------------------------------------------------------------------

/// One part of an outgoing multi-part deep scanline file. Identical
/// shape to [`DeepScanlineInput`] plus a unique `name` slot — the
/// per-part `name` attribute is mandatory on every multi-part header
/// per the OpenEXR spec.
pub struct MultipartDeepScanlinePart<'a> {
    /// Part name (must be unique across all parts in the file).
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Channels in alphabetical order (sub-sampled channels not
    /// supported on the deep path).
    pub channels: Vec<Channel>,
    /// One u32 per pixel (`width * height` long) — how many samples
    /// this pixel carries.
    pub samples_per_pixel: &'a [u32],
    /// One f32 slice per channel, each `samples_per_pixel.iter().sum()`
    /// long in pixel-scan order. UINT stored as the u32 bits (matching
    /// the [`DeepExrImage`] convention).
    pub channel_samples: Vec<&'a [f32]>,
    pub compression: Compression,
}

/// Encode a multi-part deep scanline EXR (version-field bits 0x1800).
///
/// Each part is validated independently (alphabetical channel order,
/// `samples_per_pixel` length, per-channel sample-count totals,
/// compression in {NONE, RLE, ZIPS}, unique non-empty name, no
/// sub-sampling). The header table mirrors the per-part required
/// attribute set the reader expects.
///
/// Self-roundtrips through [`parse_exr_deep_multipart`]; intended also
/// to be readable by the reference `exrmultipart -separate` /
/// `exrheader` flow.
pub fn encode_exr_multipart_deep_scanline(parts: &[MultipartDeepScanlinePart]) -> Result<Vec<u8>> {
    if parts.is_empty() {
        return Err(ExrError::invalid(
            "encode_exr_multipart_deep_scanline: at least one part required".to_string(),
        ));
    }

    // ---- Validate every part up front (mirrors single-part rules). ----
    for (i, p) in parts.iter().enumerate() {
        if p.name.is_empty() {
            return Err(ExrError::invalid(format!("deep part {i}: empty name")));
        }
        for (j, other) in parts.iter().enumerate() {
            if j != i && other.name == p.name {
                return Err(ExrError::invalid(format!(
                    "duplicate deep part name '{}' (parts {i} and {j})",
                    p.name
                )));
            }
        }
        if !matches!(
            p.compression,
            Compression::None | Compression::Rle | Compression::Zips
        ) {
            return Err(ExrError::unsupported(format!(
                "deep part '{}' compression {:?} (reference encoder accepts \
                 only NONE/RLE/ZIPS for deep — ZIP is listed in the spec page \
                 but exrinfo rejects it with EXR_ERR_INVALID_ATTR)",
                p.name, p.compression
            )));
        }
        let pixels = (p.width as usize) * (p.height as usize);
        if p.width == 0 || p.height == 0 {
            return Err(ExrError::invalid(format!(
                "deep part '{}': dataWindow {}x{} must be > 0",
                p.name, p.width, p.height
            )));
        }
        if p.samples_per_pixel.len() != pixels {
            return Err(ExrError::invalid(format!(
                "deep part '{}': samples_per_pixel len {} != width*height = {pixels}",
                p.name,
                p.samples_per_pixel.len()
            )));
        }
        if p.channels.len() != p.channel_samples.len() {
            return Err(ExrError::invalid(format!(
                "deep part '{}': channels.len()={} != channel_samples.len()={}",
                p.name,
                p.channels.len(),
                p.channel_samples.len()
            )));
        }
        for win in p.channels.windows(2) {
            if win[0].name >= win[1].name {
                return Err(ExrError::invalid(format!(
                    "deep part '{}': channels not alphabetical: '{}' >= '{}'",
                    p.name, win[0].name, win[1].name
                )));
            }
        }
        let total_samples: u64 = p.samples_per_pixel.iter().map(|&n| n as u64).sum();
        for (ch, slc) in p.channels.iter().zip(p.channel_samples.iter()) {
            if ch.x_sampling != 1 || ch.y_sampling != 1 {
                return Err(ExrError::unsupported(format!(
                    "deep part '{}': sub-sampled channel '{}' (deep path 1x1 only)",
                    p.name, ch.name
                )));
            }
            if slc.len() != total_samples as usize {
                return Err(ExrError::invalid(format!(
                    "deep part '{}': channel '{}' sample slice len {} != \
                     total_samples {total_samples}",
                    p.name,
                    ch.name,
                    slc.len()
                )));
            }
        }
    }

    // ---- Per-part header byte blocks + chunk counts. ----
    let mut header_byte_blocks: Vec<Vec<u8>> = Vec::with_capacity(parts.len());
    let mut chunk_counts: Vec<usize> = Vec::with_capacity(parts.len());

    for p in parts {
        let block_h = p.compression.scanlines_per_block();
        let cc = p.height.div_ceil(block_h) as usize;
        chunk_counts.push(cc);
        let max_samples = p.samples_per_pixel.iter().copied().max().unwrap_or(0) as i32;
        let attrs = build_deep_part_attrs(p, cc as i32, max_samples);
        let mut hb = Vec::with_capacity(256);
        for a in &attrs {
            hb.extend_from_slice(a.name.as_bytes());
            hb.push(0);
            let (type_name, payload) = encode_attribute_value(&a.value);
            hb.extend_from_slice(type_name.as_bytes());
            hb.push(0);
            hb.extend_from_slice(&(payload.len() as i32).to_le_bytes());
            hb.extend_from_slice(&payload);
        }
        header_byte_blocks.push(hb);
    }

    // ---- Stitch magic + version + headers + double-NUL terminator. ----
    let version = VersionField::from_u32(2 | 0x800 | 0x1000); // non_image + multipart
    let mut out: Vec<u8> = Vec::with_capacity(1024);
    out.extend_from_slice(&EXR_MAGIC.to_le_bytes());
    out.extend_from_slice(&version.to_u32().to_le_bytes());
    for hb in &header_byte_blocks {
        out.extend_from_slice(hb);
        out.push(0); // per-part header terminator
    }
    out.push(0); // double-NUL = end-of-all-headers

    // ---- Build per-chunk payloads (still raw, before assembly). ----
    struct ChunkBlob {
        part_idx: u32,
        y: i32,
        packed_table: Vec<u8>,
        packed_data: Vec<u8>,
        unpacked_data_len: u64,
    }
    let mut chunks_by_part: Vec<Vec<ChunkBlob>> = Vec::with_capacity(parts.len());

    for (part_idx, p) in parts.iter().enumerate() {
        let block_h = p.compression.scanlines_per_block();
        let cc = chunk_counts[part_idx];

        // Pre-compute per-row cumulative-EXCLUSIVE sample offsets so we
        // can slice each channel's samples for a given row range.
        let mut row_sample_starts: Vec<u64> = Vec::with_capacity(p.height as usize + 1);
        row_sample_starts.push(0);
        for r in 0..p.height as usize {
            let row_base = r * p.width as usize;
            let row_sum: u64 = p.samples_per_pixel[row_base..row_base + p.width as usize]
                .iter()
                .map(|&n| n as u64)
                .sum();
            let last = *row_sample_starts.last().unwrap();
            row_sample_starts.push(last + row_sum);
        }

        let mut part_chunks: Vec<ChunkBlob> = Vec::with_capacity(cc);
        for block_idx in 0..cc {
            let row0 = (block_idx as u32) * block_h;
            let rows_in_block = (p.height - row0).min(block_h) as usize;
            let entries = rows_in_block * p.width as usize;

            // Per-row cumulative-inclusive offset table.
            let mut table_bytes = Vec::with_capacity(entries * 4);
            for r in 0..rows_in_block {
                let dst_row = row0 as usize + r;
                let row_slice = &p.samples_per_pixel
                    [dst_row * p.width as usize..(dst_row + 1) * p.width as usize];
                let cumulative = cumulative_inclusive(row_slice);
                for c in cumulative {
                    table_bytes.extend_from_slice(&c.to_le_bytes());
                }
            }

            // Slice each channel's samples for this block — non-interleaved
            // (ch0 all samples, then ch1, ...).
            let block_sample_start = row_sample_starts[row0 as usize] as usize;
            let block_sample_end = row_sample_starts[row0 as usize + rows_in_block] as usize;
            let block_sample_count = block_sample_end - block_sample_start;
            let bpp_total: usize = p
                .channels
                .iter()
                .map(|c| c.pixel_type.bytes_per_sample())
                .sum();
            let mut sample_bytes: Vec<u8> = Vec::with_capacity(block_sample_count * bpp_total);
            for (ch_idx, ch) in p.channels.iter().enumerate() {
                let slc = &p.channel_samples[ch_idx][block_sample_start..block_sample_end];
                for &v in slc {
                    match ch.pixel_type {
                        PixelType::Half => sample_bytes
                            .extend_from_slice(&crate::half::f32_to_half(v).to_le_bytes()),
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

            let packed_table = compress_buffer(&table_bytes, p.compression)?;
            let packed_data = compress_buffer(&sample_bytes, p.compression)?;
            part_chunks.push(ChunkBlob {
                part_idx: part_idx as u32,
                y: row0 as i32,
                packed_table,
                packed_data,
                unpacked_data_len: sample_bytes.len() as u64,
            });
        }
        chunks_by_part.push(part_chunks);
    }

    // ---- Compute chunk offsets after offset tables, then emit. ----
    let header_bytes_so_far = out.len();
    let total_chunks: usize = chunk_counts.iter().sum();
    let offset_table_bytes = total_chunks * 8;
    let chunks_start = header_bytes_so_far + offset_table_bytes;

    // Walk chunks in part-order (part_0 chunks then part_1 chunks ...);
    // record absolute offsets so we can fill per-part offset tables.
    // Each deep multipart chunk on disk is:
    //   i32 part_number (4) | i32 Y (4) | u64 packed_table (8)
    //   | u64 packed_data (8) | u64 unpacked_data (8)
    //   | packed_table_bytes | packed_sample_bytes
    // → 4 + 4 + 24 = 32 bytes of header + the two byte blobs.
    let mut per_part_table: Vec<Vec<u64>> = vec![Vec::new(); parts.len()];
    let mut running = chunks_start;
    for part_chunks in &chunks_by_part {
        for c in part_chunks {
            per_part_table[c.part_idx as usize].push(running as u64);
            running += 32 + c.packed_table.len() + c.packed_data.len();
        }
    }

    // Emit concatenated offset tables (part 0, part 1, ...).
    for table in &per_part_table {
        for &o in table {
            out.extend_from_slice(&o.to_le_bytes());
        }
    }

    // Emit chunks in the same part-order we accounted for above.
    for part_chunks in &chunks_by_part {
        for c in part_chunks {
            out.extend_from_slice(&(c.part_idx as i32).to_le_bytes());
            out.extend_from_slice(&c.y.to_le_bytes());
            out.extend_from_slice(&(c.packed_table.len() as u64).to_le_bytes());
            out.extend_from_slice(&(c.packed_data.len() as u64).to_le_bytes());
            out.extend_from_slice(&c.unpacked_data_len.to_le_bytes());
            out.extend_from_slice(&c.packed_table);
            out.extend_from_slice(&c.packed_data);
        }
    }

    Ok(out)
}

/// Per-part attribute set for a deep scanline multipart part — strict
/// superset of the flat-multipart required attrs (adds
/// `maxSamplesPerPixel`, `type = "deepscanline"`, `version = 1`).
fn build_deep_part_attrs(
    part: &MultipartDeepScanlinePart,
    chunk_count: i32,
    max_samples: i32,
) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (part.width - 1) as i32,
        y_max: (part.height - 1) as i32,
    };
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
            name: "maxSamplesPerPixel".to_string(),
            value: AttributeValue::Other {
                type_name: "int".to_string(),
                data: max_samples.to_le_bytes().to_vec(),
            },
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
    ]
}

// ---------------------------------------------------------------------
// Single-part deep TILED WRITE + READ (round 130).
//
// Layout (single-part, version-field bit 0x800 set — `non_image` only):
//
//   magic(4) | version(4 with non_image bit set, NOT single_tile = 0x802)
//
// Empirical-spec note: single-part deep-tiled files DO NOT set the
// `single_tile` (0x200) bit. The `tiles[tiledesc]` attribute + the
// `type = "deeptile"` string-attribute are the discriminators; setting
// the `single_tile` bit alongside `non_image` causes `exrheader` to
// reject the file ("Unable to open"). Matches the OpenEXR File
// Layout convention for deep files: deep formats use the non_image bit
// alone for single-part, and add `multipart` (0x1000) for multi-part.
//   header attrs (channels, chunkCount[int], compression, dataWindow,
//     displayWindow, lineOrder, maxSamplesPerPixel[int],
//     pixelAspectRatio, screenWindowCenter, screenWindowWidth,
//     tiles[tiledesc], type[string="deeptile"], version[int=1])
//   NUL terminator
//   tile-offset table: chunkCount * u64 LE absolute byte offsets
//   tile chunks each on disk as:
//     i32 tile_x          (column index in the tile grid)
//     i32 tile_y          (row index in the tile grid)
//     i32 lvlx            (always 0 for ONE_LEVEL)
//     i32 lvly            (always 0 for ONE_LEVEL)
//     u64 packed_pixel_offset_table_size
//     u64 packed_sample_data_size
//     u64 unpacked_sample_data_size
//     packed_pixel_offset_table_bytes
//     packed_sample_data_bytes
//
// The per-tile pixel-offset table holds `tile_h * tile_w` cumulative
// i32 entries, one per column per row of the tile rectangle (matching
// the deep-scanline convention but rectangularly per-tile). After
// decompression the sample data is non-interleaved (all of channel 0's
// samples for this tile in pixel-scan order, then channel 1's, ...).
//
// Edge tiles store only the valid pixel rectangle (last row / column
// tiles may be smaller than `tileX × tileY`), matching the flat tiled
// encoder.
//
// ONE_LEVEL only. Multi-level deep tiled (MIPMAP/RIPMAP) is a followup.
// Compression NONE / RLE / ZIPS (matching the deep scanline encoder;
// deep ZIP rejected by the reference `exrinfo` validator).
// ---------------------------------------------------------------------

/// Input descriptor for [`encode_exr_deep_tiled`].
pub struct DeepTiledInput<'a> {
    pub width: u32,
    pub height: u32,
    /// Tile pixel dimensions. Both must be > 0; edge tiles store only
    /// the valid pixel rectangle (i.e. last row/column tiles may be
    /// smaller than `tile_x × tile_y`).
    pub tile_x: u32,
    pub tile_y: u32,
    /// Channels in alphabetical order (sub-sampled channels not
    /// supported on the deep path).
    pub channels: Vec<Channel>,
    /// One u32 per pixel (`width * height` long) — how many samples this
    /// pixel carries.
    pub samples_per_pixel: &'a [u32],
    /// One f32 slice per channel, each `samples_per_pixel.iter().sum()`
    /// long, in pixel-scan order. UINT stored as the u32 bits cast to
    /// f32 (matching the [`DeepExrImage`] convention).
    pub channel_samples: Vec<&'a [f32]>,
    pub compression: Compression,
}

/// Single-part deep-tiled EXR returned by [`parse_exr_deep_tiled`].
///
/// Pixel data is materialised into the same flat layout
/// [`DeepExrImage`] uses for scanline files: `samples_per_pixel` is
/// `width * height` long; each `channel_samples[ch]` is
/// `samples_per_pixel.iter().sum()` long in pixel-scan order. The
/// tile-grid structure is fully reassembled into row-major pixel
/// coordinates before return — callers don't have to know the file was
/// tiled.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepTiledImage {
    pub data_window: Box2i,
    pub display_window: Box2i,
    pub line_order: LineOrder,
    pub compression: Compression,
    /// Tile dimensions as recorded in the `tiles[tiledesc]` attribute.
    pub tile_x: u32,
    pub tile_y: u32,
    pub channels: Vec<Channel>,
    pub samples_per_pixel: Vec<u32>,
    pub channel_samples: Vec<Vec<f32>>,
    pub attributes: Vec<Attribute>,
}

impl DeepTiledImage {
    pub fn width(&self) -> u32 {
        self.data_window.width()
    }
    pub fn height(&self) -> u32 {
        self.data_window.height()
    }
    pub fn total_samples(&self) -> u64 {
        self.samples_per_pixel.iter().map(|&n| n as u64).sum()
    }
}

/// Encode a single-part `type="deeptile"` ONE_LEVEL deep-tiled EXR.
///
/// Self-roundtrips through [`parse_exr_deep_tiled`]; intended also to
/// be readable by the reference `exrheader` + `exrmetrics --convert`
/// flow.
pub fn encode_exr_deep_tiled(input: &DeepTiledInput) -> Result<Vec<u8>> {
    // ---- Validate input (mirrors deep scanline + flat tile rules). ----
    if !matches!(
        input.compression,
        Compression::None | Compression::Rle | Compression::Zips
    ) {
        return Err(ExrError::unsupported(format!(
            "deep tiled encode compression {:?} (reference encoder accepts \
             only NONE/RLE/ZIPS for deep — ZIP is listed in the spec page but \
             exrinfo rejects it with EXR_ERR_INVALID_ATTR)",
            input.compression
        )));
    }
    if input.width == 0 || input.height == 0 {
        return Err(ExrError::invalid(format!(
            "deep tiled dataWindow {}x{} must be > 0",
            input.width, input.height
        )));
    }
    if input.tile_x == 0 || input.tile_y == 0 {
        return Err(ExrError::invalid(format!(
            "deep tile size {}×{} must both be > 0",
            input.tile_x, input.tile_y
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
                "deep tiled encode + sub-sampled channel '{}'",
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
    for win in input.channels.windows(2) {
        if win[0].name >= win[1].name {
            return Err(ExrError::invalid(format!(
                "deep tiled channels not alphabetical: '{}' >= '{}'",
                win[0].name, win[1].name
            )));
        }
    }

    let tx_count = input.width.div_ceil(input.tile_x);
    let ty_count = input.height.div_ceil(input.tile_y);
    let chunk_count = (tx_count * ty_count) as usize;
    let max_samples = input.samples_per_pixel.iter().copied().max().unwrap_or(0) as i32;

    // ---- Header attributes. ----
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (input.width - 1) as i32,
        y_max: (input.height - 1) as i32,
    };
    let mut tiledesc = Vec::with_capacity(9);
    tiledesc.extend_from_slice(&input.tile_x.to_le_bytes());
    tiledesc.extend_from_slice(&input.tile_y.to_le_bytes());
    tiledesc.push(0x00); // ONE_LEVEL + ROUND_DOWN

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
                data: b"deeptile".to_vec(),
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

    // Single-part deep-tiled files use the non_image bit (0x800) ONLY —
    // the reference encoder rejects files that also set single_tile
    // (0x200) here. The `tiles` attribute + `type="deeptile"` string
    // attribute carry the tile-ness signal instead.
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

    // ---- Build per-tile payloads in ty-outer, tx-inner order. ----
    struct ChunkBlob {
        tx: u32,
        ty: u32,
        packed_table: Vec<u8>,
        packed_data: Vec<u8>,
        unpacked_data_len: u64,
    }
    let mut chunks: Vec<ChunkBlob> = Vec::with_capacity(chunk_count);

    let w = input.width as usize;
    let bpp_total: usize = input
        .channels
        .iter()
        .map(|c| c.pixel_type.bytes_per_sample())
        .sum();

    // Pre-compute cumulative-EXCLUSIVE per-pixel sample offsets so we can
    // slice each channel's samples by pixel index.
    let pixel_sample_starts: Vec<u64> = {
        let mut v = Vec::with_capacity(pixels + 1);
        v.push(0u64);
        let mut acc: u64 = 0;
        for &n in input.samples_per_pixel {
            acc += n as u64;
            v.push(acc);
        }
        v
    };

    for ty in 0..ty_count {
        for tx in 0..tx_count {
            let x0 = tx * input.tile_x;
            let y0 = ty * input.tile_y;
            let x1 = (x0 + input.tile_x).min(input.width);
            let y1 = (y0 + input.tile_y).min(input.height);
            let tw = (x1 - x0) as usize;
            let th = (y1 - y0) as usize;
            let entries = tw * th;

            // Per-tile pixel-offset table: `tw * th * 4` bytes of
            // cumulative-inclusive i32 entries (row-major within the
            // tile's valid pixel rectangle). The reference reader
            // unpacks the compressed table to exactly this size; for
            // NONE compression the reference happens to round up to
            // `tile_x * tile_y * 4` bytes on disk because its in-memory
            // buffer is full-tile-sized, but for ZIPS/RLE it reports
            // `unpacked = tw * th * 4`, which is the canonical encoded
            // size we always emit here.
            let mut table_bytes = Vec::with_capacity(entries * 4);
            let mut tile_spp: Vec<u32> = Vec::with_capacity(entries);
            for r in 0..th {
                let dst_y = y0 as usize + r;
                let mut row_acc: i32 = 0;
                for c in 0..tw {
                    let dst_x = x0 as usize + c;
                    let n = input.samples_per_pixel[dst_y * w + dst_x];
                    tile_spp.push(n);
                    row_acc = row_acc.checked_add(n as i32).ok_or_else(|| {
                        ExrError::invalid(format!(
                            "deep tile ({tx},{ty}) row {r}: cumulative offset overflows i32"
                        ))
                    })?;
                    table_bytes.extend_from_slice(&row_acc.to_le_bytes());
                }
            }

            // Assemble channel-major non-interleaved sample bytes for
            // this tile.
            let tile_total_samples: u64 = tile_spp.iter().map(|&n| n as u64).sum();
            let mut sample_bytes: Vec<u8> =
                Vec::with_capacity(tile_total_samples as usize * bpp_total);
            for (ch_idx, ch) in input.channels.iter().enumerate() {
                let plane = input.channel_samples[ch_idx];
                for r in 0..th {
                    let dst_y = y0 as usize + r;
                    for c in 0..tw {
                        let dst_x = x0 as usize + c;
                        let p = dst_y * w + dst_x;
                        let s_start = pixel_sample_starts[p] as usize;
                        let s_end = pixel_sample_starts[p + 1] as usize;
                        for &v in &plane[s_start..s_end] {
                            match ch.pixel_type {
                                PixelType::Half => sample_bytes
                                    .extend_from_slice(&crate::half::f32_to_half(v).to_le_bytes()),
                                PixelType::Float => {
                                    sample_bytes.extend_from_slice(&v.to_le_bytes())
                                }
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
                }
            }

            let packed_table = compress_buffer(&table_bytes, input.compression)?;
            let packed_data = compress_buffer(&sample_bytes, input.compression)?;
            chunks.push(ChunkBlob {
                tx,
                ty,
                packed_table,
                packed_data,
                unpacked_data_len: sample_bytes.len() as u64,
            });
        }
    }

    // ---- Compute absolute tile-chunk offsets. ----
    // Per-tile-chunk header on disk = 4 i32 coords + 3 u64 sizes = 40 B.
    let offset_table_bytes = chunk_count * 8;
    let chunks_start = header_bytes.len() + offset_table_bytes;
    let mut absolute_offsets: Vec<u64> = Vec::with_capacity(chunk_count);
    let mut running = chunks_start;
    for c in &chunks {
        absolute_offsets.push(running as u64);
        running += 40 + c.packed_table.len() + c.packed_data.len();
    }

    let mut out = Vec::with_capacity(running);
    out.extend_from_slice(&header_bytes);
    for &o in &absolute_offsets {
        out.extend_from_slice(&o.to_le_bytes());
    }
    for c in &chunks {
        out.extend_from_slice(&(c.tx as i32).to_le_bytes());
        out.extend_from_slice(&(c.ty as i32).to_le_bytes());
        out.extend_from_slice(&0i32.to_le_bytes()); // lvlx
        out.extend_from_slice(&0i32.to_le_bytes()); // lvly
        out.extend_from_slice(&(c.packed_table.len() as u64).to_le_bytes());
        out.extend_from_slice(&(c.packed_data.len() as u64).to_le_bytes());
        out.extend_from_slice(&c.unpacked_data_len.to_le_bytes());
        out.extend_from_slice(&c.packed_table);
        out.extend_from_slice(&c.packed_data);
    }
    Ok(out)
}

/// Parse a single-part `type="deeptile"` ONE_LEVEL deep-tiled EXR back
/// into a [`DeepTiledImage`].
///
/// Compression: NONE / RLE / ZIPS (matching the encoder; deep ZIP
/// rejected by the reference encoder). Sub-sampled channels are not
/// permitted in tiled files (per the EXR file format), so we reject any
/// channel with `xSampling != 1 || ySampling != 1`.
///
/// Multi-level tiled deep (MIPMAP / RIPMAP) is a followup; this round
/// only accepts ONE_LEVEL (`tiledesc.mode == 0x00`).
pub fn parse_exr_deep_tiled(bytes: &[u8]) -> Result<DeepTiledImage> {
    let header = parse_header_allow_deep(bytes)?;
    if header.version.multipart {
        return Err(ExrError::unsupported(
            "multi-part deep tiled EXR (use parse_exr_multipart_deep_tiled)".to_string(),
        ));
    }
    if !header.version.non_image {
        return Err(ExrError::invalid(
            "parse_exr_deep_tiled called on a flat (non-deep) file".to_string(),
        ));
    }
    // Single-part deep-tiled files do NOT set the single_tile (0x200) bit
    // — they discriminate via type="deeptile" + the tiles[tiledesc]
    // attribute. We accept either presence/absence of the bit defensively
    // (the format string attribute is the source of truth), but rely on
    // the type attribute below to confirm this is a deeptile file.
    let part_type = find_string_attr(&header.attributes, "type").ok_or_else(|| {
        ExrError::invalid("deep tiled file missing required 'type' attribute".to_string())
    })?;
    if part_type != "deeptile" {
        return Err(ExrError::unsupported(format!(
            "deep tiled file type='{part_type}' (only 'deeptile' supported here — \
             'deepscanline' routes through parse_exr_deep_scanline)"
        )));
    }
    let chunk_count = find_int_attr(&header.attributes, "chunkCount").ok_or_else(|| {
        ExrError::invalid("deep tiled file missing required 'chunkCount' attribute".to_string())
    })? as usize;
    let data_window = find_box2i(&header.attributes, "dataWindow").ok_or_else(|| {
        ExrError::invalid("deep tiled file missing required 'dataWindow' attribute".to_string())
    })?;
    let display_window = find_box2i(&header.attributes, "displayWindow").unwrap_or(data_window);
    let line_order =
        find_line_order(&header.attributes, "lineOrder").unwrap_or(LineOrder::IncreasingY);
    let compression = find_compression(&header.attributes, "compression").ok_or_else(|| {
        ExrError::invalid("deep tiled file missing required 'compression' attribute".to_string())
    })?;
    if !matches!(
        compression,
        Compression::None | Compression::Rle | Compression::Zips
    ) {
        return Err(ExrError::invalid(format!(
            "deep tiled file uses compression {compression:?} (reference encoder \
             accepts only NONE/RLE/ZIPS for deep)"
        )));
    }
    let channels = find_channels(&header.attributes).ok_or_else(|| {
        ExrError::invalid("deep tiled file missing required 'channels' attribute".to_string())
    })?;
    let mut sorted_channels = channels.clone();
    sorted_channels.sort_by(|a, b| a.name.cmp(&b.name));
    for ch in &sorted_channels {
        if ch.x_sampling != 1 || ch.y_sampling != 1 {
            return Err(ExrError::unsupported(format!(
                "deep tiled + sub-sampled channel '{}' (spec requires 1×1 in tiled files)",
                ch.name
            )));
        }
    }

    // tiles[tiledesc] payload: u32 xSize | u32 ySize | u8 mode.
    let tile_attr = header
        .attributes
        .iter()
        .find(|a| a.name == "tiles")
        .ok_or_else(|| {
            ExrError::invalid("deep tiled file missing required 'tiles' attribute".to_string())
        })?;
    let (tile_x, tile_y) = match &tile_attr.value {
        AttributeValue::Other { type_name, data } if type_name == "tiledesc" && data.len() == 9 => {
            let xs = u32::from_le_bytes(data[0..4].try_into().unwrap());
            let ys = u32::from_le_bytes(data[4..8].try_into().unwrap());
            let mode = data[8];
            // ONE_LEVEL = low nibble 0; MIPMAP_LEVELS = 1; ROUND_DOWN = high
            // nibble 0. Multi-level deep tiled (MIPMAP) is now handled by
            // [`parse_exr_deep_tiled_mipmap`] — point callers at it rather
            // than reject outright. RIPMAP deep tiled is still a followup.
            if (mode & 0x0F) == 0x01 {
                return Err(ExrError::unsupported(
                    "single-part MIPMAP_LEVELS deep tiled EXR \
                     (use parse_exr_deep_tiled_mipmap)"
                        .to_string(),
                ));
            }
            if mode != 0x00 {
                return Err(ExrError::unsupported(format!(
                    "deep tiled tiledesc mode=0x{mode:02x} (only 0x00 = ONE_LEVEL + \
                     ROUND_DOWN and 0x01 = MIPMAP_LEVELS + ROUND_DOWN currently \
                     supported — RIPMAP deep tiled is a followup)"
                )));
            }
            if xs == 0 || ys == 0 {
                return Err(ExrError::invalid(format!(
                    "deep tiled tiledesc tile size {xs}×{ys} must both be > 0"
                )));
            }
            (xs, ys)
        }
        _ => {
            return Err(ExrError::invalid(format!(
                "deep tiled tiles attribute has unexpected shape: {:?}",
                tile_attr.value
            )));
        }
    };

    let width = data_window.width();
    let height = data_window.height();
    if width == 0 || height == 0 {
        return Err(ExrError::invalid(format!(
            "deep tiled dataWindow {width}×{height} must be > 0"
        )));
    }
    let tx_count = width.div_ceil(tile_x);
    let ty_count = height.div_ceil(tile_y);
    let expected_chunks = (tx_count * ty_count) as usize;
    if chunk_count != expected_chunks {
        return Err(ExrError::invalid(format!(
            "deep tiled chunkCount={chunk_count} disagrees with tile-grid math ({expected_chunks})"
        )));
    }

    // Offset table.
    let mut pos = header.end_offset;
    if pos + chunk_count * 8 > bytes.len() {
        return Err(ExrError::invalid(
            "deep tiled offset table runs past EOF".to_string(),
        ));
    }
    let mut offsets = Vec::with_capacity(chunk_count);
    for _ in 0..chunk_count {
        let off = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
        offsets.push(off);
        pos += 8;
    }

    // Per-tile materialisation: first decode every tile into a sparse
    // per-pixel samples_per_pixel grid, then assemble flat per-channel
    // sample vectors in pixel-scan order from the per-tile sample slabs
    // (we cannot push directly into channel_samples in tile-arrival
    // order because the on-disk order is tile-major, not pixel-major).
    let pixels = (width as usize) * (height as usize);
    let mut samples_per_pixel: Vec<u32> = vec![0; pixels];
    // Per-tile decoded sample buffers, indexed by (tx + ty * tx_count).
    struct TileDecoded {
        // tx, ty, tw, th and one Vec<f32> per channel of length
        // tile_total_samples (channel-major within the tile).
        tx: u32,
        ty: u32,
        tw: u32,
        th: u32,
        channel_samples: Vec<Vec<f32>>,
    }
    let mut tile_decoded: Vec<Option<TileDecoded>> = (0..expected_chunks).map(|_| None).collect();

    let block_bpp: usize = sorted_channels
        .iter()
        .map(|c| c.pixel_type.bytes_per_sample())
        .sum();

    for (chunk_idx, &block_off) in offsets.iter().enumerate() {
        // Chunk header: 4 i32 + 3 u64 = 40 bytes.
        if block_off + 40 > bytes.len() {
            return Err(ExrError::invalid(format!(
                "deep tile chunk {chunk_idx} header past EOF"
            )));
        }
        let tx = i32::from_le_bytes(bytes[block_off..block_off + 4].try_into().unwrap());
        let ty = i32::from_le_bytes(bytes[block_off + 4..block_off + 8].try_into().unwrap());
        let lvlx = i32::from_le_bytes(bytes[block_off + 8..block_off + 12].try_into().unwrap());
        let lvly = i32::from_le_bytes(bytes[block_off + 12..block_off + 16].try_into().unwrap());
        if lvlx != 0 || lvly != 0 {
            return Err(ExrError::unsupported(format!(
                "deep tiled chunk {chunk_idx} has lvlx={lvlx} lvly={lvly} \
                 (only ONE_LEVEL supported — non-zero level is a followup)"
            )));
        }
        let packed_table =
            u64::from_le_bytes(bytes[block_off + 16..block_off + 24].try_into().unwrap()) as usize;
        let packed_data =
            u64::from_le_bytes(bytes[block_off + 24..block_off + 32].try_into().unwrap()) as usize;
        let unpacked_data =
            u64::from_le_bytes(bytes[block_off + 32..block_off + 40].try_into().unwrap()) as usize;

        if tx < 0 || ty < 0 || (tx as u32) >= tx_count || (ty as u32) >= ty_count {
            return Err(ExrError::invalid(format!(
                "deep tile chunk {chunk_idx}: tx={tx} ty={ty} outside grid {tx_count}×{ty_count}"
            )));
        }
        let tx_u = tx as u32;
        let ty_u = ty as u32;
        let x0 = tx_u * tile_x;
        let y0 = ty_u * tile_y;
        let x1 = (x0 + tile_x).min(width);
        let y1 = (y0 + tile_y).min(height);
        let tw = x1 - x0;
        let th = y1 - y0;
        let full_tw = tile_x as usize;
        let full_th = tile_y as usize;
        // The per-tile pixel-offset table is `tw * th * 4` bytes for
        // ZIPS/RLE compression (the canonical encoded size). The
        // reference encoder's NONE-compression path happens to
        // round up to `tile_x * tile_y * 4` bytes on disk because its
        // in-memory buffer is full-tile-sized; accept both sizes so we
        // can round-trip files produced by `exrmetrics --convert -z
        // none` as well as our own. Padding rows/columns repeat the
        // last valid cumulative so trimming to (tw, th) recovers the
        // valid pixel data either way.
        let entries = (tw * th) as usize;
        let full_entries = full_tw * full_th;
        let row_stride;
        let unpacked_table_size;
        if compression == Compression::None && packed_table == full_entries * 4 {
            unpacked_table_size = full_entries * 4;
            row_stride = full_tw;
        } else {
            unpacked_table_size = entries * 4;
            row_stride = tw as usize;
        }

        let table_start = block_off + 40;
        let table_end = table_start + packed_table;
        let data_start = table_end;
        let data_end = data_start + packed_data;
        if data_end > bytes.len() {
            return Err(ExrError::invalid(format!(
                "deep tile chunk {chunk_idx}: payload runs past EOF"
            )));
        }

        let table_bytes = decompress_buffer(
            &bytes[table_start..table_end],
            unpacked_table_size,
            compression,
        )?;
        let mut cumulative_flat: Vec<i32> = Vec::with_capacity(unpacked_table_size / 4);
        for ch in table_bytes.chunks_exact(4) {
            cumulative_flat.push(i32::from_le_bytes(ch.try_into().unwrap()));
        }
        if cumulative_flat.len() != unpacked_table_size / 4 {
            return Err(ExrError::invalid(format!(
                "deep tile chunk {chunk_idx}: offset-table size mismatch ({} != {})",
                cumulative_flat.len(),
                unpacked_table_size / 4
            )));
        }

        // Per-row of the tile, derive per-pixel sample counts. When the
        // table is padded (NONE compression + full-tile size), trim to
        // the first `tw` columns and first `th` rows.
        let mut tile_total_samples: u64 = 0;
        for r in 0..th as usize {
            let row_base = r * row_stride;
            let row_slice = &cumulative_flat[row_base..row_base + tw as usize];
            let per_pixel = per_pixel_from_cumulative(row_slice)?;
            let dst_y = y0 as usize + r;
            let dst_base = dst_y * width as usize + x0 as usize;
            for (i, &n) in per_pixel.iter().enumerate() {
                samples_per_pixel[dst_base + i] = n;
                tile_total_samples += n as u64;
            }
        }

        let expected_unpacked = tile_total_samples as usize * block_bpp;
        if expected_unpacked != unpacked_data {
            return Err(ExrError::invalid(format!(
                "deep tile chunk {chunk_idx}: derived unpacked_data={expected_unpacked} \
                 disagrees with header unpacked_data={unpacked_data}"
            )));
        }
        let sample_bytes =
            decompress_buffer(&bytes[data_start..data_end], unpacked_data, compression)?;

        // Decode each channel's slice of the tile into a Vec<f32> in
        // pixel-scan order within the tile (row-major). We store this
        // per-tile rather than scattering directly so we can re-emit
        // pixel-major channel samples in the second pass.
        let mut p = 0usize;
        let mut per_channel: Vec<Vec<f32>> =
            (0..sorted_channels.len()).map(|_| Vec::new()).collect();
        for (ch_idx, ch) in sorted_channels.iter().enumerate() {
            let bps = ch.pixel_type.bytes_per_sample();
            let need = tile_total_samples as usize * bps;
            if p + need > sample_bytes.len() {
                return Err(ExrError::invalid(format!(
                    "deep tile chunk {chunk_idx}: channel {} bytes past payload end",
                    ch.name
                )));
            }
            for s in 0..(tile_total_samples as usize) {
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
                per_channel[ch_idx].push(v);
            }
            p += need;
        }
        if p != sample_bytes.len() {
            return Err(ExrError::invalid(format!(
                "deep tile chunk {chunk_idx}: consumed {p} of {} payload bytes",
                sample_bytes.len()
            )));
        }

        let tile_grid_idx = (ty_u * tx_count + tx_u) as usize;
        if tile_decoded[tile_grid_idx].is_some() {
            return Err(ExrError::invalid(format!(
                "deep tile ({tx_u},{ty_u}) appears more than once in the offset table"
            )));
        }
        tile_decoded[tile_grid_idx] = Some(TileDecoded {
            tx: tx_u,
            ty: ty_u,
            tw,
            th,
            channel_samples: per_channel,
        });
    }

    // ---- Second pass: re-emit channel samples in pixel-scan (row-major)
    // order across the whole image. For each pixel, walk the tile it
    // belongs to and copy the per-tile channel slab's slice for that
    // pixel into the global per-channel vector. ----
    let total_samples: u64 = samples_per_pixel.iter().map(|&n| n as u64).sum();
    let mut channel_samples: Vec<Vec<f32>> = (0..sorted_channels.len())
        .map(|_| Vec::with_capacity(total_samples as usize))
        .collect();

    // For each tile, precompute per-pixel-within-tile cumulative starts
    // (so we can slice each channel's per-tile vector for any pixel in
    // the tile).
    struct TileStarts {
        pixel_starts: Vec<u64>, // length tw*th + 1, cumulative-exclusive
    }
    let mut tile_starts: Vec<TileStarts> = Vec::with_capacity(expected_chunks);
    for td_opt in &tile_decoded {
        let td = td_opt.as_ref().ok_or_else(|| {
            ExrError::invalid("deep tile grid missing one or more tiles".to_string())
        })?;
        let entries = (td.tw * td.th) as usize;
        let mut starts = Vec::with_capacity(entries + 1);
        starts.push(0u64);
        let x0 = td.tx * tile_x;
        let y0 = td.ty * tile_y;
        let mut acc: u64 = 0;
        for r in 0..td.th as usize {
            for c in 0..td.tw as usize {
                let dst_y = y0 as usize + r;
                let dst_x = x0 as usize + c;
                acc += samples_per_pixel[dst_y * width as usize + dst_x] as u64;
                starts.push(acc);
            }
        }
        tile_starts.push(TileStarts {
            pixel_starts: starts,
        });
    }

    for y in 0..height as usize {
        let ty = (y / tile_y as usize) as u32;
        let y_in_tile = y - (ty as usize) * tile_y as usize;
        for x in 0..width as usize {
            let tx = (x / tile_x as usize) as u32;
            let x_in_tile = x - (tx as usize) * tile_x as usize;
            let tile_grid_idx = (ty * tx_count + tx) as usize;
            let td = tile_decoded[tile_grid_idx].as_ref().unwrap();
            let pixel_within_tile = y_in_tile * td.tw as usize + x_in_tile;
            let s_start = tile_starts[tile_grid_idx].pixel_starts[pixel_within_tile] as usize;
            let s_end = tile_starts[tile_grid_idx].pixel_starts[pixel_within_tile + 1] as usize;
            for (ch_idx, dst) in channel_samples
                .iter_mut()
                .enumerate()
                .take(sorted_channels.len())
            {
                dst.extend_from_slice(&td.channel_samples[ch_idx][s_start..s_end]);
            }
        }
    }

    Ok(DeepTiledImage {
        data_window,
        display_window,
        line_order,
        compression,
        tile_x,
        tile_y,
        channels: sorted_channels,
        samples_per_pixel,
        channel_samples,
        attributes: header.attributes,
    })
}

// ---------------------------------------------------------------------
// Multi-part deep TILED WRITE + READ (round 181).
//
// Composition of the round-127 multi-part deep-scanline writer + the
// round-130 single-part deep-tiled chunk layout. Each part is a
// `type="deeptile"` ONE_LEVEL deep-tiled image; chunks are concatenated
// across all parts with an `i32 part_number` prefix (matching the
// `parse_exr_multipart` / `parse_exr_deep_multipart` linear-scan
// convention since `exrmultipart -combine`-built files emit zero-filled
// offset tables for parts beyond the first, so a robust reader cannot
// rely on the per-part offset tables).
//
// Per-chunk on-disk shape (per part):
//   i32 part_number
//   i32 tile_x         (column in tile grid)
//   i32 tile_y         (row in tile grid)
//   i32 lvlx           (always 0 — ONE_LEVEL only)
//   i32 lvly           (always 0 — ONE_LEVEL only)
//   u64 packed_pixel_offset_table_size
//   u64 packed_sample_data_size
//   u64 unpacked_sample_data_size
//   packed_pixel_offset_table_bytes
//   packed_sample_data_bytes
//
// → 4 + (4 + 4 + 4 + 4) + 24 = 44 bytes of header per tile chunk.
//
// Version field: `multipart | non_image = 0x1800` only. As with the
// single-part deep-tiled writer, the `single_tile` (0x200) bit is NOT
// set: the per-part `tiles[tiledesc]` attribute + `type="deeptile"`
// string attribute are the tile-ness discriminators.
//
// Per-tile pixel-offset table holds `tile_h * tile_w` cumulative-
// inclusive i32 entries, row-major within each tile's valid pixel
// rectangle (edge tiles trim to their valid extent). For NONE-
// compression the reader also accepts files that pad to the full
// `tile_x * tile_y * 4` bytes (matching the reference encoder's
// behaviour, mirrored from the single-part deep-tiled reader).
//
// Sample data is non-interleaved (channel-major within each tile).
// Compression NONE / RLE / ZIPS (deep ZIP rejected to match the
// reference `exrinfo` validator).
//
// MIPMAP/RIPMAP-level multi-part deep-tiled is a followup; this round
// only emits + accepts ONE_LEVEL (`tiledesc.mode == 0x00`).
// ---------------------------------------------------------------------

/// One part of a multi-part deep-tiled file, for input to
/// [`encode_exr_multipart_deep_tiled`].
///
/// `name` is mandatory and must be unique across all parts.
pub struct MultipartDeepTiledPart<'a> {
    /// Part name (must be unique across all parts in the file).
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Tile pixel dimensions. Both must be > 0; edge tiles store only
    /// their valid pixel rectangle.
    pub tile_x: u32,
    pub tile_y: u32,
    /// Channels in alphabetical order (sub-sampled channels not
    /// supported on the deep path).
    pub channels: Vec<Channel>,
    /// One u32 per pixel (`width * height` long) — how many samples
    /// this pixel carries.
    pub samples_per_pixel: &'a [u32],
    /// One f32 slice per channel, each `samples_per_pixel.iter().sum()`
    /// long, in pixel-scan order. UINT stored as the u32 bits cast to
    /// f32 (matching the [`DeepExrImage`] convention).
    pub channel_samples: Vec<&'a [f32]>,
    pub compression: Compression,
}

/// One part of a multi-part deep-tiled file, returned from
/// [`parse_exr_multipart_deep_tiled`].
///
/// Pixel data is materialised into the flat layout used by
/// [`DeepTiledImage`]: `samples_per_pixel` is `width * height` long;
/// each `channel_samples[ch]` is `samples_per_pixel.iter().sum()` long
/// in pixel-scan order. The tile-grid structure is fully reassembled
/// into row-major pixel coordinates before return — callers don't have
/// to know the part was tiled.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepTiledPart {
    pub name: String,
    pub data_window: Box2i,
    pub display_window: Box2i,
    pub line_order: LineOrder,
    pub compression: Compression,
    pub tile_x: u32,
    pub tile_y: u32,
    pub channels: Vec<Channel>,
    pub samples_per_pixel: Vec<u32>,
    pub channel_samples: Vec<Vec<f32>>,
    pub attributes: Vec<Attribute>,
}

impl DeepTiledPart {
    pub fn width(&self) -> u32 {
        self.data_window.width()
    }
    pub fn height(&self) -> u32 {
        self.data_window.height()
    }
    pub fn total_samples(&self) -> u64 {
        self.samples_per_pixel.iter().map(|&n| n as u64).sum()
    }
}

/// Encode a multi-part deep-tiled EXR (version-field bits 0x1800).
///
/// Each part is validated independently (alphabetical channel order,
/// `samples_per_pixel` length, per-channel sample-count totals,
/// compression in {NONE, RLE, ZIPS}, unique non-empty name, no
/// sub-sampling, tile dimensions > 0). Tile chunks are emitted ty-outer
/// tx-inner within each part (matching the single-part deep-tiled
/// writer); part chunks are concatenated in part-order.
///
/// Self-roundtrips through [`parse_exr_multipart_deep_tiled`].
pub fn encode_exr_multipart_deep_tiled(parts: &[MultipartDeepTiledPart]) -> Result<Vec<u8>> {
    if parts.is_empty() {
        return Err(ExrError::invalid(
            "encode_exr_multipart_deep_tiled: at least one part required".to_string(),
        ));
    }

    // ---- Validate every part up front (mirrors deep-tiled rules). ----
    for (i, p) in parts.iter().enumerate() {
        if p.name.is_empty() {
            return Err(ExrError::invalid(format!(
                "deep tiled part {i}: empty name"
            )));
        }
        for (j, other) in parts.iter().enumerate() {
            if j != i && other.name == p.name {
                return Err(ExrError::invalid(format!(
                    "duplicate deep tiled part name '{}' (parts {i} and {j})",
                    p.name
                )));
            }
        }
        if !matches!(
            p.compression,
            Compression::None | Compression::Rle | Compression::Zips
        ) {
            return Err(ExrError::unsupported(format!(
                "deep tiled part '{}' compression {:?} (reference encoder accepts \
                 only NONE/RLE/ZIPS for deep — ZIP is listed in the spec page but \
                 exrinfo rejects it with EXR_ERR_INVALID_ATTR)",
                p.name, p.compression
            )));
        }
        if p.width == 0 || p.height == 0 {
            return Err(ExrError::invalid(format!(
                "deep tiled part '{}': dataWindow {}x{} must be > 0",
                p.name, p.width, p.height
            )));
        }
        if p.tile_x == 0 || p.tile_y == 0 {
            return Err(ExrError::invalid(format!(
                "deep tiled part '{}': tile size {}×{} must both be > 0",
                p.name, p.tile_x, p.tile_y
            )));
        }
        let pixels = (p.width as usize) * (p.height as usize);
        if p.samples_per_pixel.len() != pixels {
            return Err(ExrError::invalid(format!(
                "deep tiled part '{}': samples_per_pixel len {} != width*height = {pixels}",
                p.name,
                p.samples_per_pixel.len()
            )));
        }
        if p.channels.len() != p.channel_samples.len() {
            return Err(ExrError::invalid(format!(
                "deep tiled part '{}': channels.len()={} != channel_samples.len()={}",
                p.name,
                p.channels.len(),
                p.channel_samples.len()
            )));
        }
        for win in p.channels.windows(2) {
            if win[0].name >= win[1].name {
                return Err(ExrError::invalid(format!(
                    "deep tiled part '{}': channels not alphabetical: '{}' >= '{}'",
                    p.name, win[0].name, win[1].name
                )));
            }
        }
        let total_samples: u64 = p.samples_per_pixel.iter().map(|&n| n as u64).sum();
        for (ch, slc) in p.channels.iter().zip(p.channel_samples.iter()) {
            if ch.x_sampling != 1 || ch.y_sampling != 1 {
                return Err(ExrError::unsupported(format!(
                    "deep tiled part '{}': sub-sampled channel '{}' (deep tiled path \
                     requires 1×1 sampling)",
                    p.name, ch.name
                )));
            }
            if slc.len() != total_samples as usize {
                return Err(ExrError::invalid(format!(
                    "deep tiled part '{}': channel '{}' sample slice len {} != \
                     total_samples {total_samples}",
                    p.name,
                    ch.name,
                    slc.len()
                )));
            }
        }
    }

    // ---- Per-part chunk counts + max-sample stats. ----
    let mut chunk_counts: Vec<usize> = Vec::with_capacity(parts.len());
    let mut tx_counts: Vec<u32> = Vec::with_capacity(parts.len());
    let mut ty_counts: Vec<u32> = Vec::with_capacity(parts.len());
    for p in parts {
        let tx_count = p.width.div_ceil(p.tile_x);
        let ty_count = p.height.div_ceil(p.tile_y);
        chunk_counts.push((tx_count * ty_count) as usize);
        tx_counts.push(tx_count);
        ty_counts.push(ty_count);
    }

    // ---- Per-part header byte blocks. ----
    let mut header_byte_blocks: Vec<Vec<u8>> = Vec::with_capacity(parts.len());
    for (i, p) in parts.iter().enumerate() {
        let max_samples = p.samples_per_pixel.iter().copied().max().unwrap_or(0) as i32;
        let attrs = build_deep_tiled_part_attrs(p, chunk_counts[i] as i32, max_samples);
        let mut hb = Vec::with_capacity(256);
        for a in &attrs {
            hb.extend_from_slice(a.name.as_bytes());
            hb.push(0);
            let (type_name, payload) = encode_attribute_value(&a.value);
            hb.extend_from_slice(type_name.as_bytes());
            hb.push(0);
            hb.extend_from_slice(&(payload.len() as i32).to_le_bytes());
            hb.extend_from_slice(&payload);
        }
        header_byte_blocks.push(hb);
    }

    // ---- Stitch magic + version + headers + double-NUL terminator. ----
    // Single-part deep-tiled used 0x800 only; multi-part adds 0x1000.
    // We deliberately do NOT set single_tile (0x200) — the per-part
    // `tiles[tiledesc]` attribute + `type="deeptile"` carry the
    // tile-ness signal (mirrors the single-part deep-tiled discipline).
    let version = VersionField::from_u32(2 | 0x800 | 0x1000);
    let mut out: Vec<u8> = Vec::with_capacity(2048);
    out.extend_from_slice(&EXR_MAGIC.to_le_bytes());
    out.extend_from_slice(&version.to_u32().to_le_bytes());
    for hb in &header_byte_blocks {
        out.extend_from_slice(hb);
        out.push(0); // per-part header terminator
    }
    out.push(0); // double-NUL = end-of-all-headers

    // ---- Build per-tile payloads. ----
    struct TileBlob {
        part_idx: u32,
        tx: u32,
        ty: u32,
        packed_table: Vec<u8>,
        packed_data: Vec<u8>,
        unpacked_data_len: u64,
    }
    let mut tiles_by_part: Vec<Vec<TileBlob>> = Vec::with_capacity(parts.len());

    for (part_idx, p) in parts.iter().enumerate() {
        let tx_count = tx_counts[part_idx];
        let ty_count = ty_counts[part_idx];
        let w = p.width as usize;
        let pixels = (p.width as usize) * (p.height as usize);

        let bpp_total: usize = p
            .channels
            .iter()
            .map(|c| c.pixel_type.bytes_per_sample())
            .sum();

        // Pre-compute cumulative-EXCLUSIVE per-pixel sample offsets so we
        // can slice each channel's samples by pixel index.
        let pixel_sample_starts: Vec<u64> = {
            let mut v = Vec::with_capacity(pixels + 1);
            v.push(0u64);
            let mut acc: u64 = 0;
            for &n in p.samples_per_pixel {
                acc += n as u64;
                v.push(acc);
            }
            v
        };

        let mut part_tiles: Vec<TileBlob> = Vec::with_capacity(chunk_counts[part_idx]);
        for ty in 0..ty_count {
            for tx in 0..tx_count {
                let x0 = tx * p.tile_x;
                let y0 = ty * p.tile_y;
                let x1 = (x0 + p.tile_x).min(p.width);
                let y1 = (y0 + p.tile_y).min(p.height);
                let tw = (x1 - x0) as usize;
                let th = (y1 - y0) as usize;
                let entries = tw * th;

                let mut table_bytes = Vec::with_capacity(entries * 4);
                let mut tile_spp: Vec<u32> = Vec::with_capacity(entries);
                for r in 0..th {
                    let dst_y = y0 as usize + r;
                    let mut row_acc: i32 = 0;
                    for c in 0..tw {
                        let dst_x = x0 as usize + c;
                        let n = p.samples_per_pixel[dst_y * w + dst_x];
                        tile_spp.push(n);
                        row_acc = row_acc.checked_add(n as i32).ok_or_else(|| {
                            ExrError::invalid(format!(
                                "deep tiled part '{}' tile ({tx},{ty}) row {r}: \
                                 cumulative offset overflows i32",
                                p.name
                            ))
                        })?;
                        table_bytes.extend_from_slice(&row_acc.to_le_bytes());
                    }
                }

                let tile_total_samples: u64 = tile_spp.iter().map(|&n| n as u64).sum();
                let mut sample_bytes: Vec<u8> =
                    Vec::with_capacity(tile_total_samples as usize * bpp_total);
                for (ch_idx, ch) in p.channels.iter().enumerate() {
                    let plane = p.channel_samples[ch_idx];
                    for r in 0..th {
                        let dst_y = y0 as usize + r;
                        for c in 0..tw {
                            let dst_x = x0 as usize + c;
                            let pi = dst_y * w + dst_x;
                            let s_start = pixel_sample_starts[pi] as usize;
                            let s_end = pixel_sample_starts[pi + 1] as usize;
                            for &v in &plane[s_start..s_end] {
                                match ch.pixel_type {
                                    PixelType::Half => sample_bytes.extend_from_slice(
                                        &crate::half::f32_to_half(v).to_le_bytes(),
                                    ),
                                    PixelType::Float => {
                                        sample_bytes.extend_from_slice(&v.to_le_bytes())
                                    }
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
                    }
                }

                let packed_table = compress_buffer(&table_bytes, p.compression)?;
                let packed_data = compress_buffer(&sample_bytes, p.compression)?;
                part_tiles.push(TileBlob {
                    part_idx: part_idx as u32,
                    tx,
                    ty,
                    packed_table,
                    packed_data,
                    unpacked_data_len: sample_bytes.len() as u64,
                });
            }
        }
        tiles_by_part.push(part_tiles);
    }

    // ---- Compute absolute offsets after concatenated offset tables. ----
    // Per-tile-chunk header on disk:
    //   i32 part_number + i32 tx + i32 ty + i32 lvlx + i32 lvly
    //   + u64 packed_table + u64 packed_data + u64 unpacked_data
    // = 4*5 + 24 = 44 bytes
    let header_bytes_so_far = out.len();
    let total_chunks: usize = chunk_counts.iter().sum();
    let offset_table_bytes = total_chunks * 8;
    let chunks_start = header_bytes_so_far + offset_table_bytes;

    let mut per_part_table: Vec<Vec<u64>> = vec![Vec::new(); parts.len()];
    let mut running = chunks_start;
    for part_tiles in &tiles_by_part {
        for c in part_tiles {
            per_part_table[c.part_idx as usize].push(running as u64);
            running += 44 + c.packed_table.len() + c.packed_data.len();
        }
    }

    // Emit concatenated offset tables (part 0, part 1, ...).
    for table in &per_part_table {
        for &o in table {
            out.extend_from_slice(&o.to_le_bytes());
        }
    }

    // Emit chunks in the same part-order.
    for part_tiles in &tiles_by_part {
        for c in part_tiles {
            out.extend_from_slice(&(c.part_idx as i32).to_le_bytes());
            out.extend_from_slice(&(c.tx as i32).to_le_bytes());
            out.extend_from_slice(&(c.ty as i32).to_le_bytes());
            out.extend_from_slice(&0i32.to_le_bytes()); // lvlx
            out.extend_from_slice(&0i32.to_le_bytes()); // lvly
            out.extend_from_slice(&(c.packed_table.len() as u64).to_le_bytes());
            out.extend_from_slice(&(c.packed_data.len() as u64).to_le_bytes());
            out.extend_from_slice(&c.unpacked_data_len.to_le_bytes());
            out.extend_from_slice(&c.packed_table);
            out.extend_from_slice(&c.packed_data);
        }
    }

    Ok(out)
}

/// Per-part attribute set for a deep tiled multipart part — strict
/// superset of the single-part deep-tiled required attrs (adds `name`,
/// matching the multi-part deep-scanline writer's `build_deep_part_attrs`).
fn build_deep_tiled_part_attrs(
    part: &MultipartDeepTiledPart,
    chunk_count: i32,
    max_samples: i32,
) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (part.width - 1) as i32,
        y_max: (part.height - 1) as i32,
    };
    let mut tiledesc = Vec::with_capacity(9);
    tiledesc.extend_from_slice(&part.tile_x.to_le_bytes());
    tiledesc.extend_from_slice(&part.tile_y.to_le_bytes());
    tiledesc.push(0x00); // ONE_LEVEL + ROUND_DOWN
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
            name: "maxSamplesPerPixel".to_string(),
            value: AttributeValue::Other {
                type_name: "int".to_string(),
                data: max_samples.to_le_bytes().to_vec(),
            },
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
                type_name: "string".to_string(),
                data: b"deeptile".to_vec(),
            },
        },
        Attribute {
            name: "version".to_string(),
            value: AttributeValue::Other {
                type_name: "int".to_string(),
                data: 1i32.to_le_bytes().to_vec(),
            },
        },
    ]
}

/// Parse a multi-part deep-tiled EXR (version-field bits 0x1800).
///
/// Every part must carry `type = "deeptile"` plus the standard per-part
/// required attributes (`name`, `chunkCount`, `dataWindow`,
/// `displayWindow`, `channels`, `compression`, `lineOrder`,
/// `pixelAspectRatio`, `screenWindowCenter`, `screenWindowWidth`,
/// `tiles[tiledesc]` ONE_LEVEL) plus the deep-specific `version=1` +
/// `maxSamplesPerPixel`.
///
/// On-disk layout mirrors the single-part deep-tiled writer per part,
/// with the chunk record prefixed by an `i32 part_number` (the
/// multi-part shape used elsewhere in this crate).
///
/// Compression: NONE / RLE / ZIPS only.
pub fn parse_exr_multipart_deep_tiled(bytes: &[u8]) -> Result<Vec<DeepTiledPart>> {
    let parts = parse_multipart_headers_allow_deep(bytes)?;
    if parts.is_empty() {
        return Err(ExrError::invalid(
            "multi-part deep tiled file has no parts".to_string(),
        ));
    }
    for (i, part) in parts.iter().enumerate() {
        let part_type = find_string_attr(&part.attributes, "type").ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part deep tiled: part {i} missing required 'type' attribute"
            ))
        })?;
        if part_type != "deeptile" {
            return Err(ExrError::unsupported(format!(
                "multi-part deep tiled: part {i} type='{part_type}' \
                 (only 'deeptile' supported — 'deepscanline' routes through \
                 parse_exr_deep_multipart)"
            )));
        }
    }
    if !parts[0].version.non_image {
        return Err(ExrError::invalid(
            "parse_exr_multipart_deep_tiled called on a multi-part file without the \
             non_image (deep) version bit set"
                .to_string(),
        ));
    }
    if !parts[0].version.multipart {
        return Err(ExrError::invalid(
            "parse_exr_multipart_deep_tiled called on a non-multipart EXR".to_string(),
        ));
    }

    /// Per-tile decoded channel samples (tile-extent `tw`, `th` + one
    /// `Vec<f32>` per channel, channel-major within the tile in
    /// pixel-scan order).
    struct TileDecoded {
        tw: u32,
        th: u32,
        channel_samples: Vec<Vec<f32>>,
    }
    struct PartState {
        name: String,
        data_window: Box2i,
        display_window: Box2i,
        line_order: LineOrder,
        compression: Compression,
        channels: Vec<Channel>,
        attributes: Vec<Attribute>,
        chunk_count: usize,
        tile_x: u32,
        tile_y: u32,
        width: u32,
        height: u32,
        tx_count: u32,
        ty_count: u32,
        samples_per_pixel: Vec<u32>,
        /// Indexed by `ty * tx_count + tx`.
        tile_decoded: Vec<Option<TileDecoded>>,
    }

    let mut state: Vec<PartState> = Vec::with_capacity(parts.len());
    let mut chunk_counts: Vec<usize> = Vec::with_capacity(parts.len());

    for (i, part) in parts.iter().enumerate() {
        let name = find_string_attr(&part.attributes, "name").ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part deep tiled part {i} missing required 'name' attribute"
            ))
        })?;
        let chunk_count = crate::decoder::find_chunk_count(&part.attributes).ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part deep tiled part {i} ('{name}') missing required 'chunkCount' attribute"
            ))
        })?;
        let data_window = find_box2i(&part.attributes, "dataWindow").ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part deep tiled part {i} ('{name}') missing required 'dataWindow' attribute"
            ))
        })?;
        let display_window = find_box2i(&part.attributes, "displayWindow").unwrap_or(data_window);
        let line_order =
            find_line_order(&part.attributes, "lineOrder").unwrap_or(LineOrder::IncreasingY);
        let compression = find_compression(&part.attributes, "compression").ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part deep tiled part {i} ('{name}') missing required 'compression' attribute"
            ))
        })?;
        if !matches!(
            compression,
            Compression::None | Compression::Rle | Compression::Zips
        ) {
            return Err(ExrError::invalid(format!(
                "multi-part deep tiled part {i} ('{name}') uses compression \
                 {compression:?} (reference encoder accepts only NONE/RLE/ZIPS for deep)"
            )));
        }
        let channels = find_channels(&part.attributes).ok_or_else(|| {
            ExrError::invalid(format!(
                "multi-part deep tiled part {i} ('{name}') missing required 'channels' attribute"
            ))
        })?;
        let mut sorted_channels = channels.clone();
        sorted_channels.sort_by(|a, b| a.name.cmp(&b.name));
        for ch in &sorted_channels {
            if ch.x_sampling != 1 || ch.y_sampling != 1 {
                return Err(ExrError::unsupported(format!(
                    "multi-part deep tiled part {i} ('{name}'): sub-sampled channel '{}' \
                     (tiled files require 1×1 sampling)",
                    ch.name
                )));
            }
        }
        // tiles[tiledesc] payload: u32 xSize | u32 ySize | u8 mode.
        let tile_attr = part
            .attributes
            .iter()
            .find(|a| a.name == "tiles")
            .ok_or_else(|| {
                ExrError::invalid(format!(
                    "multi-part deep tiled part {i} ('{name}') missing required 'tiles' attribute"
                ))
            })?;
        let (tile_x, tile_y) = match &tile_attr.value {
            AttributeValue::Other { type_name, data }
                if type_name == "tiledesc" && data.len() == 9 =>
            {
                let xs = u32::from_le_bytes(data[0..4].try_into().unwrap());
                let ys = u32::from_le_bytes(data[4..8].try_into().unwrap());
                let mode = data[8];
                if mode != 0x00 {
                    return Err(ExrError::unsupported(format!(
                        "multi-part deep tiled part {i} ('{name}'): tiledesc mode=0x{mode:02x} \
                         (only 0x00 = ONE_LEVEL + ROUND_DOWN supported — multi-level deep is a followup)"
                    )));
                }
                if xs == 0 || ys == 0 {
                    return Err(ExrError::invalid(format!(
                        "multi-part deep tiled part {i} ('{name}'): tile size {xs}×{ys} must both be > 0"
                    )));
                }
                (xs, ys)
            }
            _ => {
                return Err(ExrError::invalid(format!(
                    "multi-part deep tiled part {i} ('{name}'): tiles attribute has unexpected shape: {:?}",
                    tile_attr.value
                )));
            }
        };
        let width = data_window.width();
        let height = data_window.height();
        if width == 0 || height == 0 {
            return Err(ExrError::invalid(format!(
                "multi-part deep tiled part {i} ('{name}'): dataWindow {width}×{height} must be > 0"
            )));
        }
        let tx_count = width.div_ceil(tile_x);
        let ty_count = height.div_ceil(tile_y);
        let expected_chunks = (tx_count * ty_count) as usize;
        if chunk_count != expected_chunks {
            return Err(ExrError::invalid(format!(
                "multi-part deep tiled part {i} ('{name}'): chunkCount={chunk_count} \
                 disagrees with tile-grid math ({expected_chunks})"
            )));
        }
        chunk_counts.push(chunk_count);
        let pixels = (width as usize) * (height as usize);
        state.push(PartState {
            name,
            data_window,
            display_window,
            line_order,
            compression,
            channels: sorted_channels,
            attributes: part.attributes.clone(),
            chunk_count,
            tile_x,
            tile_y,
            width,
            height,
            tx_count,
            ty_count,
            samples_per_pixel: vec![0u32; pixels],
            tile_decoded: (0..expected_chunks).map(|_| None).collect(),
        });
    }

    // Skip past the concatenated offset tables (may be zero-filled).
    let total_chunks: usize = chunk_counts.iter().sum();
    let tables_start = parts.last().unwrap().end_offset;
    let chunk_scan_start = tables_start + total_chunks * 8;
    if chunk_scan_start > bytes.len() {
        return Err(ExrError::invalid(format!(
            "multi-part deep tiled offset tables run past EOF (need {chunk_scan_start}, have {})",
            bytes.len()
        )));
    }

    // Linear scan: each tile chunk has the part-number-prefixed layout
    // described in the encoder comment above.
    let mut scan_pos = chunk_scan_start;
    for _ in 0..total_chunks {
        // 5 i32 + 3 u64 = 44 bytes of chunk header.
        if scan_pos + 44 > bytes.len() {
            return Err(ExrError::invalid(format!(
                "multi-part deep tiled: unexpected EOF at chunk scan position {scan_pos}"
            )));
        }
        let part_num = i32::from_le_bytes(bytes[scan_pos..scan_pos + 4].try_into().unwrap());
        let tx = i32::from_le_bytes(bytes[scan_pos + 4..scan_pos + 8].try_into().unwrap());
        let ty = i32::from_le_bytes(bytes[scan_pos + 8..scan_pos + 12].try_into().unwrap());
        let lvlx = i32::from_le_bytes(bytes[scan_pos + 12..scan_pos + 16].try_into().unwrap());
        let lvly = i32::from_le_bytes(bytes[scan_pos + 16..scan_pos + 20].try_into().unwrap());
        let packed_table =
            u64::from_le_bytes(bytes[scan_pos + 20..scan_pos + 28].try_into().unwrap()) as usize;
        let packed_data =
            u64::from_le_bytes(bytes[scan_pos + 28..scan_pos + 36].try_into().unwrap()) as usize;
        let unpacked_data =
            u64::from_le_bytes(bytes[scan_pos + 36..scan_pos + 44].try_into().unwrap()) as usize;
        if part_num < 0 || part_num as usize >= state.len() {
            return Err(ExrError::invalid(format!(
                "multi-part deep tiled chunk at {scan_pos}: part_number={part_num} out of range 0..{}",
                state.len()
            )));
        }
        let part_idx = part_num as usize;
        let ps = &mut state[part_idx];
        if lvlx != 0 || lvly != 0 {
            return Err(ExrError::unsupported(format!(
                "multi-part deep tiled part {part_idx} ('{}'): chunk lvlx={lvlx} lvly={lvly} \
                 (only ONE_LEVEL supported in this round)",
                ps.name
            )));
        }
        if tx < 0 || ty < 0 || (tx as u32) >= ps.tx_count || (ty as u32) >= ps.ty_count {
            return Err(ExrError::invalid(format!(
                "multi-part deep tiled part {part_idx} ('{}'): tx={tx} ty={ty} outside grid {}×{}",
                ps.name, ps.tx_count, ps.ty_count
            )));
        }
        let tx_u = tx as u32;
        let ty_u = ty as u32;
        let x0 = tx_u * ps.tile_x;
        let y0 = ty_u * ps.tile_y;
        let x1 = (x0 + ps.tile_x).min(ps.width);
        let y1 = (y0 + ps.tile_y).min(ps.height);
        let tw = x1 - x0;
        let th = y1 - y0;
        let full_tw = ps.tile_x as usize;
        let full_th = ps.tile_y as usize;
        // Same NONE-padding accommodation as the single-part reader.
        let entries = (tw * th) as usize;
        let full_entries = full_tw * full_th;
        let row_stride;
        let unpacked_table_size;
        if ps.compression == Compression::None && packed_table == full_entries * 4 {
            unpacked_table_size = full_entries * 4;
            row_stride = full_tw;
        } else {
            unpacked_table_size = entries * 4;
            row_stride = tw as usize;
        }

        let table_start = scan_pos + 44;
        let table_end = table_start + packed_table;
        let data_start = table_end;
        let data_end = data_start + packed_data;
        if data_end > bytes.len() {
            return Err(ExrError::invalid(format!(
                "multi-part deep tiled part {part_idx} ('{}'): chunk payload runs past EOF",
                ps.name
            )));
        }

        let table_bytes = decompress_buffer(
            &bytes[table_start..table_end],
            unpacked_table_size,
            ps.compression,
        )?;
        let mut cumulative_flat: Vec<i32> = Vec::with_capacity(unpacked_table_size / 4);
        for ch in table_bytes.chunks_exact(4) {
            cumulative_flat.push(i32::from_le_bytes(ch.try_into().unwrap()));
        }
        if cumulative_flat.len() != unpacked_table_size / 4 {
            return Err(ExrError::invalid(format!(
                "multi-part deep tiled part {part_idx} ('{}'): offset-table size mismatch ({} != {})",
                ps.name,
                cumulative_flat.len(),
                unpacked_table_size / 4
            )));
        }

        let mut tile_total_samples: u64 = 0;
        let width = ps.width;
        for r in 0..th as usize {
            let row_base = r * row_stride;
            let row_slice = &cumulative_flat[row_base..row_base + tw as usize];
            let per_pixel = per_pixel_from_cumulative(row_slice)?;
            let dst_y = y0 as usize + r;
            let dst_base = dst_y * width as usize + x0 as usize;
            for (i, &n) in per_pixel.iter().enumerate() {
                ps.samples_per_pixel[dst_base + i] = n;
                tile_total_samples += n as u64;
            }
        }

        let block_bpp: usize = ps
            .channels
            .iter()
            .map(|c| c.pixel_type.bytes_per_sample())
            .sum();
        let expected_unpacked = tile_total_samples as usize * block_bpp;
        if expected_unpacked != unpacked_data {
            return Err(ExrError::invalid(format!(
                "multi-part deep tiled part {part_idx} ('{}'): derived unpacked_data={expected_unpacked} \
                 disagrees with header unpacked_data={unpacked_data}",
                ps.name
            )));
        }
        let sample_bytes =
            decompress_buffer(&bytes[data_start..data_end], unpacked_data, ps.compression)?;

        // Channel-major per-tile decode (matching single-part reader).
        let mut p = 0usize;
        let channel_types: Vec<(PixelType, String)> = ps
            .channels
            .iter()
            .map(|c| (c.pixel_type, c.name.clone()))
            .collect();
        let mut per_channel: Vec<Vec<f32>> = (0..channel_types.len()).map(|_| Vec::new()).collect();
        for (ch_idx, (pixel_type, ch_name)) in channel_types.iter().enumerate() {
            let bps = pixel_type.bytes_per_sample();
            let need = tile_total_samples as usize * bps;
            if p + need > sample_bytes.len() {
                return Err(ExrError::invalid(format!(
                    "multi-part deep tiled part {part_idx} ('{}'): channel '{ch_name}' bytes past payload end",
                    ps.name
                )));
            }
            for s in 0..(tile_total_samples as usize) {
                let off = p + s * bps;
                let v = match pixel_type {
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
                per_channel[ch_idx].push(v);
            }
            p += need;
        }
        if p != sample_bytes.len() {
            return Err(ExrError::invalid(format!(
                "multi-part deep tiled part {part_idx} ('{}'): consumed {p} of {} payload bytes",
                ps.name,
                sample_bytes.len()
            )));
        }

        let tile_grid_idx = (ty_u * ps.tx_count + tx_u) as usize;
        if ps.tile_decoded[tile_grid_idx].is_some() {
            return Err(ExrError::invalid(format!(
                "multi-part deep tiled part {part_idx} ('{}'): tile ({tx_u},{ty_u}) appears more than once",
                ps.name
            )));
        }
        ps.tile_decoded[tile_grid_idx] = Some(TileDecoded {
            tw,
            th,
            channel_samples: per_channel,
        });

        scan_pos = data_end;
        let _ = ps.chunk_count; // reserved for future bounds checks
    }

    // Second pass: per part, re-emit channel samples in pixel-scan order.
    let mut out_parts: Vec<DeepTiledPart> = Vec::with_capacity(state.len());
    for ps in state {
        let pixels = (ps.width as usize) * (ps.height as usize);
        for (idx, slot) in ps.tile_decoded.iter().enumerate() {
            if slot.is_none() {
                return Err(ExrError::invalid(format!(
                    "multi-part deep tiled part '{}': tile grid missing entry {idx}",
                    ps.name
                )));
            }
        }
        let total_samples: u64 = ps.samples_per_pixel.iter().map(|&n| n as u64).sum();
        let mut channel_samples: Vec<Vec<f32>> = (0..ps.channels.len())
            .map(|_| Vec::with_capacity(total_samples as usize))
            .collect();

        // Per-tile pixel-start tables, for slicing per-tile channel slabs.
        let mut tile_pixel_starts: Vec<Vec<u64>> = Vec::with_capacity(ps.tile_decoded.len());
        for slot in &ps.tile_decoded {
            let td = slot.as_ref().unwrap();
            let x0 = ((tile_pixel_starts.len() as u32) % ps.tx_count) * ps.tile_x;
            let y0 = ((tile_pixel_starts.len() as u32) / ps.tx_count) * ps.tile_y;
            let mut starts = Vec::with_capacity((td.tw * td.th) as usize + 1);
            starts.push(0u64);
            let mut acc: u64 = 0;
            for r in 0..td.th as usize {
                for c in 0..td.tw as usize {
                    let dst_y = y0 as usize + r;
                    let dst_x = x0 as usize + c;
                    acc += ps.samples_per_pixel[dst_y * ps.width as usize + dst_x] as u64;
                    starts.push(acc);
                }
            }
            tile_pixel_starts.push(starts);
        }

        for y in 0..ps.height as usize {
            let ty = (y / ps.tile_y as usize) as u32;
            let y_in_tile = y - (ty as usize) * ps.tile_y as usize;
            for x in 0..ps.width as usize {
                let tx = (x / ps.tile_x as usize) as u32;
                let x_in_tile = x - (tx as usize) * ps.tile_x as usize;
                let tile_grid_idx = (ty * ps.tx_count + tx) as usize;
                let td = ps.tile_decoded[tile_grid_idx].as_ref().unwrap();
                let pixel_within_tile = y_in_tile * td.tw as usize + x_in_tile;
                let s_start = tile_pixel_starts[tile_grid_idx][pixel_within_tile] as usize;
                let s_end = tile_pixel_starts[tile_grid_idx][pixel_within_tile + 1] as usize;
                for (ch_idx, dst) in channel_samples
                    .iter_mut()
                    .enumerate()
                    .take(ps.channels.len())
                {
                    dst.extend_from_slice(&td.channel_samples[ch_idx][s_start..s_end]);
                }
            }
        }
        let _ = pixels;

        out_parts.push(DeepTiledPart {
            name: ps.name,
            data_window: ps.data_window,
            display_window: ps.display_window,
            line_order: ps.line_order,
            compression: ps.compression,
            tile_x: ps.tile_x,
            tile_y: ps.tile_y,
            channels: ps.channels,
            samples_per_pixel: ps.samples_per_pixel,
            channel_samples,
            attributes: ps.attributes,
        });
    }
    Ok(out_parts)
}

// ---------------------------------------------------------------------
// Round 208: single-part deep tiled MIPMAP_LEVELS WRITE + READ.
//
// File layout (composes the round-130 single-part deep-tiled chunk shape
// with the round-78 single-part flat MIPMAP iteration order):
//
//   magic(4) | version(4 — non_image=0x800 only; single_tile NOT set)
//   header attributes (channels, chunkCount[int], compression, dataWindow,
//     displayWindow, lineOrder, maxSamplesPerPixel[int], pixelAspectRatio,
//     screenWindowCenter, screenWindowWidth,
//     tiles[tiledesc, mode=0x01 = MIPMAP_LEVELS + ROUND_DOWN],
//     type[string="deeptile"], version[int=1])
//   NUL terminator
//   tile offset table: chunkCount * u64 LE absolute byte offsets, where
//     chunkCount = sum over levels 0..N-1 of
//     ceil(level_w / tile_x) * ceil(level_h / tile_y).
//   tile chunks, each:
//     i32 tx | i32 ty | i32 lvlx | i32 lvly
//     u64 packed_pixel_offset_table_size
//     u64 packed_sample_data_size
//     u64 unpacked_sample_data_size
//     packed_pixel_offset_table_bytes
//     packed_sample_data_bytes
//
// Per the OpenEXR Technical Introduction (the diagonal-only MIPMAP
// iteration), the offset table walks levels `0..N-1` ascending and within
// each level emits tile chunks INCREASING_Y row-major (ty outer, tx
// inner); chunk header carries `lvlx == lvly == level`.
//
// Per-tile pixel-offset table holds `tile_h * tile_w` cumulative-inclusive
// i32 entries, row-major within each tile's valid pixel rectangle (edge
// tiles trim to their valid extent). For NONE compression the reader also
// accepts files that pad to the full `tile_x * tile_y * 4` bytes per the
// round-130 single-part deep-tiled discipline.
//
// Sample data is non-interleaved (channel-major within each tile).
// Compression NONE / RLE / ZIPS (deep ZIP rejected to match the reference
// `exrinfo` validator and the round-130 deep-tiled ONE_LEVEL writer).
//
// ROUND_DOWN only. RIPMAP-level deep tiled is a followup.
// ---------------------------------------------------------------------

/// One level of a deep-tiled mipmap pyramid: explicit width/height plus
/// per-pixel sample counts and per-channel samples for that level. Plane
/// `channel_samples[c]` is `samples_per_pixel.iter().sum()` long, in
/// pixel-scan (row-major) order. UINT channels store the u32 bits as f32
/// (matching the [`DeepExrImage`] convention).
pub struct DeepMipmapTiledLevelInput<'a> {
    pub width: u32,
    pub height: u32,
    /// `width * height` long.
    pub samples_per_pixel: &'a [u32],
    /// One f32 slice per channel.
    pub channel_samples: Vec<&'a [f32]>,
}

/// Input descriptor for [`encode_exr_deep_tiled_mipmap`].
pub struct DeepMipmapTiledInput<'a> {
    /// Tile pixel dimensions. Both must be > 0; edge tiles store only
    /// the valid pixel rectangle (i.e. last row/column tiles in a level
    /// may be smaller than `tile_x × tile_y`).
    pub tile_x: u32,
    pub tile_y: u32,
    /// Channels in alphabetical order (sub-sampled channels are not
    /// permitted in tiled files per the EXR spec).
    pub channels: Vec<Channel>,
    /// Pyramid: level 0 is the full-resolution data; level `l` has
    /// dimensions `mipmap_level_dim(level0_w, l, false) ×
    /// mipmap_level_dim(level0_h, l, false)`. Pyramid length must equal
    /// `mipmap_level_count(max(w, h), false)`.
    pub pyramid: Vec<DeepMipmapTiledLevelInput<'a>>,
    pub compression: Compression,
}

/// One decoded level of a deep-tiled MIPMAP pyramid.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepTiledMipmapLevel {
    pub width: u32,
    pub height: u32,
    /// `width * height` long.
    pub samples_per_pixel: Vec<u32>,
    /// One `Vec<f32>` per channel; total length per channel equals the
    /// sum of `samples_per_pixel`.
    pub channel_samples: Vec<Vec<f32>>,
}

/// Single-part deep-tiled MIPMAP_LEVELS EXR returned by
/// [`parse_exr_deep_tiled_mipmap`].
#[derive(Debug, Clone, PartialEq)]
pub struct DeepMipmapTiledImage {
    pub data_window: Box2i,
    pub display_window: Box2i,
    pub line_order: LineOrder,
    pub compression: Compression,
    /// Tile dimensions as recorded in the `tiles[tiledesc]` attribute.
    pub tile_x: u32,
    pub tile_y: u32,
    pub channels: Vec<Channel>,
    /// One entry per level (`0..N-1`).
    pub levels: Vec<DeepTiledMipmapLevel>,
    pub attributes: Vec<Attribute>,
}

impl DeepMipmapTiledImage {
    pub fn width(&self) -> u32 {
        self.data_window.width()
    }
    pub fn height(&self) -> u32 {
        self.data_window.height()
    }
    pub fn level_count(&self) -> usize {
        self.levels.len()
    }
}

/// Encode a single-part `type="deeptile"` MIPMAP_LEVELS deep-tiled EXR.
///
/// Self-roundtrips through [`parse_exr_deep_tiled_mipmap`].
pub fn encode_exr_deep_tiled_mipmap(input: &DeepMipmapTiledInput) -> Result<Vec<u8>> {
    // ---- Validate input. ----
    if !matches!(
        input.compression,
        Compression::None | Compression::Rle | Compression::Zips
    ) {
        return Err(ExrError::unsupported(format!(
            "deep mipmap tiled encode compression {:?} (reference encoder accepts \
             only NONE/RLE/ZIPS for deep — ZIP rejected by exrinfo with \
             EXR_ERR_INVALID_ATTR)",
            input.compression
        )));
    }
    if input.tile_x == 0 || input.tile_y == 0 {
        return Err(ExrError::invalid(format!(
            "deep mipmap tile size {}×{} must both be > 0",
            input.tile_x, input.tile_y
        )));
    }
    if input.pyramid.is_empty() {
        return Err(ExrError::invalid(
            "deep mipmap pyramid must have at least one level".to_string(),
        ));
    }
    for ch in &input.channels {
        if ch.x_sampling != 1 || ch.y_sampling != 1 {
            return Err(ExrError::unsupported(format!(
                "deep mipmap tiled encode + sub-sampled channel '{}' \
                 (spec requires 1×1 in tiled files)",
                ch.name
            )));
        }
    }
    for win in input.channels.windows(2) {
        if win[0].name >= win[1].name {
            return Err(ExrError::invalid(format!(
                "deep mipmap tiled channels not alphabetical: '{}' >= '{}'",
                win[0].name, win[1].name
            )));
        }
    }

    let width = input.pyramid[0].width;
    let height = input.pyramid[0].height;
    if width == 0 || height == 0 {
        return Err(ExrError::invalid(format!(
            "deep mipmap tiled level-0 {width}×{height} must be > 0"
        )));
    }
    let expected_levels = crate::decoder::mipmap_level_count(width.max(height), false);
    if input.pyramid.len() as u32 != expected_levels {
        return Err(ExrError::invalid(format!(
            "deep mipmap pyramid has {} levels, expected {expected_levels} for {width}×{height} ROUND_DOWN",
            input.pyramid.len()
        )));
    }
    for (l, lvl) in input.pyramid.iter().enumerate() {
        let want_w = crate::decoder::mipmap_level_dim(width, l as u32, false);
        let want_h = crate::decoder::mipmap_level_dim(height, l as u32, false);
        if lvl.width != want_w || lvl.height != want_h {
            return Err(ExrError::invalid(format!(
                "deep mipmap level {l} is {}×{} but spec requires {want_w}×{want_h} (ROUND_DOWN)",
                lvl.width, lvl.height
            )));
        }
        let need_pixels = (lvl.width as usize) * (lvl.height as usize);
        if lvl.samples_per_pixel.len() != need_pixels {
            return Err(ExrError::invalid(format!(
                "deep mipmap level {l} samples_per_pixel len {} != {}*{} = {need_pixels}",
                lvl.samples_per_pixel.len(),
                lvl.width,
                lvl.height
            )));
        }
        if lvl.channel_samples.len() != input.channels.len() {
            return Err(ExrError::invalid(format!(
                "deep mipmap level {l} has {} channel slices but {} channels declared",
                lvl.channel_samples.len(),
                input.channels.len()
            )));
        }
        let level_total: u64 = lvl.samples_per_pixel.iter().map(|&n| n as u64).sum();
        for (ch, slc) in input.channels.iter().zip(lvl.channel_samples.iter()) {
            if slc.len() != level_total as usize {
                return Err(ExrError::invalid(format!(
                    "deep mipmap level {l} channel '{}' sample slice len {} != level total {level_total}",
                    ch.name,
                    slc.len()
                )));
            }
        }
    }

    // chunkCount = sum of tile-grid size per level.
    let mut chunk_count: usize = 0;
    for lvl in &input.pyramid {
        let tx = lvl.width.div_ceil(input.tile_x);
        let ty = lvl.height.div_ceil(input.tile_y);
        chunk_count += (tx as usize) * (ty as usize);
    }

    // maxSamplesPerPixel across all levels.
    let max_samples = input
        .pyramid
        .iter()
        .flat_map(|lvl| lvl.samples_per_pixel.iter().copied())
        .max()
        .unwrap_or(0) as i32;

    // ---- Header attributes. ----
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (width - 1) as i32,
        y_max: (height - 1) as i32,
    };
    let mut tiledesc = Vec::with_capacity(9);
    tiledesc.extend_from_slice(&input.tile_x.to_le_bytes());
    tiledesc.extend_from_slice(&input.tile_y.to_le_bytes());
    tiledesc.push(0x01); // MIPMAP_LEVELS + ROUND_DOWN

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
                data: b"deeptile".to_vec(),
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

    // Single-part deep tiled files use the non_image (0x800) bit ONLY.
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
    header_bytes.push(0);

    // ---- Build per-tile chunk payloads in (level, ty, tx) order. ----
    struct ChunkBlob {
        tx: u32,
        ty: u32,
        lvl: u32,
        packed_table: Vec<u8>,
        packed_data: Vec<u8>,
        unpacked_data_len: u64,
    }
    let mut chunks: Vec<ChunkBlob> = Vec::with_capacity(chunk_count);
    let bpp_total: usize = input
        .channels
        .iter()
        .map(|c| c.pixel_type.bytes_per_sample())
        .sum();

    for (l, lvl) in input.pyramid.iter().enumerate() {
        let lvl_idx = l as u32;
        let lw = lvl.width as usize;
        let tx_count = lvl.width.div_ceil(input.tile_x);
        let ty_count = lvl.height.div_ceil(input.tile_y);
        // Pre-compute per-pixel sample starts so we can slice into
        // channel slices by pixel index.
        let pixel_sample_starts: Vec<u64> = {
            let mut v = Vec::with_capacity(lw * lvl.height as usize + 1);
            v.push(0u64);
            let mut acc: u64 = 0;
            for &n in lvl.samples_per_pixel {
                acc += n as u64;
                v.push(acc);
            }
            v
        };
        for ty in 0..ty_count {
            for tx in 0..tx_count {
                let x0 = tx * input.tile_x;
                let y0 = ty * input.tile_y;
                let x1 = (x0 + input.tile_x).min(lvl.width);
                let y1 = (y0 + input.tile_y).min(lvl.height);
                let tw = (x1 - x0) as usize;
                let th = (y1 - y0) as usize;
                let entries = tw * th;

                let mut table_bytes = Vec::with_capacity(entries * 4);
                let mut tile_spp: Vec<u32> = Vec::with_capacity(entries);
                for r in 0..th {
                    let dst_y = y0 as usize + r;
                    let mut row_acc: i32 = 0;
                    for c in 0..tw {
                        let dst_x = x0 as usize + c;
                        let n = lvl.samples_per_pixel[dst_y * lw + dst_x];
                        tile_spp.push(n);
                        row_acc = row_acc.checked_add(n as i32).ok_or_else(|| {
                            ExrError::invalid(format!(
                                "deep mipmap tile (lvl={l}, tx={tx}, ty={ty}) \
                                 row {r}: cumulative offset overflows i32"
                            ))
                        })?;
                        table_bytes.extend_from_slice(&row_acc.to_le_bytes());
                    }
                }

                let tile_total_samples: u64 = tile_spp.iter().map(|&n| n as u64).sum();
                let mut sample_bytes: Vec<u8> =
                    Vec::with_capacity(tile_total_samples as usize * bpp_total);
                for (ch_idx, ch) in input.channels.iter().enumerate() {
                    let plane = lvl.channel_samples[ch_idx];
                    for r in 0..th {
                        let dst_y = y0 as usize + r;
                        for c in 0..tw {
                            let dst_x = x0 as usize + c;
                            let p = dst_y * lw + dst_x;
                            let s_start = pixel_sample_starts[p] as usize;
                            let s_end = pixel_sample_starts[p + 1] as usize;
                            for &v in &plane[s_start..s_end] {
                                match ch.pixel_type {
                                    PixelType::Half => sample_bytes.extend_from_slice(
                                        &crate::half::f32_to_half(v).to_le_bytes(),
                                    ),
                                    PixelType::Float => {
                                        sample_bytes.extend_from_slice(&v.to_le_bytes())
                                    }
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
                    }
                }

                let packed_table = compress_buffer(&table_bytes, input.compression)?;
                let packed_data = compress_buffer(&sample_bytes, input.compression)?;
                chunks.push(ChunkBlob {
                    tx,
                    ty,
                    lvl: lvl_idx,
                    packed_table,
                    packed_data,
                    unpacked_data_len: sample_bytes.len() as u64,
                });
            }
        }
    }

    // ---- Compute absolute tile-chunk offsets. ----
    // Per-tile-chunk header on disk = 4 i32 coords + 3 u64 sizes = 40 B.
    let offset_table_bytes = chunk_count * 8;
    let chunks_start = header_bytes.len() + offset_table_bytes;
    let mut absolute_offsets: Vec<u64> = Vec::with_capacity(chunk_count);
    let mut running = chunks_start;
    for c in &chunks {
        absolute_offsets.push(running as u64);
        running += 40 + c.packed_table.len() + c.packed_data.len();
    }

    let mut out = Vec::with_capacity(running);
    out.extend_from_slice(&header_bytes);
    for &o in &absolute_offsets {
        out.extend_from_slice(&o.to_le_bytes());
    }
    for c in &chunks {
        out.extend_from_slice(&(c.tx as i32).to_le_bytes());
        out.extend_from_slice(&(c.ty as i32).to_le_bytes());
        out.extend_from_slice(&(c.lvl as i32).to_le_bytes()); // lvlx
        out.extend_from_slice(&(c.lvl as i32).to_le_bytes()); // lvly (== lvlx for MIPMAP diagonal)
        out.extend_from_slice(&(c.packed_table.len() as u64).to_le_bytes());
        out.extend_from_slice(&(c.packed_data.len() as u64).to_le_bytes());
        out.extend_from_slice(&c.unpacked_data_len.to_le_bytes());
        out.extend_from_slice(&c.packed_table);
        out.extend_from_slice(&c.packed_data);
    }
    Ok(out)
}

/// Parse a single-part `type="deeptile"` MIPMAP_LEVELS deep-tiled EXR
/// back into a [`DeepMipmapTiledImage`].
///
/// Compression: NONE / RLE / ZIPS. The reader uses a linear chunk scan
/// (matching the round-130 single-part deep-tiled ONE_LEVEL reader and
/// the round-78 flat MIPMAP convention) for robustness against
/// zero-filled offset tables.
pub fn parse_exr_deep_tiled_mipmap(bytes: &[u8]) -> Result<DeepMipmapTiledImage> {
    let header = parse_header_allow_deep(bytes)?;
    if header.version.multipart {
        return Err(ExrError::unsupported(
            "multi-part deep tiled MIPMAP EXR (use parse_exr_multipart_deep_tiled or \
             the future multi-part MIPMAP deep entry)"
                .to_string(),
        ));
    }
    if !header.version.non_image {
        return Err(ExrError::invalid(
            "parse_exr_deep_tiled_mipmap called on a flat (non-deep) file".to_string(),
        ));
    }
    let part_type = find_string_attr(&header.attributes, "type").ok_or_else(|| {
        ExrError::invalid("deep mipmap tiled file missing required 'type' attribute".to_string())
    })?;
    if part_type != "deeptile" {
        return Err(ExrError::unsupported(format!(
            "deep mipmap tiled file type='{part_type}' (only 'deeptile' supported)"
        )));
    }
    let chunk_count = find_int_attr(&header.attributes, "chunkCount").ok_or_else(|| {
        ExrError::invalid(
            "deep mipmap tiled file missing required 'chunkCount' attribute".to_string(),
        )
    })? as usize;
    let data_window = find_box2i(&header.attributes, "dataWindow").ok_or_else(|| {
        ExrError::invalid(
            "deep mipmap tiled file missing required 'dataWindow' attribute".to_string(),
        )
    })?;
    let display_window = find_box2i(&header.attributes, "displayWindow").unwrap_or(data_window);
    let line_order =
        find_line_order(&header.attributes, "lineOrder").unwrap_or(LineOrder::IncreasingY);
    let compression = find_compression(&header.attributes, "compression").ok_or_else(|| {
        ExrError::invalid(
            "deep mipmap tiled file missing required 'compression' attribute".to_string(),
        )
    })?;
    if !matches!(
        compression,
        Compression::None | Compression::Rle | Compression::Zips
    ) {
        return Err(ExrError::invalid(format!(
            "deep mipmap tiled file uses compression {compression:?} \
             (only NONE/RLE/ZIPS accepted for deep)"
        )));
    }
    let channels = find_channels(&header.attributes).ok_or_else(|| {
        ExrError::invalid(
            "deep mipmap tiled file missing required 'channels' attribute".to_string(),
        )
    })?;
    let mut sorted_channels = channels.clone();
    sorted_channels.sort_by(|a, b| a.name.cmp(&b.name));
    for ch in &sorted_channels {
        if ch.x_sampling != 1 || ch.y_sampling != 1 {
            return Err(ExrError::unsupported(format!(
                "deep mipmap tiled + sub-sampled channel '{}' \
                 (spec requires 1×1 in tiled files)",
                ch.name
            )));
        }
    }

    // tiles[tiledesc]: u32 xSize | u32 ySize | u8 mode. We accept only
    // mode 0x01 (MIPMAP_LEVELS + ROUND_DOWN).
    let tile_attr = header
        .attributes
        .iter()
        .find(|a| a.name == "tiles")
        .ok_or_else(|| {
            ExrError::invalid(
                "deep mipmap tiled file missing required 'tiles' attribute".to_string(),
            )
        })?;
    let (tile_x, tile_y) = match &tile_attr.value {
        AttributeValue::Other { type_name, data } if type_name == "tiledesc" && data.len() == 9 => {
            let xs = u32::from_le_bytes(data[0..4].try_into().unwrap());
            let ys = u32::from_le_bytes(data[4..8].try_into().unwrap());
            let mode = data[8];
            if mode != 0x01 {
                return Err(ExrError::unsupported(format!(
                    "deep mipmap tiled tiledesc mode=0x{mode:02x} \
                     (parse_exr_deep_tiled_mipmap requires mode=0x01 = \
                     MIPMAP_LEVELS + ROUND_DOWN; ONE_LEVEL routes through \
                     parse_exr_deep_tiled)"
                )));
            }
            if xs == 0 || ys == 0 {
                return Err(ExrError::invalid(format!(
                    "deep mipmap tiled tile size {xs}×{ys} must both be > 0"
                )));
            }
            (xs, ys)
        }
        _ => {
            return Err(ExrError::invalid(format!(
                "deep mipmap tiled tiles attribute has unexpected shape: {:?}",
                tile_attr.value
            )));
        }
    };

    let width = data_window.width();
    let height = data_window.height();
    if width == 0 || height == 0 {
        return Err(ExrError::invalid(format!(
            "deep mipmap tiled dataWindow {width}×{height} must be > 0"
        )));
    }

    // Compute the expected pyramid (sizes + chunk count) and pre-allocate
    // level data structures so we can scatter incoming tiles by (lvl, tx,
    // ty) regardless of arrival order.
    let n_levels = crate::decoder::mipmap_level_count(width.max(height), false);
    struct LevelMeta {
        width: u32,
        height: u32,
        tx_count: u32,
        ty_count: u32,
    }
    let mut metas: Vec<LevelMeta> = Vec::with_capacity(n_levels as usize);
    let mut expected_chunks: usize = 0;
    for l in 0..n_levels {
        let lw = crate::decoder::mipmap_level_dim(width, l, false);
        let lh = crate::decoder::mipmap_level_dim(height, l, false);
        let tx_count = lw.div_ceil(tile_x);
        let ty_count = lh.div_ceil(tile_y);
        expected_chunks += (tx_count as usize) * (ty_count as usize);
        metas.push(LevelMeta {
            width: lw,
            height: lh,
            tx_count,
            ty_count,
        });
    }
    if chunk_count != expected_chunks {
        return Err(ExrError::invalid(format!(
            "deep mipmap tiled chunkCount={chunk_count} disagrees with pyramid \
             total {expected_chunks} ({n_levels} levels, tile {tile_x}×{tile_y})"
        )));
    }

    // Offset table.
    let mut pos = header.end_offset;
    if pos + chunk_count * 8 > bytes.len() {
        return Err(ExrError::invalid(
            "deep mipmap tiled offset table runs past EOF".to_string(),
        ));
    }
    let mut offsets = Vec::with_capacity(chunk_count);
    for _ in 0..chunk_count {
        let off = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap()) as usize;
        offsets.push(off);
        pos += 8;
    }

    // For each level, pre-allocate `samples_per_pixel`; we scatter tiles
    // into it as we decode.
    let mut level_spp: Vec<Vec<u32>> = metas
        .iter()
        .map(|m| vec![0u32; (m.width as usize) * (m.height as usize)])
        .collect();

    // Per-level, per-tile decoded payloads; will be reassembled into
    // pixel-major order after the scan.
    struct TileDecoded {
        tx: u32,
        ty: u32,
        tw: u32,
        th: u32,
        channel_samples: Vec<Vec<f32>>,
    }
    let mut level_tiles: Vec<Vec<Option<TileDecoded>>> = metas
        .iter()
        .map(|m| {
            (0..(m.tx_count as usize) * (m.ty_count as usize))
                .map(|_| None)
                .collect()
        })
        .collect();

    let block_bpp: usize = sorted_channels
        .iter()
        .map(|c| c.pixel_type.bytes_per_sample())
        .sum();

    for (chunk_idx, &block_off) in offsets.iter().enumerate() {
        if block_off + 40 > bytes.len() {
            return Err(ExrError::invalid(format!(
                "deep mipmap tile chunk {chunk_idx} header past EOF"
            )));
        }
        let tx = i32::from_le_bytes(bytes[block_off..block_off + 4].try_into().unwrap());
        let ty = i32::from_le_bytes(bytes[block_off + 4..block_off + 8].try_into().unwrap());
        let lvlx = i32::from_le_bytes(bytes[block_off + 8..block_off + 12].try_into().unwrap());
        let lvly = i32::from_le_bytes(bytes[block_off + 12..block_off + 16].try_into().unwrap());
        if lvlx != lvly {
            return Err(ExrError::invalid(format!(
                "deep mipmap tile chunk {chunk_idx}: lvlx={lvlx} != lvly={lvly} \
                 (MIPMAP diagonal requires equal levels — RIPMAP would be a \
                 separate codepath)"
            )));
        }
        if lvlx < 0 || (lvlx as u32) >= n_levels {
            return Err(ExrError::invalid(format!(
                "deep mipmap tile chunk {chunk_idx}: lvl={lvlx} outside [0, {n_levels})"
            )));
        }
        let lvl = lvlx as usize;
        let meta = &metas[lvl];
        if tx < 0 || ty < 0 || (tx as u32) >= meta.tx_count || (ty as u32) >= meta.ty_count {
            return Err(ExrError::invalid(format!(
                "deep mipmap tile chunk {chunk_idx}: tx={tx} ty={ty} outside \
                 level {lvl}'s grid {}×{}",
                meta.tx_count, meta.ty_count
            )));
        }
        let tx_u = tx as u32;
        let ty_u = ty as u32;
        let packed_table =
            u64::from_le_bytes(bytes[block_off + 16..block_off + 24].try_into().unwrap()) as usize;
        let packed_data =
            u64::from_le_bytes(bytes[block_off + 24..block_off + 32].try_into().unwrap()) as usize;
        let unpacked_data =
            u64::from_le_bytes(bytes[block_off + 32..block_off + 40].try_into().unwrap()) as usize;

        let x0 = tx_u * tile_x;
        let y0 = ty_u * tile_y;
        let x1 = (x0 + tile_x).min(meta.width);
        let y1 = (y0 + tile_y).min(meta.height);
        let tw = x1 - x0;
        let th = y1 - y0;
        let full_tw = tile_x as usize;
        let full_th = tile_y as usize;
        let entries = (tw * th) as usize;
        let full_entries = full_tw * full_th;
        let row_stride;
        let unpacked_table_size;
        if compression == Compression::None && packed_table == full_entries * 4 {
            unpacked_table_size = full_entries * 4;
            row_stride = full_tw;
        } else {
            unpacked_table_size = entries * 4;
            row_stride = tw as usize;
        }

        let table_start = block_off + 40;
        let table_end = table_start + packed_table;
        let data_start = table_end;
        let data_end = data_start + packed_data;
        if data_end > bytes.len() {
            return Err(ExrError::invalid(format!(
                "deep mipmap tile chunk {chunk_idx}: payload runs past EOF"
            )));
        }

        let table_bytes = decompress_buffer(
            &bytes[table_start..table_end],
            unpacked_table_size,
            compression,
        )?;
        let mut cumulative_flat: Vec<i32> = Vec::with_capacity(unpacked_table_size / 4);
        for ch in table_bytes.chunks_exact(4) {
            cumulative_flat.push(i32::from_le_bytes(ch.try_into().unwrap()));
        }
        if cumulative_flat.len() != unpacked_table_size / 4 {
            return Err(ExrError::invalid(format!(
                "deep mipmap tile chunk {chunk_idx}: offset-table size mismatch \
                 ({} != {})",
                cumulative_flat.len(),
                unpacked_table_size / 4
            )));
        }

        let mut tile_total_samples: u64 = 0;
        let spp_buf = &mut level_spp[lvl];
        let lw = meta.width as usize;
        for r in 0..th as usize {
            let row_base = r * row_stride;
            let row_slice = &cumulative_flat[row_base..row_base + tw as usize];
            let per_pixel = per_pixel_from_cumulative(row_slice)?;
            let dst_y = y0 as usize + r;
            let dst_base = dst_y * lw + x0 as usize;
            for (i, &n) in per_pixel.iter().enumerate() {
                spp_buf[dst_base + i] = n;
                tile_total_samples += n as u64;
            }
        }

        let expected_unpacked = tile_total_samples as usize * block_bpp;
        if expected_unpacked != unpacked_data {
            return Err(ExrError::invalid(format!(
                "deep mipmap tile chunk {chunk_idx}: derived unpacked_data={expected_unpacked} \
                 disagrees with header unpacked_data={unpacked_data}"
            )));
        }
        let sample_bytes =
            decompress_buffer(&bytes[data_start..data_end], unpacked_data, compression)?;

        let mut p = 0usize;
        let mut per_channel: Vec<Vec<f32>> =
            (0..sorted_channels.len()).map(|_| Vec::new()).collect();
        for (ch_idx, ch) in sorted_channels.iter().enumerate() {
            let bps = ch.pixel_type.bytes_per_sample();
            let need = tile_total_samples as usize * bps;
            if p + need > sample_bytes.len() {
                return Err(ExrError::invalid(format!(
                    "deep mipmap tile chunk {chunk_idx}: channel {} bytes past payload end",
                    ch.name
                )));
            }
            for s in 0..(tile_total_samples as usize) {
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
                per_channel[ch_idx].push(v);
            }
            p += need;
        }
        if p != sample_bytes.len() {
            return Err(ExrError::invalid(format!(
                "deep mipmap tile chunk {chunk_idx}: consumed {p} of {} payload bytes",
                sample_bytes.len()
            )));
        }

        let grid_idx = (ty_u as usize) * (meta.tx_count as usize) + tx_u as usize;
        if level_tiles[lvl][grid_idx].is_some() {
            return Err(ExrError::invalid(format!(
                "deep mipmap tile (lvl={lvl}, tx={tx_u}, ty={ty_u}) \
                 appears more than once in the offset table"
            )));
        }
        level_tiles[lvl][grid_idx] = Some(TileDecoded {
            tx: tx_u,
            ty: ty_u,
            tw,
            th,
            channel_samples: per_channel,
        });
    }

    // ---- Reassemble each level's per-channel samples in pixel-scan order. ----
    let mut levels: Vec<DeepTiledMipmapLevel> = Vec::with_capacity(n_levels as usize);
    for (lvl, meta) in metas.iter().enumerate() {
        let spp = std::mem::take(&mut level_spp[lvl]);
        let total_samples: u64 = spp.iter().map(|&n| n as u64).sum();
        let mut channel_samples: Vec<Vec<f32>> = (0..sorted_channels.len())
            .map(|_| Vec::with_capacity(total_samples as usize))
            .collect();
        // Per-tile sample starts (cumulative within tile pixel order).
        struct TileStarts {
            pixel_starts: Vec<u64>,
        }
        let mut tile_starts: Vec<TileStarts> = Vec::with_capacity(level_tiles[lvl].len());
        for td_opt in &level_tiles[lvl] {
            let td = td_opt.as_ref().ok_or_else(|| {
                ExrError::invalid(format!(
                    "deep mipmap tiled level {lvl} grid is missing one or more tiles"
                ))
            })?;
            let entries = (td.tw * td.th) as usize;
            let mut starts = Vec::with_capacity(entries + 1);
            starts.push(0u64);
            let x0 = td.tx * tile_x;
            let y0 = td.ty * tile_y;
            let mut acc: u64 = 0;
            for r in 0..td.th as usize {
                for c in 0..td.tw as usize {
                    let dst_y = y0 as usize + r;
                    let dst_x = x0 as usize + c;
                    acc += spp[dst_y * meta.width as usize + dst_x] as u64;
                    starts.push(acc);
                }
            }
            tile_starts.push(TileStarts {
                pixel_starts: starts,
            });
        }

        for y in 0..meta.height as usize {
            let ty = (y / tile_y as usize) as u32;
            let y_in_tile = y - (ty as usize) * tile_y as usize;
            for x in 0..meta.width as usize {
                let tx = (x / tile_x as usize) as u32;
                let x_in_tile = x - (tx as usize) * tile_x as usize;
                let grid_idx = (ty as usize) * (meta.tx_count as usize) + tx as usize;
                let td = level_tiles[lvl][grid_idx].as_ref().unwrap();
                let pixel_within_tile = y_in_tile * td.tw as usize + x_in_tile;
                let s_start = tile_starts[grid_idx].pixel_starts[pixel_within_tile] as usize;
                let s_end = tile_starts[grid_idx].pixel_starts[pixel_within_tile + 1] as usize;
                for (ch_idx, dst) in channel_samples
                    .iter_mut()
                    .enumerate()
                    .take(sorted_channels.len())
                {
                    dst.extend_from_slice(&td.channel_samples[ch_idx][s_start..s_end]);
                }
            }
        }

        levels.push(DeepTiledMipmapLevel {
            width: meta.width,
            height: meta.height,
            samples_per_pixel: spp,
            channel_samples,
        });
    }

    Ok(DeepMipmapTiledImage {
        data_window,
        display_window,
        line_order,
        compression,
        tile_x,
        tile_y,
        channels: sorted_channels,
        levels,
        attributes: header.attributes,
    })
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
        // ZIP (16-line block) is rejected by the reference encoder
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

    // -----------------------------------------------------------------
    // Round-127 multi-part deep WRITE self-roundtrip tests.
    // -----------------------------------------------------------------

    #[test]
    fn deep_multipart_two_parts_roundtrip_zips_none() {
        let (spp_a, planes_a) = synthetic_deep(8, 4);
        let (spp_b, planes_b) = synthetic_deep(8, 4);
        let parts = vec![
            MultipartDeepScanlinePart {
                name: "partA".to_string(),
                width: 8,
                height: 4,
                channels: mk_channels_rgba_float(),
                samples_per_pixel: &spp_a,
                channel_samples: vec![&planes_a[0], &planes_a[1], &planes_a[2], &planes_a[3]],
                compression: Compression::Zips,
            },
            MultipartDeepScanlinePart {
                name: "partB".to_string(),
                width: 8,
                height: 4,
                channels: mk_channels_rgba_float(),
                samples_per_pixel: &spp_b,
                channel_samples: vec![&planes_b[0], &planes_b[1], &planes_b[2], &planes_b[3]],
                compression: Compression::None,
            },
        ];
        let bytes = encode_exr_multipart_deep_scanline(&parts).unwrap();
        let got = parse_exr_deep_multipart(&bytes).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].name, "partA");
        assert_eq!(got[0].compression, Compression::Zips);
        assert_eq!(got[0].samples_per_pixel, spp_a);
        for (c, w) in got[0].channel_samples.iter().zip(planes_a.iter()) {
            assert_eq!(c, w);
        }
        assert_eq!(got[1].name, "partB");
        assert_eq!(got[1].compression, Compression::None);
        assert_eq!(got[1].samples_per_pixel, spp_b);
        for (c, w) in got[1].channel_samples.iter().zip(planes_b.iter()) {
            assert_eq!(c, w);
        }
    }

    #[test]
    fn deep_multipart_three_parts_roundtrip_mixed_compression() {
        // ZIPS / NONE / RLE, all with FLOAT channels for bit-exact roundtrip.
        let (spp_a, planes_a) = synthetic_deep(6, 3);
        let (spp_b, planes_b) = synthetic_deep(6, 3);
        let (spp_c, planes_c) = synthetic_deep(6, 3);
        let parts = vec![
            MultipartDeepScanlinePart {
                name: "alpha".to_string(),
                width: 6,
                height: 3,
                channels: mk_channels_rgba_float(),
                samples_per_pixel: &spp_a,
                channel_samples: vec![&planes_a[0], &planes_a[1], &planes_a[2], &planes_a[3]],
                compression: Compression::Zips,
            },
            MultipartDeepScanlinePart {
                name: "beta".to_string(),
                width: 6,
                height: 3,
                channels: mk_channels_rgba_float(),
                samples_per_pixel: &spp_b,
                channel_samples: vec![&planes_b[0], &planes_b[1], &planes_b[2], &planes_b[3]],
                compression: Compression::None,
            },
            MultipartDeepScanlinePart {
                name: "gamma".to_string(),
                width: 6,
                height: 3,
                channels: mk_channels_rgba_float(),
                samples_per_pixel: &spp_c,
                channel_samples: vec![&planes_c[0], &planes_c[1], &planes_c[2], &planes_c[3]],
                compression: Compression::Rle,
            },
        ];
        let bytes = encode_exr_multipart_deep_scanline(&parts).unwrap();
        let got = parse_exr_deep_multipart(&bytes).unwrap();
        assert_eq!(got.len(), 3);
        let expected_names = ["alpha", "beta", "gamma"];
        let expected_compressions = [Compression::Zips, Compression::None, Compression::Rle];
        let expected_spp = [&spp_a, &spp_b, &spp_c];
        let expected_planes = [&planes_a, &planes_b, &planes_c];
        for (i, g) in got.iter().enumerate() {
            assert_eq!(g.name, expected_names[i]);
            assert_eq!(g.compression, expected_compressions[i]);
            assert_eq!(g.samples_per_pixel, **expected_spp[i]);
            for (gc, wc) in g.channel_samples.iter().zip(expected_planes[i].iter()) {
                assert_eq!(gc, wc, "{} channel mismatch", expected_names[i]);
            }
        }
    }

    #[test]
    fn deep_multipart_multi_chunk_zips_roundtrip() {
        // Height 12 with ZIPS (1-line blocks) → 12 chunks per part.
        let (spp_a, planes_a) = synthetic_deep(10, 12);
        let (spp_b, planes_b) = synthetic_deep(10, 12);
        let parts = vec![
            MultipartDeepScanlinePart {
                name: "foo".to_string(),
                width: 10,
                height: 12,
                channels: mk_channels_rgba_float(),
                samples_per_pixel: &spp_a,
                channel_samples: vec![&planes_a[0], &planes_a[1], &planes_a[2], &planes_a[3]],
                compression: Compression::Zips,
            },
            MultipartDeepScanlinePart {
                name: "bar".to_string(),
                width: 10,
                height: 12,
                channels: mk_channels_rgba_float(),
                samples_per_pixel: &spp_b,
                channel_samples: vec![&planes_b[0], &planes_b[1], &planes_b[2], &planes_b[3]],
                compression: Compression::Zips,
            },
        ];
        let bytes = encode_exr_multipart_deep_scanline(&parts).unwrap();
        let got = parse_exr_deep_multipart(&bytes).unwrap();
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].samples_per_pixel, spp_a);
        assert_eq!(got[1].samples_per_pixel, spp_b);
        for (g, want) in got[0].channel_samples.iter().zip(planes_a.iter()) {
            assert_eq!(g, want);
        }
        for (g, want) in got[1].channel_samples.iter().zip(planes_b.iter()) {
            assert_eq!(g, want);
        }
    }

    #[test]
    fn deep_multipart_rejects_empty_parts() {
        let r = encode_exr_multipart_deep_scanline(&[]);
        assert!(r.is_err(), "must reject zero-part input");
    }

    #[test]
    fn deep_multipart_rejects_duplicate_names() {
        let (spp, planes) = synthetic_deep(4, 2);
        let parts = vec![
            MultipartDeepScanlinePart {
                name: "dup".to_string(),
                width: 4,
                height: 2,
                channels: mk_channels_rgba_float(),
                samples_per_pixel: &spp,
                channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
                compression: Compression::None,
            },
            MultipartDeepScanlinePart {
                name: "dup".to_string(),
                width: 4,
                height: 2,
                channels: mk_channels_rgba_float(),
                samples_per_pixel: &spp,
                channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
                compression: Compression::None,
            },
        ];
        let r = encode_exr_multipart_deep_scanline(&parts);
        assert!(r.is_err(), "must reject duplicate part names");
    }

    #[test]
    fn deep_multipart_rejects_zip_compression() {
        let (spp, planes) = synthetic_deep(4, 2);
        let parts = vec![MultipartDeepScanlinePart {
            name: "x".to_string(),
            width: 4,
            height: 2,
            channels: mk_channels_rgba_float(),
            samples_per_pixel: &spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::Zip,
        }];
        let r = encode_exr_multipart_deep_scanline(&parts);
        assert!(
            r.is_err(),
            "must reject ZIP compression on multipart deep (exrinfo rejects it)"
        );
    }

    #[test]
    fn deep_multipart_all_zero_samples() {
        let spp = vec![0u32; 8];
        let empty: Vec<f32> = Vec::new();
        let parts = vec![MultipartDeepScanlinePart {
            name: "z".to_string(),
            width: 4,
            height: 2,
            channels: mk_channels_rgba_float(),
            samples_per_pixel: &spp,
            channel_samples: vec![&empty, &empty, &empty, &empty],
            compression: Compression::Zips,
        }];
        let bytes = encode_exr_multipart_deep_scanline(&parts).unwrap();
        let got = parse_exr_deep_multipart(&bytes).unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].samples_per_pixel, spp);
        for c in &got[0].channel_samples {
            assert!(c.is_empty());
        }
    }

    // -----------------------------------------------------------------
    // Round 130: deep tiled (single-part, ONE_LEVEL) WRITE + READ.
    // -----------------------------------------------------------------

    #[test]
    fn deep_tiled_roundtrip_none_8x8_in_16x16() {
        let (spp, planes) = synthetic_deep(16, 16);
        let input = DeepTiledInput {
            width: 16,
            height: 16,
            tile_x: 8,
            tile_y: 8,
            channels: mk_channels_rgba_float(),
            samples_per_pixel: &spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::None,
        };
        let bytes = encode_exr_deep_tiled(&input).unwrap();
        let img = parse_exr_deep_tiled(&bytes).unwrap();
        assert_eq!(img.width(), 16);
        assert_eq!(img.height(), 16);
        assert_eq!(img.tile_x, 8);
        assert_eq!(img.tile_y, 8);
        assert_eq!(img.samples_per_pixel, spp);
        for (got, want) in img.channel_samples.iter().zip(planes.iter()) {
            assert_eq!(got, want);
        }
    }

    #[test]
    fn deep_tiled_roundtrip_zips_8x4_in_24x12() {
        // 24×12 with 8×4 tiles → 3 cols × 3 rows = 9 chunks; no edge tiles.
        let (spp, planes) = synthetic_deep(24, 12);
        let input = DeepTiledInput {
            width: 24,
            height: 12,
            tile_x: 8,
            tile_y: 4,
            channels: mk_channels_rgba_float(),
            samples_per_pixel: &spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::Zips,
        };
        let bytes = encode_exr_deep_tiled(&input).unwrap();
        let img = parse_exr_deep_tiled(&bytes).unwrap();
        assert_eq!(img.samples_per_pixel, spp);
        for (got, want) in img.channel_samples.iter().zip(planes.iter()) {
            assert_eq!(got, want);
        }
    }

    #[test]
    fn deep_tiled_roundtrip_rle_edge_tiles_13x9_in_4x4() {
        // 13×9 with 4×4 tiles → 4 cols × 3 rows = 12 chunks; right column
        // (tx=3) holds 1px-wide tiles, bottom row (ty=2) holds 1px-tall
        // tiles — exercises the edge-tile clipping in both axes.
        let (spp, planes) = synthetic_deep(13, 9);
        let input = DeepTiledInput {
            width: 13,
            height: 9,
            tile_x: 4,
            tile_y: 4,
            channels: mk_channels_rgba_float(),
            samples_per_pixel: &spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::Rle,
        };
        let bytes = encode_exr_deep_tiled(&input).unwrap();
        let img = parse_exr_deep_tiled(&bytes).unwrap();
        assert_eq!(img.width(), 13);
        assert_eq!(img.height(), 9);
        assert_eq!(img.samples_per_pixel, spp);
        for (got, want) in img.channel_samples.iter().zip(planes.iter()) {
            assert_eq!(got, want);
        }
    }

    #[test]
    fn deep_tiled_rejects_zip_compression() {
        // ZIP rejected by the reference exrinfo for deep data — mirror.
        let (spp, planes) = synthetic_deep(4, 4);
        let input = DeepTiledInput {
            width: 4,
            height: 4,
            tile_x: 4,
            tile_y: 4,
            channels: mk_channels_rgba_float(),
            samples_per_pixel: &spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::Zip,
        };
        let r = encode_exr_deep_tiled(&input);
        assert!(r.is_err(), "deep tiled encoder must reject ZIP compression");
    }

    #[test]
    fn deep_tiled_rejects_zero_tile_size() {
        let spp = vec![0u32; 16];
        let empty: Vec<f32> = Vec::new();
        let input = DeepTiledInput {
            width: 4,
            height: 4,
            tile_x: 0,
            tile_y: 4,
            channels: mk_channels_rgba_float(),
            samples_per_pixel: &spp,
            channel_samples: vec![&empty, &empty, &empty, &empty],
            compression: Compression::None,
        };
        assert!(encode_exr_deep_tiled(&input).is_err());
    }

    #[test]
    fn deep_tiled_rejects_subsampled_channels() {
        // Tiled deep files must be 1×1 sampled per the spec — match the
        // restriction the flat tiled encoder already enforces.
        let chs = vec![Channel {
            name: "Y".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 2,
            y_sampling: 2,
        }];
        let spp = vec![0u32; 16];
        let empty: Vec<f32> = Vec::new();
        let input = DeepTiledInput {
            width: 4,
            height: 4,
            tile_x: 4,
            tile_y: 4,
            channels: chs,
            samples_per_pixel: &spp,
            channel_samples: vec![&empty],
            compression: Compression::None,
        };
        assert!(encode_exr_deep_tiled(&input).is_err());
    }

    #[test]
    fn deep_tiled_all_zero_samples_8x8_in_4x4() {
        // Degenerate but spec-legal: every pixel carries 0 samples.
        let w = 8u32;
        let h = 8u32;
        let spp = vec![0u32; (w * h) as usize];
        let empty: Vec<f32> = Vec::new();
        let input = DeepTiledInput {
            width: w,
            height: h,
            tile_x: 4,
            tile_y: 4,
            channels: mk_channels_rgba_float(),
            samples_per_pixel: &spp,
            channel_samples: vec![&empty, &empty, &empty, &empty],
            compression: Compression::Zips,
        };
        let bytes = encode_exr_deep_tiled(&input).unwrap();
        let img = parse_exr_deep_tiled(&bytes).unwrap();
        assert_eq!(img.samples_per_pixel, spp);
        for c in &img.channel_samples {
            assert!(c.is_empty());
        }
        // Header sanity: version-field carries non_image (0x800) ONLY —
        // single-part deep-tiled files must NOT set single_tile (0x200);
        // the reference encoder `exrheader` rejects files with both
        // bits set. The tile-ness signal lives in the `tiles[tiledesc]`
        // attribute + the `type="deeptile"` string attribute.
        let v = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(v & 0x800, 0x800, "non_image bit must be set");
        assert_eq!(v & 0x200, 0, "single_tile bit must NOT be set");
        assert_eq!(v & 0x1000, 0, "multipart bit must NOT be set");
    }

    #[test]
    fn deep_tiled_parse_rejects_flat_file() {
        // Hand-build a flat (non-deep) tiled-looking header and ensure
        // the deep tiled parser refuses (non_image bit not set).
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&EXR_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&(2u32 | 0x200).to_le_bytes()); // single_tile only
        bytes.push(0); // empty header terminator
        let r = parse_exr_deep_tiled(&bytes);
        assert!(r.is_err());
    }

    #[test]
    fn deep_tiled_parse_rejects_scanline_deep() {
        // Deep scanline (non_image bit set, single_tile NOT set) must be
        // rejected by parse_exr_deep_tiled — it's the wrong parser.
        let (spp, planes) = synthetic_deep(8, 4);
        let input = DeepScanlineInput {
            width: 8,
            height: 4,
            channels: mk_channels_rgba_float(),
            samples_per_pixel: &spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::None,
        };
        let bytes = encode_exr_deep_scanline(&input).unwrap();
        let r = parse_exr_deep_tiled(&bytes);
        assert!(r.is_err());
    }

    // ----- Round 181: multi-part deep TILED WRITE + READ ------------------

    fn assert_mp_deep_tiled_part_roundtrip(
        part: &DeepTiledPart,
        spp: &[u32],
        planes: &[Vec<f32>; 4],
    ) {
        assert_eq!(part.samples_per_pixel, spp);
        for (ch_idx, plane) in planes.iter().enumerate() {
            let got = &part.channel_samples[ch_idx];
            assert_eq!(
                got.len(),
                plane.len(),
                "channel {ch_idx} sample-count mismatch"
            );
            for (i, (g, e)) in got.iter().zip(plane.iter()).enumerate() {
                assert_eq!(
                    g, e,
                    "channel {ch_idx} sample {i} mismatch: got {g} expected {e}"
                );
            }
        }
    }

    #[test]
    fn mp_deep_tiled_two_parts_zips_roundtrip() {
        let (w, h, tx, ty) = (8u32, 6u32, 4u32, 3u32);
        let (spp0, planes0) = synthetic_deep(w, h);
        let (spp1, planes1) = synthetic_deep(w, h);
        let channels = mk_channels_rgba_float();
        let parts = vec![
            MultipartDeepTiledPart {
                name: "left".to_string(),
                width: w,
                height: h,
                tile_x: tx,
                tile_y: ty,
                channels: channels.clone(),
                samples_per_pixel: &spp0,
                channel_samples: vec![&planes0[0], &planes0[1], &planes0[2], &planes0[3]],
                compression: Compression::Zips,
            },
            MultipartDeepTiledPart {
                name: "right".to_string(),
                width: w,
                height: h,
                tile_x: tx,
                tile_y: ty,
                channels: channels.clone(),
                samples_per_pixel: &spp1,
                channel_samples: vec![&planes1[0], &planes1[1], &planes1[2], &planes1[3]],
                compression: Compression::Zips,
            },
        ];
        let bytes = encode_exr_multipart_deep_tiled(&parts).unwrap();

        // Version field carries multipart (0x1000) + non_image (0x800).
        // Must NOT carry single_tile (0x200), matching the single-part
        // deep-tiled discipline.
        let v = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(v & 0x1000, 0x1000, "multipart bit must be set");
        assert_eq!(v & 0x800, 0x800, "non_image bit must be set");
        assert_eq!(v & 0x200, 0, "single_tile bit must NOT be set");

        let read = parse_exr_multipart_deep_tiled(&bytes).unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].name, "left");
        assert_eq!(read[1].name, "right");
        assert_eq!(read[0].tile_x, tx);
        assert_eq!(read[0].tile_y, ty);
        assert_mp_deep_tiled_part_roundtrip(&read[0], &spp0, &planes0);
        assert_mp_deep_tiled_part_roundtrip(&read[1], &spp1, &planes1);
    }

    #[test]
    fn mp_deep_tiled_three_parts_mixed_compression_roundtrip() {
        let (w, h, tx, ty) = (12u32, 8u32, 4u32, 4u32);
        let (spp0, planes0) = synthetic_deep(w, h);
        let (spp1, planes1) = synthetic_deep(w, h);
        let (spp2, planes2) = synthetic_deep(w, h);
        let channels = mk_channels_rgba_float();
        let parts = vec![
            MultipartDeepTiledPart {
                name: "alpha".to_string(),
                width: w,
                height: h,
                tile_x: tx,
                tile_y: ty,
                channels: channels.clone(),
                samples_per_pixel: &spp0,
                channel_samples: vec![&planes0[0], &planes0[1], &planes0[2], &planes0[3]],
                compression: Compression::None,
            },
            MultipartDeepTiledPart {
                name: "beta".to_string(),
                width: w,
                height: h,
                tile_x: tx,
                tile_y: ty,
                channels: channels.clone(),
                samples_per_pixel: &spp1,
                channel_samples: vec![&planes1[0], &planes1[1], &planes1[2], &planes1[3]],
                compression: Compression::Rle,
            },
            MultipartDeepTiledPart {
                name: "gamma".to_string(),
                width: w,
                height: h,
                tile_x: tx,
                tile_y: ty,
                channels: channels.clone(),
                samples_per_pixel: &spp2,
                channel_samples: vec![&planes2[0], &planes2[1], &planes2[2], &planes2[3]],
                compression: Compression::Zips,
            },
        ];
        let bytes = encode_exr_multipart_deep_tiled(&parts).unwrap();
        let read = parse_exr_multipart_deep_tiled(&bytes).unwrap();
        assert_eq!(read.len(), 3);
        assert_eq!(read[0].compression, Compression::None);
        assert_eq!(read[1].compression, Compression::Rle);
        assert_eq!(read[2].compression, Compression::Zips);
        assert_mp_deep_tiled_part_roundtrip(&read[0], &spp0, &planes0);
        assert_mp_deep_tiled_part_roundtrip(&read[1], &spp1, &planes1);
        assert_mp_deep_tiled_part_roundtrip(&read[2], &spp2, &planes2);
    }

    #[test]
    fn mp_deep_tiled_edge_tiles_roundtrip() {
        // 13x9 image / 4x3 tiles forces 4x3 grid where the right column
        // (width 1) and bottom row (height 0) tiles are partial — exercises
        // the edge-trimming path in both encoder and reader.
        let (w, h, tx, ty) = (13u32, 9u32, 4u32, 3u32);
        let (spp0, planes0) = synthetic_deep(w, h);
        let (spp1, planes1) = synthetic_deep(w, h);
        let channels = mk_channels_rgba_float();
        let parts = vec![
            MultipartDeepTiledPart {
                name: "edge0".to_string(),
                width: w,
                height: h,
                tile_x: tx,
                tile_y: ty,
                channels: channels.clone(),
                samples_per_pixel: &spp0,
                channel_samples: vec![&planes0[0], &planes0[1], &planes0[2], &planes0[3]],
                compression: Compression::Zips,
            },
            MultipartDeepTiledPart {
                name: "edge1".to_string(),
                width: w,
                height: h,
                tile_x: tx,
                tile_y: ty,
                channels,
                samples_per_pixel: &spp1,
                channel_samples: vec![&planes1[0], &planes1[1], &planes1[2], &planes1[3]],
                compression: Compression::Rle,
            },
        ];
        let bytes = encode_exr_multipart_deep_tiled(&parts).unwrap();
        let read = parse_exr_multipart_deep_tiled(&bytes).unwrap();
        assert_eq!(read.len(), 2);
        assert_mp_deep_tiled_part_roundtrip(&read[0], &spp0, &planes0);
        assert_mp_deep_tiled_part_roundtrip(&read[1], &spp1, &planes1);
    }

    #[test]
    fn mp_deep_tiled_rejects_empty_parts() {
        let r = encode_exr_multipart_deep_tiled(&[]);
        assert!(r.is_err());
    }

    #[test]
    fn mp_deep_tiled_rejects_duplicate_names() {
        let (spp, planes) = synthetic_deep(4, 4);
        let channels = mk_channels_rgba_float();
        let parts = vec![
            MultipartDeepTiledPart {
                name: "dupe".to_string(),
                width: 4,
                height: 4,
                tile_x: 2,
                tile_y: 2,
                channels: channels.clone(),
                samples_per_pixel: &spp,
                channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
                compression: Compression::Zips,
            },
            MultipartDeepTiledPart {
                name: "dupe".to_string(),
                width: 4,
                height: 4,
                tile_x: 2,
                tile_y: 2,
                channels,
                samples_per_pixel: &spp,
                channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
                compression: Compression::Zips,
            },
        ];
        assert!(encode_exr_multipart_deep_tiled(&parts).is_err());
    }

    #[test]
    fn mp_deep_tiled_rejects_deep_zip() {
        let (spp, planes) = synthetic_deep(4, 4);
        let parts = vec![MultipartDeepTiledPart {
            name: "zip".to_string(),
            width: 4,
            height: 4,
            tile_x: 2,
            tile_y: 2,
            channels: mk_channels_rgba_float(),
            samples_per_pixel: &spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::Zip,
        }];
        assert!(encode_exr_multipart_deep_tiled(&parts).is_err());
    }

    #[test]
    fn mp_deep_tiled_rejects_single_part_file() {
        // Single-part deep-tiled bytes must not parse through the multi-part
        // entry (the wrong parser).
        let (spp, planes) = synthetic_deep(8, 4);
        let input = DeepTiledInput {
            width: 8,
            height: 4,
            tile_x: 4,
            tile_y: 2,
            channels: mk_channels_rgba_float(),
            samples_per_pixel: &spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::Zips,
        };
        let bytes = encode_exr_deep_tiled(&input).unwrap();
        assert!(parse_exr_multipart_deep_tiled(&bytes).is_err());
    }

    #[test]
    fn mp_deep_tiled_rejects_deep_scanline_multipart() {
        // Multi-part deep-scanline bytes must not parse through the deep-
        // tiled multi-part entry.
        let (spp, planes) = synthetic_deep(6, 4);
        let scan_parts = vec![MultipartDeepScanlinePart {
            name: "scan".to_string(),
            width: 6,
            height: 4,
            channels: mk_channels_rgba_float(),
            samples_per_pixel: &spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::Zips,
        }];
        let bytes = encode_exr_multipart_deep_scanline(&scan_parts).unwrap();
        assert!(parse_exr_multipart_deep_tiled(&bytes).is_err());
    }

    #[test]
    fn mp_deep_tiled_all_zero_samples_roundtrip() {
        // Edge case: every pixel has 0 samples in one part (total_samples=0
        // → empty channel buffers). Exercises the zero-payload tile path.
        let (w, h, tx, ty) = (6u32, 6u32, 3u32, 3u32);
        let zero_spp: Vec<u32> = vec![0; (w * h) as usize];
        let empty: Vec<f32> = Vec::new();
        let (spp1, planes1) = synthetic_deep(w, h);
        let channels = mk_channels_rgba_float();
        let parts = vec![
            MultipartDeepTiledPart {
                name: "zero".to_string(),
                width: w,
                height: h,
                tile_x: tx,
                tile_y: ty,
                channels: channels.clone(),
                samples_per_pixel: &zero_spp,
                channel_samples: vec![&empty, &empty, &empty, &empty],
                compression: Compression::Zips,
            },
            MultipartDeepTiledPart {
                name: "nonzero".to_string(),
                width: w,
                height: h,
                tile_x: tx,
                tile_y: ty,
                channels,
                samples_per_pixel: &spp1,
                channel_samples: vec![&planes1[0], &planes1[1], &planes1[2], &planes1[3]],
                compression: Compression::Rle,
            },
        ];
        let bytes = encode_exr_multipart_deep_tiled(&parts).unwrap();
        let read = parse_exr_multipart_deep_tiled(&bytes).unwrap();
        assert_eq!(read.len(), 2);
        assert_eq!(read[0].samples_per_pixel, zero_spp);
        for c in &read[0].channel_samples {
            assert!(c.is_empty());
        }
        assert_mp_deep_tiled_part_roundtrip(&read[1], &spp1, &planes1);
    }

    // -----------------------------------------------------------------
    // Round 208: single-part deep tiled MIPMAP_LEVELS WRITE + READ.
    // -----------------------------------------------------------------

    /// Build a MIPMAP pyramid of deep samples per level. Level `l` has
    /// dimensions `mipmap_level_dim(w0, l, false) ×
    /// mipmap_level_dim(h0, l, false)`. Each level is independent
    /// synthetic data (NOT box-filtered from level 0) — for the deep path
    /// per-pixel sample counts and per-sample values are not naturally
    /// box-filterable, and the file format itself does not mandate any
    /// relation between levels (the file just stores each level's deep
    /// pixels independently).
    fn synthetic_deep_mipmap(w0: u32, h0: u32) -> Vec<(Vec<u32>, [Vec<f32>; 4])> {
        let n_levels = crate::decoder::mipmap_level_count(w0.max(h0), false);
        let mut out = Vec::with_capacity(n_levels as usize);
        for l in 0..n_levels {
            let lw = crate::decoder::mipmap_level_dim(w0, l, false);
            let lh = crate::decoder::mipmap_level_dim(h0, l, false);
            out.push(synthetic_deep(lw, lh));
        }
        out
    }

    /// Build a [`DeepMipmapTiledInput`] from a pyramid of synthetic deep
    /// data with explicit level-0 dimensions.
    #[allow(clippy::type_complexity)]
    fn build_deep_mipmap<'a>(
        w0: u32,
        h0: u32,
        levels: &'a [(Vec<u32>, [Vec<f32>; 4])],
        tile_x: u32,
        tile_y: u32,
        compression: Compression,
    ) -> DeepMipmapTiledInput<'a> {
        let mut pyramid: Vec<DeepMipmapTiledLevelInput<'a>> = Vec::with_capacity(levels.len());
        for (l, (spp, planes)) in levels.iter().enumerate() {
            let lw = crate::decoder::mipmap_level_dim(w0, l as u32, false);
            let lh = crate::decoder::mipmap_level_dim(h0, l as u32, false);
            pyramid.push(DeepMipmapTiledLevelInput {
                width: lw,
                height: lh,
                samples_per_pixel: spp,
                channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            });
        }
        DeepMipmapTiledInput {
            tile_x,
            tile_y,
            channels: mk_channels_rgba_float(),
            pyramid,
            compression,
        }
    }

    fn assert_deep_mipmap_roundtrip(
        img: &DeepMipmapTiledImage,
        levels: &[(Vec<u32>, [Vec<f32>; 4])],
    ) {
        assert_eq!(img.level_count(), levels.len());
        for (l, (spp, planes)) in levels.iter().enumerate() {
            let lvl = &img.levels[l];
            assert_eq!(&lvl.samples_per_pixel, spp, "level {l} spp mismatch");
            for (ch_idx, plane) in planes.iter().enumerate() {
                assert_eq!(
                    &lvl.channel_samples[ch_idx], plane,
                    "level {l} ch {ch_idx} sample mismatch"
                );
            }
        }
    }

    #[test]
    fn deep_mipmap_tiled_roundtrip_none_16x16_tile_8() {
        // 16×16 → 5 levels (16, 8, 4, 2, 1). 8×8 tiles → level 0 has
        // 4 tiles, level 1 has 1 tile, levels 2-4 each have 1 tile.
        let (w0, h0) = (16u32, 16u32);
        let pyramid = synthetic_deep_mipmap(w0, h0);
        let input = build_deep_mipmap(w0, h0, &pyramid, 8, 8, Compression::None);
        let bytes = encode_exr_deep_tiled_mipmap(&input).unwrap();
        let img = parse_exr_deep_tiled_mipmap(&bytes).unwrap();
        assert_eq!(img.width(), w0);
        assert_eq!(img.height(), h0);
        assert_eq!(img.tile_x, 8);
        assert_eq!(img.tile_y, 8);
        assert_deep_mipmap_roundtrip(&img, &pyramid);
    }

    #[test]
    fn deep_mipmap_tiled_roundtrip_zips_24x16_tile_8x4() {
        // 24×16 → 5 levels (24, 12, 6, 3, 1). Edge tiles exercised at
        // levels where dims aren't multiples of (8,4).
        let (w0, h0) = (24u32, 16u32);
        let pyramid = synthetic_deep_mipmap(w0, h0);
        let input = build_deep_mipmap(w0, h0, &pyramid, 8, 4, Compression::Zips);
        let bytes = encode_exr_deep_tiled_mipmap(&input).unwrap();
        let img = parse_exr_deep_tiled_mipmap(&bytes).unwrap();
        assert_deep_mipmap_roundtrip(&img, &pyramid);
    }

    #[test]
    fn deep_mipmap_tiled_roundtrip_rle_edge_tiles_13x9_tile_4() {
        // 13×9 → 4 levels (13, 6, 3, 1). 4×4 tiles → level 0 has 4×3
        // = 12 tiles with edge tiles on right and bottom; lower levels
        // collapse to 1-2 tiles. Exercises RLE + edge tiles + the chunk
        // count math at every level.
        let (w0, h0) = (13u32, 9u32);
        let pyramid = synthetic_deep_mipmap(w0, h0);
        let input = build_deep_mipmap(w0, h0, &pyramid, 4, 4, Compression::Rle);
        let bytes = encode_exr_deep_tiled_mipmap(&input).unwrap();
        let img = parse_exr_deep_tiled_mipmap(&bytes).unwrap();
        assert_deep_mipmap_roundtrip(&img, &pyramid);
    }

    #[test]
    fn deep_mipmap_tiled_rejects_zip_compression() {
        // ZIP rejected by the reference exrinfo for deep data — match.
        let (w0, h0) = (8u32, 8u32);
        let pyramid = synthetic_deep_mipmap(w0, h0);
        let input = build_deep_mipmap(w0, h0, &pyramid, 4, 4, Compression::Zip);
        assert!(encode_exr_deep_tiled_mipmap(&input).is_err());
    }

    #[test]
    fn deep_mipmap_tiled_rejects_wrong_pyramid_length() {
        let (w0, h0) = (16u32, 16u32);
        let mut pyramid = synthetic_deep_mipmap(w0, h0);
        // 16×16 → 5 levels; truncate to 3.
        pyramid.truncate(3);
        let input = build_deep_mipmap(w0, h0, &pyramid, 8, 8, Compression::Zips);
        let r = encode_exr_deep_tiled_mipmap(&input);
        assert!(
            r.is_err(),
            "must reject pyramid with wrong number of levels"
        );
    }

    #[test]
    fn deep_mipmap_tiled_rejects_subsampled_channels() {
        let (w0, h0) = (8u32, 8u32);
        let pyramid = synthetic_deep_mipmap(w0, h0);
        let mut input = build_deep_mipmap(w0, h0, &pyramid, 4, 4, Compression::Zips);
        input.channels = vec![Channel {
            name: "Y".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 2,
            y_sampling: 2,
        }];
        // pyramid uses 4-channel planes; this test is just verifying the
        // sub-sampled-channel guard fires before the channel-count check.
        assert!(encode_exr_deep_tiled_mipmap(&input).is_err());
    }

    #[test]
    fn deep_mipmap_tiled_rejects_zero_tile_size() {
        let (w0, h0) = (8u32, 8u32);
        let pyramid = synthetic_deep_mipmap(w0, h0);
        let input = build_deep_mipmap(w0, h0, &pyramid, 0, 4, Compression::Zips);
        assert!(encode_exr_deep_tiled_mipmap(&input).is_err());
    }

    #[test]
    fn deep_mipmap_tiled_rejects_empty_pyramid() {
        let input = DeepMipmapTiledInput {
            tile_x: 4,
            tile_y: 4,
            channels: mk_channels_rgba_float(),
            pyramid: Vec::new(),
            compression: Compression::Zips,
        };
        assert!(encode_exr_deep_tiled_mipmap(&input).is_err());
    }

    #[test]
    fn parse_exr_deep_tiled_redirects_mipmap_to_new_entry() {
        // Encode a MIPMAP deep tiled file, then assert that the legacy
        // single-part deep tiled reader (which only accepts ONE_LEVEL)
        // surfaces a pointer error referring to the new entry rather
        // than mis-parsing it.
        let (w0, h0) = (8u32, 8u32);
        let pyramid = synthetic_deep_mipmap(w0, h0);
        let input = build_deep_mipmap(w0, h0, &pyramid, 4, 4, Compression::Zips);
        let bytes = encode_exr_deep_tiled_mipmap(&input).unwrap();
        let err = parse_exr_deep_tiled(&bytes).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("parse_exr_deep_tiled_mipmap"),
            "expected pointer to parse_exr_deep_tiled_mipmap, got: {msg}"
        );
    }

    #[test]
    fn parse_exr_deep_tiled_mipmap_rejects_one_level_file() {
        // Symmetric guard: encode a single-part ONE_LEVEL deep tiled
        // file and verify the MIPMAP reader rejects it explicitly rather
        // than mis-parsing.
        let (spp, planes) = synthetic_deep(8, 8);
        let one_level = DeepTiledInput {
            width: 8,
            height: 8,
            tile_x: 4,
            tile_y: 4,
            channels: mk_channels_rgba_float(),
            samples_per_pixel: &spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::Zips,
        };
        let bytes = encode_exr_deep_tiled(&one_level).unwrap();
        let err = parse_exr_deep_tiled_mipmap(&bytes).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("mode=0x00") || msg.contains("ONE_LEVEL"),
            "expected ONE_LEVEL rejection, got: {msg}"
        );
    }

    #[test]
    fn deep_mipmap_tiled_non_power_of_two_16x12_tile_4() {
        // 16×12 → 5 levels (16, 8, 4, 2, 1). Note that ROUND_DOWN halves
        // each dim independently; level dims are
        // (mipmap_level_dim(16, l), mipmap_level_dim(12, l)).
        let (w0, h0) = (16u32, 12u32);
        let pyramid = synthetic_deep_mipmap(w0, h0);
        let input = build_deep_mipmap(w0, h0, &pyramid, 4, 4, Compression::Zips);
        let bytes = encode_exr_deep_tiled_mipmap(&input).unwrap();
        let img = parse_exr_deep_tiled_mipmap(&bytes).unwrap();
        assert_eq!(img.level_count() as u32, std::cmp::max(w0, h0).ilog2() + 1);
        assert_deep_mipmap_roundtrip(&img, &pyramid);
    }

    #[test]
    fn deep_mipmap_tiled_version_field_bits() {
        // Single-part deep MIPMAP tiled discipline: only the non_image
        // (0x800) bit is set; multipart (0x1000) MUST NOT be set, and
        // single_tile (0x200) MUST NOT be set (mirroring the round-130
        // ONE_LEVEL single-part deep tiled writer).
        let (w0, h0) = (8u32, 8u32);
        let pyramid = synthetic_deep_mipmap(w0, h0);
        let input = build_deep_mipmap(w0, h0, &pyramid, 4, 4, Compression::None);
        let bytes = encode_exr_deep_tiled_mipmap(&input).unwrap();
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let bits = version & 0xFFFFFF00;
        assert_eq!(
            bits & 0x800,
            0x800,
            "non_image bit must be set for single-part deep tiled"
        );
        assert_eq!(
            bits & 0x1000,
            0,
            "multipart bit must NOT be set for single-part deep tiled"
        );
        assert_eq!(
            bits & 0x200,
            0,
            "single_tile bit must NOT be set for deep tiled \
             (type='deeptile' + tiles attribute carry the signal)"
        );
    }
}
