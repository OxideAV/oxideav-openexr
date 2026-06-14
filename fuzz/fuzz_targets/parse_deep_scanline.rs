#![no_main]

//! Coverage-guided fuzz harness for `oxideav_openexr::parse_exr_deep_scanline`.
//!
//! Contract under test: every byte slice produces either
//! `Ok(DeepExrImage)` or `Err(ExrError::*)`. Panics, debug-mode integer
//! overflows, index-out-of-bounds, and attacker-claimed-length
//! allocations are bugs.
//!
//! The deep scanline layout the decoder walks is, after the header:
//!
//! ```text
//! offset_table : chunkCount * u64   (file offset of each chunk)
//! per chunk    : i32 Y
//!                u64 packed_table_bytes
//!                u64 packed_data_bytes
//!                u64 unpacked_data_bytes
//!                <packed sample-count table>
//!                <packed sample payload>
//! ```
//!
//! Every u64 in that structure is turned into a buffer offset or an
//! allocation size, so a hostile file can claim arbitrary 64-bit values
//! there. The harness reaches that surface two ways:
//!
//!   1. Raw mode — hand the fuzz bytes straight to the decoder. Catches
//!      malformation in the header walker and shallow chunk parsing.
//!   2. Overlay mode — build a structurally valid deep file with the
//!      crate's own writer, then splice fuzz-controlled bytes over the
//!      offset table and block-header region. This reaches the deep
//!      block-header arithmetic without the fuzzer having to first
//!      rediscover a valid header from scratch.

use libfuzzer_sys::fuzz_target;
use oxideav_openexr::deep::{encode_exr_deep_scanline, DeepScanlineInput};
use oxideav_openexr::parse_exr_deep_scanline;
use oxideav_openexr::types::{Channel, Compression, PixelType};

fn mk_channels() -> Vec<Channel> {
    // Two HALF channels in alphabetical order (the decoder sorts, but
    // the writer wants them pre-sorted).
    vec![
        Channel {
            name: "A".to_string(),
            pixel_type: PixelType::Half,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        },
        Channel {
            name: "Z".to_string(),
            pixel_type: PixelType::Half,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        },
    ]
}

/// Build a small valid deep-scanline file. Returns `None` if the writer
/// rejects the inputs (keeps the harness panic-free on its own setup).
fn base_file(compression: Compression) -> Option<Vec<u8>> {
    let width = 4u32;
    let height = 3u32;
    let n = (width * height) as usize;
    let spp: Vec<u32> = (0..n).map(|i| (i as u32) % 3).collect();
    let total: usize = spp.iter().map(|&v| v as usize).sum();
    let ch_a: Vec<f32> = (0..total).map(|i| i as f32 * 0.5).collect();
    let ch_z: Vec<f32> = (0..total).map(|i| i as f32 * 0.25).collect();
    let input = DeepScanlineInput {
        width,
        height,
        channels: mk_channels(),
        samples_per_pixel: &spp,
        channel_samples: vec![&ch_a, &ch_z],
        compression,
    };
    encode_exr_deep_scanline(&input).ok()
}

fuzz_target!(|data: &[u8]| {
    // 1. Raw mode.
    let _ = parse_exr_deep_scanline(data);

    if data.is_empty() {
        return;
    }

    // 2. Overlay mode. First byte selects the base compression; the rest
    // is spliced over the bytes that follow the header (offset table +
    // block headers + payload).
    let compression = match data[0] % 3 {
        0 => Compression::None,
        1 => Compression::Rle,
        _ => Compression::Zips,
    };
    let Some(mut file) = base_file(compression) else {
        return;
    };
    let overlay = &data[1..];

    // Splice the overlay over the offset table + chunk bodies, keeping
    // the header intact so the fuzzer reaches the deep chunk arithmetic
    // (offset table u64s, per-block packed_table / packed_data /
    // unpacked_data) without first rediscovering a valid header. The
    // header runs from byte 8 to the double-NUL terminator (an empty
    // attribute name = a lone 0x00 byte after the last attribute);
    // everything after it is offset table + chunks.
    let keep = header_end(&file).unwrap_or(8).min(file.len());
    let region = &mut file[keep..];
    let take = overlay.len().min(region.len());
    region[..take].copy_from_slice(&overlay[..take]);

    let _ = parse_exr_deep_scanline(&file);
});

/// Locate the byte just past the header's double-NUL terminator in a file
/// produced by the crate's own deep writer. Returns the offset where the
/// offset table begins, or `None` if the terminator isn't found.
fn header_end(file: &[u8]) -> Option<usize> {
    // Skip magic(4) + version(4); then walk attributes. Each attribute is
    // name\0 type\0 size(i32) payload. An empty name (lone 0x00 where a
    // name would start) terminates the header.
    let mut p = 8usize;
    loop {
        let b = *file.get(p)?;
        if b == 0 {
            return Some(p + 1);
        }
        // name
        p = next_nul(file, p)? + 1;
        // type
        p = next_nul(file, p)? + 1;
        // size
        let size = i32::from_le_bytes(file.get(p..p + 4)?.try_into().ok()?);
        if size < 0 {
            return None;
        }
        p = p.checked_add(4)?.checked_add(size as usize)?;
        if p > file.len() {
            return None;
        }
    }
}

fn next_nul(file: &[u8], from: usize) -> Option<usize> {
    file[from..].iter().position(|&b| b == 0).map(|i| from + i)
}
