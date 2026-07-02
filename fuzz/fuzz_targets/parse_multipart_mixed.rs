#![no_main]

//! Coverage-guided fuzz harness for
//! `oxideav_openexr::parse_exr_multipart_mixed`.
//!
//! Contract under test: every byte slice produces either
//! `Ok(Vec<MultipartMixedImage>)` or `Err(ExrError::*)`. Panics,
//! debug-mode integer overflows, index-out-of-bounds, and
//! attacker-claimed-length allocations are bugs.
//!
//! The mixed multi-part layout the decoder walks is, after the chained
//! per-part headers (each terminated by a lone 0x00, the chain by a
//! second 0x00):
//!
//! ```text
//! offset_tables : sum(chunkCount_i) * u64
//! per chunk     : i32 part_number, then one of four body shapes —
//!   scanline      : i32 Y, i32 size, payload
//!   tile          : 4×i32 (tx ty lvlx lvly), i32 size, payload
//!   deep scanline : i32 Y, 3×u64 sizes, table, data
//!   deep tile     : 4×i32 coords, 3×u64 sizes, table, data
//! ```
//!
//! The chunk-body shape is selected by the *part's* declared `type`,
//! the coordinates select tiles/levels inside per-part decode state,
//! and every u64 becomes a buffer offset or allocation bound — all of
//! it fuzz-controlled. Two modes:
//!
//!   1. Raw mode — hand the fuzz bytes straight to the decoder.
//!   2. Overlay mode — build a structurally valid mixed file (flat
//!      scanline part + multi-level deep MIPMAP tiled part) with the
//!      crate's own writer, then splice fuzz bytes over the offset
//!      tables + chunk region so the fuzzer reaches the per-part
//!      chunk-dispatch arithmetic without rediscovering two valid
//!      part headers from scratch.

use libfuzzer_sys::fuzz_target;
use oxideav_openexr::deep::DeepMipmapTiledLevelInput;
use oxideav_openexr::multipart_mixed_encoder::{
    encode_exr_multipart_mixed, parse_exr_multipart_mixed, MultipartMixedPart,
};
use oxideav_openexr::types::{Channel, Compression, PixelType};

fn mk_channels() -> Vec<Channel> {
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

/// Build a small valid mixed file: one flat scanline part + one deep
/// MIPMAP tiled part (8×8, tile 4×4 → levels 8/4/2/1 → 4+1+1+1 = 7
/// deep chunks). Returns `None` if the writer rejects the inputs.
fn base_file(compression: Compression) -> Option<Vec<u8>> {
    let w = 8u32;
    let h = 8u32;
    let pixels = (w * h) as usize;
    let flat: Vec<f32> = (0..pixels).map(|i| i as f32 * 0.125).collect();

    // Per-level deep data, held alive for the borrow in the input.
    let levels_data: Vec<(u32, u32, Vec<u32>, Vec<f32>, Vec<f32>)> = (0..4u32)
        .map(|l| {
            let lw = (w >> l).max(1);
            let lh = (h >> l).max(1);
            let n = (lw * lh) as usize;
            let spp: Vec<u32> = (0..n).map(|i| (i as u32) % 3).collect();
            let total: usize = spp.iter().map(|&v| v as usize).sum();
            let a: Vec<f32> = (0..total).map(|i| i as f32 * 0.5).collect();
            let z: Vec<f32> = (0..total).map(|i| i as f32 * 0.25).collect();
            (lw, lh, spp, a, z)
        })
        .collect();
    let pyramid: Vec<DeepMipmapTiledLevelInput> = levels_data
        .iter()
        .map(|(lw, lh, spp, a, z)| DeepMipmapTiledLevelInput {
            width: *lw,
            height: *lh,
            samples_per_pixel: spp,
            channel_samples: vec![a, z],
        })
        .collect();

    encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "flat".to_string(),
            width: w,
            height: h,
            channels: mk_channels(),
            planes: vec![&flat, &flat],
            compression,
        },
        MultipartMixedPart::DeepTiledMipmap {
            name: "dmip".to_string(),
            tile_x: 4,
            tile_y: 4,
            channels: mk_channels(),
            pyramid,
            compression: match compression {
                // Deep parts accept only NONE/ZIPS/RLE.
                Compression::None => Compression::None,
                Compression::Rle => Compression::Rle,
                _ => Compression::Zips,
            },
        },
    ])
    .ok()
}

fuzz_target!(|data: &[u8]| {
    // 1. Raw mode.
    let _ = parse_exr_multipart_mixed(data);

    if data.is_empty() {
        return;
    }

    // 2. Overlay mode. First byte selects the base compression; the
    // rest is spliced over everything after the header chain (offset
    // tables + chunk bodies).
    let compression = match data[0] % 3 {
        0 => Compression::None,
        1 => Compression::Rle,
        _ => Compression::Zips,
    };
    let Some(mut file) = base_file(compression) else {
        return;
    };
    let overlay = &data[1..];

    let keep = multipart_header_end(&file).unwrap_or(8).min(file.len());
    let region = &mut file[keep..];
    let take = overlay.len().min(region.len());
    region[..take].copy_from_slice(&overlay[..take]);

    let _ = parse_exr_multipart_mixed(&file);
});

/// Locate the byte just past the multi-part header chain's terminating
/// second NUL in a file produced by the crate's own mixed writer. Each
/// part header is a run of attributes ended by a lone 0x00; a further
/// 0x00 where the next part header would start ends the chain.
fn multipart_header_end(file: &[u8]) -> Option<usize> {
    let mut p = 8usize; // skip magic + version
    loop {
        // Walk one part header's attributes.
        loop {
            let b = *file.get(p)?;
            if b == 0 {
                p += 1; // end of this part header
                break;
            }
            p = next_nul(file, p)? + 1; // name
            p = next_nul(file, p)? + 1; // type
            let size = i32::from_le_bytes(file.get(p..p + 4)?.try_into().ok()?);
            if size < 0 {
                return None;
            }
            p = p.checked_add(4)?.checked_add(size as usize)?;
            if p > file.len() {
                return None;
            }
        }
        if *file.get(p)? == 0 {
            return Some(p + 1); // chain terminator
        }
    }
}

fn next_nul(file: &[u8], from: usize) -> Option<usize> {
    file[from..].iter().position(|&b| b == 0).map(|i| from + i)
}
