#![no_main]

//! Coverage-guided fuzz harness for the single-part FLAT decode entry
//! points: `oxideav_openexr::parse_exr` (scanline + tiled ONE_LEVEL)
//! and `oxideav_openexr::parse_exr_tiled_multilevel` (MIPMAP / RIPMAP
//! pyramids). These are the oldest public readers but — unlike the
//! deep and mixed walkers — were never fuzzed, and they are the ONLY
//! route into the PXR24 byte-plane/delta, B44/B44A 4×4-block, PIZ
//! (bitmap / range LUT / wavelet / canonical-Huffman) and DWAA/DWAB
//! (rule block / sub-streams / AC run coding / IDCT) decoders, whose
//! per-chunk arithmetic (reorg-size accounting, block bit-unpacking,
//! raw-fallback length discrimination) is entirely attacker-controlled.
//!
//! Contract under test: every byte slice produces either `Ok(..)` or
//! `Err(ExrError::*)`. Panics, debug-mode integer overflows,
//! index-out-of-bounds, and attacker-claimed-length allocations are
//! bugs.
//!
//! Two modes:
//!
//!   1. Raw mode — hand the fuzz bytes to both entry points.
//!   2. Overlay mode — build a small structurally valid file with the
//!      crate's own writers (first byte selects among scanline / tiled
//!      / mipmap / ripmap shapes, compression schemes including PXR24,
//!      B44/B44A, PIZ and DWAA/DWAB, and the three lineOrder storage
//!      layouts), then
//!      splice the remaining fuzz bytes over everything past the
//!      header (offset table + chunk region) so the fuzzer reaches the
//!      per-chunk decode arithmetic without rediscovering a valid
//!      attribute table from scratch.

use libfuzzer_sys::fuzz_target;
use oxideav_openexr::{
    build_box_filter_pyramid, build_box_filter_ripmap,
    encode_exr_scanline_rgba_float_with_line_order, encode_exr_tiled_mipmap,
    encode_exr_tiled_rgba_float_with_line_order, encode_exr_tiled_ripmap, parse_exr,
    parse_exr_tiled_multilevel, Channel, Compression, LineOrder, PixelType,
};

fn rgba(w: u32, h: u32) -> Vec<f32> {
    (0..(w * h * 4) as usize)
        .map(|i| ((i * 13) % 97) as f32 / 96.0)
        .collect()
}

fn gray_channel(pt: PixelType) -> Vec<Channel> {
    vec![Channel {
        name: "G".to_string(),
        pixel_type: pt,
        p_linear: false,
        x_sampling: 1,
        y_sampling: 1,
    }]
}

/// Build one of a spread of valid single-part flat files. `sel` picks
/// the shape × compression × line-order combination.
fn base_file(sel: u8) -> Option<Vec<u8>> {
    let comp = match sel % 9 {
        0 => Compression::None,
        1 => Compression::Zip,
        2 => Compression::Rle,
        3 => Compression::Pxr24,
        4 => Compression::B44,
        5 => Compression::B44a,
        // Round 439: PIZ (bitmap/LUT/wavelet/Huffman) and DWAA/DWAB
        // (rule block, four sub-streams, AC/DC coding, IDCT) are decode
        // arithmetic wholly driven by attacker-controlled chunk bytes.
        6 => Compression::Piz,
        7 => Compression::Dwaa,
        _ => Compression::Dwab,
    };
    let lo = match (sel / 9) % 2 {
        0 => LineOrder::IncreasingY,
        _ => LineOrder::DecreasingY,
    };
    match (sel / 18) % 4 {
        // Scanline 9×37 (partial final block for every block height).
        0 => encode_exr_scanline_rgba_float_with_line_order(9, 37, &rgba(9, 37), comp, lo).ok(),
        // Tiled ONE_LEVEL with edge tiles; exercise RANDOM_Y too.
        1 => {
            let lo = if (sel / 9) % 2 == 0 {
                LineOrder::RandomY
            } else {
                lo
            };
            encode_exr_tiled_rgba_float_with_line_order(13, 9, &rgba(13, 9), comp, 5, 4, lo).ok()
        }
        // MIPMAP pyramid, single gray channel (HALF exercises B44
        // block packing; FLOAT exercises PXR24 3-byte planes).
        2 => {
            let pt = if sel % 2 == 0 {
                PixelType::Half
            } else {
                PixelType::Float
            };
            let plane: Vec<f32> = (0..16 * 16).map(|i| (i % 29) as f32 / 28.0).collect();
            let pyr = build_box_filter_pyramid(16, 16, &[plane]);
            encode_exr_tiled_mipmap(&gray_channel(pt), &pyr, comp, 5, 5).ok()
        }
        // RIPMAP grid.
        _ => {
            let plane: Vec<f32> = (0..16 * 8).map(|i| (i % 31) as f32 / 30.0).collect();
            let pyr = build_box_filter_ripmap(16, 8, &[plane]);
            encode_exr_tiled_ripmap(&gray_channel(PixelType::Half), &pyr, comp, 4, 4).ok()
        }
    }
}

/// Byte position just past the single-part header's terminating NUL
/// (i.e. where the offset table begins) in a file produced by the
/// crate's own writers.
fn header_end(file: &[u8]) -> Option<usize> {
    let mut p = 8usize; // magic + version
    loop {
        if *file.get(p)? == 0 {
            return Some(p + 1);
        }
        // name NUL
        while *file.get(p)? != 0 {
            p += 1;
        }
        p += 1;
        // type NUL
        while *file.get(p)? != 0 {
            p += 1;
        }
        p += 1;
        let size = i32::from_le_bytes([
            *file.get(p)?,
            *file.get(p + 1)?,
            *file.get(p + 2)?,
            *file.get(p + 3)?,
        ]);
        p = p.checked_add(4)?.checked_add(usize::try_from(size).ok()?)?;
    }
}

fuzz_target!(|data: &[u8]| {
    // 1. Raw mode — both entry points.
    let _ = parse_exr(data);
    let _ = parse_exr_tiled_multilevel(data);

    if data.len() < 2 {
        return;
    }

    // 2. Overlay mode.
    let Some(mut file) = base_file(data[0]) else {
        return;
    };
    let overlay = &data[1..];
    let keep = header_end(&file).unwrap_or(8).min(file.len());
    let region = &mut file[keep..];
    let take = overlay.len().min(region.len());
    region[..take].copy_from_slice(&overlay[..take]);

    let _ = parse_exr(&file);
    let _ = parse_exr_tiled_multilevel(&file);
});
