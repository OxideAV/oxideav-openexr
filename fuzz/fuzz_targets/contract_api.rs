#![no_main]

//! Coverage-guided fuzz harness for the image-crate API contract root:
//! `probe`, `info`, `decode` / `decode_with` (lenient and strict, with a
//! layer / part selection), `decode_rgb8` / `decode_rgba8` and
//! `decode_all`. These wrap the depth readers already fuzzed by
//! `parse_flat` / `parse_multipart_mixed` / `parse_deep_scanline`, so
//! what this target adds is the contract plumbing itself: the header-only
//! `info` path (part shape, channel-set planning, chromaticities →
//! colour), the limit checks that must fire before any allocation, the
//! part / layer selection, the view mapping (luma/chroma reconstruction,
//! sub-sampled chroma), and the multi-part skip / strict logic of
//! `decode_all`.
//!
//! Contract under test: every byte slice produces either `Ok(..)` or
//! `Err(ExrError::*)`; `probe` never fails. Panics, debug-mode integer
//! overflows, index-out-of-bounds and attacker-claimed-length allocations
//! are bugs. A successful `decode` must also survive `to_rgb8` /
//! `to_rgba8` and `encode` (the view the decoder built is a valid image).
//!
//! Two modes:
//!
//!   1. Raw mode — hand the fuzz bytes to every root function.
//!   2. Overlay mode — build a small structurally valid file with the
//!      contract encoder (first byte selects the layout × compression ×
//!      channel-layout × part-shape combination, including a multi-part
//!      file and a layer-prefixed one), then splice the remaining fuzz
//!      bytes over everything past the header so the fuzzer reaches the
//!      chunk decoders and the view mapping without rediscovering a
//!      valid attribute table.

use libfuzzer_sys::fuzz_target;
use oxideav_openexr::{
    decode, decode_all, decode_all_with, decode_rgb8, decode_rgba8, decode_with, encode,
    encode_all, info, probe, ColourLayout, Compression, DecodeOptions, EncodeOptions, ExrImage,
    Frame, LevelMode, PixelFormat,
};

fn ramp(w: u32, h: u32, format: PixelFormat) -> ExrImage {
    let comps = format.components();
    let samples: Vec<f32> = (0..(w * h) as usize * comps)
        .map(|i| ((i * 13) % 97) as f32 / 96.0 - 0.25)
        .collect();
    ExrImage::from_f32(w, h, format, &samples).expect("ramp geometry")
}

/// A structurally valid file for selector byte `sel`, or `None` for the
/// combinations the writer refuses.
fn base_file(sel: u8) -> Option<Vec<u8>> {
    let compression = match sel & 0x0f {
        0 => Compression::None,
        1 => Compression::Rle,
        2 => Compression::Zips,
        3 => Compression::Zip,
        4 => Compression::Piz,
        5 => Compression::Pxr24,
        6 => Compression::B44,
        7 => Compression::B44a,
        8 => Compression::Dwaa,
        9 => Compression::Dwab,
        _ => return None,
    };
    let format = match (sel >> 4) & 0x3 {
        0 => PixelFormat::GrayF32Le,
        1 => PixelFormat::RgbF32Le,
        _ => PixelFormat::RgbaF32Le,
    };
    let shape = (sel >> 6) & 0x3;
    let (w, h) = (8u32, 6u32);
    let img = ramp(w, h, format);
    let opts = EncodeOptions::default().with_compression(compression);
    match shape {
        // Scanline, RGB layout, odd one layered.
        0 => encode(&img, &opts.with_layer(if sel & 0x10 != 0 { "diffuse" } else { "" })).ok(),
        // Luminance/chroma with 2x2 chroma (gray writes Y).
        1 => encode(&img, &opts.with_colour(ColourLayout::LumaChroma)).ok(),
        // Tiled with a level mode.
        2 => encode(
            &img,
            &opts.with_tile_size(4).with_levels(match sel & 0x3 {
                0 => LevelMode::One,
                1 => LevelMode::Mipmap,
                _ => LevelMode::Ripmap,
            }),
        )
        .ok(),
        // Multi-part: this image plus a smaller second part.
        _ => encode_all(
            &[
                Frame::new(img, 0).with_name("left".to_string()),
                Frame::new(ramp(4, 2, PixelFormat::RgbF32Le), 1).with_name("right".to_string()),
            ],
            &opts,
        )
        .ok(),
    }
}

/// Byte offset just past the header(s): the end of the attribute table
/// (single-part) or of the double-NUL that closes the multi-part header
/// list. `None` when the walk falls off the buffer.
fn header_end(file: &[u8]) -> Option<usize> {
    let version = u32::from_le_bytes([*file.get(4)?, *file.get(5)?, *file.get(6)?, *file.get(7)?]);
    let multipart = version & 0x1000 != 0;
    let mut p = 8usize;
    loop {
        if *file.get(p)? == 0 {
            p += 1;
            if !multipart || *file.get(p)? == 0 {
                return Some(if multipart { p + 1 } else { p });
            }
            continue;
        }
        while *file.get(p)? != 0 {
            p += 1;
        }
        p += 1;
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

fn exercise(bytes: &[u8], sel: u8) {
    let _ = probe(bytes);
    let _ = info(bytes);
    if let Ok(img) = decode(bytes) {
        let _ = img.to_rgb8();
        let _ = img.to_rgba8();
        let _ = encode(&img, &EncodeOptions::default());
    }
    let _ = decode_rgb8(bytes);
    let _ = decode_rgba8(bytes);
    // Tight limits must fail before any plane is allocated; strict and
    // the part / layer selectors exercise the remaining branches.
    let _ = decode_with(bytes, &DecodeOptions::default().with_max_bytes(1u64 << 20));
    let strict = DecodeOptions::default().with_strict(true);
    let _ = decode_with(bytes, &strict);
    let _ = decode_with(bytes, &DecodeOptions::default().with_layer("diffuse"));
    let _ = decode_with(bytes, &DecodeOptions::default().with_part(u32::from(sel & 1)));
    let _ = decode_with(bytes, &DecodeOptions::default().with_part_name("right"));
    let _ = decode_all(bytes);
    let _ = decode_all_with(bytes, &strict);
}

fuzz_target!(|data: &[u8]| {
    // 1. Raw mode.
    exercise(data, 0);

    if data.len() < 2 {
        return;
    }

    // 2. Overlay mode over a memoised writer-built base.
    static BASES: std::sync::OnceLock<Vec<Option<Vec<u8>>>> = std::sync::OnceLock::new();
    let bases = BASES.get_or_init(|| (0..=255u8).map(base_file).collect());
    let Some(mut file) = bases[data[0] as usize].clone() else {
        return;
    };
    let overlay = &data[1..];
    let keep = header_end(&file).unwrap_or(8).min(file.len());
    let region = &mut file[keep..];
    let take = overlay.len().min(region.len());
    region[..take].copy_from_slice(&overlay[..take]);
    exercise(&file, data[0]);
});
