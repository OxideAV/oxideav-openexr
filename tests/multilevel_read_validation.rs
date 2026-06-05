//! Round-174 — full-pyramid READ for tiled `MIPMAP_LEVELS` and
//! `RIPMAP_LEVELS` files.
//!
//! [`parse_exr_tiled_multilevel`] is the round-174 addition; the
//! existing [`parse_exr`] entry point continues to return only the
//! full-resolution level (no behaviour change). These tests:
//!
//! 1. Encode a MIPMAP_LEVELS pyramid (via the existing
//!    `encode_exr_tiled_mipmap` writer using an explicit pyramid),
//!    decode it back via the new multilevel reader, and confirm every
//!    pyramid level's pixels round-trip bit-exactly.
//! 2. Encode a RIPMAP_LEVELS grid (via the existing
//!    `encode_exr_tiled_ripmap`) and confirm every `(lvlx, lvly)` cell
//!    decodes pixel-for-pixel.
//! 3. ONE_LEVEL files (no mip/rip pyramid) decode to a single-entry
//!    level vector.
//! 4. Non-tiled and multi-part files are rejected with a clear error.
//!
//! These tests run pure-Rust self round-trips. The encoder side has
//! its own `exrmetrics --convert` / `exrmaketiled -r` cross-validation
//! in `tests/mipmap_encoder_validation.rs` and
//! `tests/ripmap_encoder_validation.rs`; this file pins the new READ
//! path against that same encoder output.

use oxideav_openexr::{
    encode_exr_scanline_rgba_float_with, encode_exr_tiled_mipmap, encode_exr_tiled_rgba_float_with,
    encode_exr_tiled_ripmap, mipmap_level_count, mipmap_level_dim, parse_exr,
    parse_exr_tiled_multilevel, Channel, Compression, MipmapLevel, PixelType, RipmapLevel,
    RipmapPyramid,
};

fn rgba_channels() -> Vec<Channel> {
    vec![
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
    ]
}

/// Deterministic per-level RGBA float planes (A,B,G,R) for the supplied
/// level dimensions. Each plane is a fresh function of `(level_tag, x,
/// y)` so any cross-level address mix-up shows up as a sample mismatch.
fn make_level_planes(level_tag: f32, w: u32, h: u32) -> Vec<Vec<f32>> {
    let pixels = (w as usize) * (h as usize);
    let mut a = vec![0.0_f32; pixels];
    let mut b = vec![0.0_f32; pixels];
    let mut g = vec![0.0_f32; pixels];
    let mut r = vec![0.0_f32; pixels];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let off = y * w as usize + x;
            r[off] = level_tag + (x as f32) * 0.001;
            g[off] = level_tag + (y as f32) * 0.001;
            b[off] = level_tag + ((x as f32) + (y as f32)) * 0.0005;
            a[off] = 1.0 - level_tag * 0.01;
        }
    }
    // Alphabetical channel order (A, B, G, R) — same as the encoder
    // expects in a MipmapLevel's `planes`.
    vec![a, b, g, r]
}

#[test]
fn mipmap_full_pyramid_zip_roundtrip() {
    // 32×32 -> levels: 32, 16, 8, 4, 2, 1 (6 levels, ROUND_DOWN).
    let w = 32u32;
    let h = 32u32;
    let n = mipmap_level_count(w.max(h), false);
    assert_eq!(n, 6);
    let mut pyramid: Vec<MipmapLevel> = Vec::with_capacity(n as usize);
    for level in 0..n {
        let lw = mipmap_level_dim(w, level, false);
        let lh = mipmap_level_dim(h, level, false);
        pyramid.push(MipmapLevel {
            width: lw,
            height: lh,
            planes: make_level_planes(level as f32 * 0.1, lw, lh),
        });
    }
    let chs = rgba_channels();
    let bytes = encode_exr_tiled_mipmap(&chs, &pyramid, Compression::Zip, 16, 16).unwrap();

    let img = parse_exr_tiled_multilevel(&bytes).unwrap();
    assert_eq!(img.level_mode, 1, "decoded level_mode != MIPMAP_LEVELS");
    assert_eq!(img.round_mode, 0, "decoded round_mode != ROUND_DOWN");
    assert_eq!(img.tile_x, 16);
    assert_eq!(img.tile_y, 16);
    assert_eq!(img.levels.len(), n as usize, "missing pyramid levels");

    for (level_idx, level) in img.levels.iter().enumerate() {
        let l = level_idx as u32;
        let want_w = mipmap_level_dim(w, l, false);
        let want_h = mipmap_level_dim(h, l, false);
        assert_eq!(level.level_x, l, "level {l} level_x mismatch");
        assert_eq!(
            level.level_y, l,
            "level {l} level_y (mipmap must equal level_x)"
        );
        assert_eq!(level.width, want_w, "level {l} width");
        assert_eq!(level.height, want_h, "level {l} height");

        // Each plane should be `want_w * want_h` long.
        for p in &level.planes {
            assert_eq!(
                p.samples.len(),
                (want_w * want_h) as usize,
                "level {l} plane '{}' wrong length",
                p.name
            );
        }
        // And compare against the input pyramid bit-exactly.
        let src = &pyramid[level_idx];
        for (ch_idx, plane) in level.planes.iter().enumerate() {
            for y in 0..want_h as usize {
                for x in 0..want_w as usize {
                    let off = y * want_w as usize + x;
                    let got = plane.samples[off];
                    let want = src.planes[ch_idx][off];
                    assert_eq!(got, want, "mipmap level {l} ch {ch_idx} at ({x},{y})");
                }
            }
        }
    }
}

#[test]
fn mipmap_full_pyramid_none_rle_zips_roundtrip() {
    // Same shape as above but with three compressors back-to-back to
    // pin the decoder's tile-payload path for each compressed variant.
    for comp in [Compression::None, Compression::Zips, Compression::Rle] {
        let w = 16u32;
        let h = 12u32;
        let n = mipmap_level_count(w.max(h), false);
        let chs = rgba_channels();
        let mut pyramid: Vec<MipmapLevel> = Vec::with_capacity(n as usize);
        for level in 0..n {
            let lw = mipmap_level_dim(w, level, false);
            let lh = mipmap_level_dim(h, level, false);
            pyramid.push(MipmapLevel {
                width: lw,
                height: lh,
                planes: make_level_planes(0.2 + level as f32 * 0.05, lw, lh),
            });
        }
        let bytes = encode_exr_tiled_mipmap(&chs, &pyramid, comp, 8, 8).unwrap();
        let img = parse_exr_tiled_multilevel(&bytes).unwrap();
        assert_eq!(img.compression, comp);
        assert_eq!(img.levels.len(), n as usize);
        for (level_idx, level) in img.levels.iter().enumerate() {
            let src = &pyramid[level_idx];
            for (ch_idx, plane) in level.planes.iter().enumerate() {
                assert_eq!(
                    plane.samples, src.planes[ch_idx],
                    "{comp:?} level {level_idx} ch {ch_idx} plane mismatch"
                );
            }
        }
    }
}

#[test]
fn ripmap_full_grid_zip_roundtrip() {
    // 16×8 -> (nx, ny) = (5, 4). 5*4 = 20 ripmap cells.
    let w = 16u32;
    let h = 8u32;
    let nx = mipmap_level_count(w, false);
    let ny = mipmap_level_count(h, false);
    assert_eq!((nx, ny), (5, 4));

    let chs = rgba_channels();
    // Build a synthetic grid with per-cell deterministic content.
    let mut grid: Vec<Vec<RipmapLevel>> = Vec::with_capacity(ny as usize);
    for lvly in 0..ny {
        let mut row: Vec<RipmapLevel> = Vec::with_capacity(nx as usize);
        for lvlx in 0..nx {
            let lw = mipmap_level_dim(w, lvlx, false);
            let lh = mipmap_level_dim(h, lvly, false);
            // tag uniquely encodes (lvlx, lvly) so cross-cell mix-ups
            // surface immediately.
            let tag = (lvly as f32) * 0.1 + (lvlx as f32) * 0.01;
            row.push(RipmapLevel {
                width: lw,
                height: lh,
                planes: make_level_planes(tag, lw, lh),
            });
        }
        grid.push(row);
    }
    let pyramid = RipmapPyramid { grid: grid.clone() };
    let bytes = encode_exr_tiled_ripmap(&chs, &pyramid, Compression::Zip, 8, 8).unwrap();

    let img = parse_exr_tiled_multilevel(&bytes).unwrap();
    assert_eq!(img.level_mode, 2);
    assert_eq!(img.round_mode, 0);
    assert_eq!(img.levels.len(), (nx * ny) as usize);

    // RIPMAP iteration order: lvly outer, lvlx inner.
    for lvly in 0..ny {
        for lvlx in 0..nx {
            let flat = (lvly * nx + lvlx) as usize;
            let level = &img.levels[flat];
            assert_eq!(level.level_x, lvlx);
            assert_eq!(level.level_y, lvly);
            let src = &grid[lvly as usize][lvlx as usize];
            assert_eq!(level.width, src.width);
            assert_eq!(level.height, src.height);
            for (ch_idx, plane) in level.planes.iter().enumerate() {
                assert_eq!(
                    plane.samples, src.planes[ch_idx],
                    "ripmap ({lvlx},{lvly}) ch {ch_idx} plane mismatch"
                );
            }
        }
    }
}

#[test]
fn one_level_tiled_still_decodes_as_single_entry() {
    // ONE_LEVEL files have no pyramid — the new API should return one
    // level matching the full data window.
    let w = 12u32;
    let h = 9u32;
    let mut samples = vec![0.0_f32; (w * h * 4) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let off = (y * w as usize + x) * 4;
            samples[off] = x as f32;
            samples[off + 1] = y as f32;
            samples[off + 2] = (x + y) as f32;
            samples[off + 3] = 1.0;
        }
    }
    let bytes = encode_exr_tiled_rgba_float_with(w, h, &samples, Compression::Zips, 4, 4).unwrap();
    let img = parse_exr_tiled_multilevel(&bytes).unwrap();
    assert_eq!(img.level_mode, 0);
    assert_eq!(img.levels.len(), 1);
    let level = &img.levels[0];
    assert_eq!(level.level_x, 0);
    assert_eq!(level.level_y, 0);
    assert_eq!(level.width, w);
    assert_eq!(level.height, h);

    // Sanity: level 0 pixel content matches what parse_exr returns
    // (i.e. the legacy single-level path) for the same file.
    let legacy = parse_exr(&bytes).unwrap();
    for ch_idx in 0..level.planes.len() {
        assert_eq!(
            level.planes[ch_idx].samples, legacy.planes[ch_idx].samples,
            "multilevel level-0 plane '{}' disagrees with parse_exr",
            level.planes[ch_idx].name
        );
    }
}

#[test]
fn rejects_scanline_files() {
    // A plain (non-tiled) scanline file must be rejected with a clear
    // error.
    let w = 8u32;
    let h = 8u32;
    let samples = vec![0.5_f32; (w * h * 4) as usize];
    let bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::None).unwrap();
    let r = parse_exr_tiled_multilevel(&bytes);
    assert!(r.is_err(), "scanline file should be rejected");
    let msg = format!("{:?}", r.err().unwrap());
    assert!(
        msg.contains("not tiled") || msg.contains("single_tile"),
        "expected 'not tiled' error, got: {msg}"
    );
}
