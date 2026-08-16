//! Lossy compression (PXR24 / B44 / B44A) through the single-part
//! multilevel writers (`encode_exr_tiled_mipmap` /
//! `encode_exr_tiled_ripmap`) — previously these writers accepted only
//! NONE / ZIP / ZIPS / RLE while the ONE_LEVEL tiled and multi-part
//! mixed writers already carried the lossy schemes.
//!
//! Validation strategy: self round-trip through
//! `parse_exr_tiled_multilevel` with the scheme's quantisation
//! tolerance, plus reference-binary cross-checks (auto-skip when
//! absent): `exrheader` must echo the compression, `exrinfo` must
//! accept the file, and `exrmetrics --convert -z none` must re-encode
//! it with the level-0 pixels **bit-exactly equal** to our own decode
//! of the same file (both sides sit post-quantisation, so agreement is
//! exact, making this a decoder-vs-decoder comparison).

use std::process::Command;

use oxideav_openexr::{
    build_box_filter_pyramid, build_box_filter_ripmap, encode_exr_tiled_mipmap,
    encode_exr_tiled_ripmap, parse_exr, parse_exr_tiled_multilevel, Channel, Compression,
    PixelType,
};

fn tool_available(bin: &str) -> bool {
    Command::new(bin)
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

fn tempdir() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-mllossy-{nanos}-{pid}-{seq}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

fn channel(name: &str, pt: PixelType) -> Channel {
    Channel {
        name: name.to_string(),
        pixel_type: pt,
        p_linear: false,
        x_sampling: 1,
        y_sampling: 1,
    }
}

/// Scattered (high per-block dynamic range) plane — used for schemes
/// whose error bound does not depend on block content.
fn scattered_plane(w: u32, h: u32) -> Vec<f32> {
    (0..(w * h) as usize)
        .map(|i| ((i * 37) % 509) as f32 / 508.0)
        .collect()
}

/// Smooth gradient plane — B44's 6-bit shifted block differences give a
/// per-sample error proportional to the 4×4 block's dynamic range, so
/// its tolerance check uses image-like smooth data (the bit-exact
/// reference-vs-ours comparison below is the correctness anchor either
/// way).
fn smooth_plane(w: u32, h: u32) -> Vec<f32> {
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            out.push((x + y) as f32 / (w + h) as f32);
        }
    }
    out
}

fn test_plane_for(comp: Compression, w: u32, h: u32) -> Vec<f32> {
    match comp {
        Compression::B44 | Compression::B44a => smooth_plane(w, h),
        _ => scattered_plane(w, h),
    }
}

/// Per-scheme absolute tolerance for values in [0, 1]:
/// PXR24 rounds FLOAT mantissas to 15 bits; B44/B44A quantise HALF
/// blocks with 6-bit shifted differences (larger error).
fn tolerance(comp: Compression, pt: PixelType) -> f32 {
    match (comp, pt) {
        // HALF itself has ~2^-11 steps near 1.0; B44 block quantisation
        // can add a few steps on top.
        (Compression::B44 | Compression::B44a, PixelType::Half) => 1.0 / 64.0,
        // The f32 test values are not exactly representable in binary16,
        // so HALF channels see plain half-rounding even under schemes
        // that carry the HALF codes losslessly.
        (_, PixelType::Half) => 1.0 / 2048.0,
        (Compression::Pxr24, _) => 1.0 / 32768.0,
        _ => 0.0,
    }
}

fn check_mipmap(comp: Compression, pt: PixelType) {
    let (w, h) = (32u32, 32u32);
    let chs = vec![channel("G", pt)];
    let plane = test_plane_for(comp, w, h);
    let pyramid = build_box_filter_pyramid(w, h, std::slice::from_ref(&plane));
    let bytes = encode_exr_tiled_mipmap(&chs, &pyramid, comp, 16, 16).unwrap();

    let img = parse_exr_tiled_multilevel(&bytes).unwrap();
    assert_eq!(img.levels.len(), pyramid.len());
    let tol = tolerance(comp, pt);
    for (lvl, want) in img.levels.iter().zip(pyramid.iter()) {
        assert_eq!((lvl.width, lvl.height), (want.width, want.height));
        for (i, (got, want_v)) in lvl.planes[0]
            .samples
            .iter()
            .zip(want.planes[0].iter())
            .enumerate()
        {
            assert!(
                (got - want_v).abs() <= tol,
                "{comp:?}/{pt:?} level {} sample {i}: got {got}, want {want_v} (tol {tol})",
                lvl.level_x
            );
        }
    }

    reference_check(&bytes, &img.levels[0].planes[0].samples, w, h, comp);
}

fn check_ripmap(comp: Compression, pt: PixelType) {
    let (w, h) = (32u32, 16u32);
    let chs = vec![channel("G", pt)];
    let plane = test_plane_for(comp, w, h);
    let pyramid = build_box_filter_ripmap(w, h, std::slice::from_ref(&plane));
    let bytes = encode_exr_tiled_ripmap(&chs, &pyramid, comp, 8, 8).unwrap();

    let img = parse_exr_tiled_multilevel(&bytes).unwrap();
    let tol = tolerance(comp, pt);
    let mut idx = 0;
    for (lvly, row) in pyramid.grid.iter().enumerate() {
        for (lvlx, cell) in row.iter().enumerate() {
            let lvl = &img.levels[idx];
            idx += 1;
            assert_eq!(
                (lvl.level_x, lvl.level_y, lvl.width, lvl.height),
                (lvlx as u32, lvly as u32, cell.width, cell.height)
            );
            for (i, (got, want_v)) in lvl.planes[0]
                .samples
                .iter()
                .zip(cell.planes[0].iter())
                .enumerate()
            {
                assert!(
                    (got - want_v).abs() <= tol,
                    "{comp:?}/{pt:?} cell ({lvlx},{lvly}) sample {i}: got {got}, want {want_v}"
                );
            }
        }
    }
    assert_eq!(idx, img.levels.len());

    reference_check(&bytes, &img.levels[0].planes[0].samples, w, h, comp);
}

/// exrheader echoes the compression, exrinfo accepts, and the
/// reference decode (via convert-to-NONE) equals OUR decode bit-exact.
fn reference_check(bytes: &[u8], our_level0: &[f32], w: u32, h: u32, comp: Compression) {
    let echo = match comp {
        Compression::Pxr24 => "pxr24:",
        Compression::B44 => "b44:",
        Compression::B44a => "b44a:",
        _ => unreachable!(),
    };
    let dir = tempdir();
    let path = format!("{dir}/ml.exr");
    std::fs::write(&path, bytes).unwrap();

    if tool_available("exrheader") {
        let out = Command::new("exrheader").arg(&path).output().unwrap();
        assert!(out.status.success(), "exrheader rejected {comp:?} file");
        let text = String::from_utf8_lossy(&out.stdout).to_lowercase();
        assert!(
            text.contains(echo),
            "exrheader did not echo {echo}:\n{text}"
        );
    } else {
        eprintln!("exrheader not available, skipping ({comp:?})");
    }

    if tool_available("exrinfo") {
        let out = Command::new("exrinfo").arg(&path).output().unwrap();
        assert!(
            out.status.success(),
            "exrinfo rejected {comp:?} multilevel file: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    } else {
        eprintln!("exrinfo not available, skipping ({comp:?})");
    }

    if tool_available("exrmetrics") {
        let conv = format!("{dir}/conv.exr");
        let out = Command::new("exrmetrics")
            .args(["--convert", "-z", "none", "-o", &conv, &path])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "exrmetrics --convert rejected {comp:?} multilevel file: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let converted = std::fs::read(&conv).unwrap();
        let img = parse_exr(&converted).unwrap();
        assert_eq!((img.width(), img.height()), (w, h));
        for (i, (reference, ours)) in img.planes[0]
            .samples
            .iter()
            .zip(our_level0.iter())
            .enumerate()
        {
            assert_eq!(
                reference, ours,
                "{comp:?} level-0 sample {i}: reference decode {reference} != our decode {ours}"
            );
        }
    } else {
        eprintln!("exrmetrics not available, skipping ({comp:?})");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mipmap_pxr24_float() {
    check_mipmap(Compression::Pxr24, PixelType::Float);
}

#[test]
fn mipmap_pxr24_half_lossless() {
    // PXR24 is lossless for HALF channels.
    check_mipmap(Compression::Pxr24, PixelType::Half);
}

#[test]
fn mipmap_b44_half() {
    check_mipmap(Compression::B44, PixelType::Half);
}

#[test]
fn mipmap_b44a_half() {
    check_mipmap(Compression::B44a, PixelType::Half);
}

#[test]
fn mipmap_b44_float_raw_copy() {
    // B44 copies FLOAT channels raw — lossless.
    check_mipmap(Compression::B44, PixelType::Float);
}

#[test]
fn ripmap_pxr24_float() {
    check_ripmap(Compression::Pxr24, PixelType::Float);
}

#[test]
fn ripmap_b44_half() {
    check_ripmap(Compression::B44, PixelType::Half);
}

#[test]
fn ripmap_b44a_half() {
    check_ripmap(Compression::B44a, PixelType::Half);
}
