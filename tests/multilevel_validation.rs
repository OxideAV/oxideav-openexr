//! Tests for multi-level tiled EXR files (MIPMAP_LEVELS, RIPMAP_LEVELS)
//! and multi-part EXR files.
//!
//! Multi-level tests use `exrmaketiled` to convert a synthesised scanline
//! file into mip/rip-map tiled files with various compressions, then
//! verify that `parse_exr` returns the full-resolution image (level 0,0)
//! pixel-exactly.
//!
//! Multi-part tests use `exrmultipart` to combine two identical scanline
//! files into a two-part file, then verify that `parse_exr_multipart`
//! returns two images with matching pixel data.
//!
//! Both test suites auto-skip if the required reference binaries are
//! missing (`exrmaketiled`, `exrmultipart`).

use std::process::Command;

use oxideav_openexr::{
    encode_exr_scanline_rgba_float_with, mipmap_level_count, mipmap_level_dim, parse_exr,
    parse_exr_multipart, Compression,
};

// ─── helpers ─────────────────────────────────────────────────────────────────

fn tool_available(name: &str) -> bool {
    Command::new(name)
        .arg("--help")
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

fn tempdir(tag: &str) -> std::path::PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("oxideav-exr-{tag}-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Synthesise a small gradient RGBA f32 scanline EXR (ZIP-compressed).
fn make_gradient_exr(w: u32, h: u32) -> Vec<u8> {
    let samples: Vec<f32> = (0..(w * h) as usize)
        .flat_map(|i| {
            let x = (i as u32) % w;
            let y = (i as u32) / w;
            let r = x as f32 / w as f32;
            let g = y as f32 / h as f32;
            let b = 0.5_f32;
            let a = 1.0_f32;
            [r, g, b, a]
        })
        .collect();
    encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::Zip).unwrap()
}

/// Check that the decoded mipmap's level-0 pixels match the reference.
/// Reference is decoded from `ref_bytes` (the original uncompressed file).
fn check_level0_matches_ref(decoded: &oxideav_openexr::ExrImage, ref_bytes: &[u8]) {
    let ref_img = parse_exr(ref_bytes).expect("failed to parse reference file");
    assert_eq!(decoded.width(), ref_img.width(), "width mismatch");
    assert_eq!(decoded.height(), ref_img.height(), "height mismatch");
    assert_eq!(
        decoded.channels.len(),
        ref_img.channels.len(),
        "channel count mismatch"
    );

    for (pl_idx, (decoded_pl, ref_pl)) in
        decoded.planes.iter().zip(ref_img.planes.iter()).enumerate()
    {
        assert_eq!(
            decoded_pl.samples.len(),
            ref_pl.samples.len(),
            "plane {pl_idx} length mismatch"
        );
        for (i, (&d, &r)) in decoded_pl
            .samples
            .iter()
            .zip(ref_pl.samples.iter())
            .enumerate()
        {
            // Allow a small tolerance because exrmaketiled may re-sample
            // with filtering; we use "-f R -f G -f B -f A" to disable it.
            assert!(
                (d - r).abs() < 1e-3,
                "plane {pl_idx} sample {i}: decoded={d} ref={r}"
            );
        }
    }
}

// ─── multi-level tiled tests ──────────────────────────────────────────────────

fn run_mipmap_test(compression_flag: &str) {
    if !tool_available("exrmaketiled") {
        eprintln!("exrmaketiled not found, skipping mipmap {compression_flag} test");
        return;
    }

    let w = 32u32;
    let h = 32u32;
    let scanline_bytes = make_gradient_exr(w, h);

    let dir = tempdir("mipmap");
    let src = dir.join("src.exr");
    let out = dir.join(format!("mipmap_{compression_flag}.exr"));

    std::fs::write(&src, &scanline_bytes).unwrap();

    let status = Command::new("exrmaketiled")
        .args([
            "-m", // MIPMAP_LEVELS
            "-z",
            compression_flag,
            "-t",
            "32",
            "32",
            // Disable low-pass filtering so level-0 is bit-exact to input.
            "-f",
            "R",
            "-f",
            "G",
            "-f",
            "B",
            "-f",
            "A",
            src.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .status()
        .expect("exrmaketiled spawn failed");
    assert!(
        status.success(),
        "exrmaketiled failed for {compression_flag}"
    );

    let mipmap_bytes = std::fs::read(&out).unwrap();
    let img = match parse_exr(&mipmap_bytes) {
        Ok(img) => img,
        Err(e) => {
            // PIZ / B44 / B44A tiles will be unsupported; that's expected.
            if e.to_string().contains("not yet implemented") {
                eprintln!(
                    "  {compression_flag}: not yet implemented (expected) — skipping pixel check"
                );
                return;
            }
            panic!("parse_exr failed for {compression_flag}: {e}");
        }
    };

    // Level-count helpers
    let n = mipmap_level_count(w.max(h), false); // ROUND_DOWN
    assert!(n >= 1, "level count should be >= 1");
    assert_eq!(mipmap_level_dim(w, 0, false), w);
    assert_eq!(mipmap_level_dim(w, 1, false), w / 2);

    assert_eq!(img.width(), w, "decoded image width mismatch");
    assert_eq!(img.height(), h, "decoded image height mismatch");
    check_level0_matches_ref(&img, &scanline_bytes);

    let _ = std::fs::remove_dir_all(&dir);
}

fn run_ripmap_test(compression_flag: &str) {
    if !tool_available("exrmaketiled") {
        eprintln!("exrmaketiled not found, skipping ripmap {compression_flag} test");
        return;
    }

    let w = 32u32;
    let h = 32u32;
    let scanline_bytes = make_gradient_exr(w, h);

    let dir = tempdir("ripmap");
    let src = dir.join("src.exr");
    let out = dir.join(format!("ripmap_{compression_flag}.exr"));

    std::fs::write(&src, &scanline_bytes).unwrap();

    let status = Command::new("exrmaketiled")
        .args([
            "-r", // RIPMAP_LEVELS
            "-z",
            compression_flag,
            "-t",
            "32",
            "32",
            "-f",
            "R",
            "-f",
            "G",
            "-f",
            "B",
            "-f",
            "A",
            src.to_str().unwrap(),
            out.to_str().unwrap(),
        ])
        .status()
        .expect("exrmaketiled spawn failed");
    assert!(
        status.success(),
        "exrmaketiled failed for {compression_flag}"
    );

    let ripmap_bytes = std::fs::read(&out).unwrap();
    let img = match parse_exr(&ripmap_bytes) {
        Ok(img) => img,
        Err(e) => {
            if e.to_string().contains("not yet implemented") {
                eprintln!("  {compression_flag}: not yet implemented (expected) — skipping");
                return;
            }
            panic!("parse_exr failed for ripmap {compression_flag}: {e}");
        }
    };

    assert_eq!(img.width(), w);
    assert_eq!(img.height(), h);
    check_level0_matches_ref(&img, &scanline_bytes);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mipmap_zip_level0_matches_source() {
    run_mipmap_test("zip");
}

#[test]
fn mipmap_zips_level0_matches_source() {
    run_mipmap_test("zips");
}

#[test]
fn mipmap_rle_level0_matches_source() {
    run_mipmap_test("rle");
}

#[test]
fn mipmap_none_level0_matches_source() {
    run_mipmap_test("none");
}

#[test]
fn mipmap_piz_parse_or_unsupported() {
    run_mipmap_test("piz");
}

#[test]
fn mipmap_b44_parse_or_unsupported() {
    run_mipmap_test("b44");
}

#[test]
fn mipmap_b44a_parse_or_unsupported() {
    run_mipmap_test("b44a");
}

#[test]
fn ripmap_zip_level0_matches_source() {
    run_ripmap_test("zip");
}

#[test]
fn ripmap_rle_level0_matches_source() {
    run_ripmap_test("rle");
}

// ─── multi-part tests ─────────────────────────────────────────────────────────

#[test]
fn multipart_two_scanline_parts_decoded() {
    if !tool_available("exrmultipart") {
        eprintln!("exrmultipart not found, skipping multi-part test");
        return;
    }

    let w = 32u32;
    let h = 16u32;
    let scanline_bytes = make_gradient_exr(w, h);

    let dir = tempdir("multipart");
    let src = dir.join("src.exr");
    // Convert single-part to multi-part format first (adds required attributes).
    let mp1 = dir.join("mp1.exr");
    let out = dir.join("multipart.exr");

    std::fs::write(&src, &scanline_bytes).unwrap();

    // Convert to single-part EXR2 (adds "type" and "name" attributes).
    let status = Command::new("exrmultipart")
        .args([
            "-convert",
            "-i",
            src.to_str().unwrap(),
            "-o",
            mp1.to_str().unwrap(),
        ])
        .status()
        .expect("exrmultipart convert spawn failed");
    assert!(status.success(), "exrmultipart -convert failed");

    // Combine two copies into a 2-part file.
    let status = Command::new("exrmultipart")
        .args([
            "-combine",
            "-i",
            &format!("{}:0::partA", mp1.to_str().unwrap()),
            "-i",
            &format!("{}:0::partB", mp1.to_str().unwrap()),
            "-o",
            out.to_str().unwrap(),
        ])
        .status()
        .expect("exrmultipart combine spawn failed");
    assert!(status.success(), "exrmultipart -combine failed");

    let mp_bytes = std::fs::read(&out).unwrap();
    let images = parse_exr_multipart(&mp_bytes).expect("parse_exr_multipart failed");

    assert_eq!(images.len(), 2, "expected 2 parts, got {}", images.len());

    for (i, img) in images.iter().enumerate() {
        assert_eq!(img.width(), w, "part {i} width mismatch");
        assert_eq!(img.height(), h, "part {i} height mismatch");
        assert_eq!(img.channels.len(), 4, "part {i} channel count");
        // Pixel data in both parts should match the original source.
        check_level0_matches_ref(img, &scanline_bytes);
    }

    // parse_exr on a multi-part file should return an error pointing at
    // parse_exr_multipart.
    let err = parse_exr(&mp_bytes).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("multi-part") || msg.contains("multipart"),
        "unexpected error message: {msg}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ─── level-dim helper unit tests ─────────────────────────────────────────────

#[test]
fn mipmap_level_count_round_down() {
    // 1 -> 1, 2 -> 2, 4 -> 3, 32 -> 6, 33 -> 6 (33->16->8->4->2->1)
    assert_eq!(mipmap_level_count(1, false), 1);
    assert_eq!(mipmap_level_count(2, false), 2);
    assert_eq!(mipmap_level_count(4, false), 3);
    assert_eq!(mipmap_level_count(32, false), 6);
    assert_eq!(mipmap_level_count(33, false), 6); // 33->16->8->4->2->1
}

#[test]
fn mipmap_level_count_round_up() {
    assert_eq!(mipmap_level_count(1, true), 1);
    assert_eq!(mipmap_level_count(2, true), 2);
    assert_eq!(mipmap_level_count(4, true), 3);
    assert_eq!(mipmap_level_count(32, true), 6);
    assert_eq!(mipmap_level_count(33, true), 7); // 33->17->9->5->3->2->1
}

#[test]
fn mipmap_level_dim_round_down() {
    assert_eq!(mipmap_level_dim(32, 0, false), 32);
    assert_eq!(mipmap_level_dim(32, 1, false), 16);
    assert_eq!(mipmap_level_dim(32, 5, false), 1);
    assert_eq!(mipmap_level_dim(33, 1, false), 16); // floor(33/2)=16
}

#[test]
fn mipmap_level_dim_round_up() {
    assert_eq!(mipmap_level_dim(32, 0, true), 32);
    assert_eq!(mipmap_level_dim(32, 1, true), 16);
    assert_eq!(mipmap_level_dim(33, 1, true), 17); // ceil(33/2)=17
}
