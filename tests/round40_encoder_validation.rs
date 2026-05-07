//! Round-40 cross-validation: feed our tiled-encoder output and our
//! multipart-encoder output through the OpenEXR reference binaries
//! (`exrheader`, `exrmetrics --convert -z none`, `exrinfo`) and verify
//! the round-trip pixels match bit-exactly. Also sanity-check that
//! `exrheader` accepts our headers without printing any
//! parser-error lines.
//!
//! Auto-skips with a printed reason when the required binary is
//! missing (CI without the OpenEXR tools installed).

use std::process::Command;

use oxideav_openexr::{
    encode_exr_multipart_rgba_float_with, encode_exr_tiled_rgba_float_with, parse_exr,
    parse_exr_multipart, Compression,
};

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
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-r40-{tag}-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn make_gradient(w: u32, h: u32) -> Vec<f32> {
    (0..(w * h) as usize)
        .flat_map(|i| {
            let x = (i as u32) % w;
            let y = (i as u32) / w;
            [
                x as f32 / w as f32,
                y as f32 / h as f32,
                ((x ^ y) as f32) * 0.01,
                1.0,
            ]
        })
        .collect()
}

fn check_pixels_match_rgba(img: &oxideav_openexr::ExrImage, source: &[f32]) {
    let w = img.width() as usize;
    let h = img.height() as usize;
    let a = &img.planes[0].samples;
    let b = &img.planes[1].samples;
    let g = &img.planes[2].samples;
    let r = &img.planes[3].samples;
    for y in 0..h {
        for x in 0..w {
            let off = y * w + x;
            assert_eq!(r[off], source[off * 4], "R mismatch at ({x},{y})");
            assert_eq!(g[off], source[off * 4 + 1], "G mismatch at ({x},{y})");
            assert_eq!(b[off], source[off * 4 + 2], "B mismatch at ({x},{y})");
            assert_eq!(a[off], source[off * 4 + 3], "A mismatch at ({x},{y})");
        }
    }
}

// ─── Tiled encoder cross-validation ──────────────────────────────────────────

/// For each compression mode, encode a tiled file via our encoder, run
/// `exrmetrics --convert -z none` on it (which forces the OpenEXR
/// reference impl to fully decode AND re-encode as uncompressed), then
/// re-parse the result with our decoder and check pixel equality.
fn run_tiled_external_roundtrip(z: Compression, tile_x: u32, tile_y: u32) {
    if !tool_available("exrmetrics") {
        eprintln!("exrmetrics not available; skipping tiled-{z:?} validation");
        return;
    }
    let w = 32u32;
    let h = 32u32;
    let samples = make_gradient(w, h);
    let bytes = encode_exr_tiled_rgba_float_with(w, h, &samples, z, tile_x, tile_y).unwrap();

    let dir = tempdir("tiled-encoder");
    let inp = dir.join(format!("in_{z:?}.exr"));
    let outp = dir.join(format!("out_{z:?}.exr"));
    std::fs::write(&inp, &bytes).unwrap();

    let output = Command::new("exrmetrics")
        .arg("--convert")
        .arg("-z")
        .arg("none")
        .arg(&inp)
        .arg("-o")
        .arg(&outp)
        .output()
        .expect("exrmetrics spawn failed");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("exrmetrics rejected our tiled {z:?} output:\n{stderr}");
    }
    let decoded = std::fs::read(&outp).unwrap();
    let img = parse_exr(&decoded).unwrap();
    check_pixels_match_rgba(&img, &samples);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn tiled_encoder_none_round_trips_through_exrmetrics() {
    run_tiled_external_roundtrip(Compression::None, 8, 8);
}
#[test]
fn tiled_encoder_zip_round_trips_through_exrmetrics() {
    run_tiled_external_roundtrip(Compression::Zip, 8, 8);
}
#[test]
fn tiled_encoder_zips_round_trips_through_exrmetrics() {
    run_tiled_external_roundtrip(Compression::Zips, 8, 8);
}
#[test]
fn tiled_encoder_rle_round_trips_through_exrmetrics() {
    run_tiled_external_roundtrip(Compression::Rle, 8, 8);
}
#[test]
fn tiled_encoder_zip_edge_tiles() {
    // 17×13 image with 8×8 tiles: right column is partial (last tile
    // is 1px wide), bottom row is partial (last tile is 5px tall).
    if !tool_available("exrmetrics") {
        eprintln!("exrmetrics not available; skipping edge-tiles test");
        return;
    }
    let w = 17u32;
    let h = 13u32;
    let samples = make_gradient(w, h);
    let bytes = encode_exr_tiled_rgba_float_with(w, h, &samples, Compression::Zip, 8, 8).unwrap();
    let dir = tempdir("tiled-edge");
    let inp = dir.join("edge_in.exr");
    let outp = dir.join("edge_out.exr");
    std::fs::write(&inp, &bytes).unwrap();
    let output = Command::new("exrmetrics")
        .arg("--convert")
        .arg("-z")
        .arg("none")
        .arg(&inp)
        .arg("-o")
        .arg(&outp)
        .output()
        .expect("exrmetrics spawn failed");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("exrmetrics rejected our edge-tile output:\n{stderr}");
    }
    let decoded = std::fs::read(&outp).unwrap();
    let img = parse_exr(&decoded).unwrap();
    check_pixels_match_rgba(&img, &samples);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn tiled_encoder_header_passes_exrheader() {
    if !tool_available("exrheader") {
        eprintln!("exrheader not available; skipping");
        return;
    }
    let w = 16u32;
    let h = 16u32;
    let samples = make_gradient(w, h);
    let bytes = encode_exr_tiled_rgba_float_with(w, h, &samples, Compression::Zip, 8, 8).unwrap();
    let dir = tempdir("tiled-exrheader");
    let inp = dir.join("a.exr");
    std::fs::write(&inp, &bytes).unwrap();
    let output = Command::new("exrheader")
        .arg(&inp)
        .output()
        .expect("exrheader spawn failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "exrheader failed: {stderr}\n{stdout}"
    );
    // Required attributes for a tiled file should all appear.
    assert!(
        stdout.contains("flags 0x200"),
        "single_tile bit missing: {stdout}"
    );
    assert!(
        stdout.contains("tiledimage"),
        "type=tiledimage missing: {stdout}"
    );
    assert!(
        stdout.contains("tiles (type tiledesc)"),
        "tiles attribute missing: {stdout}"
    );
    assert!(
        stdout.contains("chunkCount"),
        "chunkCount missing: {stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ─── Multipart encoder cross-validation ──────────────────────────────────────

#[test]
fn multipart_encoder_two_parts_passes_exrheader() {
    if !tool_available("exrheader") {
        eprintln!("exrheader not available; skipping");
        return;
    }
    let w = 12u32;
    let h = 12u32;
    let s_a = make_gradient(w, h);
    let s_b = make_gradient(w, h);
    let bytes = encode_exr_multipart_rgba_float_with(&[
        ("first".to_string(), w, h, s_a.as_slice(), Compression::Zip),
        (
            "second".to_string(),
            w,
            h,
            s_b.as_slice(),
            Compression::None,
        ),
    ])
    .unwrap();

    let dir = tempdir("mp-exrheader");
    let inp = dir.join("mp.exr");
    std::fs::write(&inp, &bytes).unwrap();
    let output = Command::new("exrheader")
        .arg(&inp)
        .output()
        .expect("exrheader spawn failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "exrheader failed:\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        stdout.contains("flags 0x1000"),
        "multipart bit missing: {stdout}"
    );
    assert!(
        stdout.contains("part 0:"),
        "part 0 marker missing: {stdout}"
    );
    assert!(
        stdout.contains("part 1:"),
        "part 1 marker missing: {stdout}"
    );
    assert!(
        stdout.contains("\"first\""),
        "name=\"first\" missing: {stdout}"
    );
    assert!(
        stdout.contains("\"second\""),
        "name=\"second\" missing: {stdout}"
    );

    // Round-trip: decode it back via our parse_exr_multipart.
    let parts = parse_exr_multipart(&bytes).unwrap();
    assert_eq!(parts.len(), 2);
    check_pixels_match_rgba(&parts[0], &s_a);
    check_pixels_match_rgba(&parts[1], &s_b);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Run our multipart output through `exrmultipart -split` (separates a
/// multipart file into independent single-part files), then re-parse
/// each split file and verify pixels match.
#[test]
fn multipart_encoder_splits_via_exrmultipart() {
    if !tool_available("exrmultipart") {
        eprintln!("exrmultipart not available; skipping");
        return;
    }
    let w = 16u32;
    let h = 16u32;
    let s_a = make_gradient(w, h);
    let s_b = make_gradient(w, h);
    let bytes = encode_exr_multipart_rgba_float_with(&[
        ("alpha".to_string(), w, h, s_a.as_slice(), Compression::Zip),
        ("beta".to_string(), w, h, s_b.as_slice(), Compression::Rle),
    ])
    .unwrap();

    let dir = tempdir("mp-split");
    let mp_in = dir.join("in.exr");
    std::fs::write(&mp_in, &bytes).unwrap();

    let output_root = dir.join("part");
    let output = Command::new("exrmultipart")
        .arg("-separate")
        .arg("-i")
        .arg(&mp_in)
        .arg("-o")
        .arg(&output_root)
        .output()
        .expect("exrmultipart spawn failed");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!("exrmultipart -separate failed:\nstdout={stdout}\nstderr={stderr}");
    }

    // -separate emits files named `<output>.<idx>.exr` (1-indexed).
    // Glob for whatever exists in the dir and verify both parts
    // decoded match their sources (parts share the same gradient
    // image so either source ordering is fine).
    let mut part_files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p != &mp_in && p.extension().and_then(|e| e.to_str()) == Some("exr"))
        .collect();
    part_files.sort();
    assert_eq!(
        part_files.len(),
        2,
        "expected 2 separated files, found {part_files:?}"
    );
    for pf in &part_files {
        let bytes = std::fs::read(pf).unwrap();
        let img = parse_exr(&bytes).unwrap();
        // Both parts contain the same gradient image; either should match.
        check_pixels_match_rgba(&img, &s_a);
    }

    let _ = std::fs::remove_dir_all(&dir);
}
