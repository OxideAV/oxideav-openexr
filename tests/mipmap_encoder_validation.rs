//! Cross-validation of our MIPMAP_LEVELS tiled encoder against the OpenEXR
//! reference binaries.
//!
//! The round-78 mipmap encoder writes single-part `MIPMAP_LEVELS` tiled
//! EXR files (`tiledesc.level_mode = 1`) with a multi-level pyramid built
//! by a 2×2 box filter. We validate by:
//!
//! 1. Encoding a synthesised gradient.
//! 2. Asking `exrmetrics --convert -z none` to decode + re-encode the
//!    file as an uncompressed scanline file. If our chunks are spec-
//!    legal, that decode succeeds.
//! 3. Re-parsing the converted file with our own scanline decoder and
//!    confirming level-0 RGBA matches the input pixel-exactly.
//!
//! Optionally `exrheader` is invoked to confirm it reports `mipmap` as
//! the level mode (string match in stdout).
//!
//! All tests auto-skip when the reference binaries are absent.

use std::process::Command;

use oxideav_openexr::{
    encode_exr_tiled_rgba_float_mipmap_box_filter, mipmap_level_count_round_down, parse_exr,
    Compression,
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
    let dir = std::env::temp_dir().join(format!("oxideav-exr-mipmap-{tag}-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn make_image(w: u32, h: u32) -> Vec<f32> {
    let mut s = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            s.push(x as f32 / w as f32);
            s.push(y as f32 / h as f32);
            s.push(((x ^ y) as f32) * 0.01);
            s.push(1.0);
        }
    }
    s
}

fn run_exrmetrics_roundtrip(compression: Compression, tag: &str) {
    if !tool_available("exrmetrics") {
        eprintln!("exrmetrics not found, skipping mipmap-encoder validation ({tag})");
        return;
    }
    let w = 32u32;
    let h = 32u32;
    let samples = make_image(w, h);
    let our_bytes =
        encode_exr_tiled_rgba_float_mipmap_box_filter(w, h, &samples, compression, 16, 16)
            .expect("our mipmap encoder failed");

    let dir = tempdir(tag);
    let our_path = dir.join(format!("ours_{tag}.exr"));
    let conv_path = dir.join(format!("conv_{tag}.exr"));
    std::fs::write(&our_path, &our_bytes).unwrap();

    // Ask exrmetrics to re-emit the file as an uncompressed scanline EXR.
    // Note exrmetrics `--convert -z none` reads our tiled mipmap file (this
    // is the spec-conformance check) and writes a fresh uncompressed EXR.
    let output = Command::new("exrmetrics")
        .args([
            "--convert",
            "-z",
            "none",
            our_path.to_str().unwrap(),
            "-o",
            conv_path.to_str().unwrap(),
        ])
        .output()
        .expect("exrmetrics --convert spawn failed");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "exrmetrics --convert refused our mipmap file ({tag}): status={:?}\nstderr: {stderr}\nstdout: {stdout}",
            output.status
        );
    }

    // Decode the converted file with our scanline decoder and confirm
    // level-0 pixels match the input.
    let conv_bytes = std::fs::read(&conv_path).unwrap();
    let img = parse_exr(&conv_bytes).expect("parse_exr on converted file failed");
    assert_eq!(img.width(), w, "converted width mismatch ({tag})");
    assert_eq!(img.height(), h, "converted height mismatch ({tag})");

    // Pixel-exact check on level 0.
    let a = &img.planes[0].samples;
    let b = &img.planes[1].samples;
    let g = &img.planes[2].samples;
    let r = &img.planes[3].samples;
    for y in 0..h as usize {
        for x in 0..w as usize {
            let off = y * w as usize + x;
            assert!(
                (r[off] - samples[off * 4]).abs() < 1e-6,
                "R mismatch at ({x},{y}) ({tag})"
            );
            assert!(
                (g[off] - samples[off * 4 + 1]).abs() < 1e-6,
                "G mismatch at ({x},{y}) ({tag})"
            );
            assert!(
                (b[off] - samples[off * 4 + 2]).abs() < 1e-6,
                "B mismatch at ({x},{y}) ({tag})"
            );
            assert!(
                (a[off] - samples[off * 4 + 3]).abs() < 1e-6,
                "A mismatch at ({x},{y}) ({tag})"
            );
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mipmap_encoder_exrmetrics_accepts_none() {
    run_exrmetrics_roundtrip(Compression::None, "none");
}

#[test]
fn mipmap_encoder_exrmetrics_accepts_zip() {
    run_exrmetrics_roundtrip(Compression::Zip, "zip");
}

#[test]
fn mipmap_encoder_exrmetrics_accepts_zips() {
    run_exrmetrics_roundtrip(Compression::Zips, "zips");
}

#[test]
fn mipmap_encoder_exrmetrics_accepts_rle() {
    run_exrmetrics_roundtrip(Compression::Rle, "rle");
}

#[test]
fn mipmap_encoder_exrheader_reports_mipmap_levels() {
    if !tool_available("exrheader") {
        eprintln!("exrheader not found, skipping mipmap header check");
        return;
    }
    let w = 32u32;
    let h = 32u32;
    let samples = make_image(w, h);
    let our_bytes =
        encode_exr_tiled_rgba_float_mipmap_box_filter(w, h, &samples, Compression::Zip, 16, 16)
            .expect("our mipmap encoder failed");

    let dir = tempdir("header");
    let path = dir.join("mipmap.exr");
    std::fs::write(&path, &our_bytes).unwrap();

    let output = Command::new("exrheader")
        .arg(path.to_str().unwrap())
        .output()
        .expect("exrheader spawn failed");
    assert!(
        output.status.success(),
        "exrheader failed on our file: status={:?}\nstderr: {}\nstdout: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    // exrheader prints the tiles attribute. We just confirm it
    // recognises the file as a tiled-mipmap file (string match on
    // "mipmap" or "MIPMAP_LEVELS"; both occur in CLI versions).
    let lower = stdout.to_lowercase();
    assert!(
        lower.contains("mipmap"),
        "exrheader output did not mention mipmap: {stdout}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mipmap_pyramid_level_count_helper() {
    // Sanity check on the public helper.
    assert_eq!(mipmap_level_count_round_down(32, 32), 6);
    assert_eq!(mipmap_level_count_round_down(16, 8), 5);
    assert_eq!(mipmap_level_count_round_down(1, 1), 1);
    assert_eq!(mipmap_level_count_round_down(7, 7), 3); // 7→3→1
}
