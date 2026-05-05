//! Validate that our encoded output parses cleanly through the
//! `exrheader` binary if it's present on $PATH. The test is auto-skipped
//! (with a printed reason) when the binary is missing — that's fine for
//! CI machines without an OpenEXR install.
//!
//! `exrheader` is used as an opaque oracle: we don't read its source,
//! we just check it exits zero and emits text that mentions our four
//! channels. Per workspace policy, no ILM/Academy code is consulted.

use std::process::Command;

use oxideav_openexr::{encode_exr_scanline_rgba_float_with, Compression};

fn exrheader_available() -> bool {
    Command::new("exrheader")
        .arg("--help")
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

fn run_exrheader_on(bytes: &[u8]) {
    let dir = tempdir();
    let path = format!("{dir}/oxideav-openexr-test.exr");
    std::fs::write(&path, bytes).unwrap();

    let output = Command::new("exrheader")
        .arg(&path)
        .output()
        .expect("exrheader spawn failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "exrheader returned non-zero on our output\nstdout: {stdout}\nstderr: {stderr}"
    );

    // Look for evidence the four canonical RGBA channels survived.
    for ch in ["R", "G", "B", "A"] {
        assert!(
            stdout.contains(&format!("{ch},")) || stdout.contains(&format!("{ch} ")),
            "exrheader output missing channel '{ch}'\nstdout: {stdout}"
        );
    }

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

fn tempdir() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-test-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

#[test]
fn exrheader_accepts_zip_output() {
    if !exrheader_available() {
        eprintln!("exrheader not available on PATH, skipping validation");
        return;
    }
    let w = 8;
    let h = 8;
    let samples: Vec<f32> = (0..(w * h * 4)).map(|i| (i as f32) * 0.01).collect();
    let bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::Zip).unwrap();
    run_exrheader_on(&bytes);
}

#[test]
fn exrheader_accepts_no_compression_output() {
    if !exrheader_available() {
        eprintln!("exrheader not available on PATH, skipping validation");
        return;
    }
    let w = 4;
    let h = 4;
    let samples: Vec<f32> = (0..(w * h * 4)).map(|i| (i as f32) * 0.05).collect();
    let bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::None).unwrap();
    run_exrheader_on(&bytes);
}
