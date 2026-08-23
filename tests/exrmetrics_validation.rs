//! Cross-validate that bytes our encoder writes (with each supported
//! compression mode) decode losslessly through the OpenEXR reference
//! `exrmetrics --convert -z none` pipeline back to a NO_COMPRESSION
//! file we can re-parse and compare against the input samples
//! bit-exactly.
//!
//! This catches the failure mode where our encoder's predictor /
//! interleave / RLE pipeline drifts from the reference encoder —
//! the self-roundtrip stays green but the bytes are not actually
//! spec-compliant. Auto-skips when `exrmetrics` is missing.

use std::process::Command;

use oxideav_openexr::{encode_exr_scanline_rgba_float_with, parse_exr, Compression};

fn exrmetrics_available() -> bool {
    Command::new("exrmetrics")
        .arg("--help")
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
    let dir = std::env::temp_dir().join(format!(
        "oxideav-openexr-exrmetrics-test-{nanos}-{pid}-{seq}"
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

fn run_external_roundtrip(z: Compression) {
    if !exrmetrics_available() {
        eprintln!("exrmetrics not available, skipping ({z:?})");
        return;
    }
    let w = 16u32;
    let h = 16u32;
    let samples: Vec<f32> = (0..(w * h * 4) as usize)
        .map(|i| (i as f32) * 0.0125)
        .collect();
    let bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, z).unwrap();

    let dir = tempdir();
    let in_path = format!("{dir}/in.exr");
    let out_path = format!("{dir}/out.exr");
    std::fs::write(&in_path, &bytes).unwrap();

    let output = Command::new("exrmetrics")
        .arg("--convert")
        .arg("-z")
        .arg("none")
        .arg(&in_path)
        .arg("-o")
        .arg(&out_path)
        .output();
    let output = match output {
        Ok(o) => o,
        Err(e) => {
            eprintln!("exrmetrics spawn failed ({e}); skipping ({z:?})");
            return;
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("exrmetrics rejected our {z:?} output:\n{stderr}");
    }
    let decoded_bytes = std::fs::read(&out_path).unwrap();
    let img = parse_exr(&decoded_bytes).unwrap();

    // Every sample must match exactly.
    let a = &img.planes[0].samples;
    let b = &img.planes[1].samples;
    let g = &img.planes[2].samples;
    let r = &img.planes[3].samples;
    let wu = w as usize;
    let hu = h as usize;
    for y in 0..hu {
        for x in 0..wu {
            let off = y * wu + x;
            assert_eq!(r[off], samples[off * 4], "{z:?} R mismatch ({x},{y})");
            assert_eq!(g[off], samples[off * 4 + 1], "{z:?} G mismatch ({x},{y})");
            assert_eq!(b[off], samples[off * 4 + 2], "{z:?} B mismatch ({x},{y})");
            assert_eq!(a[off], samples[off * 4 + 3], "{z:?} A mismatch ({x},{y})");
        }
    }

    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn exrmetrics_decodes_our_zip() {
    run_external_roundtrip(Compression::Zip);
}

#[test]
fn exrmetrics_decodes_our_zips() {
    run_external_roundtrip(Compression::Zips);
}

#[test]
fn exrmetrics_decodes_our_rle() {
    run_external_roundtrip(Compression::Rle);
}

#[test]
fn exrmetrics_decodes_our_none() {
    run_external_roundtrip(Compression::None);
}

/// Files using the two High-Throughput JPEG 2000 compression codes
/// (10 = 256-line, 11 = 32-line blocks) are beyond this crate's
/// compression matrix; a reference-produced HTJ2K file must fail with
/// a clean `Err` naming the unknown compression byte — never a panic,
/// a misdecode, or an attacker-sized allocation. Auto-skips when the
/// installed `exrmetrics` build cannot write HTJ2K.
#[test]
fn reference_htj2k_files_are_rejected_cleanly() {
    if !exrmetrics_available() {
        eprintln!("exrmetrics not available, skipping");
        return;
    }
    let w = 16u32;
    let h = 16u32;
    let samples: Vec<f32> = (0..(w * h * 4) as usize)
        .map(|i| (i as f32) * 0.0125)
        .collect();
    let none_bytes =
        encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::None).unwrap();

    for (mode, code) in [("htj2k256", 10u8), ("htj2k32", 11u8)] {
        let dir = tempdir();
        let in_path = format!("{dir}/in.exr");
        let out_path = format!("{dir}/out.exr");
        std::fs::write(&in_path, &none_bytes).unwrap();

        let output = Command::new("exrmetrics")
            .arg("--convert")
            .arg("-z")
            .arg(mode)
            .arg(&in_path)
            .arg("-o")
            .arg(&out_path)
            .output();
        let ok = matches!(&output, Ok(o) if o.status.success());
        if !ok {
            eprintln!("exrmetrics cannot write {mode}; skipping");
            let _ = std::fs::remove_file(&in_path);
            let _ = std::fs::remove_dir_all(&dir);
            continue;
        }
        let htj2k_bytes = std::fs::read(&out_path).unwrap();

        let err = parse_exr(&htj2k_bytes)
            .expect_err("an HTJ2K-compressed file must not parse as a supported one");
        let msg = format!("{err}");
        assert!(
            msg.contains(&format!("unknown compression byte {code}")),
            "{mode}: expected a clean unknown-compression rejection, got: {msg}"
        );

        let _ = std::fs::remove_file(&in_path);
        let _ = std::fs::remove_file(&out_path);
        let _ = std::fs::remove_dir(&dir);
    }
}
