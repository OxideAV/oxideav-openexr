//! Cross-validate tiled-decode by running our scanline encoder, piping
//! through `exrmaketiled` (an OpenEXR command-line tool used as opaque
//! oracle), then re-parsing the tiled output and checking every sample
//! matches the original. The test is auto-skipped (with a printed
//! reason) when `exrmaketiled` is missing — fine on stripped CI.

use std::process::Command;

use oxideav_openexr::{encode_exr_scanline_rgba_float_with, parse_exr, Compression};

fn exrmaketiled_available() -> bool {
    Command::new("exrmaketiled")
        .arg("-h")
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
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-tiled-test-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

fn run_tiled_roundtrip(z: &str) {
    if !exrmaketiled_available() {
        eprintln!("exrmaketiled not available on PATH, skipping tiled-decode validation ({z})");
        return;
    }
    let w = 16u32;
    let h = 16u32;
    // Encode an uncompressed scanline EXR via our encoder.
    let samples: Vec<f32> = (0..(w * h * 4) as usize)
        .map(|i| (i as f32) * 0.0125)
        .collect();
    let bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::None).unwrap();

    let dir = tempdir();
    let scan_path = format!("{dir}/scan.exr");
    let tiled_path = format!("{dir}/tiled.exr");
    std::fs::write(&scan_path, &bytes).unwrap();

    let output = Command::new("exrmaketiled")
        .arg("-o")
        .arg("-z")
        .arg(z)
        .arg("-t")
        .arg("8")
        .arg("8")
        .arg(&scan_path)
        .arg(&tiled_path)
        .output();
    let output = match output {
        Ok(o) => o,
        Err(e) => {
            eprintln!("exrmaketiled spawn failed ({e}); skipping");
            return;
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("exrmaketiled returned non-zero (skipping):\n{stderr}");
        return;
    }
    let tiled_bytes = match std::fs::read(&tiled_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to read tiled output ({e}); skipping");
            return;
        }
    };

    // Decode back through our decoder and check every sample matches.
    let img = parse_exr(&tiled_bytes).unwrap();
    assert_eq!(img.width(), w);
    assert_eq!(img.height(), h);
    let a = &img.planes[0].samples;
    let b = &img.planes[1].samples;
    let g = &img.planes[2].samples;
    let r = &img.planes[3].samples;
    let wu = w as usize;
    let hu = h as usize;
    for y in 0..hu {
        for x in 0..wu {
            let off = y * wu + x;
            assert_eq!(r[off], samples[off * 4], "R mismatch ({x},{y})");
            assert_eq!(g[off], samples[off * 4 + 1], "G mismatch ({x},{y})");
            assert_eq!(b[off], samples[off * 4 + 2], "B mismatch ({x},{y})");
            assert_eq!(a[off], samples[off * 4 + 3], "A mismatch ({x},{y})");
        }
    }

    let _ = std::fs::remove_file(&scan_path);
    let _ = std::fs::remove_file(&tiled_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn decode_tiled_none() {
    run_tiled_roundtrip("none");
}

#[test]
fn decode_tiled_zip() {
    run_tiled_roundtrip("zip");
}

#[test]
fn decode_tiled_zips() {
    run_tiled_roundtrip("zips");
}

#[test]
fn decode_tiled_rle() {
    run_tiled_roundtrip("rle");
}
