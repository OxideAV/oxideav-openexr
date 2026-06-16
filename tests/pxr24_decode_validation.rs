//! PXR24 scanline-decode validation.
//!
//! PXR24 is lossy for FLOAT channels: each binary32 sample is reduced to
//! a 24-bit representation (round mantissa to 15 bits, drop the low byte)
//! before the byte-plane delta + zlib stage. HALF/UINT are preserved.
//!
//! Two layers of coverage:
//!
//! 1. Pure unit checks of the 24-bit reduction against the observer-spec
//!    §1.1 worked-fact table (no external tool needed).
//! 2. A black-box round-trip: encode a FLOAT RGBA scanline EXR with our
//!    encoder, transcode it to PXR24 with the reference `exrmetrics`
//!    CLI (`-z pxr24`), decode the result with our new PXR24 reader, and
//!    assert every sample equals the spec's 24-bit reduction of the
//!    original. This proves our inflate + byte-plane prefix-sum + 24-bit
//!    reconstruction matches the reference encoder bit-for-bit. Auto-
//!    skips when `exrmetrics` is unavailable.

use std::process::Command;

use oxideav_openexr::{encode_exr_scanline_rgba_float, parse_exr};

/// Mirror of the PXR24 24-bit float reduction (observer-spec §1.1):
/// returns the binary32 value a FLOAT sample collapses to after PXR24
/// encode+decode.
fn pxr24_reduce(f: f32) -> f32 {
    let w = f.to_bits();
    let s = w & 0x8000_0000;
    let e = w & 0x7f80_0000;
    let m = w & 0x007f_ffff;
    let code = if e == 0x7f80_0000 {
        // inf / NaN
        if m == 0 {
            e >> 8
        } else {
            let mm = m >> 8;
            (e >> 8) | mm | u32::from(mm == 0)
        }
    } else {
        let mut i = ((e | m) + (m & 0x80)) >> 8;
        if i >= 0x7f8000 {
            i = (e | m) >> 8;
        }
        (s >> 8) | i
    };
    f32::from_bits((code & 0x00ff_ffff) << 8)
}

/// 24-bit hex code (top 24 bits of the reduced binary32) for a value.
fn pxr24_code(f: f32) -> u32 {
    pxr24_reduce(f).to_bits() >> 8
}

#[test]
fn pxr24_reduction_worked_facts() {
    // Observer-spec §1.1 worked-fact table (binary32 -> 24-bit hex).
    assert_eq!(pxr24_code(1.0), 0x3f8000, "1.0");
    assert_eq!(pxr24_code(0.5), 0x3f0000, "0.5");
    assert_eq!(pxr24_code(std::f32::consts::PI), 0x404910, "pi");
    assert_eq!(pxr24_code(65504.0), 0x477fe0, "65504");
    assert_eq!(pxr24_code(1e30), 0x7149f3, "1e30");
}

#[test]
fn pxr24_reduction_specials() {
    // Infinity: mantissa zero.
    assert_eq!(pxr24_code(f32::INFINITY), 0x7f8000);
    // NaN never collapses to infinity (low mantissa forced non-zero).
    let n = pxr24_code(f32::NAN);
    assert_eq!(n & 0x7f8000, 0x7f8000, "NaN keeps inf exponent");
    assert_ne!(n & 0x007fff, 0, "NaN keeps a mantissa bit");
    // Negative values fold the sign in unchanged.
    assert_eq!(pxr24_code(-1.0), 0xbf8000);
}

fn exrmetrics_available() -> bool {
    Command::new("exrmetrics")
        .arg("--help")
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

fn tempdir() -> std::path::PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-pxr24-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Encode a FLOAT RGBA scanline EXR, transcode it to PXR24 via the
/// reference CLI, decode it back, and check every sample equals the
/// spec 24-bit reduction of the original.
fn external_pxr24_roundtrip(w: u32, h: u32) {
    if !exrmetrics_available() {
        eprintln!("exrmetrics not available, skipping PXR24 round-trip ({w}x{h})");
        return;
    }

    // Spread values across a few decades so the FLOAT reduction is
    // exercised over a range of exponents, including sub-1.0 values.
    let samples: Vec<f32> = (0..(w * h * 4) as usize)
        .map(|i| (i as f32) * 0.013 + 0.001)
        .collect();
    let bytes = encode_exr_scanline_rgba_float(w, h, &samples).unwrap();

    let dir = tempdir();
    let in_path = dir.join("in.exr");
    let out_path = dir.join("pxr24.exr");
    std::fs::write(&in_path, &bytes).unwrap();

    let output = Command::new("exrmetrics")
        .arg("--convert")
        .arg("-z")
        .arg("pxr24")
        .arg(&in_path)
        .arg("-o")
        .arg(&out_path)
        .output();
    let output = match output {
        Ok(o) => o,
        Err(e) => {
            eprintln!("exrmetrics spawn failed ({e}); skipping");
            return;
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("exrmetrics failed to write PXR24:\n{stderr}");
    }

    let pxr_bytes = std::fs::read(&out_path).unwrap();
    let img = parse_exr(&pxr_bytes).unwrap();
    assert_eq!(
        img.compression,
        oxideav_openexr::Compression::Pxr24,
        "decoded file should report PXR24"
    );

    let wu = w as usize;
    for (ci, name) in ["R", "G", "B", "A"].iter().enumerate() {
        let plane = img
            .planes
            .iter()
            .find(|p| &p.name == name)
            .unwrap_or_else(|| panic!("missing plane {name}"));
        for y in 0..h as usize {
            for x in 0..wu {
                let off = y * wu + x;
                let got = plane.samples[off];
                let want = pxr24_reduce(samples[off * 4 + ci]);
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "PXR24 {name}[{x},{y}] got={got} want={want} (orig={})",
                    samples[off * 4 + ci]
                );
            }
        }
    }

    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn pxr24_decode_single_block() {
    // 8x8 fits in a single 16-line PXR24 block.
    external_pxr24_roundtrip(8, 8);
}

#[test]
fn pxr24_decode_multi_block_odd_width() {
    // 13x40 spans three 16-line blocks with a non-power-of-two width,
    // exercising the per-block boundary and short final block.
    external_pxr24_roundtrip(13, 40);
}
