//! PXR24 scanline-encode validation.
//!
//! PXR24 is lossy for FLOAT channels: each binary32 sample is reduced to
//! a 24-bit representation (round mantissa to 15 bits, drop the low byte)
//! before the byte-plane delta + zlib stage. HALF/UINT are preserved.
//!
//! Three layers of coverage:
//!
//! 1. A pure self round-trip: encode a FLOAT RGBA scanline EXR as PXR24
//!    with our encoder, decode it back with our PXR24 reader, and assert
//!    every sample equals the spec's 24-bit reduction of the original.
//!    This proves the encoder's byte-plane + horizontal-delta + 24-bit
//!    reduction is the exact inverse of the (separately reference-
//!    validated) decoder, with no external tool needed.
//!
//! 2. A reference cross-check: encode PXR24 with our encoder, ask the
//!    reference `exrmetrics` CLI to transcode our file to ZIP, then
//!    decode the ZIP result with our (reference-validated) ZIP reader and
//!    assert it still equals the spec 24-bit reduction. This proves the
//!    reference decoder accepts our PXR24 bytes and recovers the same
//!    pixels. Auto-skips when `exrmetrics` is unavailable.
//!
//! 3. A raw-fallback check: encode incompressible random FLOAT data as
//!    PXR24 and confirm it still round-trips (the encoder must store the
//!    reorganised stream uncompressed when deflate doesn't shrink it).

use std::process::Command;

use oxideav_openexr::{encode_exr_scanline_rgba_float_with, parse_exr, Compression};

/// Mirror of the PXR24 24-bit float reduction (observer-spec §1.1):
/// returns the binary32 value a FLOAT sample collapses to after PXR24
/// encode+decode.
fn pxr24_reduce(f: f32) -> f32 {
    let w = f.to_bits();
    let s = w & 0x8000_0000;
    let e = w & 0x7f80_0000;
    let m = w & 0x007f_ffff;
    let code = if e == 0x7f80_0000 {
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
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-pxr24enc-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Build a deterministic FLOAT sample buffer spread across several
/// decades so the 24-bit reduction is exercised over many exponents.
fn ramp_samples(w: u32, h: u32) -> Vec<f32> {
    (0..(w * h * 4) as usize)
        .map(|i| (i as f32) * 0.013 + 0.001)
        .collect()
}

/// Assert each plane sample of `img` equals the spec 24-bit reduction of
/// the corresponding original RGBA sample.
fn assert_planes_reduced(img: &oxideav_openexr::ExrImage, samples: &[f32], w: u32, h: u32) {
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
}

/// Layer 1 — our encode, our decode.
fn self_roundtrip(w: u32, h: u32) {
    let samples = ramp_samples(w, h);
    let bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::Pxr24).unwrap();
    let img = parse_exr(&bytes).unwrap();
    assert_eq!(
        img.compression,
        Compression::Pxr24,
        "re-parsed file should report PXR24"
    );
    assert_planes_reduced(&img, &samples, w, h);
}

#[test]
fn pxr24_encode_self_roundtrip_single_block() {
    // 8x8 fits in a single 16-line PXR24 block.
    self_roundtrip(8, 8);
}

#[test]
fn pxr24_encode_self_roundtrip_multi_block_odd_width() {
    // 13x40 spans three 16-line blocks with a non-power-of-two width,
    // exercising the per-block boundary and short final block.
    self_roundtrip(13, 40);
}

#[test]
fn pxr24_encode_raw_fallback_roundtrips() {
    // Incompressible data: a simple xorshift PRNG over the full f32 bit
    // pattern domain (avoiding inf/NaN exponents so the reduction has a
    // single fixed point and the round-trip is exact to the reduction).
    let (w, h) = (16u32, 16u32);
    let mut state: u32 = 0x1234_5678;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // Keep exponent in [1, 0xfd] so we get ordinary finite floats.
        let e = (state & 0xff).clamp(1, 0xfd);
        let bits = (state & 0x807f_ffff) | (e << 23);
        f32::from_bits(bits)
    };
    let samples: Vec<f32> = (0..(w * h * 4) as usize).map(|_| next()).collect();
    let bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::Pxr24).unwrap();
    let img = parse_exr(&bytes).unwrap();
    assert_planes_reduced(&img, &samples, w, h);
}

/// Layer 2 — our encode, reference decode (via exrmetrics transcode to
/// ZIP, then our reference-validated ZIP reader).
fn reference_cross_check(w: u32, h: u32) {
    if !exrmetrics_available() {
        eprintln!("exrmetrics not available, skipping PXR24 encode cross-check ({w}x{h})");
        return;
    }
    let samples = ramp_samples(w, h);
    let bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::Pxr24).unwrap();

    let dir = tempdir();
    let in_path = dir.join("ours_pxr24.exr");
    let out_path = dir.join("ref_zip.exr");
    std::fs::write(&in_path, &bytes).unwrap();

    let output = Command::new("exrmetrics")
        .arg("--convert")
        .arg("-z")
        .arg("zip")
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
        panic!("exrmetrics failed to read our PXR24 / write ZIP:\n{stderr}");
    }

    let zip_bytes = std::fs::read(&out_path).unwrap();
    let img = parse_exr(&zip_bytes).unwrap();
    assert_eq!(img.compression, Compression::Zip);
    assert_planes_reduced(&img, &samples, w, h);

    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn pxr24_encode_reference_accepts_single_block() {
    reference_cross_check(8, 8);
}

#[test]
fn pxr24_encode_reference_accepts_multi_block() {
    reference_cross_check(13, 40);
}
