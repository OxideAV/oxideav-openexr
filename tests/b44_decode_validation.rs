//! B44 / B44A scanline-decode validation (observer-spec §2).
//!
//! B44 is a fixed-ratio lossy compressor for `HALF` channels (`FLOAT` /
//! `UINT` are copied raw). Because the quantisation is lossy, the
//! ground-truth a B44 file decodes to is whatever the reference encoder
//! produced — so the validation compares our B44 decode against the
//! reference's own B44 decode, surfaced by re-converting the same B44 file
//! back to `NONE` with the reference CLI and reading that with our
//! (already-validated) uncompressed reader.
//!
//! Flow:
//!   1. Encode a HALF-channel scanline EXR with our encoder (NONE).
//!   2. `exrmetrics --convert -z b44`  → a B44 file.
//!   3. `exrmetrics --convert -z none` (from the B44 file) → the reference
//!      decode of that B44 file, losslessly re-stored.
//!   4. Decode the B44 file with our new reader and assert every HALF
//!      sample bit-matches the reference decode.
//!
//! Auto-skips when `exrmetrics` is unavailable. A pure unit layer checks
//! the inverse-log table sentinels without any external tool.

use std::process::Command;

use oxideav_openexr::types::{Channel, PixelType};
use oxideav_openexr::{encode_exr_scanline, parse_exr, Compression};

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
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-b44-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn convert(input: &std::path::Path, out: &std::path::Path, z: &str) -> bool {
    let output = Command::new("exrmetrics")
        .arg("--convert")
        .arg("-z")
        .arg(z)
        .arg(input)
        .arg("-o")
        .arg(out)
        .output();
    match output {
        Ok(o) if o.status.success() => true,
        Ok(o) => {
            eprintln!(
                "exrmetrics -z {z} failed:\n{}",
                String::from_utf8_lossy(&o.stderr)
            );
            false
        }
        Err(e) => {
            eprintln!("exrmetrics spawn failed ({e})");
            false
        }
    }
}

/// Build a HALF-channel EXR (A,B,G,R) with our NONE encoder, transcode it
/// to B44 via the reference, then assert our B44 decode matches the
/// reference's own B44 decode (surfaced by a B44→NONE re-conversion).
///
/// `p_linear` flags the colour channels as perceptually linear, which
/// engages the inverse-log dequantisation table on decode. Note B44A is
/// only exercised with `p_linear == false`: OpenEXR 3.4.x's B44A decoder
/// zeroes pLinear channels (its plain-B44 decoder of the identical data
/// does not), so a pLinear B44A "reference" decode is not a valid oracle.
/// The pLinear log-table path is covered through the plain-B44 cases,
/// where the reference is self-consistent.
fn external_b44_roundtrip(
    w: u32,
    h: u32,
    scheme: &str,
    p_linear: bool,
    gen: impl Fn(usize, usize) -> f32,
) {
    if !exrmetrics_available() {
        eprintln!("exrmetrics not available, skipping B44 round-trip ({w}x{h})");
        return;
    }

    let mk = |name: &str, pl: bool| Channel {
        name: name.to_string(),
        pixel_type: PixelType::Half,
        p_linear: pl,
        x_sampling: 1,
        y_sampling: 1,
    };
    // A is always a data channel (non-linear); B/G/R follow `p_linear`.
    let channels = vec![
        mk("A", false),
        mk("B", p_linear),
        mk("G", p_linear),
        mk("R", p_linear),
    ];

    let pixels = (w * h) as usize;
    // One plane per channel in alphabetical order A, B, G, R.
    let planes_owned: Vec<Vec<f32>> = (0..4)
        .map(|ci| (0..pixels).map(|i| gen(i, ci)).collect())
        .collect();
    let planes: Vec<&[f32]> = planes_owned.iter().map(|v| v.as_slice()).collect();

    let attrs = {
        // Reuse the encoder's default attribute set shape by encoding once
        // and re-reading; simpler to build directly here.
        use oxideav_openexr::types::{Attribute, AttributeValue, Box2i, LineOrder};
        let win = Box2i {
            x_min: 0,
            y_min: 0,
            x_max: (w - 1) as i32,
            y_max: (h - 1) as i32,
        };
        vec![
            Attribute {
                name: "channels".to_string(),
                value: AttributeValue::Channels(channels.clone()),
            },
            Attribute {
                name: "compression".to_string(),
                value: AttributeValue::Compression(Compression::None),
            },
            Attribute {
                name: "dataWindow".to_string(),
                value: AttributeValue::Box2i(win),
            },
            Attribute {
                name: "displayWindow".to_string(),
                value: AttributeValue::Box2i(win),
            },
            Attribute {
                name: "lineOrder".to_string(),
                value: AttributeValue::LineOrder(LineOrder::IncreasingY),
            },
            Attribute {
                name: "pixelAspectRatio".to_string(),
                value: AttributeValue::Float(1.0),
            },
            Attribute {
                name: "screenWindowCenter".to_string(),
                value: AttributeValue::V2f(0.0, 0.0),
            },
            Attribute {
                name: "screenWindowWidth".to_string(),
                value: AttributeValue::Float(1.0),
            },
        ]
    };

    let bytes = encode_exr_scanline(w, h, &channels, &planes, Compression::None, attrs).unwrap();

    let dir = tempdir();
    let in_path = dir.join("in.exr");
    let b44_path = dir.join("b44.exr");
    let ref_path = dir.join("ref_none.exr");
    std::fs::write(&in_path, &bytes).unwrap();

    if !convert(&in_path, &b44_path, scheme) {
        eprintln!("skipping: could not produce {scheme} file");
        return;
    }
    if !convert(&b44_path, &ref_path, "none") {
        eprintln!("skipping: could not re-convert {scheme} -> none");
        return;
    }

    // Our decode of the B44 file.
    let b44_bytes = std::fs::read(&b44_path).unwrap();
    let ours = parse_exr(&b44_bytes).unwrap();
    let want_compression = match scheme {
        "b44" => Compression::B44,
        "b44a" => Compression::B44a,
        _ => unreachable!(),
    };
    assert_eq!(
        ours.compression, want_compression,
        "decoded file should report {scheme}"
    );

    // Reference's own decode of the same B44 file, re-stored as NONE.
    let ref_bytes = std::fs::read(&ref_path).unwrap();
    let reference = parse_exr(&ref_bytes).unwrap();
    assert_eq!(reference.compression, Compression::None);

    let wu = w as usize;
    for name in ["A", "B", "G", "R"] {
        let ours_plane = ours
            .planes
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("our decode missing plane {name}"));
        let ref_plane = reference
            .planes
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("reference decode missing plane {name}"));
        for y in 0..h as usize {
            for x in 0..wu {
                let off = y * wu + x;
                let got = ours_plane.samples[off];
                let want = ref_plane.samples[off];
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{scheme} {name}[{x},{y}] ours={got} reference={want}"
                );
            }
        }
    }

    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&b44_path);
    let _ = std::fs::remove_file(&ref_path);
    let _ = std::fs::remove_dir(&dir);
}

/// A smooth HDR gradient that exercises the logarithmic quantisation over
/// a range of exponents, with a per-channel offset so the four planes
/// differ.
fn gradient(i: usize, ci: usize) -> f32 {
    let base = (i as f32) * 0.05 + 0.01;
    base * (1.0 + ci as f32 * 0.37)
}

/// A field with large constant runs to trigger B44A 3-byte flat blocks.
fn flat_regions(i: usize, ci: usize) -> f32 {
    // Blocks of 64 identical samples, value depends on the block index.
    let block = i / 64;
    ((block % 5) as f32) * 2.0 + ci as f32 * 0.25
}

#[test]
fn b44_decode_single_block_plinear() {
    // 8x8 fits in a single 32-line B44 chunk. pLinear engages the inverse
    // log dequantisation table on R/G/B.
    external_b44_roundtrip(8, 8, "b44", true, gradient);
}

#[test]
fn b44_decode_odd_size_edge_replication() {
    // 13x37: neither dimension is a multiple of 4, exercising the right-
    // column / bottom-row edge replication, and 37 rows spans two B44
    // chunks (32 + 5). pLinear on.
    external_b44_roundtrip(13, 37, "b44", true, gradient);
}

#[test]
fn b44_decode_nonlinear() {
    // Non-pLinear channels bypass the log table — the unpacked HALF code
    // is used directly.
    external_b44_roundtrip(16, 16, "b44", false, gradient);
}

#[test]
fn b44a_decode_flat_blocks() {
    // Constant regions trigger B44A's 3-byte flat-block special case.
    // Non-pLinear (the reference's B44A pLinear decode is unreliable).
    external_b44_roundtrip(16, 16, "b44a", false, flat_regions);
}

#[test]
fn b44a_decode_gradient() {
    // B44A on a gradient (no flat blocks) must still match the reference.
    external_b44_roundtrip(20, 20, "b44a", false, gradient);
}
