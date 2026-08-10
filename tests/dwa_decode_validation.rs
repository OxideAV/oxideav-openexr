//! DWAA / DWAB decode validation against reference-produced files
//! (observer-spec `openexr-piz-dwa-observer-spec.md` §3).
//!
//! Strategy: write a NONE file with our encoder, convert it to DWAA /
//! DWAB with a reference tool (opaque process), then convert THAT file
//! back to NONE with the same tool. Our decode of the DWA file must
//! agree with the reference's own decode of the identical bytes — both
//! sides sit post-quantisation, so the comparison is bit-exact if our
//! IDCT / LUT / CSC pipeline reproduces the reference arithmetic.
//! Auto-skips when the tool is missing.

use std::process::Command;

use oxideav_openexr::{
    encode_exr_scanline, parse_exr, Attribute, AttributeValue, Box2i, Channel, Compression,
    LineOrder, PixelType,
};

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
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-dwadec-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn reference_convert(input_bytes: &[u8], z: &str, level: Option<u32>) -> Option<Vec<u8>> {
    if !exrmetrics_available() {
        eprintln!("exrmetrics not available, skipping");
        return None;
    }
    let dir = tempdir();
    let in_path = dir.join("in.exr");
    let out_path = dir.join("out.exr");
    std::fs::write(&in_path, input_bytes).unwrap();
    let mut cmd = Command::new("exrmetrics");
    cmd.arg("--convert").arg("-z").arg(z);
    if let Some(l) = level {
        cmd.arg("-l").arg(l.to_string());
    }
    let output = cmd.arg(&in_path).arg("-o").arg(&out_path).output().ok()?;
    assert!(
        output.status.success(),
        "reference conversion to {z} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let out = std::fs::read(&out_path).unwrap();
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    let _ = std::fs::remove_dir(&dir);
    Some(out)
}

fn ch(name: &str, pt: PixelType) -> Channel {
    Channel {
        name: name.to_string(),
        pixel_type: pt,
        p_linear: false,
        x_sampling: 1,
        y_sampling: 1,
    }
}

fn base_attrs(w: u32, h: u32, chans: &[Channel], z: Compression) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (w - 1) as i32,
        y_max: (h - 1) as i32,
    };
    vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(chans.to_vec()),
        },
        Attribute {
            name: "compression".to_string(),
            value: AttributeValue::Compression(z),
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
}

/// Compare our decode of the DWA bytes against the reference's own
/// re-encode to NONE of the same bytes. Returns (samples, mismatches,
/// max_abs_diff).
fn compare_against_reference_decode(dwa_bytes: &[u8], none_of_dwa: &[u8]) -> (usize, usize, f32) {
    let ours = parse_exr(dwa_bytes).unwrap();
    let theirs = parse_exr(none_of_dwa).unwrap();
    assert_eq!(ours.planes.len(), theirs.planes.len());
    let mut total = 0usize;
    let mut bad = 0usize;
    let mut max_diff = 0.0f32;
    for (pa, pb) in ours.planes.iter().zip(theirs.planes.iter()) {
        assert_eq!(pa.samples.len(), pb.samples.len());
        for (a, b) in pa.samples.iter().zip(pb.samples.iter()) {
            total += 1;
            if a.to_bits() != b.to_bits() {
                bad += 1;
                max_diff = max_diff.max((a - b).abs());
            }
        }
    }
    (total, bad, max_diff)
}

fn run_case(name: &str, chans: Vec<Channel>, planes: Vec<Vec<f32>>, w: u32, h: u32, z: &str) {
    let refs: Vec<&[f32]> = planes.iter().map(|p| p.as_slice()).collect();
    let attrs = base_attrs(w, h, &chans, Compression::None);
    let none_bytes = encode_exr_scanline(w, h, &chans, &refs, Compression::None, attrs).unwrap();
    let Some(dwa_bytes) = reference_convert(&none_bytes, z, None) else {
        return;
    };
    let Some(none_of_dwa) = reference_convert(&dwa_bytes, "none", None) else {
        return;
    };
    let (total, bad, max_diff) = compare_against_reference_decode(&dwa_bytes, &none_of_dwa);
    assert_eq!(
        bad, 0,
        "{name}: {bad}/{total} samples differ from the reference decode (max diff {max_diff})"
    );
}

/// Smooth HDR-ish plane exercising both LUT branches (values below and
/// above 1.0).
fn smooth_plane(w: u32, h: u32, scale: f32, offset: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            out.push(offset + scale * (fx * fx + 0.5 * fy + 0.25 * fx * fy));
        }
    }
    out
}

#[test]
fn reference_dwaa_lone_y_decodes_bit_exact() {
    let (w, h) = (64u32, 48u32);
    run_case(
        "lone Y",
        vec![ch("Y", PixelType::Half)],
        vec![smooth_plane(w, h, 3.0, 0.1)],
        w,
        h,
        "dwaa",
    );
}

#[test]
fn reference_dwaa_rgba_decodes_bit_exact() {
    let (w, h) = (70u32, 45u32);
    run_case(
        "RGBA",
        vec![
            ch("A", PixelType::Half),
            ch("B", PixelType::Half),
            ch("G", PixelType::Half),
            ch("R", PixelType::Half),
        ],
        vec![
            smooth_plane(w, h, 1.0, 0.0),
            smooth_plane(w, h, 2.0, 0.25),
            smooth_plane(w, h, 1.5, 0.1),
            smooth_plane(w, h, 0.8, 0.4),
        ],
        w,
        h,
        "dwaa",
    );
}

#[test]
fn reference_dwab_rgb_float_decodes_bit_exact() {
    let (w, h) = (33u32, 300u32); // > 256 rows: two DWAB chunks
    run_case(
        "RGB float dwab",
        vec![
            ch("B", PixelType::Float),
            ch("G", PixelType::Float),
            ch("R", PixelType::Float),
        ],
        vec![
            smooth_plane(w, h, 4.0, 0.0),
            smooth_plane(w, h, 2.0, 0.5),
            smooth_plane(w, h, 1.0, 0.05),
        ],
        w,
        h,
        "dwab",
    );
}

#[test]
fn reference_dwaa_noise_stress_decodes_bit_exact() {
    // Noisy content with negatives and 100+ blocks: hammers the rounding
    // boundaries of the IDCT / CSC / LUT chain far harder than the
    // smooth planes.
    let (w, h) = (96u32, 64u32);
    let n = (w * h) as usize;
    let mut state = 0x2468_ACE1u32;
    let mut rnd = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        ((state >> 8) as f32 / (1 << 24) as f32) * 8.0 - 2.0
    };
    let r: Vec<f32> = (0..n).map(|_| rnd()).collect();
    let g: Vec<f32> = (0..n).map(|_| rnd()).collect();
    let b: Vec<f32> = (0..n).map(|_| rnd()).collect();
    run_case(
        "noise RGB",
        vec![
            ch("B", PixelType::Half),
            ch("G", PixelType::Half),
            ch("R", PixelType::Half),
        ],
        vec![b, g, r],
        w,
        h,
        "dwaa",
    );
}

#[test]
fn reference_dwaa_mixed_verbatim_rle_decodes_bit_exact() {
    let (w, h) = (40u32, 40u32);
    // Y (lossy) + A (RLE) + Z FLOAT (verbatim) + id UINT (verbatim).
    let n = (w * h) as usize;
    run_case(
        "mixed schemes",
        vec![
            ch("A", PixelType::Half),
            ch("Y", PixelType::Half),
            ch("Z", PixelType::Float),
            ch("id", PixelType::Uint),
        ],
        vec![
            (0..n).map(|i| (i % 512) as f32 / 512.0).collect(),
            smooth_plane(w, h, 5.0, 0.2),
            (0..n).map(|i| (i as f32) * 0.125).collect(),
            (0..n).map(|i| ((i * 37) % 100_000) as f32).collect(),
        ],
        w,
        h,
        "dwaa",
    );
}
