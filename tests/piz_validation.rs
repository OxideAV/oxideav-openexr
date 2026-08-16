//! PIZ scanline validation (observer-spec `openexr-piz-dwa-observer-spec.md` §2).
//!
//! Three axes, all lossless and therefore bit-exact:
//!
//! 1. Self round-trip: our PIZ encoder's bytes decode back through our
//!    own decoder to the identical samples.
//! 2. Reference-encoded decode: a reference conversion tool (invoked as
//!    an opaque process) re-encodes one of our NONE files with PIZ; our
//!    decoder must recover the identical samples.
//! 3. Our-encoded reference decode: the reference tool converts our PIZ
//!    file back to NONE; parsing that must again give identical samples.
//!
//! The external-tool tests auto-skip when `exrmetrics` is missing.

use std::process::Command;

use oxideav_openexr::{
    encode_exr_scanline, encode_exr_scanline_rgba_float_with, parse_exr, Attribute, AttributeValue,
    Box2i, Channel, Compression, LineOrder, PixelType,
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
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-piz-test-{nanos}-{pid}-{seq}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Convert `input_bytes` with the reference tool to the given
/// compression name; returns the converted bytes, or None if the tool is
/// unavailable.
fn reference_convert(input_bytes: &[u8], z: &str) -> Option<Vec<u8>> {
    if !exrmetrics_available() {
        eprintln!("exrmetrics not available, skipping");
        return None;
    }
    let dir = tempdir();
    let in_path = dir.join("in.exr");
    let out_path = dir.join("out.exr");
    std::fs::write(&in_path, input_bytes).unwrap();
    let output = Command::new("exrmetrics")
        .arg("--convert")
        .arg("-z")
        .arg(z)
        .arg(&in_path)
        .arg("-o")
        .arg(&out_path)
        .output()
        .ok()?;
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

fn assert_images_bit_exact(a: &oxideav_openexr::ExrImage, b: &oxideav_openexr::ExrImage) {
    assert_eq!(a.channels.len(), b.channels.len());
    for (ca, cb) in a.channels.iter().zip(b.channels.iter()) {
        assert_eq!(ca.name, cb.name);
    }
    for (pa, pb) in a.planes.iter().zip(b.planes.iter()) {
        assert_eq!(pa.samples.len(), pb.samples.len());
        for (i, (x, y)) in pa.samples.iter().zip(pb.samples.iter()).enumerate() {
            assert!(x.to_bits() == y.to_bits(), "sample {i} differs: {x} vs {y}");
        }
    }
}

/// Structured HDR-ish test pattern with runs, gradients and outliers —
/// exercises both wavelet variants' input regimes.
fn test_samples(w: u32, h: u32) -> Vec<f32> {
    let mut s = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            s.push((x as f32 / w as f32) * 4.0);
            s.push((y as f32 % 11.0) * 0.125);
            s.push(if (x + y) % 7 == 0 { 1000.5 } else { 0.0 });
            s.push(1.0);
        }
    }
    s
}

#[test]
fn piz_self_roundtrip_rgba_float() {
    for (w, h) in [(16u32, 16u32), (61, 37), (8, 100), (1, 1), (200, 3)] {
        let samples = test_samples(w, h);
        let bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::Piz).unwrap();
        let img = parse_exr(&bytes).unwrap();
        assert_eq!(img.compression, Compression::Piz);
        let wu = w as usize;
        for y in 0..h as usize {
            for x in 0..wu {
                let off = y * wu + x;
                assert_eq!(img.planes[3].samples[off], samples[off * 4], "R ({x},{y})");
                assert_eq!(img.planes[2].samples[off], samples[off * 4 + 1]);
                assert_eq!(img.planes[1].samples[off], samples[off * 4 + 2]);
                assert_eq!(img.planes[0].samples[off], samples[off * 4 + 3]);
            }
        }
    }
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

#[test]
fn piz_self_roundtrip_mixed_pixel_types() {
    let (w, h) = (45u32, 70u32);
    let n = (w * h) as usize;
    let chans = vec![
        ch("F", PixelType::Float),
        ch("H", PixelType::Half),
        ch("U", PixelType::Uint),
    ];
    let f: Vec<f32> = (0..n).map(|i| (i as f32) * 0.37 - 100.0).collect();
    let hp: Vec<f32> = (0..n).map(|i| ((i % 97) as f32) * 0.25).collect();
    let u: Vec<f32> = (0..n).map(|i| ((i * i) % 100_000) as f32).collect();
    let planes: Vec<&[f32]> = vec![&f, &hp, &u];
    let attrs = base_attrs(w, h, &chans, Compression::Piz);
    let bytes = encode_exr_scanline(w, h, &chans, &planes, Compression::Piz, attrs).unwrap();
    let img = parse_exr(&bytes).unwrap();
    for (i, &v) in f.iter().enumerate() {
        assert_eq!(img.planes[0].samples[i].to_bits(), v.to_bits(), "F {i}");
    }
    for (i, &v) in hp.iter().enumerate() {
        // HALF channel: value survives the f32->half->f32 ladder exactly
        // for these quarter-step inputs below the half range.
        assert_eq!(img.planes[1].samples[i], v, "H {i}");
    }
    for (i, &v) in u.iter().enumerate() {
        assert_eq!(img.planes[2].samples[i], v, "U {i}");
    }
}

#[test]
fn piz_self_roundtrip_subsampled_luma_chroma() {
    let (w, h) = (32u32, 20u32);
    let mut by = ch("BY", PixelType::Half);
    by.x_sampling = 2;
    by.y_sampling = 2;
    let mut ry = ch("RY", PixelType::Half);
    ry.x_sampling = 2;
    ry.y_sampling = 2;
    let yc = ch("Y", PixelType::Half);
    let chans = vec![by, ry, yc];
    let sw = (w as usize).div_ceil(2);
    let sh = (h as usize).div_ceil(2);
    let by_p: Vec<f32> = (0..sw * sh).map(|i| ((i % 31) as f32) * 0.0625).collect();
    let ry_p: Vec<f32> = (0..sw * sh).map(|i| ((i % 17) as f32) * -0.125).collect();
    let y_p: Vec<f32> = (0..(w * h) as usize)
        .map(|i| ((i % 41) as f32) * 0.03125)
        .collect();
    let planes: Vec<&[f32]> = vec![&by_p, &ry_p, &y_p];
    let attrs = base_attrs(w, h, &chans, Compression::Piz);
    let bytes = encode_exr_scanline(w, h, &chans, &planes, Compression::Piz, attrs).unwrap();
    let img = parse_exr(&bytes).unwrap();
    for (pi, src) in [&by_p, &ry_p, &y_p].iter().enumerate() {
        for (i, &v) in src.iter().enumerate() {
            assert_eq!(img.planes[pi].samples[i], v, "plane {pi} sample {i}");
        }
    }
}

#[test]
fn reference_piz_encoding_decodes_bit_exact() {
    let (w, h) = (61u32, 70u32);
    let samples = test_samples(w, h);
    let none_bytes =
        encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::None).unwrap();
    let Some(piz_bytes) = reference_convert(&none_bytes, "piz") else {
        return;
    };
    let img_none = parse_exr(&none_bytes).unwrap();
    let img_piz = parse_exr(&piz_bytes).unwrap();
    assert_eq!(img_piz.compression, Compression::Piz);
    assert_images_bit_exact(&img_none, &img_piz);
}

#[test]
fn our_piz_encoding_accepted_and_decoded_by_reference() {
    let (w, h) = (61u32, 70u32);
    let samples = test_samples(w, h);
    let piz_bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::Piz).unwrap();
    let Some(none_bytes) = reference_convert(&piz_bytes, "none") else {
        return;
    };
    let img_ours = parse_exr(&piz_bytes).unwrap();
    let img_ref = parse_exr(&none_bytes).unwrap();
    assert_images_bit_exact(&img_ours, &img_ref);
}

#[test]
fn reference_piz_mixed_types_decodes_bit_exact() {
    // FLOAT + HALF + UINT channels through the reference PIZ encoder:
    // exercises the two-words-per-sample wavelet interleave.
    let (w, h) = (40u32, 37u32);
    let n = (w * h) as usize;
    let chans = vec![
        ch("F", PixelType::Float),
        ch("H", PixelType::Half),
        ch("U", PixelType::Uint),
    ];
    let f: Vec<f32> = (0..n).map(|i| ((i % 13) as f32) * 0.03125).collect();
    let hp: Vec<f32> = (0..n).map(|i| ((i % 5) as f32) * 0.25).collect();
    let u: Vec<f32> = (0..n).map(|i| ((i * 7) % 9) as f32).collect();
    let planes: Vec<&[f32]> = vec![&f, &hp, &u];
    let attrs = base_attrs(w, h, &chans, Compression::None);
    let none_bytes = encode_exr_scanline(w, h, &chans, &planes, Compression::None, attrs).unwrap();
    let Some(piz_bytes) = reference_convert(&none_bytes, "piz") else {
        return;
    };
    let img_none = parse_exr(&none_bytes).unwrap();
    let img_piz = parse_exr(&piz_bytes).unwrap();
    assert_images_bit_exact(&img_none, &img_piz);
}

#[test]
fn piz_incompressible_chunk_takes_raw_fallback() {
    // Noise-like data with a dense value alphabet compresses poorly;
    // whether or not the encoder falls back, the round-trip must hold
    // and the reference must agree.
    let (w, h) = (13u32, 9u32);
    let mut samples = Vec::with_capacity((w * h * 4) as usize);
    let mut state = 0x12345678u32;
    for _ in 0..(w * h * 4) {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        samples.push(((state >> 8) as f32) / (1 << 24) as f32);
    }
    let piz_bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::Piz).unwrap();
    let img = parse_exr(&piz_bytes).unwrap();
    for (i, chunk) in samples.chunks_exact(4).enumerate() {
        assert_eq!(img.planes[3].samples[i], chunk[0]);
        assert_eq!(img.planes[2].samples[i], chunk[1]);
        assert_eq!(img.planes[1].samples[i], chunk[2]);
        assert_eq!(img.planes[0].samples[i], chunk[3]);
    }
    if let Some(none_bytes) = reference_convert(&piz_bytes, "none") {
        let img_ref = parse_exr(&none_bytes).unwrap();
        assert_images_bit_exact(&img, &img_ref);
    }
}
