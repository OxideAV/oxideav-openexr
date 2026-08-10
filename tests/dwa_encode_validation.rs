//! DWAA / DWAB encode validation (observer-spec
//! `openexr-piz-dwa-observer-spec.md` §3).
//!
//! Two anchors:
//!
//! 1. Self round-trip: our version-2 chunks decode through our own
//!    decoder; lossy channels within the quantisation tolerance,
//!    RLE / verbatim channels byte-exact.
//! 2. Reference decode: the reference tool (opaque process) converts
//!    our DWA file to NONE; its decode must agree with our decode of
//!    the same DWA bytes **bit-exactly** (both sides sit
//!    post-quantisation). Auto-skips when the tool is missing.

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
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-dwaenc-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

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
        "reference rejected our DWA file:\n{}",
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

/// Encode with the given compression, self-round-trip within `tol`
/// (relative to magnitude, lossy channels only), then have the
/// reference decode the same bytes and compare bit-exactly with ours.
#[allow(clippy::too_many_arguments)]
fn run_case(
    name: &str,
    chans: Vec<Channel>,
    planes: Vec<Vec<f32>>,
    lossless: Vec<bool>,
    w: u32,
    h: u32,
    z: Compression,
    tol: f32,
) {
    let refs: Vec<&[f32]> = planes.iter().map(|p| p.as_slice()).collect();
    let attrs = base_attrs(w, h, &chans, z);
    let dwa_bytes = encode_exr_scanline(w, h, &chans, &refs, z, attrs).unwrap();

    let ours = parse_exr(&dwa_bytes).unwrap();
    assert_eq!(ours.compression, z);
    for (pi, plane) in planes.iter().enumerate() {
        for (i, (&want, got)) in plane.iter().zip(ours.planes[pi].samples.iter()).enumerate() {
            if lossless[pi] {
                // RLE / verbatim channels: exact up to the declared
                // pixel-type ladder (HALF channels round once).
                let expected = match chans[pi].pixel_type {
                    PixelType::Half => {
                        oxideav_openexr::half::half_to_f32(oxideav_openexr::half::f32_to_half(want))
                    }
                    PixelType::Uint => (want + 0.5).floor(),
                    PixelType::Float => want,
                };
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "{name}: lossless channel {} sample {i}",
                    chans[pi].name
                );
            } else {
                assert!(
                    (got - want).abs() <= tol * want.abs().max(1.0),
                    "{name}: lossy channel {} sample {i}: got {got}, want {want}",
                    chans[pi].name
                );
            }
        }
    }

    // Reference decode of the identical bytes must agree with ours
    // bit-exactly.
    if let Some(none_bytes) = reference_convert(&dwa_bytes, "none") {
        let theirs = parse_exr(&none_bytes).unwrap();
        for (pi, (pa, pb)) in ours.planes.iter().zip(theirs.planes.iter()).enumerate() {
            for (i, (a, b)) in pa.samples.iter().zip(pb.samples.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name}: reference decode of OUR chunk differs at plane {pi} sample {i}: {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn our_dwaa_lone_y_is_reference_compatible() {
    let (w, h) = (64u32, 48u32);
    run_case(
        "lone Y",
        vec![ch("Y", PixelType::Half)],
        vec![smooth_plane(w, h, 3.0, 0.1)],
        vec![false],
        w,
        h,
        Compression::Dwaa,
        0.02,
    );
}

#[test]
fn our_dwaa_rgba_is_reference_compatible() {
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
        vec![true, false, false, false],
        w,
        h,
        Compression::Dwaa,
        0.05,
    );
}

#[test]
fn our_dwab_rgb_float_is_reference_compatible() {
    let (w, h) = (33u32, 300u32); // two DWAB chunks
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
        vec![false, false, false],
        w,
        h,
        Compression::Dwab,
        0.05,
    );
}

#[test]
fn our_dwaa_mixed_schemes_are_reference_compatible() {
    let (w, h) = (40u32, 40u32);
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
            (0..n).map(|i| ((i % 512) as f32 / 512.0) * 0.5).collect(),
            smooth_plane(w, h, 5.0, 0.2),
            (0..n).map(|i| (i as f32) * 0.125).collect(),
            (0..n).map(|i| ((i * 37) % 100_000) as f32).collect(),
        ],
        vec![true, false, true, true],
        w,
        h,
        Compression::Dwaa,
        0.05,
    );
}

#[test]
fn our_dwaa_odd_dims_edge_mirror_is_reference_compatible() {
    // Non-multiple-of-8 dims exercise the mirrored edge blocks.
    let (w, h) = (13u32, 11u32);
    run_case(
        "odd dims",
        vec![ch("Y", PixelType::Half)],
        vec![smooth_plane(w, h, 2.0, 0.3)],
        vec![false],
        w,
        h,
        Compression::Dwaa,
        0.05,
    );
}

#[test]
fn dwa_compression_level_attribute_is_honoured() {
    // A higher dwaCompressionLevel must not break decodability and
    // should not enlarge the file.
    let (w, h) = (64u32, 64u32);
    let chans = vec![ch("Y", PixelType::Half)];
    let plane = smooth_plane(w, h, 3.0, 0.1);
    let refs: Vec<&[f32]> = vec![&plane];

    let attrs_default = base_attrs(w, h, &chans, Compression::Dwaa);
    let bytes_default =
        encode_exr_scanline(w, h, &chans, &refs, Compression::Dwaa, attrs_default).unwrap();

    let mut attrs_coarse = base_attrs(w, h, &chans, Compression::Dwaa);
    attrs_coarse.push(Attribute {
        name: "dwaCompressionLevel".to_string(),
        value: AttributeValue::Float(500.0),
    });
    let bytes_coarse =
        encode_exr_scanline(w, h, &chans, &refs, Compression::Dwaa, attrs_coarse).unwrap();

    assert!(
        bytes_coarse.len() <= bytes_default.len(),
        "level 500 produced a larger file ({} vs {})",
        bytes_coarse.len(),
        bytes_default.len()
    );
    let img = parse_exr(&bytes_coarse).unwrap();
    for (i, &want) in plane.iter().enumerate() {
        let got = img.planes[0].samples[i];
        assert!(
            (got - want).abs() <= 0.2 * want.abs().max(1.0),
            "coarse decode too far off at {i}: {got} vs {want}"
        );
    }
}
