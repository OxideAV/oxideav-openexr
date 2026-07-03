//! Sub-sampled (luminance/chroma) channel layouts through the lossy
//! compressors — PXR24, B44, B44A (round 385).
//!
//! The round-73 sub-sampled scanline coverage stopped at the lossless
//! schemes (NONE / ZIP / ZIPS / RLE). The observer-spec's §0 framing and
//! §2.1 chunk reorganisation both account for `y_sampling`, and the
//! encoder / decoder already thread the sampling factors through the
//! PXR24 byte-plane builder and the B44 per-channel plane gather — this
//! suite pins that behaviour:
//!
//! 1. **PXR24 + FLOAT 4:2:0** — self-roundtrip must equal the §1.1
//!    24-bit reduction (computed independently here from the spec) on
//!    every plane, full-res and sub-sampled alike.
//! 2. **B44 / B44A + HALF luminance/chroma** (`Y` at 1×1, `BY` / `RY`
//!    at 2×2, `pLinear` on the classic layout) — decode must be a fixed
//!    point (decode → re-encode → decode is bit-stable).
//! 3. **Reference cross-checks** — `exrmetrics` transcodes our
//!    sub-sampled PXR24 / B44 / B44A bytes to NONE and the reference
//!    decode must bit-match our own decode (auto-skips when the binary
//!    is missing). `exrheader` must report the sampling factors.
//!
//! Chunk shapes exercised: single-chunk (h ≤ lines/block), multi-chunk
//! with a partial trailing block, and odd (non-multiple-of-sampling,
//! non-multiple-of-4) dimensions.

use std::process::Command;

use oxideav_openexr::{
    encode_exr_scanline, parse_exr, Attribute, AttributeValue, Box2i, Channel, Compression,
    ExrImage, LineOrder, PixelType,
};

fn exrmetrics_available() -> bool {
    Command::new("exrmetrics")
        .arg("--help")
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

fn exrheader_available() -> bool {
    Command::new("exrheader")
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
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-sublossy-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `exrmetrics --convert -z <z>` as an opaque process.
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
        Ok(o) => {
            if !o.status.success() {
                eprintln!(
                    "exrmetrics -z {z} failed:\n{}",
                    String::from_utf8_lossy(&o.stderr)
                );
            }
            o.status.success()
        }
        Err(e) => {
            eprintln!("exrmetrics spawn failed ({e})");
            false
        }
    }
}

/// Ceil-divide an image dimension by a sampling factor (matches the
/// number of sample columns/rows a channel contributes when the data
/// window starts at 0: coordinates divisible by the factor).
fn sub_dim(d: u32, s: u32) -> u32 {
    d.div_ceil(s)
}

/// Independent implementation of the observer-spec §1.1 PXR24 FLOAT
/// 24-bit reduction, used as the expected-value oracle.
fn pxr24_reduce(v: f32) -> f32 {
    let bits = v.to_bits();
    let sign = bits & 0x8000_0000;
    let em = bits & 0x7fff_ffff; // exponent | mantissa
    let e = bits & 0x7f80_0000;
    let m = bits & 0x007f_ffff;
    let i = if e == 0x7f80_0000 {
        if m == 0 {
            // Infinity: mantissa zero.
            em >> 8
        } else {
            // NaN: keep top 15 mantissa bits, force non-zero.
            let m15 = m >> 8;
            (e >> 8) | m15 | u32::from(m15 == 0)
        }
    } else {
        // Finite: add the round bit and shift; on exponent overflow
        // redo by truncation.
        let r = (em + (m & 0x80)) >> 8;
        if r >= 0x7f_8000 {
            em >> 8
        } else {
            r
        }
    };
    f32::from_bits(((sign >> 8) | i) << 8)
}

/// Build a luminance/chroma channel list: `Y` full-res plus `BY` / `RY`
/// sub-sampled by `(sx, sy)`. Alphabetical order (the file order) is
/// BY, RY, Y.
fn luma_chroma_channels(pt: PixelType, p_linear: bool, sx: i32, sy: i32) -> Vec<Channel> {
    vec![
        Channel {
            name: "BY".to_string(),
            pixel_type: pt,
            p_linear,
            x_sampling: sx,
            y_sampling: sy,
        },
        Channel {
            name: "RY".to_string(),
            pixel_type: pt,
            p_linear,
            x_sampling: sx,
            y_sampling: sy,
        },
        Channel {
            name: "Y".to_string(),
            pixel_type: pt,
            p_linear,
            x_sampling: 1,
            y_sampling: 1,
        },
    ]
}

fn base_attrs(w: u32, h: u32, channels: &[Channel], compression: Compression) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (w - 1) as i32,
        y_max: (h - 1) as i32,
    };
    vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(channels.to_vec()),
        },
        Attribute {
            name: "compression".to_string(),
            value: AttributeValue::Compression(compression),
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

/// Generate luma (full-res) + chroma (sub-sampled) planes. `flat` makes
/// the chroma planes constant so B44A's 3-byte blocks fire.
fn make_planes(w: u32, h: u32, sx: u32, sy: u32, flat: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut y_plane = Vec::with_capacity((w * h) as usize);
    for yy in 0..h {
        for xx in 0..w {
            // Positive HDR-ish gradient (pLinear-friendly).
            y_plane.push(0.25 + (xx as f32) * 0.375 + (yy as f32) * 1.5);
        }
    }
    let cw = sub_dim(w, sx);
    let ch = sub_dim(h, sy);
    let mut by = Vec::with_capacity((cw * ch) as usize);
    let mut ry = Vec::with_capacity((cw * ch) as usize);
    for cy in 0..ch {
        for cx in 0..cw {
            if flat {
                by.push(0.75);
                ry.push(2.5);
            } else {
                by.push(0.5 + (cx as f32) * 0.25 + (cy as f32) * 0.0625);
                ry.push(8.0 - (cx as f32) * 0.5 + (cy as f32) * 0.125);
            }
        }
    }
    (by, ry, y_plane)
}

fn encode_luma_chroma(
    w: u32,
    h: u32,
    compression: Compression,
    pt: PixelType,
    p_linear: bool,
    flat: bool,
) -> Vec<u8> {
    let channels = luma_chroma_channels(pt, p_linear, 2, 2);
    let (by, ry, y_plane) = make_planes(w, h, 2, 2, flat);
    let planes: Vec<&[f32]> = vec![by.as_slice(), ry.as_slice(), y_plane.as_slice()];
    let attrs = base_attrs(w, h, &channels, compression);
    encode_exr_scanline(w, h, &channels, &planes, compression, attrs).unwrap()
}

fn plane<'a>(img: &'a ExrImage, name: &str) -> &'a [f32] {
    &img.planes
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("plane {name} missing"))
        .samples
}

fn assert_planes_bitmatch(a: &ExrImage, b: &ExrImage, w: u32, h: u32, ctx: &str) {
    for name in ["BY", "RY", "Y"] {
        let pa = plane(a, name);
        let pb = plane(b, name);
        let (pw, ph) = if name == "Y" {
            (w, h)
        } else {
            (sub_dim(w, 2), sub_dim(h, 2))
        };
        assert_eq!(pa.len(), (pw * ph) as usize, "{ctx}: {name} plane size");
        assert_eq!(pa.len(), pb.len(), "{ctx}: {name} plane size mismatch");
        for i in 0..pa.len() {
            assert_eq!(
                pa[i].to_bits(),
                pb[i].to_bits(),
                "{ctx}: {name}[{i}] {} vs {}",
                pa[i],
                pb[i]
            );
        }
    }
}

// ---------------------------------------------------------------------
// 1. PXR24 + FLOAT 4:2:0 — exact against the independent §1.1 oracle.
// ---------------------------------------------------------------------

fn pxr24_yuv420_matches_reduction(w: u32, h: u32) {
    let bytes = encode_luma_chroma(w, h, Compression::Pxr24, PixelType::Float, false, false);
    let img = parse_exr(&bytes).unwrap();
    assert_eq!(img.compression, Compression::Pxr24);

    let (by_src, ry_src, y_src) = make_planes(w, h, 2, 2, false);
    let cases = [("BY", &by_src), ("RY", &ry_src), ("Y", &y_src)];
    for (name, src) in cases {
        let got = plane(&img, name);
        assert_eq!(got.len(), src.len(), "{name} plane size {w}x{h}");
        for (i, (&g, &s)) in got.iter().zip(src.iter()).enumerate() {
            let want = pxr24_reduce(s);
            assert_eq!(
                g.to_bits(),
                want.to_bits(),
                "{name}[{i}] ({w}x{h}): got {g}, want reduce({s}) = {want}"
            );
        }
    }
}

#[test]
fn pxr24_yuv420_single_chunk() {
    // 16 rows = exactly one PXR24 chunk.
    pxr24_yuv420_matches_reduction(16, 16);
}

#[test]
fn pxr24_yuv420_partial_trailing_chunk() {
    // 16 + 4 rows: the trailing chunk covers 4 image rows / 2 chroma rows.
    pxr24_yuv420_matches_reduction(8, 20);
}

#[test]
fn pxr24_yuv420_odd_dims() {
    // Odd width AND height: chroma planes are ceil(13/2) x ceil(9/2) = 7x5.
    pxr24_yuv420_matches_reduction(13, 9);
}

// ---------------------------------------------------------------------
// 2. B44 / B44A + HALF luminance/chroma — decode is a fixed point.
// ---------------------------------------------------------------------

fn b44_luma_chroma_fixed_point(w: u32, h: u32, scheme: Compression, p_linear: bool, flat: bool) {
    let bytes = encode_luma_chroma(w, h, scheme, PixelType::Half, p_linear, flat);
    let img1 = parse_exr(&bytes).unwrap();
    assert_eq!(img1.compression, scheme);

    // Re-encode the decoded planes and re-decode: bit-stable pixels.
    let channels = luma_chroma_channels(PixelType::Half, p_linear, 2, 2);
    let by = plane(&img1, "BY").to_vec();
    let ry = plane(&img1, "RY").to_vec();
    let y_plane = plane(&img1, "Y").to_vec();
    let planes: Vec<&[f32]> = vec![by.as_slice(), ry.as_slice(), y_plane.as_slice()];
    let attrs = base_attrs(w, h, &channels, scheme);
    let bytes2 = encode_exr_scanline(w, h, &channels, &planes, scheme, attrs).unwrap();
    let img2 = parse_exr(&bytes2).unwrap();
    assert_planes_bitmatch(
        &img1,
        &img2,
        w,
        h,
        &format!("{scheme:?} fixed-point {w}x{h}"),
    );
}

#[test]
fn b44_luma_chroma_single_chunk_plinear() {
    // 32 rows = one B44 chunk; chroma contributes 16 rows of its plane.
    b44_luma_chroma_fixed_point(16, 32, Compression::B44, true, false);
}

#[test]
fn b44_luma_chroma_two_chunks_plinear() {
    // 40 rows = 32 + 8: the chroma plane splits 16 + 4 across chunks.
    b44_luma_chroma_fixed_point(16, 40, Compression::B44, true, false);
}

#[test]
fn b44_luma_chroma_odd_dims_nonlinear() {
    // 13x9: chroma plane 7x5 — both planes need 4x4 edge replication.
    b44_luma_chroma_fixed_point(13, 9, Compression::B44, false, false);
}

#[test]
fn b44a_luma_chroma_flat_chroma_nonlinear() {
    // Constant chroma: B44A's 3-byte flat blocks fire on the
    // sub-sampled planes while the luma gradient stays 14-byte packed.
    b44_luma_chroma_fixed_point(16, 32, Compression::B44a, false, true);
}

#[test]
fn b44a_flat_chroma_compresses_below_b44() {
    // The flat chroma planes must actually shrink under B44A relative
    // to plain B44 (3-byte vs 14-byte blocks on the same content).
    let b44 = encode_luma_chroma(32, 32, Compression::B44, PixelType::Half, false, true);
    let b44a = encode_luma_chroma(32, 32, Compression::B44a, PixelType::Half, false, true);
    assert!(
        b44a.len() < b44.len(),
        "B44A ({}) should be smaller than B44 ({}) on flat chroma",
        b44a.len(),
        b44.len()
    );
}

// ---------------------------------------------------------------------
// 3. Reference cross-checks (auto-skip without the binaries).
// ---------------------------------------------------------------------

fn reference_bitmatch(w: u32, h: u32, scheme: Compression, pt: PixelType, p_linear: bool) {
    if !exrmetrics_available() {
        eprintln!("exrmetrics not available, skipping {scheme:?} sub-sampled cross-check");
        return;
    }
    let bytes = encode_luma_chroma(w, h, scheme, pt, p_linear, false);
    let ours = parse_exr(&bytes).unwrap();

    let dir = tempdir();
    let in_path = dir.join("ours.exr");
    let none_path = dir.join("ref_none.exr");
    std::fs::write(&in_path, &bytes).unwrap();
    assert!(
        convert(&in_path, &none_path, "none"),
        "reference could not read our sub-sampled {scheme:?} file ({w}x{h})"
    );
    let ref_bytes = std::fs::read(&none_path).unwrap();
    let reference = parse_exr(&ref_bytes).unwrap();
    assert_eq!(reference.compression, Compression::None);
    assert_planes_bitmatch(
        &ours,
        &reference,
        w,
        h,
        &format!("reference {scheme:?} {w}x{h}"),
    );

    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&none_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn pxr24_yuv420_reference_bitmatch() {
    reference_bitmatch(16, 20, Compression::Pxr24, PixelType::Float, false);
}

#[test]
fn b44_luma_chroma_reference_bitmatch_plinear() {
    reference_bitmatch(16, 40, Compression::B44, PixelType::Half, true);
}

/// Pin an observed reference-validator constraint: the reference reader
/// refuses to *open* a sub-sampled file whose data-window extent is not
/// a multiple of the sampling factor (13x9 with 2x2 chroma), for every
/// compression scheme. Our own reader is a permissive superset (the
/// odd-dims self-roundtrips above decode ceil-sized chroma planes), but
/// files meant for reference-tool consumption must keep sub-sampled
/// extents divisible by the sampling factor. If a future reference
/// version starts accepting these files, this test fails and the
/// cross-checks above should gain odd-dims cases.
#[test]
fn reference_rejects_odd_extent_subsampled() {
    if !exrmetrics_available() {
        eprintln!("exrmetrics not available, skipping");
        return;
    }
    for (scheme, pt) in [
        (Compression::Pxr24, PixelType::Float),
        (Compression::B44, PixelType::Half),
    ] {
        let bytes = encode_luma_chroma(13, 9, scheme, pt, false, false);
        let dir = tempdir();
        let in_path = dir.join("odd.exr");
        let none_path = dir.join("odd_none.exr");
        std::fs::write(&in_path, &bytes).unwrap();
        assert!(
            !convert(&in_path, &none_path, "none"),
            "reference unexpectedly accepted an odd-extent sub-sampled {scheme:?} file"
        );
        let _ = std::fs::remove_file(&in_path);
        let _ = std::fs::remove_file(&none_path);
        let _ = std::fs::remove_dir(&dir);
    }
}

#[test]
fn b44a_luma_chroma_reference_bitmatch_nonlinear() {
    // Non-pLinear on B44A: the reference B44A decoder zeroes pLinear
    // channels (see README), so the cross-check runs on data channels.
    reference_bitmatch(16, 32, Compression::B44a, PixelType::Half, false);
}

#[test]
fn exrheader_reports_sampling_on_b44_luma_chroma() {
    if !exrheader_available() {
        eprintln!("exrheader not available, skipping");
        return;
    }
    let bytes = encode_luma_chroma(16, 32, Compression::B44, PixelType::Half, true, false);
    let dir = tempdir();
    let in_path = dir.join("lc.exr");
    std::fs::write(&in_path, &bytes).unwrap();
    let output = Command::new("exrheader")
        .arg(&in_path)
        .output()
        .expect("exrheader spawn");
    assert!(
        output.status.success(),
        "exrheader rejected our sub-sampled B44 file:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    for name in ["BY", "RY", "Y"] {
        assert!(text.contains(name), "channel {name} missing:\n{text}");
    }
    // The header dump must reflect the 2x2 sampling on the chroma
    // channels (rendered as "sampling 2 2").
    assert!(
        text.contains("sampling 2 2"),
        "2x2 sampling not reported:\n{text}"
    );
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_dir(&dir);
}
