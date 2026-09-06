//! Luminance/chroma (`Y` / `RY` / `BY`) colour reconstruction validated
//! against a reference EXR tool as an opaque process.
//!
//! The tool used is `exr2aces`: it reads an image through the
//! reference's RGB(A) interface — so a luminance/chroma file is
//! reconstructed to RGB by the reference's own conversion — converts the
//! primaries to the ACES container space, and writes the result in the
//! *same* channel layout it read (a `Y RY BY` input comes back as
//! `Y RY BY` under the ACES luminance weights, with a `chromaticities`
//! attribute naming the new primaries). Feeding it both an RGB file and
//! our luminance/chroma encoding of the same pixels therefore gives two
//! independent views of one image after an identical primaries
//! conversion:
//!
//! 1. **Definition** (filter-independent) — for an image whose chroma
//!    is constant (`RGB = k · (r0, g0, b0)` with only `k` varying) the
//!    `RY` / `BY` ratios are the same at every pixel, so whatever
//!    reconstruction / reduction filter the reference applies, its
//!    `RY` / `BY` output must equal the ratios we compute from its RGB
//!    output. This pins the `(R − Y) / Y` form and the
//!    chromaticities-derived weights (the ACES weights are far from
//!    BT.709: a negative blue weight).
//! 2. **Reconstruction** (tolerance) — on a smooth gradient our decode
//!    of the reference-written `Y RY BY` must match the reference's RGB
//!    output of the RGB file to a colour-level tolerance. The two
//!    sides use different chroma filters (ours is bilinear / tent), so
//!    this is a tolerance check, tightest in the interior.
//! 3. **Acceptance** — the reference accepts our luminance/chroma
//!    files (the tool exits 0 and `exrheader` reports the channels).
//!
//! All checks auto-skip when the binaries are absent.

use std::path::{Path, PathBuf};
use std::process::Command;

use oxideav_openexr::{
    encode_exr_scanline, luma_chroma_to_rgb, luminance_weights, luminance_weights_of, parse_exr,
    rgb_to_luma_chroma, Attribute, AttributeValue, Box2i, Channel, ChromaPlane, Compression,
    ExrImage, LineOrder, PixelType, BT709_CHROMATICITIES,
};

fn tool_available(name: &str) -> bool {
    Command::new(name)
        .arg("--help")
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

fn tempdir() -> PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-lumachroma-{nanos}-{pid}-{seq}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn exr2aces(input: &Path, output: &Path) -> bool {
    let out = Command::new("exr2aces")
        .arg(input)
        .arg(output)
        .output()
        .expect("spawn exr2aces");
    if !out.status.success() {
        eprintln!("exr2aces failed: {}", String::from_utf8_lossy(&out.stderr));
    }
    out.status.success()
}

fn ch(name: &str, pt: PixelType, sx: i32, sy: i32) -> Channel {
    Channel {
        name: name.to_string(),
        pixel_type: pt,
        p_linear: false,
        x_sampling: sx,
        y_sampling: sy,
    }
}

fn attrs(w: u32, h: u32, chs: &[Channel]) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (w - 1) as i32,
        y_max: (h - 1) as i32,
    };
    vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(chs.to_vec()),
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
}

fn plane<'a>(img: &'a ExrImage, name: &str) -> &'a [f32] {
    let idx = img
        .channels
        .iter()
        .position(|c| c.name == name)
        .unwrap_or_else(|| panic!("channel {name} missing"));
    &img.planes[idx].samples
}

/// Write `rgb.exr` (B G R) and `yc.exr` (BY RY Y, chroma at 2×2) for
/// the given planes and return their paths.
fn write_pair(dir: &Path, w: u32, h: u32, r: &[f32], g: &[f32], b: &[f32]) -> (PathBuf, PathBuf) {
    let wts = luminance_weights(&BT709_CHROMATICITIES);
    let yc = rgb_to_luma_chroma(w, h, [r, g, b], wts, (2, 2)).unwrap();
    let chs = vec![
        ch("BY", PixelType::Float, 2, 2),
        ch("RY", PixelType::Float, 2, 2),
        ch("Y", PixelType::Float, 1, 1),
    ];
    let yc_bytes = encode_exr_scanline(
        w,
        h,
        &chs,
        &[&yc.by, &yc.ry, &yc.y],
        Compression::None,
        attrs(w, h, &chs),
    )
    .unwrap();
    let chs = vec![
        ch("B", PixelType::Float, 1, 1),
        ch("G", PixelType::Float, 1, 1),
        ch("R", PixelType::Float, 1, 1),
    ];
    let rgb_bytes =
        encode_exr_scanline(w, h, &chs, &[b, g, r], Compression::None, attrs(w, h, &chs)).unwrap();
    let yc_path = dir.join("yc.exr");
    let rgb_path = dir.join("rgb.exr");
    std::fs::write(&yc_path, yc_bytes).unwrap();
    std::fs::write(&rgb_path, rgb_bytes).unwrap();
    (rgb_path, yc_path)
}

fn read(path: &Path) -> ExrImage {
    parse_exr(&std::fs::read(path).unwrap()).unwrap()
}

#[test]
fn reference_chroma_ratios_match_ours_for_constant_chroma() {
    if !tool_available("exr2aces") {
        eprintln!("exr2aces not available, skipping");
        return;
    }
    let (w, h) = (24u32, 12u32);
    let n = (w * h) as usize;
    // Constant chroma, varying luminance: RGB = k · (0.6, 0.35, 0.8).
    let k = |i: usize| 0.05 + 1.5 * (i as f32) / (n as f32);
    let r: Vec<f32> = (0..n).map(|i| 0.6 * k(i)).collect();
    let g: Vec<f32> = (0..n).map(|i| 0.35 * k(i)).collect();
    let b: Vec<f32> = (0..n).map(|i| 0.8 * k(i)).collect();
    let dir = tempdir();
    let (rgb_path, yc_path) = write_pair(&dir, w, h, &r, &g, &b);
    let rgb_out = dir.join("rgb_aces.exr");
    let yc_out = dir.join("yc_aces.exr");
    assert!(exr2aces(&rgb_path, &rgb_out));
    assert!(exr2aces(&yc_path, &yc_out));
    let ra = read(&rgb_out);
    let ya = read(&yc_out);
    assert_eq!(
        ya.channels
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        ["BY", "RY", "Y"]
    );
    assert_eq!(
        (ya.channels[0].x_sampling, ya.channels[0].y_sampling),
        (2, 2)
    );

    // The reference tagged its output with the ACES primaries; the
    // weights derived from that attribute must be far from BT.709.
    let wa = luminance_weights_of(&ya.attributes);
    assert!(wa[2] < 0.0, "ACES blue weight is negative: {wa:?}");

    // Our ratios from the reference's RGB output (full resolution).
    let mine = rgb_to_luma_chroma(
        w,
        h,
        [plane(&ra, "R"), plane(&ra, "G"), plane(&ra, "B")],
        wa,
        (1, 1),
    )
    .unwrap();
    let (ry_ref, by_ref, y_ref) = (plane(&ya, "RY"), plane(&ya, "BY"), plane(&ya, "Y"));
    let pw = (w / 2) as usize;
    for jy in 0..(h / 2) as usize {
        for jx in 0..pw {
            let j = jy * pw + jx;
            let i = (2 * jy) * w as usize + 2 * jx;
            // HALF output: ~3 significant decimal digits.
            assert!(
                (ry_ref[j] - mine.ry[i]).abs() <= 2e-3 * (1.0 + mine.ry[i].abs()),
                "RY sample {j}: reference {} vs ours {}",
                ry_ref[j],
                mine.ry[i]
            );
            assert!(
                (by_ref[j] - mine.by[i]).abs() <= 2e-3 * (1.0 + mine.by[i].abs()),
                "BY sample {j}: reference {} vs ours {}",
                by_ref[j],
                mine.by[i]
            );
        }
    }
    for (i, (&got, &expect)) in y_ref.iter().zip(&mine.y).enumerate() {
        assert!(
            (got - expect).abs() <= 2e-3 * (1.0 + expect.abs()),
            "Y px{i}: reference {got} vs ours {expect}"
        );
    }
    // Full decode of the reference-written file equals the reference's
    // RGB output (constant chroma ⇒ no filter dependence).
    let dec = luma_chroma_to_rgb(
        w,
        h,
        y_ref,
        ChromaPlane {
            samples: ry_ref,
            x_sampling: 2,
            y_sampling: 2,
        },
        ChromaPlane {
            samples: by_ref,
            x_sampling: 2,
            y_sampling: 2,
        },
        wa,
    )
    .unwrap();
    for i in 0..n {
        for (name, got, exp) in [
            ("R", dec.r[i], plane(&ra, "R")[i]),
            ("G", dec.g[i], plane(&ra, "G")[i]),
            ("B", dec.b[i], plane(&ra, "B")[i]),
        ] {
            assert!(
                (got - exp).abs() <= 4e-3 * (1.0 + exp.abs()),
                "{name} px{i}: ours {got} vs reference {exp}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn smooth_gradient_reconstructs_within_colour_tolerance() {
    if !tool_available("exr2aces") {
        eprintln!("exr2aces not available, skipping");
        return;
    }
    let (w, h) = (32u32, 16u32);
    let n = (w * h) as usize;
    let mut r = vec![0f32; n];
    let mut g = vec![0f32; n];
    let mut b = vec![0f32; n];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = y * w as usize + x;
            let fx = x as f32 / (w - 1) as f32;
            let fy = y as f32 / (h - 1) as f32;
            r[i] = 0.1 + 0.8 * fx;
            g[i] = 0.2 + 0.6 * fy;
            b[i] = 0.9 - 0.7 * fx * fy;
        }
    }
    let dir = tempdir();
    let (rgb_path, yc_path) = write_pair(&dir, w, h, &r, &g, &b);
    let rgb_out = dir.join("rgb_aces.exr");
    let yc_out = dir.join("yc_aces.exr");
    assert!(exr2aces(&rgb_path, &rgb_out));
    assert!(exr2aces(&yc_path, &yc_out));
    let ra = read(&rgb_out);
    let ya = read(&yc_out);
    let wa = luminance_weights_of(&ya.attributes);
    let dec = luma_chroma_to_rgb(
        w,
        h,
        plane(&ya, "Y"),
        ChromaPlane {
            samples: plane(&ya, "RY"),
            x_sampling: 2,
            y_sampling: 2,
        },
        ChromaPlane {
            samples: plane(&ya, "BY"),
            x_sampling: 2,
            y_sampling: 2,
        },
        wa,
    )
    .unwrap();
    // Luminance is filter-independent: tight.
    let y_ref = plane(&ya, "Y");
    for (i, &got) in y_ref.iter().enumerate() {
        let expect =
            wa[0] * plane(&ra, "R")[i] + wa[1] * plane(&ra, "G")[i] + wa[2] * plane(&ra, "B")[i];
        assert!((got - expect).abs() <= 5e-3, "Y px{i}: {got} vs {expect}");
    }
    // Colour: interior pixels within 0.02, edges (where the two chroma
    // filters disagree most) within 0.1, on data in [0, 1].
    for i in 0..n {
        let (x, y) = (i % w as usize, i / w as usize);
        let interior = x >= 2 && x + 2 < w as usize && y >= 2 && y + 2 < h as usize;
        let tol = if interior { 0.02 } else { 0.1 };
        for (name, got, exp) in [
            ("R", dec.r[i], plane(&ra, "R")[i]),
            ("G", dec.g[i], plane(&ra, "G")[i]),
            ("B", dec.b[i], plane(&ra, "B")[i]),
        ] {
            assert!(
                (got - exp).abs() <= tol,
                "{name} px({x},{y}): ours {got} vs reference {exp} (tol {tol})"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn reference_header_tool_reports_our_luma_chroma_layout() {
    if !tool_available("exr2aces") || !tool_available("exrheader") {
        eprintln!("reference tools not available, skipping");
        return;
    }
    let (w, h) = (8u32, 6u32);
    let n = (w * h) as usize;
    let r: Vec<f32> = (0..n).map(|i| (i % 8) as f32 / 8.0).collect();
    let g: Vec<f32> = (0..n).map(|i| (i / 8) as f32 / 6.0).collect();
    let b: Vec<f32> = (0..n).map(|_| 0.5).collect();
    let dir = tempdir();
    let (_rgb_path, yc_path) = write_pair(&dir, w, h, &r, &g, &b);
    let out = Command::new("exrheader").arg(&yc_path).output().unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("BY, 32-bit floating-point, sampling 2 2"),
        "{text}"
    );
    assert!(
        text.contains("RY, 32-bit floating-point, sampling 2 2"),
        "{text}"
    );
    assert!(
        text.contains("Y, 32-bit floating-point, sampling 1 1"),
        "{text}"
    );
    assert!(exr2aces(&yc_path, &dir.join("accepted.exr")));
    let _ = std::fs::remove_dir_all(&dir);
}
