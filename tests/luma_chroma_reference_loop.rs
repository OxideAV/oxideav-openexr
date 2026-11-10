//! Closed-loop black-box check of the luminance/chroma decoder AND
//! encoder through a reference tool: `exr2aces` reads an image through
//! the reference RGBA interface (a `Y RY BY` file is decoded with the
//! reference's reconstruction, conversion and saturation limiting),
//! converts it to the ACES primaries and writes it back in the layout it
//! read (re-encoded by the reference luminance/chroma writer). Feeding
//! it a file already tagged with the ACES primaries makes the primaries
//! conversion an identity, so its output's stored planes can be compared
//! with `encode(decode(input))` computed by this crate: the `Y` plane
//! must match bit-for-bit; the ratio planes are bounded (the writer's
//! chroma path beyond the traced 16×16 fixture is not characterised).
//!
//! Auto-skips when the binary is absent.

use std::path::PathBuf;
use std::process::Command;

use oxideav_openexr::{
    Attribute, AttributeValue, Box2i, Channel, ChromaPlane, Chromaticities, Compression, LineOrder,
    PixelType, encode_exr_scanline, luma_chroma_to_rgb, luminance_weights, parse_exr,
    rgb_to_luma_chroma,
};

const AP0: Chromaticities = Chromaticities {
    red_x: 0.7347,
    red_y: 0.2653,
    green_x: 0.0,
    green_y: 1.0,
    blue_x: 0.0001,
    blue_y: -0.077,
    white_x: 0.32168,
    white_y: 0.33767,
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
    let dir = std::env::temp_dir().join(format!(
        "oxideav-openexr-ycloop-{nanos}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Deterministic pseudo-random in [0, 1).
fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32) / (1u64 << 24) as f32
}

struct Loop {
    exact: usize,
    total: usize,
    max_abs: f32,
    y_exact: usize,
    y_total: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// Smooth colour gradients.
    Smooth,
    /// Gradients plus per-pixel noise (the saturation limiter fires).
    Noisy,
    /// One chromaticity, luminance ramp.
    ConstantChroma,
}

fn run_loop(dir: &std::path::Path, w: u32, h: u32, seed: u64, kind: Kind) -> Loop {
    let noisy = kind == Kind::Noisy;
    let n = (w * h) as usize;
    let mut s = seed;
    let mut r = vec![0f32; n];
    let mut g = vec![0f32; n];
    let mut b = vec![0f32; n];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = y * w as usize + x;
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            if kind == Kind::ConstantChroma {
                let k = 0.2 + fx + 0.5 * fy;
                (r[i], g[i], b[i]) = (0.6 * k, 0.5 * k, 0.3 * k);
                continue;
            }
            let jitter = if noisy { 0.3 * lcg(&mut s) } else { 0.0 };
            r[i] = 0.2 + 0.7 * fx + jitter;
            g[i] = 0.3 + 0.5 * fy + if noisy { 0.2 * lcg(&mut s) } else { 0.0 };
            b[i] = 0.8 - 0.6 * fx * fy + if noisy { 0.4 * lcg(&mut s) } else { 0.0 };
        }
    }
    let wts = luminance_weights(&AP0);
    let yc = rgb_to_luma_chroma(w, h, [&r, &g, &b], wts, (2, 2)).unwrap();
    let ch = |name: &str, s: i32| Channel::new(name.to_string(), PixelType::Half, false, s, s);
    let chs = vec![ch("BY", 2), ch("RY", 2), ch("Y", 1)];
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: w as i32 - 1,
        y_max: h as i32 - 1,
    };
    let attrs = vec![
        Attribute::new("channels".into(), AttributeValue::Channels(chs.clone())),
        Attribute::new(
            "compression".into(),
            AttributeValue::Compression(Compression::None),
        ),
        Attribute::new("dataWindow".into(), AttributeValue::Box2i(win)),
        Attribute::new("displayWindow".into(), AttributeValue::Box2i(win)),
        Attribute::new(
            "lineOrder".into(),
            AttributeValue::LineOrder(LineOrder::IncreasingY),
        ),
        Attribute::new("pixelAspectRatio".into(), AttributeValue::Float(1.0)),
        Attribute::new("screenWindowCenter".into(), AttributeValue::V2f(0.0, 0.0)),
        Attribute::new("screenWindowWidth".into(), AttributeValue::Float(1.0)),
        Attribute::new("chromaticities".into(), AttributeValue::Chromaticities(AP0)),
    ];
    let bytes = encode_exr_scanline(
        w,
        h,
        &chs,
        &[&yc.by, &yc.ry, &yc.y],
        Compression::None,
        attrs,
    )
    .unwrap();
    let input = dir.join(format!("in-{w}x{h}-{seed}.exr"));
    let output = dir.join(format!("out-{w}x{h}-{seed}.exr"));
    std::fs::write(&input, bytes).unwrap();
    let st = Command::new("exr2aces")
        .arg(&input)
        .arg(&output)
        .status()
        .unwrap();
    assert!(st.success());
    let out = parse_exr(&std::fs::read(&output).unwrap()).unwrap();
    // Ours: decode our file, re-encode the decoded RGB.
    let rgb = luma_chroma_to_rgb(
        w,
        h,
        &yc.y,
        ChromaPlane::new(&yc.ry, 2, 2),
        ChromaPlane::new(&yc.by, 2, 2),
        wts,
    )
    .unwrap();
    let again = rgb_to_luma_chroma(w, h, [&rgb.r, &rgb.g, &rgb.b], wts, (2, 2)).unwrap();
    let plane = |name: &str| {
        let i = out.channels.iter().position(|c| c.name == name).unwrap();
        out.planes[i].samples.clone()
    };
    let mut l = Loop {
        exact: 0,
        total: 0,
        max_abs: 0.0,
        y_exact: 0,
        y_total: 0,
    };
    for (mine, name) in [(&again.y, "Y"), (&again.ry, "RY"), (&again.by, "BY")] {
        let theirs = plane(name);
        assert_eq!(mine.len(), theirs.len(), "{name}");
        let exact = mine
            .iter()
            .zip(&theirs)
            .filter(|(a, t)| a.to_bits() == t.to_bits())
            .count();
        if name == "Y" {
            l.y_exact = exact;
            l.y_total = mine.len();
        }
        for (a, t) in mine.iter().zip(&theirs) {
            l.total += 1;
            if a.to_bits() == t.to_bits() {
                l.exact += 1;
            } else {
                l.max_abs = l.max_abs.max((a - t).abs());
            }
        }
    }
    l
}

#[test]
fn reference_decode_and_re_encode_match_ours() {
    if !tool_available("exr2aces") {
        eprintln!("exr2aces not available, skipping");
        return;
    }
    let dir = tempdir();
    for (w, h, seed, kind) in [
        (16, 16, 1, Kind::Smooth),
        (24, 20, 2, Kind::Noisy),
        (32, 8, 3, Kind::Noisy),
        (6, 4, 4, Kind::Noisy),
        (40, 30, 5, Kind::Smooth),
        (2, 2, 6, Kind::Noisy),
        (16, 16, 7, Kind::ConstantChroma),
        (40, 30, 8, Kind::ConstantChroma),
        (6, 4, 9, Kind::ConstantChroma),
    ] {
        let l = run_loop(&dir, w, h, seed, kind);
        eprintln!(
            "{w}x{h} seed {seed} {kind:?}: Y {}/{} exact; all planes {}/{} exact, max |d| {}",
            l.y_exact, l.y_total, l.exact, l.total, l.max_abs
        );
        // The luminance plane closes the loop bit-exactly: the
        // reference's decode of our file preserves the luminance our
        // decoder computes, and its writer stores it as ours does.
        assert_eq!(l.y_exact, l.y_total, "{w}x{h} seed {seed}");
        // The 2×2 ratio planes agree to a colour-level bound only: the
        // writer's chroma path is characterised on one 16×16 fixture
        // (see the crate README's luminance/chroma notes).
        assert!(l.max_abs <= 0.125, "{w}x{h} seed {seed}: {}", l.max_abs);
        // A single chromaticity closes the whole loop bit-exactly.
        if kind == Kind::ConstantChroma {
            assert_eq!(l.exact, l.total, "{w}x{h} seed {seed}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Our high-level luminance/chroma writes, read back by reference
/// tools: `exrinfo` reports the 2×2 chroma, and `exrmetrics --convert`
/// (a full read + re-write through the reference) reproduces every
/// stored plane bit-for-bit, for each compression scheme.
#[test]
fn reference_tools_read_our_luma_chroma_writes() {
    use oxideav_openexr::{ColourLayout, EncodeOptions, ExrImage, PixelFormat, decode, encode};
    if !tool_available("exrinfo") || !tool_available("exrmetrics") {
        eprintln!("exrinfo / exrmetrics not available, skipping");
        return;
    }
    let dir = tempdir();
    let (w, h) = (36u32, 22u32);
    let mut s = 11u64;
    let samples: Vec<f32> = (0..(w * h * 4) as usize)
        .map(|i| {
            if i % 4 == 3 {
                1.0 - (i / 4) as f32 / (w * h) as f32
            } else {
                0.05 + 0.9 * lcg(&mut s)
            }
        })
        .collect();
    let img = ExrImage::from_f32(w, h, PixelFormat::RgbaF32Le, &samples).unwrap();
    for comp in [
        Compression::None,
        Compression::Rle,
        Compression::Zips,
        Compression::Zip,
        Compression::Piz,
        Compression::Pxr24,
        Compression::B44,
        Compression::B44a,
    ] {
        let opts = EncodeOptions::default()
            .with_colour(ColourLayout::LumaChroma)
            .with_pixel_type(PixelType::Half)
            .with_compression(comp);
        let bytes = encode(&img, &opts).unwrap();
        let ours = dir.join(format!("ours-{comp:?}.exr"));
        let copy = dir.join(format!("copy-{comp:?}.exr"));
        std::fs::write(&ours, &bytes).unwrap();
        let info = Command::new("exrinfo").arg(&ours).output().unwrap();
        assert!(info.status.success(), "{comp:?}");
        let text = String::from_utf8_lossy(&info.stdout);
        assert!(text.contains("'RY': half samp 2 2"), "{comp:?}: {text}");
        assert!(text.contains("'BY': half samp 2 2"), "{comp:?}: {text}");
        assert!(text.contains("'Y': half samp 1 1"), "{comp:?}: {text}");
        let st = Command::new("exrmetrics")
            .arg("--convert")
            .arg(&ours)
            .arg("-o")
            .arg(&copy)
            .output()
            .unwrap();
        assert!(
            st.status.success(),
            "{comp:?}: {}",
            String::from_utf8_lossy(&st.stderr)
        );
        let a = parse_exr(&bytes).unwrap();
        let b = parse_exr(&std::fs::read(&copy).unwrap()).unwrap();
        assert_eq!(a.channels, b.channels, "{comp:?}");
        for (pa, pb) in a.planes.iter().zip(&b.planes) {
            assert_eq!(
                pa.samples.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                pb.samples.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{comp:?}"
            );
        }
        assert_eq!(
            decode(&bytes).unwrap().pixels(),
            decode(&std::fs::read(&copy).unwrap()).unwrap().pixels()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
