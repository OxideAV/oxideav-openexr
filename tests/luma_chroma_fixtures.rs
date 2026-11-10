//! Luminance/chroma (`Y` / `RY` / `BY`) decode and encode against the
//! traced fixture corpus in `tests/fixtures/ryby/` (each directory: an
//! `input.exr`, the reference RGBA decode as binary16 and binary32 dumps,
//! and for the encoder case the reference-written stored planes).
//!
//! Dumps are little-endian, row-major from the top data-window row, RGBA
//! interleaved. The comparison is on the binary16 code words of every
//! colour sample; alpha is checked against the file's `A` (or `1.0`).

use std::path::PathBuf;

use oxideav_openexr::{
    DecodeOptions, ExrImage, LayerKind, PixelFormat, channel_view, decode, decode_with,
    default_view, enumerate_layers, luminance_weights, luminance_weights_of, parse_exr,
    rgb_to_luma_chroma,
};

fn dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ryby")
        .join(name)
}

fn read(name: &str, file: &str) -> Vec<u8> {
    std::fs::read(dir(name).join(file)).unwrap()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn u16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect()
}

/// The image as interleaved RGBA `f32` (alpha `1.0` when absent, gray
/// replicated).
fn rgba(img: &ExrImage) -> Vec<f32> {
    let comps = match img.format {
        PixelFormat::RgbaF32Le => 4,
        PixelFormat::RgbF32Le => 3,
        PixelFormat::GrayF32Le => 1,
        f => panic!("unexpected format {f:?}"),
    };
    let plane = &img.planes[0];
    let mut out = Vec::new();
    for y in 0..img.height as usize {
        let row = &plane.data[y * plane.stride..];
        let px = f32s(&row[..img.width as usize * comps * 4]);
        for p in px.chunks_exact(comps) {
            match comps {
                4 => out.extend_from_slice(p),
                3 => out.extend_from_slice(&[p[0], p[1], p[2], 1.0]),
                _ => out.extend_from_slice(&[p[0], p[0], p[0], 1.0]),
            }
        }
    }
    out
}

/// binary16 code of an `f32` that must be exactly representable.
fn half_bits(v: f32) -> u16 {
    let h = oxideav_openexr::half::f32_to_half(v);
    assert_eq!(
        oxideav_openexr::half::half_to_f32(h).to_bits(),
        v.to_bits(),
        "{v} is not a binary16 value"
    );
    h
}

/// Per-fixture comparison outcome.
#[derive(Debug, Default)]
struct Score {
    pixels: usize,
    exact_pixels: usize,
    samples: usize,
    exact_samples: usize,
    /// Largest distance in binary16 units of the last place over the
    /// non-exact samples.
    max_ulp: u32,
    /// Largest absolute difference of a non-exact sample.
    max_abs: f32,
}

fn ulp_distance(a: u16, b: u16) -> u32 {
    // Map sign-magnitude codes onto a monotone integer line.
    let key = |h: u16| -> i32 {
        if h & 0x8000 != 0 {
            -i32::from(h & 0x7fff)
        } else {
            i32::from(h)
        }
    };
    key(a).abs_diff(key(b))
}

fn score(got: &[f32], expected_f16: &[u16], expected_f32: &[f32]) -> Score {
    assert_eq!(got.len(), expected_f16.len());
    assert_eq!(got.len(), expected_f32.len());
    let mut s = Score::default();
    for (px, (g, (e16, e32))) in got
        .as_chunks::<4>()
        .0
        .iter()
        .zip(
            expected_f16
                .as_chunks::<4>()
                .0
                .iter()
                .zip(expected_f32.as_chunks::<4>().0),
        )
        .enumerate()
    {
        let mut pixel_exact = true;
        for c in 0..4 {
            // The two dumps carry identical values.
            assert_eq!(
                oxideav_openexr::half::half_to_f32(e16[c]).to_bits(),
                e32[c].to_bits(),
                "dump mismatch px{px}"
            );
            let gh = half_bits(g[c]);
            s.samples += 1;
            if gh == e16[c] {
                s.exact_samples += 1;
            } else {
                pixel_exact = false;
                s.max_ulp = s.max_ulp.max(ulp_distance(gh, e16[c]));
                s.max_abs = s.max_abs.max((g[c] - e32[c]).abs());
            }
        }
        s.pixels += 1;
        if pixel_exact {
            s.exact_pixels += 1;
        }
    }
    s
}

fn check_decode(name: &str, layer: &str, expected: &str) -> Score {
    let bytes = read(name, "input.exr");
    let img = if layer.is_empty() {
        decode(&bytes).unwrap()
    } else {
        decode_with(&bytes, &DecodeOptions::new().with_layer(layer)).unwrap()
    };
    let got = rgba(&img);
    let s = score(
        &got,
        &u16s(&read(name, &format!("{expected}.f16"))),
        &f32s(&read(name, &format!("{expected}.f32"))),
    );
    eprintln!("{name} [{layer}]: {s:?}");
    s
}

fn assert_bit_exact(name: &str) {
    let s = check_decode(name, "", "expected.rgba");
    assert_eq!(s.exact_samples, s.samples, "{name}: {s:?}");
}

#[test]
fn const_rec709_is_bit_exact() {
    assert_bit_exact("ryby-const-rec709");
}

#[test]
fn impulse_rec709_is_bit_exact() {
    assert_bit_exact("ryby-impulse-rec709");
}

#[test]
fn edge_rec709_is_bit_exact() {
    assert_bit_exact("ryby-edge-rec709");
}

#[test]
fn random_rec709_is_bit_exact() {
    assert_bit_exact("ryby-random-rec709");
}

#[test]
fn random_ap0_is_bit_exact() {
    assert_bit_exact("ryby-random-ap0");
}

#[test]
fn luma_alpha_is_bit_exact() {
    assert_bit_exact("ryby-luma-alpha");
}

#[test]
fn encoder_roundtrip_decode_is_bit_exact() {
    assert_bit_exact("ryby-encoder-roundtrip");
}

#[test]
fn saturated_rec709_is_bit_exact() {
    // Saturation limiting, Y <= 0 pixels (signed zeros included) and
    // the edge-neighbour ring.
    assert_bit_exact("ryby-saturated-rec709");
}

#[test]
fn weights_reproduce_the_tabulated_bits() {
    let bt709 = luminance_weights(&oxideav_openexr::BT709_CHROMATICITIES);
    assert_eq!(
        bt709.map(f32::to_bits),
        [0x3e59_be0b, 0x3f37_154b, 0x3d93_d991]
    );
    let ap0 = parse_exr(&read("ryby-random-ap0", "input.exr")).unwrap();
    assert_eq!(
        luminance_weights_of(&ap0.attributes).map(f32::to_bits),
        [0x3eb0_1c5f, 0x3f3a_6918, 0xbd93_ba3b]
    );
    // Writing the default primaries explicitly is the same as omitting
    // them.
    assert_eq!(luminance_weights_of(&[]), bt709);
}

#[test]
fn raw_planes_stay_reachable_as_stored() {
    let p = parse_exr(&read("ryby-encoder-roundtrip", "input.exr")).unwrap();
    let plane = |n: &str| {
        let i = p.channels.iter().position(|c| c.name == n).unwrap();
        (p.channels[i].x_sampling, p.planes[i].samples.clone())
    };
    for (n, file, s) in [
        ("Y", "expected.stored-Y.f32", 1),
        ("RY", "expected.stored-RY.f32", 2),
        ("BY", "expected.stored-BY.f32", 2),
    ] {
        let (sampling, samples) = plane(n);
        assert_eq!(sampling, s);
        let exp = f32s(&read("ryby-encoder-roundtrip", file));
        assert_eq!(
            samples.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            exp.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{n}"
        );
    }
}

#[test]
fn encoder_reproduces_the_stored_planes() {
    let name = "ryby-encoder-roundtrip";
    let input = f32s(&read(name, "input.rgba.f32"));
    let (w, h) = (16u32, 16u32);
    let n = (w * h) as usize;
    let comp = |c: usize| -> Vec<f32> { (0..n).map(|i| input[4 * i + c]).collect() };
    let (r, g, b) = (comp(0), comp(1), comp(2));
    let weights = luminance_weights_of(&[]);
    let yc = rgb_to_luma_chroma(w, h, [&r, &g, &b], weights, (2, 2)).unwrap();
    let mut exact = 0;
    let mut total = 0;
    for (got, file) in [
        (&yc.y, "expected.stored-Y.f32"),
        (&yc.ry, "expected.stored-RY.f32"),
        (&yc.by, "expected.stored-BY.f32"),
    ] {
        let exp = f32s(&read(name, file));
        assert_eq!(got.len(), exp.len(), "{file}");
        for (i, (a, e)) in got.iter().zip(&exp).enumerate() {
            total += 1;
            if a.to_bits() == e.to_bits() {
                exact += 1;
            } else {
                eprintln!("{file}[{i}]: {a} vs {e}");
            }
        }
    }
    eprintln!("{name} stored planes: {exact}/{total} exact");
    assert_eq!(exact, total);
}

#[test]
fn multiview_layers_decode_bit_exact() {
    let name = "ryby-layers-multiview";
    // Default view (bare names; the stray `left.Y` is ignored), the
    // default view by name, the other view, layered views and a
    // non-view layer holding only `R`.
    for (layer, expected) in [
        ("", "expected.rgba"),
        ("left", "expected.left.rgba"),
        ("right", "expected.right.rgba"),
        ("spec.left", "expected.spec.left.rgba"),
        ("spec.right", "expected.spec.right.rgba"),
        ("diffuse", "expected.diffuse.rgba"),
    ] {
        let s = check_decode(name, layer, expected);
        assert_eq!(s.exact_samples, s.samples, "{name} [{layer}]: {s:?}");
    }
    // `spec` alone names no channel set (only `spec.left.*` /
    // `spec.right.*` exist): a caller error rather than an empty frame.
    let bytes = read(name, "input.exr");
    assert!(matches!(
        decode_with(&bytes, &DecodeOptions::new().with_layer("spec")),
        Err(oxideav_openexr::ExrError::InvalidData(_))
    ));
}

#[test]
fn multiview_channel_views_match_the_reference_classification() {
    let name = "ryby-layers-multiview";
    let part = parse_exr(&read(name, "input.exr")).unwrap();
    assert_eq!(default_view(&part.attributes), Some("left"));
    let text = String::from_utf8(read(name, "expected.views.txt")).unwrap();
    let mut checked = 0;
    for line in text.lines() {
        let Some(rest) = line.trim().strip_prefix("channel ") else {
            continue;
        };
        let (chan, view) = rest.split_once("->").unwrap();
        let chan = chan.trim();
        let view = view.trim().trim_start_matches("view ").trim_matches('\'');
        let expect = (!view.is_empty()).then_some(view);
        assert_eq!(channel_view(chan, &part.attributes), expect, "{chan}");
        checked += 1;
    }
    assert_eq!(checked, part.channels.len());
    assert_eq!(part.views(), ["left", "right"]);
    let right_y = part.channels.iter().position(|c| c.name == "right.Y");
    assert_eq!(part.channel_view(right_y.unwrap()), Some("right"));
    assert_eq!(part.channel_view(999), None);
    // The header-only description carries the same layers and views.
    let info = oxideav_openexr::info(&read(name, "input.exr")).unwrap();
    assert_eq!(info.views, ["left", "right"]);
    assert_eq!(info.layers, part.layers());
    // The enumerated layers carry the same views.
    let layers = enumerate_layers(&part.channels, &part.attributes);
    let summary: Vec<(&str, Option<&str>, LayerKind)> = layers
        .iter()
        .map(|l| (l.name.as_str(), l.view.as_deref(), l.kind))
        .collect();
    assert_eq!(
        summary,
        [
            ("", Some("left"), LayerKind::LumaChromaAlpha),
            ("diffuse", None, LayerKind::PartialRgb),
            ("left", Some("left"), LayerKind::Gray),
            ("right", Some("right"), LayerKind::LumaChromaAlpha),
            ("spec.left", Some("left"), LayerKind::LumaChroma),
            ("spec.right", Some("right"), LayerKind::LumaChroma),
        ]
    );
}
