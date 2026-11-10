//! The traced luminance/chroma fixture corpus (`tests/fixtures/ryby/`)
//! decoded through the framework path: the registry decoder resolved
//! from a `RuntimeContext` by codec id, with the `layer` option for the
//! multi-view fixture. Every colour sample must equal the reference
//! RGBA decode bit-for-bit (alpha: the file's `A`, else `1.0`).
#![cfg(feature = "registry")]

use std::path::PathBuf;

use oxideav_core::{CodecId, CodecParameters, Frame, Packet, RuntimeContext, TimeBase};

fn read(name: &str, file: &str) -> Vec<u8> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ryby")
        .join(name)
        .join(file);
    std::fs::read(p).unwrap()
}

fn decode(bytes: Vec<u8>, layer: &str) -> (usize, Vec<f32>) {
    let mut ctx = RuntimeContext::new();
    oxideav_openexr::register(&mut ctx);
    let mut params = CodecParameters::video(CodecId::new(oxideav_openexr::CODEC_ID_STR));
    if !layer.is_empty() {
        params.options.insert("layer", layer.to_string());
    }
    let mut dec = ctx.codecs.first_decoder(&params).unwrap();
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), bytes))
        .unwrap();
    let Frame::Video(v) = dec.receive_frame().unwrap() else {
        panic!("expected a video frame");
    };
    let plane = &v.planes[0];
    let samples: Vec<f32> = plane
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    (plane.stride, samples)
}

fn check(name: &str, layer: &str, expected: &str, width: usize) -> usize {
    let (stride, got) = decode(read(name, "input.exr"), layer);
    let comps = stride / (width * 4);
    let exp: Vec<u16> = read(name, &format!("{expected}.f16"))
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    let pixels = exp.len() / 4;
    assert_eq!(got.len(), pixels * comps, "{name} [{layer}]");
    for px in 0..pixels {
        for c in 0..4 {
            let v = match (comps, c) {
                (3, 3) => 1.0,
                (1, 3) => 1.0,
                (1, _) => got[px],
                _ => got[px * comps + c],
            };
            let h = oxideav_openexr::half::f32_to_half(v);
            assert_eq!(
                h,
                exp[px * 4 + c],
                "{name} [{layer}] px {px} c {c}: {v} vs {}",
                oxideav_openexr::half::half_to_f32(exp[px * 4 + c])
            );
            assert_eq!(oxideav_openexr::half::half_to_f32(h).to_bits(), v.to_bits());
        }
    }
    pixels * 4
}

#[test]
fn every_fixture_decodes_bit_exactly_through_the_registry() {
    let mut samples = 0;
    for (name, w) in [
        ("ryby-const-rec709", 8),
        ("ryby-impulse-rec709", 32),
        ("ryby-edge-rec709", 16),
        ("ryby-random-rec709", 24),
        ("ryby-saturated-rec709", 16),
        ("ryby-random-ap0", 16),
        ("ryby-luma-alpha", 8),
        ("ryby-encoder-roundtrip", 16),
    ] {
        samples += check(name, "", "expected.rgba", w);
    }
    for (layer, expected) in [
        ("", "expected.rgba"),
        ("left", "expected.left.rgba"),
        ("right", "expected.right.rgba"),
        ("spec.left", "expected.spec.left.rgba"),
        ("spec.right", "expected.spec.right.rgba"),
        ("diffuse", "expected.diffuse.rgba"),
    ] {
        samples += check("ryby-layers-multiview", layer, expected, 8);
    }
    assert_eq!(samples, 11_776);
}
