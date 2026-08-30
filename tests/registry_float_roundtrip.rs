//! Framework-path round trips through the real registry: resolve the
//! codec by id from a `RuntimeContext`, encode a scene-referred float
//! frame, decode the packet back, and compare — exact for FLOAT
//! channels, half-rounded for HALF — across every pixel format and
//! compression scheme the registry encoder exposes.
#![cfg(feature = "registry")]

use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, RuntimeContext, VideoFrame};
use oxideav_openexr::half::{f32_to_half, half_to_f32};
use oxideav_openexr::{parse_exr, PixelType};

const LOSSLESS: [&str; 5] = ["none", "rle", "zips", "zip", "piz"];
const LOSSY: [&str; 5] = ["pxr24", "b44", "b44a", "dwaa", "dwab"];
const FORMATS: [(PixelFormat, usize); 3] = [
    (PixelFormat::RgbaF32Le, 4),
    (PixelFormat::RgbF32Le, 3),
    (PixelFormat::GrayF32Le, 1),
];

fn ctx() -> RuntimeContext {
    let mut ctx = RuntimeContext::new();
    oxideav_openexr::register(&mut ctx);
    ctx
}

/// Scene-referred test signal: speculars above 1.0, negative
/// excursions, and a smooth ramp so the lossy codecs stay well-behaved.
fn sample(w: u32, x: usize, y: usize, c: usize) -> f32 {
    let t = (y * w as usize + x) as f32 / (w as f32 * 4.0);
    let base = 0.25 + t * 1.5 + c as f32 * 0.375;
    if (x + y) % 7 == 0 {
        base * 12.0
    } else if (x * 3 + y) % 11 == 0 {
        -base * 0.5
    } else {
        base
    }
}

/// Smooth scene-referred ramp (still above 1.0 at the far corner) for
/// the lossy legs: the block-quantising schemes bound their error by
/// the local magnitude, so a spiky signal would only measure the codec
/// design, not the framework plumbing.
fn smooth(w: u32, x: usize, y: usize, c: usize) -> f32 {
    let t = (y * w as usize + x) as f32 / (w as f32 * 4.0);
    0.125 + t * 1.5 + c as f32 * 0.375
}

fn frame(w: u32, h: u32, comps: usize, signal: fn(u32, usize, usize, usize) -> f32) -> VideoFrame {
    let mut data = Vec::with_capacity((w * h) as usize * comps * 4);
    for y in 0..h as usize {
        for x in 0..w as usize {
            for c in 0..comps {
                data.extend_from_slice(&signal(w, x, y, c).to_le_bytes());
            }
        }
    }
    VideoFrame {
        pts: None,
        planes: vec![oxideav_core::VideoPlane {
            stride: w as usize * comps * 4,
            data,
        }],
    }
}

fn roundtrip(
    w: u32,
    h: u32,
    format: PixelFormat,
    comps: usize,
    pixel_type: &str,
    compression: &str,
    signal: fn(u32, usize, usize, usize) -> f32,
) -> (Vec<u8>, VideoFrame) {
    let ctx = ctx();
    let cid = CodecId::new(oxideav_openexr::CODEC_ID_STR);
    let mut params = CodecParameters::video(cid.clone());
    params.width = Some(w);
    params.height = Some(h);
    params.pixel_format = Some(format);
    params.options.insert("pixel_type", pixel_type);
    params.options.insert("compression", compression);
    let mut enc = ctx.codecs.first_encoder(&params).expect("encoder");
    enc.send_frame(&Frame::Video(frame(w, h, comps, signal)))
        .unwrap();
    let pkt = enc.receive_packet().unwrap();
    assert!(pkt.flags.keyframe);

    let mut dec = ctx
        .codecs
        .first_decoder(&CodecParameters::video(cid))
        .expect("decoder");
    dec.send_packet(&pkt).unwrap();
    match dec.receive_frame().unwrap() {
        Frame::Video(v) => (pkt.data, v),
        _ => panic!("expected video frame"),
    }
}

fn read(vf: &VideoFrame, comps: usize, x: usize, y: usize, c: usize) -> f32 {
    let off = y * vf.planes[0].stride + (x * comps + c) * 4;
    f32::from_le_bytes(vf.planes[0].data[off..off + 4].try_into().unwrap())
}

#[test]
fn float_channels_round_trip_bit_exact_through_the_registry() {
    let (w, h) = (37u32, 19u32);
    for &(format, comps) in &FORMATS {
        for codec in LOSSLESS {
            let (bytes, vf) = roundtrip(w, h, format, comps, "float", codec, sample);
            let img = parse_exr(&bytes).unwrap();
            assert_eq!(img.channels.len(), comps, "{format:?}/{codec}");
            assert!(img
                .channels
                .iter()
                .all(|c| c.pixel_type == PixelType::Float));
            assert_eq!(vf.planes[0].stride, format.plane_row_bytes(0, w).unwrap());
            assert_eq!(
                vf.planes[0].data.len(),
                format.frame_size_bytes(w, h).unwrap()
            );
            for y in 0..h as usize {
                for x in 0..w as usize {
                    for c in 0..comps {
                        assert_eq!(
                            read(&vf, comps, x, y, c).to_bits(),
                            sample(w, x, y, c).to_bits(),
                            "{format:?}/{codec} ({x},{y}) c{c}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn half_channels_round_trip_half_rounded_through_the_registry() {
    let (w, h) = (33u32, 17u32);
    for &(format, comps) in &FORMATS {
        for codec in LOSSLESS {
            let (bytes, vf) = roundtrip(w, h, format, comps, "half", codec, sample);
            let img = parse_exr(&bytes).unwrap();
            assert!(img.channels.iter().all(|c| c.pixel_type == PixelType::Half));
            for y in 0..h as usize {
                for x in 0..w as usize {
                    for c in 0..comps {
                        let expect = half_to_f32(f32_to_half(sample(w, x, y, c)));
                        assert_eq!(
                            read(&vf, comps, x, y, c).to_bits(),
                            expect.to_bits(),
                            "{format:?}/{codec} ({x},{y}) c{c}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn lossy_codecs_round_trip_within_tolerance_through_the_registry() {
    let (w, h) = (32u32, 24u32);
    for &(format, comps) in &FORMATS {
        for codec in LOSSY {
            for pixel_type in ["float", "half"] {
                let (_bytes, vf) = roundtrip(w, h, format, comps, pixel_type, codec, smooth);
                assert_eq!(
                    vf.planes[0].data.len(),
                    format.frame_size_bytes(w, h).unwrap()
                );
                let mut worst = 0.0f32;
                for y in 0..h as usize {
                    for x in 0..w as usize {
                        for c in 0..comps {
                            let want = smooth(w, x, y, c);
                            let got = read(&vf, comps, x, y, c);
                            assert!(got.is_finite(), "{format:?}/{codec}/{pixel_type}");
                            let rel = (got - want).abs() / want.abs();
                            worst = worst.max(rel);
                        }
                    }
                }
                // B44/B44A quantise 4x4 HALF blocks against the block's
                // magnitude range, which lands around 5% on the small
                // end of this ramp; the other schemes sit well below.
                assert!(
                    worst < 0.10,
                    "{format:?}/{codec}/{pixel_type}: worst relative error {worst}"
                );
            }
        }
    }
}
