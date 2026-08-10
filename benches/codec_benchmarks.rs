//! Criterion benchmarks for the scanline codec paths (round 385).
//!
//! Measures decode (`parse_exr`) and encode (`encode_exr_scanline`)
//! throughput on a 256×256 four-channel (A,B,G,R) image across every
//! supported compression scheme, for both HALF and FLOAT pixel types,
//! plus the binary16 conversion primitives that sit on the HALF hot
//! path. Throughput is reported in raw interleaved pixel bytes so the
//! schemes are comparable.
//!
//! Run with:
//! ```sh
//! CARGO_TARGET_DIR=/tmp/oxideav-openexr-bench cargo bench -p oxideav-openexr
//! ```

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use oxideav_openexr::{
    encode_exr_scanline, parse_exr, Attribute, AttributeValue, Box2i, Channel, Compression,
    LineOrder, PixelType,
};

const W: u32 = 256;
const H: u32 = 256;

fn channels(pt: PixelType) -> Vec<Channel> {
    ["A", "B", "G", "R"]
        .iter()
        .map(|n| Channel {
            name: n.to_string(),
            pixel_type: pt,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        })
        .collect()
}

fn attrs(w: u32, h: u32, chs: &[Channel], compression: Compression) -> Vec<Attribute> {
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

/// Deterministic photographic-ish planes: smooth gradients with a bit
/// of structure so ZIP/RLE see realistic (compressible but not flat)
/// data and B44's shift search does real work.
fn make_planes(w: u32, h: u32) -> Vec<Vec<f32>> {
    let mut planes = Vec::with_capacity(4);
    for c in 0..4usize {
        let mut p = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let fx = x as f32 / w as f32;
                let fy = y as f32 / h as f32;
                let v = match c {
                    0 => 1.0,                                            // A: opaque
                    1 => 0.25 + 4.0 * fx * fy,                           // B
                    2 => 0.5 + 2.0 * (1.0 - fx) + ((x / 16) % 2) as f32, // G: blocky
                    _ => 8.0 * fy + 0.125 * ((x % 7) as f32),            // R: stripes
                };
                p.push(v);
            }
        }
        planes.push(p);
    }
    planes
}

fn encode_with(compression: Compression, pt: PixelType, planes: &[Vec<f32>]) -> Vec<u8> {
    let chs = channels(pt);
    let refs: Vec<&[f32]> = planes.iter().map(|v| v.as_slice()).collect();
    encode_exr_scanline(
        W,
        H,
        &chs,
        &refs,
        compression,
        attrs(W, H, &chs, compression),
    )
    .unwrap()
}

fn schemes() -> Vec<(&'static str, Compression)> {
    vec![
        ("none", Compression::None),
        ("rle", Compression::Rle),
        ("zips", Compression::Zips),
        ("zip", Compression::Zip),
        ("pxr24", Compression::Pxr24),
        ("piz", Compression::Piz),
        ("b44", Compression::B44),
        ("b44a", Compression::B44a),
        ("dwaa", Compression::Dwaa),
        ("dwab", Compression::Dwab),
    ]
}

fn bench_decode(c: &mut Criterion) {
    let planes = make_planes(W, H);
    for (pt_name, pt) in [("half", PixelType::Half), ("float", PixelType::Float)] {
        let mut group = c.benchmark_group(format!("decode_{pt_name}"));
        let raw_bytes = (W * H) as u64 * 4 * pt.bytes_per_sample() as u64;
        group.throughput(Throughput::Bytes(raw_bytes));
        for (name, scheme) in schemes() {
            let bytes = encode_with(scheme, pt, &planes);
            group.bench_function(name, |b| {
                b.iter(|| parse_exr(black_box(&bytes)).unwrap());
            });
        }
        group.finish();
    }
}

fn bench_encode(c: &mut Criterion) {
    let planes = make_planes(W, H);
    for (pt_name, pt) in [("half", PixelType::Half), ("float", PixelType::Float)] {
        let mut group = c.benchmark_group(format!("encode_{pt_name}"));
        let raw_bytes = (W * H) as u64 * 4 * pt.bytes_per_sample() as u64;
        group.throughput(Throughput::Bytes(raw_bytes));
        for (name, scheme) in schemes() {
            group.bench_function(name, |b| {
                b.iter(|| black_box(encode_with(scheme, pt, black_box(&planes))));
            });
        }
        group.finish();
    }
}

fn bench_half_primitives(c: &mut Criterion) {
    // The binary16 conversions sit on every HALF sample both ways.
    let codes: Vec<u16> = (0..=u16::MAX).collect();
    let floats: Vec<f32> = codes
        .iter()
        .map(|&h| oxideav_openexr::half::half_to_f32(h))
        .collect();

    let mut group = c.benchmark_group("half");
    group.throughput(Throughput::Elements(codes.len() as u64));
    group.bench_function("half_to_f32_all_codes", |b| {
        b.iter(|| {
            let mut acc = 0f32;
            for &h in &codes {
                acc += oxideav_openexr::half::half_to_f32(black_box(h));
            }
            black_box(acc)
        });
    });
    group.bench_function("f32_to_half_all_codes", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for &f in &floats {
                acc = acc.wrapping_add(u32::from(oxideav_openexr::half::f32_to_half(black_box(f))));
            }
            black_box(acc)
        });
    });
    group.finish();
}

fn bench_decode_tiled(c: &mut Criterion) {
    // Tiled ONE_LEVEL decode (64×64 tiles): exercises the tile scatter
    // path, which is distinct from the scanline block scatter.
    let planes = make_planes(W, H);
    for (pt_name, pt) in [("half", PixelType::Half), ("float", PixelType::Float)] {
        let mut group = c.benchmark_group(format!("decode_tiled_{pt_name}"));
        let raw_bytes = (W * H) as u64 * 4 * pt.bytes_per_sample() as u64;
        group.throughput(Throughput::Bytes(raw_bytes));
        for (name, scheme) in schemes() {
            let chs = channels(pt);
            let refs: Vec<&[f32]> = planes.iter().map(|v| v.as_slice()).collect();
            let bytes =
                oxideav_openexr::encode_exr_tiled(W, H, &chs, &refs, scheme, 64, 64).unwrap();
            group.bench_function(name, |b| {
                b.iter(|| parse_exr(black_box(&bytes)).unwrap());
            });
        }
        group.finish();
    }
}

criterion_group!(
    benches,
    bench_decode,
    bench_decode_tiled,
    bench_encode,
    bench_half_primitives
);
criterion_main!(benches);
