//! Decode / encode throughput of the contract API per compression
//! scheme, for scanline and tiled files, HALF and FLOAT channels, at
//! one or more worker budgets.
//!
//! ```sh
//! cargo run --release --example throughput            # budgets 1 and 8
//! cargo run --release --example throughput -- 1 4 16  # chosen budgets
//! EXR_BENCH_ONLY=ZIP,PIZ cargo run --release --example throughput
//! ```
//!
//! Fixture: 2048 x 1024 RGBA (2.1 Mpx), smooth gradients with fine
//! noise (compressible, not flat). Each figure is megapixels per
//! second, the median of five runs. The output is a Markdown table.

use std::time::Instant;

use oxideav_openexr::{
    Compression, DecodeOptions, EncodeOptions, ExrImage, ExrPixelFormat, PixelType, decode_with,
    encode,
};

const WIDTH: u32 = 2048;
const HEIGHT: u32 = 1024;
const RUNS: usize = 5;

fn fixture() -> ExrImage {
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut noise = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    let mut samples = Vec::with_capacity((WIDTH * HEIGHT * 4) as usize);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let (u, v) = (x as f32 / WIDTH as f32, y as f32 / HEIGHT as f32);
            let n = noise() * 0.02;
            samples.push(0.8 * u + 0.1 * (v * 40.0).sin().abs() + n);
            samples.push(0.6 * v + 0.2 * (u * 25.0).cos().abs() + n);
            samples.push(0.5 * (u + v) * 0.5 + n);
            samples.push(1.0);
        }
    }
    ExrImage::from_f32(WIDTH, HEIGHT, ExrPixelFormat::RgbaF32Le, &samples).expect("fixture")
}

/// Megapixels per second of `f`, median of [`RUNS`].
fn mpx_per_second(mut f: impl FnMut()) -> f64 {
    let mut times: Vec<f64> = (0..RUNS)
        .map(|_| {
            let start = Instant::now();
            f();
            start.elapsed().as_secs_f64()
        })
        .collect();
    times.sort_by(f64::total_cmp);
    (WIDTH as f64 * HEIGHT as f64) / 1e6 / times[RUNS / 2]
}

fn main() {
    let budgets: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let budgets = if budgets.is_empty() {
        vec![1, 8]
    } else {
        budgets
    };
    let image = fixture();
    let schemes = [
        ("NONE", Compression::None),
        ("RLE", Compression::Rle),
        ("ZIPS", Compression::Zips),
        ("ZIP", Compression::Zip),
        ("PIZ", Compression::Piz),
        ("PXR24", Compression::Pxr24),
        ("B44", Compression::B44),
        ("B44A", Compression::B44a),
        ("DWAA", Compression::Dwaa),
        ("DWAB", Compression::Dwab),
    ];
    let mut header = String::from("| Scheme | Layout | Type |");
    let mut rule = String::from("|---|---|---|");
    for b in &budgets {
        header += &format!(" decode x{b} | encode x{b} |");
        rule += "---:|---:|";
    }
    println!("{header}\n{rule}");
    // `EXR_BENCH_ONLY=ZIP,PIZ` restricts the run to the named schemes.
    let only = std::env::var("EXR_BENCH_ONLY").unwrap_or_default();
    for (name, compression) in schemes {
        if !only.is_empty() && !only.split(',').any(|s| s == name) {
            continue;
        }
        for (layout, tile) in [("scanline", 0u32), ("tiled 256", 256)] {
            for (type_name, pixel_type) in [("HALF", PixelType::Half), ("FLOAT", PixelType::Float)]
            {
                let mut row = format!("| {name} | {layout} | {type_name} |");
                for &threads in &budgets {
                    let opts = EncodeOptions::default()
                        .with_compression(compression)
                        .with_pixel_type(pixel_type)
                        .with_tile_size(tile)
                        .with_threads(threads);
                    let bytes = encode(&image, &opts).expect("encode");
                    let read = DecodeOptions::default().with_threads(threads);
                    let decode = mpx_per_second(|| {
                        std::hint::black_box(decode_with(&bytes, &read).expect("decode"));
                    });
                    let encode = mpx_per_second(|| {
                        std::hint::black_box(encode(&image, &opts).expect("encode"));
                    });
                    row += &format!(" {decode:.0} | {encode:.0} |");
                }
                println!("{row}");
            }
        }
    }
}
