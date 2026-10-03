//! Tiny helper that writes a 4×4 RGBA float ZIP-compressed EXR to the
//! path passed as argv[1], used for ad-hoc validation against the
//! `exrheader` / `exrinfo` command-line tools (and as a smoke test for
//! Cargo example builds).

use oxideav_openexr::{encode, Compression, EncodeOptions, ExrImage, PixelFormat};

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: encode_sample <out.exr>");
    let w = 4;
    let h = 4;
    let samples: Vec<f32> = (0..(w * h * 4)).map(|i| (i as f32) * 0.05).collect();
    let img = ExrImage::from_f32(w, h, PixelFormat::RgbaF32Le, &samples).unwrap();
    let opts = EncodeOptions::default().with_compression(Compression::Zip);
    let bytes = encode(&img, &opts).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    println!("wrote {} bytes to {}", bytes.len(), path);
}
