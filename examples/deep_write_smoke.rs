//! Smoke example: write a small deep scanline EXR via
//! [`oxideav_openexr::encode_exr_deep_scanline`]. Useful for ad-hoc
//! validation against the OpenEXR CLI tools (`exrheader`, `exrinfo`,
//! `exrmetrics --convert`); the automated cross-validation lives in
//! `tests/deep_validation.rs`.

use oxideav_openexr::{
    encode_exr_deep_scanline, Channel, Compression, DeepScanlineInput, PixelType,
};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/oxide-deep.exr".to_string());
    let w = 8u32;
    let h = 4u32;
    let chs: Vec<Channel> = ["A", "B", "G", "R"]
        .iter()
        .map(|n| Channel {
            name: n.to_string(),
            pixel_type: PixelType::Half,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        })
        .collect();
    let pixels = (w * h) as usize;
    let spp: Vec<u32> = (0..pixels as u32).map(|i| i % 4).collect();
    let total: usize = spp.iter().sum::<u32>() as usize;
    let mk = |scale: f32| -> Vec<f32> { (0..total).map(|i| (i as f32) * scale).collect() };
    let a = mk(0.05);
    let b = mk(0.1);
    let g = mk(0.15);
    let r = mk(0.2);
    let input = DeepScanlineInput {
        width: w,
        height: h,
        channels: chs,
        samples_per_pixel: &spp,
        channel_samples: vec![&a, &b, &g, &r],
        compression: Compression::Zips,
    };
    let bytes = encode_exr_deep_scanline(&input).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    eprintln!("wrote {} ({} bytes)", path, bytes.len());
}
