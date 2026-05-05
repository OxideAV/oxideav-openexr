//! Self-roundtrip integration tests for `oxideav-openexr`.
//!
//! Each test hard-asserts that encoding a synthetic float RGBA image
//! and re-parsing it yields the original samples bit-exactly. Since
//! the encoder writes FLOAT channels (not HALF), there's no precision
//! loss — the only conversion is `f32 -> 4 LE bytes -> f32` and
//! optionally a zlib round-trip.

use oxideav_openexr::{
    encode_exr_scanline_rgba_float, encode_exr_scanline_rgba_float_with, parse_exr, Compression,
    PixelType,
};

fn make_sample_image(w: u32, h: u32) -> Vec<f32> {
    // Mix of sub-1.0 values, exactly-1.0, and HDR (>1.0) values so the
    // round-trip catches any clamping / range-truncation bugs.
    let mut s = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = (x as f32 / w as f32) * 1.5; // up to 1.5
            let g = (y as f32 / h as f32) * 1.5;
            let b = ((x ^ y) as f32 / 255.0) * 8.0; // HDR up to 8.0
            let a = if (x + y) % 2 == 0 { 0.25 } else { 1.75 };
            s.extend_from_slice(&[r, g, b, a]);
        }
    }
    s
}

#[test]
fn roundtrip_no_compression_small() {
    let w = 8;
    let h = 4;
    let samples = make_sample_image(w, h);
    let bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::None).unwrap();
    let img = parse_exr(&bytes).unwrap();
    assert_eq!(img.width(), w);
    assert_eq!(img.height(), h);
    assert_eq!(img.compression, Compression::None);
    assert_planes_match(&img, &samples);
}

#[test]
fn roundtrip_zip_small() {
    let w = 16;
    let h = 16;
    let samples = make_sample_image(w, h);
    let bytes = encode_exr_scanline_rgba_float(w, h, &samples).unwrap();
    let img = parse_exr(&bytes).unwrap();
    assert_eq!(img.width(), w);
    assert_eq!(img.height(), h);
    assert_eq!(img.compression, Compression::Zip);
    assert_planes_match(&img, &samples);
}

#[test]
fn roundtrip_zip_multi_block() {
    // 40 rows > one ZIP block (16 rows), so we exercise three blocks.
    let w = 12;
    let h = 40;
    let samples = make_sample_image(w, h);
    let bytes = encode_exr_scanline_rgba_float(w, h, &samples).unwrap();
    let img = parse_exr(&bytes).unwrap();
    assert_eq!(img.height(), h);
    assert_planes_match(&img, &samples);
}

#[test]
fn header_round_trip_attributes() {
    let w = 4;
    let h = 4;
    let samples = vec![0.5_f32; (w * h * 4) as usize];
    let bytes = encode_exr_scanline_rgba_float(w, h, &samples).unwrap();
    let img = parse_exr(&bytes).unwrap();
    assert_eq!(img.pixel_aspect_ratio, 1.0);
    assert_eq!(img.screen_window_center, (0.0, 0.0));
    assert_eq!(img.screen_window_width, 1.0);
    let names: Vec<&str> = img.channels.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["A", "B", "G", "R"]);
    for ch in &img.channels {
        assert_eq!(ch.pixel_type, PixelType::Float);
        assert_eq!(ch.x_sampling, 1);
        assert_eq!(ch.y_sampling, 1);
    }
}

#[test]
fn rejects_bad_magic() {
    let mut bytes = encode_exr_scanline_rgba_float(2, 2, &[0.0_f32; 16]).unwrap();
    bytes[0] = 0xFF;
    assert!(parse_exr(&bytes).is_err());
}

#[test]
fn rejects_truncated_offset_table() {
    let bytes = encode_exr_scanline_rgba_float(4, 4, &[0.0_f32; 64]).unwrap();
    // Truncate inside the offset table region.
    let truncated = &bytes[..bytes.len().min(60)];
    assert!(parse_exr(truncated).is_err());
}

fn assert_planes_match(img: &oxideav_openexr::ExrImage, source_rgba: &[f32]) {
    let w = img.width() as usize;
    let h = img.height() as usize;
    // Channels are alphabetical: A=0, B=1, G=2, R=3.
    let a = &img.planes[0].samples;
    let b = &img.planes[1].samples;
    let g = &img.planes[2].samples;
    let r = &img.planes[3].samples;
    for y in 0..h {
        for x in 0..w {
            let off = y * w + x;
            assert_eq!(r[off], source_rgba[off * 4], "R mismatch at ({x},{y})");
            assert_eq!(g[off], source_rgba[off * 4 + 1], "G mismatch at ({x},{y})");
            assert_eq!(b[off], source_rgba[off * 4 + 2], "B mismatch at ({x},{y})");
            assert_eq!(a[off], source_rgba[off * 4 + 3], "A mismatch at ({x},{y})");
        }
    }
}
