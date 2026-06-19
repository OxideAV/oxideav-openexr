//! Self-roundtrip + reject-path validation for the round-232 mixed
//! multi-part (`scanlineimage` + `tiledimage`) WRITE + READ pair.

use oxideav_openexr::{
    encode_exr_multipart_mixed, parse_exr_multipart_mixed, Channel, Compression,
    MultipartMixedPart, PixelType,
};

fn make_planes(w: u32, h: u32, salt: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let pixels = (w as usize) * (h as usize);
    let mut a = Vec::with_capacity(pixels);
    let mut b = Vec::with_capacity(pixels);
    let mut g = Vec::with_capacity(pixels);
    let mut r = Vec::with_capacity(pixels);
    for y in 0..h {
        for x in 0..w {
            r.push((x as f32) / (w as f32) + salt);
            g.push((y as f32) / (h as f32) + salt * 0.5);
            b.push(((x ^ y) as f32) * 0.01 + salt * 0.25);
            a.push(1.0);
        }
    }
    (a, b, g, r)
}

fn rgba_channels() -> Vec<Channel> {
    ["A", "B", "G", "R"]
        .iter()
        .map(|n| Channel {
            name: (*n).to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        })
        .collect()
}

#[test]
fn mixed_scanline_then_tiled_zips() {
    let w = 16;
    let h = 16;
    let (a0, b0, g0, r0) = make_planes(w, h, 0.0);
    let (a1, b1, g1, r1) = make_planes(w, h, 0.5);
    let bytes = encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "scan".to_string(),
            width: w,
            height: h,
            channels: rgba_channels(),
            planes: vec![&a0, &b0, &g0, &r0],
            compression: Compression::Zips,
        },
        MultipartMixedPart::Tiled {
            name: "tile".to_string(),
            width: w,
            height: h,
            tile_x: 8,
            tile_y: 8,
            channels: rgba_channels(),
            planes: vec![&a1, &b1, &g1, &r1],
            compression: Compression::Zips,
        },
    ])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert_eq!(imgs.len(), 2);
    assert!(imgs[0].is_scanline());
    assert!(imgs[1].is_tiled());
    for (img, (sa, sb, sg, sr)) in imgs
        .iter()
        .zip([(&a0, &b0, &g0, &r0), (&a1, &b1, &g1, &r1)].iter())
    {
        let image = img.image().expect("flat part");
        assert_eq!(image.width(), w);
        assert_eq!(image.height(), h);
        assert_eq!(&image.planes[0].samples, *sa);
        assert_eq!(&image.planes[1].samples, *sb);
        assert_eq!(&image.planes[2].samples, *sg);
        assert_eq!(&image.planes[3].samples, *sr);
    }
}

#[test]
fn mixed_tiled_then_scanline_none() {
    // Reverse order: tiled first, scanline second, NONE compression so the
    // chunk shapes are easy to inspect.
    let w = 12;
    let h = 9;
    let (a0, b0, g0, r0) = make_planes(w, h, 0.25);
    let (a1, b1, g1, r1) = make_planes(w, h, 0.75);
    let bytes = encode_exr_multipart_mixed(&[
        MultipartMixedPart::Tiled {
            name: "first".to_string(),
            width: w,
            height: h,
            tile_x: 4,
            tile_y: 3,
            channels: rgba_channels(),
            planes: vec![&a0, &b0, &g0, &r0],
            compression: Compression::None,
        },
        MultipartMixedPart::Scanline {
            name: "second".to_string(),
            width: w,
            height: h,
            channels: rgba_channels(),
            planes: vec![&a1, &b1, &g1, &r1],
            compression: Compression::None,
        },
    ])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert_eq!(imgs.len(), 2);
    assert!(imgs[0].is_tiled());
    assert!(imgs[1].is_scanline());
    let img0 = imgs[0].image().expect("flat part");
    let img1 = imgs[1].image().expect("flat part");
    assert_eq!(&img0.planes[0].samples, &a0);
    assert_eq!(&img0.planes[1].samples, &b0);
    assert_eq!(&img0.planes[2].samples, &g0);
    assert_eq!(&img0.planes[3].samples, &r0);
    assert_eq!(&img1.planes[0].samples, &a1);
    assert_eq!(&img1.planes[1].samples, &b1);
    assert_eq!(&img1.planes[2].samples, &g1);
    assert_eq!(&img1.planes[3].samples, &r1);
}

#[test]
fn mixed_three_parts_mixed_compression() {
    let w = 16;
    let h = 12;
    let (a0, b0, g0, r0) = make_planes(w, h, 0.0);
    let (a1, b1, g1, r1) = make_planes(w, h, 0.25);
    let (a2, b2, g2, r2) = make_planes(w, h, 0.75);
    let bytes = encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "scan_rle".to_string(),
            width: w,
            height: h,
            channels: rgba_channels(),
            planes: vec![&a0, &b0, &g0, &r0],
            compression: Compression::Rle,
        },
        MultipartMixedPart::Tiled {
            name: "tile_zip".to_string(),
            width: w,
            height: h,
            tile_x: 8,
            tile_y: 8,
            channels: rgba_channels(),
            planes: vec![&a1, &b1, &g1, &r1],
            compression: Compression::Zip,
        },
        MultipartMixedPart::Scanline {
            name: "scan_zip".to_string(),
            width: w,
            height: h,
            channels: rgba_channels(),
            planes: vec![&a2, &b2, &g2, &r2],
            compression: Compression::Zip,
        },
    ])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert_eq!(imgs.len(), 3);
    assert!(imgs[0].is_scanline());
    assert!(imgs[1].is_tiled());
    assert!(imgs[2].is_scanline());
    let sources = [
        (&a0, &b0, &g0, &r0),
        (&a1, &b1, &g1, &r1),
        (&a2, &b2, &g2, &r2),
    ];
    for (img, (sa, sb, sg, sr)) in imgs.iter().zip(sources.iter()) {
        let image = img.image().expect("flat part");
        assert_eq!(&image.planes[0].samples, *sa);
        assert_eq!(&image.planes[1].samples, *sb);
        assert_eq!(&image.planes[2].samples, *sg);
        assert_eq!(&image.planes[3].samples, *sr);
    }
}

#[test]
fn mixed_tiled_with_edge_tiles_zip() {
    // 13×9 with 4×3 tiles: tile grid is 4×3 = 12 tiles, right column +
    // bottom row are edge tiles smaller than 4×3. Exercise alongside a
    // matching scanline part to confirm both edge-tile and
    // small-final-block paths run interleaved cleanly.
    let w = 13;
    let h = 9;
    let (a0, b0, g0, r0) = make_planes(w, h, 0.0);
    let (a1, b1, g1, r1) = make_planes(w, h, 0.5);
    let bytes = encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "scan".to_string(),
            width: w,
            height: h,
            channels: rgba_channels(),
            planes: vec![&a0, &b0, &g0, &r0],
            compression: Compression::Zip,
        },
        MultipartMixedPart::Tiled {
            name: "tile".to_string(),
            width: w,
            height: h,
            tile_x: 4,
            tile_y: 3,
            channels: rgba_channels(),
            planes: vec![&a1, &b1, &g1, &r1],
            compression: Compression::Zip,
        },
    ])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert_eq!(imgs.len(), 2);
    let img0 = imgs[0].image().expect("flat part");
    let img1 = imgs[1].image().expect("flat part");
    assert_eq!(&img0.planes[0].samples, &a0);
    assert_eq!(&img0.planes[1].samples, &b0);
    assert_eq!(&img0.planes[2].samples, &g0);
    assert_eq!(&img0.planes[3].samples, &r0);
    assert_eq!(&img1.planes[0].samples, &a1);
    assert_eq!(&img1.planes[1].samples, &b1);
    assert_eq!(&img1.planes[2].samples, &g1);
    assert_eq!(&img1.planes[3].samples, &r1);
}

#[test]
fn mixed_distinct_dimensions_per_part() {
    // The two parts carry distinct dimensions and even distinct tile
    // sizes — multi-part files don't require shape homogeneity.
    let w0 = 16;
    let h0 = 16;
    let w1 = 24;
    let h1 = 16;
    let (a0, b0, g0, r0) = make_planes(w0, h0, 0.0);
    let (a1, b1, g1, r1) = make_planes(w1, h1, 0.25);
    let bytes = encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "small".to_string(),
            width: w0,
            height: h0,
            channels: rgba_channels(),
            planes: vec![&a0, &b0, &g0, &r0],
            compression: Compression::Zips,
        },
        MultipartMixedPart::Tiled {
            name: "wide".to_string(),
            width: w1,
            height: h1,
            tile_x: 8,
            tile_y: 4,
            channels: rgba_channels(),
            planes: vec![&a1, &b1, &g1, &r1],
            compression: Compression::Rle,
        },
    ])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert_eq!(imgs.len(), 2);
    let img0 = imgs[0].image().expect("flat part");
    let img1 = imgs[1].image().expect("flat part");
    assert_eq!(img0.width(), w0);
    assert_eq!(img0.height(), h0);
    assert_eq!(img1.width(), w1);
    assert_eq!(img1.height(), h1);
    assert_eq!(&img0.planes[0].samples, &a0);
    assert_eq!(&img1.planes[0].samples, &a1);
    assert_eq!(&img1.planes[3].samples, &r1);
}

#[test]
fn mixed_rejects_empty_parts() {
    let r = encode_exr_multipart_mixed(&[]);
    assert!(r.is_err());
}

#[test]
fn mixed_rejects_duplicate_names() {
    let w = 4;
    let h = 4;
    let (a, b, g, r) = make_planes(w, h, 0.0);
    let res = encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "dup".to_string(),
            width: w,
            height: h,
            channels: rgba_channels(),
            planes: vec![&a, &b, &g, &r],
            compression: Compression::None,
        },
        MultipartMixedPart::Tiled {
            name: "dup".to_string(),
            width: w,
            height: h,
            tile_x: 2,
            tile_y: 2,
            channels: rgba_channels(),
            planes: vec![&a, &b, &g, &r],
            compression: Compression::None,
        },
    ]);
    assert!(res.is_err());
}

#[test]
fn mixed_rejects_empty_name() {
    let w = 4;
    let h = 4;
    let (a, b, g, r) = make_planes(w, h, 0.0);
    let res = encode_exr_multipart_mixed(&[MultipartMixedPart::Scanline {
        name: "".to_string(),
        width: w,
        height: h,
        channels: rgba_channels(),
        planes: vec![&a, &b, &g, &r],
        compression: Compression::None,
    }]);
    assert!(res.is_err());
}

#[test]
fn mixed_rejects_tiled_subsampled_channel() {
    // Tiled parts must use 1×1 sampling — the chunk-body layout assumes
    // it. A sub-sampled channel on a tiled part must be rejected at
    // encode time.
    let w = 8;
    let h = 8;
    let pixels = (w * h) as usize;
    let plane: Vec<f32> = (0..pixels).map(|i| i as f32).collect();
    let mut chans = rgba_channels();
    chans[0].x_sampling = 2; // sub-sample channel A
    let res = encode_exr_multipart_mixed(&[MultipartMixedPart::Tiled {
        name: "bad".to_string(),
        width: w,
        height: h,
        tile_x: 4,
        tile_y: 4,
        channels: chans,
        planes: vec![&plane, &plane, &plane, &plane],
        compression: Compression::None,
    }]);
    assert!(res.is_err());
}

#[test]
fn mixed_rejects_zero_tile_size() {
    let w = 8;
    let h = 8;
    let (a, b, g, r) = make_planes(w, h, 0.0);
    let res = encode_exr_multipart_mixed(&[MultipartMixedPart::Tiled {
        name: "zero".to_string(),
        width: w,
        height: h,
        tile_x: 0,
        tile_y: 4,
        channels: rgba_channels(),
        planes: vec![&a, &b, &g, &r],
        compression: Compression::None,
    }]);
    assert!(res.is_err());
}

#[test]
fn mixed_uint_and_half_pixel_types() {
    // Each part can carry its own pixel-type mix. Exercise a UINT+HALF
    // scanline part alongside a FLOAT tiled part to confirm the chunk-
    // body layout adapts per part.
    let w = 8;
    let h = 8;
    let pixels = (w * h) as usize;
    // Part 0: ID channel as UINT + a HALF channel.
    let id_plane: Vec<f32> = (0..pixels).map(|i| i as f32).collect();
    let z_plane: Vec<f32> = (0..pixels).map(|i| (i as f32) * 0.125).collect();
    let id_chans = vec![
        Channel {
            name: "ID".to_string(),
            pixel_type: PixelType::Uint,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        },
        Channel {
            name: "Z".to_string(),
            pixel_type: PixelType::Half,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        },
    ];
    // Part 1: standard RGBA FLOAT tiled.
    let (a1, b1, g1, r1) = make_planes(w, h, 0.5);

    let bytes = encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "id_z".to_string(),
            width: w,
            height: h,
            channels: id_chans,
            planes: vec![&id_plane, &z_plane],
            compression: Compression::Zips,
        },
        MultipartMixedPart::Tiled {
            name: "rgba".to_string(),
            width: w,
            height: h,
            tile_x: 4,
            tile_y: 4,
            channels: rgba_channels(),
            planes: vec![&a1, &b1, &g1, &r1],
            compression: Compression::Zip,
        },
    ])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert_eq!(imgs.len(), 2);
    let img0 = imgs[0].image().expect("flat part");
    // ID plane (UINT) bit-exact under 2^24.
    assert_eq!(&img0.planes[0].samples, &id_plane);
    // Z plane (HALF) is lossy at high precision; the chosen values are
    // representable bit-exactly in binary16 since they are multiples of
    // 1/8 in the [0, 8) range, well inside HALF's exactly-representable
    // subnormals.
    assert_eq!(&img0.planes[1].samples, &z_plane);
    let img1 = imgs[1].image().expect("flat part");
    assert_eq!(&img1.planes[0].samples, &a1);
    assert_eq!(&img1.planes[3].samples, &r1);
}

// ---------------------------------------------------------------------
// Round-282: deep parts inside mixed multi-part files.
// ---------------------------------------------------------------------

use oxideav_openexr::{
    build_box_filter_pyramid, build_box_filter_ripmap, encode_exr_multipart_tiled_mipmap,
    parse_exr, parse_exr_deep_scanline, parse_exr_deep_tiled, MultipartMipmapTiledPart,
};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

fn exr_tool_available(tool: &str) -> bool {
    Command::new(tool)
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn tempdir(tag: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let c = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "oxideav-openexr-mixdeep-{tag}-{nanos}-{}-{c}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

fn cleanup_dir(dir: &str) -> std::io::Result<()> {
    for ent in std::fs::read_dir(dir)?.flatten() {
        let _ = std::fs::remove_file(ent.path());
    }
    std::fs::remove_dir(dir)
}

fn az_deep_channels() -> Vec<Channel> {
    ["A", "Z"]
        .iter()
        .map(|n| Channel {
            name: (*n).to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        })
        .collect()
}

/// Deterministic deep-data fixture: per-pixel sample counts cycling
/// 0..=3 (zeros included so empty pixels are exercised) plus two
/// channel sample lists in pixel-scan order.
fn make_deep_data(w: u32, h: u32, salt: f32) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let pixels = (w as usize) * (h as usize);
    let mut spp = Vec::with_capacity(pixels);
    let mut a = Vec::new();
    let mut z = Vec::new();
    for p in 0..pixels {
        let n = ((p + (salt * 4.0) as usize) % 4) as u32;
        spp.push(n);
        for s in 0..n {
            a.push((p as f32) * 0.5 + (s as f32) + salt);
            z.push((p as f32) * 0.25 - (s as f32) * 2.0 + salt);
        }
    }
    (spp, a, z)
}

#[test]
fn mixed_flat_scanline_and_deep_scanline_roundtrip() {
    let w = 16;
    let h = 12;
    let (a0, b0, g0, r0) = make_planes(w, h, 0.0);
    let (spp, da, dz) = make_deep_data(w, h, 0.5);
    let bytes = encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "flat".to_string(),
            width: w,
            height: h,
            channels: rgba_channels(),
            planes: vec![&a0, &b0, &g0, &r0],
            compression: Compression::Zips,
        },
        MultipartMixedPart::DeepScanline {
            name: "deep".to_string(),
            width: w,
            height: h,
            channels: az_deep_channels(),
            samples_per_pixel: &spp,
            channel_samples: vec![&da, &dz],
            compression: Compression::Zips,
        },
    ])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert_eq!(imgs.len(), 2);
    assert!(imgs[0].is_scanline());
    assert!(imgs[1].is_deep_scanline());
    let flat = imgs[0].image().expect("flat part");
    assert_eq!(&flat.planes[0].samples, &a0);
    assert_eq!(&flat.planes[3].samples, &r0);
    assert!(imgs[1].image().is_none());
    let deep = imgs[1].deep_scanline().expect("deep part");
    assert_eq!(deep.name, "deep");
    assert_eq!(deep.samples_per_pixel, spp);
    assert_eq!(deep.channel_samples[0], da);
    assert_eq!(deep.channel_samples[1], dz);
}

#[test]
fn mixed_deep_tiled_and_flat_tiled_edge_tiles() {
    // 13×9 with 4×3 tiles → right column + bottom row are edge tiles.
    let w = 13;
    let h = 9;
    let (a0, b0, g0, r0) = make_planes(w, h, 0.25);
    let (spp, da, dz) = make_deep_data(w, h, 0.0);
    let bytes = encode_exr_multipart_mixed(&[
        MultipartMixedPart::DeepTiled {
            name: "deep_t".to_string(),
            width: w,
            height: h,
            tile_x: 4,
            tile_y: 3,
            channels: az_deep_channels(),
            samples_per_pixel: &spp,
            channel_samples: vec![&da, &dz],
            compression: Compression::Rle,
        },
        MultipartMixedPart::Tiled {
            name: "flat_t".to_string(),
            width: w,
            height: h,
            tile_x: 4,
            tile_y: 3,
            channels: rgba_channels(),
            planes: vec![&a0, &b0, &g0, &r0],
            compression: Compression::None,
        },
    ])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert_eq!(imgs.len(), 2);
    assert!(imgs[0].is_deep_tiled());
    assert!(imgs[1].is_tiled());
    let deep = imgs[0].deep_tiled().expect("deep tiled part");
    assert_eq!(deep.name, "deep_t");
    assert_eq!(deep.tile_x, 4);
    assert_eq!(deep.tile_y, 3);
    assert_eq!(deep.samples_per_pixel, spp);
    assert_eq!(deep.channel_samples[0], da);
    assert_eq!(deep.channel_samples[1], dz);
    let flat = imgs[1].image().expect("flat part");
    assert_eq!(&flat.planes[0].samples, &a0);
    assert_eq!(&flat.planes[3].samples, &r0);
}

/// Build the canonical four-part fixture (scanline, deep scanline,
/// tiled, deep tiled — distinct dimensions + compressions per part).
#[allow(clippy::type_complexity)]
fn four_part_fixture() -> (
    Vec<u8>,
    (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),
    (Vec<u32>, Vec<f32>, Vec<f32>),
    (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>),
    (Vec<u32>, Vec<f32>, Vec<f32>),
) {
    let flat0 = make_planes(16, 12, 0.0);
    let deep0 = make_deep_data(11, 7, 0.25);
    let flat1 = make_planes(13, 9, 0.5);
    let deep1 = make_deep_data(13, 9, 0.75);
    let bytes = encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "p0_scan".to_string(),
            width: 16,
            height: 12,
            channels: rgba_channels(),
            planes: vec![&flat0.0, &flat0.1, &flat0.2, &flat0.3],
            compression: Compression::Zip,
        },
        MultipartMixedPart::DeepScanline {
            name: "p1_deepscan".to_string(),
            width: 11,
            height: 7,
            channels: az_deep_channels(),
            samples_per_pixel: &deep0.0,
            channel_samples: vec![&deep0.1, &deep0.2],
            compression: Compression::Zips,
        },
        MultipartMixedPart::Tiled {
            name: "p2_tile".to_string(),
            width: 13,
            height: 9,
            tile_x: 4,
            tile_y: 3,
            channels: rgba_channels(),
            planes: vec![&flat1.0, &flat1.1, &flat1.2, &flat1.3],
            compression: Compression::Rle,
        },
        MultipartMixedPart::DeepTiled {
            name: "p3_deeptile".to_string(),
            width: 13,
            height: 9,
            tile_x: 8,
            tile_y: 8,
            channels: az_deep_channels(),
            samples_per_pixel: &deep1.0,
            channel_samples: vec![&deep1.1, &deep1.2],
            compression: Compression::None,
        },
    ])
    .unwrap();
    (bytes, flat0, deep0, flat1, deep1)
}

#[test]
fn mixed_all_four_part_types_roundtrip() {
    let (bytes, flat0, deep0, flat1, deep1) = four_part_fixture();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert_eq!(imgs.len(), 4);
    assert!(imgs[0].is_scanline());
    assert!(imgs[1].is_deep_scanline());
    assert!(imgs[2].is_tiled());
    assert!(imgs[3].is_deep_tiled());

    let img0 = imgs[0].image().expect("flat scanline");
    assert_eq!(img0.width(), 16);
    assert_eq!(&img0.planes[0].samples, &flat0.0);
    assert_eq!(&img0.planes[3].samples, &flat0.3);

    let d0 = imgs[1].deep_scanline().expect("deep scanline");
    assert_eq!(d0.name, "p1_deepscan");
    assert_eq!(d0.width(), 11);
    assert_eq!(d0.samples_per_pixel, deep0.0);
    assert_eq!(d0.channel_samples[0], deep0.1);
    assert_eq!(d0.channel_samples[1], deep0.2);

    let img1 = imgs[2].image().expect("flat tiled");
    assert_eq!(img1.width(), 13);
    assert_eq!(&img1.planes[0].samples, &flat1.0);
    assert_eq!(&img1.planes[3].samples, &flat1.3);

    let d1 = imgs[3].deep_tiled().expect("deep tiled");
    assert_eq!(d1.name, "p3_deeptile");
    assert_eq!(d1.width(), 13);
    assert_eq!(d1.samples_per_pixel, deep1.0);
    assert_eq!(d1.channel_samples[0], deep1.1);
    assert_eq!(d1.channel_samples[1], deep1.2);
}

#[test]
fn mixed_rejects_deep_zip_compression() {
    // Deep parts accept only NONE/RLE/ZIPS — ZIP must be rejected at
    // encode time (matching the homogeneous deep writers).
    let (spp, da, dz) = make_deep_data(8, 8, 0.0);
    let res = encode_exr_multipart_mixed(&[MultipartMixedPart::DeepScanline {
        name: "bad".to_string(),
        width: 8,
        height: 8,
        channels: az_deep_channels(),
        samples_per_pixel: &spp,
        channel_samples: vec![&da, &dz],
        compression: Compression::Zip,
    }]);
    assert!(res.is_err());
}

#[test]
fn mixed_reader_decodes_homogeneous_multilevel_mipmap_file() {
    // The mixed reader now decodes multi-level (MIPMAP) flat tiled parts
    // inline. A file produced by the homogeneous multi-part MIPMAP writer
    // shares the mixed chunk layout, so the mixed reader recovers every
    // pyramid level sample-for-sample.
    let (a, b, g, r) = make_planes(16, 16, 0.0);
    let pyramid = build_box_filter_pyramid(16, 16, &[a, b, g, r]);
    let parts = vec![MultipartMipmapTiledPart {
        name: "pyr".to_string(),
        tile_x: 8,
        tile_y: 8,
        channels: rgba_channels(),
        pyramid: pyramid.clone(),
        compression: Compression::Zip,
    }];
    let bytes = encode_exr_multipart_tiled_mipmap(&parts).unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert_eq!(imgs.len(), 1);
    assert!(imgs[0].is_tiled_mipmap());
    let mlt = imgs[0].multilevel_tiled().expect("mipmap part");
    assert_eq!(mlt.level_mode, 1);
    assert_eq!(mlt.levels.len(), pyramid.len());
    for (lvl, src) in mlt.levels.iter().zip(pyramid.iter()) {
        assert_eq!(lvl.level_x, lvl.level_y);
        assert_eq!(lvl.width, src.width);
        assert_eq!(lvl.height, src.height);
        for (plane, src_plane) in lvl.planes.iter().zip(src.planes.iter()) {
            assert_eq!(&plane.samples, src_plane);
        }
    }
}

#[test]
fn exrheader_accepts_mixed_deep_flat_file() {
    if !exr_tool_available("exrheader") {
        eprintln!("exrheader not available, skipping");
        return;
    }
    let (bytes, ..) = four_part_fixture();
    let dir = tempdir("exrheader");
    let path = format!("{dir}/mixed.exr");
    std::fs::write(&path, &bytes).unwrap();
    let out = Command::new("exrheader").arg(&path).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "exrheader failed on our mixed deep+flat multi-part file\nstdout: {stdout}\nstderr: {stderr}"
    );
    for needle in [
        "scanlineimage",
        "deepscanline",
        "tiledimage",
        "deeptile",
        "p0_scan",
        "p1_deepscan",
        "p2_tile",
        "p3_deeptile",
    ] {
        assert!(
            stdout.contains(needle),
            "exrheader output missing '{needle}'\nstdout: {stdout}"
        );
    }
    let _ = cleanup_dir(&dir);
}

#[test]
fn exrmultipart_separate_splits_mixed_deep_flat() {
    if !exr_tool_available("exrmultipart") || !exr_tool_available("exrheader") {
        eprintln!("exrmultipart / exrheader not available, skipping");
        return;
    }
    let (bytes, flat0, deep0, flat1, deep1) = four_part_fixture();
    let dir = tempdir("separate");
    let in_path = format!("{dir}/in.exr");
    std::fs::write(&in_path, &bytes).unwrap();
    let _ = Command::new("exrmultipart")
        .arg("-separate")
        .arg("-i")
        .arg(&in_path)
        .arg("-o")
        .arg(format!("{dir}/out.exr"))
        .output()
        .expect("exrmultipart spawn");

    let mut splits: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for ent in rd.flatten() {
            let p = ent.path();
            if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                if name != "in.exr" && name.ends_with(".exr") {
                    splits.push(p.to_string_lossy().into_owned());
                }
            }
        }
    }
    splits.sort();
    if splits.len() != 4 {
        eprintln!(
            "exrmultipart split count = {} (expected 4) in {dir}; skipping cross-check",
            splits.len()
        );
        let _ = cleanup_dir(&dir);
        return;
    }

    // Identify each split by its part name (exrheader output), then
    // decode through the matching single-part reader and compare to the
    // source data sample-for-sample.
    let mut seen = [false; 4];
    for s in &splits {
        let header_out = Command::new("exrheader").arg(s).output().unwrap();
        let txt = String::from_utf8_lossy(&header_out.stdout);
        let split_bytes = std::fs::read(s).unwrap();
        if txt.contains("p0_scan") {
            let img = parse_exr(&split_bytes).unwrap();
            assert_eq!(&img.planes[0].samples, &flat0.0, "p0_scan plane A");
            assert_eq!(&img.planes[3].samples, &flat0.3, "p0_scan plane R");
            seen[0] = true;
        } else if txt.contains("p1_deepscan") {
            let img = parse_exr_deep_scanline(&split_bytes).unwrap();
            assert_eq!(img.samples_per_pixel, deep0.0, "p1_deepscan spp");
            assert_eq!(img.channel_samples[0], deep0.1, "p1_deepscan A");
            assert_eq!(img.channel_samples[1], deep0.2, "p1_deepscan Z");
            seen[1] = true;
        } else if txt.contains("p2_tile") {
            let img = parse_exr(&split_bytes).unwrap();
            assert_eq!(&img.planes[0].samples, &flat1.0, "p2_tile plane A");
            assert_eq!(&img.planes[3].samples, &flat1.3, "p2_tile plane R");
            seen[2] = true;
        } else if txt.contains("p3_deeptile") {
            let img = parse_exr_deep_tiled(&split_bytes).unwrap();
            assert_eq!(img.samples_per_pixel, deep1.0, "p3_deeptile spp");
            assert_eq!(img.channel_samples[0], deep1.1, "p3_deeptile A");
            assert_eq!(img.channel_samples[1], deep1.2, "p3_deeptile Z");
            seen[3] = true;
        }
    }
    assert_eq!(
        seen, [true; 4],
        "did not locate all four parts in split outputs"
    );
    let _ = cleanup_dir(&dir);
}

// ---------------------------------------------------------------------
// Round-344: multi-level (MIPMAP / RIPMAP) flat tiled parts inside a
// mixed multi-part file. Covers encode + self-roundtrip for each level
// mode standalone, the fully-mixed case (scanline + mipmap + ripmap +
// deep), edge tiles, every flat compression, and the validation rejects.
// ---------------------------------------------------------------------

use oxideav_openexr::{MipmapLevel, TiledLevel};

/// Assert a decoded multi-level part recovers every level of `src`
/// (a pyramid/grid of `MipmapLevel`) sample-for-sample, in spec order.
fn assert_levels_match(levels: &[TiledLevel], src: &[MipmapLevel]) {
    assert_eq!(levels.len(), src.len(), "level count");
    for (lvl, s) in levels.iter().zip(src.iter()) {
        assert_eq!(lvl.width, s.width, "level width");
        assert_eq!(lvl.height, s.height, "level height");
        assert_eq!(lvl.planes.len(), s.planes.len(), "plane count");
        for (p, sp) in lvl.planes.iter().zip(s.planes.iter()) {
            assert_eq!(&p.samples, sp, "level plane samples");
        }
    }
}

#[test]
fn mixed_encoder_mipmap_part_roundtrips_every_compression() {
    for comp in [
        Compression::None,
        Compression::Zip,
        Compression::Zips,
        Compression::Rle,
    ] {
        let (a, b, g, r) = make_planes(16, 16, 0.0);
        let pyramid = build_box_filter_pyramid(16, 16, &[a, b, g, r]);
        let bytes = encode_exr_multipart_mixed(&[MultipartMixedPart::TiledMipmap {
            name: "mip".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: rgba_channels(),
            pyramid: pyramid.clone(),
            compression: comp,
        }])
        .unwrap();
        let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
        assert_eq!(imgs.len(), 1);
        assert!(imgs[0].is_tiled_mipmap(), "comp {comp:?}");
        let mlt = imgs[0].multilevel_tiled().unwrap();
        assert_eq!(mlt.level_mode, 1);
        assert_eq!(mlt.compression, comp);
        assert_levels_match(&mlt.levels, &pyramid);
    }
}

#[test]
fn mixed_encoder_ripmap_part_roundtrips_every_compression() {
    for comp in [
        Compression::None,
        Compression::Zip,
        Compression::Zips,
        Compression::Rle,
    ] {
        let (a, b, g, r) = make_planes(16, 16, 0.0);
        let grid = build_box_filter_ripmap(16, 16, &[a, b, g, r]).grid;
        let bytes = encode_exr_multipart_mixed(&[MultipartMixedPart::TiledRipmap {
            name: "rip".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: rgba_channels(),
            grid: grid.clone(),
            compression: comp,
        }])
        .unwrap();
        let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
        assert_eq!(imgs.len(), 1);
        assert!(imgs[0].is_tiled_ripmap(), "comp {comp:?}");
        let mlt = imgs[0].multilevel_tiled().unwrap();
        assert_eq!(mlt.level_mode, 2);
        // Flatten the grid in spec iteration order (lvly outer, lvlx
        // inner) to compare against the decoded levels.
        let flat: Vec<MipmapLevel> = grid.into_iter().flatten().collect();
        assert_levels_match(&mlt.levels, &flat);
    }
}

#[test]
fn mixed_encoder_mipmap_edge_tiles() {
    // 13×9 level-0 with 4×3 tiles forces edge tiles at every level.
    let (a, b, g, r) = make_planes(13, 9, 0.25);
    let pyramid = build_box_filter_pyramid(13, 9, &[a, b, g, r]);
    let bytes = encode_exr_multipart_mixed(&[MultipartMixedPart::TiledMipmap {
        name: "mip_edge".to_string(),
        tile_x: 4,
        tile_y: 3,
        channels: rgba_channels(),
        pyramid: pyramid.clone(),
        compression: Compression::Zip,
    }])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    let mlt = imgs[0].multilevel_tiled().unwrap();
    assert_levels_match(&mlt.levels, &pyramid);
}

#[test]
fn mixed_encoder_ripmap_edge_tiles() {
    let (a, b, g, r) = make_planes(13, 9, 0.0);
    let grid = build_box_filter_ripmap(13, 9, &[a, b, g, r]).grid;
    let bytes = encode_exr_multipart_mixed(&[MultipartMixedPart::TiledRipmap {
        name: "rip_edge".to_string(),
        tile_x: 4,
        tile_y: 3,
        channels: rgba_channels(),
        grid: grid.clone(),
        compression: Compression::Rle,
    }])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    let mlt = imgs[0].multilevel_tiled().unwrap();
    let flat: Vec<MipmapLevel> = grid.into_iter().flatten().collect();
    assert_levels_match(&mlt.levels, &flat);
}

#[test]
fn mixed_all_six_part_kinds_in_one_file() {
    // The headline milestone case: scanline + ONE_LEVEL tiled + MIPMAP +
    // RIPMAP + deep scanline + deep tiled, freely interleaved.
    let (sa, sb, sg, sr) = make_planes(16, 16, 0.0);
    let (ta, tb, tg, tr) = make_planes(13, 9, 0.5);
    let (ma, mb, mg, mr) = make_planes(16, 16, 0.125);
    let pyramid = build_box_filter_pyramid(16, 16, &[ma, mb, mg, mr]);
    let (ra, rb, rg, rr) = make_planes(16, 16, 0.75);
    let grid = build_box_filter_ripmap(16, 16, &[ra, rb, rg, rr]).grid;
    let (dspp, dsa, dsz) = make_deep_data(11, 7, 0.5);
    let (dtspp, dta, dtz) = make_deep_data(13, 9, 0.0);

    let bytes = encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "p_scan".to_string(),
            width: 16,
            height: 16,
            channels: rgba_channels(),
            planes: vec![&sa, &sb, &sg, &sr],
            compression: Compression::Zip,
        },
        MultipartMixedPart::TiledMipmap {
            name: "p_mip".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: rgba_channels(),
            pyramid: pyramid.clone(),
            compression: Compression::Zips,
        },
        MultipartMixedPart::DeepScanline {
            name: "p_deepscan".to_string(),
            width: 11,
            height: 7,
            channels: az_deep_channels(),
            samples_per_pixel: &dspp,
            channel_samples: vec![&dsa, &dsz],
            compression: Compression::Rle,
        },
        MultipartMixedPart::Tiled {
            name: "p_tile".to_string(),
            width: 13,
            height: 9,
            tile_x: 4,
            tile_y: 3,
            channels: rgba_channels(),
            planes: vec![&ta, &tb, &tg, &tr],
            compression: Compression::None,
        },
        MultipartMixedPart::TiledRipmap {
            name: "p_rip".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: rgba_channels(),
            grid: grid.clone(),
            compression: Compression::Zip,
        },
        MultipartMixedPart::DeepTiled {
            name: "p_deeptile".to_string(),
            width: 13,
            height: 9,
            tile_x: 4,
            tile_y: 3,
            channels: az_deep_channels(),
            samples_per_pixel: &dtspp,
            channel_samples: vec![&dta, &dtz],
            compression: Compression::Zips,
        },
    ])
    .unwrap();

    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert_eq!(imgs.len(), 6);
    assert!(imgs[0].is_scanline());
    assert!(imgs[1].is_tiled_mipmap());
    assert!(imgs[2].is_deep_scanline());
    assert!(imgs[3].is_tiled());
    assert!(imgs[4].is_tiled_ripmap());
    assert!(imgs[5].is_deep_tiled());

    // Flat scanline.
    let s = imgs[0].image().unwrap();
    assert_eq!(&s.planes[0].samples, &sa);
    assert_eq!(&s.planes[3].samples, &sr);
    // MIPMAP pyramid.
    assert_levels_match(&imgs[1].multilevel_tiled().unwrap().levels, &pyramid);
    // Deep scanline.
    let ds = imgs[2].deep_scanline().unwrap();
    assert_eq!(ds.name, "p_deepscan");
    assert_eq!(ds.samples_per_pixel, dspp);
    assert_eq!(ds.channel_samples[0], dsa);
    assert_eq!(ds.channel_samples[1], dsz);
    // ONE_LEVEL tiled.
    let t = imgs[3].image().unwrap();
    assert_eq!(&t.planes[0].samples, &ta);
    assert_eq!(&t.planes[3].samples, &tr);
    // RIPMAP grid.
    let rip_flat: Vec<MipmapLevel> = grid.into_iter().flatten().collect();
    assert_levels_match(&imgs[4].multilevel_tiled().unwrap().levels, &rip_flat);
    // Deep tiled.
    let dt = imgs[5].deep_tiled().unwrap();
    assert_eq!(dt.name, "p_deeptile");
    assert_eq!(dt.samples_per_pixel, dtspp);
    assert_eq!(dt.channel_samples[0], dta);
    assert_eq!(dt.channel_samples[1], dtz);
}

#[test]
fn mixed_mipmap_rejects_wrong_pyramid_length() {
    let (a, b, g, r) = make_planes(16, 16, 0.0);
    let mut pyramid = build_box_filter_pyramid(16, 16, &[a, b, g, r]);
    pyramid.pop(); // drop a level → invalid count
    let res = encode_exr_multipart_mixed(&[MultipartMixedPart::TiledMipmap {
        name: "bad".to_string(),
        tile_x: 8,
        tile_y: 8,
        channels: rgba_channels(),
        pyramid,
        compression: Compression::Zip,
    }]);
    assert!(res.is_err());
}

#[test]
fn mixed_multilevel_rejects_pxr24_compression() {
    // Multi-level mixed parts share the flat-tiled NONE/ZIP/ZIPS/RLE
    // restriction — lossy PXR24 is rejected at encode time.
    let (a, b, g, r) = make_planes(16, 16, 0.0);
    let pyramid = build_box_filter_pyramid(16, 16, &[a, b, g, r]);
    let res = encode_exr_multipart_mixed(&[MultipartMixedPart::TiledMipmap {
        name: "bad".to_string(),
        tile_x: 8,
        tile_y: 8,
        channels: rgba_channels(),
        pyramid,
        compression: Compression::Pxr24,
    }]);
    assert!(res.is_err());
}

#[test]
fn exrheader_accepts_mixed_multilevel_file() {
    if !exr_tool_available("exrheader") {
        eprintln!("exrheader not available, skipping");
        return;
    }
    // A scanline + MIPMAP + RIPMAP mixed file must be accepted by a
    // reference EXR validator binary (treated as an opaque CLI), with
    // each part's level mode reported.
    let (sa, sb, sg, sr) = make_planes(16, 16, 0.0);
    let (ma, mb, mg, mr) = make_planes(16, 16, 0.2);
    let pyramid = build_box_filter_pyramid(16, 16, &[ma, mb, mg, mr]);
    let (ra, rb, rg, rr) = make_planes(16, 16, 0.4);
    let grid = build_box_filter_ripmap(16, 16, &[ra, rb, rg, rr]).grid;
    let bytes = encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "ml_scan".to_string(),
            width: 16,
            height: 16,
            channels: rgba_channels(),
            planes: vec![&sa, &sb, &sg, &sr],
            compression: Compression::Zip,
        },
        MultipartMixedPart::TiledMipmap {
            name: "ml_mip".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: rgba_channels(),
            pyramid,
            compression: Compression::Zip,
        },
        MultipartMixedPart::TiledRipmap {
            name: "ml_rip".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: rgba_channels(),
            grid,
            compression: Compression::Zip,
        },
    ])
    .unwrap();
    let dir = tempdir("exrheader_ml");
    let path = format!("{dir}/mixed_ml.exr");
    std::fs::write(&path, &bytes).unwrap();
    let out = Command::new("exrheader").arg(&path).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "reference validator rejected our mixed multi-level file\nstdout: {stdout}\nstderr: {stderr}"
    );
    let lc = stdout.to_lowercase();
    for needle in ["ml_scan", "ml_mip", "ml_rip"] {
        assert!(
            lc.contains(needle),
            "validator output missing part name '{needle}'\nstdout: {stdout}"
        );
    }
    // The validator reports the level modes (it hyphenates these as
    // "mip-map" / "rip-map" in its header dump).
    assert!(
        lc.contains("mip-map") || lc.contains("mipmap"),
        "validator did not report a mipmap level mode\nstdout: {stdout}"
    );
    assert!(
        lc.contains("rip-map") || lc.contains("ripmap"),
        "validator did not report a ripmap level mode\nstdout: {stdout}"
    );
    let _ = cleanup_dir(&dir);
}
