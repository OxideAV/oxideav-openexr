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
        let image = img.image();
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
    let img0 = imgs[0].image();
    let img1 = imgs[1].image();
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
        let image = img.image();
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
    let img0 = imgs[0].image();
    let img1 = imgs[1].image();
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
    let img0 = imgs[0].image();
    let img1 = imgs[1].image();
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
    let img0 = imgs[0].image();
    // ID plane (UINT) bit-exact under 2^24.
    assert_eq!(&img0.planes[0].samples, &id_plane);
    // Z plane (HALF) is lossy at high precision; the chosen values are
    // representable bit-exactly in binary16 since they are multiples of
    // 1/8 in the [0, 8) range, well inside HALF's exactly-representable
    // subnormals.
    assert_eq!(&img0.planes[1].samples, &z_plane);
    let img1 = imgs[1].image();
    assert_eq!(&img1.planes[0].samples, &a1);
    assert_eq!(&img1.planes[3].samples, &r1);
}
