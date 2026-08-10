//! PIZ across the tiled / multilevel / multi-part surface
//! (observer-spec `openexr-piz-dwa-observer-spec.md` §2 — a tile is a
//! self-contained chunk with origin row 0 and 1×1 sampling).
//!
//! PIZ is lossless, so every comparison here is bit-exact: self
//! round-trips through each reader, plus reference-binary cross-checks
//! (opaque process; auto-skip when absent) in both directions for the
//! tiled paths.

use std::process::Command;

use oxideav_openexr::{
    build_box_filter_pyramid, build_box_filter_ripmap, encode_exr_multipart,
    encode_exr_multipart_mixed, encode_exr_multipart_tiled, encode_exr_multipart_tiled_mipmap,
    encode_exr_multipart_tiled_ripmap, encode_exr_tiled, encode_exr_tiled_mipmap,
    encode_exr_tiled_ripmap, parse_exr, parse_exr_multipart, parse_exr_multipart_mixed,
    parse_exr_multipart_tiled, parse_exr_multipart_tiled_multilevel, parse_exr_tiled_multilevel,
    Channel, Compression, MipmapLevel, MultipartMipmapTiledPart, MultipartMixedImage,
    MultipartMixedPart, MultipartRipmapTiledPart, MultipartScanlinePart, MultipartTiledPart,
    PixelType,
};

fn exrmetrics_available() -> bool {
    Command::new("exrmetrics")
        .arg("--help")
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

fn tempdir() -> std::path::PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-piz-tiled-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn reference_convert(input_bytes: &[u8], z: &str) -> Option<Vec<u8>> {
    if !exrmetrics_available() {
        eprintln!("exrmetrics not available, skipping");
        return None;
    }
    let dir = tempdir();
    let in_path = dir.join("in.exr");
    let out_path = dir.join("out.exr");
    std::fs::write(&in_path, input_bytes).unwrap();
    let output = Command::new("exrmetrics")
        .arg("--convert")
        .arg("-z")
        .arg(z)
        .arg(&in_path)
        .arg("-o")
        .arg(&out_path)
        .output()
        .ok()?;
    assert!(
        output.status.success(),
        "reference conversion to {z} failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let out = std::fs::read(&out_path).unwrap();
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    let _ = std::fs::remove_dir(&dir);
    Some(out)
}

fn ch(name: &str, pt: PixelType) -> Channel {
    Channel {
        name: name.to_string(),
        pixel_type: pt,
        p_linear: false,
        x_sampling: 1,
        y_sampling: 1,
    }
}

/// Half-friendly structured plane: quarter steps and outliers.
fn plane(w: u32, h: u32, seed: u32) -> Vec<f32> {
    (0..(w * h) as usize)
        .map(|i| {
            let k = (i as u32).wrapping_mul(7).wrapping_add(seed) % 53;
            if k == 51 {
                2048.0
            } else {
                (k as f32) * 0.25
            }
        })
        .collect()
}

#[test]
fn piz_tiled_one_level_roundtrip_and_reference() {
    // Odd image size + edge tiles.
    let (w, h) = (45u32, 21u32);
    let chs = vec![ch("G", PixelType::Half), ch("Z", PixelType::Float)];
    let g = plane(w, h, 1);
    let z = plane(w, h, 9);
    let planes: Vec<&[f32]> = vec![&g, &z];
    let bytes = encode_exr_tiled(w, h, &chs, &planes, Compression::Piz, 16, 16).unwrap();
    let img = parse_exr(&bytes).unwrap();
    assert_eq!(img.compression, Compression::Piz);
    for (i, &v) in g.iter().enumerate() {
        assert_eq!(img.planes[0].samples[i], v, "G {i}");
    }
    for (i, &v) in z.iter().enumerate() {
        assert_eq!(img.planes[1].samples[i].to_bits(), v.to_bits(), "Z {i}");
    }
    // Reference decode of our tiled PIZ file.
    if let Some(none_bytes) = reference_convert(&bytes, "none") {
        let img_ref = parse_exr(&none_bytes).unwrap();
        for (pa, pb) in img.planes.iter().zip(img_ref.planes.iter()) {
            for (i, (a, b)) in pa.samples.iter().zip(pb.samples.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "reference decode differs at {i}");
            }
        }
    }
    // Reference PIZ encode of the equivalent NONE tiled file, decoded by us.
    let none_ours = encode_exr_tiled(w, h, &chs, &planes, Compression::None, 16, 16).unwrap();
    if let Some(piz_ref) = reference_convert(&none_ours, "piz") {
        let img_ref = parse_exr(&piz_ref).unwrap();
        assert_eq!(img_ref.compression, Compression::Piz);
        for (pa, pb) in img.planes.iter().zip(img_ref.planes.iter()) {
            for (i, (a, b)) in pa.samples.iter().zip(pb.samples.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "our decode of reference PIZ differs at {i}"
                );
            }
        }
    }
}

#[test]
fn piz_mipmap_and_ripmap_roundtrip() {
    let (w, h) = (40u32, 24u32);
    let chs = vec![ch("G", PixelType::Half)];
    let p = plane(w, h, 3);
    let pyramid = build_box_filter_pyramid(w, h, std::slice::from_ref(&p));
    let bytes = encode_exr_tiled_mipmap(&chs, &pyramid, Compression::Piz, 16, 16).unwrap();
    let img = parse_exr_tiled_multilevel(&bytes).unwrap();
    assert_eq!(img.levels.len(), pyramid.len());
    for (lvl, want) in img.levels.iter().zip(pyramid.iter()) {
        for (i, (got, want_v)) in lvl.planes[0]
            .samples
            .iter()
            .zip(want.planes[0].iter())
            .enumerate()
        {
            // The box filter output is f32; HALF storage rounds it once.
            let want_h =
                oxideav_openexr::half::half_to_f32(oxideav_openexr::half::f32_to_half(*want_v));
            assert_eq!(*got, want_h, "mipmap level sample {i}");
        }
    }

    let ripmap = build_box_filter_ripmap(w, h, std::slice::from_ref(&p));
    let bytes = encode_exr_tiled_ripmap(&chs, &ripmap, Compression::Piz, 8, 8).unwrap();
    let img = parse_exr_tiled_multilevel(&bytes).unwrap();
    let expected_cells: usize = ripmap.grid.iter().map(|row| row.len()).sum();
    assert_eq!(img.levels.len(), expected_cells);
}

#[test]
fn piz_multipart_scanline_roundtrip() {
    let (w, h) = (33u32, 70u32);
    let g0 = plane(w, h, 5);
    let g1 = plane(w, h, 11);
    let parts = vec![
        MultipartScanlinePart {
            name: "left".to_string(),
            width: w,
            height: h,
            channels: vec![ch("G", PixelType::Half)],
            planes: vec![&g0],
            compression: Compression::Piz,
        },
        MultipartScanlinePart {
            name: "right".to_string(),
            width: w,
            height: h,
            channels: vec![ch("G", PixelType::Half)],
            planes: vec![&g1],
            compression: Compression::Zip,
        },
    ];
    let bytes = encode_exr_multipart(&parts).unwrap();
    let imgs = parse_exr_multipart(&bytes).unwrap();
    assert_eq!(imgs.len(), 2);
    for (i, &v) in g0.iter().enumerate() {
        assert_eq!(imgs[0].planes[0].samples[i], v, "part0 {i}");
    }
    for (i, &v) in g1.iter().enumerate() {
        assert_eq!(imgs[1].planes[0].samples[i], v, "part1 {i}");
    }
}

#[test]
fn piz_multipart_tiled_and_multilevel_roundtrip() {
    let (w, h) = (26u32, 18u32);
    let g = plane(w, h, 2);
    let tiled_parts = vec![MultipartTiledPart {
        name: "t0".to_string(),
        width: w,
        height: h,
        tile_x: 8,
        tile_y: 8,
        channels: vec![ch("G", PixelType::Half)],
        planes: vec![&g],
        compression: Compression::Piz,
    }];
    let bytes = encode_exr_multipart_tiled(&tiled_parts).unwrap();
    let imgs = parse_exr_multipart_tiled(&bytes).unwrap();
    for (i, &v) in g.iter().enumerate() {
        assert_eq!(imgs[0].planes[0].samples[i], v, "tiled {i}");
    }

    let pyramid = build_box_filter_pyramid(w, h, std::slice::from_ref(&g));
    let pyr_levels: Vec<MipmapLevel> = pyramid.clone();
    let parts = vec![MultipartMipmapTiledPart {
        name: "m0".to_string(),
        tile_x: 8,
        tile_y: 8,
        channels: vec![ch("G", PixelType::Half)],
        pyramid: pyr_levels,
        compression: Compression::Piz,
    }];
    let bytes = encode_exr_multipart_tiled_mipmap(&parts).unwrap();
    let mp = parse_exr_multipart_tiled_multilevel(&bytes).unwrap();
    assert_eq!(mp[0].levels.len(), pyramid.len());

    let ripmap = build_box_filter_ripmap(w, h, std::slice::from_ref(&g));
    let rparts = vec![MultipartRipmapTiledPart {
        name: "r0".to_string(),
        tile_x: 8,
        tile_y: 8,
        channels: vec![ch("G", PixelType::Half)],
        pyramid: ripmap,
        compression: Compression::Piz,
    }];
    let bytes = encode_exr_multipart_tiled_ripmap(&rparts).unwrap();
    let mp = parse_exr_multipart_tiled_multilevel(&bytes).unwrap();
    assert!(!mp[0].levels.is_empty());
}

#[test]
fn piz_multipart_mixed_roundtrip() {
    let (w, h) = (24u32, 40u32);
    let s = plane(w, h, 7);
    let t = plane(w, h, 13);
    let parts = vec![
        MultipartMixedPart::Scanline {
            name: "scan".to_string(),
            width: w,
            height: h,
            channels: vec![ch("G", PixelType::Half)],
            planes: vec![&s],
            compression: Compression::Piz,
        },
        MultipartMixedPart::Tiled {
            name: "tile".to_string(),
            width: w,
            height: h,
            tile_x: 16,
            tile_y: 16,
            channels: vec![ch("G", PixelType::Half)],
            planes: vec![&t],
            compression: Compression::Piz,
        },
    ];
    let bytes = encode_exr_multipart_mixed(&parts).unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert_eq!(imgs.len(), 2);
    let MultipartMixedImage::Scanline(img0) = &imgs[0] else {
        panic!("part 0 should be scanline");
    };
    let MultipartMixedImage::Tiled(img1) = &imgs[1] else {
        panic!("part 1 should be tiled");
    };
    for (i, &v) in s.iter().enumerate() {
        assert_eq!(img0.planes[0].samples[i], v, "scan {i}");
    }
    for (i, &v) in t.iter().enumerate() {
        assert_eq!(img1.planes[0].samples[i], v, "tile {i}");
    }
}
