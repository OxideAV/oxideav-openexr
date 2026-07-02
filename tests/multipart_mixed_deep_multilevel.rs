//! Round-382 self-roundtrip + reject-path validation for multi-level
//! (MIPMAP / RIPMAP) **deep** tiled parts inside the mixed multi-part
//! WRITE + READ pair (`encode_exr_multipart_mixed` /
//! `parse_exr_multipart_mixed`).
//!
//! Previously a mixed file could only carry ONE_LEVEL deep tiled parts;
//! MIPMAP / RIPMAP deep tiled files needed the dedicated
//! `parse_exr_multipart_deep_tiled_{mipmap,ripmap}` readers. These tests
//! exercise the new inline mixed-path support at every deep compression
//! (NONE / ZIPS / RLE), for both level modes, including edge tiles and
//! files that interleave the multi-level deep parts with flat + deep
//! ONE_LEVEL parts.
#![allow(clippy::type_complexity)]

use oxideav_openexr::{
    encode_exr_multipart_mixed, parse_exr_multipart_mixed, Channel, Compression,
    DeepMipmapTiledLevelInput, DeepRipmapTiledLevelInput, MultipartMixedPart, PixelType,
};

fn channels_rgba_float() -> Vec<Channel> {
    ["A", "B", "G", "R"]
        .iter()
        .map(|n| Channel {
            name: n.to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        })
        .collect()
}

/// Synthetic deep pixel data for one level: variable per-pixel sample
/// counts (0..3) and four channel planes filled distinctly.
fn build_level(w: u32, h: u32, salt: f32) -> (Vec<u32>, [Vec<f32>; 4]) {
    let pixels = (w * h) as usize;
    let spp: Vec<u32> = (0..pixels).map(|i| (i as u32) % 4).collect();
    let total: usize = spp.iter().sum::<u32>() as usize;
    let mk = |scale: f32| -> Vec<f32> { (0..total).map(|i| (i as f32) * scale + salt).collect() };
    (spp, [mk(0.05), mk(0.1), mk(0.15), mk(0.2)])
}

fn level_dim(full: u32, level: u32) -> u32 {
    (full >> level).max(1)
}

fn mipmap_level_count(w0: u32, h0: u32) -> u32 {
    let mut d = w0.max(h0);
    let mut n = 1u32;
    while d > 1 {
        d /= 2;
        n += 1;
    }
    n
}

/// Build all backing per-level (spp, planes) tuples for a ROUND_DOWN
/// mipmap pyramid.
fn build_mipmap_pyramid(w0: u32, h0: u32) -> Vec<(Vec<u32>, [Vec<f32>; 4])> {
    let n = mipmap_level_count(w0, h0);
    (0..n)
        .map(|l| build_level(level_dim(w0, l), level_dim(h0, l), l as f32))
        .collect()
}

/// Build all backing cells for a ROUND_DOWN ripmap grid, `grid[ly][lx]`.
fn build_ripmap_grid(w0: u32, h0: u32) -> Vec<Vec<(Vec<u32>, [Vec<f32>; 4])>> {
    let nx = mipmap_level_count(w0, w0); // count over width only
    let ny = mipmap_level_count(h0, h0);
    (0..ny)
        .map(|ly| {
            (0..nx)
                .map(|lx| build_level(level_dim(w0, lx), level_dim(h0, ly), (ly * nx + lx) as f32))
                .collect()
        })
        .collect()
}

/// MIPMAP level inputs with correct dimensions attached.
fn mipmap_inputs_with_dims<'a>(
    w0: u32,
    h0: u32,
    pyr: &'a [(Vec<u32>, [Vec<f32>; 4])],
) -> Vec<DeepMipmapTiledLevelInput<'a>> {
    pyr.iter()
        .enumerate()
        .map(|(l, (spp, planes))| DeepMipmapTiledLevelInput {
            width: level_dim(w0, l as u32),
            height: level_dim(h0, l as u32),
            samples_per_pixel: spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
        })
        .collect()
}

/// RIPMAP grid inputs with correct dimensions attached.
fn ripmap_inputs_with_dims<'a>(
    w0: u32,
    h0: u32,
    grid: &'a [Vec<(Vec<u32>, [Vec<f32>; 4])>],
) -> Vec<Vec<DeepRipmapTiledLevelInput<'a>>> {
    grid.iter()
        .enumerate()
        .map(|(ly, row)| {
            row.iter()
                .enumerate()
                .map(|(lx, (spp, planes))| DeepRipmapTiledLevelInput {
                    width: level_dim(w0, lx as u32),
                    height: level_dim(h0, ly as u32),
                    samples_per_pixel: spp,
                    channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
                })
                .collect()
        })
        .collect()
}

fn deep_compressions() -> [Compression; 3] {
    [Compression::None, Compression::Zips, Compression::Rle]
}

// ---------------------------------------------------------------------
// MIPMAP self-roundtrip.
// ---------------------------------------------------------------------

fn assert_mipmap_roundtrip(w0: u32, h0: u32, tile_x: u32, tile_y: u32, z: Compression) {
    let pyr = build_mipmap_pyramid(w0, h0);
    let levels = mipmap_inputs_with_dims(w0, h0, &pyr);
    let bytes = encode_exr_multipart_mixed(&[MultipartMixedPart::DeepTiledMipmap {
        name: "dmip".to_string(),
        tile_x,
        tile_y,
        channels: channels_rgba_float(),
        pyramid: levels,
        compression: z,
    }])
    .unwrap_or_else(|e| panic!("encode {z:?} {w0}x{h0}: {e}"));
    let imgs =
        parse_exr_multipart_mixed(&bytes).unwrap_or_else(|e| panic!("parse {z:?} {w0}x{h0}: {e}"));
    assert_eq!(imgs.len(), 1);
    assert!(imgs[0].is_deep_tiled_mipmap());
    let part = imgs[0].deep_tiled_mipmap().unwrap();
    assert_eq!(part.name, "dmip");
    assert_eq!(part.compression, z);
    assert_eq!(part.levels.len(), pyr.len());
    for (l, (lvl, (spp, planes))) in part.levels.iter().zip(pyr.iter()).enumerate() {
        assert_eq!(lvl.width, level_dim(w0, l as u32), "level {l} width");
        assert_eq!(lvl.height, level_dim(h0, l as u32), "level {l} height");
        assert_eq!(&lvl.samples_per_pixel, spp, "level {l} spp {z:?}");
        for (c, (got, want)) in lvl.channel_samples.iter().zip(planes.iter()).enumerate() {
            assert_eq!(got, want, "level {l} channel {c} {z:?}");
        }
    }
}

#[test]
fn mixed_deep_mipmap_roundtrip_all_compressions_16x16() {
    for z in deep_compressions() {
        assert_mipmap_roundtrip(16, 16, 8, 8, z);
    }
}

#[test]
fn mixed_deep_mipmap_roundtrip_edge_tiles_24x16() {
    // Non-power-of-two width + tile sizes that don't divide levels evenly
    // exercise edge tiles at multiple levels.
    for z in deep_compressions() {
        assert_mipmap_roundtrip(24, 16, 8, 4, z);
    }
}

// ---------------------------------------------------------------------
// RIPMAP self-roundtrip.
// ---------------------------------------------------------------------

fn assert_ripmap_roundtrip(w0: u32, h0: u32, tile_x: u32, tile_y: u32, z: Compression) {
    let grid = build_ripmap_grid(w0, h0);
    let cells = ripmap_inputs_with_dims(w0, h0, &grid);
    let bytes = encode_exr_multipart_mixed(&[MultipartMixedPart::DeepTiledRipmap {
        name: "drip".to_string(),
        tile_x,
        tile_y,
        channels: channels_rgba_float(),
        grid: cells,
        compression: z,
    }])
    .unwrap_or_else(|e| panic!("encode {z:?} {w0}x{h0}: {e}"));
    let imgs =
        parse_exr_multipart_mixed(&bytes).unwrap_or_else(|e| panic!("parse {z:?} {w0}x{h0}: {e}"));
    assert_eq!(imgs.len(), 1);
    assert!(imgs[0].is_deep_tiled_ripmap());
    let part = imgs[0].deep_tiled_ripmap().unwrap();
    assert_eq!(part.name, "drip");
    assert_eq!(part.grid.len(), grid.len());
    for (ly, (got_row, want_row)) in part.grid.iter().zip(grid.iter()).enumerate() {
        assert_eq!(got_row.len(), want_row.len(), "row {ly} len");
        for (lx, (cell, (spp, planes))) in got_row.iter().zip(want_row.iter()).enumerate() {
            assert_eq!(cell.level_x, lx as u32);
            assert_eq!(cell.level_y, ly as u32);
            assert_eq!(cell.width, level_dim(w0, lx as u32), "cell ({lx},{ly}) w");
            assert_eq!(cell.height, level_dim(h0, ly as u32), "cell ({lx},{ly}) h");
            assert_eq!(&cell.samples_per_pixel, spp, "cell ({lx},{ly}) spp {z:?}");
            for (c, (g, w)) in cell.channel_samples.iter().zip(planes.iter()).enumerate() {
                assert_eq!(g, w, "cell ({lx},{ly}) channel {c} {z:?}");
            }
        }
    }
}

#[test]
fn mixed_deep_ripmap_roundtrip_all_compressions_16x16() {
    for z in deep_compressions() {
        assert_ripmap_roundtrip(16, 16, 8, 8, z);
    }
}

#[test]
fn mixed_deep_ripmap_roundtrip_edge_tiles_24x12() {
    for z in deep_compressions() {
        assert_ripmap_roundtrip(24, 12, 8, 8, z);
    }
}

// ---------------------------------------------------------------------
// Reject paths.
// ---------------------------------------------------------------------

#[test]
fn mixed_deep_mipmap_rejects_lossy_compression() {
    let pyr = build_mipmap_pyramid(16, 16);
    let levels = mipmap_inputs_with_dims(16, 16, &pyr);
    let r = encode_exr_multipart_mixed(&[MultipartMixedPart::DeepTiledMipmap {
        name: "d".to_string(),
        tile_x: 8,
        tile_y: 8,
        channels: channels_rgba_float(),
        pyramid: levels,
        compression: Compression::Pxr24,
    }]);
    assert!(r.is_err(), "PXR24 must be rejected for deep mipmap parts");
}

#[test]
fn mixed_deep_mipmap_rejects_wrong_pyramid_length() {
    let mut pyr = build_mipmap_pyramid(16, 16);
    pyr.pop(); // drop the last level -> wrong count
    let levels = mipmap_inputs_with_dims(16, 16, &pyr);
    let r = encode_exr_multipart_mixed(&[MultipartMixedPart::DeepTiledMipmap {
        name: "d".to_string(),
        tile_x: 8,
        tile_y: 8,
        channels: channels_rgba_float(),
        pyramid: levels,
        compression: Compression::Zips,
    }]);
    assert!(r.is_err(), "short pyramid must be rejected");
}

#[test]
fn mixed_deep_ripmap_rejects_subsampled_channel() {
    let grid = build_ripmap_grid(16, 16);
    let cells = ripmap_inputs_with_dims(16, 16, &grid);
    let mut chans = channels_rgba_float();
    chans[0].x_sampling = 2; // only the channel sampling is bad
    let r = encode_exr_multipart_mixed(&[MultipartMixedPart::DeepTiledRipmap {
        name: "d".to_string(),
        tile_x: 8,
        tile_y: 8,
        channels: chans,
        grid: cells,
        compression: Compression::Zips,
    }]);
    assert!(r.is_err(), "sub-sampled deep channel must be rejected");
}

// ---------------------------------------------------------------------
// Kitchen-sink: every part type in one file.
// ---------------------------------------------------------------------

/// One mixed file interleaving all eight supported part kinds — flat
/// scanline, flat ONE_LEVEL tiled, flat MIPMAP tiled, flat RIPMAP tiled,
/// deep scanline, deep ONE_LEVEL tiled, deep MIPMAP tiled and deep
/// RIPMAP tiled — in an order that alternates flat and deep so the
/// reader's per-part chunk-shape dispatch is exercised across every
/// transition.
#[test]
fn mixed_kitchen_sink_all_eight_part_types() {
    use oxideav_openexr::MipmapLevel;

    let w = 16u32;
    let h = 16u32;
    let pixels = (w * h) as usize;

    // Flat planes for the scanline + ONE_LEVEL tiled parts.
    let flat_a: Vec<f32> = (0..pixels).map(|i| (i as f32) * 0.01).collect();
    let flat_b: Vec<f32> = (0..pixels).map(|i| (i as f32) * 0.02).collect();
    let flat_g: Vec<f32> = (0..pixels).map(|i| (i as f32) * 0.03).collect();
    let flat_r: Vec<f32> = (0..pixels).map(|i| (i as f32) * 0.04).collect();
    let flat_planes: Vec<&[f32]> = vec![&flat_a, &flat_b, &flat_g, &flat_r];

    // Flat multi-level pyramids/grids (owned MipmapLevel data).
    let n_levels = mipmap_level_count(w, h);
    let flat_pyramid: Vec<MipmapLevel> = (0..n_levels)
        .map(|l| {
            let lw = level_dim(w, l);
            let lh = level_dim(h, l);
            let n = (lw * lh) as usize;
            MipmapLevel {
                width: lw,
                height: lh,
                planes: (0..4)
                    .map(|c| (0..n).map(|i| (i + c) as f32 * 0.1 + l as f32).collect())
                    .collect(),
            }
        })
        .collect();
    let flat_grid: Vec<Vec<MipmapLevel>> = (0..n_levels)
        .map(|ly| {
            (0..n_levels)
                .map(|lx| {
                    let lw = level_dim(w, lx);
                    let lh = level_dim(h, ly);
                    let n = (lw * lh) as usize;
                    MipmapLevel {
                        width: lw,
                        height: lh,
                        planes: (0..4)
                            .map(|c| {
                                (0..n)
                                    .map(|i| (i + c) as f32 * 0.2 + (ly * 8 + lx) as f32)
                                    .collect()
                            })
                            .collect(),
                    }
                })
                .collect()
        })
        .collect();

    // Deep data.
    let (ds_spp, ds_planes) = build_level(w, h, 100.0);
    let (dt_spp, dt_planes) = build_level(w, h, 200.0);
    let deep_pyr = build_mipmap_pyramid(w, h);
    let deep_grid = build_ripmap_grid(w, h);

    let parts = vec![
        MultipartMixedPart::DeepTiledMipmap {
            name: "deep_mip".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: channels_rgba_float(),
            pyramid: mipmap_inputs_with_dims(w, h, &deep_pyr),
            compression: Compression::Zips,
        },
        MultipartMixedPart::Scanline {
            name: "scan".to_string(),
            width: w,
            height: h,
            channels: channels_rgba_float(),
            planes: flat_planes.clone(),
            compression: Compression::Zip,
        },
        MultipartMixedPart::DeepTiledRipmap {
            name: "deep_rip".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: channels_rgba_float(),
            grid: ripmap_inputs_with_dims(w, h, &deep_grid),
            compression: Compression::Rle,
        },
        MultipartMixedPart::TiledMipmap {
            name: "flat_mip".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: channels_rgba_float(),
            pyramid: flat_pyramid.clone(),
            compression: Compression::Zips,
        },
        MultipartMixedPart::DeepScanline {
            name: "deep_scan".to_string(),
            width: w,
            height: h,
            channels: channels_rgba_float(),
            samples_per_pixel: &ds_spp,
            channel_samples: vec![&ds_planes[0], &ds_planes[1], &ds_planes[2], &ds_planes[3]],
            compression: Compression::Zips,
        },
        MultipartMixedPart::Tiled {
            name: "flat_tile".to_string(),
            width: w,
            height: h,
            tile_x: 8,
            tile_y: 8,
            channels: channels_rgba_float(),
            planes: flat_planes.clone(),
            compression: Compression::Rle,
        },
        MultipartMixedPart::DeepTiled {
            name: "deep_tile".to_string(),
            width: w,
            height: h,
            tile_x: 8,
            tile_y: 8,
            channels: channels_rgba_float(),
            samples_per_pixel: &dt_spp,
            channel_samples: vec![&dt_planes[0], &dt_planes[1], &dt_planes[2], &dt_planes[3]],
            compression: Compression::None,
        },
        MultipartMixedPart::TiledRipmap {
            name: "flat_rip".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: channels_rgba_float(),
            grid: flat_grid.clone(),
            compression: Compression::Zip,
        },
    ];

    let bytes = encode_exr_multipart_mixed(&parts).expect("kitchen-sink encode");
    let imgs = parse_exr_multipart_mixed(&bytes).expect("kitchen-sink parse");
    assert_eq!(imgs.len(), 8);

    // Part 0: deep MIPMAP.
    let p0 = imgs[0].deep_tiled_mipmap().expect("part 0 kind");
    assert_eq!(p0.name, "deep_mip");
    assert_eq!(p0.levels.len(), deep_pyr.len());
    for (lvl, (spp, planes)) in p0.levels.iter().zip(deep_pyr.iter()) {
        assert_eq!(&lvl.samples_per_pixel, spp);
        for (got, want) in lvl.channel_samples.iter().zip(planes.iter()) {
            assert_eq!(got, want);
        }
    }

    // Part 1: flat scanline.
    let p1 = imgs[1].image().expect("part 1 kind");
    for (plane, want) in p1.planes.iter().zip([&flat_a, &flat_b, &flat_g, &flat_r]) {
        assert_eq!(&plane.samples, want);
    }

    // Part 2: deep RIPMAP.
    let p2 = imgs[2].deep_tiled_ripmap().expect("part 2 kind");
    assert_eq!(p2.name, "deep_rip");
    for (got_row, want_row) in p2.grid.iter().zip(deep_grid.iter()) {
        for (cell, (spp, planes)) in got_row.iter().zip(want_row.iter()) {
            assert_eq!(&cell.samples_per_pixel, spp);
            for (g, w) in cell.channel_samples.iter().zip(planes.iter()) {
                assert_eq!(g, w);
            }
        }
    }

    // Part 3: flat MIPMAP.
    let p3 = imgs[3].multilevel_tiled().expect("part 3 kind");
    assert_eq!(p3.levels.len(), flat_pyramid.len());
    for (lvl, want) in p3.levels.iter().zip(flat_pyramid.iter()) {
        for (got_plane, want_plane) in lvl.planes.iter().zip(want.planes.iter()) {
            assert_eq!(&got_plane.samples, want_plane);
        }
    }

    // Part 4: deep scanline.
    let p4 = imgs[4].deep_scanline().expect("part 4 kind");
    assert_eq!(p4.samples_per_pixel, ds_spp);
    for (g, w) in p4.channel_samples.iter().zip(ds_planes.iter()) {
        assert_eq!(g, w);
    }

    // Part 5: flat ONE_LEVEL tiled.
    let p5 = imgs[5].image().expect("part 5 kind");
    for (plane, want) in p5.planes.iter().zip([&flat_a, &flat_b, &flat_g, &flat_r]) {
        assert_eq!(&plane.samples, want);
    }

    // Part 6: deep ONE_LEVEL tiled.
    let p6 = imgs[6].deep_tiled().expect("part 6 kind");
    assert_eq!(p6.samples_per_pixel, dt_spp);
    for (g, w) in p6.channel_samples.iter().zip(dt_planes.iter()) {
        assert_eq!(g, w);
    }

    // Part 7: flat RIPMAP.
    let p7 = imgs[7].multilevel_tiled().expect("part 7 kind");
    let flat_grid_linear: Vec<&MipmapLevel> = flat_grid.iter().flatten().collect();
    assert_eq!(p7.levels.len(), flat_grid_linear.len());
    for (lvl, want) in p7.levels.iter().zip(flat_grid_linear) {
        for (got_plane, want_plane) in lvl.planes.iter().zip(want.planes.iter()) {
            assert_eq!(&got_plane.samples, want_plane);
        }
    }
}

// ---------------------------------------------------------------------
// External validator acceptance (black-box).
// ---------------------------------------------------------------------

fn tool_available(name: &str) -> bool {
    std::process::Command::new(name)
        .arg("--help")
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

fn tempdir() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-mixed-ml-deep-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

/// A mixed file whose parts are a flat scanline part + a deep MIPMAP
/// tiled part + a deep RIPMAP tiled part must be accepted by an external
/// EXR header reader, and the dump must report the deep parts' types and
/// level modes.
#[test]
fn exrheader_accepts_mixed_file_with_multilevel_deep_parts() {
    if !tool_available("exrheader") {
        eprintln!("exrheader not available, skipping");
        return;
    }
    let w = 16u32;
    let h = 16u32;
    let pixels = (w * h) as usize;
    let flat: Vec<f32> = (0..pixels).map(|i| (i as f32) * 0.01).collect();
    let deep_pyr = build_mipmap_pyramid(w, h);
    let deep_grid = build_ripmap_grid(w, h);
    let bytes = encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "flat".to_string(),
            width: w,
            height: h,
            channels: channels_rgba_float(),
            planes: vec![&flat, &flat, &flat, &flat],
            compression: Compression::Zips,
        },
        MultipartMixedPart::DeepTiledMipmap {
            name: "dmip".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: channels_rgba_float(),
            pyramid: mipmap_inputs_with_dims(w, h, &deep_pyr),
            compression: Compression::Zips,
        },
        MultipartMixedPart::DeepTiledRipmap {
            name: "drip".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: channels_rgba_float(),
            grid: ripmap_inputs_with_dims(w, h, &deep_grid),
            compression: Compression::Rle,
        },
    ])
    .unwrap();
    let dir = tempdir();
    let path = format!("{dir}/mixed_ml_deep.exr");
    std::fs::write(&path, &bytes).unwrap();
    let out = std::process::Command::new("exrheader")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "exrheader rejected our mixed multi-level-deep file:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("deeptile"),
        "exrheader output didn't mention 'deeptile':\n{stdout}"
    );
    assert!(
        stdout.contains("scanlineimage"),
        "exrheader output didn't mention 'scanlineimage':\n{stdout}"
    );
    assert!(
        stdout.contains("mip-map") || stdout.contains("mipmap") || stdout.contains("MIPMAP"),
        "exrheader output didn't indicate a mip-map level mode:\n{stdout}"
    );
    assert!(
        stdout.contains("rip-map") || stdout.contains("ripmap") || stdout.contains("RIPMAP"),
        "exrheader output didn't indicate a rip-map level mode:\n{stdout}"
    );
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

/// Same wire bytes through a second, independent external inspector.
#[test]
fn exrinfo_accepts_mixed_file_with_multilevel_deep_parts() {
    if !tool_available("exrinfo") {
        eprintln!("exrinfo not available, skipping");
        return;
    }
    let w = 16u32;
    let h = 16u32;
    let deep_pyr = build_mipmap_pyramid(w, h);
    let bytes = encode_exr_multipart_mixed(&[MultipartMixedPart::DeepTiledMipmap {
        name: "dmip".to_string(),
        tile_x: 8,
        tile_y: 8,
        channels: channels_rgba_float(),
        pyramid: mipmap_inputs_with_dims(w, h, &deep_pyr),
        compression: Compression::Zips,
    }])
    .unwrap();
    let dir = tempdir();
    let path = format!("{dir}/mixed_dmip.exr");
    std::fs::write(&path, &bytes).unwrap();
    let out = std::process::Command::new("exrinfo")
        .arg("-v")
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "exrinfo rejected our mixed deep-MIPMAP file:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("deeptile"),
        "exrinfo output didn't mention 'deeptile':\n{stdout}"
    );
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}
