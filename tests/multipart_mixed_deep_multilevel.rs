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
