//! Round-382 hostile-size sweep over every dedicated deep reader.
//!
//! Each deep chunk header carries three `u64` byte counts read straight
//! off the wire (`packed_table`, `packed_data`, `unpacked_data`). Before
//! this round, seven of the eight deep chunk walks summed
//! `table_start + packed_table + packed_data` with unchecked `usize`
//! addition — a debug-build add-overflow panic on a hostile value near
//! `u64::MAX`. All eight now route through one overflow-safe bounds
//! helper.
//!
//! Rather than pinpointing each size field's offset (brittle across
//! writers), each test slides an 8-byte `0xFF` window across the entire
//! encoded file: every corrupted variant must parse to `Ok` or `Err`,
//! never panic. The window is guaranteed to land exactly on every u64
//! size field at some position.
#![allow(clippy::type_complexity)]

use oxideav_openexr::deep::{
    encode_exr_deep_scanline, encode_exr_deep_tiled, encode_exr_deep_tiled_mipmap,
    encode_exr_deep_tiled_ripmap, encode_exr_multipart_deep_scanline,
    encode_exr_multipart_deep_tiled, encode_exr_multipart_deep_tiled_mipmap,
    encode_exr_multipart_deep_tiled_ripmap, parse_exr_deep_multipart, parse_exr_deep_scanline,
    parse_exr_deep_tiled, parse_exr_deep_tiled_mipmap, parse_exr_deep_tiled_ripmap,
    parse_exr_multipart_deep_tiled, parse_exr_multipart_deep_tiled_mipmap,
    parse_exr_multipart_deep_tiled_ripmap, DeepMipmapTiledInput, DeepMipmapTiledLevelInput,
    DeepRipmapTiledInput, DeepRipmapTiledLevelInput, DeepScanlineInput, DeepTiledInput,
    MultipartDeepMipmapTiledPart, MultipartDeepRipmapTiledPart, MultipartDeepScanlinePart,
    MultipartDeepTiledPart,
};
use oxideav_openexr::types::{Channel, Compression, PixelType};

fn channels_half() -> Vec<Channel> {
    ["A", "Z"]
        .iter()
        .map(|n| Channel {
            name: n.to_string(),
            pixel_type: PixelType::Half,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        })
        .collect()
}

fn build_level(w: u32, h: u32) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let pixels = (w * h) as usize;
    let spp: Vec<u32> = (0..pixels).map(|i| (i as u32) % 3).collect();
    let total: usize = spp.iter().sum::<u32>() as usize;
    let a: Vec<f32> = (0..total).map(|i| i as f32 * 0.5).collect();
    let z: Vec<f32> = (0..total).map(|i| i as f32 * 0.25).collect();
    (spp, a, z)
}

/// Slide an 8-byte 0xFF window across the file; every variant must
/// return without panicking. `parse` is the reader under test.
fn sweep<F: Fn(&[u8])>(bytes: &[u8], parse: F) {
    for pos in 0..bytes.len().saturating_sub(8) {
        let mut m = bytes.to_vec();
        m[pos..pos + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        parse(&m);
    }
}

#[test]
fn deep_scanline_hostile_sizes_never_panic() {
    let (spp, a, z) = build_level(6, 4);
    let bytes = encode_exr_deep_scanline(&DeepScanlineInput {
        width: 6,
        height: 4,
        channels: channels_half(),
        samples_per_pixel: &spp,
        channel_samples: vec![&a, &z],
        compression: Compression::None,
    })
    .unwrap();
    assert!(parse_exr_deep_scanline(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr_deep_scanline(b);
    });
}

#[test]
fn deep_tiled_hostile_sizes_never_panic() {
    let (spp, a, z) = build_level(8, 8);
    let bytes = encode_exr_deep_tiled(&DeepTiledInput {
        width: 8,
        height: 8,
        tile_x: 4,
        tile_y: 4,
        channels: channels_half(),
        samples_per_pixel: &spp,
        channel_samples: vec![&a, &z],
        compression: Compression::None,
    })
    .unwrap();
    assert!(parse_exr_deep_tiled(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr_deep_tiled(b);
    });
}

fn mipmap_levels(w0: u32, h0: u32) -> Vec<(u32, u32, Vec<u32>, Vec<f32>, Vec<f32>)> {
    let mut out = Vec::new();
    let mut l = 0u32;
    loop {
        let lw = (w0 >> l).max(1);
        let lh = (h0 >> l).max(1);
        let (spp, a, z) = build_level(lw, lh);
        out.push((lw, lh, spp, a, z));
        if lw == 1 && lh == 1 {
            break;
        }
        l += 1;
    }
    out
}

#[test]
fn deep_tiled_mipmap_hostile_sizes_never_panic() {
    let levels = mipmap_levels(8, 8);
    let pyramid: Vec<DeepMipmapTiledLevelInput> = levels
        .iter()
        .map(|(lw, lh, spp, a, z)| DeepMipmapTiledLevelInput {
            width: *lw,
            height: *lh,
            samples_per_pixel: spp,
            channel_samples: vec![a, z],
        })
        .collect();
    let bytes = encode_exr_deep_tiled_mipmap(&DeepMipmapTiledInput {
        tile_x: 4,
        tile_y: 4,
        channels: channels_half(),
        pyramid,
        compression: Compression::None,
    })
    .unwrap();
    assert!(parse_exr_deep_tiled_mipmap(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr_deep_tiled_mipmap(b);
    });
}

fn ripmap_grid(w0: u32, h0: u32) -> Vec<Vec<(u32, u32, Vec<u32>, Vec<f32>, Vec<f32>)>> {
    let nlv = |d: u32| {
        let mut n = 1u32;
        let mut v = d;
        while v > 1 {
            v /= 2;
            n += 1;
        }
        n
    };
    (0..nlv(h0))
        .map(|ly| {
            (0..nlv(w0))
                .map(|lx| {
                    let lw = (w0 >> lx).max(1);
                    let lh = (h0 >> ly).max(1);
                    let (spp, a, z) = build_level(lw, lh);
                    (lw, lh, spp, a, z)
                })
                .collect()
        })
        .collect()
}

#[test]
fn deep_tiled_ripmap_hostile_sizes_never_panic() {
    let grid_data = ripmap_grid(8, 4);
    let grid: Vec<Vec<DeepRipmapTiledLevelInput>> = grid_data
        .iter()
        .map(|row| {
            row.iter()
                .map(|(lw, lh, spp, a, z)| DeepRipmapTiledLevelInput {
                    width: *lw,
                    height: *lh,
                    samples_per_pixel: spp,
                    channel_samples: vec![a, z],
                })
                .collect()
        })
        .collect();
    let bytes = encode_exr_deep_tiled_ripmap(&DeepRipmapTiledInput {
        tile_x: 4,
        tile_y: 4,
        channels: channels_half(),
        grid,
        compression: Compression::None,
    })
    .unwrap();
    assert!(parse_exr_deep_tiled_ripmap(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr_deep_tiled_ripmap(b);
    });
}

#[test]
fn multipart_deep_scanline_hostile_sizes_never_panic() {
    let (spp, a, z) = build_level(6, 4);
    let bytes = encode_exr_multipart_deep_scanline(&[MultipartDeepScanlinePart {
        name: "p0".to_string(),
        width: 6,
        height: 4,
        channels: channels_half(),
        samples_per_pixel: &spp,
        channel_samples: vec![&a, &z],
        compression: Compression::None,
    }])
    .unwrap();
    assert!(parse_exr_deep_multipart(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr_deep_multipart(b);
    });
}

#[test]
fn multipart_deep_tiled_hostile_sizes_never_panic() {
    let (spp, a, z) = build_level(8, 8);
    let bytes = encode_exr_multipart_deep_tiled(&[MultipartDeepTiledPart {
        name: "p0".to_string(),
        width: 8,
        height: 8,
        tile_x: 4,
        tile_y: 4,
        channels: channels_half(),
        samples_per_pixel: &spp,
        channel_samples: vec![&a, &z],
        compression: Compression::None,
    }])
    .unwrap();
    assert!(parse_exr_multipart_deep_tiled(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr_multipart_deep_tiled(b);
    });
}

#[test]
fn multipart_deep_tiled_mipmap_hostile_sizes_never_panic() {
    let levels = mipmap_levels(8, 8);
    let pyramid: Vec<DeepMipmapTiledLevelInput> = levels
        .iter()
        .map(|(lw, lh, spp, a, z)| DeepMipmapTiledLevelInput {
            width: *lw,
            height: *lh,
            samples_per_pixel: spp,
            channel_samples: vec![a, z],
        })
        .collect();
    let bytes = encode_exr_multipart_deep_tiled_mipmap(&[MultipartDeepMipmapTiledPart {
        name: "p0".to_string(),
        tile_x: 4,
        tile_y: 4,
        channels: channels_half(),
        pyramid,
        compression: Compression::None,
    }])
    .unwrap();
    assert!(parse_exr_multipart_deep_tiled_mipmap(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr_multipart_deep_tiled_mipmap(b);
    });
}

#[test]
fn multipart_deep_tiled_ripmap_hostile_sizes_never_panic() {
    let grid_data = ripmap_grid(8, 4);
    let grid: Vec<Vec<DeepRipmapTiledLevelInput>> = grid_data
        .iter()
        .map(|row| {
            row.iter()
                .map(|(lw, lh, spp, a, z)| DeepRipmapTiledLevelInput {
                    width: *lw,
                    height: *lh,
                    samples_per_pixel: spp,
                    channel_samples: vec![a, z],
                })
                .collect()
        })
        .collect();
    let bytes = encode_exr_multipart_deep_tiled_ripmap(&[MultipartDeepRipmapTiledPart {
        name: "p0".to_string(),
        tile_x: 4,
        tile_y: 4,
        channels: channels_half(),
        grid,
        compression: Compression::None,
    }])
    .unwrap();
    assert!(parse_exr_multipart_deep_tiled_ripmap(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr_multipart_deep_tiled_ripmap(b);
    });
}
