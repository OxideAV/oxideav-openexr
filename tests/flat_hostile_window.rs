//! Round-382 hostile-window sweep over the flat (non-deep) readers.
//!
//! Companion to `tests/deep_hostile_sizes.rs`: slide an 8-byte `0xFF`
//! window across every byte position of a writer-produced file and
//! assert the matching reader returns `Ok` or `Err` without panicking,
//! debug-overflowing, or indexing out of bounds. The window lands on
//! every offset-table entry, chunk-header field, and attribute length
//! at some position, so this covers the flat scanline, tiled
//! (ONE_LEVEL / MIPMAP / RIPMAP) and homogeneous multi-part readers
//! with the same contract already enforced on the deep and mixed
//! paths.
#![allow(clippy::type_complexity)]

use oxideav_openexr::{
    encode_exr_multipart, encode_exr_multipart_tiled, encode_exr_scanline_rgba_float,
    encode_exr_tiled_rgba_float_mipmap_box_filter, encode_exr_tiled_rgba_float_ripmap_box_filter,
    encode_exr_tiled_rgba_float_with, parse_exr, parse_exr_multipart, parse_exr_multipart_tiled,
    parse_exr_tiled_multilevel, Channel, Compression, MultipartScanlinePart, MultipartTiledPart,
    PixelType,
};

fn rgba(w: u32, h: u32) -> Vec<f32> {
    (0..(w * h * 4) as usize).map(|i| i as f32 * 0.01).collect()
}

fn channels_float() -> Vec<Channel> {
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

fn sweep<F: Fn(&[u8])>(bytes: &[u8], parse: F) {
    for pos in 0..bytes.len().saturating_sub(8) {
        let mut m = bytes.to_vec();
        m[pos..pos + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        parse(&m);
        // A zero window catches division-by-zero paths (tile sizes,
        // sampling factors) that the all-ones window can't reach.
        m[pos..pos + 8].copy_from_slice(&0u64.to_le_bytes());
        parse(&m);
    }
}

#[test]
fn scanline_hostile_window_never_panics() {
    let bytes = encode_exr_scanline_rgba_float(8, 6, &rgba(8, 6)).unwrap();
    assert!(parse_exr(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr(b);
    });
}

#[test]
fn tiled_one_level_hostile_window_never_panics() {
    let bytes =
        encode_exr_tiled_rgba_float_with(8, 8, &rgba(8, 8), Compression::Zip, 4, 4).unwrap();
    assert!(parse_exr(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr(b);
    });
}

#[test]
fn tiled_mipmap_hostile_window_never_panics() {
    let bytes =
        encode_exr_tiled_rgba_float_mipmap_box_filter(8, 8, &rgba(8, 8), Compression::Zips, 4, 4)
            .unwrap();
    assert!(parse_exr_tiled_multilevel(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr_tiled_multilevel(b);
    });
}

#[test]
fn tiled_ripmap_hostile_window_never_panics() {
    let bytes =
        encode_exr_tiled_rgba_float_ripmap_box_filter(8, 4, &rgba(8, 4), Compression::Rle, 4, 4)
            .unwrap();
    assert!(parse_exr_tiled_multilevel(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr_tiled_multilevel(b);
    });
}

#[test]
fn multipart_scanline_hostile_window_never_panics() {
    let w = 8u32;
    let h = 6u32;
    let n = (w * h) as usize;
    let p: Vec<f32> = (0..n).map(|i| i as f32 * 0.02).collect();
    let bytes = encode_exr_multipart(&[
        MultipartScanlinePart {
            name: "p0".to_string(),
            width: w,
            height: h,
            channels: channels_float(),
            planes: vec![&p, &p, &p, &p],
            compression: Compression::Zips,
        },
        MultipartScanlinePart {
            name: "p1".to_string(),
            width: w,
            height: h,
            channels: channels_float(),
            planes: vec![&p, &p, &p, &p],
            compression: Compression::Rle,
        },
    ])
    .unwrap();
    assert!(parse_exr_multipart(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr_multipart(b);
    });
}

#[test]
fn multipart_tiled_hostile_window_never_panics() {
    let w = 8u32;
    let h = 8u32;
    let n = (w * h) as usize;
    let p: Vec<f32> = (0..n).map(|i| i as f32 * 0.02).collect();
    let bytes = encode_exr_multipart_tiled(&[MultipartTiledPart {
        name: "p0".to_string(),
        width: w,
        height: h,
        tile_x: 4,
        tile_y: 4,
        channels: channels_float(),
        planes: vec![&p, &p, &p, &p],
        compression: Compression::Zip,
    }])
    .unwrap();
    assert!(parse_exr_multipart_tiled(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr_multipart_tiled(b);
    });
}

#[test]
fn multipart_tiled_mipmap_hostile_window_never_panics() {
    use oxideav_openexr::{
        encode_exr_multipart_tiled_mipmap, parse_exr_multipart_tiled_multilevel, MipmapLevel,
        MultipartMipmapTiledPart,
    };
    let w = 8u32;
    let h = 8u32;
    let mut pyramid = Vec::new();
    let mut l = 0u32;
    loop {
        let lw = (w >> l).max(1);
        let lh = (h >> l).max(1);
        let n = (lw * lh) as usize;
        pyramid.push(MipmapLevel {
            width: lw,
            height: lh,
            planes: (0..4)
                .map(|c| (0..n).map(|i| (i + c) as f32 * 0.1).collect())
                .collect(),
        });
        if lw == 1 && lh == 1 {
            break;
        }
        l += 1;
    }
    let bytes = encode_exr_multipart_tiled_mipmap(&[MultipartMipmapTiledPart {
        name: "p0".to_string(),
        tile_x: 4,
        tile_y: 4,
        channels: channels_float(),
        pyramid,
        compression: Compression::Zips,
    }])
    .unwrap();
    assert!(parse_exr_multipart_tiled_multilevel(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr_multipart_tiled_multilevel(b);
    });
}

#[test]
fn multipart_tiled_ripmap_hostile_window_never_panics() {
    use oxideav_openexr::{
        encode_exr_multipart_tiled_ripmap, parse_exr_multipart_tiled_multilevel, MipmapLevel,
        MultipartRipmapTiledPart, RipmapPyramid,
    };
    let w = 8u32;
    let h = 4u32;
    let nlv = |d: u32| {
        let mut n = 1u32;
        let mut v = d;
        while v > 1 {
            v /= 2;
            n += 1;
        }
        n
    };
    let grid: Vec<Vec<MipmapLevel>> = (0..nlv(h))
        .map(|ly| {
            (0..nlv(w))
                .map(|lx| {
                    let lw = (w >> lx).max(1);
                    let lh = (h >> ly).max(1);
                    let n = (lw * lh) as usize;
                    MipmapLevel {
                        width: lw,
                        height: lh,
                        planes: (0..4)
                            .map(|c| (0..n).map(|i| (i + c) as f32 * 0.2).collect())
                            .collect(),
                    }
                })
                .collect()
        })
        .collect();
    let bytes = encode_exr_multipart_tiled_ripmap(&[MultipartRipmapTiledPart {
        name: "p0".to_string(),
        tile_x: 4,
        tile_y: 4,
        channels: channels_float(),
        pyramid: RipmapPyramid { grid },
        compression: Compression::Rle,
    }])
    .unwrap();
    assert!(parse_exr_multipart_tiled_multilevel(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr_multipart_tiled_multilevel(b);
    });
}

#[test]
fn mixed_deep_ripmap_hostile_window_never_panics() {
    use oxideav_openexr::{
        encode_exr_multipart_mixed, parse_exr_multipart_mixed, DeepRipmapTiledLevelInput,
        MultipartMixedPart,
    };
    let w = 8u32;
    let h = 4u32;
    let nlv = |d: u32| {
        let mut n = 1u32;
        let mut v = d;
        while v > 1 {
            v /= 2;
            n += 1;
        }
        n
    };
    // Owned deep per-cell data.
    let cells: Vec<Vec<(u32, u32, Vec<u32>, Vec<f32>)>> = (0..nlv(h))
        .map(|ly| {
            (0..nlv(w))
                .map(|lx| {
                    let lw = (w >> lx).max(1);
                    let lh = (h >> ly).max(1);
                    let n = (lw * lh) as usize;
                    let spp: Vec<u32> = (0..n).map(|i| (i as u32) % 2).collect();
                    let total: usize = spp.iter().sum::<u32>() as usize;
                    let s: Vec<f32> = (0..total).map(|i| i as f32 * 0.5).collect();
                    (lw, lh, spp, s)
                })
                .collect()
        })
        .collect();
    let single_channel = vec![Channel {
        name: "Z".to_string(),
        pixel_type: PixelType::Float,
        p_linear: false,
        x_sampling: 1,
        y_sampling: 1,
    }];
    let grid: Vec<Vec<DeepRipmapTiledLevelInput>> = cells
        .iter()
        .map(|row| {
            row.iter()
                .map(|(lw, lh, spp, s)| DeepRipmapTiledLevelInput {
                    width: *lw,
                    height: *lh,
                    samples_per_pixel: spp,
                    channel_samples: vec![s],
                })
                .collect()
        })
        .collect();
    let bytes = encode_exr_multipart_mixed(&[MultipartMixedPart::DeepTiledRipmap {
        name: "drip".to_string(),
        tile_x: 4,
        tile_y: 4,
        channels: single_channel,
        grid,
        compression: Compression::None,
    }])
    .unwrap();
    assert!(parse_exr_multipart_mixed(&bytes).is_ok());
    sweep(&bytes, |b| {
        let _ = parse_exr_multipart_mixed(b);
    });
}
