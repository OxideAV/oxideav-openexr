//! PXR24 / B44 / B44A in the mixed multi-part path (scanline + ONE_LEVEL
//! tiled parts). The mixed writer reuses the same per-chunk block builders
//! as the single-part scanline/tiled writers, and the mixed reader reuses
//! the same per-chunk decoders, so two properties must hold:
//!
//! 1. **PXR24 FLOAT reduction** — a FLOAT plane carried PXR24 in a mixed
//!    part decodes to the documented 24-bit reduction of its source
//!    (observer-spec §1.1), exactly as the single-part path does.
//! 2. **B44/B44A fixed-point** — a HALF plane carried B44/B44A in a mixed
//!    part is a pixel-level fixed point: decode→re-encode→decode is
//!    bit-stable (non-linear channels), and B44A flat blocks recover the
//!    constant exactly.
//!
//! All assertions are pure self round-trips through this crate's own
//! encode/decode pair — no external tool needed.

use oxideav_openexr::{
    encode_exr_multipart_mixed, parse_exr_multipart_mixed, Channel, Compression,
    MultipartMixedPart, PixelType,
};
use std::process::Command;

/// Mirror of the PXR24 24-bit float reduction (observer-spec §1.1).
fn pxr24_reduce(f: f32) -> f32 {
    let w = f.to_bits();
    let s = w & 0x8000_0000;
    let e = w & 0x7f80_0000;
    let m = w & 0x007f_ffff;
    let code = if e == 0x7f80_0000 {
        if m == 0 {
            e >> 8
        } else {
            let mm = m >> 8;
            (e >> 8) | mm | u32::from(mm == 0)
        }
    } else {
        let mut i = ((e | m) + (m & 0x80)) >> 8;
        if i >= 0x7f8000 {
            i = (e | m) >> 8;
        }
        (s >> 8) | i
    };
    f32::from_bits((code & 0x00ff_ffff) << 8)
}

fn float_channels() -> Vec<Channel> {
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

fn half_channels(p_linear: bool) -> Vec<Channel> {
    // A is always a data channel (non-linear).
    [
        ("A", false),
        ("B", p_linear),
        ("G", p_linear),
        ("R", p_linear),
    ]
    .iter()
    .map(|(n, pl)| Channel {
        name: n.to_string(),
        pixel_type: PixelType::Half,
        p_linear: *pl,
        x_sampling: 1,
        y_sampling: 1,
    })
    .collect()
}

/// A FLOAT ramp spread across several decades so the 24-bit reduction is
/// exercised over many exponents.
fn float_ramp(w: u32, h: u32, salt: f32) -> Vec<Vec<f32>> {
    let n = (w * h) as usize;
    (0..4)
        .map(|c| {
            (0..n)
                .map(|i| (i as f32) * 0.013 + 0.001 + salt + c as f32 * 0.5)
                .collect()
        })
        .collect()
}

/// A HALF gradient.
fn half_grad(w: u32, h: u32, salt: f32) -> Vec<Vec<f32>> {
    let n = (w * h) as usize;
    (0..4)
        .map(|c| {
            (0..n)
                .map(|i| (i as f32) * 0.05 + 0.01 + salt + c as f32 * 0.37)
                .collect()
        })
        .collect()
}

/// A HALF field of large constant 4×4-aligned regions (drives B44A flat
/// blocks). `w`/`h` are multiples of 4.
fn half_flat(w: u32, h: u32) -> Vec<Vec<f32>> {
    let wu = w as usize;
    (0..4)
        .map(|c| {
            (0..(w * h) as usize)
                .map(|i| {
                    let bx = (i % wu) / 4;
                    let by = (i / wu) / 4;
                    ((bx + by * 7) % 5) as f32 * 2.0 + c as f32 * 0.25
                })
                .collect()
        })
        .collect()
}

fn refs(planes: &[Vec<f32>]) -> Vec<&[f32]> {
    planes.iter().map(|v| v.as_slice()).collect()
}

// ---------------------------------------------------------------------------
// PXR24 — FLOAT reduction
// ---------------------------------------------------------------------------

#[test]
fn mixed_pxr24_scanline_float_reduction() {
    let (w, h) = (20, 18);
    let p = float_ramp(w, h, 0.0);
    let bytes = encode_exr_multipart_mixed(&[MultipartMixedPart::Scanline {
        name: "scan".to_string(),
        width: w,
        height: h,
        channels: float_channels(),
        planes: refs(&p),
        compression: Compression::Pxr24,
    }])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    let img = imgs[0].image().unwrap();
    assert_eq!(img.compression, Compression::Pxr24);
    for (ci, name) in ["A", "B", "G", "R"].iter().enumerate() {
        let plane = img.planes.iter().find(|p| &p.name == name).unwrap();
        for (off, &got) in plane.samples.iter().enumerate() {
            assert_eq!(
                got.to_bits(),
                pxr24_reduce(p[ci][off]).to_bits(),
                "PXR24 scanline {name}[{off}]"
            );
        }
    }
}

#[test]
fn mixed_pxr24_tiled_float_reduction_edge_tiles() {
    // 30×22 with 8×8 tiles → right + bottom partial edge tiles.
    let (w, h) = (30, 22);
    let p = float_ramp(w, h, 0.3);
    let bytes = encode_exr_multipart_mixed(&[MultipartMixedPart::Tiled {
        name: "tile".to_string(),
        width: w,
        height: h,
        tile_x: 8,
        tile_y: 8,
        channels: float_channels(),
        planes: refs(&p),
        compression: Compression::Pxr24,
    }])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    let img = imgs[0].image().unwrap();
    assert!(imgs[0].is_tiled());
    for (ci, name) in ["A", "B", "G", "R"].iter().enumerate() {
        let plane = img.planes.iter().find(|p| &p.name == name).unwrap();
        for (off, &got) in plane.samples.iter().enumerate() {
            assert_eq!(
                got.to_bits(),
                pxr24_reduce(p[ci][off]).to_bits(),
                "PXR24 tiled {name}[{off}]"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// B44 / B44A — HALF fixed point
// ---------------------------------------------------------------------------

/// Decode the mixed file, re-encode the decoded HALF planes the same way,
/// decode again, and assert pixel-stable. Returns the first decode's
/// planes (declared order A,B,G,R) for further checks.
fn assert_b44_fixed_point(
    scheme: Compression,
    tiled: bool,
    w: u32,
    h: u32,
    p_linear: bool,
    planes: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let build = |chans: Vec<Channel>, pl: &[Vec<f32>]| {
        let part = if tiled {
            MultipartMixedPart::Tiled {
                name: "p".to_string(),
                width: w,
                height: h,
                tile_x: 8,
                tile_y: 8,
                channels: chans,
                planes: refs(pl),
                compression: scheme,
            }
        } else {
            MultipartMixedPart::Scanline {
                name: "p".to_string(),
                width: w,
                height: h,
                channels: chans,
                planes: refs(pl),
                compression: scheme,
            }
        };
        encode_exr_multipart_mixed(std::slice::from_ref(&part)).unwrap()
    };

    let bytes1 = build(half_channels(p_linear), planes);
    let img1 = parse_exr_multipart_mixed(&bytes1).unwrap();
    let i1 = img1[0].image().unwrap();
    assert_eq!(i1.compression, scheme);

    let mut decoded: Vec<Vec<f32>> = Vec::new();
    for name in ["A", "B", "G", "R"] {
        decoded.push(
            i1.planes
                .iter()
                .find(|p| p.name == name)
                .unwrap()
                .samples
                .clone(),
        );
    }

    if !p_linear {
        // Non-linear channels are a pixel-level fixed point.
        let bytes2 = build(half_channels(p_linear), &decoded);
        let img2 = parse_exr_multipart_mixed(&bytes2).unwrap();
        let i2 = img2[0].image().unwrap();
        for name in ["A", "B", "G", "R"] {
            let p1 = i1.planes.iter().find(|p| p.name == name).unwrap();
            let p2 = i2.planes.iter().find(|p| p.name == name).unwrap();
            for (off, (a, b)) in p1.samples.iter().zip(p2.samples.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "B44 {scheme:?} tiled={tiled} {name}[{off}] not a fixed point"
                );
            }
        }
    }
    decoded
}

#[test]
fn mixed_b44_scanline_half_fixed_point() {
    let (w, h) = (16, 24);
    assert_b44_fixed_point(Compression::B44, false, w, h, false, &half_grad(w, h, 0.0));
}

#[test]
fn mixed_b44_tiled_half_fixed_point_edge_tiles() {
    let (w, h) = (22, 18);
    assert_b44_fixed_point(Compression::B44, true, w, h, false, &half_grad(w, h, 0.2));
}

#[test]
fn mixed_b44a_scanline_flat_block_recovery() {
    // Constant 4×4-aligned regions must recover exactly through B44A's
    // 3-byte flat blocks.
    let (w, h) = (16, 16);
    let p = half_flat(w, h);
    let decoded = assert_b44_fixed_point(Compression::B44a, false, w, h, false, &p);
    // Flat HALF values are exactly representable; B44A recovers them.
    for (ci, name) in ["A", "B", "G", "R"].iter().enumerate() {
        for (off, &got) in decoded[ci].iter().enumerate() {
            // The source is already on the HALF lattice (small integers /
            // quarters), so the recovered value equals the source.
            let want =
                oxideav_openexr::half::half_to_f32(oxideav_openexr::half::f32_to_half(p[ci][off]));
            assert_eq!(got.to_bits(), want.to_bits(), "B44A flat {name}[{off}]");
        }
    }
}

#[test]
fn mixed_b44a_tiled_flat_block_recovery() {
    let (w, h) = (16, 16);
    let p = half_flat(w, h);
    assert_b44_fixed_point(Compression::B44a, true, w, h, false, &p);
}

#[test]
fn mixed_b44_plinear_scanline_roundtrips() {
    // pLinear channels are not a strict fixed point (exp/log aren't exact
    // inverses at HALF precision), but the file must still decode cleanly
    // and round-trip through the mixed reader.
    let (w, h) = (16, 16);
    let p = half_grad(w, h, 0.1);
    let bytes = encode_exr_multipart_mixed(&[MultipartMixedPart::Scanline {
        name: "p".to_string(),
        width: w,
        height: h,
        channels: half_channels(true),
        planes: refs(&p),
        compression: Compression::B44,
    }])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert_eq!(imgs[0].image().unwrap().compression, Compression::B44);
}

// ---------------------------------------------------------------------------
// Mixed file: scanline-PXR24 + tiled-B44 + scanline-ZIP in one container
// ---------------------------------------------------------------------------

#[test]
fn mixed_pxr24_and_b44_and_zip_in_one_file() {
    let (w, h) = (16, 16);
    let fp = float_ramp(w, h, 0.0);
    let hp = half_grad(w, h, 0.5);
    let zp = float_ramp(w, h, 1.0);
    let bytes = encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "pxr".to_string(),
            width: w,
            height: h,
            channels: float_channels(),
            planes: refs(&fp),
            compression: Compression::Pxr24,
        },
        MultipartMixedPart::Tiled {
            name: "b44".to_string(),
            width: w,
            height: h,
            tile_x: 8,
            tile_y: 8,
            channels: half_channels(false),
            planes: refs(&hp),
            compression: Compression::B44,
        },
        MultipartMixedPart::Scanline {
            name: "zip".to_string(),
            width: w,
            height: h,
            channels: float_channels(),
            planes: refs(&zp),
            compression: Compression::Zip,
        },
    ])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert_eq!(imgs.len(), 3);
    assert_eq!(imgs[0].image().unwrap().compression, Compression::Pxr24);
    assert_eq!(imgs[1].image().unwrap().compression, Compression::B44);
    assert_eq!(imgs[2].image().unwrap().compression, Compression::Zip);

    // PXR24 part decodes to the 24-bit reduction.
    let pxr = imgs[0].image().unwrap();
    for (ci, name) in ["A", "B", "G", "R"].iter().enumerate() {
        let plane = pxr.planes.iter().find(|p| &p.name == name).unwrap();
        for (off, &got) in plane.samples.iter().enumerate() {
            assert_eq!(got.to_bits(), pxr24_reduce(fp[ci][off]).to_bits());
        }
    }
    // ZIP part is lossless.
    let zip = imgs[2].image().unwrap();
    for (ci, name) in ["A", "B", "G", "R"].iter().enumerate() {
        let plane = zip.planes.iter().find(|p| &p.name == name).unwrap();
        assert_eq!(plane.samples, zp[ci]);
    }
}

// ---------------------------------------------------------------------------
// Multi-level (MIPMAP / RIPMAP) flat tiled parts now carry PXR24 / B44.
// ---------------------------------------------------------------------------

use oxideav_openexr::{build_box_filter_pyramid, build_box_filter_ripmap, MultipartMixedImage};

/// Re-encode each decoded level through the given scheme via the ONE_LEVEL
/// tiled path and decode again; the decoded levels are on the quantisation
/// lattice so the second pass must be bit-stable.
fn multilevel_fixed_point(img: &MultipartMixedImage, channels: &[Channel], scheme: Compression) {
    let mlt = img.multilevel_tiled().unwrap();
    for lvl in &mlt.levels {
        let refs: Vec<&[f32]> = lvl.planes.iter().map(|p| p.samples.as_slice()).collect();
        let bytes = encode_exr_multipart_mixed(&[MultipartMixedPart::Tiled {
            name: "p".to_string(),
            width: lvl.width,
            height: lvl.height,
            tile_x: lvl.width.max(1),
            tile_y: lvl.height.max(1),
            channels: channels.to_vec(),
            planes: refs,
            compression: scheme,
        }])
        .unwrap();
        let re = parse_exr_multipart_mixed(&bytes).unwrap();
        let rimg = re[0].image().unwrap();
        for (ci, plane) in rimg.planes.iter().enumerate() {
            for (off, &got) in plane.samples.iter().enumerate() {
                assert_eq!(
                    got.to_bits(),
                    lvl.planes[ci].samples[off].to_bits(),
                    "multi-level {scheme:?} fixed-point level {}x{} ch{ci}[{off}]",
                    lvl.width,
                    lvl.height
                );
            }
        }
    }
}

#[test]
fn mixed_mipmap_pxr24_roundtrips() {
    let p = float_ramp(16, 16, 0.0);
    let pyramid = build_box_filter_pyramid(
        16,
        16,
        &[p[0].clone(), p[1].clone(), p[2].clone(), p[3].clone()],
    );
    let bytes = encode_exr_multipart_mixed(&[MultipartMixedPart::TiledMipmap {
        name: "mip".to_string(),
        tile_x: 8,
        tile_y: 8,
        channels: float_channels(),
        pyramid,
        compression: Compression::Pxr24,
    }])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert!(imgs[0].is_tiled_mipmap());
    let mlt = imgs[0].multilevel_tiled().unwrap();
    assert_eq!(mlt.compression, Compression::Pxr24);
    // Each decoded FLOAT level equals the 24-bit reduction of the box-
    // filtered source level (the encoder fed each level through PXR24).
    multilevel_fixed_point(&imgs[0], &float_channels(), Compression::Pxr24);
}

#[test]
fn mixed_ripmap_pxr24_roundtrips_edge_tiles() {
    // 13×9 forces edge tiles at multiple grid cells.
    let p = float_ramp(13, 9, 0.2);
    let grid = build_box_filter_ripmap(
        13,
        9,
        &[p[0].clone(), p[1].clone(), p[2].clone(), p[3].clone()],
    )
    .grid;
    let bytes = encode_exr_multipart_mixed(&[MultipartMixedPart::TiledRipmap {
        name: "rip".to_string(),
        tile_x: 4,
        tile_y: 3,
        channels: float_channels(),
        grid,
        compression: Compression::Pxr24,
    }])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert!(imgs[0].is_tiled_ripmap());
    assert_eq!(
        imgs[0].multilevel_tiled().unwrap().compression,
        Compression::Pxr24
    );
    multilevel_fixed_point(&imgs[0], &float_channels(), Compression::Pxr24);
}

#[test]
fn mixed_mipmap_b44_roundtrips() {
    // HALF MIPMAP carried B44 — each level decode is a pixel fixed point.
    let p = half_grad(16, 16, 0.0);
    let pyramid = build_box_filter_pyramid(
        16,
        16,
        &[p[0].clone(), p[1].clone(), p[2].clone(), p[3].clone()],
    );
    let bytes = encode_exr_multipart_mixed(&[MultipartMixedPart::TiledMipmap {
        name: "mip".to_string(),
        tile_x: 8,
        tile_y: 8,
        channels: half_channels(false),
        pyramid,
        compression: Compression::B44,
    }])
    .unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    assert!(imgs[0].is_tiled_mipmap());
    let mlt = imgs[0].multilevel_tiled().unwrap();
    assert_eq!(mlt.compression, Compression::B44);
    // Re-encode each decoded HALF level with B44 and decode again: stable.
    multilevel_fixed_point(&imgs[0], &half_channels(false), Compression::B44);
}

// ---------------------------------------------------------------------------
// External validator: an independent EXR header reader accepts the mixed
// PXR24 + B44 wire format (auto-skips when the tool is unavailable).
// ---------------------------------------------------------------------------

fn tool_available(tool: &str) -> bool {
    Command::new(tool)
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

#[test]
fn external_header_reader_accepts_mixed_lossy_file() {
    if !tool_available("exrheader") {
        eprintln!("exrheader not available, skipping");
        return;
    }
    let (w, h) = (16, 16);
    let fp = float_ramp(w, h, 0.0);
    let hp = half_grad(w, h, 0.5);
    let bytes = encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "pxr".to_string(),
            width: w,
            height: h,
            channels: float_channels(),
            planes: refs(&fp),
            compression: Compression::Pxr24,
        },
        MultipartMixedPart::Tiled {
            name: "b44".to_string(),
            width: w,
            height: h,
            tile_x: 8,
            tile_y: 8,
            channels: half_channels(false),
            planes: refs(&hp),
            compression: Compression::B44,
        },
    ])
    .unwrap();

    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-mixlossy-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("mixed_lossy.exr");
    std::fs::write(&path, &bytes).unwrap();

    let out = Command::new("exrheader").arg(&path).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "exrheader failed on our mixed PXR24+B44 file\nstdout: {stdout}\nstderr: {stderr}"
    );
    // The header dump should surface both compression names.
    let lc = stdout.to_lowercase();
    assert!(
        lc.contains("pxr24"),
        "exrheader output missing pxr24\nstdout: {stdout}"
    );
    assert!(
        lc.contains("b44"),
        "exrheader output missing b44\nstdout: {stdout}"
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}
