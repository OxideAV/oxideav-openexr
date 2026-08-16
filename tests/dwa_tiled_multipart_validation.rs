//! DWAA / DWAB across the tiled / multilevel / multi-part surface
//! (observer-spec `openexr-piz-dwa-observer-spec.md` §3 — a tile is one
//! self-contained chunk).
//!
//! Lossy channels round-trip within the quantisation tolerance; the
//! tiled path is additionally cross-checked against the reference
//! binary in both directions (opaque process, auto-skip when absent) —
//! reference-encoded tiled DWA decodes bit-exactly to the reference's
//! own decode, and our tiled DWA chunks are accepted and decoded
//! bit-identically.

use std::process::Command;

use oxideav_openexr::{
    build_box_filter_pyramid, encode_exr_multipart, encode_exr_multipart_mixed,
    encode_exr_multipart_tiled, encode_exr_tiled, encode_exr_tiled_mipmap, parse_exr,
    parse_exr_multipart, parse_exr_multipart_mixed, parse_exr_multipart_tiled,
    parse_exr_tiled_multilevel, Channel, Compression, MultipartMixedImage, MultipartMixedPart,
    MultipartScanlinePart, MultipartTiledPart, PixelType,
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
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-dwatile-{nanos}-{pid}-{seq}"));
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

fn smooth_plane(w: u32, h: u32, scale: f32, offset: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let fx = x as f32 / w as f32;
            let fy = y as f32 / h as f32;
            out.push(offset + scale * (fx * fx + 0.5 * fy + 0.25 * fx * fy));
        }
    }
    out
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "{what}: sample {i}: got {g}, want {w}"
        );
    }
}

#[test]
fn dwa_tiled_one_level_roundtrip_and_reference() {
    // Odd size: edge tiles exercise per-tile mirroring.
    let (w, h) = (45u32, 21u32);
    let chs = vec![ch("G", PixelType::Half)];
    let g = smooth_plane(w, h, 2.0, 0.2);
    let planes: Vec<&[f32]> = vec![&g];
    let bytes = encode_exr_tiled(w, h, &chs, &planes, Compression::Dwaa, 16, 16).unwrap();
    let img = parse_exr(&bytes).unwrap();
    assert_eq!(img.compression, Compression::Dwaa);
    assert_close(&img.planes[0].samples, &g, 0.05, "tiled dwaa self");

    // Our tiled DWA decoded by the reference must equal our decode.
    if let Some(none_bytes) = reference_convert(&bytes, "none") {
        let img_ref = parse_exr(&none_bytes).unwrap();
        for (i, (a, b)) in img.planes[0]
            .samples
            .iter()
            .zip(img_ref.planes[0].samples.iter())
            .enumerate()
        {
            assert_eq!(a.to_bits(), b.to_bits(), "reference decode differs at {i}");
        }
    }
    // Reference-encoded tiled DWA decoded by us must equal the
    // reference's own decode.
    let none_ours = encode_exr_tiled(w, h, &chs, &planes, Compression::None, 16, 16).unwrap();
    if let Some(dwa_ref) = reference_convert(&none_ours, "dwaa") {
        let ours = parse_exr(&dwa_ref).unwrap();
        if let Some(none_of_dwa) = reference_convert(&dwa_ref, "none") {
            let theirs = parse_exr(&none_of_dwa).unwrap();
            for (i, (a, b)) in ours.planes[0]
                .samples
                .iter()
                .zip(theirs.planes[0].samples.iter())
                .enumerate()
            {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "our decode of reference tiled DWA differs at {i}"
                );
            }
        }
    }
}

#[test]
fn dwa_mipmap_roundtrip() {
    let (w, h) = (40u32, 24u32);
    let chs = vec![ch("Y", PixelType::Half)];
    let p = smooth_plane(w, h, 3.0, 0.1);
    let pyramid = build_box_filter_pyramid(w, h, std::slice::from_ref(&p));
    let bytes = encode_exr_tiled_mipmap(&chs, &pyramid, Compression::Dwab, 16, 16).unwrap();
    let img = parse_exr_tiled_multilevel(&bytes).unwrap();
    assert_eq!(img.levels.len(), pyramid.len());
    for (lvl, want) in img.levels.iter().zip(pyramid.iter()) {
        assert_close(
            &lvl.planes[0].samples,
            &want.planes[0],
            0.05,
            "mipmap dwab level",
        );
    }
}

#[test]
fn dwa_multipart_scanline_and_tiled_roundtrip() {
    let (w, h) = (33u32, 40u32);
    let g0 = smooth_plane(w, h, 1.5, 0.3);
    let parts = vec![MultipartScanlinePart {
        name: "p0".to_string(),
        width: w,
        height: h,
        channels: vec![ch("G", PixelType::Half)],
        planes: vec![&g0],
        compression: Compression::Dwaa,
    }];
    let bytes = encode_exr_multipart(&parts).unwrap();
    let imgs = parse_exr_multipart(&bytes).unwrap();
    assert_close(&imgs[0].planes[0].samples, &g0, 0.05, "mp scanline dwaa");

    let tiled = vec![MultipartTiledPart {
        name: "t0".to_string(),
        width: w,
        height: h,
        tile_x: 16,
        tile_y: 16,
        channels: vec![ch("G", PixelType::Half)],
        planes: vec![&g0],
        compression: Compression::Dwaa,
    }];
    let bytes = encode_exr_multipart_tiled(&tiled).unwrap();
    let imgs = parse_exr_multipart_tiled(&bytes).unwrap();
    assert_close(&imgs[0].planes[0].samples, &g0, 0.05, "mp tiled dwaa");
}

#[test]
fn dwa_multipart_mixed_roundtrip() {
    let (w, h) = (24u32, 40u32);
    let s = smooth_plane(w, h, 2.0, 0.1);
    let t = smooth_plane(w, h, 1.0, 0.6);
    let parts = vec![
        MultipartMixedPart::Scanline {
            name: "scan".to_string(),
            width: w,
            height: h,
            channels: vec![ch("Y", PixelType::Half)],
            planes: vec![&s],
            compression: Compression::Dwaa,
        },
        MultipartMixedPart::Tiled {
            name: "tile".to_string(),
            width: w,
            height: h,
            tile_x: 16,
            tile_y: 16,
            channels: vec![ch("Y", PixelType::Half)],
            planes: vec![&t],
            compression: Compression::Dwab,
        },
    ];
    let bytes = encode_exr_multipart_mixed(&parts).unwrap();
    let imgs = parse_exr_multipart_mixed(&bytes).unwrap();
    let MultipartMixedImage::Scanline(img0) = &imgs[0] else {
        panic!("part 0 should be scanline");
    };
    let MultipartMixedImage::Tiled(img1) = &imgs[1] else {
        panic!("part 1 should be tiled");
    };
    assert_close(&img0.planes[0].samples, &s, 0.05, "mixed scanline dwaa");
    assert_close(&img1.planes[0].samples, &t, 0.05, "mixed tiled dwab");
}
