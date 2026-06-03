//! Cross-validation for the round-73 deep scanline reader/writer.
//!
//! Self-roundtrip lives in the module's `#[cfg(test)] mod tests`; this
//! file exercises the read path against `exrheader` and `exrmetrics
//! --convert -z none`, which both ship in the homebrew openexr
//! package. The flow is:
//!
//!   our writer (deep ZIPS)
//!     -> exrmetrics --convert -z none -> on-disk deep NONE file
//!     -> our parse_exr_deep_scanline -> pixel data
//!     -> compare bit-exact against source channel samples
//!
//! That covers two important properties:
//!
//!   (a) the bytes we emit are spec-compliant (otherwise `exrmetrics`
//!       wouldn't decompress them);
//!   (b) our reader handles a deep file authored by the reference
//!       implementation, not just by ourselves.
//!
//! Both `exrheader` and `exrmetrics` auto-skip when the binary is
//! missing.

use std::process::Command;

use oxideav_openexr::{
    encode_exr_deep_scanline, encode_exr_deep_tiled, encode_exr_deep_tiled_mipmap,
    encode_exr_deep_tiled_ripmap, encode_exr_multipart_deep_scanline, parse_exr_deep_multipart,
    parse_exr_deep_scanline, parse_exr_deep_tiled, parse_exr_deep_tiled_mipmap,
    parse_exr_deep_tiled_ripmap, Channel, Compression, DeepMipmapTiledInput,
    DeepMipmapTiledLevelInput, DeepRipmapTiledInput, DeepRipmapTiledLevelInput, DeepScanlineInput,
    DeepTiledInput, MultipartDeepScanlinePart, PixelType,
};

fn tool_available(name: &str) -> bool {
    Command::new(name)
        .arg("--help")
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

fn exrheader_available() -> bool {
    // exrheader has no --help; check by spawning with no args (returns
    // usage to stderr and exits nonzero, but the binary exists).
    Command::new("exrheader").output().is_ok()
}

fn tempdir() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-deep-test-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

fn build_synthetic(w: u32, h: u32) -> (Vec<u32>, [Vec<f32>; 4]) {
    let pixels = (w * h) as usize;
    let mut spp = Vec::with_capacity(pixels);
    for i in 0..pixels {
        spp.push((i as u32) % 4);
    }
    let total: usize = spp.iter().sum::<u32>() as usize;
    let mk = |scale: f32| -> Vec<f32> { (0..total).map(|i| (i as f32) * scale).collect() };
    (spp, [mk(0.05), mk(0.1), mk(0.15), mk(0.2)])
}

/// FLOAT-typed channels: bit-exact round-trip (no quantization step,
/// unlike HALF where 0.05_f32 != 0.05 in binary16).
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

#[test]
fn exrheader_accepts_our_deep_file() {
    if !exrheader_available() {
        eprintln!("exrheader not available, skipping");
        return;
    }
    let (spp, planes) = build_synthetic(8, 4);
    let input = DeepScanlineInput {
        width: 8,
        height: 4,
        channels: channels_rgba_float(),
        samples_per_pixel: &spp,
        channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
        compression: Compression::Zips,
    };
    let bytes = encode_exr_deep_scanline(&input).unwrap();
    let dir = tempdir();
    let path = format!("{dir}/deep.exr");
    std::fs::write(&path, &bytes).unwrap();
    let out = Command::new("exrheader").arg(&path).output().unwrap();
    assert!(
        out.status.success(),
        "exrheader rejected our deep file:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // Make sure the header dump mentions deepscanline.
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("deepscanline"),
        "exrheader output didn't mention 'deepscanline':\n{stdout}"
    );
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

fn cross_roundtrip_via_exrmetrics(z: Compression) {
    if !tool_available("exrmetrics") {
        eprintln!("exrmetrics not available, skipping ({z:?})");
        return;
    }
    let (spp, planes) = build_synthetic(8, 4);
    let input = DeepScanlineInput {
        width: 8,
        height: 4,
        channels: channels_rgba_float(),
        samples_per_pixel: &spp,
        channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
        compression: z,
    };
    let bytes = encode_exr_deep_scanline(&input).unwrap();
    let dir = tempdir();
    let in_path = format!("{dir}/in.exr");
    let out_path = format!("{dir}/out.exr");
    std::fs::write(&in_path, &bytes).unwrap();
    let out = Command::new("exrmetrics")
        .arg("--convert")
        .arg("-z")
        .arg("none")
        .arg(&in_path)
        .arg("-o")
        .arg(&out_path)
        .output()
        .expect("exrmetrics spawn");
    if !out.status.success() {
        panic!(
            "exrmetrics rejected our deep {z:?} output:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let decoded_bytes = std::fs::read(&out_path).unwrap();
    let img = parse_exr_deep_scanline(&decoded_bytes).unwrap();
    assert_eq!(img.samples_per_pixel, spp);
    for (got, want) in img.channel_samples.iter().zip(planes.iter()) {
        assert_eq!(got, want, "deep cross-roundtrip channel mismatch (z={z:?})");
    }
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn exrmetrics_decodes_our_deep_none() {
    cross_roundtrip_via_exrmetrics(Compression::None);
}

#[test]
fn exrmetrics_decodes_our_deep_zips_multiline() {
    // 20 chunks (1 line per chunk for ZIPS) exercises the offset table
    // sizing past a trivial 1-or-2-chunk case.
    if !tool_available("exrmetrics") {
        eprintln!("exrmetrics not available, skipping");
        return;
    }
    let (spp, planes) = build_synthetic(12, 20);
    let input = DeepScanlineInput {
        width: 12,
        height: 20,
        channels: channels_rgba_float(),
        samples_per_pixel: &spp,
        channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
        compression: Compression::Zips,
    };
    let bytes = encode_exr_deep_scanline(&input).unwrap();
    let dir = tempdir();
    let in_path = format!("{dir}/in.exr");
    let out_path = format!("{dir}/out.exr");
    std::fs::write(&in_path, &bytes).unwrap();
    let out = Command::new("exrmetrics")
        .arg("--convert")
        .arg("-z")
        .arg("none")
        .arg(&in_path)
        .arg("-o")
        .arg(&out_path)
        .output()
        .expect("exrmetrics spawn");
    if !out.status.success() {
        panic!(
            "exrmetrics rejected our deep ZIPS multi-chunk output:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let decoded_bytes = std::fs::read(&out_path).unwrap();
    let img = parse_exr_deep_scanline(&decoded_bytes).unwrap();
    assert_eq!(img.samples_per_pixel, spp);
    for (got, want) in img.channel_samples.iter().zip(planes.iter()) {
        assert_eq!(got, want, "deep ZIPS cross-roundtrip channel mismatch");
    }
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn exrmetrics_decodes_our_deep_zips() {
    cross_roundtrip_via_exrmetrics(Compression::Zips);
}

#[test]
fn exrmetrics_decodes_our_deep_rle() {
    cross_roundtrip_via_exrmetrics(Compression::Rle);
}

// ----------------------------------------------------------------------
// Round-92 multi-part deep READ validation.
//
// Strategy: write two distinct single-part deep .exr files with our
// writer (different compressions + different pixel patterns), feed them
// to `exrmultipart -combine` to produce a real multi-part deep file
// (version-field bits 0x1800, per-part `type=deepscanline`, per-part
// `name`), then read it back with `parse_exr_deep_multipart` and
// confirm every sample round-trips bit-exactly.
// ----------------------------------------------------------------------

fn build_synthetic_seeded(w: u32, h: u32, scale_base: f32) -> (Vec<u32>, [Vec<f32>; 4]) {
    let pixels = (w * h) as usize;
    let mut spp = Vec::with_capacity(pixels);
    for i in 0..pixels {
        spp.push((i as u32) % 4);
    }
    let total: usize = spp.iter().sum::<u32>() as usize;
    let mk = |scale: f32| -> Vec<f32> { (0..total).map(|i| (i as f32) * scale).collect() };
    (
        spp,
        [
            mk(scale_base + 0.05),
            mk(scale_base + 0.10),
            mk(scale_base + 0.15),
            mk(scale_base + 0.20),
        ],
    )
}

#[test]
fn deep_multipart_two_parts_via_exrmultipart_combine() {
    if !tool_available("exrmultipart") {
        eprintln!("exrmultipart not available, skipping");
        return;
    }
    // Write part A (ZIPS) and part B (NONE) as standalone deep files.
    let (spp_a, planes_a) = build_synthetic_seeded(8, 4, 0.0);
    let (spp_b, planes_b) = build_synthetic_seeded(8, 4, 0.5);
    let ch = channels_rgba_float();
    let a_bytes = encode_exr_deep_scanline(&DeepScanlineInput {
        width: 8,
        height: 4,
        channels: ch.clone(),
        samples_per_pixel: &spp_a,
        channel_samples: vec![&planes_a[0], &planes_a[1], &planes_a[2], &planes_a[3]],
        compression: Compression::Zips,
    })
    .unwrap();
    let b_bytes = encode_exr_deep_scanline(&DeepScanlineInput {
        width: 8,
        height: 4,
        channels: ch,
        samples_per_pixel: &spp_b,
        channel_samples: vec![&planes_b[0], &planes_b[1], &planes_b[2], &planes_b[3]],
        compression: Compression::None,
    })
    .unwrap();

    let dir = tempdir();
    let a_path = format!("{dir}/partA.exr");
    let b_path = format!("{dir}/partB.exr");
    let combined_path = format!("{dir}/combined.exr");
    std::fs::write(&a_path, &a_bytes).unwrap();
    std::fs::write(&b_path, &b_bytes).unwrap();

    let out = Command::new("exrmultipart")
        .arg("-combine")
        .arg("-i")
        .arg(format!("{a_path}::partA"))
        .arg(format!("{b_path}::partB"))
        .arg("-o")
        .arg(&combined_path)
        .output()
        .expect("exrmultipart spawn");
    assert!(
        out.status.success(),
        "exrmultipart -combine failed: stdout={} stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let combined = std::fs::read(&combined_path).unwrap();
    let parts = parse_exr_deep_multipart(&combined).unwrap();
    assert_eq!(parts.len(), 2, "expected two parts after combine");

    // Order is the order we passed to -combine.
    assert_eq!(parts[0].name, "partA");
    assert_eq!(parts[0].compression, Compression::Zips);
    assert_eq!(parts[0].samples_per_pixel, spp_a);
    for (got, want) in parts[0].channel_samples.iter().zip(planes_a.iter()) {
        assert_eq!(got, want, "partA channel mismatch");
    }

    assert_eq!(parts[1].name, "partB");
    assert_eq!(parts[1].compression, Compression::None);
    assert_eq!(parts[1].samples_per_pixel, spp_b);
    for (got, want) in parts[1].channel_samples.iter().zip(planes_b.iter()) {
        assert_eq!(got, want, "partB channel mismatch");
    }

    let _ = std::fs::remove_file(&a_path);
    let _ = std::fs::remove_file(&b_path);
    let _ = std::fs::remove_file(&combined_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn deep_multipart_three_parts_mixed_compressions() {
    if !tool_available("exrmultipart") {
        eprintln!("exrmultipart not available, skipping");
        return;
    }
    // Three distinct deep parts: ZIPS / NONE / RLE.
    let (spp_a, planes_a) = build_synthetic_seeded(6, 3, 0.0);
    let (spp_b, planes_b) = build_synthetic_seeded(6, 3, 1.0);
    let (spp_c, planes_c) = build_synthetic_seeded(6, 3, 2.0);
    let ch = channels_rgba_float();
    let mk_bytes = |spp: &[u32], planes: &[Vec<f32>; 4], z: Compression| -> Vec<u8> {
        encode_exr_deep_scanline(&DeepScanlineInput {
            width: 6,
            height: 3,
            channels: ch.clone(),
            samples_per_pixel: spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: z,
        })
        .unwrap()
    };
    let a_bytes = mk_bytes(&spp_a, &planes_a, Compression::Zips);
    let b_bytes = mk_bytes(&spp_b, &planes_b, Compression::None);
    let c_bytes = mk_bytes(&spp_c, &planes_c, Compression::Rle);

    let dir = tempdir();
    let a_path = format!("{dir}/a.exr");
    let b_path = format!("{dir}/b.exr");
    let c_path = format!("{dir}/c.exr");
    let combined_path = format!("{dir}/combined.exr");
    std::fs::write(&a_path, &a_bytes).unwrap();
    std::fs::write(&b_path, &b_bytes).unwrap();
    std::fs::write(&c_path, &c_bytes).unwrap();

    let out = Command::new("exrmultipart")
        .arg("-combine")
        .arg("-i")
        .arg(format!("{a_path}::alpha"))
        .arg(format!("{b_path}::beta"))
        .arg(format!("{c_path}::gamma"))
        .arg("-o")
        .arg(&combined_path)
        .output()
        .expect("exrmultipart spawn");
    assert!(out.status.success(), "exrmultipart -combine failed");

    let combined = std::fs::read(&combined_path).unwrap();
    let parts = parse_exr_deep_multipart(&combined).unwrap();
    assert_eq!(parts.len(), 3);
    let expected = [
        ("alpha", Compression::Zips, &spp_a, &planes_a),
        ("beta", Compression::None, &spp_b, &planes_b),
        ("gamma", Compression::Rle, &spp_c, &planes_c),
    ];
    for (got, (name, comp, spp, planes)) in parts.iter().zip(expected.iter()) {
        assert_eq!(&got.name, name);
        assert_eq!(got.compression, *comp);
        assert_eq!(got.samples_per_pixel, **spp);
        for (got_ch, want_ch) in got.channel_samples.iter().zip(planes.iter()) {
            assert_eq!(got_ch, want_ch, "{name} channel mismatch");
        }
    }

    let _ = std::fs::remove_file(&a_path);
    let _ = std::fs::remove_file(&b_path);
    let _ = std::fs::remove_file(&c_path);
    let _ = std::fs::remove_file(&combined_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn deep_multipart_multi_row_zips_via_combine() {
    // Larger dimensions so each part has many ZIPS chunks.
    if !tool_available("exrmultipart") {
        eprintln!("exrmultipart not available, skipping");
        return;
    }
    let (spp_a, planes_a) = build_synthetic_seeded(12, 10, 0.0);
    let (spp_b, planes_b) = build_synthetic_seeded(12, 10, 0.25);
    let ch = channels_rgba_float();
    let mk_bytes = |spp: &[u32], planes: &[Vec<f32>; 4]| -> Vec<u8> {
        encode_exr_deep_scanline(&DeepScanlineInput {
            width: 12,
            height: 10,
            channels: ch.clone(),
            samples_per_pixel: spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::Zips,
        })
        .unwrap()
    };
    let a_bytes = mk_bytes(&spp_a, &planes_a);
    let b_bytes = mk_bytes(&spp_b, &planes_b);

    let dir = tempdir();
    let a_path = format!("{dir}/a.exr");
    let b_path = format!("{dir}/b.exr");
    let combined_path = format!("{dir}/combined.exr");
    std::fs::write(&a_path, &a_bytes).unwrap();
    std::fs::write(&b_path, &b_bytes).unwrap();
    let out = Command::new("exrmultipart")
        .arg("-combine")
        .arg("-i")
        .arg(format!("{a_path}::foo"))
        .arg(format!("{b_path}::bar"))
        .arg("-o")
        .arg(&combined_path)
        .output()
        .expect("exrmultipart spawn");
    assert!(out.status.success(), "exrmultipart -combine failed");

    let parts = parse_exr_deep_multipart(&std::fs::read(&combined_path).unwrap()).unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].name, "foo");
    assert_eq!(parts[1].name, "bar");
    assert_eq!(parts[0].samples_per_pixel, spp_a);
    assert_eq!(parts[1].samples_per_pixel, spp_b);
    for (got, want) in parts[0].channel_samples.iter().zip(planes_a.iter()) {
        assert_eq!(got, want);
    }
    for (got, want) in parts[1].channel_samples.iter().zip(planes_b.iter()) {
        assert_eq!(got, want);
    }
    let _ = std::fs::remove_file(&a_path);
    let _ = std::fs::remove_file(&b_path);
    let _ = std::fs::remove_file(&combined_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn deep_multipart_rejects_flat_multipart() {
    use oxideav_openexr::{encode_exr_multipart_rgba_float_with, Compression};
    // A flat multi-part file should not be readable as a deep multi-part.
    let bytes = encode_exr_multipart_rgba_float_with(&[(
        "x".to_string(),
        4,
        4,
        &vec![0.0f32; 4 * 4 * 4],
        Compression::None,
    )])
    .unwrap();
    let r = parse_exr_deep_multipart(&bytes);
    assert!(
        r.is_err(),
        "flat multipart must be rejected by the deep walker"
    );
}

#[test]
fn parse_exr_multipart_rejects_deep_multipart() {
    // The flat multipart walker must redirect deep multipart input to
    // parse_exr_deep_multipart.
    if !tool_available("exrmultipart") {
        eprintln!("exrmultipart not available, skipping");
        return;
    }
    use oxideav_openexr::parse_exr_multipart;
    let (spp_a, planes_a) = build_synthetic_seeded(4, 2, 0.0);
    let (spp_b, planes_b) = build_synthetic_seeded(4, 2, 0.5);
    let ch = channels_rgba_float();
    let mk = |spp: &[u32], planes: &[Vec<f32>; 4]| -> Vec<u8> {
        encode_exr_deep_scanline(&DeepScanlineInput {
            width: 4,
            height: 2,
            channels: ch.clone(),
            samples_per_pixel: spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            compression: Compression::None,
        })
        .unwrap()
    };
    let a_bytes = mk(&spp_a, &planes_a);
    let b_bytes = mk(&spp_b, &planes_b);
    let dir = tempdir();
    let a_path = format!("{dir}/a.exr");
    let b_path = format!("{dir}/b.exr");
    let combined_path = format!("{dir}/combined.exr");
    std::fs::write(&a_path, &a_bytes).unwrap();
    std::fs::write(&b_path, &b_bytes).unwrap();
    let out = Command::new("exrmultipart")
        .arg("-combine")
        .arg("-i")
        .arg(format!("{a_path}::a"))
        .arg(format!("{b_path}::b"))
        .arg("-o")
        .arg(&combined_path)
        .output()
        .expect("exrmultipart spawn");
    assert!(out.status.success());
    let combined = std::fs::read(&combined_path).unwrap();
    let r = parse_exr_multipart(&combined);
    assert!(r.is_err(), "parse_exr_multipart must reject deep parts");
    let _ = std::fs::remove_file(&a_path);
    let _ = std::fs::remove_file(&b_path);
    let _ = std::fs::remove_file(&combined_path);
    let _ = std::fs::remove_dir(&dir);
}

// ----------------------------------------------------------------------
// Round-127 multi-part deep WRITE validation.
//
// Strategy: build a multi-part deep file directly with our new
// `encode_exr_multipart_deep_scanline`, then validate:
//
//   1. `exrheader` accepts the file and reports each part's
//      `type = "deepscanline"` + `name`.
//   2. `exrmultipart -separate -i <our> -o <prefix>` splits the file into
//      one .exr per part; each per-part output is a valid single-part
//      deep scanline file readable by our own
//      `parse_exr_deep_scanline` with bit-exact pixel data.
//
// That demonstrates the bytes we emit are spec-compliant in both the
// multi-part chain layout AND in each part's deep-chunk body, since
// exrmultipart -separate would otherwise fail to extract the per-part
// chunks.
// ----------------------------------------------------------------------

#[test]
fn exrheader_accepts_our_multipart_deep_file() {
    if !exrheader_available() {
        eprintln!("exrheader not available, skipping");
        return;
    }
    let (spp_a, planes_a) = build_synthetic_seeded(8, 4, 0.0);
    let (spp_b, planes_b) = build_synthetic_seeded(8, 4, 0.5);
    let ch = channels_rgba_float();
    let parts = vec![
        MultipartDeepScanlinePart {
            name: "partA".to_string(),
            width: 8,
            height: 4,
            channels: ch.clone(),
            samples_per_pixel: &spp_a,
            channel_samples: vec![&planes_a[0], &planes_a[1], &planes_a[2], &planes_a[3]],
            compression: Compression::Zips,
        },
        MultipartDeepScanlinePart {
            name: "partB".to_string(),
            width: 8,
            height: 4,
            channels: ch,
            samples_per_pixel: &spp_b,
            channel_samples: vec![&planes_b[0], &planes_b[1], &planes_b[2], &planes_b[3]],
            compression: Compression::None,
        },
    ];
    let bytes = encode_exr_multipart_deep_scanline(&parts).unwrap();
    let dir = tempdir();
    let path = format!("{dir}/multipart_deep.exr");
    std::fs::write(&path, &bytes).unwrap();
    let out = Command::new("exrheader").arg(&path).output().unwrap();
    assert!(
        out.status.success(),
        "exrheader rejected our multipart deep file:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("deepscanline"),
        "exrheader output didn't mention 'deepscanline':\n{stdout}"
    );
    // Each part name should appear in the header dump.
    assert!(
        stdout.contains("partA"),
        "exrheader output didn't mention 'partA':\n{stdout}"
    );
    assert!(
        stdout.contains("partB"),
        "exrheader output didn't mention 'partB':\n{stdout}"
    );
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn exrmultipart_separate_splits_our_multipart_deep_two_parts() {
    if !tool_available("exrmultipart") {
        eprintln!("exrmultipart not available, skipping");
        return;
    }
    let (spp_a, planes_a) = build_synthetic_seeded(8, 4, 0.0);
    let (spp_b, planes_b) = build_synthetic_seeded(8, 4, 0.5);
    let ch = channels_rgba_float();
    let parts = vec![
        MultipartDeepScanlinePart {
            name: "partA".to_string(),
            width: 8,
            height: 4,
            channels: ch.clone(),
            samples_per_pixel: &spp_a,
            channel_samples: vec![&planes_a[0], &planes_a[1], &planes_a[2], &planes_a[3]],
            compression: Compression::Zips,
        },
        MultipartDeepScanlinePart {
            name: "partB".to_string(),
            width: 8,
            height: 4,
            channels: ch,
            samples_per_pixel: &spp_b,
            channel_samples: vec![&planes_b[0], &planes_b[1], &planes_b[2], &planes_b[3]],
            compression: Compression::None,
        },
    ];
    let bytes = encode_exr_multipart_deep_scanline(&parts).unwrap();
    let dir = tempdir();
    let in_path = format!("{dir}/in.exr");
    let out_prefix = format!("{dir}/sep");
    std::fs::write(&in_path, &bytes).unwrap();

    // exrmultipart -separate emits `<prefix>.<N>.exr` per part.
    let out = Command::new("exrmultipart")
        .arg("-separate")
        .arg("-i")
        .arg(&in_path)
        .arg("-o")
        .arg(&out_prefix)
        .output()
        .expect("exrmultipart spawn");
    assert!(
        out.status.success(),
        "exrmultipart -separate failed:\n stdout: {}\n stderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    // Read each per-part single-part deep file back through our parser.
    let p0 = format!("{out_prefix}.1.exr");
    let p1 = format!("{out_prefix}.2.exr");
    let img_a = parse_exr_deep_scanline(&std::fs::read(&p0).unwrap()).unwrap();
    let img_b = parse_exr_deep_scanline(&std::fs::read(&p1).unwrap()).unwrap();
    assert_eq!(img_a.samples_per_pixel, spp_a);
    for (g, w) in img_a.channel_samples.iter().zip(planes_a.iter()) {
        assert_eq!(g, w, "partA channel mismatch after -separate");
    }
    assert_eq!(img_b.samples_per_pixel, spp_b);
    for (g, w) in img_b.channel_samples.iter().zip(planes_b.iter()) {
        assert_eq!(g, w, "partB channel mismatch after -separate");
    }

    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&p0);
    let _ = std::fs::remove_file(&p1);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn exrmultipart_separate_splits_our_multipart_deep_three_parts() {
    if !tool_available("exrmultipart") {
        eprintln!("exrmultipart not available, skipping");
        return;
    }
    let (spp_a, planes_a) = build_synthetic_seeded(6, 3, 0.0);
    let (spp_b, planes_b) = build_synthetic_seeded(6, 3, 0.5);
    let (spp_c, planes_c) = build_synthetic_seeded(6, 3, 1.0);
    let ch = channels_rgba_float();
    let parts = vec![
        MultipartDeepScanlinePart {
            name: "alpha".to_string(),
            width: 6,
            height: 3,
            channels: ch.clone(),
            samples_per_pixel: &spp_a,
            channel_samples: vec![&planes_a[0], &planes_a[1], &planes_a[2], &planes_a[3]],
            compression: Compression::Zips,
        },
        MultipartDeepScanlinePart {
            name: "beta".to_string(),
            width: 6,
            height: 3,
            channels: ch.clone(),
            samples_per_pixel: &spp_b,
            channel_samples: vec![&planes_b[0], &planes_b[1], &planes_b[2], &planes_b[3]],
            compression: Compression::None,
        },
        MultipartDeepScanlinePart {
            name: "gamma".to_string(),
            width: 6,
            height: 3,
            channels: ch,
            samples_per_pixel: &spp_c,
            channel_samples: vec![&planes_c[0], &planes_c[1], &planes_c[2], &planes_c[3]],
            compression: Compression::Rle,
        },
    ];
    let bytes = encode_exr_multipart_deep_scanline(&parts).unwrap();
    let dir = tempdir();
    let in_path = format!("{dir}/in.exr");
    let out_prefix = format!("{dir}/sep");
    std::fs::write(&in_path, &bytes).unwrap();
    let out = Command::new("exrmultipart")
        .arg("-separate")
        .arg("-i")
        .arg(&in_path)
        .arg("-o")
        .arg(&out_prefix)
        .output()
        .expect("exrmultipart spawn");
    assert!(
        out.status.success(),
        "exrmultipart -separate failed:\n stdout: {}\n stderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let p0 = format!("{out_prefix}.1.exr");
    let p1 = format!("{out_prefix}.2.exr");
    let p2 = format!("{out_prefix}.3.exr");
    let img_a = parse_exr_deep_scanline(&std::fs::read(&p0).unwrap()).unwrap();
    let img_b = parse_exr_deep_scanline(&std::fs::read(&p1).unwrap()).unwrap();
    let img_c = parse_exr_deep_scanline(&std::fs::read(&p2).unwrap()).unwrap();
    assert_eq!(img_a.samples_per_pixel, spp_a);
    assert_eq!(img_b.samples_per_pixel, spp_b);
    assert_eq!(img_c.samples_per_pixel, spp_c);
    for (g, w) in img_a.channel_samples.iter().zip(planes_a.iter()) {
        assert_eq!(g, w, "alpha mismatch");
    }
    for (g, w) in img_b.channel_samples.iter().zip(planes_b.iter()) {
        assert_eq!(g, w, "beta mismatch");
    }
    for (g, w) in img_c.channel_samples.iter().zip(planes_c.iter()) {
        assert_eq!(g, w, "gamma mismatch");
    }
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&p0);
    let _ = std::fs::remove_file(&p1);
    let _ = std::fs::remove_file(&p2);
    let _ = std::fs::remove_dir(&dir);
}

// ----------------------------------------------------------------------
// Round-130 single-part deep TILED WRITE + READ validation.
//
// Strategy: build a single-part deep-tiled file directly with our new
// `encode_exr_deep_tiled`, then validate:
//
//   1. `exrheader` accepts the file and reports `type="deeptile"` +
//      the `tiles` tiledesc attribute.
//   2. `exrmetrics --convert -z none` decompresses our deep-tiled file
//      and re-emits it; we then re-parse the decompressed file with
//      `parse_exr_deep_tiled` and confirm every sample round-trips
//      bit-exactly.
//
// Per the empirical convention from the reference encoder, single-part deep-tiled
// files set the non_image (0x800) version-field bit ONLY; the
// `tiles[tiledesc]` attribute + `type="deeptile"` string-attribute are
// what discriminate this file format from deep scanline. `exrheader`
// rejects files that also set the single_tile (0x200) bit.
// ----------------------------------------------------------------------

#[test]
fn exrheader_accepts_our_deep_tiled_file() {
    if !exrheader_available() {
        eprintln!("exrheader not available, skipping");
        return;
    }
    let (spp, planes) = build_synthetic(16, 12);
    let input = DeepTiledInput {
        width: 16,
        height: 12,
        tile_x: 8,
        tile_y: 4,
        channels: channels_rgba_float(),
        samples_per_pixel: &spp,
        channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
        compression: Compression::Zips,
    };
    let bytes = encode_exr_deep_tiled(&input).unwrap();
    let dir = tempdir();
    let path = format!("{dir}/deep_tiled.exr");
    std::fs::write(&path, &bytes).unwrap();
    let out = Command::new("exrheader").arg(&path).output().unwrap();
    assert!(
        out.status.success(),
        "exrheader rejected our deep-tiled file:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("deeptile"),
        "exrheader output didn't mention 'deeptile':\n{stdout}"
    );
    assert!(
        stdout.contains("tiles") && stdout.contains("tile size"),
        "exrheader output didn't mention 'tiles' / 'tile size':\n{stdout}"
    );
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

fn deep_tiled_cross_roundtrip_via_exrmetrics(z: Compression, w: u32, h: u32, tx: u32, ty: u32) {
    if !tool_available("exrmetrics") {
        eprintln!("exrmetrics not available, skipping ({z:?})");
        return;
    }
    let (spp, planes) = build_synthetic(w, h);
    let input = DeepTiledInput {
        width: w,
        height: h,
        tile_x: tx,
        tile_y: ty,
        channels: channels_rgba_float(),
        samples_per_pixel: &spp,
        channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
        compression: z,
    };
    let bytes = encode_exr_deep_tiled(&input).unwrap();
    let dir = tempdir();
    let in_path = format!("{dir}/in.exr");
    let out_path = format!("{dir}/out.exr");
    std::fs::write(&in_path, &bytes).unwrap();
    let out = Command::new("exrmetrics")
        .arg("--convert")
        .arg("-z")
        .arg("none")
        .arg(&in_path)
        .arg("-o")
        .arg(&out_path)
        .output()
        .expect("exrmetrics spawn");
    assert!(
        out.status.success(),
        "exrmetrics rejected our deep-tiled {z:?} output:\n stdout: {}\n stderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let decoded_bytes = std::fs::read(&out_path).unwrap();
    let img = parse_exr_deep_tiled(&decoded_bytes).unwrap();
    assert_eq!(img.width(), w);
    assert_eq!(img.height(), h);
    assert_eq!(img.tile_x, tx);
    assert_eq!(img.tile_y, ty);
    assert_eq!(img.samples_per_pixel, spp);
    for (got, want) in img.channel_samples.iter().zip(planes.iter()) {
        assert_eq!(
            got, want,
            "deep tiled cross-roundtrip channel mismatch (z={z:?})"
        );
    }
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn exrmetrics_decodes_our_deep_tiled_none() {
    deep_tiled_cross_roundtrip_via_exrmetrics(Compression::None, 16, 12, 8, 4);
}

#[test]
fn exrmetrics_decodes_our_deep_tiled_zips() {
    deep_tiled_cross_roundtrip_via_exrmetrics(Compression::Zips, 16, 12, 8, 4);
}

#[test]
fn exrmetrics_decodes_our_deep_tiled_rle() {
    deep_tiled_cross_roundtrip_via_exrmetrics(Compression::Rle, 16, 12, 8, 4);
}

#[test]
fn exrmetrics_decodes_our_deep_tiled_edge_tiles_13x9_in_4x4() {
    // 13×9 with 4×4 tiles → right column + bottom row are partial tiles,
    // exercising the edge-clipping path through the reference tool.
    deep_tiled_cross_roundtrip_via_exrmetrics(Compression::Zips, 13, 9, 4, 4);
}

#[test]
fn our_writer_and_reader_deep_tiled_full_roundtrip() {
    // Pure-Rust round-trip: our writer → our reader, no external tools.
    // Larger image with edge tiles in both axes exercises the second
    // pass that re-emits channel samples in pixel-scan order from the
    // per-tile sample slabs.
    let (spp, planes) = build_synthetic(23, 17);
    let input = DeepTiledInput {
        width: 23,
        height: 17,
        tile_x: 6,
        tile_y: 5,
        channels: channels_rgba_float(),
        samples_per_pixel: &spp,
        channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
        compression: Compression::Zips,
    };
    let bytes = encode_exr_deep_tiled(&input).unwrap();
    let img = parse_exr_deep_tiled(&bytes).unwrap();
    assert_eq!(img.width(), 23);
    assert_eq!(img.height(), 17);
    assert_eq!(img.tile_x, 6);
    assert_eq!(img.tile_y, 5);
    assert_eq!(img.samples_per_pixel, spp);
    for (got, want) in img.channel_samples.iter().zip(planes.iter()) {
        assert_eq!(got, want);
    }
}

#[test]
fn our_writer_and_reader_multipart_deep_full_roundtrip() {
    // Pure-Rust round-trip: our writer → our reader, no external tools.
    // Larger height with ZIPS to exercise many chunks per part.
    let (spp_a, planes_a) = build_synthetic_seeded(10, 12, 0.0);
    let (spp_b, planes_b) = build_synthetic_seeded(10, 12, 0.25);
    let ch = channels_rgba_float();
    let parts = vec![
        MultipartDeepScanlinePart {
            name: "left".to_string(),
            width: 10,
            height: 12,
            channels: ch.clone(),
            samples_per_pixel: &spp_a,
            channel_samples: vec![&planes_a[0], &planes_a[1], &planes_a[2], &planes_a[3]],
            compression: Compression::Zips,
        },
        MultipartDeepScanlinePart {
            name: "right".to_string(),
            width: 10,
            height: 12,
            channels: ch,
            samples_per_pixel: &spp_b,
            channel_samples: vec![&planes_b[0], &planes_b[1], &planes_b[2], &planes_b[3]],
            compression: Compression::Rle,
        },
    ];
    let bytes = encode_exr_multipart_deep_scanline(&parts).unwrap();
    let got = parse_exr_deep_multipart(&bytes).unwrap();
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].name, "left");
    assert_eq!(got[0].compression, Compression::Zips);
    assert_eq!(got[0].samples_per_pixel, spp_a);
    for (g, w) in got[0].channel_samples.iter().zip(planes_a.iter()) {
        assert_eq!(g, w);
    }
    assert_eq!(got[1].name, "right");
    assert_eq!(got[1].compression, Compression::Rle);
    assert_eq!(got[1].samples_per_pixel, spp_b);
    for (g, w) in got[1].channel_samples.iter().zip(planes_b.iter()) {
        assert_eq!(g, w);
    }
}

// ----------------------------------------------------------------------
// Round 208: single-part deep tiled MIPMAP_LEVELS validation.
// ----------------------------------------------------------------------

fn build_deep_mipmap_synthetic(w0: u32, h0: u32) -> Vec<(Vec<u32>, [Vec<f32>; 4])> {
    let mut max_dim = w0.max(h0);
    let mut n = 1u32;
    while max_dim > 1 {
        max_dim /= 2;
        n += 1;
    }
    let mut out = Vec::with_capacity(n as usize);
    for l in 0..n {
        let lw = (w0 >> l).max(1);
        let lh = (h0 >> l).max(1);
        out.push(build_synthetic(lw, lh));
    }
    out
}

#[test]
fn exrheader_accepts_our_deep_tiled_mipmap_file() {
    if !exrheader_available() {
        eprintln!("exrheader not available, skipping");
        return;
    }
    let (w0, h0) = (16u32, 16u32);
    let pyramid = build_deep_mipmap_synthetic(w0, h0);
    let pyramid_levels: Vec<DeepMipmapTiledLevelInput> = pyramid
        .iter()
        .enumerate()
        .map(|(l, (spp, planes))| {
            let lw = (w0 >> l).max(1);
            let lh = (h0 >> l).max(1);
            DeepMipmapTiledLevelInput {
                width: lw,
                height: lh,
                samples_per_pixel: spp,
                channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            }
        })
        .collect();
    let input = DeepMipmapTiledInput {
        tile_x: 8,
        tile_y: 8,
        channels: channels_rgba_float(),
        pyramid: pyramid_levels,
        compression: Compression::Zips,
    };
    let bytes = encode_exr_deep_tiled_mipmap(&input).unwrap();
    let dir = tempdir();
    let path = format!("{dir}/deep_tiled_mipmap.exr");
    std::fs::write(&path, &bytes).unwrap();
    let out = Command::new("exrheader").arg(&path).output().unwrap();
    assert!(
        out.status.success(),
        "exrheader rejected our deep-tiled MIPMAP file:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("deeptile"),
        "exrheader output didn't mention 'deeptile':\n{stdout}"
    );
    assert!(
        stdout.contains("mip-map") || stdout.contains("mipmap") || stdout.contains("MIPMAP"),
        "exrheader output didn't indicate mip-map level mode:\n{stdout}"
    );
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn deep_tiled_mipmap_full_pure_rust_roundtrip_24x16() {
    // Larger non-power-of-two image with ZIPS compression to exercise:
    //   * the offset-table sizing across multiple levels,
    //   * edge tiles in the partial bottom-row of level 0,
    //   * the per-tile pixel-offset table for non-edge and edge tiles,
    //   * channel reassembly into pixel-major order from the per-tile
    //     channel-major sample slabs.
    let (w0, h0) = (24u32, 16u32);
    let pyramid = build_deep_mipmap_synthetic(w0, h0);
    let pyramid_levels: Vec<DeepMipmapTiledLevelInput> = pyramid
        .iter()
        .enumerate()
        .map(|(l, (spp, planes))| {
            let lw = (w0 >> l).max(1);
            let lh = (h0 >> l).max(1);
            DeepMipmapTiledLevelInput {
                width: lw,
                height: lh,
                samples_per_pixel: spp,
                channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            }
        })
        .collect();
    let input = DeepMipmapTiledInput {
        tile_x: 8,
        tile_y: 4,
        channels: channels_rgba_float(),
        pyramid: pyramid_levels,
        compression: Compression::Zips,
    };
    let bytes = encode_exr_deep_tiled_mipmap(&input).unwrap();
    let img = parse_exr_deep_tiled_mipmap(&bytes).unwrap();
    assert_eq!(img.width(), w0);
    assert_eq!(img.height(), h0);
    assert_eq!(img.tile_x, 8);
    assert_eq!(img.tile_y, 4);
    assert_eq!(img.level_count(), pyramid.len());
    for (l, (spp, planes)) in pyramid.iter().enumerate() {
        assert_eq!(&img.levels[l].samples_per_pixel, spp, "level {l} spp");
        for (ch_idx, p) in planes.iter().enumerate() {
            assert_eq!(
                &img.levels[l].channel_samples[ch_idx], p,
                "level {l} ch {ch_idx}"
            );
        }
    }
}

// ----------------------------------------------------------------------
// Round 214: single-part deep tiled RIPMAP_LEVELS validation.
// ----------------------------------------------------------------------

#[allow(clippy::type_complexity)]
fn build_deep_ripmap_synthetic(w0: u32, h0: u32) -> Vec<Vec<(Vec<u32>, [Vec<f32>; 4])>> {
    // Match the lib-test convention: each grid cell carries an
    // independent `build_synthetic(cell_w, cell_h)` payload. RIPMAP does
    // not constrain inter-cell sample relationships at the file-format
    // level; this just exercises the per-cell encode/decode plumbing.
    let count_levels = |dim: u32| -> u32 {
        let mut n = 1u32;
        let mut d = dim;
        while d > 1 {
            d /= 2;
            n += 1;
        }
        n
    };
    let level_dim = |dim: u32, l: u32| -> u32 { (dim >> l).max(1) };
    let nx = count_levels(w0);
    let ny = count_levels(h0);
    let mut grid = Vec::with_capacity(ny as usize);
    for lvly in 0..ny {
        let lh = level_dim(h0, lvly);
        let mut row = Vec::with_capacity(nx as usize);
        for lvlx in 0..nx {
            let lw = level_dim(w0, lvlx);
            row.push(build_synthetic(lw, lh));
        }
        grid.push(row);
    }
    grid
}

#[test]
fn exrheader_accepts_our_deep_tiled_ripmap_file() {
    if !exrheader_available() {
        eprintln!("exrheader not available, skipping");
        return;
    }
    let (w0, h0) = (16u32, 16u32);
    let grid = build_deep_ripmap_synthetic(w0, h0);
    let level_dim = |dim: u32, l: u32| -> u32 { (dim >> l).max(1) };
    let mut input_grid: Vec<Vec<DeepRipmapTiledLevelInput>> = Vec::with_capacity(grid.len());
    for (lvly, row) in grid.iter().enumerate() {
        let lh = level_dim(h0, lvly as u32);
        let mut out_row: Vec<DeepRipmapTiledLevelInput> = Vec::with_capacity(row.len());
        for (lvlx, (spp, planes)) in row.iter().enumerate() {
            let lw = level_dim(w0, lvlx as u32);
            out_row.push(DeepRipmapTiledLevelInput {
                width: lw,
                height: lh,
                samples_per_pixel: spp,
                channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            });
        }
        input_grid.push(out_row);
    }
    let input = DeepRipmapTiledInput {
        tile_x: 8,
        tile_y: 8,
        channels: channels_rgba_float(),
        grid: input_grid,
        compression: Compression::Zips,
    };
    let bytes = encode_exr_deep_tiled_ripmap(&input).unwrap();
    let dir = tempdir();
    let path = format!("{dir}/deep_tiled_ripmap.exr");
    std::fs::write(&path, &bytes).unwrap();
    let out = Command::new("exrheader").arg(&path).output().unwrap();
    assert!(
        out.status.success(),
        "exrheader rejected our deep-tiled RIPMAP file:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("deeptile"),
        "exrheader output didn't mention 'deeptile':\n{stdout}"
    );
    assert!(
        stdout.contains("rip-map") || stdout.contains("ripmap") || stdout.contains("RIPMAP"),
        "exrheader output didn't indicate rip-map level mode:\n{stdout}"
    );
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn deep_tiled_ripmap_full_pure_rust_roundtrip_24x16() {
    // Non-power-of-two image with ZIPS compression exercises:
    //   * the offset-table sizing across the full (nx * ny) grid,
    //   * edge tiles in partial rows/columns at several cells,
    //   * the per-tile pixel-offset table for non-edge and edge tiles,
    //   * channel reassembly into pixel-major order from the per-tile
    //     channel-major sample slabs in every grid cell.
    let (w0, h0) = (24u32, 16u32);
    let grid = build_deep_ripmap_synthetic(w0, h0);
    let level_dim = |dim: u32, l: u32| -> u32 { (dim >> l).max(1) };
    let mut input_grid: Vec<Vec<DeepRipmapTiledLevelInput>> = Vec::with_capacity(grid.len());
    for (lvly, row) in grid.iter().enumerate() {
        let lh = level_dim(h0, lvly as u32);
        let mut out_row: Vec<DeepRipmapTiledLevelInput> = Vec::with_capacity(row.len());
        for (lvlx, (spp, planes)) in row.iter().enumerate() {
            let lw = level_dim(w0, lvlx as u32);
            out_row.push(DeepRipmapTiledLevelInput {
                width: lw,
                height: lh,
                samples_per_pixel: spp,
                channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
            });
        }
        input_grid.push(out_row);
    }
    let input = DeepRipmapTiledInput {
        tile_x: 8,
        tile_y: 4,
        channels: channels_rgba_float(),
        grid: input_grid,
        compression: Compression::Zips,
    };
    let bytes = encode_exr_deep_tiled_ripmap(&input).unwrap();
    let img = parse_exr_deep_tiled_ripmap(&bytes).unwrap();
    assert_eq!(img.width(), w0);
    assert_eq!(img.height(), h0);
    assert_eq!(img.tile_x, 8);
    assert_eq!(img.tile_y, 4);
    let (nx, ny) = img.level_counts();
    assert_eq!(ny as usize, grid.len());
    assert_eq!(nx as usize, grid[0].len());
    for (lvly, row) in grid.iter().enumerate() {
        for (lvlx, (spp, planes)) in row.iter().enumerate() {
            let cell = &img.grid[lvly][lvlx];
            assert_eq!(
                &cell.samples_per_pixel, spp,
                "cell (lvlx={lvlx}, lvly={lvly}) spp"
            );
            for (ch_idx, p) in planes.iter().enumerate() {
                assert_eq!(
                    &cell.channel_samples[ch_idx], p,
                    "cell (lvlx={lvlx}, lvly={lvly}) ch {ch_idx}"
                );
            }
        }
    }
}
