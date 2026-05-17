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
    encode_exr_deep_scanline, parse_exr_deep_scanline, Channel, Compression, DeepScanlineInput,
    PixelType,
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
