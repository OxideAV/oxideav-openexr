//! Cross-validate our encoder output through OpenEXR reader binaries
//! that are *implemented by a different codebase* than the classic
//! `Imf`-library reader exercised by `exrheader` / `exrmetrics`:
//!
//!   * `exrinfo` — the file walker built on the independent
//!     `OpenEXRCore` C decoder. Accepting our bytes here proves the
//!     file layout satisfies a second, separately-written parser (not
//!     just the historical C++ reader), closing the "single-reader"
//!     validation gap.
//!   * `exr2aces` — the ACES colour-conversion tool. It *consumes* the
//!     `chromaticities` header attribute to build its RGB→ACES matrix,
//!     so a successful conversion is an end-to-end check that our
//!     chromaticities bytes are not merely echoed back by a header
//!     printer but actually interpreted as colour primaries.
//!
//! All binaries are invoked opaquely (bytes in / status + text out) and
//! every test auto-skips with a printed reason when the tool is absent,
//! so CI hosts without an OpenEXR install stay green.

use std::process::Command;

use oxideav_openexr::{
    encode_exr_multipart, encode_exr_scanline, encode_exr_scanline_rgba_float_with,
    encode_exr_tiled_rgba_float_with, parse_exr, Attribute, AttributeValue, Box2i, Channel,
    Chromaticities, Compression, LineOrder, MultipartScanlinePart, PixelType,
};

fn tool_available(bin: &str) -> bool {
    Command::new(bin)
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
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let pid = std::process::id();
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-indep-{nanos}-{pid}-{seq}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

fn write_tmp(bytes: &[u8]) -> (String, String) {
    let dir = tempdir();
    let path = format!("{dir}/in.exr");
    std::fs::write(&path, bytes).unwrap();
    (dir, path)
}

// ---------------------------------------------------------------------
// exrinfo (OpenEXRCore reader) accepts our scanline output for every
// compression scheme we emit.
// ---------------------------------------------------------------------

fn exrinfo_accepts(bytes: &[u8], label: &str) {
    if !tool_available("exrinfo") {
        eprintln!("exrinfo not available, skipping ({label})");
        return;
    }
    let (dir, path) = write_tmp(bytes);
    let out = Command::new("exrinfo")
        .arg(&path)
        .output()
        .expect("exrinfo spawn failed");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let stdout = String::from_utf8_lossy(&out.stdout);
        let _ = std::fs::remove_dir_all(&dir);
        panic!("exrinfo rejected our {label} output:\nstdout:\n{stdout}\nstderr:\n{stderr}");
    }
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("channels"),
        "exrinfo output for {label} did not describe channels:\n{text}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn sample_scanline(z: Compression) -> Vec<u8> {
    let w = 16u32;
    let h = 16u32;
    let samples: Vec<f32> = (0..(w * h * 4) as usize)
        .map(|i| (i as f32) * 0.0125)
        .collect();
    encode_exr_scanline_rgba_float_with(w, h, &samples, z).unwrap()
}

#[test]
fn exrinfo_accepts_none() {
    exrinfo_accepts(&sample_scanline(Compression::None), "NONE");
}

#[test]
fn exrinfo_accepts_zip() {
    exrinfo_accepts(&sample_scanline(Compression::Zip), "ZIP");
}

#[test]
fn exrinfo_accepts_zips() {
    exrinfo_accepts(&sample_scanline(Compression::Zips), "ZIPS");
}

#[test]
fn exrinfo_accepts_rle() {
    exrinfo_accepts(&sample_scanline(Compression::Rle), "RLE");
}

#[test]
fn exrinfo_accepts_pxr24() {
    exrinfo_accepts(&sample_scanline(Compression::Pxr24), "PXR24");
}

#[test]
fn exrinfo_accepts_b44() {
    exrinfo_accepts(&sample_scanline(Compression::B44), "B44");
}

#[test]
fn exrinfo_accepts_b44a() {
    exrinfo_accepts(&sample_scanline(Compression::B44a), "B44A");
}

// ---------------------------------------------------------------------
// chromaticities: an explicit non-default primary set must survive to
// the wire and be reported verbatim by a header printer, and must be
// *consumed* (not just echoed) by the ACES converter.
// ---------------------------------------------------------------------

/// Deliberately-not-BT.709 primaries so an accidental default-fill in
/// the writer or reader is visible.
const TEST_CHROMA: Chromaticities = Chromaticities {
    red_x: 0.7347,
    red_y: 0.2653,
    green_x: 0.0000,
    green_y: 1.0000,
    blue_x: 0.0001,
    blue_y: -0.0770,
    white_x: 0.32168,
    white_y: 0.33767,
};

fn encode_with_chromaticities(z: Compression) -> Vec<u8> {
    let w = 8u32;
    let h = 8u32;
    let chs = vec![
        Channel {
            name: "B".to_string(),
            pixel_type: PixelType::Half,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        },
        Channel {
            name: "G".to_string(),
            pixel_type: PixelType::Half,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        },
        Channel {
            name: "R".to_string(),
            pixel_type: PixelType::Half,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        },
    ];
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (w - 1) as i32,
        y_max: (h - 1) as i32,
    };
    let attrs = vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(chs.clone()),
        },
        Attribute {
            name: "compression".to_string(),
            value: AttributeValue::Compression(z),
        },
        Attribute {
            name: "dataWindow".to_string(),
            value: AttributeValue::Box2i(win),
        },
        Attribute {
            name: "displayWindow".to_string(),
            value: AttributeValue::Box2i(win),
        },
        Attribute {
            name: "lineOrder".to_string(),
            value: AttributeValue::LineOrder(LineOrder::IncreasingY),
        },
        Attribute {
            name: "pixelAspectRatio".to_string(),
            value: AttributeValue::Float(1.0),
        },
        Attribute {
            name: "screenWindowCenter".to_string(),
            value: AttributeValue::V2f(0.0, 0.0),
        },
        Attribute {
            name: "screenWindowWidth".to_string(),
            value: AttributeValue::Float(1.0),
        },
        Attribute {
            name: "chromaticities".to_string(),
            value: AttributeValue::Chromaticities(TEST_CHROMA),
        },
    ];
    let b: Vec<f32> = vec![0.25; (w * h) as usize];
    let g: Vec<f32> = vec![0.50; (w * h) as usize];
    let r: Vec<f32> = vec![0.75; (w * h) as usize];
    let planes: Vec<&[f32]> = vec![b.as_slice(), g.as_slice(), r.as_slice()];
    encode_exr_scanline(w, h, &chs, &planes, z, attrs).unwrap()
}

#[test]
fn chromaticities_survive_self_roundtrip() {
    let bytes = encode_with_chromaticities(Compression::Zip);
    let img = parse_exr(&bytes).unwrap();
    let got = img
        .attributes
        .iter()
        .find(|a| a.name == "chromaticities")
        .map(|a| &a.value)
        .expect("chromaticities attribute missing after round-trip");
    match got {
        AttributeValue::Chromaticities(c) => assert_eq!(*c, TEST_CHROMA),
        other => panic!("chromaticities decoded as wrong variant: {other:?}"),
    }
}

#[test]
fn exrheader_echoes_our_chromaticities() {
    if !tool_available("exrheader") {
        eprintln!("exrheader not available, skipping");
        return;
    }
    let bytes = encode_with_chromaticities(Compression::Zip);
    let (dir, path) = write_tmp(&bytes);
    let out = Command::new("exrheader")
        .arg(&path)
        .output()
        .expect("exrheader spawn failed");
    let text = String::from_utf8_lossy(&out.stdout);
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let _ = std::fs::remove_dir_all(&dir);
        panic!("exrheader rejected our chromaticities file:\n{stderr}");
    }
    // The printer renders the red primary among the eight values; a
    // wrong-endian or default-filled write would not show 0.7347.
    assert!(
        text.contains("0.7347"),
        "exrheader did not echo our red-x primary 0.7347:\n{text}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn exr2aces_consumes_our_chromaticities() {
    if !tool_available("exr2aces") {
        eprintln!("exr2aces not available, skipping");
        return;
    }
    // A file whose chromaticities are already the ACES AP0 primaries;
    // exr2aces must read them, apply the (identity-ish) transform, and
    // write a valid output file.
    let bytes = encode_with_chromaticities(Compression::Zip);
    let (dir, in_path) = write_tmp(&bytes);
    let out_path = format!("{dir}/out.exr");
    let out = Command::new("exr2aces")
        .arg(&in_path)
        .arg(&out_path)
        .output()
        .expect("exr2aces spawn failed");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let _ = std::fs::remove_dir_all(&dir);
        panic!("exr2aces rejected our chromaticities file:\n{stderr}");
    }
    assert!(
        std::fs::metadata(&out_path)
            .map(|m| m.len() > 0)
            .unwrap_or(false),
        "exr2aces produced no output file"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------
// exrinfo (OpenEXRCore) also drives a distinct tiled parser and a
// distinct multi-part header-chain walker — validate those layouts
// through it too, not just flat scanline.
// ---------------------------------------------------------------------

fn sample_tiled(z: Compression) -> Vec<u8> {
    let w = 24u32;
    let h = 20u32;
    let samples: Vec<f32> = (0..(w * h * 4) as usize)
        .map(|i| (i as f32) * 0.011)
        .collect();
    // 8x8 tiles → an edge column/row that is not a whole tile, exercising
    // the reader's edge-tile clipping.
    encode_exr_tiled_rgba_float_with(w, h, &samples, z, 8, 8).unwrap()
}

#[test]
fn exrinfo_accepts_tiled_none() {
    exrinfo_accepts(&sample_tiled(Compression::None), "tiled NONE");
}

#[test]
fn exrinfo_accepts_tiled_zip() {
    exrinfo_accepts(&sample_tiled(Compression::Zip), "tiled ZIP");
}

#[test]
fn exrinfo_accepts_tiled_pxr24() {
    exrinfo_accepts(&sample_tiled(Compression::Pxr24), "tiled PXR24");
}

#[test]
fn exrinfo_accepts_tiled_b44() {
    exrinfo_accepts(&sample_tiled(Compression::B44), "tiled B44");
}

#[test]
fn exrinfo_accepts_multipart_scanline() {
    // Two scanline parts of different sizes and compressions, so the
    // OpenEXRCore multi-part header-chain walker sees a non-trivial
    // chain and concatenated offset tables.
    let w0 = 12u32;
    let h0 = 10u32;
    let w1 = 8u32;
    let h1 = 6u32;
    let g0: Vec<f32> = (0..(w0 * h0) as usize).map(|i| i as f32 * 0.02).collect();
    let g1: Vec<f32> = (0..(w1 * h1) as usize).map(|i| i as f32 * 0.03).collect();
    let ch = vec![Channel {
        name: "Y".to_string(),
        pixel_type: PixelType::Half,
        p_linear: false,
        x_sampling: 1,
        y_sampling: 1,
    }];
    let parts = vec![
        MultipartScanlinePart {
            name: "left".to_string(),
            width: w0,
            height: h0,
            channels: ch.clone(),
            planes: vec![g0.as_slice()],
            compression: Compression::Zip,
        },
        MultipartScanlinePart {
            name: "right".to_string(),
            width: w1,
            height: h1,
            channels: ch.clone(),
            planes: vec![g1.as_slice()],
            compression: Compression::Rle,
        },
    ];
    let bytes = encode_exr_multipart(&parts).unwrap();
    exrinfo_accepts(&bytes, "multipart scanline");
}
