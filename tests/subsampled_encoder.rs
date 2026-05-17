//! Sub-sampled channel encoder validation (round 73).
//!
//! Encodes scanline EXR files where one or more channels carries
//! `x_sampling != 1` or `y_sampling != 1`. Self-roundtrip through
//! `parse_exr` is mandatory; cross-validation through `exrmetrics
//! --convert -z none` is best-effort (auto-skips when the binary is
//! missing) and confirms the bytes we emit are spec-compliant.
//!
//! Layout we exercise: 4:2:0-style chroma (Y at 1×1, U/V at 2×2). The
//! decoder's plane allocator already returns per-channel slices sized
//! to each channel's sub-sampled dimensions; the round-73 encoder
//! mirrors that on the write side.

use std::process::Command;

use oxideav_openexr::{
    encode_exr_scanline, header::VersionField, parse_exr, Attribute, AttributeValue, Box2i,
    Channel, Compression, ExrImage, LineOrder, PixelType,
};

fn exrmetrics_available() -> bool {
    Command::new("exrmetrics")
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
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-sub-test-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

fn make_yuv420_planes(w: u32, h: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    // Y at 1×1, U and V at 2×2.
    let mut y = Vec::with_capacity((w * h) as usize);
    for yy in 0..h {
        for xx in 0..w {
            y.push((xx as f32 / w as f32) + (yy as f32 / h as f32));
        }
    }
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let mut u = Vec::with_capacity((cw * ch) as usize);
    let mut v = Vec::with_capacity((cw * ch) as usize);
    for cy in 0..ch {
        for cx in 0..cw {
            u.push((cx as f32) * 0.125 + (cy as f32) * 0.0625);
            v.push(1.0 - ((cx as f32) * 0.125));
        }
    }
    (y, u, v)
}

fn yuv420_attrs(w: u32, h: u32, compression: Compression) -> Vec<Attribute> {
    // Channels alphabetical: U, V, Y.
    let chs = vec![
        Channel {
            name: "U".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 2,
            y_sampling: 2,
        },
        Channel {
            name: "V".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 2,
            y_sampling: 2,
        },
        Channel {
            name: "Y".to_string(),
            pixel_type: PixelType::Float,
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
    vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(chs),
        },
        Attribute {
            name: "compression".to_string(),
            value: AttributeValue::Compression(compression),
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
    ]
}

fn encode_yuv420(
    w: u32,
    h: u32,
    compression: Compression,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let (y, u, v) = make_yuv420_planes(w, h);
    let chs = match &yuv420_attrs(w, h, compression)[0].value {
        AttributeValue::Channels(c) => c.clone(),
        _ => unreachable!(),
    };
    let planes: Vec<&[f32]> = vec![u.as_slice(), v.as_slice(), y.as_slice()];
    let bytes = encode_exr_scanline(
        w,
        h,
        &chs,
        &planes,
        compression,
        yuv420_attrs(w, h, compression),
    )
    .unwrap();
    let _ = VersionField::from_u32(2); // touch the type so it's used somewhere
    (bytes, y, u, v)
}

fn assert_yuv420_matches(
    img: &ExrImage,
    w: u32,
    h: u32,
    y_src: &[f32],
    u_src: &[f32],
    v_src: &[f32],
) {
    // Channels alphabetical: U, V, Y -> planes[0]=U, [1]=V, [2]=Y.
    let cw = w.div_ceil(2);
    let ch = h.div_ceil(2);
    let u_got = &img.planes[0].samples;
    let v_got = &img.planes[1].samples;
    let y_got = &img.planes[2].samples;
    assert_eq!(
        u_got.len(),
        (cw * ch) as usize,
        "U plane size {} expected {}",
        u_got.len(),
        cw * ch
    );
    assert_eq!(v_got.len(), (cw * ch) as usize);
    assert_eq!(y_got.len(), (w * h) as usize);
    for i in 0..(w * h) as usize {
        assert_eq!(y_got[i], y_src[i], "Y mismatch at {i}");
    }
    for i in 0..(cw * ch) as usize {
        assert_eq!(u_got[i], u_src[i], "U mismatch at {i}");
        assert_eq!(v_got[i], v_src[i], "V mismatch at {i}");
    }
}

#[test]
fn yuv420_none_self_roundtrip() {
    let (bytes, y, u, v) = encode_yuv420(16, 12, Compression::None);
    let img = parse_exr(&bytes).unwrap();
    assert_yuv420_matches(&img, 16, 12, &y, &u, &v);
}

#[test]
fn yuv420_zip_self_roundtrip() {
    // Use a 20-row image so ZIP has a partial trailing block (16 + 4).
    let (bytes, y, u, v) = encode_yuv420(8, 20, Compression::Zip);
    let img = parse_exr(&bytes).unwrap();
    assert_yuv420_matches(&img, 8, 20, &y, &u, &v);
}

#[test]
fn yuv420_zips_self_roundtrip() {
    let (bytes, y, u, v) = encode_yuv420(12, 10, Compression::Zips);
    let img = parse_exr(&bytes).unwrap();
    assert_yuv420_matches(&img, 12, 10, &y, &u, &v);
}

#[test]
fn yuv420_rle_self_roundtrip() {
    let (bytes, y, u, v) = encode_yuv420(10, 6, Compression::Rle);
    let img = parse_exr(&bytes).unwrap();
    assert_yuv420_matches(&img, 10, 6, &y, &u, &v);
}

#[test]
fn yuv420_zip_exrmetrics_roundtrip() {
    if !exrmetrics_available() {
        eprintln!("exrmetrics not available, skipping");
        return;
    }
    let w = 16u32;
    let h = 12u32;
    let (bytes, y_src, u_src, v_src) = encode_yuv420(w, h, Compression::Zip);
    let dir = tempdir();
    let in_path = format!("{dir}/in.exr");
    let out_path = format!("{dir}/out.exr");
    std::fs::write(&in_path, &bytes).unwrap();
    let output = Command::new("exrmetrics")
        .arg("--convert")
        .arg("-z")
        .arg("none")
        .arg(&in_path)
        .arg("-o")
        .arg(&out_path)
        .output()
        .expect("exrmetrics spawn");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("exrmetrics rejected our sub-sampled ZIP output:\n{stderr}");
    }
    let decoded = std::fs::read(&out_path).unwrap();
    let img = parse_exr(&decoded).unwrap();
    assert_yuv420_matches(&img, w, h, &y_src, &u_src, &v_src);
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn yuv420_none_exrmetrics_roundtrip() {
    if !exrmetrics_available() {
        eprintln!("exrmetrics not available, skipping");
        return;
    }
    let w = 12u32;
    let h = 8u32;
    let (bytes, y_src, u_src, v_src) = encode_yuv420(w, h, Compression::None);
    let dir = tempdir();
    let in_path = format!("{dir}/in.exr");
    let out_path = format!("{dir}/out.exr");
    std::fs::write(&in_path, &bytes).unwrap();
    let output = Command::new("exrmetrics")
        .arg("--convert")
        .arg("-z")
        .arg("none")
        .arg(&in_path)
        .arg("-o")
        .arg(&out_path)
        .output()
        .expect("exrmetrics spawn");
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("exrmetrics rejected our sub-sampled NONE output:\n{stderr}");
    }
    let decoded = std::fs::read(&out_path).unwrap();
    let img = parse_exr(&decoded).unwrap();
    assert_yuv420_matches(&img, w, h, &y_src, &u_src, &v_src);
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);
    let _ = std::fs::remove_dir(&dir);
}
