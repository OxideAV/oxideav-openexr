//! FLOAT → HALF rounding at the subnormal boundary, cross-checked
//! against a reference EXR tool as an opaque process: a FLOAT scanline
//! file holding magnitudes around 2^-25 .. 2^-24 is converted to HALF
//! by `exrmetrics --convert --pixelmode half`, and the HALF codes it
//! wrote must equal what the crate's `f32_to_half` produces for the
//! same samples. Values in (2^-25, 2^-24) round up to the smallest
//! subnormal (0x0001) under round-to-nearest-even; 2^-25 itself is the
//! tie and rounds to zero. Auto-skips when the binary is absent.

use std::process::Command;

use oxideav_openexr::half::f32_to_half;
use oxideav_openexr::{
    encode_exr_scanline, parse_exr, Attribute, AttributeValue, Box2i, Channel, Compression,
    LineOrder, PixelType,
};

fn tool_available(name: &str) -> bool {
    Command::new(name)
        .arg("--help")
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

#[test]
fn reference_half_conversion_matches_f32_to_half_at_the_subnormal_boundary() {
    if !tool_available("exrmetrics") {
        eprintln!("exrmetrics not available, skipping");
        return;
    }
    let vals: Vec<f32> = [
        0x3340_0000u32,
        0x3301_47ae,
        0x3300_0000,
        0x337f_be77,
        0x3380_0000,
        0x33c0_0000,
        0x32ff_ffff,
        0xb340_0000,
        0x3f80_0000,
        0x477f_e000,
        0x0000_0001,
    ]
    .iter()
    .map(|&b| f32::from_bits(b))
    .collect();
    let w = vals.len() as u32;
    let ch = Channel {
        name: "Y".to_string(),
        pixel_type: PixelType::Float,
        p_linear: false,
        x_sampling: 1,
        y_sampling: 1,
    };
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (w - 1) as i32,
        y_max: 0,
    };
    let attrs = vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(vec![ch.clone()]),
        },
        Attribute {
            name: "compression".to_string(),
            value: AttributeValue::Compression(Compression::None),
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
    ];
    let bytes = encode_exr_scanline(w, 1, &[ch], &[&vals], Compression::None, attrs).unwrap();
    let dir = std::env::temp_dir().join(format!(
        "oxideav-openexr-halfsub-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("float.exr");
    let dst = dir.join("half.exr");
    std::fs::write(&src, &bytes).unwrap();
    let out = Command::new("exrmetrics")
        .args(["--convert", "--pixelmode", "half", "-o"])
        .arg(&dst)
        .arg(&src)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "exrmetrics failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let img = parse_exr(&std::fs::read(&dst).unwrap()).unwrap();
    assert_eq!(img.channels[0].pixel_type, PixelType::Half);
    for (v, got) in vals.iter().zip(img.planes[0].samples.iter()) {
        // `got` is the reference's HALF widened exactly, so re-encoding
        // it is lossless and yields the code it wrote.
        let theirs = f32_to_half(*got);
        let ours = f32_to_half(*v);
        assert_eq!(
            ours,
            theirs,
            "f32 {:#010x}: ours {ours:#06x} vs reference {theirs:#06x}",
            v.to_bits()
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
