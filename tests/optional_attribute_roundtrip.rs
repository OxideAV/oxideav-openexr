//! Round-trip + reference-binary interop coverage for the optional
//! standard attribute types added to [`AttributeValue`]: `v2d`, `v3d`,
//! `rational`, `timecode`, `keycode`, and `stringvector`.
//!
//! Three layers of coverage mirror `typed_attribute_roundtrip.rs`:
//!
//! 1. **Algebraic round-trip** — encode each variant through
//!    [`encode_attribute_value`] and re-parse through
//!    [`parse_attribute_value`], asserting equality and the on-disk
//!    payload byte length (16 for `v2d`, 24 for `v3d`, 8 for `rational`
//!    / `timecode`, 28 for `keycode`, variable for `stringvector`).
//!
//! 2. **Full-header round-trip** — assemble a scanline header carrying
//!    every new variant under realistic attribute names, encode +
//!    re-parse via [`parse_exr`], and assert each survives with the
//!    typed variant intact.
//!
//! 3. **`exrheader` interop** — if the `exrheader` binary is reachable
//!    via `$PATH`, write the bytes to a temp file and invoke `exrheader`
//!    as an opaque process. The layouts implemented here were derived
//!    purely from this validator's text rendering (BCD time fields,
//!    keycode field order + ranges, `rational` `n/d` form). Skipped with
//!    a printed reason when the binary is missing.

use std::process::Command;

use oxideav_openexr::header::{encode_attribute_value, parse_attribute_value};
use oxideav_openexr::{
    encode_exr_scanline, parse_exr, Attribute, AttributeValue, Box2i, Channel, Compression,
    Keycode, LineOrder, PixelType, Timecode,
};

// ---------------------------------------------------------------------------
// Layer 1: algebraic round-trips + on-disk payload size.
// ---------------------------------------------------------------------------

fn rt(value: AttributeValue) -> (String, usize, AttributeValue) {
    let (type_name, data) = encode_attribute_value(&value);
    let len = data.len();
    let parsed = parse_attribute_value(&type_name, &data).unwrap();
    (type_name, len, parsed)
}

#[test]
fn v2d_roundtrip() {
    let (ty, len, parsed) = rt(AttributeValue::V2d(1.5, -2.5));
    assert_eq!(ty, "v2d");
    assert_eq!(len, 16);
    assert_eq!(parsed, AttributeValue::V2d(1.5, -2.5));
    // bit-exact f64 preservation incl. extremes + NaN.
    let nan = f64::from_bits(0x7ff8_0000_0000_abcd);
    match rt(AttributeValue::V2d(f64::MIN, nan)).2 {
        AttributeValue::V2d(a, b) => {
            assert_eq!(a.to_bits(), f64::MIN.to_bits());
            assert_eq!(b.to_bits(), nan.to_bits());
        }
        other => panic!("expected V2d, got {other:?}"),
    }
}

#[test]
fn v3d_roundtrip() {
    let (ty, len, parsed) = rt(AttributeValue::V3d(1.0, 2.0, 3.0));
    assert_eq!(ty, "v3d");
    assert_eq!(len, 24);
    assert_eq!(parsed, AttributeValue::V3d(1.0, 2.0, 3.0));
}

#[test]
fn rational_roundtrip() {
    for (n, d) in [(24i32, 1u32), (-1, 2), (30000, 1001), (i32::MIN, u32::MAX)] {
        let (ty, len, parsed) = rt(AttributeValue::Rational(n, d));
        assert_eq!(ty, "rational");
        assert_eq!(len, 8);
        assert_eq!(parsed, AttributeValue::Rational(n, d));
    }
}

#[test]
fn timecode_roundtrip_and_bcd_accessors() {
    // 0x01020304 renders as 01:02:03:04 in the reference validator.
    let tc = Timecode {
        time_and_flags: 0x0102_0304,
        user_data: 0xdead_beef,
    };
    let (ty, len, parsed) = rt(AttributeValue::Timecode(tc));
    assert_eq!(ty, "timecode");
    assert_eq!(len, 8);
    assert_eq!(parsed, AttributeValue::Timecode(tc));
    assert_eq!(tc.hours(), 1);
    assert_eq!(tc.minutes(), 2);
    assert_eq!(tc.seconds(), 3);
    assert_eq!(tc.frames(), 4);

    // 23:59:29:24 — the maximum NTSC drop-frame-ish corner the validator
    // accepted in probing.
    let tc2 = Timecode {
        time_and_flags: 0x2359_2924,
        user_data: 0,
    };
    assert_eq!(tc2.hours(), 23);
    assert_eq!(tc2.minutes(), 59);
    assert_eq!(tc2.seconds(), 29);
    assert_eq!(tc2.frames(), 24);
}

#[test]
fn keycode_roundtrip_field_order() {
    let kc = Keycode {
        film_mfc_code: 10,
        film_type: 5,
        prefix: 123456,
        count: 1000,
        perf_offset: 30,
        perfs_per_frame: 4,
        perfs_per_count: 60,
    };
    let (ty, len, parsed) = rt(AttributeValue::Keycode(kc));
    assert_eq!(ty, "keycode");
    assert_eq!(len, 28);
    assert_eq!(parsed, AttributeValue::Keycode(kc));

    // Confirm the on-disk field order is exactly the seven i32 in the
    // declared sequence (the order the reference validator labels).
    let (_, data) = encode_attribute_value(&AttributeValue::Keycode(kc));
    let read = |i: usize| i32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap());
    assert_eq!(read(0), 10);
    assert_eq!(read(1), 5);
    assert_eq!(read(2), 123456);
    assert_eq!(read(3), 1000);
    assert_eq!(read(4), 30);
    assert_eq!(read(5), 4);
    assert_eq!(read(6), 60);
}

#[test]
fn stringvector_roundtrip() {
    for entries in [
        vec![],
        vec!["alpha".to_string()],
        vec!["foo".to_string(), "".to_string(), "bar".to_string()],
        vec!["日本語".to_string(), "🎬".to_string()],
    ] {
        let (ty, _len, parsed) = rt(AttributeValue::StringVector(entries.clone()));
        assert_eq!(ty, "stringvector");
        assert_eq!(parsed, AttributeValue::StringVector(entries));
    }
}

#[test]
fn parse_rejects_malformed_sizes() {
    assert!(parse_attribute_value("v2d", &[0u8; 8]).is_err());
    assert!(parse_attribute_value("v3d", &[0u8; 16]).is_err());
    assert!(parse_attribute_value("rational", &[0u8; 4]).is_err());
    assert!(parse_attribute_value("timecode", &[0u8; 4]).is_err());
    assert!(parse_attribute_value("keycode", &[0u8; 20]).is_err());
    // stringvector with a length field that overruns the payload.
    let mut bad = Vec::new();
    bad.extend_from_slice(&100i32.to_le_bytes());
    bad.extend_from_slice(b"short");
    assert!(parse_attribute_value("stringvector", &bad).is_err());
}

// ---------------------------------------------------------------------------
// Layer 2: full-header round-trip through parse_exr.
// ---------------------------------------------------------------------------

fn base_attributes(w: u32, h: u32) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (w - 1) as i32,
        y_max: (h - 1) as i32,
    };
    let chs = vec![
        Channel {
            name: "A".to_string(),
            pixel_type: PixelType::Half,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        },
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
    vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(chs),
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
    ]
}

fn build_file(w: u32, h: u32, extra: Vec<Attribute>) -> Vec<u8> {
    let mut attrs = base_attributes(w, h);
    attrs.extend(extra);
    let chs = match &attrs[0].value {
        AttributeValue::Channels(c) => c.clone(),
        _ => unreachable!(),
    };
    let plane = vec![0.5f32; (w * h) as usize];
    let planes: Vec<&[f32]> = vec![&plane, &plane, &plane, &plane];
    encode_exr_scanline(w, h, &chs, &planes, Compression::None, attrs).unwrap()
}

#[test]
fn full_header_roundtrip_through_parse_exr() {
    let extra = vec![
        Attribute {
            name: "worldToNDC".to_string(),
            value: AttributeValue::V3d(1.0, 2.0, 3.0),
        },
        Attribute {
            name: "originXY".to_string(),
            value: AttributeValue::V2d(-4.0, 5.0),
        },
        Attribute {
            name: "framesPerSecond".to_string(),
            value: AttributeValue::Rational(24, 1),
        },
        Attribute {
            name: "timeCode".to_string(),
            value: AttributeValue::Timecode(Timecode {
                time_and_flags: 0x0102_0304,
                user_data: 0,
            }),
        },
        Attribute {
            name: "keyCode".to_string(),
            value: AttributeValue::Keycode(Keycode {
                film_mfc_code: 1,
                film_type: 2,
                prefix: 3,
                count: 4,
                perf_offset: 5,
                perfs_per_frame: 4,
                perfs_per_count: 64,
            }),
        },
        Attribute {
            name: "channelTags".to_string(),
            value: AttributeValue::StringVector(vec![
                "diffuse".to_string(),
                "specular".to_string(),
            ]),
        },
    ];
    let bytes = build_file(4, 4, extra.clone());
    let img = parse_exr(&bytes).unwrap();
    for want in &extra {
        let got = img
            .attributes
            .iter()
            .find(|a| a.name == want.name)
            .unwrap_or_else(|| panic!("attribute {} missing after round-trip", want.name));
        assert_eq!(got.value, want.value, "mismatch for {}", want.name);
    }
}

// ---------------------------------------------------------------------------
// Layer 3: exrheader interop (opaque validator).
// ---------------------------------------------------------------------------

fn exrheader_available() -> bool {
    Command::new("exrheader")
        .arg("--help")
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

fn exrheader_text(bytes: &[u8]) -> Option<String> {
    if !exrheader_available() {
        eprintln!("exrheader not available on PATH, skipping validation");
        return None;
    }
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("oxideav-openexr-optattr-{nanos}.exr"));
    std::fs::write(&path, bytes).unwrap();
    let output = Command::new("exrheader")
        .arg(&path)
        .output()
        .expect("exrheader spawn failed");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "exrheader returned non-zero\nstdout: {stdout}\nstderr: {stderr}"
    );
    let _ = std::fs::remove_file(&path);
    Some(stdout)
}

#[test]
fn exrheader_renders_optional_attributes() {
    let extra = vec![
        Attribute {
            name: "framesPerSecond".to_string(),
            value: AttributeValue::Rational(24, 1),
        },
        Attribute {
            name: "timeCode".to_string(),
            value: AttributeValue::Timecode(Timecode {
                time_and_flags: 0x0102_0304,
                user_data: 0,
            }),
        },
        Attribute {
            name: "keyCode".to_string(),
            value: AttributeValue::Keycode(Keycode {
                film_mfc_code: 10,
                film_type: 5,
                prefix: 123456,
                count: 1000,
                perf_offset: 30,
                perfs_per_frame: 4,
                perfs_per_count: 60,
            }),
        },
        Attribute {
            name: "layerNames".to_string(),
            value: AttributeValue::StringVector(vec!["alpha".to_string(), "beta".to_string()]),
        },
    ];
    let bytes = build_file(2, 2, extra);
    let Some(text) = exrheader_text(&bytes) else {
        return;
    };
    // rational n/d form.
    assert!(
        text.contains("24/1"),
        "exrheader did not render the rational as 24/1:\n{text}"
    );
    // timecode BCD time field.
    assert!(
        text.contains("01:02:03:04"),
        "exrheader did not render the timecode BCD time:\n{text}"
    );
    // keycode named fields decoded from our field order.
    assert!(
        text.contains("prefix 123456"),
        "exrheader did not decode keycode prefix:\n{text}"
    );
    assert!(
        text.contains("perfs per count 60"),
        "exrheader did not decode keycode perfs-per-count:\n{text}"
    );
    // stringvector entries.
    assert!(
        text.contains("\"alpha\"") && text.contains("\"beta\""),
        "exrheader did not render the stringvector entries:\n{text}"
    );
}
