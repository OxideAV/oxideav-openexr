//! Typed-attribute round-trip + reference-binary interop coverage for the
//! attribute variants added to [`AttributeValue`]: `Int`, `Double`,
//! `String`, `V2i`, `V3i`, `V3f`, `M33f`, `M44f`, `M33d`, `M44d`,
//! `Chromaticities`.
//!
//! Three layers of coverage:
//!
//! 1. **Algebraic round-trip** — for each variant, build an `Attribute`,
//!    encode through [`encode_attribute_value`], re-parse through
//!    [`parse_attribute_value`], and assert equality on the
//!    `AttributeValue`. Confirms the on-disk size matches each spec
//!    table (4 / 8 / 12 / 36 / 64 / 32 bytes plus the variable-length
//!    `String`) and that the LE byte order is exactly inverse.
//!
//! 2. **Full-header round-trip** — assemble a scanline `ParsedHeader`
//!    that carries every new typed variant under realistic attribute
//!    names (`worldToCamera` for `m44f`, `chromaticities`, `name` for
//!    `string`, etc.), encode via [`encode_header`] + a minimal
//!    scanline body, re-parse via [`parse_exr`], and assert every
//!    inserted attribute is present with the typed variant and the
//!    exact value.
//!
//! 3. **`exrheader` interop** — if the `exrheader` binary is reachable
//!    via `$PATH`, write the encoded bytes to a temp file and invoke
//!    `exrheader` as an opaque process (input bytes in, stdout text
//!    out). Assert it exits zero and that its emitted attribute lines
//!    mention the new attribute names. The test is skipped (with a
//!    printed reason) when the binary is missing — same shape as the
//!    pre-existing `exrheader_validation.rs` skip path.

use std::process::Command;

use oxideav_openexr::{
    encode_exr_scanline, parse_exr, Attribute, AttributeValue, Box2i, Channel, Chromaticities,
    Compression, LineOrder, PixelType,
};

// ---------------------------------------------------------------------------
// Layer 1: algebraic round-trips through parse/encode_attribute_value.
// ---------------------------------------------------------------------------

/// Helper that runs `value` through encode->parse and returns the parsed
/// value (the `type_name` is derived by the encoder, matching the spec
/// table for each variant).
fn rt(value: AttributeValue) -> AttributeValue {
    let (type_name, data) = oxideav_openexr::header::encode_attribute_value(&value);
    oxideav_openexr::header::parse_attribute_value(&type_name, &data).unwrap()
}

#[test]
fn int_roundtrip_extremes() {
    for v in [0i32, 1, -1, i32::MIN, i32::MAX, 100_000, -100_000] {
        assert_eq!(rt(AttributeValue::Int(v)), AttributeValue::Int(v));
    }
}

#[test]
fn double_roundtrip_bits() {
    let cases = [
        0.0_f64,
        -0.0,
        1.0,
        -1.0,
        std::f64::consts::PI,
        std::f64::consts::E,
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN,
        1e-300,
        1e300,
    ];
    for v in cases {
        let parsed = rt(AttributeValue::Double(v));
        match parsed {
            AttributeValue::Double(p) => assert_eq!(p.to_bits(), v.to_bits()),
            other => panic!("expected Double, got {other:?}"),
        }
    }
    // NaN bit-pattern preservation.
    let nan = f64::from_bits(0x7ff8_0000_0000_1234);
    let parsed = rt(AttributeValue::Double(nan));
    match parsed {
        AttributeValue::Double(p) => assert_eq!(p.to_bits(), nan.to_bits()),
        other => panic!("expected Double(NaN), got {other:?}"),
    }
}

#[test]
fn string_roundtrip_ascii_and_utf8() {
    for s in [
        "",
        "scanlineimage",
        "tiledimage",
        "deeptile",
        "render-layer/diffuse_direct.001",
        "héllo, wörld", // multi-byte UTF-8
        "日本語",       // 3-byte CJK
        "🎬🎞️",         // 4-byte + ZWJ sequence
    ] {
        assert_eq!(
            rt(AttributeValue::String(s.to_string())),
            AttributeValue::String(s.to_string()),
            "string round-trip failed for {s:?}"
        );
    }
}

#[test]
fn vector_roundtrips() {
    assert_eq!(rt(AttributeValue::V2i(-7, 42)), AttributeValue::V2i(-7, 42));
    assert_eq!(
        rt(AttributeValue::V3i(i32::MIN, 0, i32::MAX)),
        AttributeValue::V3i(i32::MIN, 0, i32::MAX)
    );
    assert_eq!(
        rt(AttributeValue::V3f(1.5, -2.25, 3.125)),
        AttributeValue::V3f(1.5, -2.25, 3.125)
    );
}

#[test]
fn matrix_roundtrips() {
    let m33 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    assert_eq!(rt(AttributeValue::M33f(m33)), AttributeValue::M33f(m33));

    let m44 = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    assert_eq!(rt(AttributeValue::M44f(m44)), AttributeValue::M44f(m44));

    // Double-precision matrices use values that have no exact f32
    // representation, so a round-trip through f32 would corrupt them —
    // proving the m33d/m44d path preserves full f64 width.
    let m33d = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, std::f64::consts::PI];
    assert_eq!(rt(AttributeValue::M33d(m33d)), AttributeValue::M33d(m33d));

    let m44d = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0 / 3.0,
        std::f64::consts::E,
        std::f64::consts::PI,
        -1.0 / 7.0,
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::MIN,
    ];
    assert_eq!(rt(AttributeValue::M44d(m44d)), AttributeValue::M44d(m44d));
}

#[test]
fn chromaticities_roundtrip_bt709() {
    // BT.709 primaries (a widely-used reference set; pure-math values).
    let c = Chromaticities {
        red_x: 0.6400,
        red_y: 0.3300,
        green_x: 0.3000,
        green_y: 0.6000,
        blue_x: 0.1500,
        blue_y: 0.0600,
        white_x: 0.3127,
        white_y: 0.3290,
    };
    let v = rt(AttributeValue::Chromaticities(c));
    match v {
        AttributeValue::Chromaticities(p) => {
            assert_eq!(p.red_x.to_bits(), c.red_x.to_bits());
            assert_eq!(p.red_y.to_bits(), c.red_y.to_bits());
            assert_eq!(p.green_x.to_bits(), c.green_x.to_bits());
            assert_eq!(p.green_y.to_bits(), c.green_y.to_bits());
            assert_eq!(p.blue_x.to_bits(), c.blue_x.to_bits());
            assert_eq!(p.blue_y.to_bits(), c.blue_y.to_bits());
            assert_eq!(p.white_x.to_bits(), c.white_x.to_bits());
            assert_eq!(p.white_y.to_bits(), c.white_y.to_bits());
        }
        other => panic!("expected Chromaticities, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Layer 1b: payload size + endianness sanity (the spec table values).
// ---------------------------------------------------------------------------

#[test]
fn payload_sizes_match_spec_table() {
    let cases: &[(AttributeValue, &str, usize)] = &[
        (AttributeValue::Int(0), "int", 4),
        (AttributeValue::Double(0.0), "double", 8),
        (AttributeValue::V2i(0, 0), "v2i", 8),
        (AttributeValue::V3i(0, 0, 0), "v3i", 12),
        (AttributeValue::V3f(0.0, 0.0, 0.0), "v3f", 12),
        (AttributeValue::M33f([0.0; 9]), "m33f", 36),
        (AttributeValue::M44f([0.0; 16]), "m44f", 64),
        (AttributeValue::M33d([0.0; 9]), "m33d", 72),
        (AttributeValue::M44d([0.0; 16]), "m44d", 128),
        (
            AttributeValue::Chromaticities(Chromaticities {
                red_x: 0.0,
                red_y: 0.0,
                green_x: 0.0,
                green_y: 0.0,
                blue_x: 0.0,
                blue_y: 0.0,
                white_x: 0.0,
                white_y: 0.0,
            }),
            "chromaticities",
            32,
        ),
    ];
    for (val, expected_name, expected_size) in cases {
        let (type_name, data) = oxideav_openexr::header::encode_attribute_value(val);
        assert_eq!(type_name, *expected_name, "type-name mismatch for {val:?}");
        assert_eq!(
            data.len(),
            *expected_size,
            "payload size mismatch for {val:?}"
        );
    }
}

#[test]
fn int_endianness_is_little_endian() {
    let (_, data) =
        oxideav_openexr::header::encode_attribute_value(&AttributeValue::Int(0x0123_4567));
    // 0x0123_4567 -> LE byte order
    assert_eq!(data, vec![0x67, 0x45, 0x23, 0x01]);
}

#[test]
fn rejects_short_int_payload() {
    let r = oxideav_openexr::header::parse_attribute_value("int", &[0, 0, 0]);
    assert!(r.is_err(), "3-byte int payload must error");
}

#[test]
fn rejects_oversize_m33f_payload() {
    let r = oxideav_openexr::header::parse_attribute_value("m33f", &[0u8; 37]);
    assert!(r.is_err(), "37-byte m33f payload must error");
}

#[test]
fn rejects_wrong_size_double_matrix_payloads() {
    // m33d is exactly 72 bytes (9 × f64); m44d is exactly 128 (16 × f64).
    assert!(
        oxideav_openexr::header::parse_attribute_value("m33d", &[0u8; 71]).is_err(),
        "71-byte m33d payload must error"
    );
    assert!(
        oxideav_openexr::header::parse_attribute_value("m33d", &[0u8; 73]).is_err(),
        "73-byte m33d payload must error"
    );
    assert!(
        oxideav_openexr::header::parse_attribute_value("m44d", &[0u8; 127]).is_err(),
        "127-byte m44d payload must error"
    );
    assert!(
        oxideav_openexr::header::parse_attribute_value("m44d", &[0u8; 129]).is_err(),
        "129-byte m44d payload must error"
    );
    // Exact sizes parse.
    assert!(oxideav_openexr::header::parse_attribute_value("m33d", &[0u8; 72]).is_ok());
    assert!(oxideav_openexr::header::parse_attribute_value("m44d", &[0u8; 128]).is_ok());
}

// ---------------------------------------------------------------------------
// Layer 2: full-file round-trip through encode_exr_scanline -> parse_exr.
// ---------------------------------------------------------------------------

fn minimal_scanline_with_extra_attrs(extra: Vec<Attribute>) -> Vec<u8> {
    // 4x4 single-channel HALF file with one custom-extra attribute set.
    let w = 4u32;
    let h = 4u32;
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (w - 1) as i32,
        y_max: (h - 1) as i32,
    };
    let chs = vec![Channel {
        name: "Y".to_string(),
        pixel_type: PixelType::Half,
        p_linear: false,
        x_sampling: 1,
        y_sampling: 1,
    }];
    let plane: Vec<f32> = (0..(w * h)).map(|i| i as f32 * 0.125).collect();
    let mut attrs = vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(chs.clone()),
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
    attrs.extend(extra);
    encode_exr_scanline(w, h, &chs, &[&plane], Compression::None, attrs).unwrap()
}

#[test]
fn full_header_roundtrip_all_typed_variants() {
    let chromat = Chromaticities {
        red_x: 0.708,
        red_y: 0.292,
        green_x: 0.170,
        green_y: 0.797,
        blue_x: 0.131,
        blue_y: 0.046,
        white_x: 0.32168,
        white_y: 0.33767,
    };
    let m33 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let m44 = [
        1.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 6.0, 0.0, 0.0, 1.0, 7.0, 0.0, 0.0, 0.0, 1.0,
    ];

    let extras = vec![
        Attribute {
            name: "chromaticities".to_string(),
            value: AttributeValue::Chromaticities(chromat),
        },
        Attribute {
            name: "exposure".to_string(),
            value: AttributeValue::Float(1.0),
        },
        Attribute {
            name: "name".to_string(),
            value: AttributeValue::String("layer-0".to_string()),
        },
        Attribute {
            name: "version".to_string(),
            value: AttributeValue::Int(7),
        },
        Attribute {
            name: "captureRate".to_string(),
            value: AttributeValue::Double(23.976023976),
        },
        Attribute {
            name: "tileOrigin".to_string(),
            value: AttributeValue::V2i(-1, 2),
        },
        Attribute {
            name: "voxelOrigin".to_string(),
            value: AttributeValue::V3i(-1, 0, 1),
        },
        Attribute {
            name: "cameraDirection".to_string(),
            value: AttributeValue::V3f(0.0, 0.0, -1.0),
        },
        Attribute {
            name: "colorMatrix".to_string(),
            value: AttributeValue::M33f(m33),
        },
        Attribute {
            name: "worldToCamera".to_string(),
            value: AttributeValue::M44f(m44),
        },
        Attribute {
            name: "colorMatrixD".to_string(),
            value: AttributeValue::M33d([0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3]),
        },
        Attribute {
            name: "worldToCameraD".to_string(),
            value: AttributeValue::M44d([
                std::f64::consts::PI,
                0.0,
                0.0,
                5.0,
                0.0,
                std::f64::consts::E,
                0.0,
                6.0,
                0.0,
                0.0,
                1.0 / 3.0,
                7.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ]),
        },
    ];

    let bytes = minimal_scanline_with_extra_attrs(extras.clone());
    let img = parse_exr(&bytes).unwrap();

    // Every extra attribute must round-trip as its typed variant with
    // bit-equal payload.
    for want in &extras {
        let got = img
            .attributes
            .iter()
            .find(|a| a.name == want.name)
            .unwrap_or_else(|| panic!("missing attribute {:?} after round-trip", want.name));
        assert_eq!(
            &got.value, &want.value,
            "attribute {:?} mismatched after round-trip",
            want.name
        );
    }

    // Sanity: no extra attribute fell through to AttributeValue::Other.
    for want in &extras {
        let got = img.attributes.iter().find(|a| a.name == want.name).unwrap();
        if let AttributeValue::Other { type_name, .. } = &got.value {
            panic!(
                "attribute {:?} fell through to Other(type_name={type_name:?})",
                want.name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Layer 3: `exrheader` interop — invoked as an opaque process (input
// bytes in, stdout text out). Auto-skipped when the binary is missing.
// ---------------------------------------------------------------------------

fn exrheader_available() -> bool {
    Command::new("exrheader")
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
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-typed-attr-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

#[test]
fn exrheader_accepts_file_with_every_typed_attribute() {
    if !exrheader_available() {
        eprintln!("exrheader not available on PATH, skipping interop validation");
        return;
    }

    let extras = vec![
        Attribute {
            name: "chromaticities".to_string(),
            value: AttributeValue::Chromaticities(Chromaticities {
                red_x: 0.708,
                red_y: 0.292,
                green_x: 0.170,
                green_y: 0.797,
                blue_x: 0.131,
                blue_y: 0.046,
                white_x: 0.32168,
                white_y: 0.33767,
            }),
        },
        Attribute {
            name: "exposure".to_string(),
            value: AttributeValue::Float(1.5),
        },
        Attribute {
            name: "owner".to_string(),
            value: AttributeValue::String("oxideav round 238b".to_string()),
        },
        Attribute {
            name: "isoSpeed".to_string(),
            value: AttributeValue::Int(800),
        },
        Attribute {
            name: "captureRate".to_string(),
            value: AttributeValue::Double(23.976023976),
        },
        Attribute {
            name: "originXY".to_string(),
            value: AttributeValue::V2i(-1, 2),
        },
        Attribute {
            name: "originXYZ".to_string(),
            value: AttributeValue::V3i(-1, 0, 1),
        },
        Attribute {
            name: "lookVector".to_string(),
            value: AttributeValue::V3f(0.0, 0.0, -1.0),
        },
        Attribute {
            name: "colorTransform".to_string(),
            value: AttributeValue::M33f([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        },
        Attribute {
            name: "worldToCamera".to_string(),
            value: AttributeValue::M44f([
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ]),
        },
        Attribute {
            name: "colorTransformD".to_string(),
            value: AttributeValue::M33d([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        },
        Attribute {
            name: "worldToCameraD".to_string(),
            value: AttributeValue::M44d([
                1.0, 0.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, 0.0, //
                0.0, 0.0, 1.0, 0.0, //
                0.0, 0.0, 0.0, 1.0,
            ]),
        },
    ];

    let bytes = minimal_scanline_with_extra_attrs(extras.clone());

    let dir = tempdir();
    let path = format!("{dir}/typed-attrs.exr");
    std::fs::write(&path, &bytes).unwrap();

    let out = Command::new("exrheader")
        .arg(&path)
        .output()
        .expect("exrheader spawn failed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "exrheader returned non-zero\nstdout: {stdout}\nstderr: {stderr}"
    );

    // Every attribute name we added should appear somewhere in
    // exrheader's text output (it prints `name (type): value`).
    for a in &extras {
        assert!(
            stdout.contains(&a.name),
            "exrheader output missing attribute {:?}\n--- stdout ---\n{stdout}",
            a.name
        );
    }

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}
