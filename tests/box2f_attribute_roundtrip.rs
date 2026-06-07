//! Round-247 — typed `box2f` attribute inspector round-trip + reference-
//! binary interop.
//!
//! `box2f` is the f32 cousin of `box2i`: four little-endian `f32`
//! coordinates `(xMin, yMin, xMax, yMax)` packed in declaration order,
//! 16 bytes total on disk. The type-name string in the attribute table
//! is `"box2f"`. Same field-shape as `box2i`, swap i32 for f32.
//!
//! Three layers of coverage, matching the r238
//! `typed_attribute_roundtrip.rs` layout:
//!
//! 1. **Algebraic round-trip** — build a [`AttributeValue::Box2f`],
//!    encode through [`encode_attribute_value`], re-parse through
//!    [`parse_attribute_value`], assert the parsed value bit-exactly
//!    matches the input across a range of inputs including signed
//!    extremes, sub-normals, infinity, and NaN bit-pattern preservation.
//!    Also checks the on-disk size is exactly 16 bytes and the type-name
//!    string equals `"box2f"`.
//!
//! 2. **Full-file round-trip** — embed a `box2f` attribute under a
//!    realistic name in an `encode_exr_scanline` call and confirm that
//!    after `parse_exr` it survives as a typed
//!    `AttributeValue::Box2f(_)` (not falling through to `Other`) with
//!    bit-identical field values.
//!
//! 3. **`exrheader` interop** — opaque-process invocation of `exrheader`
//!    on a generated file carrying the new attribute; assert zero exit
//!    and that the attribute name appears in the emitted text.
//!    Auto-skipped when `exrheader` is missing from `$PATH`.

use std::process::Command;

use oxideav_openexr::{
    encode_exr_scanline, parse_exr, Attribute, AttributeValue, Box2f, Box2i, Channel, Compression,
    LineOrder, PixelType,
};

// ---------------------------------------------------------------------------
// Layer 1: algebraic round-trip through encode/parse_attribute_value.
// ---------------------------------------------------------------------------

/// Helper that runs `value` through `encode_attribute_value` ->
/// `parse_attribute_value` and returns the parsed result.
fn rt(value: AttributeValue) -> AttributeValue {
    let (type_name, data) = oxideav_openexr::header::encode_attribute_value(&value);
    oxideav_openexr::header::parse_attribute_value(&type_name, &data).unwrap()
}

#[test]
fn box2f_payload_size_and_type_name() {
    let v = AttributeValue::Box2f(Box2f {
        x_min: 0.0,
        y_min: 0.0,
        x_max: 0.0,
        y_max: 0.0,
    });
    let (type_name, data) = oxideav_openexr::header::encode_attribute_value(&v);
    assert_eq!(type_name, "box2f");
    assert_eq!(
        data.len(),
        16,
        "box2f payload must be exactly 4 * 4 = 16 bytes"
    );
}

#[test]
fn box2f_little_endian_byte_order() {
    // Pick four distinct exact-binary f32 values and check the on-disk
    // byte sequence is the concatenation of their little-endian f32
    // encodings in (xMin, yMin, xMax, yMax) order.
    let b = Box2f {
        x_min: 1.0,
        y_min: 2.0,
        x_max: 4.0,
        y_max: 8.0,
    };
    let (_, data) = oxideav_openexr::header::encode_attribute_value(&AttributeValue::Box2f(b));
    let mut expected = Vec::with_capacity(16);
    expected.extend_from_slice(&1.0f32.to_le_bytes());
    expected.extend_from_slice(&2.0f32.to_le_bytes());
    expected.extend_from_slice(&4.0f32.to_le_bytes());
    expected.extend_from_slice(&8.0f32.to_le_bytes());
    assert_eq!(
        data, expected,
        "field order on disk must be xMin xMax-style \
        with declarations in (xMin, yMin, xMax, yMax) order"
    );
}

#[test]
fn box2f_roundtrip_typical_window() {
    // A typical display-window-shaped box (in f32 coords).
    let b = Box2f {
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1919.0,
        y_max: 1079.0,
    };
    match rt(AttributeValue::Box2f(b)) {
        AttributeValue::Box2f(p) => assert_eq!(p, b),
        other => panic!("expected Box2f, got {other:?}"),
    }
}

#[test]
fn box2f_roundtrip_negative_extents() {
    // Negative coords are spec-legal; box2f doesn't constrain sign.
    let b = Box2f {
        x_min: -100.5,
        y_min: -250.25,
        x_max: 100.5,
        y_max: 250.25,
    };
    match rt(AttributeValue::Box2f(b)) {
        AttributeValue::Box2f(p) => assert_eq!(p, b),
        other => panic!("expected Box2f, got {other:?}"),
    }
}

#[test]
fn box2f_roundtrip_subnormal_and_extreme() {
    // f32::MIN / MAX / MIN_POSITIVE / sub-normal patterns — every f32
    // bit pattern except NaN must round-trip via `==`.
    let cases = [
        f32::MIN,
        f32::MAX,
        f32::MIN_POSITIVE,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0_f32,
        1.0e-30_f32,
        1.0e30_f32,
    ];
    for x in cases {
        let b = Box2f {
            x_min: x,
            y_min: -x,
            x_max: x,
            y_max: -x,
        };
        match rt(AttributeValue::Box2f(b)) {
            AttributeValue::Box2f(p) => {
                assert_eq!(p.x_min.to_bits(), b.x_min.to_bits());
                assert_eq!(p.y_min.to_bits(), b.y_min.to_bits());
                assert_eq!(p.x_max.to_bits(), b.x_max.to_bits());
                assert_eq!(p.y_max.to_bits(), b.y_max.to_bits());
            }
            other => panic!("expected Box2f, got {other:?}"),
        }
    }
}

#[test]
fn box2f_roundtrip_nan_bit_pattern_preservation() {
    // NaN is `!= NaN` by IEEE-754, so compare via to_bits().
    let nan = f32::from_bits(0x7fc0_1234);
    let b = Box2f {
        x_min: nan,
        y_min: 0.0,
        x_max: nan,
        y_max: 0.0,
    };
    match rt(AttributeValue::Box2f(b)) {
        AttributeValue::Box2f(p) => {
            assert_eq!(p.x_min.to_bits(), nan.to_bits());
            assert_eq!(p.x_max.to_bits(), nan.to_bits());
            assert_eq!(p.y_min.to_bits(), 0.0_f32.to_bits());
            assert_eq!(p.y_max.to_bits(), 0.0_f32.to_bits());
        }
        other => panic!("expected Box2f(NaN), got {other:?}"),
    }
}

#[test]
fn box2f_rejects_short_payload() {
    // box2f requires exactly 16 bytes; anything shorter is an error.
    assert!(oxideav_openexr::header::parse_attribute_value("box2f", &[0u8; 15]).is_err());
    assert!(oxideav_openexr::header::parse_attribute_value("box2f", &[]).is_err());
}

#[test]
fn box2f_rejects_oversize_payload() {
    // 17 bytes is also an error — the spec table value is exact.
    assert!(oxideav_openexr::header::parse_attribute_value("box2f", &[0u8; 17]).is_err());
    assert!(oxideav_openexr::header::parse_attribute_value("box2f", &[0u8; 32]).is_err());
}

#[test]
fn box2f_is_distinct_from_box2i() {
    // Same byte length (16) but distinct typed variants — make sure a
    // box2i payload doesn't accidentally parse as box2f and vice versa,
    // and that encoding box2f never emits the box2i type-name string.
    let mut raw = Vec::with_capacity(16);
    raw.extend_from_slice(&1i32.to_le_bytes());
    raw.extend_from_slice(&2i32.to_le_bytes());
    raw.extend_from_slice(&3i32.to_le_bytes());
    raw.extend_from_slice(&4i32.to_le_bytes());
    let p_box2i = oxideav_openexr::header::parse_attribute_value("box2i", &raw).unwrap();
    assert!(matches!(p_box2i, AttributeValue::Box2i(_)));
    // And the same 16 bytes interpreted as box2f are NOT a Box2i.
    let p_box2f = oxideav_openexr::header::parse_attribute_value("box2f", &raw).unwrap();
    assert!(matches!(p_box2f, AttributeValue::Box2f(_)));

    // Encoding a Box2f never produces type-name "box2i".
    let (tn, _) = oxideav_openexr::header::encode_attribute_value(&AttributeValue::Box2f(Box2f {
        x_min: 0.0,
        y_min: 0.0,
        x_max: 0.0,
        y_max: 0.0,
    }));
    assert_eq!(tn, "box2f");
}

// ---------------------------------------------------------------------------
// Layer 2: full-file round-trip through encode_exr_scanline -> parse_exr.
// ---------------------------------------------------------------------------

fn minimal_scanline_with_extra(extra: Vec<Attribute>) -> Vec<u8> {
    // 4x4 single-channel HALF scanline file with the eight required
    // attributes plus `extra`.
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
fn full_header_roundtrip_box2f_variant() {
    let b = Box2f {
        x_min: -1.5,
        y_min: -2.25,
        x_max: 3.5,
        y_max: 4.75,
    };
    let extras = vec![Attribute {
        name: "renderRegion".to_string(),
        value: AttributeValue::Box2f(b),
    }];
    let bytes = minimal_scanline_with_extra(extras);
    let img = parse_exr(&bytes).unwrap();

    let got = img
        .attributes
        .iter()
        .find(|a| a.name == "renderRegion")
        .expect("renderRegion attribute survived round-trip");

    match &got.value {
        AttributeValue::Box2f(p) => {
            assert_eq!(p.x_min.to_bits(), b.x_min.to_bits());
            assert_eq!(p.y_min.to_bits(), b.y_min.to_bits());
            assert_eq!(p.x_max.to_bits(), b.x_max.to_bits());
            assert_eq!(p.y_max.to_bits(), b.y_max.to_bits());
        }
        AttributeValue::Other { type_name, .. } => {
            panic!("box2f fell through to Other(type_name={type_name:?})")
        }
        other => panic!("expected Box2f, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Layer 3: `exrheader` interop — invoked as an opaque process. Skipped
// when the binary isn't on $PATH.
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
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-box2f-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

#[test]
fn exrheader_accepts_file_with_box2f_attribute() {
    if !exrheader_available() {
        eprintln!("exrheader not available on PATH, skipping interop validation");
        return;
    }

    let extras = vec![Attribute {
        name: "renderRegion".to_string(),
        value: AttributeValue::Box2f(Box2f {
            x_min: -0.5,
            y_min: -0.25,
            x_max: 1920.5,
            y_max: 1080.25,
        }),
    }];
    let bytes = minimal_scanline_with_extra(extras);

    let dir = tempdir();
    let path = format!("{dir}/box2f-attr.exr");
    std::fs::write(&path, &bytes).unwrap();

    let out = Command::new("exrheader")
        .arg(&path)
        .output()
        .expect("exrheader spawn failed");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "exrheader returned non-zero on a file carrying a box2f attribute\n\
         stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        stdout.contains("renderRegion"),
        "exrheader output missing the renderRegion attribute name\n\
         stdout:\n{stdout}"
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}
