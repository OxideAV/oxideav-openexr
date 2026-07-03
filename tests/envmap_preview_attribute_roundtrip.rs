//! Typed `envmap` / `preview` / `floatvector` / `deepImageState` header
//! attributes (round 385).
//!
//! Layouts derived empirically from the opaque `exrheader` validator
//! (the round-273 methodology): `envmap` is a single byte the validator
//! renders as "latitude-longitude map" (0) / "cube-face map" (1);
//! `preview` is two little-endian `u32` dimensions plus `4·w·h` pixel
//! bytes (the validator renders "W by H pixels" and refuses files whose
//! payload size disagrees with the dimensions); `floatvector` is a
//! sequence of little-endian `f32` (the validator refuses payloads that
//! are not a multiple of 4 bytes); `deepImageState` is a single byte
//! (the validator refuses wider payloads).
//!
//! Coverage: algebraic round-trips through
//! `encode_attribute_value` / `parse_attribute_value`, malformed-size
//! rejection, full scanline-file round-trips through
//! `encode_exr_scanline` + `parse_exr`, and `exrheader` interop
//! (auto-skipped when the binary is absent).

use std::process::Command;

use oxideav_openexr::header::{encode_attribute_value, parse_attribute_value};
use oxideav_openexr::{
    encode_exr_scanline, parse_exr, Attribute, AttributeValue, Box2i, Channel, Compression, EnvMap,
    LineOrder, PixelType, Preview,
};

fn exrheader_available() -> bool {
    Command::new("exrheader")
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

fn roundtrip(v: &AttributeValue) -> AttributeValue {
    let (type_name, bytes) = encode_attribute_value(v);
    parse_attribute_value(&type_name, &bytes).unwrap()
}

fn sample_preview() -> Preview {
    Preview {
        width: 2,
        height: 3,
        rgba: (0..24u8).collect(),
    }
}

// ---------------------------------------------------------------------
// Algebraic round-trips + on-disk layout pins.
// ---------------------------------------------------------------------

#[test]
fn envmap_algebraic_roundtrip_and_layout() {
    for (variant, byte) in [
        (EnvMap::LatLong, 0u8),
        (EnvMap::Cube, 1),
        (EnvMap::Unknown(9), 9),
        (EnvMap::Unknown(255), 255),
    ] {
        let v = AttributeValue::EnvMap(variant);
        let (ty, bytes) = encode_attribute_value(&v);
        assert_eq!(ty, "envmap");
        assert_eq!(bytes, vec![byte], "{variant:?} on-disk byte");
        assert_eq!(roundtrip(&v), v);
    }
    // from_byte/to_byte are inverses across the whole byte range.
    for b in 0..=255u8 {
        assert_eq!(EnvMap::from_byte(b).to_byte(), b);
    }
}

#[test]
fn preview_algebraic_roundtrip_and_layout() {
    let p = sample_preview();
    let v = AttributeValue::Preview(p.clone());
    let (ty, bytes) = encode_attribute_value(&v);
    assert_eq!(ty, "preview");
    assert_eq!(bytes.len(), 8 + 24);
    assert_eq!(&bytes[0..4], &2u32.to_le_bytes());
    assert_eq!(&bytes[4..8], &3u32.to_le_bytes());
    assert_eq!(&bytes[8..], &p.rgba[..]);
    assert_eq!(roundtrip(&v), v);

    // Zero-dimension preview: header only.
    let empty = AttributeValue::Preview(Preview {
        width: 0,
        height: 0,
        rgba: Vec::new(),
    });
    let (_, bytes) = encode_attribute_value(&empty);
    assert_eq!(bytes.len(), 8);
    assert_eq!(roundtrip(&empty), empty);
}

#[test]
fn floatvector_algebraic_roundtrip_and_layout() {
    let cases: Vec<Vec<f32>> = vec![
        vec![],
        vec![1.0],
        vec![1.0, 2.5, -3.75, f32::MIN, f32::MAX, f32::MIN_POSITIVE],
        vec![f32::INFINITY, f32::NEG_INFINITY, -0.0],
    ];
    for values in cases {
        let v = AttributeValue::FloatVector(values.clone());
        let (ty, bytes) = encode_attribute_value(&v);
        assert_eq!(ty, "floatvector");
        assert_eq!(bytes.len(), values.len() * 4);
        for (i, val) in values.iter().enumerate() {
            assert_eq!(&bytes[i * 4..i * 4 + 4], &val.to_le_bytes());
        }
        assert_eq!(roundtrip(&v), v);
    }
    // NaN bit-pattern preservation.
    let nan = f32::from_bits(0x7fc0_1234);
    let v = AttributeValue::FloatVector(vec![nan]);
    match roundtrip(&v) {
        AttributeValue::FloatVector(out) => {
            assert_eq!(out.len(), 1);
            assert_eq!(out[0].to_bits(), 0x7fc0_1234);
        }
        other => panic!("wrong variant {other:?}"),
    }
}

#[test]
fn deep_image_state_algebraic_roundtrip_and_layout() {
    for b in [0u8, 1, 2, 3, 200] {
        let v = AttributeValue::DeepImageState(b);
        let (ty, bytes) = encode_attribute_value(&v);
        assert_eq!(ty, "deepImageState");
        assert_eq!(bytes, vec![b]);
        assert_eq!(roundtrip(&v), v);
    }
}

// ---------------------------------------------------------------------
// Malformed-size rejection.
// ---------------------------------------------------------------------

#[test]
fn malformed_sizes_rejected() {
    // envmap: exactly 1 byte.
    assert!(parse_attribute_value("envmap", &[]).is_err());
    assert!(parse_attribute_value("envmap", &[0, 0]).is_err());
    // deepImageState: exactly 1 byte.
    assert!(parse_attribute_value("deepImageState", &[]).is_err());
    assert!(parse_attribute_value("deepImageState", &[1, 0, 0, 0]).is_err());
    // floatvector: multiple of 4.
    assert!(parse_attribute_value("floatvector", &[1, 2, 3]).is_err());
    assert!(parse_attribute_value("floatvector", &[1]).is_err());
    // preview: at least the 8-byte dimension header...
    assert!(parse_attribute_value("preview", &[2, 0, 0, 0]).is_err());
    // ...and exactly 8 + 4·w·h pixel bytes.
    let mut short = Vec::new();
    short.extend_from_slice(&2u32.to_le_bytes());
    short.extend_from_slice(&2u32.to_le_bytes());
    short.extend_from_slice(&[0u8; 15]); // needs 16
    assert!(parse_attribute_value("preview", &short).is_err());
    let mut long = Vec::new();
    long.extend_from_slice(&2u32.to_le_bytes());
    long.extend_from_slice(&2u32.to_le_bytes());
    long.extend_from_slice(&[0u8; 17]);
    assert!(parse_attribute_value("preview", &long).is_err());
    // Hostile dimensions whose 4·w·h overflows 32 bits must error, not
    // wrap into a small expected size.
    let mut hostile = Vec::new();
    hostile.extend_from_slice(&u32::MAX.to_le_bytes());
    hostile.extend_from_slice(&u32::MAX.to_le_bytes());
    hostile.extend_from_slice(&[0u8; 16]);
    assert!(parse_attribute_value("preview", &hostile).is_err());
}

// ---------------------------------------------------------------------
// Full-file round-trip through encode_exr_scanline + parse_exr.
// ---------------------------------------------------------------------

fn encode_with(extra: Vec<Attribute>) -> Vec<u8> {
    let w = 4u32;
    let h = 4u32;
    let chs = vec![Channel {
        name: "G".to_string(),
        pixel_type: PixelType::Half,
        p_linear: false,
        x_sampling: 1,
        y_sampling: 1,
    }];
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (w - 1) as i32,
        y_max: (h - 1) as i32,
    };
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
    let g: Vec<f32> = vec![0.5; (w * h) as usize];
    let planes: Vec<&[f32]> = vec![g.as_slice()];
    encode_exr_scanline(w, h, &chs, &planes, Compression::None, attrs).unwrap()
}

fn find<'a>(img: &'a oxideav_openexr::ExrImage, name: &str) -> &'a AttributeValue {
    &img.attributes
        .iter()
        .find(|a| a.name == name)
        .unwrap_or_else(|| panic!("attribute {name} missing"))
        .value
}

#[test]
fn full_file_roundtrip_all_four_types() {
    let preview = sample_preview();
    let bytes = encode_with(vec![
        Attribute {
            name: "envmap".to_string(),
            value: AttributeValue::EnvMap(EnvMap::Cube),
        },
        Attribute {
            name: "preview".to_string(),
            value: AttributeValue::Preview(preview.clone()),
        },
        Attribute {
            name: "sampleWeights".to_string(),
            value: AttributeValue::FloatVector(vec![0.25, 0.5, 0.25]),
        },
        Attribute {
            name: "deepImageState".to_string(),
            value: AttributeValue::DeepImageState(2),
        },
    ]);
    let img = parse_exr(&bytes).unwrap();
    assert_eq!(find(&img, "envmap"), &AttributeValue::EnvMap(EnvMap::Cube));
    assert_eq!(find(&img, "preview"), &AttributeValue::Preview(preview));
    assert_eq!(
        find(&img, "sampleWeights"),
        &AttributeValue::FloatVector(vec![0.25, 0.5, 0.25])
    );
    assert_eq!(
        find(&img, "deepImageState"),
        &AttributeValue::DeepImageState(2)
    );
}

// ---------------------------------------------------------------------
// exrheader interop (opaque process; auto-skip when absent).
// ---------------------------------------------------------------------

#[test]
fn exrheader_renders_envmap_and_preview() {
    if !exrheader_available() {
        eprintln!("exrheader not available, skipping");
        return;
    }
    let cases: Vec<(Vec<Attribute>, Vec<&str>)> = vec![
        (
            vec![Attribute {
                name: "envmap".to_string(),
                value: AttributeValue::EnvMap(EnvMap::LatLong),
            }],
            vec!["envmap", "latitude-longitude map"],
        ),
        (
            vec![Attribute {
                name: "envmap".to_string(),
                value: AttributeValue::EnvMap(EnvMap::Cube),
            }],
            vec!["envmap", "cube-face map"],
        ),
        (
            vec![Attribute {
                name: "preview".to_string(),
                value: AttributeValue::Preview(Preview {
                    width: 5,
                    height: 7,
                    rgba: vec![0x40; 4 * 5 * 7],
                }),
            }],
            vec!["preview", "5 by 7 pixels"],
        ),
        (
            vec![
                Attribute {
                    name: "sampleWeights".to_string(),
                    value: AttributeValue::FloatVector(vec![1.0, 2.0]),
                },
                Attribute {
                    name: "deepImageState".to_string(),
                    value: AttributeValue::DeepImageState(1),
                },
            ],
            vec![
                "sampleWeights (type floatvector)",
                "deepImageState (type deepImageState)",
            ],
        ),
    ];
    for (attrs, expects) in cases {
        let bytes = encode_with(attrs);
        let dir = std::env::temp_dir();
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = dir.join(format!("oxideav-openexr-attr-{nanos}.exr"));
        std::fs::write(&path, &bytes).unwrap();
        let out = Command::new("exrheader")
            .arg(&path)
            .output()
            .expect("exrheader spawn");
        assert!(
            out.status.success(),
            "exrheader rejected the file:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        for e in expects {
            assert!(text.contains(e), "expected {e:?} in:\n{text}");
        }
        let _ = std::fs::remove_file(&path);
    }
}
