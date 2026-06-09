//! Round-265 — typed `tiledesc` attribute inspector round-trip + reference-
//! binary interop.
//!
//! `tiledesc` is the 9-byte payload that the OpenEXR `tiles` attribute
//! carries on every tiled file. Layout on disk:
//!
//! ```text
//! u32 x_size          ; little-endian, > 0
//! u32 y_size          ; little-endian, > 0
//! u8  mode            ; high nibble = round mode (0=DOWN, 1=UP),
//!                       low  nibble = level mode (0=ONE_LEVEL,
//!                                                 1=MIPMAP_LEVELS,
//!                                                 2=RIPMAP_LEVELS)
//! ```
//!
//! Until this round the `parse_attribute_value` fall-through returned the
//! 9-byte payload as `AttributeValue::Other { type_name: "tiledesc", data
//! }`. The `tiledesc_from_attribute` helper inspected the bytes inline,
//! so reads worked, but the type was opaque on the public API surface.
//!
//! This round adds the typed [`AttributeValue::TileDesc(TileDesc)`]
//! variant alongside the round-238 / round-247 typed inspectors for
//! `int` / `double` / `string` / `v2i` / `v3i` / `v3f` / `m33f` / `m44f`
//! / `chromaticities` / `box2f`. Existing call sites that consumed
//! `AttributeValue::Other { type_name: "tiledesc", .. }` (the multi-part
//! / deep-tiled `tiles` lookups in `deep.rs`) accept BOTH the typed
//! variant AND the legacy `Other` shape, so the encoder sites that still
//! emit `Other` (round-40 + round-78 + round-124 + round-130 + round-181
//! + round-192 + round-196 + round-202 + round-208 + round-214 + round-220
//! + round-227 + round-232) keep working unchanged. The reader now
//!   returns the typed variant on every file.
//!
//! Three layers of coverage, matching the round-238
//! `typed_attribute_roundtrip.rs` + round-247 `box2f_attribute_roundtrip.rs`
//! layout:
//!
//! 1. **Algebraic round-trip** — build an
//!    [`AttributeValue::TileDesc`], encode through
//!    [`encode_attribute_value`], re-parse through
//!    [`parse_attribute_value`], assert bit-exact equality across every
//!    (level_mode, round_mode) combination the format permits plus
//!    representative `x_size` / `y_size` values including 1 (minimum
//!    permitted size) and large powers of two.
//!
//! 2. **Full-file round-trip** — generate a real tiled file via
//!    [`encode_exr_tiled_rgba_float_with`] and confirm that after
//!    [`parse_header`] the `tiles` attribute surfaces as the new typed
//!    [`AttributeValue::TileDesc`] variant (not falling through to
//!    `Other`). Decoder side (the round-2 tile decoder, the round-78
//!    MIPMAP decoder, the round-124 RIPMAP decoder, plus the deep tiled
//!    variants in `deep.rs`) accepts both forms so the same files keep
//!    decoding without source changes.
//!
//! 3. **`exrheader` interop** — opaque-process invocation of `exrheader`
//!    on a generated file carrying the new typed `tiledesc`; assert zero
//!    exit and that the `tiles` attribute appears in the emitted text.
//!    Auto-skipped when `exrheader` is missing from `$PATH`.

use std::process::Command;

use oxideav_openexr::{
    encode_exr_tiled_rgba_float_with, parse_header, tiled::TileDesc, AttributeValue, Compression,
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
fn tiledesc_payload_size_and_type_name() {
    let v = AttributeValue::TileDesc(TileDesc {
        x_size: 64,
        y_size: 64,
        level_mode: 0,
        round_mode: 0,
    });
    let (type_name, data) = oxideav_openexr::header::encode_attribute_value(&v);
    assert_eq!(type_name, "tiledesc");
    assert_eq!(
        data.len(),
        9,
        "tiledesc payload must be exactly 2 * u32 + 1 byte = 9 bytes"
    );
}

#[test]
fn tiledesc_roundtrip_one_level_round_down() {
    let td = TileDesc {
        x_size: 64,
        y_size: 64,
        level_mode: 0,
        round_mode: 0,
    };
    match rt(AttributeValue::TileDesc(td)) {
        AttributeValue::TileDesc(p) => assert_eq!(p, td),
        other => panic!("expected TileDesc, got {other:?}"),
    }
}

#[test]
fn tiledesc_roundtrip_every_level_and_round_mode() {
    // The format permits level_mode in {0, 1, 2} and round_mode in {0, 1}.
    for lvl in 0u8..=2 {
        for rnd in 0u8..=1 {
            let td = TileDesc {
                x_size: 128,
                y_size: 64,
                level_mode: lvl,
                round_mode: rnd,
            };
            match rt(AttributeValue::TileDesc(td)) {
                AttributeValue::TileDesc(p) => assert_eq!(
                    p, td,
                    "round-trip drift for level_mode={lvl} round_mode={rnd}"
                ),
                other => panic!("expected TileDesc, got {other:?}"),
            }
        }
    }
}

#[test]
fn tiledesc_roundtrip_extreme_sizes() {
    // Minimum permitted size (1×1), large power of two, and an asymmetric
    // non-power-of-two pair to confirm field independence.
    let cases = [(1u32, 1u32), (4096, 4096), (65535, 31), (32, 65535)];
    for (xs, ys) in cases {
        let td = TileDesc {
            x_size: xs,
            y_size: ys,
            level_mode: 0,
            round_mode: 0,
        };
        match rt(AttributeValue::TileDesc(td)) {
            AttributeValue::TileDesc(p) => assert_eq!(p, td),
            other => panic!("expected TileDesc for {xs}×{ys}, got {other:?}"),
        }
    }
}

#[test]
fn tiledesc_byte_layout_matches_spec_bytes() {
    // Pin: x_size = 32 (LE 20 00 00 00), y_size = 16 (LE 10 00 00 00),
    // mode = MIPMAP (1) + ROUND_UP (1) -> 0x11.
    let td = TileDesc {
        x_size: 32,
        y_size: 16,
        level_mode: 1,
        round_mode: 1,
    };
    let (_, data) = oxideav_openexr::header::encode_attribute_value(&AttributeValue::TileDesc(td));
    let mut expected = Vec::with_capacity(9);
    expected.extend_from_slice(&32u32.to_le_bytes());
    expected.extend_from_slice(&16u32.to_le_bytes());
    expected.push(0x11);
    assert_eq!(data, expected);
}

#[test]
fn tiledesc_parses_legacy_other_payload() {
    // The reader still accepts a manually-built 9-byte payload through
    // the public parse_attribute_value entry. This guards the round-2
    // behaviour: prior callers that synthesised a tiledesc by hand and
    // re-parsed it must keep working.
    let mut bytes = Vec::with_capacity(9);
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&8u32.to_le_bytes());
    bytes.push(0x02); // RIPMAP + ROUND_DOWN
    let parsed = oxideav_openexr::header::parse_attribute_value("tiledesc", &bytes).unwrap();
    match parsed {
        AttributeValue::TileDesc(td) => {
            assert_eq!(td.x_size, 16);
            assert_eq!(td.y_size, 8);
            assert_eq!(td.level_mode, 2);
            assert_eq!(td.round_mode, 0);
        }
        other => panic!("expected TileDesc, got {other:?}"),
    }
}

#[test]
fn tiledesc_rejects_short_payload() {
    assert!(oxideav_openexr::header::parse_attribute_value("tiledesc", &[0u8; 8]).is_err());
}

#[test]
fn tiledesc_rejects_oversize_payload() {
    assert!(oxideav_openexr::header::parse_attribute_value("tiledesc", &[0u8; 10]).is_err());
}

// ---------------------------------------------------------------------------
// Layer 2: full-file round-trip through encode_exr_tiled + parse_header.
// ---------------------------------------------------------------------------

fn tiny_tiled_file(compression: Compression) -> Vec<u8> {
    let w = 8u32;
    let h = 8u32;
    let mut samples = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            samples.push((x as f32) * 0.1);
            samples.push((y as f32) * 0.1);
            samples.push(0.5);
            samples.push(1.0);
        }
    }
    encode_exr_tiled_rgba_float_with(w, h, &samples, compression, 4, 4).unwrap()
}

#[test]
fn tiled_file_surfaces_typed_tiledesc() {
    let bytes = tiny_tiled_file(Compression::Zip);
    let header = parse_header(&bytes).unwrap();
    let tiles_attr = header
        .attributes
        .iter()
        .find(|a| a.name == "tiles")
        .expect("tiled file must carry the tiles attribute");
    match &tiles_attr.value {
        AttributeValue::TileDesc(td) => {
            assert_eq!(td.x_size, 4);
            assert_eq!(td.y_size, 4);
            assert_eq!(td.level_mode, 0, "ONE_LEVEL");
            assert_eq!(td.round_mode, 0, "ROUND_DOWN");
        }
        other => {
            panic!("tiles attribute must surface as the typed TileDesc variant, got {other:?}")
        }
    }
}

#[test]
fn tiled_file_with_zips_compression_surfaces_typed_tiledesc() {
    // Repeat under a different compression to confirm the typed variant
    // surfaces independent of the per-tile payload encoding.
    let bytes = tiny_tiled_file(Compression::Zips);
    let header = parse_header(&bytes).unwrap();
    let tiles_attr = header
        .attributes
        .iter()
        .find(|a| a.name == "tiles")
        .unwrap();
    assert!(
        matches!(tiles_attr.value, AttributeValue::TileDesc(_)),
        "expected typed TileDesc on ZIPS tiled file, got {:?}",
        tiles_attr.value
    );
}

// ---------------------------------------------------------------------------
// Layer 3: exrheader interop on a generated tiled file.
// ---------------------------------------------------------------------------

fn exrheader_available() -> bool {
    Command::new("exrheader")
        .arg("--help")
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

#[test]
fn exrheader_accepts_generated_tiled_file_and_mentions_tiles() {
    if !exrheader_available() {
        eprintln!("skipping: `exrheader` not on PATH");
        return;
    }
    let bytes = tiny_tiled_file(Compression::None);
    let tmp = tempfile_path("openexr_r265_tiledesc.exr");
    std::fs::write(&tmp, &bytes).expect("write tmp file");
    let out = Command::new("exrheader")
        .arg(&tmp)
        .output()
        .expect("run exrheader");
    let _ = std::fs::remove_file(&tmp);
    assert!(
        out.status.success(),
        "exrheader exited non-zero: stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("tiles"),
        "exrheader output must mention the `tiles` attribute, stdout was:\n{stdout}"
    );
}

fn tempfile_path(suffix: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    p.push(format!("{nanos}-{suffix}"));
    p
}
