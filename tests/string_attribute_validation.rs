//! Black-box validation that the typed `AttributeValue::String` variant
//! parses, encodes, and re-parses through `exrheader` identically to the
//! legacy `AttributeValue::Other { type_name: "string", .. }` shape.
//!
//! The motivating use is multi-part EXR, where every part carries a
//! required `type` (one of `scanlineimage` / `tiledimage` /
//! `deepscanline` / `deeptile`) and a required `name` — both encoded as
//! `string`-typed attributes. The encoders in `src/` still construct
//! these as `Other { type_name: "string", data: utf-8 }`; the typed
//! parser path lifts them back to `String(...)` on read. This test
//! confirms the round-trip is observably correct against the external
//! `exrheader` oracle (binary used as opaque process — no source
//! consulted).
//!
//! The test is auto-skipped when `exrheader` is missing from `$PATH`
//! (mirrors the policy in `tests/exrheader_validation.rs`).

use std::process::Command;

use oxideav_openexr::{
    encode_exr_scanline_rgba_float_with, parse_exr, AttributeValue, Compression,
};

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
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-string-attr-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

/// Pure-Rust round-trip — independent of any external binary. Confirms
/// that an EXR encoded by this crate (which still produces string
/// attributes via the legacy `Other { "string", utf-8 }` shape) is
/// parsed back into the typed `AttributeValue::String(...)` variant.
#[test]
fn parsed_back_as_typed_string_after_self_roundtrip() {
    let w = 4;
    let h = 4;
    let samples: Vec<f32> = (0..(w * h * 4)).map(|i| (i as f32) * 0.05).collect();
    let bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::None).unwrap();

    // parse_exr only returns the level-0 image, but the header is also
    // parsed via parse_header internally — we read the full header
    // separately for attribute inspection.
    let _img = parse_exr(&bytes).expect("parse_exr must succeed");
    let header =
        oxideav_openexr::parse_header(&bytes).expect("parse_header must succeed on our own output");

    // Single-part scanline encoder doesn't emit a `name` / `type`
    // attribute (those are only required on multi-part files), so we
    // only assert the typed path works when a string-typed attribute
    // IS present. Inject one via a manual header round-trip:
    let mut attrs = header.attributes.clone();
    attrs.push(oxideav_openexr::Attribute {
        name: "comments".into(),
        value: AttributeValue::String("round 238 typed-string attribute landing".to_string()),
    });
    let raw = oxideav_openexr::encode_header(oxideav_openexr::VersionField::from_u32(2), &attrs);
    let reparsed = oxideav_openexr::parse_header(&raw).unwrap();

    // The injected attribute must come back as the typed variant.
    let injected = reparsed
        .attributes
        .iter()
        .find(|a| a.name == "comments")
        .expect("injected comments attribute must survive round-trip");
    match &injected.value {
        AttributeValue::String(s) => {
            assert_eq!(s, "round 238 typed-string attribute landing");
        }
        other => panic!("expected typed String variant, got {other:?}"),
    }
}

/// Cross-validate through `exrheader` (used as an opaque binary
/// oracle): a hand-crafted header carrying a typed-String `comments`
/// attribute must be readable by exrheader and the string must appear
/// in its dump.
#[test]
fn exrheader_reads_typed_string_attribute() {
    if !exrheader_available() {
        eprintln!("exrheader not available on PATH, skipping validation");
        return;
    }
    let w = 4;
    let h = 4;
    let samples: Vec<f32> = (0..(w * h * 4)).map(|i| (i as f32) * 0.05).collect();
    let bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::None).unwrap();

    // Splice a typed-String `comments` attribute into the parsed
    // header, then re-encode the prefix + the original scanline body.
    let header = oxideav_openexr::parse_header(&bytes).unwrap();
    let body = &bytes[header.end_offset..];
    let needle = "oxideav-openexr r238 typed string check";
    let mut attrs = header.attributes.clone();
    attrs.push(oxideav_openexr::Attribute {
        name: "comments".into(),
        value: AttributeValue::String(needle.to_string()),
    });
    let new_header =
        oxideav_openexr::encode_header(oxideav_openexr::VersionField::from_u32(2), &attrs);

    // Rewrite the line-offset table: every entry shifts by the
    // (new_header_len - old_header_len) delta. The single-part
    // scanline offset table is `chunk_count * u64 LE`, immediately
    // following the header. chunk_count is derived from the parsed
    // dataWindow + compression (1 line per block for NONE).
    let delta = new_header.len() as i64 - header.end_offset as i64;
    let chunk_count = h as usize; // NONE compression -> 1 scanline per block
    let mut spliced = Vec::with_capacity(new_header.len() + body.len());
    spliced.extend_from_slice(&new_header);
    for i in 0..chunk_count {
        let off = i * 8;
        let old = u64::from_le_bytes(body[off..off + 8].try_into().unwrap());
        let new = (old as i64 + delta) as u64;
        spliced.extend_from_slice(&new.to_le_bytes());
    }
    spliced.extend_from_slice(&body[chunk_count * 8..]);

    // Sanity: our own parser must accept the spliced file (round-trip
    // through the level-0 decoder).
    let _img = parse_exr(&spliced).expect("spliced file must self-parse");

    // Now hand the bytes to exrheader and verify the string appears.
    let dir = tempdir();
    let path = format!("{dir}/oxideav-openexr-string-attr.exr");
    std::fs::write(&path, &spliced).unwrap();
    let output = Command::new("exrheader")
        .arg(&path)
        .output()
        .expect("exrheader spawn failed");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "exrheader returned non-zero on our spliced output\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains(needle),
        "exrheader output did not contain our typed-string attribute value {needle:?}\nstdout: {stdout}"
    );

    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}
