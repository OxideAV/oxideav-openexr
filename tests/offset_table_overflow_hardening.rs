//! Untrusted-input hardening: a hostile line/tile offset-table entry
//! must produce a clean `Err`, never an arithmetic-overflow panic.
//!
//! Each EXR data chunk is located through an offset table whose entries
//! are absolute `u64` byte positions read straight off the wire. A
//! malicious file can set an entry to a value near `u64::MAX`; the
//! decoder casts it to `usize` and then bounds-checks the chunk header
//! (`off + 8` for scanline blocks, `off + 20` for tiles). Computing
//! `off + N` before the comparison overflows `usize` for a near-max
//! offset — a panic in debug builds, a wrap into an out-of-bounds slice
//! (also a panic) in release. The decoder must reject such offsets with
//! an error instead.
//!
//! These tests build a *valid* file with the crate's own encoder, then
//! corrupt the first offset-table entry, and assert the parser returns
//! `Err` without unwinding. Each test would panic (test failure) against
//! the pre-hardening decoder.

use oxideav_openexr::{
    encode_exr_scanline_rgba_float_with, encode_exr_tiled_rgba_float_mipmap_box_filter,
    encode_exr_tiled_rgba_float_with, parse_exr, parse_exr_tiled_multilevel, parse_header,
    Compression,
};

/// Overwrite the 8-byte offset-table entry at `table_index` (entry 0 is
/// the first entry, located immediately after the header's terminating
/// NUL) with `value`.
fn corrupt_offset_entry(bytes: &mut [u8], table_index: usize, value: u64) {
    let header = parse_header(bytes).expect("encoder must emit a parseable header");
    let entry_pos = header.end_offset + table_index * 8;
    bytes[entry_pos..entry_pos + 8].copy_from_slice(&value.to_le_bytes());
}

fn ramp(w: u32, h: u32) -> Vec<f32> {
    (0..(w * h * 4) as usize)
        .map(|i| (i as f32) * 0.0125)
        .collect()
}

#[test]
fn scanline_hostile_offset_entry_is_rejected_not_panicking() {
    let (w, h) = (8u32, 8u32);
    let samples = ramp(w, h);
    // NONE compression => one block per scanline => offset table has `h`
    // entries directly after the header.
    let mut bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::None).unwrap();

    // u64::MAX would overflow `block_off + 8` in the bounds check.
    corrupt_offset_entry(&mut bytes, 0, u64::MAX);
    let res = parse_exr(&bytes);
    assert!(
        res.is_err(),
        "hostile near-max scanline offset must yield Err, got Ok"
    );

    // A value that is in-range as a usize but whose `+8` still lands past
    // EOF must also be rejected (the ordinary past-EOF path), proving the
    // overflow guard didn't change the non-overflow rejection behaviour.
    let mut bytes2 =
        encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::None).unwrap();
    let just_past = bytes2.len() as u64;
    corrupt_offset_entry(&mut bytes2, 0, just_past);
    assert!(parse_exr(&bytes2).is_err());
}

#[test]
fn scanline_offset_overflowing_payload_size_is_rejected() {
    let (w, h) = (8u32, 8u32);
    let samples = ramp(w, h);
    let mut bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::None).unwrap();
    // Point the first block at the very end of the file so its 8-byte
    // header parses, but the (attacker-uncontrolled) payload_size read
    // there is whatever bytes happen to sit at EOF-8; the parser must not
    // overflow `payload_start + payload_size`. Use an offset that leaves
    // exactly 8 bytes so the header is readable.
    let near_end = (bytes.len() - 8) as u64;
    corrupt_offset_entry(&mut bytes, 0, near_end);
    // Either Ok-with-fewer-pixels is impossible here (size mismatch) so it
    // must be Err; the point is it does not panic.
    assert!(parse_exr(&bytes).is_err());
}

#[test]
fn tiled_hostile_offset_entry_is_rejected_not_panicking() {
    let (w, h) = (16u32, 16u32);
    let samples = ramp(w, h);
    let mut bytes =
        encode_exr_tiled_rgba_float_with(w, h, &samples, Compression::None, 8, 8).unwrap();

    corrupt_offset_entry(&mut bytes, 0, u64::MAX);
    let res = parse_exr(&bytes);
    assert!(
        res.is_err(),
        "hostile near-max tile offset must yield Err, got Ok"
    );
}

#[test]
fn multilevel_tiled_hostile_offset_entry_is_rejected_not_panicking() {
    let (w, h) = (16u32, 16u32);
    let samples = ramp(w, h);
    let mut bytes =
        encode_exr_tiled_rgba_float_mipmap_box_filter(w, h, &samples, Compression::None, 8, 8)
            .unwrap();

    corrupt_offset_entry(&mut bytes, 0, u64::MAX);
    let res = parse_exr_tiled_multilevel(&bytes);
    assert!(
        res.is_err(),
        "hostile near-max multilevel-tile offset must yield Err, got Ok"
    );
}

#[test]
fn well_formed_files_still_decode_after_hardening() {
    // Regression guard: the checked-arithmetic changes must not break the
    // happy path. Decode each layout untouched.
    let (w, h) = (8u32, 8u32);
    let samples = ramp(w, h);

    let scan = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::None).unwrap();
    let img = parse_exr(&scan).unwrap();
    assert_eq!(img.width(), w);
    assert_eq!(img.height(), h);

    let (tw, th) = (16u32, 16u32);
    let tsamples = ramp(tw, th);
    let tiled =
        encode_exr_tiled_rgba_float_with(tw, th, &tsamples, Compression::None, 8, 8).unwrap();
    let timg = parse_exr(&tiled).unwrap();
    assert_eq!(timg.width(), tw);
    assert_eq!(timg.height(), th);

    let mip =
        encode_exr_tiled_rgba_float_mipmap_box_filter(tw, th, &tsamples, Compression::None, 8, 8)
            .unwrap();
    let mimg = parse_exr_tiled_multilevel(&mip).unwrap();
    assert!(!mimg.levels.is_empty());
}
