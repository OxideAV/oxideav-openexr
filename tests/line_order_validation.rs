//! `lineOrder` conformance — DECREASING_Y (and, for tiled files,
//! RANDOM_Y) storage-order support, validated against the reference
//! reader binaries as opaque black boxes.
//!
//! Observer-established wire facts pinned by this suite (see
//! `tests/line_order_observer_notes.md` for the derivation record):
//!
//! 1. The chunk offset table is ALWAYS keyed in canonical top-first
//!    order — entry `i` points at the scanline block whose first row is
//!    `y_min + i * blockHeight` (or at tile `i` of the canonical
//!    ty-outer/tx-inner walk) — regardless of `lineOrder`. A table
//!    whose entry order follows the decreasing physical storage order
//!    is rejected by the reference reader with a chunk-leader error.
//! 2. `lineOrder` governs only the physical storage/streaming order of
//!    the chunks inside the file.
//! 3. RANDOM_Y is invalid for scanline images (the reference readers
//!    refuse to open such headers) and valid for tiled images.
//!
//! Every reference-binary test auto-skips with a printed reason when
//! the tool is absent so CI hosts without an OpenEXR install stay
//! green.

use std::process::Command;

use oxideav_openexr::{
    encode_exr_scanline_rgba_float_with, encode_exr_scanline_rgba_float_with_line_order,
    encode_exr_tiled_rgba_float_with, encode_exr_tiled_rgba_float_with_line_order, parse_exr,
    Compression, LineOrder,
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
    let dir = std::env::temp_dir().join(format!("oxideav-openexr-lineorder-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

/// 32×40 RGBA gradient — 3 chunks under ZIP (16 lines/block), 2 chunks
/// under B44 (32 lines/block), 40 chunks under ZIPS.
fn gradient(width: u32, height: u32) -> Vec<f32> {
    let mut px = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            px.push(x as f32 / (width - 1) as f32);
            px.push(y as f32 / (height - 1) as f32);
            px.push((x + y) as f32 / (width + height - 2) as f32);
            px.push(1.0);
        }
    }
    px
}

fn assert_planes_match(bytes: &[u8], expected: &[f32], width: u32, height: u32, tol: f32) {
    let img = parse_exr(bytes).unwrap();
    assert_eq!(img.width(), width);
    assert_eq!(img.height(), height);
    // Planes are alphabetical: A, B, G, R. Interleaved input is RGBA.
    let order = [3usize, 2, 1, 0]; // channel index in the RGBA pixel for A,B,G,R
    for (plane, &comp) in img.planes.iter().zip(order.iter()) {
        for (i, &got) in plane.samples.iter().enumerate() {
            let want = expected[i * 4 + comp];
            assert!(
                (got - want).abs() <= tol,
                "plane {} sample {i}: got {got}, want {want} (tol {tol})",
                plane.name
            );
        }
    }
}

// ---------------------------------------------------------------------
// Wire-layout pinning (binary-independent)
// ---------------------------------------------------------------------

/// Walk the header attribute table of a single-part file and return
/// (offset of the lineOrder value byte, end-of-header position).
fn find_line_order_and_header_end(bytes: &[u8]) -> (usize, usize) {
    let mut pos = 8; // magic + version
    let mut lo_off = None;
    while bytes[pos] != 0 {
        let nend = pos + bytes[pos..].iter().position(|&b| b == 0).unwrap();
        let name = &bytes[pos..nend];
        pos = nend + 1;
        let tend = pos + bytes[pos..].iter().position(|&b| b == 0).unwrap();
        pos = tend + 1;
        let size = i32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if name == b"lineOrder" {
            lo_off = Some(pos);
        }
        pos += size;
    }
    (lo_off.expect("no lineOrder attribute"), pos + 1)
}

/// DECREASING_Y scanline: the offset table must stay keyed top-first
/// (entry i ↔ block starting at y = i·blockHeight) while the physical
/// chunk storage runs bottom-first.
#[test]
fn decreasing_y_scanline_wire_layout() {
    let (w, h) = (32u32, 40u32);
    let px = gradient(w, h);
    let bytes = encode_exr_scanline_rgba_float_with_line_order(
        w,
        h,
        &px,
        Compression::Zip,
        LineOrder::DecreasingY,
    )
    .unwrap();

    let (lo_off, header_end) = find_line_order_and_header_end(&bytes);
    assert_eq!(bytes[lo_off], 1, "lineOrder attribute value must be 1");

    // 3 ZIP blocks: y = 0, 16, 32. Read the offset table.
    let mut offsets = Vec::new();
    for i in 0..3 {
        let p = header_end + i * 8;
        offsets.push(u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap()) as usize);
    }
    // Each table entry must point at the block with the canonical Y for
    // its index (top-first keying)...
    for (i, &off) in offsets.iter().enumerate() {
        let y = i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        assert_eq!(y, (i as i32) * 16, "table entry {i} keyed to wrong block");
    }
    // ...while the physical storage runs bottom-first: the first stored
    // chunk (right after the table) is the bottom block, so its file
    // offset is the SMALLEST and belongs to the LAST table entry.
    assert!(
        offsets[2] < offsets[1] && offsets[1] < offsets[0],
        "DECREASING_Y storage must be bottom-first (offsets {offsets:?})"
    );
    let first_chunk = header_end + 3 * 8;
    assert_eq!(offsets[2], first_chunk, "bottom block must be stored first");

    // Our own reader decodes it pixel-exactly (offset-table driven).
    assert_planes_match(&bytes, &px, w, h, 0.0);
}

/// IncreasingY output is byte-identical whether requested through the
/// default entry point or the explicit line-order one.
#[test]
fn increasing_y_explicit_matches_default() {
    let (w, h) = (32u32, 40u32);
    let px = gradient(w, h);
    let a = encode_exr_scanline_rgba_float_with(w, h, &px, Compression::Rle).unwrap();
    let b = encode_exr_scanline_rgba_float_with_line_order(
        w,
        h,
        &px,
        Compression::Rle,
        LineOrder::IncreasingY,
    )
    .unwrap();
    assert_eq!(a, b);
}

/// RANDOM_Y is invalid for scanline images: the writer must refuse.
#[test]
fn random_y_scanline_rejected_on_encode() {
    let (w, h) = (8u32, 8u32);
    let px = gradient(w, h);
    let err = encode_exr_scanline_rgba_float_with_line_order(
        w,
        h,
        &px,
        Compression::None,
        LineOrder::RandomY,
    )
    .unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("RANDOM_Y"),
        "error should name RANDOM_Y: {msg}"
    );
}

/// Reader leniency pin: a scanline file whose lineOrder byte was
/// corrupted to RANDOM_Y still decodes through our offset-table-driven
/// scatter (the reference readers reject such headers outright; we
/// accept, since each chunk self-describes its Y).
#[test]
fn random_y_scanline_read_leniency() {
    let (w, h) = (32u32, 40u32);
    let px = gradient(w, h);
    let mut bytes = encode_exr_scanline_rgba_float_with(w, h, &px, Compression::Zip).unwrap();
    let (lo_off, _) = find_line_order_and_header_end(&bytes);
    bytes[lo_off] = 2; // RANDOM_Y
    assert_planes_match(&bytes, &px, w, h, 0.0);
}

/// Decode-side pin: DECREASING_Y round-trips pixel-exactly through our
/// parser for every compression scheme we emit, at sizes that exercise
/// partial final blocks.
#[test]
fn decreasing_y_all_compressions_roundtrip() {
    for (comp, tol) in [
        (Compression::None, 0.0),
        (Compression::Zip, 0.0),
        (Compression::Zips, 0.0),
        (Compression::Rle, 0.0),
        // PXR24 rounds FLOAT mantissas to 15 bits (lossy).
        (Compression::Pxr24, 1.0 / 32768.0),
    ] {
        for (w, h) in [(32u32, 40u32), (7u32, 33u32), (16u32, 16u32)] {
            let px = gradient(w, h);
            let bytes = encode_exr_scanline_rgba_float_with_line_order(
                w,
                h,
                &px,
                comp,
                LineOrder::DecreasingY,
            )
            .unwrap();
            assert_planes_match(&bytes, &px, w, h, tol);
        }
    }
    // B44/B44A quantise HALF; our RGBA-float convenience writer emits
    // FLOAT channels, which B44 copies raw — still exact.
    for comp in [Compression::B44, Compression::B44a] {
        let (w, h) = (32u32, 40u32);
        let px = gradient(w, h);
        let bytes =
            encode_exr_scanline_rgba_float_with_line_order(w, h, &px, comp, LineOrder::DecreasingY)
                .unwrap();
        assert_planes_match(&bytes, &px, w, h, 0.0);
    }
}

// ---------------------------------------------------------------------
// Tiled (ONE_LEVEL) — DECREASING_Y and RANDOM_Y are both valid
// ---------------------------------------------------------------------

/// Read the tile offset table + per-entry chunk coordinates of a
/// single-part tiled file. Returns (offset, tx, ty) per table entry.
fn tiled_table_entries(bytes: &[u8], n: usize) -> Vec<(usize, i32, i32)> {
    let (_, header_end) = find_line_order_and_header_end(bytes);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let p = header_end + i * 8;
        let off = u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap()) as usize;
        let tx = i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
        let ty = i32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
        out.push((off, tx, ty));
    }
    out
}

/// DECREASING_Y tiled: table entries stay keyed to the canonical
/// ty-outer/tx-inner walk while physical storage runs bottom-row-first.
#[test]
fn decreasing_y_tiled_wire_layout() {
    let (w, h) = (32u32, 40u32); // 16×16 tiles → 2×3 grid, 6 chunks
    let px = gradient(w, h);
    let bytes = encode_exr_tiled_rgba_float_with_line_order(
        w,
        h,
        &px,
        Compression::Zip,
        16,
        16,
        LineOrder::DecreasingY,
    )
    .unwrap();

    let (lo_off, _) = find_line_order_and_header_end(&bytes);
    assert_eq!(bytes[lo_off], 1, "lineOrder attribute value must be 1");

    let entries = tiled_table_entries(&bytes, 6);
    // Canonical keying: entry i ↔ tile (i % 2, i / 2).
    for (i, &(_, tx, ty)) in entries.iter().enumerate() {
        assert_eq!((tx, ty), ((i % 2) as i32, (i / 2) as i32));
    }
    // Bottom tile row (ty=2, entries 4 and 5) must be stored first.
    let min_off = entries.iter().map(|e| e.0).min().unwrap();
    assert!(
        entries[4].0 == min_off || entries[5].0 == min_off,
        "bottom tile row must be stored first (offsets {:?})",
        entries.iter().map(|e| e.0).collect::<Vec<_>>()
    );

    assert_planes_match(&bytes, &px, w, h, 0.0);
}

/// RANDOM_Y tiled: valid layout, deterministic output, decodes exactly.
#[test]
fn random_y_tiled_roundtrip_and_deterministic() {
    let (w, h) = (32u32, 40u32);
    let px = gradient(w, h);
    let enc = || {
        encode_exr_tiled_rgba_float_with_line_order(
            w,
            h,
            &px,
            Compression::Rle,
            16,
            16,
            LineOrder::RandomY,
        )
        .unwrap()
    };
    let bytes = enc();
    assert_eq!(bytes, enc(), "RANDOM_Y output must be deterministic");

    let (lo_off, _) = find_line_order_and_header_end(&bytes);
    assert_eq!(bytes[lo_off], 2, "lineOrder attribute value must be 2");

    // Table still canonically keyed.
    let entries = tiled_table_entries(&bytes, 6);
    for (i, &(_, tx, ty)) in entries.iter().enumerate() {
        assert_eq!((tx, ty), ((i % 2) as i32, (i / 2) as i32));
    }
    // Storage must actually be shuffled (not canonical order).
    let offs: Vec<usize> = entries.iter().map(|e| e.0).collect();
    assert!(
        offs.windows(2).any(|w| w[1] < w[0]),
        "RANDOM_Y storage unexpectedly canonical: {offs:?}"
    );

    assert_planes_match(&bytes, &px, w, h, 0.0);
}

/// IncreasingY tiled output is byte-identical through both entry points.
#[test]
fn increasing_y_tiled_explicit_matches_default() {
    let (w, h) = (32u32, 40u32);
    let px = gradient(w, h);
    let a = encode_exr_tiled_rgba_float_with(w, h, &px, Compression::Zip, 16, 16).unwrap();
    let b = encode_exr_tiled_rgba_float_with_line_order(
        w,
        h,
        &px,
        Compression::Zip,
        16,
        16,
        LineOrder::IncreasingY,
    )
    .unwrap();
    assert_eq!(a, b);
}

/// Tiled DECREASING_Y + RANDOM_Y round-trip across every compression
/// scheme, including edge tiles (13×9 tiles over 32×40).
#[test]
fn tiled_line_orders_all_compressions_roundtrip() {
    for lo in [LineOrder::DecreasingY, LineOrder::RandomY] {
        for (comp, tol) in [
            (Compression::None, 0.0),
            (Compression::Zip, 0.0),
            (Compression::Zips, 0.0),
            (Compression::Rle, 0.0),
            (Compression::Pxr24, 1.0 / 32768.0),
            (Compression::B44, 0.0),
            (Compression::B44a, 0.0),
        ] {
            let (w, h) = (32u32, 40u32);
            let px = gradient(w, h);
            let bytes =
                encode_exr_tiled_rgba_float_with_line_order(w, h, &px, comp, 13, 9, lo).unwrap();
            assert_planes_match(&bytes, &px, w, h, tol);
        }
    }
}

// ---------------------------------------------------------------------
// Reference-binary validation (auto-skip when absent)
// ---------------------------------------------------------------------

/// `exrheader` must echo "decreasing y"; `exrinfo` (the independent
/// second reader) must accept the file; `exrmetrics --convert -z none`
/// must re-encode it, and the converted file must decode pixel-exactly.
#[test]
fn decreasing_y_scanline_reference_validated() {
    let (w, h) = (32u32, 40u32);
    let px = gradient(w, h);
    let bytes = encode_exr_scanline_rgba_float_with_line_order(
        w,
        h,
        &px,
        Compression::Zip,
        LineOrder::DecreasingY,
    )
    .unwrap();

    let dir = tempdir();
    let path = format!("{dir}/dec.exr");
    std::fs::write(&path, &bytes).unwrap();

    if tool_available("exrheader") {
        let out = Command::new("exrheader").arg(&path).output().unwrap();
        assert!(out.status.success(), "exrheader rejected DECREASING_Y file");
        let text = String::from_utf8_lossy(&out.stdout);
        assert!(
            text.contains("decreasing y"),
            "exrheader did not echo decreasing y:\n{text}"
        );
    } else {
        eprintln!("exrheader not available, skipping header echo check");
    }

    if tool_available("exrinfo") {
        let out = Command::new("exrinfo").arg(&path).output().unwrap();
        assert!(
            out.status.success(),
            "exrinfo rejected DECREASING_Y file: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    } else {
        eprintln!("exrinfo not available, skipping");
    }

    if tool_available("exrmetrics") {
        let conv = format!("{dir}/conv.exr");
        let out = Command::new("exrmetrics")
            .args(["--convert", "-z", "none", "-o", &conv, &path])
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "exrmetrics --convert rejected DECREASING_Y file: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let converted = std::fs::read(&conv).unwrap();
        assert_planes_match(&converted, &px, w, h, 0.0);
    } else {
        eprintln!("exrmetrics not available, skipping convert check");
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Tiled DECREASING_Y and RANDOM_Y files must be accepted by the
/// reference readers, echo the right lineOrder, and convert
/// pixel-exactly.
#[test]
fn tiled_line_orders_reference_validated() {
    let (w, h) = (32u32, 40u32);
    let px = gradient(w, h);
    for (lo, echo) in [
        (LineOrder::DecreasingY, "decreasing y"),
        (LineOrder::RandomY, "random y"),
    ] {
        let bytes =
            encode_exr_tiled_rgba_float_with_line_order(w, h, &px, Compression::Zip, 16, 16, lo)
                .unwrap();
        let dir = tempdir();
        let path = format!("{dir}/tiled.exr");
        std::fs::write(&path, &bytes).unwrap();

        if tool_available("exrheader") {
            let out = Command::new("exrheader").arg(&path).output().unwrap();
            assert!(out.status.success(), "exrheader rejected {echo} tiled file");
            let text = String::from_utf8_lossy(&out.stdout);
            assert!(
                text.contains(echo),
                "exrheader did not echo {echo}:\n{text}"
            );
        } else {
            eprintln!("exrheader not available, skipping ({echo})");
        }

        if tool_available("exrinfo") {
            let out = Command::new("exrinfo").arg(&path).output().unwrap();
            assert!(
                out.status.success(),
                "exrinfo rejected {echo} tiled file: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        } else {
            eprintln!("exrinfo not available, skipping ({echo})");
        }

        if tool_available("exrmetrics") {
            let conv = format!("{dir}/conv.exr");
            let out = Command::new("exrmetrics")
                .args(["--convert", "-z", "none", "-o", &conv, &path])
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "exrmetrics --convert rejected {echo} tiled file: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            let converted = std::fs::read(&conv).unwrap();
            assert_planes_match(&converted, &px, w, h, 0.0);
        } else {
            eprintln!("exrmetrics not available, skipping ({echo})");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
