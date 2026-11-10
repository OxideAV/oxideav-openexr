//! The staged multi-part fixtures (`docs/image/openexr/fixtures/multipart-*`,
//! copied under `tests/fixtures/multipart/`), read against the multi-part
//! observer note:
//!
//! * layout facts (version flags, part names / types, chunk counts, table
//!   positions) as the fixture notes list them;
//! * every part decoded exactly — the SHA-1 of the decoded samples equals
//!   the per-part pixel hash the black-box producer reported, and each
//!   part of a multi-part file equals the same image's single-part file;
//! * the byte-level repack: split and combine reproduce the staged files
//!   byte for byte;
//! * the eleven rejected files fail with the violation their edit causes;
//! * the eleven tolerated files decode to the unmodified pixels, and
//!   strict mode names each irregularity.

use std::path::PathBuf;

use oxideav_openexr::{
    CombineOptions, DecodeOptions, ExrError, ExrPart, FileLayout, LayoutViolation,
    MultipartMixedImage, PartKind, PixelType, SplitOptions, check_layout, combine_parts,
    combine_parts_with, decode_all, decode_all_with, parse_exr, parse_exr_deep_multipart,
    parse_exr_deep_scanline, parse_exr_deep_tiled, parse_exr_multipart, parse_exr_multipart_mixed,
    parse_exr_multipart_tiled, parse_exr_tiled_multilevel, split_parts, split_parts_with,
};

fn fixture(rel: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/multipart")
        .join(rel);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

// ---------------------------------------------------------------------------
// SHA-1 (FIPS 180-1), for the producer's per-part pixel hashes.
// ---------------------------------------------------------------------------

fn sha1(data: &[u8]) -> String {
    let mut h: [u32; 5] = [
        0x6745_2301,
        0xEFCD_AB89,
        0x98BA_DCFE,
        0x1032_5476,
        0xC3D2_E1F0,
    ];
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in msg.as_chunks::<64>().0 {
        let mut w = [0u32; 80];
        for (i, word) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*word);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = h;
        for (i, wi) in w.iter().enumerate() {
            let (f, k) = match i / 20 {
                0 => ((b & c) | (!b & d), 0x5A82_7999),
                1 => (b ^ c ^ d, 0x6ED9_EBA1),
                2 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let t = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t;
        }
        for (s, v) in h.iter_mut().zip([a, b, c, d, e]) {
            *s = s.wrapping_add(v);
        }
    }
    h.iter().map(|v| format!("{v:08X}")).collect()
}

#[test]
fn sha1_known_answers() {
    assert_eq!(sha1(b"abc"), "A9993E364706816ABA3E25717850C26C9CD0D89D");
    assert_eq!(sha1(b""), "DA39A3EE5E6B4B0D3255BFEF95601890AFD80709");
}

/// The producer's pixel hash of a flat part: SHA-1 over the samples in
/// their stored type, pixel-interleaved, channels in the order the
/// producer command named them (`order`).
fn pixel_hash(part: &ExrPart, order: &[&str]) -> String {
    let mut bytes = Vec::new();
    let n = part.planes[0].samples.len();
    let idx: Vec<usize> = order
        .iter()
        .map(|name| {
            part.channels
                .iter()
                .position(|c| c.name == *name)
                .unwrap_or_else(|| panic!("channel {name}"))
        })
        .collect();
    assert_eq!(idx.len(), part.channels.len());
    for i in 0..n {
        for &c in &idx {
            let v = part.planes[c].samples[i];
            match part.channels[c].pixel_type {
                PixelType::Half => {
                    bytes.extend_from_slice(&oxideav_openexr::half::f32_to_half(v).to_le_bytes())
                }
                PixelType::Float => bytes.extend_from_slice(&v.to_le_bytes()),
                PixelType::Uint => bytes.extend_from_slice(&(v as u32).to_le_bytes()),
            }
        }
    }
    sha1(&bytes)
}

const SEED1_8X6: &str = "03EDC793EC6071900EDB9E82AB4A345F907B5EEA";
const SEED2_8X6: &str = "BDF3F12C";

fn flat(parts: &[MultipartMixedImage], i: usize) -> &ExrPart {
    parts[i]
        .image()
        .unwrap_or_else(|| panic!("part {i} is not a flat one-level part"))
}

// ---------------------------------------------------------------------------
// Layout facts
// ---------------------------------------------------------------------------

#[test]
fn layout_facts_match_the_fixture_notes() {
    // (file, flags, tables start, tables end, [(name, kind, chunks)])
    type Row = (&'static str, PartKind, u64);
    let cases: Vec<(&str, u32, usize, usize, Vec<Row>)> = vec![
        (
            "scan2/input.exr",
            0x1000,
            0x353,
            0x3b3,
            vec![
                ("subimage00", PartKind::Scanline, 6),
                ("subimage01", PartKind::Scanline, 6),
            ],
        ),
        (
            "scan3-zip/input.exr",
            0x1000,
            0x4e6,
            0x50e,
            vec![
                ("subimage00", PartKind::Scanline, 1),
                ("subimage01", PartKind::Scanline, 1),
                ("subimage02", PartKind::Scanline, 3),
            ],
        ),
        (
            "tiled2/input.exr",
            0x1000,
            0x385,
            0x3c5,
            vec![
                ("subimage00", PartKind::Tiled, 4),
                ("subimage01", PartKind::Tiled, 4),
            ],
        ),
        (
            "deep2/input.exr",
            0x1800,
            0x3c1,
            0x3f1,
            vec![
                ("subimage00", PartKind::DeepScanline, 3),
                ("subimage01", PartKind::DeepScanline, 3),
            ],
        ),
        (
            "mixed-constructed/input.exr",
            0x1c00,
            0xb7a,
            0xc92,
            vec![
                ("part00", PartKind::Scanline, 6),
                ("part01", PartKind::Tiled, 12),
                ("part02", PartKind::Scanline, 1),
                ("part03", PartKind::Scanline, 6),
                ("part04", PartKind::DeepScanline, 6),
                ("part05", PartKind::DeepTiled, 4),
            ],
        ),
    ];
    for (file, flags, start, end, parts) in cases {
        let bytes = fixture(file);
        let layout = check_layout(&bytes, true).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(layout.version.raw, 2 | flags, "{file}");
        assert_eq!(
            (layout.tables_start, layout.tables_end),
            (start, end),
            "{file}"
        );
        assert!(layout.irregularities.is_empty(), "{file}");
        assert_eq!(layout.parts.len(), parts.len(), "{file}");
        for (p, (name, kind, chunks)) in layout.parts.iter().zip(parts) {
            assert_eq!(p.name.as_deref(), Some(name), "{file}");
            assert_eq!(p.kind, kind, "{file} {name}");
            assert_eq!(p.chunk_count, chunks, "{file} {name}");
            assert_eq!(p.declared_chunk_count, Some(chunks as i32), "{file} {name}");
        }
        // Oracle-written files: tables, then chunks back to back to EOF.
        let map = layout.chunks(&bytes).unwrap();
        assert!(!map.rebuilt);
        let mut at = layout.tables_end;
        for r in map.parts.iter().flatten() {
            assert_eq!(r.start, at, "{file}");
            at = r.end;
        }
        assert_eq!(at, bytes.len(), "{file}");
    }
    // Single-part flags: flat scanline 0, flat tiled 0x200, deep 0x800
    // (also for deep tiled), long name 0x400.
    for (file, flags, kind) in [
        ("scan2/single.part0.exr", 0, PartKind::Scanline),
        ("tiled2/single.part0.exr", 0x200, PartKind::Tiled),
        ("deep2/single.part0.exr", 0x800, PartKind::DeepScanline),
        ("mixed-constructed/src.1.exr", 0x200, PartKind::Tiled),
        ("mixed-constructed/src.3.exr", 0x400, PartKind::Scanline),
        ("mixed-constructed/src.4.exr", 0x800, PartKind::DeepScanline),
        ("mixed-constructed/src.5.exr", 0x800, PartKind::DeepTiled),
    ] {
        let bytes = fixture(file);
        let layout = check_layout(&bytes, true).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(layout.version.raw, 2 | flags, "{file}");
        assert_eq!(layout.parts[0].kind, kind, "{file}");
    }
}

// ---------------------------------------------------------------------------
// Exact decode
// ---------------------------------------------------------------------------

#[test]
fn scanline_parts_decode_to_the_producer_hashes() {
    let bytes = fixture("scan2/input.exr");
    let parts = parse_exr_multipart(&bytes).unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(pixel_hash(&parts[0], &["R", "G", "B"]), SEED1_8X6);
    assert!(pixel_hash(&parts[1], &["R", "G", "B"]).starts_with(SEED2_8X6));
    for (i, part) in parts.iter().enumerate() {
        let single = parse_exr(&fixture(&format!("scan2/single.part{i}.exr"))).unwrap();
        assert_eq!(part.planes, single.planes, "part {i}");
        assert_eq!(part.channels, single.channels);
    }
    // The mixed reader and the contract root see the same samples.
    let mixed = parse_exr_multipart_mixed(&bytes).unwrap();
    assert_eq!(flat(&mixed, 0).planes, parts[0].planes);
    assert_eq!(flat(&mixed, 1).planes, parts[1].planes);
    let frames = decode_all(&bytes).unwrap();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].name.as_deref(), Some("subimage00"));
    assert_eq!(frames[1].part_type.as_deref(), Some("scanlineimage"));
}

#[test]
fn zip_parts_with_different_windows_decode_to_the_producer_hashes() {
    let bytes = fixture("scan3-zip/input.exr");
    let parts = parse_exr_multipart(&bytes).unwrap();
    assert_eq!(parts.len(), 3);
    // Part 0 is the seed-1 image of the uncompressed fixture, ZIP here.
    assert_eq!(pixel_hash(&parts[0], &["R", "G", "B"]), SEED1_8X6);
    assert_eq!(
        pixel_hash(&parts[1], &["R", "G", "B", "A"]),
        "DFF1769274792A917FDBD130B5CFEAC0EBEB246B"
    );
    // Part 2: data window (3,-2)-(7,37), three 16-line chunks at
    // y = -2, 14, 30.
    assert_eq!(
        (parts[2].data_window.x_min, parts[2].data_window.y_min),
        (3, -2)
    );
    assert_eq!((parts[2].width(), parts[2].height()), (5, 40));
    assert_eq!(
        pixel_hash(&parts[2], &["Y"]),
        "4393A206D4E18FF4B5C6A0E660E8AD1A48A50DED"
    );
    assert_eq!(parts[0].display_window, parts[2].display_window);
    assert_eq!(parts[0].display_window.width(), 16);
}

#[test]
fn tiled_parts_decode_to_the_producer_hashes() {
    let bytes = fixture("tiled2/input.exr");
    let parts = parse_exr_multipart_tiled(&bytes).unwrap();
    assert_eq!(parts.len(), 2);
    // Same two images as the scanline fixture, stored as 4x4 tiles.
    assert_eq!(pixel_hash(&parts[0], &["R", "G", "B"]), SEED1_8X6);
    assert!(pixel_hash(&parts[1], &["R", "G", "B"]).starts_with(SEED2_8X6));
    let single = parse_exr(&fixture("tiled2/single.part0.exr")).unwrap();
    assert_eq!(parts[0].planes, single.planes);
}

#[test]
fn deep_parts_equal_the_single_part_deep_file() {
    let bytes = fixture("deep2/input.exr");
    let parts = parse_exr_deep_multipart(&bytes).unwrap();
    assert_eq!(parts.len(), 2);
    let single = parse_exr_deep_scanline(&fixture("deep2/single.part0.exr")).unwrap();
    assert_eq!(parts[0].samples_per_pixel, single.samples_per_pixel);
    assert_eq!(parts[0].channel_samples, single.channel_samples);
    assert_eq!(parts[0].channels, single.channels);
    // One sample per pixel, five HALF channels, 4x3.
    assert_eq!(parts[0].samples_per_pixel, vec![1; 12]);
    assert_eq!(parts[1].channel_samples.len(), 5);
    // The Z channel is the constant 2.5 the producer appended.
    assert!(parts[1].channel_samples[4].iter().all(|&z| z == 2.5));
    assert_ne!(parts[0].channel_samples[0], parts[1].channel_samples[0]);
}

#[test]
fn mixed_file_decodes_every_part_exactly() {
    let bytes = fixture("mixed-constructed/input.exr");
    let parts = parse_exr_multipart_mixed(&bytes).unwrap();
    assert_eq!(parts.len(), 6);

    // 0: flat scanline, NONE.
    let p0 = flat(&parts, 0);
    assert_eq!(pixel_hash(p0, &["R", "G", "B"]), SEED1_8X6);
    assert_eq!(
        p0.planes,
        parse_exr(&fixture("mixed-constructed/src.0.exr"))
            .unwrap()
            .planes
    );

    // 1: mip-mapped tiles, ZIP, 13x7 -> 13x7, 6x3, 3x1, 1x1.
    let MultipartMixedImage::TiledMipmap(p1) = &parts[1] else {
        panic!("part 1 is a mip-mapped tiled part");
    };
    let dims: Vec<(u32, u32)> = p1.levels.iter().map(|l| (l.width, l.height)).collect();
    assert_eq!(dims, [(13, 7), (6, 3), (3, 1), (1, 1)]);
    let src1 = parse_exr_tiled_multilevel(&fixture("mixed-constructed/src.1.exr")).unwrap();
    assert_eq!(src1.levels.len(), 4);
    for (a, b) in p1.levels.iter().zip(&src1.levels) {
        assert_eq!(a.planes, b.planes);
    }

    // 2: FLOAT `ID` / `Z`, PIZ.
    let p2 = flat(&parts, 2);
    assert_eq!(
        pixel_hash(p2, &["Z", "ID"]),
        "FC420B52C0A5E2D13899C6142765DC032D5C4884"
    );
    assert_eq!(
        p2.planes,
        parse_exr(&fixture("mixed-constructed/src.2.exr"))
            .unwrap()
            .planes
    );

    // 3: a 48-byte channel name, RLE.
    let long = "a_channel_name_longer_than_31_chars_xxxxxxxxxxxxx";
    let p3 = flat(&parts, 3);
    assert!(p3.channels.iter().any(|c| c.name == long));
    assert_eq!(
        pixel_hash(p3, &["R", "G", long]),
        "0C44B3F112D2B57F4CADACA2402289A3505E7CC3"
    );
    assert_eq!(
        p3.planes,
        parse_exr(&fixture("mixed-constructed/src.3.exr"))
            .unwrap()
            .planes
    );

    // 4: deep scanline, NONE; 5: deep 4x4 tiles, ZIPS — the same image.
    let p4 = parts[4].deep_scanline().expect("part 4 deep scanline");
    let src4 = parse_exr_deep_scanline(&fixture("mixed-constructed/src.4.exr")).unwrap();
    assert_eq!(p4.samples_per_pixel, src4.samples_per_pixel);
    assert_eq!(p4.channel_samples, src4.channel_samples);
    let p5 = parts[5].deep_tiled().expect("part 5 deep tiled");
    let src5 = parse_exr_deep_tiled(&fixture("mixed-constructed/src.5.exr")).unwrap();
    assert_eq!(p5.samples_per_pixel, src5.samples_per_pixel);
    assert_eq!(p5.channel_samples, src5.channel_samples);
    assert_eq!(p4.channel_samples, p5.channel_samples);
    assert_eq!(p4.samples_per_pixel, vec![1; 48]);

    // The contract root views the RGB parts (part 3 as a partial RGB
    // triple) and skips the FLOAT `ID` / `Z` part and the deep parts.
    let frames = decode_all(&bytes).unwrap();
    let indices: Vec<u32> = frames.iter().map(|f| f.index).collect();
    assert_eq!(indices, [0, 1, 3]);
}

// ---------------------------------------------------------------------------
// Repack
// ---------------------------------------------------------------------------

#[test]
fn split_reproduces_the_single_part_files_byte_for_byte() {
    let strip = SplitOptions::new().with_strip_part_attributes(true);
    let scan = split_parts_with(&fixture("scan2/input.exr"), &strip).unwrap();
    assert_eq!(scan.len(), 2);
    assert_eq!(scan[0], fixture("scan2/single.part0.exr"));
    assert_eq!(scan[1], fixture("scan2/single.part1.exr"));

    let tiled = split_parts_with(&fixture("tiled2/input.exr"), &strip).unwrap();
    assert_eq!(tiled[0], fixture("tiled2/single.part0.exr"));

    let deep = split_parts_with(&fixture("deep2/input.exr"), &strip).unwrap();
    assert_eq!(deep[0], fixture("deep2/single.part0.exr"));

    let mixed = split_parts_with(&fixture("mixed-constructed/input.exr"), &strip).unwrap();
    assert_eq!(mixed.len(), 6);
    for (i, part) in mixed.iter().enumerate() {
        assert_eq!(
            *part,
            fixture(&format!("mixed-constructed/src.{i}.exr")),
            "part {i}"
        );
    }
}

#[test]
fn combine_reproduces_the_multi_part_files_byte_for_byte() {
    // Split keeping the headers, recombine: the staged file.
    for file in [
        "scan2/input.exr",
        "scan3-zip/input.exr",
        "tiled2/input.exr",
        "deep2/input.exr",
        "mixed-constructed/input.exr",
    ] {
        let bytes = fixture(file);
        let parts = split_parts(&bytes).unwrap();
        for p in &parts {
            // Each split part is a regular single-part file.
            check_layout(p, true).unwrap_or_else(|e| panic!("{file}: {e}"));
        }
        assert_eq!(combine_parts(&parts).unwrap(), bytes, "{file}");
    }
    // The single-part sources, named as the producer named the parts.
    let named = |names: &[&str]| CombineOptions::new().with_names(names.iter().copied());
    let scan = combine_parts_with(
        &[
            fixture("scan2/single.part0.exr"),
            fixture("scan2/single.part1.exr"),
        ],
        &named(&["subimage00", "subimage01"]),
    )
    .unwrap();
    assert_eq!(scan, fixture("scan2/input.exr"));

    let sources: Vec<Vec<u8>> = (0..6)
        .map(|i| fixture(&format!("mixed-constructed/src.{i}.exr")))
        .collect();
    let mixed = combine_parts_with(
        &sources,
        &named(&["part00", "part01", "part02", "part03", "part04", "part05"]),
    )
    .unwrap();
    assert_eq!(mixed, fixture("mixed-constructed/input.exr"));
}

#[test]
fn combine_checks_the_shared_attributes() {
    let a = fixture("scan2/single.part0.exr");
    let wide = split_parts(&fixture("scan3-zip/input.exr")).unwrap();
    // displayWindow 8x6 against 16x16.
    let err = combine_parts(&[a.clone(), wide[1].clone()]).unwrap_err();
    assert!(
        matches!(
            &err,
            ExrError::Layout(LayoutViolation::SharedAttributeMismatch { part: 1, attribute })
                if attribute == "displayWindow"
        ),
        "{err}"
    );
    // Unified on request: every part gets the union.
    let unified = combine_parts_with(
        &[a.clone(), wide[1].clone()],
        &CombineOptions::new().with_unify_display_window(true),
    )
    .unwrap();
    let layout = check_layout(&unified, true).unwrap();
    assert_eq!(layout.parts.len(), 2);
    let parts = parse_exr_multipart(&unified).unwrap();
    assert_eq!(parts[0].display_window, parts[1].display_window);
    assert_eq!(parts[0].display_window.width(), 16);
    assert_eq!(pixel_hash(&parts[0], &["R", "G", "B"]), SEED1_8X6);

    // Same name twice: refused, or renamed on request.
    let named = split_parts(&fixture("scan2/input.exr")).unwrap();
    let err = combine_parts(&[named[0].clone(), named[0].clone()]).unwrap_err();
    assert!(
        matches!(
            err,
            ExrError::Layout(LayoutViolation::DuplicatePartName {
                part: 1,
                first: 0,
                ..
            })
        ),
        "{err}"
    );
    let renamed = combine_parts_with(
        &[named[0].clone(), named[0].clone()],
        &CombineOptions::new().with_rename_duplicates(true),
    )
    .unwrap();
    let layout = check_layout(&renamed, true).unwrap();
    assert_ne!(layout.parts[0].name, layout.parts[1].name);

    // A multi-part input is not a part.
    assert!(combine_parts(&[fixture("scan2/input.exr")]).is_err());
    let none: [Vec<u8>; 0] = [];
    assert!(matches!(
        combine_parts(&none),
        Err(ExrError::Layout(LayoutViolation::NoParts))
    ));
}

// ---------------------------------------------------------------------------
// Rejected files
// ---------------------------------------------------------------------------

fn violation(file: &str) -> LayoutViolation {
    let bytes = fixture(file);
    match check_layout(&bytes, false) {
        Err(ExrError::Layout(v)) => v,
        other => panic!("{file}: expected a layout violation, got {other:?}"),
    }
}

/// Every reader refuses `file`.
fn assert_unreadable(file: &str) {
    let bytes = fixture(file);
    assert!(parse_exr_multipart_mixed(&bytes).is_err(), "{file}");
    assert!(parse_exr_multipart(&bytes).is_err(), "{file}");
    assert!(decode_all(&bytes).is_err(), "{file}");
    assert!(split_parts(&bytes).is_err(), "{file}");
}

#[test]
fn rejected_files_fail_with_the_rule_they_break() {
    use LayoutViolation as V;
    let shared = |file: &str, name: &str| match violation(file) {
        V::SharedAttributeMismatch { part: 1, attribute } => assert_eq!(attribute, name, "{file}"),
        other => panic!("{file}: {other:?}"),
    };
    shared("negative/displaywindow-differs.exr", "displayWindow");
    shared("negative/pixelaspectratio-differs.exr", "pixelAspectRatio");
    shared("negative/chromaticities-only-part1.exr", "chromaticities");
    assert_eq!(
        violation("negative/name-missing.exr"),
        V::MissingName { part: 1 }
    );
    assert_eq!(
        violation("negative/type-missing.exr"),
        V::MissingType { part: 1 }
    );
    assert_eq!(
        violation("negative/type-unknown.exr"),
        V::UnknownPartType {
            part: 1,
            found: "bogus".to_string()
        }
    );
    assert_eq!(
        violation("negative/flags-1200.exr"),
        V::ConflictingVersionFlags { flags: 0x1200 }
    );
    assert_eq!(
        violation("negative/flags-3000.exr"),
        V::UnknownVersionFlags { flags: 0x3000 }
    );
    assert!(matches!(
        violation("negative/no-empty-header.exr"),
        V::HeaderListNotTerminated { parts_read: 2, .. }
    ));
    assert_eq!(
        violation("negative/wrong-part-number.exr"),
        V::ChunkPartNumber {
            part: 0,
            chunk: 0,
            found: 1
        }
    );
    for file in [
        "negative/displaywindow-differs.exr",
        "negative/pixelaspectratio-differs.exr",
        "negative/chromaticities-only-part1.exr",
        "negative/name-missing.exr",
        "negative/type-missing.exr",
        "negative/type-unknown.exr",
        "negative/flags-1200.exr",
        "negative/flags-3000.exr",
        "negative/no-empty-header.exr",
        "negative/wrong-part-number.exr",
        "negative/flags-0000.exr",
    ] {
        assert_unreadable(file);
        assert!(
            decode_all_with(&fixture(file), &DecodeOptions::default().with_strict(true)).is_err(),
            "{file}"
        );
    }
    // The multi-part flag cleared: the header opens as one part (what
    // follows the first header is taken for its offset table) and the
    // pixel read fails.
    let cleared = fixture("negative/flags-0000.exr");
    let layout = FileLayout::read(&cleared).unwrap();
    assert_eq!(layout.parts.len(), 1);
    assert!(!layout.is_multipart());
    assert!(layout.chunks(&cleared).is_err());
    assert!(parse_exr(&cleared).is_err());
}

// ---------------------------------------------------------------------------
// Tolerated files
// ---------------------------------------------------------------------------

#[test]
fn tolerated_files_decode_to_the_unmodified_pixels() {
    use LayoutViolation as V;
    let reference = parse_exr_multipart(&fixture("scan2/input.exr")).unwrap();
    // (file, parts, the irregularity strict mode reports)
    type Check = fn(&V) -> bool;
    let cases: Vec<(&str, usize, Option<Check>)> = vec![
        (
            "tolerated/chunkcount-missing.exr",
            2,
            Some(|v| {
                matches!(
                    v,
                    V::ChunkCountMissing {
                        part: 0,
                        expected: 6
                    }
                )
            }),
        ),
        (
            "tolerated/chunkcount-wrong.exr",
            2,
            Some(|v| {
                matches!(
                    v,
                    V::ChunkCountMismatch {
                        part: 0,
                        declared: 99,
                        expected: 6
                    }
                )
            }),
        ),
        (
            "tolerated/offset-table-zero.exr",
            2,
            Some(|v| matches!(v, V::OffsetTableZero { part: 0 })),
        ),
        (
            "tolerated/duplicate-names.exr",
            2,
            Some(
                |v| matches!(v, V::DuplicatePartName { part: 1, first: 0, name } if name == "subimage00"),
            ),
        ),
        (
            "tolerated/longname-flag-clear.exr",
            2,
            Some(|v| matches!(v, V::LongNameFlagClear { part: 1, name } if name.len() == 32)),
        ),
        ("tolerated/chunks-interleaved.exr", 2, None),
        ("tolerated/timecode-differs.exr", 2, None),
        ("tolerated/screenwindow-differs.exr", 2, None),
        ("tolerated/chromaticities-only-part0.exr", 2, None),
        ("tolerated/one-part-multipart.exr", 1, None),
    ];
    for (file, count, irregular) in cases {
        let bytes = fixture(file);
        let lenient = check_layout(&bytes, false).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(lenient.parts.len(), count, "{file}");
        // Every reader accepts the file and sees the unmodified pixels.
        let parts = parse_exr_multipart(&bytes).unwrap_or_else(|e| panic!("{file}: {e}"));
        let mixed = parse_exr_multipart_mixed(&bytes).unwrap_or_else(|e| panic!("{file}: {e}"));
        let frames = decode_all(&bytes).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(
            (parts.len(), mixed.len(), frames.len()),
            (count, count, count)
        );
        for (i, part) in parts.iter().enumerate() {
            assert_eq!(part.planes, reference[i].planes, "{file} part {i}");
            assert_eq!(
                flat(&mixed, i).planes,
                reference[i].planes,
                "{file} part {i}"
            );
        }
        assert_eq!(pixel_hash(&parts[0], &["R", "G", "B"]), SEED1_8X6, "{file}");
        // Split parts are regular files with the same pixels.
        for (i, single) in split_parts(&bytes).unwrap().iter().enumerate() {
            check_layout(single, false).unwrap_or_else(|e| panic!("{file} part {i}: {e}"));
            assert_eq!(
                parse_exr(single).unwrap().planes,
                reference[i].planes,
                "{file} part {i}"
            );
        }
        // Strict mode.
        let strict = check_layout(&bytes, true);
        let strict_decode = decode_all_with(&bytes, &DecodeOptions::default().with_strict(true));
        match irregular {
            Some(check) => {
                assert!(
                    lenient.irregularities.iter().any(check),
                    "{file}: {:?}",
                    lenient.irregularities
                );
                assert!(lenient.irregularities.iter().all(V::is_tolerated), "{file}");
                match strict {
                    Err(ExrError::Layout(v)) => assert!(check(&v), "{file}: {v:?}"),
                    other => panic!("{file}: strict accepted: {other:?}"),
                }
                assert!(matches!(strict_decode, Err(ExrError::Layout(_))), "{file}");
            }
            None => {
                assert!(
                    lenient.irregularities.is_empty(),
                    "{file}: {:?}",
                    lenient.irregularities
                );
                assert!(strict.is_ok(), "{file}");
                assert_eq!(strict_decode.unwrap().len(), count, "{file}");
            }
        }
    }

    // A single-part file may carry `name`, `type`, `chunkCount`.
    let single = fixture("tolerated/single-with-multipart-attrs.exr");
    let layout = check_layout(&single, true).unwrap();
    assert!(!layout.is_multipart());
    assert_eq!(layout.parts[0].name.as_deref(), Some("subimage00"));
    let part = parse_exr(&single).unwrap();
    assert_eq!(part.planes, reference[0].planes);
    assert_eq!(pixel_hash(&part, &["R", "G", "B"]), SEED1_8X6);
}

#[test]
fn chunks_are_found_through_the_tables_not_by_position() {
    // Interleaved storage: the tables still list each part's chunks top
    // to bottom.
    let bytes = fixture("tolerated/chunks-interleaved.exr");
    let layout = FileLayout::read(&bytes).unwrap();
    let map = layout.chunks(&bytes).unwrap();
    assert!(!map.rebuilt);
    assert!(map.parts[0][1].start > map.parts[1][0].start);
    for (p, ranges) in map.parts.iter().enumerate() {
        for (k, r) in ranges.iter().enumerate() {
            let part = i32::from_le_bytes(bytes[r.start..r.start + 4].try_into().unwrap());
            let y = i32::from_le_bytes(bytes[r.start + 4..r.start + 8].try_into().unwrap());
            assert_eq!((part as usize, y as usize), (p, k));
        }
    }
    // Zeroed tables are rebuilt from the chunks themselves.
    let zero = fixture("tolerated/offset-table-zero.exr");
    let layout = FileLayout::read(&zero).unwrap();
    let rebuilt = layout.chunks(&zero).unwrap();
    assert!(rebuilt.rebuilt);
    let regular = fixture("scan2/input.exr");
    let want = FileLayout::read(&regular)
        .unwrap()
        .chunks(&regular)
        .unwrap();
    assert_eq!(rebuilt.parts, want.parts);
}
