//! Round-trip properties of the multi-part layout over writer-built
//! files mixing every part type (flat scanline, flat tiles — one level
//! and mip-mapped —, deep scanline, deep tiles):
//!
//! * what the writers emit passes the strict layout check, with exactly
//!   the chunk counts of the multi-part note's formulas;
//! * decode returns the written samples;
//! * `combine(split(f)) == f`, and every split part decodes alone;
//! * the irregularities a conforming reader tolerates (chunks stored in
//!   any order, zeroed offset tables, missing `chunkCount`) decode to
//!   the same samples, and strict mode names them;
//! * black-box readers (`exrinfo`, `exrmetrics`, run as opaque processes
//!   when installed) accept every file and every split part.

use std::process::Command;

use oxideav_openexr::{
    Attribute, AttributeValue, Box2i, Channel, Compression, ExrError, FileLayout, LayoutViolation,
    LineOrder, MultipartMixedImage, MultipartMixedPart, MultipartScanlinePart, PartKind, PixelType,
    build_box_filter_pyramid, check_layout, chunk_count, combine_parts, encode_exr_multipart,
    encode_exr_multipart_mixed, encode_exr_scanline, parse_exr, parse_exr_deep_scanline,
    parse_exr_deep_tiled, parse_exr_multipart_mixed, parse_exr_tiled_multilevel, parse_header,
    split_parts,
};

/// xorshift64* — deterministic, seedable.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len() as u64) as usize]
    }
    /// A value exactly representable as a HALF.
    fn sample(&mut self) -> f32 {
        (self.below(4096) as f32 - 2048.0) / 64.0
    }
}

const FLAT: [Compression; 5] = [
    Compression::None,
    Compression::Rle,
    Compression::Zips,
    Compression::Zip,
    Compression::Piz,
];
const DEEP: [Compression; 3] = [Compression::None, Compression::Rle, Compression::Zips];

/// Owned source data of one part.
enum Source {
    Scanline {
        w: u32,
        h: u32,
        planes: Vec<Vec<f32>>,
        z: Compression,
    },
    Tiled {
        w: u32,
        h: u32,
        tile: (u32, u32),
        planes: Vec<Vec<f32>>,
        z: Compression,
    },
    Mipmap {
        w: u32,
        h: u32,
        tile: (u32, u32),
        planes: Vec<Vec<f32>>,
        z: Compression,
    },
    DeepScanline {
        w: u32,
        h: u32,
        spp: Vec<u32>,
        samples: Vec<Vec<f32>>,
        z: Compression,
    },
    DeepTiled {
        w: u32,
        h: u32,
        tile: (u32, u32),
        spp: Vec<u32>,
        samples: Vec<Vec<f32>>,
        z: Compression,
    },
}

fn channels(n: usize, rng: &mut Rng) -> Vec<Channel> {
    let pt = rng.pick(&[PixelType::Half, PixelType::Float]);
    ["A", "B", "G", "R", "Z"][..n]
        .iter()
        .map(|c| Channel::new(c.to_string(), pt, false, 1, 1))
        .collect()
}

fn random_source(rng: &mut Rng) -> (Source, Vec<Channel>) {
    let w = 1 + rng.below(19) as u32;
    let h = 1 + rng.below(40) as u32;
    let n = 1 + rng.below(4) as usize;
    let chs = channels(n, rng);
    let tile = (1 + rng.below(6) as u32, 1 + rng.below(6) as u32);
    let flat = |rng: &mut Rng| -> Vec<Vec<f32>> {
        (0..n)
            .map(|_| (0..w * h).map(|_| rng.sample()).collect())
            .collect()
    };
    let deep = |rng: &mut Rng| -> (Vec<u32>, Vec<Vec<f32>>) {
        let spp: Vec<u32> = (0..w * h).map(|_| rng.below(4) as u32).collect();
        let total: u32 = spp.iter().sum();
        let samples = (0..n)
            .map(|_| (0..total).map(|_| rng.sample()).collect())
            .collect();
        (spp, samples)
    };
    let source = match rng.below(5) {
        0 => Source::Scanline {
            w,
            h,
            planes: flat(rng),
            z: rng.pick(&FLAT),
        },
        1 => Source::Tiled {
            w,
            h,
            tile,
            planes: flat(rng),
            z: rng.pick(&FLAT),
        },
        2 => Source::Mipmap {
            w,
            h,
            tile,
            planes: flat(rng),
            z: rng.pick(&FLAT),
        },
        3 => {
            let (spp, samples) = deep(rng);
            Source::DeepScanline {
                w,
                h,
                spp,
                samples,
                z: rng.pick(&DEEP),
            }
        }
        _ => {
            let (spp, samples) = deep(rng);
            Source::DeepTiled {
                w,
                h,
                tile,
                spp,
                samples,
                z: rng.pick(&DEEP),
            }
        }
    };
    (source, chs)
}

fn build(sources: &[(Source, Vec<Channel>)]) -> Vec<u8> {
    let parts: Vec<MultipartMixedPart> = sources
        .iter()
        .enumerate()
        .map(|(i, (s, chs))| {
            let name = format!("p{i}");
            let channels = chs.clone();
            match s {
                Source::Scanline { w, h, planes, z } => MultipartMixedPart::Scanline {
                    name,
                    width: *w,
                    height: *h,
                    channels,
                    planes: planes.iter().map(|p| p.as_slice()).collect(),
                    compression: *z,
                },
                Source::Tiled {
                    w,
                    h,
                    tile,
                    planes,
                    z,
                } => MultipartMixedPart::Tiled {
                    name,
                    width: *w,
                    height: *h,
                    tile_x: tile.0,
                    tile_y: tile.1,
                    channels,
                    planes: planes.iter().map(|p| p.as_slice()).collect(),
                    compression: *z,
                },
                Source::Mipmap {
                    w,
                    h,
                    tile,
                    planes,
                    z,
                } => MultipartMixedPart::TiledMipmap {
                    name,
                    tile_x: tile.0,
                    tile_y: tile.1,
                    channels,
                    pyramid: build_box_filter_pyramid(*w, *h, planes),
                    compression: *z,
                },
                Source::DeepScanline {
                    w,
                    h,
                    spp,
                    samples,
                    z,
                } => MultipartMixedPart::DeepScanline {
                    name,
                    width: *w,
                    height: *h,
                    channels,
                    samples_per_pixel: spp,
                    channel_samples: samples.iter().map(|p| p.as_slice()).collect(),
                    compression: *z,
                },
                Source::DeepTiled {
                    w,
                    h,
                    tile,
                    spp,
                    samples,
                    z,
                } => MultipartMixedPart::DeepTiled {
                    name,
                    width: *w,
                    height: *h,
                    tile_x: tile.0,
                    tile_y: tile.1,
                    channels,
                    samples_per_pixel: spp,
                    channel_samples: samples.iter().map(|p| p.as_slice()).collect(),
                    compression: *z,
                },
            }
        })
        .collect();
    encode_exr_multipart_mixed(&parts).expect("writer")
}

/// The chunk count the multi-part note gives a part.
fn formula(s: &Source) -> u64 {
    let ceil = |n: u32, d: u32| n.div_ceil(d) as u64;
    let lines = |z: Compression| match z {
        Compression::Zip | Compression::Pxr24 => 16,
        Compression::Piz | Compression::B44 | Compression::B44a | Compression::Dwaa => 32,
        Compression::Dwab => 256,
        _ => 1,
    };
    match s {
        Source::Scanline { h, z, .. } => ceil(*h, lines(*z)),
        Source::DeepScanline { h, .. } => *h as u64,
        Source::Tiled { w, h, tile, .. } | Source::DeepTiled { w, h, tile, .. } => {
            ceil(*w, tile.0) * ceil(*h, tile.1)
        }
        Source::Mipmap { w, h, tile, .. } => {
            let (mut lw, mut lh, mut total) = (*w, *h, 0);
            loop {
                total += ceil(lw, tile.0) * ceil(lh, tile.1);
                if lw == 1 && lh == 1 {
                    break total;
                }
                lw = (lw / 2).max(1);
                lh = (lh / 2).max(1);
            }
        }
    }
}

/// The box-filtered pyramid as a file stores it: the averaged levels
/// rounded to the channel type.
fn stored_pyramid(w: u32, h: u32, planes: &[Vec<f32>], pt: PixelType) -> Vec<Vec<Vec<f32>>> {
    use oxideav_openexr::half::{f32_to_half, half_to_f32};
    build_box_filter_pyramid(w, h, planes)
        .into_iter()
        .map(|level| {
            level
                .planes
                .into_iter()
                .map(|plane| match pt {
                    PixelType::Half => plane
                        .into_iter()
                        .map(|v| half_to_f32(f32_to_half(v)))
                        .collect(),
                    _ => plane,
                })
                .collect()
        })
        .collect()
}

fn assert_decodes(bytes: &[u8], sources: &[(Source, Vec<Channel>)], label: &str) {
    let parts = parse_exr_multipart_mixed(bytes).unwrap_or_else(|e| panic!("{label}: {e}"));
    assert_eq!(parts.len(), sources.len(), "{label}");
    for (i, (part, (source, _))) in parts.iter().zip(sources).enumerate() {
        let label = format!("{label} part {i}");
        match (part, source) {
            (MultipartMixedImage::Scanline(img), Source::Scanline { planes, .. })
            | (MultipartMixedImage::Tiled(img), Source::Tiled { planes, .. }) => {
                for (got, want) in img.planes.iter().zip(planes) {
                    assert_eq!(&got.samples, want, "{label}");
                }
            }
            (MultipartMixedImage::TiledMipmap(img), Source::Mipmap { w, h, planes, .. }) => {
                let pyramid = stored_pyramid(*w, *h, planes, img.channels[0].pixel_type);
                assert_eq!(img.levels.len(), pyramid.len(), "{label}");
                for (got, want) in img.levels.iter().zip(&pyramid) {
                    for (g, w) in got.planes.iter().zip(want) {
                        assert_eq!(&g.samples, w, "{label}");
                    }
                }
            }
            (MultipartMixedImage::DeepScanline(img), Source::DeepScanline { spp, samples, .. }) => {
                assert_eq!(&img.samples_per_pixel, spp, "{label}");
                assert_eq!(&img.channel_samples, samples, "{label}");
            }
            (MultipartMixedImage::DeepTiled(img), Source::DeepTiled { spp, samples, .. }) => {
                assert_eq!(&img.samples_per_pixel, spp, "{label}");
                assert_eq!(&img.channel_samples, samples, "{label}");
            }
            _ => panic!("{label}: part type changed"),
        }
    }
}

/// Rewrite `bytes` with its chunks stored in reverse order (offset
/// tables updated), 16 bytes of padding between tables and chunks.
fn reorder_chunks(bytes: &[u8]) -> Vec<u8> {
    let layout = FileLayout::read(bytes).unwrap();
    let map = layout.chunks(bytes).unwrap();
    let mut out = bytes[..layout.tables_end].to_vec();
    out.extend_from_slice(&[0xAB; 16]);
    let mut all: Vec<(usize, usize)> = Vec::new();
    for (p, ranges) in map.parts.iter().enumerate() {
        for k in 0..ranges.len() {
            all.push((p, k));
        }
    }
    for &(p, k) in all.iter().rev() {
        let at = out.len() as u64;
        let slot = layout.parts[p].table_offset + k * 8;
        out[slot..slot + 8].copy_from_slice(&at.to_le_bytes());
        out.extend_from_slice(&bytes[map.parts[p][k].clone()]);
    }
    out
}

fn zero_tables(bytes: &[u8]) -> Vec<u8> {
    let layout = FileLayout::read(bytes).unwrap();
    let mut out = bytes.to_vec();
    out[layout.tables_start..layout.tables_end].fill(0);
    out
}

fn tool_available(bin: &str) -> bool {
    Command::new(bin)
        .arg("--help")
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

fn scratch(label: &str) -> std::path::PathBuf {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "oxideav-openexr-layout-{}-{seq}-{label}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Header-level black-box readers (each skipped when not installed).
fn header_reader_accepts(bytes: &[u8], label: &str) {
    for tool in ["exrinfo", "exrheader"] {
        if !tool_available(tool) {
            continue;
        }
        let dir = scratch("info");
        let path = dir.join("in.exr");
        std::fs::write(&path, bytes).unwrap();
        let out = Command::new(tool).arg(&path).output().unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            out.status.success(),
            "{label}: a black-box reader ({tool}) rejected the file:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
}

/// Pixel-level black-box reader: reads every chunk of every part and
/// writes a copy; returns the copy.
fn pixel_reader_copy(bytes: &[u8], label: &str) -> Option<Vec<u8>> {
    if !tool_available("exrmetrics") {
        return None;
    }
    let dir = scratch("copy");
    let (input, output) = (dir.join("in.exr"), dir.join("out.exr"));
    std::fs::write(&input, bytes).unwrap();
    let out = Command::new("exrmetrics")
        .arg("--convert")
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()
        .unwrap();
    // The tool writes a one-part file as a single-part file; hand that
    // back as the one-part multi-part file it reads as.
    let copy = std::fs::read(&output).ok().map(|copy| {
        if FileLayout::read(&copy).is_ok_and(|l| l.is_multipart()) {
            copy
        } else {
            combine_parts(&[copy]).expect("single-part copy")
        }
    });
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        out.status.success(),
        "{label}: a black-box reader failed to read the pixels:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    copy
}

#[test]
fn mixed_files_round_trip_through_every_layout_operation() {
    let mut rng = Rng(0x0BAD_5EED_CAFE_F00D);
    let mut kinds_seen = std::collections::HashSet::new();
    for case in 0..48 {
        let n = 1 + rng.below(5) as usize;
        let sources: Vec<(Source, Vec<Channel>)> =
            (0..n).map(|_| random_source(&mut rng)).collect();
        let bytes = build(&sources);
        let label = format!("case {case}");

        // Strict layout, formula chunk counts, version flags.
        let layout = check_layout(&bytes, true).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(layout.irregularities.is_empty(), "{label}");
        let any_deep = layout.parts.iter().any(|p| p.kind.is_deep());
        assert_eq!(
            layout.version.raw,
            2 | 0x1000 | if any_deep { 0x800 } else { 0 },
            "{label}"
        );
        for (p, (s, _)) in layout.parts.iter().zip(&sources) {
            assert_eq!(p.chunk_count, formula(s), "{label} part {}", p.index);
            assert_eq!(p.declared_chunk_count, Some(formula(s) as i32), "{label}");
            assert_eq!(chunk_count(p.kind, &p.attributes).unwrap(), formula(s));
            kinds_seen.insert(p.kind);
        }
        assert_decodes(&bytes, &sources, &label);

        // Split: every part is a regular single-part file with the
        // right flags and the same samples; combine restores the file.
        let singles = split_parts(&bytes).unwrap();
        assert_eq!(singles.len(), n, "{label}");
        for (single, p) in singles.iter().zip(&layout.parts) {
            let one = check_layout(single, true).unwrap_or_else(|e| panic!("{label}: {e}"));
            let want = match p.kind {
                PartKind::Scanline => 0,
                PartKind::Tiled => 0x200,
                PartKind::DeepScanline | PartKind::DeepTiled => 0x800,
            };
            assert_eq!(one.version.raw, 2 | want, "{label} part {}", p.index);
            assert_eq!(one.parts[0].chunk_count, p.chunk_count, "{label}");
        }
        for (single, (source, _)) in singles.iter().zip(&sources) {
            match source {
                Source::Scanline { planes, .. } | Source::Tiled { planes, .. } => {
                    let part = parse_exr(single).unwrap();
                    for (got, want) in part.planes.iter().zip(planes) {
                        assert_eq!(&got.samples, want, "{label}");
                    }
                }
                Source::Mipmap { w, h, planes, .. } => {
                    let img = parse_exr_tiled_multilevel(single).unwrap();
                    let pyramid = stored_pyramid(*w, *h, planes, img.channels[0].pixel_type);
                    assert_eq!(img.levels.len(), pyramid.len(), "{label}");
                    for (got, want) in img.levels.iter().zip(&pyramid) {
                        for (g, w) in got.planes.iter().zip(want) {
                            assert_eq!(&g.samples, w, "{label}");
                        }
                    }
                }
                Source::DeepScanline { spp, samples, .. } => {
                    let img = parse_exr_deep_scanline(single).unwrap();
                    assert_eq!(
                        (&img.samples_per_pixel, &img.channel_samples),
                        (spp, samples)
                    );
                }
                Source::DeepTiled { spp, samples, .. } => {
                    let img = parse_exr_deep_tiled(single).unwrap();
                    assert_eq!(
                        (&img.samples_per_pixel, &img.channel_samples),
                        (spp, samples)
                    );
                }
            }
        }
        assert_eq!(combine_parts(&singles).unwrap(), bytes, "{label}");

        // Chunks in another physical order, after a gap: still regular
        // for a conforming reader, and for strict mode.
        let reordered = reorder_chunks(&bytes);
        check_layout(&reordered, true).unwrap_or_else(|e| panic!("{label} reordered: {e}"));
        assert_decodes(&reordered, &sources, &format!("{label} reordered"));
        assert_eq!(split_parts(&reordered).unwrap(), singles, "{label}");

        // Zeroed tables: rebuilt from the chunks.
        let zeroed = zero_tables(&bytes);
        assert_decodes(&zeroed, &sources, &format!("{label} zeroed"));
        assert!(
            matches!(
                check_layout(&zeroed, true),
                Err(ExrError::Layout(LayoutViolation::OffsetTableZero { .. }))
            ),
            "{label}"
        );
        assert_eq!(split_parts(&zeroed).unwrap(), singles, "{label}");

        // Black-box readers.
        if case < 16 {
            header_reader_accepts(&bytes, &label);
            for (i, single) in singles.iter().enumerate() {
                header_reader_accepts(single, &format!("{label} split part {i}"));
            }
            if let Some(copy) = pixel_reader_copy(&bytes, &label) {
                assert_decodes(&copy, &sources, &format!("{label} black-box copy"));
            }
            header_reader_accepts(&reordered, &format!("{label} reordered"));
            if let Some(copy) = pixel_reader_copy(&reordered, &format!("{label} reordered")) {
                assert_decodes(
                    &copy,
                    &sources,
                    &format!("{label} reordered black-box copy"),
                );
            }
        }
    }
    assert_eq!(kinds_seen.len(), 4, "every part kind exercised");
}

/// The required attributes of a `w`x`h` scanline file, then `extra`.
fn header_attributes(w: u32, h: u32, chs: &[Channel], extra: Vec<Attribute>) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: w as i32 - 1,
        y_max: h as i32 - 1,
    };
    let mut attrs = vec![
        Attribute::new(
            "channels".to_string(),
            AttributeValue::Channels(chs.to_vec()),
        ),
        Attribute::new(
            "compression".to_string(),
            AttributeValue::Compression(Compression::None),
        ),
        Attribute::new("dataWindow".to_string(), AttributeValue::Box2i(win)),
        Attribute::new("displayWindow".to_string(), AttributeValue::Box2i(win)),
        Attribute::new(
            "lineOrder".to_string(),
            AttributeValue::LineOrder(LineOrder::IncreasingY),
        ),
        Attribute::new("pixelAspectRatio".to_string(), AttributeValue::Float(1.0)),
        Attribute::new(
            "screenWindowCenter".to_string(),
            AttributeValue::V2f(0.0, 0.0),
        ),
        Attribute::new("screenWindowWidth".to_string(), AttributeValue::Float(1.0)),
    ];
    attrs.extend(extra);
    attrs
}

fn half_channels(names: &[&str]) -> Vec<Channel> {
    let mut chs: Vec<Channel> = names
        .iter()
        .map(|n| Channel::new(n.to_string(), PixelType::Half, false, 1, 1))
        .collect();
    chs.sort_by(|a, b| a.name.cmp(&b.name));
    chs
}

#[test]
fn writers_set_the_long_name_flag() {
    let long = "a_channel_name_of_more_than_thirty_one_bytes";
    let plane = vec![0.5f32; 6];
    // 31 bytes: flag clear. 32 bytes: flag set.
    for (name, flag) in [(&long[..31], 0u32), (&long[..32], 0x400)] {
        let chs = half_channels(&["R", name]);
        let single = encode_exr_scanline(
            3,
            2,
            &chs,
            &[&plane, &plane],
            Compression::None,
            header_attributes(3, 2, &chs, Vec::new()),
        )
        .unwrap();
        assert_eq!(parse_header(&single).unwrap().version.raw, 2 | flag);
        assert!(
            check_layout(&single, true)
                .unwrap()
                .irregularities
                .is_empty()
        );
        let part = parse_exr(&single).unwrap();
        assert!(part.channels.iter().any(|c| c.name == name));
        header_reader_accepts(&single, name);

        let multi = encode_exr_multipart(&[
            MultipartScanlinePart::new(
                "first".to_string(),
                3,
                2,
                half_channels(&["R"]),
                vec![&plane],
                Compression::Zip,
            ),
            MultipartScanlinePart::new(
                "second".to_string(),
                3,
                2,
                chs.clone(),
                vec![&plane, &plane],
                Compression::Zip,
            ),
        ])
        .unwrap();
        let layout = check_layout(&multi, true).unwrap();
        assert_eq!(layout.version.raw, 2 | 0x1000 | flag);
        header_reader_accepts(&multi, name);
        // The split parts carry the flag only where the name lives.
        let singles = split_parts(&multi).unwrap();
        assert_eq!(parse_header(&singles[0]).unwrap().version.raw, 2);
        assert_eq!(parse_header(&singles[1]).unwrap().version.raw, 2 | flag);
    }
    // A long attribute name sets it too.
    let chs = half_channels(&["R"]);
    let extra = vec![Attribute::new(
        long.to_string(),
        AttributeValue::String("x".to_string()),
    )];
    let single = encode_exr_scanline(
        3,
        2,
        &chs,
        &[&plane],
        Compression::None,
        header_attributes(3, 2, &chs, extra),
    )
    .unwrap();
    assert_eq!(parse_header(&single).unwrap().version.raw, 2 | 0x400);
    header_reader_accepts(&single, "long attribute name");
}

#[test]
fn writers_refuse_duplicate_part_names() {
    let plane = vec![0.25f32; 4];
    let part = |name: &str| {
        MultipartScanlinePart::new(
            name.to_string(),
            2,
            2,
            half_channels(&["Y"]),
            vec![&plane],
            Compression::None,
        )
    };
    assert!(encode_exr_multipart(&[part("a"), part("b")]).is_ok());
    let err = encode_exr_multipart(&[part("a"), part("b"), part("a")]).unwrap_err();
    assert!(
        matches!(
            err,
            ExrError::Layout(LayoutViolation::DuplicatePartName {
                part: 2,
                first: 0,
                ..
            })
        ),
        "{err}"
    );
}

#[test]
fn hostile_counts_are_refused_before_any_table_is_sized() {
    let plane = vec![0.25f32; 4];
    let bytes = encode_exr_multipart(&[MultipartScanlinePart::new(
        "a".to_string(),
        2,
        2,
        half_channels(&["Y"]),
        vec![&plane],
        Compression::None,
    )])
    .unwrap();
    // dataWindow blown up to 2^31 rows: the offset table would need
    // 16 GiB.
    let at = bytes
        .windows(17)
        .position(|w| w == b"dataWindow\0box2i\0")
        .unwrap()
        + 17
        + 4;
    let mut hostile = bytes.clone();
    hostile[at + 12..at + 16].copy_from_slice(&i32::MAX.to_le_bytes());
    for result in [
        check_layout(&hostile, false).map(|_| ()),
        parse_exr_multipart_mixed(&hostile).map(|_| ()),
        split_parts(&hostile).map(|_| ()),
    ] {
        assert!(
            matches!(
                result,
                Err(ExrError::Layout(LayoutViolation::OffsetTableTruncated {
                    part: 0,
                    ..
                }))
            ),
            "{result:?}"
        );
    }
    // A data window 2^27 pixels wide keeps the two-entry offset table
    // but declares gigabytes of samples the chunks cannot hold:
    // refused before a plane is sized, in multi-part and single-part
    // files alike.
    let mut wide = bytes.clone();
    wide[at + 8..at + 12].copy_from_slice(&(1i32 << 27).to_le_bytes());
    for result in [
        check_layout(&wide, false).map(|_| ()),
        parse_exr_multipart_mixed(&wide).map(|_| ()),
        split_parts(&wide).map(|_| ()),
    ] {
        assert!(
            matches!(
                result,
                Err(ExrError::Layout(LayoutViolation::PixelDataNotBacked {
                    part: 0,
                    ..
                }))
            ),
            "{result:?}"
        );
    }
    let single = split_parts(&bytes).unwrap().remove(0);
    let at = single
        .windows(17)
        .position(|w| w == b"dataWindow\0box2i\0")
        .unwrap()
        + 17
        + 4;
    let mut wide = single.clone();
    wide[at + 8..at + 12].copy_from_slice(&(1i32 << 27).to_le_bytes());
    assert!(parse_exr(&single).is_ok());
    assert!(matches!(
        parse_exr(&wide),
        Err(ExrError::Layout(LayoutViolation::PixelDataNotBacked {
            part: 0,
            ..
        }))
    ));
    // A header list of thousands of minimal parts costs nothing beyond
    // its own bytes and is refused at the first missing attribute.
    let mut many = bytes[..8].to_vec();
    for _ in 0..50_000 {
        many.extend_from_slice(b"a\0int\0\x04\0\0\0\0\0\0\0\0");
    }
    many.push(0);
    assert!(matches!(
        check_layout(&many, false),
        Err(ExrError::Layout(LayoutViolation::MissingName { part: 0 }))
    ));
}

/// `encode_all` writes every frame as `encode` would and combines the
/// results, so each single-part layout is available per part. Black-box
/// readers read the headers and the pixels of every variant, and of
/// single-part tiled files whose windows sit off the origin.
#[test]
fn contract_writer_layouts_are_read_by_black_box_readers() {
    use oxideav_openexr::{
        ColourLayout, EncodeOptions, ExrImage, ExrPixelFormat, Frame, LevelMode, decode,
        decode_all, encode, encode_all,
    };
    let image = |w: u32, h: u32, format: ExrPixelFormat, seed: u64| {
        let mut rng = Rng(seed);
        let n = (w * h) as usize * format.components();
        let samples: Vec<f32> = (0..n).map(|_| rng.sample().abs()).collect();
        ExrImage::from_f32(w, h, format, &samples).unwrap()
    };
    let placed = Box2i {
        x_min: -6,
        y_min: 4,
        x_max: 6,
        y_max: 12,
    };
    let frames = vec![
        Frame::new(
            image(13, 9, ExrPixelFormat::RgbaF32Le, 11)
                .with_data_window(placed)
                .unwrap(),
            0,
        )
        .with_name("beauty".to_string()),
        Frame::new(image(8, 6, ExrPixelFormat::RgbF32Le, 12), 1).with_name("left".to_string()),
        Frame::new(image(5, 17, ExrPixelFormat::GrayF32Le, 13), 2),
    ];
    let variants = [
        EncodeOptions::default(),
        EncodeOptions::default()
            .with_tile_size(4)
            .with_compression(Compression::Piz),
        EncodeOptions::default()
            .with_tile_size(5)
            .with_levels(LevelMode::Mipmap),
        EncodeOptions::default()
            .with_tile_size(4)
            .with_levels(LevelMode::Ripmap)
            .with_line_order(LineOrder::DecreasingY)
            .with_compression(Compression::Rle),
        EncodeOptions::default()
            .with_line_order(LineOrder::DecreasingY)
            .with_compression(Compression::Zips),
    ];
    for (v, opts) in variants.iter().enumerate() {
        let label = format!("encode_all variant {v}");
        let bytes = encode_all(&frames, opts).unwrap();
        let layout = check_layout(&bytes, true).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(layout.parts.len(), 3, "{label}");
        assert_eq!(layout.parts[2].name.as_deref(), Some("part2"), "{label}");
        let ours = decode_all(&bytes).unwrap();
        for (f, want) in ours.iter().zip(&frames) {
            assert_eq!(f.image.planes, want.image.planes, "{label}");
            assert_eq!(f.image.data_window, want.image.data_window, "{label}");
        }
        // combine(split) restores it, and each part decodes alone.
        let singles = split_parts(&bytes).unwrap();
        assert_eq!(combine_parts(&singles).unwrap(), bytes, "{label}");
        for (single, want) in singles.iter().zip(&frames) {
            assert_eq!(decode(single).unwrap().planes, want.image.planes, "{label}");
            header_reader_accepts(single, &label);
        }
        header_reader_accepts(&bytes, &label);
        if let Some(copy) = pixel_reader_copy(&bytes, &label) {
            let theirs = decode_all(&copy).unwrap_or_else(|e| panic!("{label} copy: {e}"));
            assert_eq!(theirs.len(), 3, "{label}");
            for (f, want) in theirs.iter().zip(&frames) {
                assert_eq!(f.image.planes, want.image.planes, "{label} copy");
                assert_eq!(f.image.data_window, want.image.data_window, "{label} copy");
            }
        }
    }

    // Luminance/chroma parts (sub-sampled chroma, scanline).
    let luma = EncodeOptions::default().with_colour(ColourLayout::LumaChroma);
    let even = vec![
        Frame::new(image(8, 6, ExrPixelFormat::RgbF32Le, 21), 0),
        Frame::new(image(4, 10, ExrPixelFormat::RgbaF32Le, 22), 1),
    ];
    let bytes = encode_all(&even, &luma).unwrap();
    check_layout(&bytes, true).unwrap();
    assert_eq!(decode_all(&bytes).unwrap().len(), 2);
    header_reader_accepts(&bytes, "luminance/chroma parts");
    assert!(pixel_reader_copy(&bytes, "luminance/chroma parts").is_none_or(|c| !c.is_empty()));

    // Single-part tiled files off the origin.
    for levels in [LevelMode::One, LevelMode::Mipmap, LevelMode::Ripmap] {
        let label = format!("tiled {levels:?} off origin");
        let opts = EncodeOptions::default()
            .with_tile_size(4)
            .with_levels(levels);
        let bytes = encode(&frames[0].image, &opts).unwrap();
        assert_eq!(parse_header(&bytes).unwrap().version.raw, 2 | 0x200);
        assert_eq!(decode(&bytes).unwrap(), frames[0].image, "{label}");
        header_reader_accepts(&bytes, &label);
        if let Some(copy) = pixel_reader_copy(&bytes, &label) {
            // (`pixel_reader_copy` hands a one-part file back as a
            // one-part multi-part file.)
            let theirs = decode_all(&copy).unwrap_or_else(|e| panic!("{label} copy: {e}"));
            assert_eq!(theirs[0].image.planes, frames[0].image.planes, "{label}");
            assert_eq!(theirs[0].image.data_window, placed, "{label}");
        }
    }
}
