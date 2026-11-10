#![no_main]

//! Coverage-guided fuzz harness for the file-layout module
//! (`oxideav_openexr::layout`): version flags, the header list, chunk
//! counts from the part geometry, offset tables, part-number prefixes,
//! and the byte-level split / combine repack.
//!
//! Contract under test: every byte slice produces `Ok` or `Err` — no
//! panic, no debug-mode overflow, no out-of-bounds index, and nothing
//! sized from a header field before the file is known to hold it (a
//! huge `chunkCount`, data window, tile grid or part count costs
//! nothing). On top of that, for every input the strict check accepts:
//!
//! * `split_parts` succeeds and each part passes the lenient check;
//! * `combine_parts(split_parts(f))` succeeds, passes the strict check
//!   and is a fixed point of split-then-combine.
//!
//! Modes (first byte):
//!
//!   0. Raw — the fuzz bytes are the file.
//!   1. Table overlay — fuzz bytes spliced over the offset tables and
//!      the start of the chunk region of a writer-built mixed file
//!      (flat scanline + flat tiles + deep scanline + deep mip-mapped
//!      tiles).
//!   2. Header patch — a short fuzz run written at a fuzz-chosen
//!      position inside the header region of that file, so attribute
//!      values (windows, tile descriptions, names, types, counts) and
//!      the version field mutate while the rest stays parseable.

use libfuzzer_sys::fuzz_target;
use oxideav_openexr::deep::DeepMipmapTiledLevelInput;
use oxideav_openexr::multipart_mixed_encoder::{MultipartMixedPart, encode_exr_multipart_mixed};
use oxideav_openexr::types::{Channel, Compression, PixelType};
use oxideav_openexr::{
    DecodeOptions, FileLayout, check_layout, combine_parts, decode_all_with,
    parse_exr_multipart_mixed, split_parts,
};

fn channels() -> Vec<Channel> {
    vec![
        Channel::new("A".to_string(), PixelType::Half, false, 1, 1),
        Channel::new("Z".to_string(), PixelType::Float, false, 1, 1),
    ]
}

fn base_file(compression: Compression) -> Option<Vec<u8>> {
    let (w, h) = (7u32, 9u32);
    let pixels = (w * h) as usize;
    let flat: Vec<f32> = (0..pixels).map(|i| i as f32 * 0.125).collect();
    let spp: Vec<u32> = (0..pixels).map(|i| (i as u32) % 3).collect();
    let total: usize = spp.iter().map(|&v| v as usize).sum();
    let deep: Vec<f32> = (0..total).map(|i| i as f32 * 0.5).collect();
    type Level = (u32, u32, Vec<u32>, Vec<f32>);
    let levels: Vec<Level> = (0..4u32)
        .map(|l| {
            let (lw, lh) = ((w >> l).max(1), (h >> l).max(1));
            let n = (lw * lh) as usize;
            let spp: Vec<u32> = (0..n).map(|i| (i as u32) % 2).collect();
            let total: usize = spp.iter().map(|&v| v as usize).sum();
            (lw, lh, spp, (0..total).map(|i| i as f32).collect())
        })
        .collect();
    let pyramid: Vec<DeepMipmapTiledLevelInput> = levels
        .iter()
        .map(|(lw, lh, spp, s)| DeepMipmapTiledLevelInput::new(*lw, *lh, spp, vec![s, s]))
        .collect();
    let deep_compression = match compression {
        Compression::None => Compression::None,
        Compression::Rle => Compression::Rle,
        _ => Compression::Zips,
    };
    encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "scan".to_string(),
            width: w,
            height: h,
            channels: channels(),
            planes: vec![&flat, &flat],
            compression,
        },
        MultipartMixedPart::Tiled {
            name: "tiles".to_string(),
            width: w,
            height: h,
            tile_x: 3,
            tile_y: 4,
            channels: channels(),
            planes: vec![&flat, &flat],
            compression,
        },
        MultipartMixedPart::DeepScanline {
            name: "deep".to_string(),
            width: w,
            height: h,
            channels: channels(),
            samples_per_pixel: &spp,
            channel_samples: vec![&deep, &deep],
            compression: deep_compression,
        },
        MultipartMixedPart::DeepTiledMipmap {
            name: "dmip".to_string(),
            tile_x: 4,
            tile_y: 3,
            channels: channels(),
            pyramid,
            compression: deep_compression,
        },
    ])
    .ok()
}

fn exercise(file: &[u8]) {
    let _ = FileLayout::read_headers(file);
    let lenient = check_layout(file, false);
    let strict = check_layout(file, true);
    let split = split_parts(file);
    let _ = parse_exr_multipart_mixed(file);
    let _ = decode_all_with(file, &DecodeOptions::default().with_strict(true));
    if lenient.is_err() {
        // The strict check never accepts what the lenient one refuses.
        assert!(strict.is_err());
        return;
    }
    if strict.is_err() {
        return;
    }
    let parts = split.expect("a strictly regular file splits");
    for p in &parts {
        check_layout(p, false).expect("split part is readable");
    }
    let combined = combine_parts(&parts).expect("split parts combine");
    check_layout(&combined, true).expect("combined file is regular");
    // Combining names an unnamed single-part input, so the fixed point
    // is reached one step later: the combined file splits and
    // recombines into itself.
    let again = split_parts(&combined).expect("combined file splits");
    assert_eq!(again.len(), parts.len());
    assert_eq!(combine_parts(&again).expect("recombine"), combined);
}

fuzz_target!(|data: &[u8]| {
    let Some((&mode, rest)) = data.split_first() else {
        return;
    };
    if mode % 3 == 0 {
        exercise(rest);
        return;
    }
    static BASES: std::sync::OnceLock<Vec<Option<Vec<u8>>>> = std::sync::OnceLock::new();
    let bases = BASES.get_or_init(|| {
        [Compression::None, Compression::Rle, Compression::Zip]
            .into_iter()
            .map(base_file)
            .collect()
    });
    let Some((&select, rest)) = rest.split_first() else {
        return;
    };
    let Some(mut file) = bases[(select % 3) as usize].clone() else {
        return;
    };
    let Ok(layout) = FileLayout::read(&file) else {
        return;
    };
    if mode % 3 == 1 {
        let region = &mut file[layout.tables_start..];
        let take = rest.len().min(region.len());
        region[..take].copy_from_slice(&rest[..take]);
    } else {
        let Some((at, patch)) = rest.split_first_chunk::<2>() else {
            return;
        };
        // Position inside the header region (version field included).
        let at = 4 + u16::from_le_bytes(*at) as usize % (layout.tables_start - 4);
        let take = patch.len().min(8).min(file.len() - at);
        file[at..at + take].copy_from_slice(&patch[..take]);
    }
    exercise(&file);
});
