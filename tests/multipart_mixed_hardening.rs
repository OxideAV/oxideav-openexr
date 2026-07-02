//! Round-382 adversarial hardening for `parse_exr_multipart_mixed`.
//!
//! Decode contract: every byte slice returns `Ok` or `Err` — never a
//! panic, debug-build integer overflow, out-of-bounds index, or an
//! attacker-claimed allocation the input can't back. These tests attack
//! the mixed multi-part reader (with emphasis on the round-382
//! multi-level deep tiled path) via systematic truncation, byte
//! flipping, targeted header surgery, and hostile 64-bit chunk-size
//! fields.
#![allow(clippy::type_complexity)]

use oxideav_openexr::{
    encode_exr_multipart_mixed, parse_exr_multipart_mixed, Channel, Compression,
    DeepMipmapTiledLevelInput, MultipartMixedPart, PixelType,
};

fn channels_rgba_float() -> Vec<Channel> {
    ["A", "B", "G", "R"]
        .iter()
        .map(|n| Channel {
            name: n.to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        })
        .collect()
}

fn build_level(w: u32, h: u32) -> (Vec<u32>, [Vec<f32>; 4]) {
    let pixels = (w * h) as usize;
    let spp: Vec<u32> = (0..pixels).map(|i| (i as u32) % 4).collect();
    let total: usize = spp.iter().sum::<u32>() as usize;
    let mk = |scale: f32| -> Vec<f32> { (0..total).map(|i| (i as f32) * scale).collect() };
    (spp, [mk(0.05), mk(0.1), mk(0.15), mk(0.2)])
}

/// A small mixed fixture: one flat scanline part + one deep MIPMAP tiled
/// part (16×16, tile 8×8 → levels 16/8/4/2/1 → 4+1+1+1+1 = 8 deep
/// chunks) — 9 chunks total with the scanline part's single ZIPS block
/// count varying by height/block math.
fn fixture() -> Vec<u8> {
    let w = 16u32;
    let h = 16u32;
    let pixels = (w * h) as usize;
    let flat: Vec<f32> = (0..pixels).map(|i| (i as f32) * 0.01).collect();
    let levels: Vec<(u32, u32, Vec<u32>, [Vec<f32>; 4])> = (0..5u32)
        .map(|l| {
            let lw = (w >> l).max(1);
            let lh = (h >> l).max(1);
            let (spp, planes) = build_level(lw, lh);
            (lw, lh, spp, planes)
        })
        .collect();
    let pyramid: Vec<DeepMipmapTiledLevelInput> = levels
        .iter()
        .map(|(lw, lh, spp, planes)| DeepMipmapTiledLevelInput {
            width: *lw,
            height: *lh,
            samples_per_pixel: spp,
            channel_samples: vec![&planes[0], &planes[1], &planes[2], &planes[3]],
        })
        .collect();
    encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "flat".to_string(),
            width: w,
            height: h,
            channels: channels_rgba_float(),
            planes: vec![&flat, &flat, &flat, &flat],
            compression: Compression::Zips,
        },
        MultipartMixedPart::DeepTiledMipmap {
            name: "dmip".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: channels_rgba_float(),
            pyramid,
            compression: Compression::Zips,
        },
    ])
    .unwrap()
}

/// Every truncated prefix must parse to `Err` (or, for the full length,
/// `Ok`) without panicking.
#[test]
fn truncated_prefixes_never_panic() {
    let bytes = fixture();
    assert!(parse_exr_multipart_mixed(&bytes).is_ok());
    for len in (0..bytes.len()).step_by(3) {
        let _ = parse_exr_multipart_mixed(&bytes[..len]);
    }
    // The exact one-byte-short case too.
    let _ = parse_exr_multipart_mixed(&bytes[..bytes.len() - 1]);
}

/// Flipping any single byte to 0xFF (and to 0x00) must never panic —
/// the result may be `Ok` (benign flip) or `Err`, but the decoder must
/// stay memory-safe and overflow-free.
#[test]
fn single_byte_corruption_never_panics() {
    let bytes = fixture();
    for pos in 0..bytes.len() {
        let mut m = bytes.clone();
        m[pos] = 0xFF;
        let _ = parse_exr_multipart_mixed(&m);
        m[pos] = 0x00;
        let _ = parse_exr_multipart_mixed(&m);
    }
}

/// Locate the byte offset of an attribute's value payload in the raw
/// header bytes: finds `name\0type\0` and returns the offset just past
/// the following i32 length field.
fn attr_value_offset(bytes: &[u8], name: &str, type_name: &str) -> Option<usize> {
    let mut pat = Vec::new();
    pat.extend_from_slice(name.as_bytes());
    pat.push(0);
    pat.extend_from_slice(type_name.as_bytes());
    pat.push(0);
    bytes
        .windows(pat.len())
        .position(|w| w == pat)
        .map(|p| p + pat.len() + 4)
}

/// A chunkCount that disagrees with the level-grid math must be
/// rejected with an error (not mis-walk the chunk list).
#[test]
fn corrupt_chunk_count_is_rejected() {
    let bytes = fixture();
    let off = attr_value_offset(&bytes, "chunkCount", "int").expect("chunkCount attr");
    for bad in [0i32, 1, 999, -5] {
        let mut m = bytes.clone();
        m[off..off + 4].copy_from_slice(&bad.to_le_bytes());
        assert!(
            parse_exr_multipart_mixed(&m).is_err(),
            "chunkCount={bad} must be rejected"
        );
    }
}

/// A tiledesc level_mode outside 0/1/2 must be rejected.
#[test]
fn corrupt_level_mode_is_rejected() {
    let bytes = fixture();
    // The deep part's tiledesc: 4-byte x, 4-byte y, 1 mode byte.
    let off = attr_value_offset(&bytes, "tiles", "tiledesc").expect("tiles attr");
    let mode_off = off + 8;
    let mut m = bytes.clone();
    m[mode_off] = 3; // unknown level mode
    assert!(
        parse_exr_multipart_mixed(&m).is_err(),
        "level_mode=3 must be rejected"
    );
    // Demoting the MIPMAP part to ONE_LEVEL makes the declared
    // chunkCount disagree with the ONE_LEVEL grid math → reject.
    let mut m = bytes.clone();
    m[mode_off] = 0;
    assert!(
        parse_exr_multipart_mixed(&m).is_err(),
        "MIPMAP part relabelled ONE_LEVEL must fail the chunk-count check"
    );
}

/// Find the absolute offset of the chunk area by locating the offset
/// table: the first table entry is a u64 equal to its own position plus
/// `8 * total_chunks`.
fn find_chunks_start(bytes: &[u8], total_chunks: usize) -> Option<usize> {
    for p in 0..bytes.len().saturating_sub(8) {
        let v = u64::from_le_bytes(bytes[p..p + 8].try_into().unwrap()) as usize;
        if v == p + 8 * total_chunks {
            return Some(v);
        }
    }
    None
}

/// A deep chunk whose u64 packed-size fields are hostile (near
/// `u64::MAX`) must yield `Err`, not a debug add-overflow panic.
#[test]
fn hostile_u64_chunk_sizes_are_rejected() {
    let bytes = fixture();
    // 1 scanline chunk (ZIPS: 16 rows = 16 blocks? ZIPS = 1 line/block,
    // so 16 chunks) + 8 deep chunks. Derive instead of hard-coding:
    // scanline ZIPS chunks = height = 16; deep chunks = 8.
    let total_chunks = 16 + 8;
    let chunks_start = find_chunks_start(&bytes, total_chunks).expect("offset table");
    // Walk to the first deep chunk (part_number == 1). Scanline chunk:
    // i32 part + i32 y + i32 size + payload.
    let mut pos = chunks_start;
    loop {
        let part = i32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        if part == 1 {
            break;
        }
        let size = i32::from_le_bytes(bytes[pos + 8..pos + 12].try_into().unwrap());
        pos += 12 + size as usize;
    }
    // Deep tiled chunk header: i32 part + 4×i32 coords, then u64
    // packed_table at +20, u64 packed_data at +28.
    for (field_off, label) in [(20usize, "packed_table"), (28usize, "packed_data")] {
        let mut m = bytes.clone();
        m[pos + field_off..pos + field_off + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(
            parse_exr_multipart_mixed(&m).is_err(),
            "hostile {label} must be rejected"
        );
        // A value that overflows only after adding the scan position.
        let sneaky = (u64::MAX / 2).to_le_bytes();
        let mut m = bytes.clone();
        m[pos + field_off..pos + field_off + 8].copy_from_slice(&sneaky);
        assert!(
            parse_exr_multipart_mixed(&m).is_err(),
            "half-max {label} must be rejected"
        );
    }
}

/// Duplicate deep tile chunks (same tile re-sent in place of another)
/// must be rejected by the duplicate-tile guard.
#[test]
fn duplicated_deep_tile_chunk_is_rejected() {
    let bytes = fixture();
    let total_chunks = 16 + 8;
    let chunks_start = find_chunks_start(&bytes, total_chunks).expect("offset table");
    // Find the first two deep chunks and overwrite the second one's
    // (tx, ty, lvlx, lvly) with the first one's coordinates.
    let mut pos = chunks_start;
    let mut deep_positions = Vec::new();
    while deep_positions.len() < 2 && pos + 12 <= bytes.len() {
        let part = i32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        if part == 1 {
            deep_positions.push(pos);
            let pt = u64::from_le_bytes(bytes[pos + 20..pos + 28].try_into().unwrap()) as usize;
            let pd = u64::from_le_bytes(bytes[pos + 28..pos + 36].try_into().unwrap()) as usize;
            pos += 44 + pt + pd;
        } else {
            let size = i32::from_le_bytes(bytes[pos + 8..pos + 12].try_into().unwrap());
            pos += 12 + size as usize;
        }
    }
    let (first, second) = (deep_positions[0], deep_positions[1]);
    let mut m = bytes.clone();
    let coords: Vec<u8> = bytes[first + 4..first + 20].to_vec();
    m[second + 4..second + 20].copy_from_slice(&coords);
    assert!(
        parse_exr_multipart_mixed(&m).is_err(),
        "re-sent deep tile coordinates must be rejected"
    );
}
