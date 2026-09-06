#![no_main]

//! Coverage-guided fuzz harness for the compressed-chunk decoders in
//! isolation: PIZ (bitmap / range LUT / wavelet / canonical Huffman),
//! DWAA/DWAB (eleven-slot header / rule block / sub-streams / AC run
//! coding / DC delta / IDCT), B44/B44A (4×4 block unpacking), PXR24
//! (byte-plane delta) and the ZIP/RLE pipelines — reached through the
//! hidden `chunk_api::decode_scanline_chunk` entry point rather than a
//! whole file, so every fuzz byte lands in chunk arithmetic instead of
//! the header parser.
//!
//! Contract under test: every byte slice produces `Ok(..)` or
//! `Err(..)`. Panics, debug-mode integer overflows, index-out-of-bounds
//! and attacker-claimed-length allocations are bugs.
//!
//! Input layout: `[scheme][channel set][width][block row][mode] payload…`.
//! `mode` bit 0 selects raw mode (the payload bytes are the chunk) or
//! overlay mode (a valid chunk is built by the crate's own scanline
//! writer for the chosen scheme and shape, then the payload bytes are
//! spliced over it at an offset taken from the next two bytes) — the
//! overlay reaches the deep decode arithmetic without the fuzzer having
//! to rediscover a well-formed Huffman table or DWA header first.

use libfuzzer_sys::fuzz_target;
use oxideav_openexr::chunk_api::decode_scanline_chunk;
use oxideav_openexr::{
    encode_exr_scanline, parse_header, Attribute, AttributeValue, Box2i, Channel, Compression,
    LineOrder, PixelType,
};

fn ch(name: &str, pt: PixelType, p_linear: bool, sx: i32, sy: i32) -> Channel {
    Channel {
        name: name.to_string(),
        pixel_type: pt,
        p_linear,
        x_sampling: sx,
        y_sampling: sy,
    }
}

/// Channel sets, alphabetical as the file layout requires.
fn channel_set(sel: u8) -> Vec<Channel> {
    match sel % 7 {
        0 => vec![
            ch("A", PixelType::Half, false, 1, 1),
            ch("B", PixelType::Half, false, 1, 1),
            ch("G", PixelType::Half, false, 1, 1),
            ch("R", PixelType::Half, false, 1, 1),
        ],
        1 => vec![
            ch("B", PixelType::Float, false, 1, 1),
            ch("G", PixelType::Float, false, 1, 1),
            ch("R", PixelType::Float, false, 1, 1),
        ],
        2 => vec![
            ch("BY", PixelType::Half, false, 2, 2),
            ch("RY", PixelType::Half, false, 2, 2),
            ch("Y", PixelType::Half, false, 1, 1),
        ],
        3 => vec![
            ch("A", PixelType::Float, false, 1, 1),
            ch("B", PixelType::Half, true, 1, 1),
            ch("G", PixelType::Half, true, 1, 1),
            ch("R", PixelType::Half, true, 1, 1),
            ch("Z", PixelType::Float, false, 1, 1),
            ch("id", PixelType::Uint, false, 1, 1),
        ],
        4 => vec![ch("Y", PixelType::Half, false, 1, 1)],
        5 => vec![
            ch("BY", PixelType::Float, false, 3, 1),
            ch("RY", PixelType::Float, false, 1, 3),
            ch("Y", PixelType::Float, false, 1, 1),
        ],
        _ => vec![
            ch("B", PixelType::Half, false, 1, 1),
            ch("G", PixelType::Uint, false, 1, 1),
            ch("R", PixelType::Half, false, 1, 1),
        ],
    }
}

fn attrs(w: u32, h: u32, chs: &[Channel], compression: Compression) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (w - 1) as i32,
        y_max: (h - 1) as i32,
    };
    vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(chs.to_vec()),
        },
        Attribute {
            name: "compression".to_string(),
            value: AttributeValue::Compression(compression),
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
    ]
}

/// A valid chunk for the shape, cut out of a one-chunk file written by
/// the crate's scanline encoder (header | one offset | y | size | payload).
fn valid_chunk(compression: Compression, chs: &[Channel], w: u32, lines: u32) -> Option<Vec<u8>> {
    let planes: Vec<Vec<f32>> = chs
        .iter()
        .enumerate()
        .map(|(c, ch)| {
            let pw = w.div_ceil(ch.x_sampling as u32) as usize;
            let ph = lines.div_ceil(ch.y_sampling as u32) as usize;
            (0..pw * ph)
                .map(|i| ((i * 7 + c * 13) % 61) as f32 / 30.0 - 0.5 + (i / pw) as f32 * 0.01)
                .collect()
        })
        .collect();
    let refs: Vec<&[f32]> = planes.iter().map(|p| p.as_slice()).collect();
    let file = encode_exr_scanline(w, lines, chs, &refs, compression, attrs(w, lines, chs, compression))
        .ok()?;
    let hdr = parse_header(&file).ok()?;
    let start = hdr.end_offset + 8 + 8;
    file.get(start..).map(|p| p.to_vec())
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 5 {
        return;
    }
    let compression = Compression::from_byte(data[0] % 10).unwrap();
    let chs = channel_set(data[1]);
    let width = (data[2] as u32 % 48) + 1;
    let lines = compression.scanlines_per_block() as usize;
    let block_y0 = (data[3] as u32 % 4) * lines as u32;
    let mode = data[4];
    let rest = &data[5..];

    if mode & 1 == 0 {
        let _ = decode_scanline_chunk(compression, rest, &chs, width, block_y0, lines);
        return;
    }
    if rest.len() < 2 {
        return;
    }
    // Overlay mode: the valid chunk is always built at block row 0 (a
    // one-chunk file); the decode still runs at `block_y0` so the row
    // parity of sub-sampled channels is exercised against it too.
    let Some(mut chunk) = valid_chunk(compression, &chs, width, lines as u32) else {
        return;
    };
    let off = u16::from_le_bytes([rest[0], rest[1]]) as usize;
    let splice = &rest[2..];
    if !chunk.is_empty() {
        let off = off % chunk.len();
        let n = splice.len().min(chunk.len() - off);
        chunk[off..off + n].copy_from_slice(&splice[..n]);
    }
    let _ = decode_scanline_chunk(compression, &chunk, &chs, width, 0, lines);
    let _ = decode_scanline_chunk(compression, &chunk, &chs, width, block_y0, lines);
});
