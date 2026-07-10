//! DECREASING_Y physical chunk storage through the LINEAR-SCAN
//! readers.
//!
//! The offset-table-driven flat readers visit chunks in canonical
//! table order, so physical storage order never matters to them. The
//! multi-part readers instead scan chunks linearly for robustness
//! against zero-filled offset tables — which makes them sensitive to
//! physical storage order unless each chunk's payload is scattered (or
//! buffered and reassembled) by its self-described coordinates. Flat
//! multi-part parts scatter by row/tile; deep TILED parts buffer
//! per-tile slabs; deep SCANLINE parts accumulate variable-length
//! sample lists, which this suite pins as order-robust: a legal
//! DECREASING_Y-stored file (chunks bottom-first, offset tables still
//! canonically keyed) must decode identically to its top-first
//! counterpart.

use oxideav_openexr::{
    encode_exr_multipart_deep_scanline, parse_exr_deep_multipart, parse_exr_multipart_mixed,
    Channel, Compression, MultipartDeepScanlinePart, PixelType,
};

fn channels() -> Vec<Channel> {
    vec![
        Channel {
            name: "A".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        },
        Channel {
            name: "Z".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        },
    ]
}

/// Deep part fixture: 4×6, sample counts cycling 0..=3, sample values
/// encode their (pixel, sample, channel, part) identity so any
/// misordering is caught.
fn part_data(w: u32, h: u32, part_tag: f32) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let n = (w * h) as usize;
    let spp: Vec<u32> = (0..n).map(|i| (i as u32) % 4).collect();
    let total: usize = spp.iter().map(|&v| v as usize).sum();
    let mut a = Vec::with_capacity(total);
    let mut z = Vec::with_capacity(total);
    let mut s_idx = 0usize;
    for (px, &cnt) in spp.iter().enumerate() {
        for s in 0..cnt {
            a.push(part_tag + px as f32 + s as f32 / 8.0);
            z.push(-(part_tag + px as f32 + s as f32 / 8.0));
            s_idx += 1;
        }
    }
    assert_eq!(s_idx, total);
    (spp, a, z)
}

/// Byte position just past the multi-part header chain's terminating
/// second NUL. Also returns the byte offset of each part's `lineOrder`
/// value byte.
fn multipart_header_end(file: &[u8]) -> (usize, Vec<usize>) {
    let mut p = 8usize;
    let mut line_order_offsets = Vec::new();
    loop {
        if file[p] == 0 {
            return (p + 1, line_order_offsets);
        }
        // One part header: attributes until a lone NUL.
        loop {
            if file[p] == 0 {
                p += 1;
                break;
            }
            let nend = p + file[p..].iter().position(|&b| b == 0).unwrap();
            let name = &file[p..nend];
            p = nend + 1;
            let tend = p + file[p..].iter().position(|&b| b == 0).unwrap();
            p = tend + 1;
            let size = i32::from_le_bytes(file[p..p + 4].try_into().unwrap()) as usize;
            p += 4;
            if name == b"lineOrder" {
                line_order_offsets.push(p);
            }
            p += size;
        }
    }
}

/// Split the chunk region of a multi-part deep scanline file into
/// records `(part, y, bytes)`.
fn split_deep_chunks(file: &[u8], chunk_start: usize, n: usize) -> Vec<(i32, i32, Vec<u8>)> {
    let mut out = Vec::with_capacity(n);
    let mut p = chunk_start;
    for _ in 0..n {
        let part = i32::from_le_bytes(file[p..p + 4].try_into().unwrap());
        let y = i32::from_le_bytes(file[p + 4..p + 8].try_into().unwrap());
        let pt = u64::from_le_bytes(file[p + 8..p + 16].try_into().unwrap()) as usize;
        let pd = u64::from_le_bytes(file[p + 16..p + 24].try_into().unwrap()) as usize;
        let end = p + 32 + pt + pd;
        out.push((part, y, file[p..end].to_vec()));
        p = end;
    }
    assert_eq!(p, file.len(), "trailing bytes after last chunk");
    out
}

/// Rebuild the file with the chunks physically stored bottom-first
/// (per part, y descending; part order preserved within equal y),
/// offset tables rewritten with canonical keying, and every part's
/// lineOrder attribute set to DECREASING_Y.
fn make_decreasing(file: &[u8], chunk_counts: &[usize], block_h: i32) -> Vec<u8> {
    let (tables_start, lo_offs) = multipart_header_end(file);
    let total: usize = chunk_counts.iter().sum();
    let chunk_start = tables_start + total * 8;
    let chunks = split_deep_chunks(file, chunk_start, total);

    let mut storage: Vec<&(i32, i32, Vec<u8>)> = chunks.iter().collect();
    // Bottom-first: sort by y descending (stable keeps part order).
    storage.sort_by_key(|c| std::cmp::Reverse(c.1));

    let mut out = file[..tables_start].to_vec();
    for &off in &lo_offs {
        out[off] = 1; // DECREASING_Y
    }
    // Placeholder tables, then chunks; record placement.
    out.resize(tables_start + total * 8, 0);
    let mut placed: Vec<(i32, i32, u64)> = Vec::with_capacity(total);
    for c in &storage {
        placed.push((c.0, c.1, out.len() as u64));
        out.extend_from_slice(&c.2);
    }
    // Canonical table keying: part-major, entry i of part p points at
    // part p's block with y = i * block_h.
    let mut table_pos = tables_start;
    for (p_idx, &cc) in chunk_counts.iter().enumerate() {
        for i in 0..cc {
            let want_y = (i as i32) * block_h;
            let off = placed
                .iter()
                .find(|&&(pp, yy, _)| pp == p_idx as i32 && yy == want_y)
                .map(|&(_, _, o)| o)
                .expect("canonical chunk missing");
            out[table_pos..table_pos + 8].copy_from_slice(&off.to_le_bytes());
            table_pos += 8;
        }
    }
    out
}

#[test]
fn deep_multipart_decreasing_storage_decodes_identically() {
    let (w, h) = (4u32, 6u32);
    let (spp0, a0, z0) = part_data(w, h, 100.0);
    let (spp1, a1, z1) = part_data(w, h, 500.0);
    let file = encode_exr_multipart_deep_scanline(&[
        MultipartDeepScanlinePart {
            name: "front".to_string(),
            width: w,
            height: h,
            channels: channels(),
            samples_per_pixel: &spp0,
            channel_samples: vec![&a0, &z0],
            compression: Compression::None,
        },
        MultipartDeepScanlinePart {
            name: "back".to_string(),
            width: w,
            height: h,
            channels: channels(),
            samples_per_pixel: &spp1,
            channel_samples: vec![&a1, &z1],
            compression: Compression::None,
        },
    ])
    .unwrap();

    let baseline = parse_exr_deep_multipart(&file).unwrap();
    // NONE compression → 1 scanline per chunk → h chunks per part.
    let dec = make_decreasing(&file, &[h as usize, h as usize], 1);
    let got = parse_exr_deep_multipart(&dec).unwrap();

    assert_eq!(got.len(), baseline.len());
    for (g, b) in got.iter().zip(baseline.iter()) {
        assert_eq!(g.name, b.name);
        assert_eq!(g.samples_per_pixel, b.samples_per_pixel, "part {}", g.name);
        assert_eq!(
            g.channel_samples, b.channel_samples,
            "part {} sample lists must be assembled in canonical pixel-scan \
             order regardless of physical chunk storage order",
            g.name
        );
    }

    // The mixed multi-part reader walks the same wire layout for
    // `deepscanline` parts; it must be order-robust too.
    let mixed_base = parse_exr_multipart_mixed(&file).unwrap();
    let mixed_dec = parse_exr_multipart_mixed(&dec).unwrap();
    for (g, b) in mixed_dec.iter().zip(mixed_base.iter()) {
        let g = g.deep_scanline().expect("deep scanline part");
        let b = b.deep_scanline().expect("deep scanline part");
        assert_eq!(g.samples_per_pixel, b.samples_per_pixel, "part {}", g.name);
        assert_eq!(g.channel_samples, b.channel_samples, "part {}", g.name);
    }
}
