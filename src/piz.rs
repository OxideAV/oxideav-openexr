//! PIZ compression (compression code 4, 32 scanlines per chunk) —
//! observer-spec `openexr-piz-dwa-observer-spec.md` §2.
//!
//! Lossless pipeline over the chunk's samples reinterpreted as 16-bit
//! code words (HALF = 1 word, FLOAT / UINT = 2 words):
//!
//! ```text
//! samples -> occupancy bitmap -> range-compaction LUT -> 2D wavelet
//!         -> Huffman payload (crate::huf) -> chunk
//! ```
//!
//! Chunk layout (§2.1): u16 `minNonZeroByte`, u16 `maxNonZeroByte`, the
//! inclusive bitmap byte run (absent when min > max — the all-zero
//! chunk), u32 Huffman byte count, Huffman payload. All little-endian.
//!
//! Within the word buffer, channel regions are laid out back to back in
//! header (name-sorted) order; a channel's region holds its chunk rows
//! top to bottom with each sample's `bytes_per_sample / 2` words
//! interleaved consecutively (component *j* of a sample at word offset
//! *j*, sample stride `B/2` words). The wavelet runs per channel per
//! component, and the choice between its 14-bit and modulo-16 variants
//! is recomputed from the bitmap (`maxValue < 16384`) — it is never
//! transmitted (§2.4).

use crate::error::{ExrError, Result};
use crate::types::Channel;

/// Shape of one compressed chunk: which channels contribute which rows.
/// Shared by the PIZ and DWA codecs (scanline blocks and tiles alike; a
/// tile is a chunk with `block_y0 = 0` and full 1×1 sampling).
pub(crate) struct ChunkShape<'a> {
    pub(crate) sorted_channels: &'a [Channel],
    /// Data-window width (full resolution; per-channel widths derive via
    /// x_sampling).
    pub(crate) width: u32,
    /// Top image row of this chunk relative to the data window.
    pub(crate) block_y0: u32,
    /// Image rows covered by this chunk.
    pub(crate) lines_in_block: usize,
}

pub(crate) struct ChannelExtent {
    /// Samples per row (sub-sampled width).
    pub(crate) nx: usize,
    /// Rows of this channel present in the chunk.
    pub(crate) ny: usize,
    /// 16-bit words per sample (1 for HALF, 2 for FLOAT / UINT).
    pub(crate) words_per_sample: usize,
}

impl ChunkShape<'_> {
    /// Per-channel extents within this chunk. A channel contributes a row
    /// only when the absolute scanline index is a multiple of its
    /// y_sampling (observer-spec §1 shared framing).
    pub(crate) fn extents(&self) -> Vec<ChannelExtent> {
        self.sorted_channels
            .iter()
            .map(|ch| {
                let ys = ch.y_sampling as u32;
                let ny = (0..self.lines_in_block as u32)
                    .filter(|&l| (self.block_y0 + l) % ys == 0)
                    .count();
                let nx = crate::decoder::subsampled_dim(self.width, ch.x_sampling as u32) as usize;
                ChannelExtent {
                    nx,
                    ny,
                    words_per_sample: ch.pixel_type.bytes_per_sample() / 2,
                }
            })
            .collect()
    }

    /// Does image row `line` (chunk-relative) include channel `ch`?
    #[inline]
    pub(crate) fn row_present(&self, ch: &Channel, line: usize) -> bool {
        (self.block_y0 + line as u32) % (ch.y_sampling as u32) == 0
    }
}

// ---------------------------------------------------------------------------
// Bitmap + range-compaction LUT (§2.2 / §2.3)
// ---------------------------------------------------------------------------

const BITMAP_BYTES: usize = 8192;

/// Forward LUT from an occupancy bitmap: renumbers present code words
/// (plus the always-present 0) contiguously from 0. Returns the LUT and
/// `maxValue` (the largest renumbered value).
fn forward_lut(bitmap: &[u8; BITMAP_BYTES]) -> (Vec<u16>, u16) {
    let mut lut = vec![0u16; 65536];
    let mut k = 0u32;
    for v in 0..65536usize {
        if v == 0 || (bitmap[v >> 3] & (1 << (v & 7))) != 0 {
            lut[v] = k as u16;
            k += 1;
        }
    }
    (lut, (k - 1) as u16)
}

/// Inverse LUT: maps renumbered values back to original code words.
fn inverse_lut(bitmap: &[u8; BITMAP_BYTES]) -> (Vec<u16>, u16) {
    let mut lut = vec![0u16; 65536];
    let mut k = 0u32;
    for v in 0..65536usize {
        if v == 0 || (bitmap[v >> 3] & (1 << (v & 7))) != 0 {
            lut[k as usize] = v as u16;
            k += 1;
        }
    }
    (lut, (k - 1) as u16)
}

// ---------------------------------------------------------------------------
// Wavelet (§2.4)
// ---------------------------------------------------------------------------

/// 14-bit variant, forward: mean (floor) and difference over signed
/// 16-bit interpretations.
#[inline]
fn wenc14(a: u16, b: u16) -> (u16, u16) {
    let (a, b) = (a as i16 as i32, b as i16 as i32);
    let l = (a + b) >> 1;
    let h = a - b;
    (l as u16, h as u16)
}

/// 14-bit variant, inverse: `a = l + ceil(h/2)` via `(h & 1) + (h >> 1)`.
#[inline]
fn wdec14(l: u16, h: u16) -> (u16, u16) {
    let (l, h) = (l as i16 as i32, h as i16 as i32);
    let a = l + (h & 1) + (h >> 1);
    let b = a - h;
    (a as u16, b as u16)
}

const MOD_OFFSET: i32 = 0x8000;
const MOD_MASK: i32 = 0xFFFF;

/// Modulo-2^16 variant, forward: offset the first operand, take the
/// logical-shift mean, and fold the difference's wraparound bit into the
/// mean via the conditional bias (§2.4).
#[inline]
fn wenc16(a: u16, b: u16) -> (u16, u16) {
    let ao = (a as i32 + MOD_OFFSET) & MOD_MASK;
    let mut l = (ao + b as i32) >> 1;
    let d = ao - b as i32;
    if d < 0 {
        l = (l + MOD_OFFSET) & MOD_MASK;
    }
    (l as u16, (d & MOD_MASK) as u16)
}

/// Modulo-2^16 variant, inverse.
#[inline]
fn wdec16(l: u16, h: u16) -> (u16, u16) {
    let b = (l as i32 - ((h as i32) >> 1)) & MOD_MASK;
    let a = (h as i32 + b - MOD_OFFSET) & MOD_MASK;
    (a as u16, b as u16)
}

/// The level steps for a channel of min-dimension `n`: p = 1, 2, 4, …
/// while 2p <= n.
fn level_steps(n: usize) -> Vec<usize> {
    let mut ps = Vec::new();
    let mut p = 1usize;
    while p * 2 <= n {
        ps.push(p);
        p *= 2;
    }
    ps
}

/// Forward hierarchical 2D wavelet over one component of one channel.
/// `base` is the word index of sample (0,0)'s component; `ox` / `oy` are
/// the word strides between horizontally / vertically adjacent samples.
fn wav_encode(
    buf: &mut [u16],
    base: usize,
    nx: usize,
    ny: usize,
    ox: usize,
    oy: usize,
    use16: bool,
) {
    // Monomorphize per variant so the step function inlines into the
    // innermost loop (a runtime fn pointer defeats inlining).
    if use16 {
        wav_encode_impl(buf, base, nx, ny, ox, oy, wenc16);
    } else {
        wav_encode_impl(buf, base, nx, ny, ox, oy, wenc14);
    }
}

fn wav_encode_impl<W: Fn(u16, u16) -> (u16, u16) + Copy>(
    buf: &mut [u16],
    base: usize,
    nx: usize,
    ny: usize,
    ox: usize,
    oy: usize,
    w: W,
) {
    for p in level_steps(nx.min(ny)) {
        let p2 = p * 2;
        let mut y = 0usize;
        while y + p2 <= ny {
            let mut x = 0usize;
            while x + p2 <= nx {
                let i00 = base + x * ox + y * oy;
                let i01 = i00 + p * ox;
                let i10 = i00 + p * oy;
                let i11 = i10 + p * ox;
                // Two horizontal steps, then two vertical steps.
                let (a, b) = w(buf[i00], buf[i01]);
                buf[i00] = a;
                buf[i01] = b;
                let (a, b) = w(buf[i10], buf[i11]);
                buf[i10] = a;
                buf[i11] = b;
                let (a, b) = w(buf[i00], buf[i10]);
                buf[i00] = a;
                buf[i10] = b;
                let (a, b) = w(buf[i01], buf[i11]);
                buf[i01] = a;
                buf[i11] = b;
                x += p2;
            }
            if nx & p != 0 {
                // Odd column: 1D vertical step on the leftover column.
                let i0 = base + x * ox + y * oy;
                let i1 = i0 + p * oy;
                let (a, b) = w(buf[i0], buf[i1]);
                buf[i0] = a;
                buf[i1] = b;
            }
            y += p2;
        }
        if ny & p != 0 {
            // Odd row: 1D horizontal steps across the leftover row.
            let mut x = 0usize;
            while x + p2 <= nx {
                let i0 = base + x * ox + y * oy;
                let i1 = i0 + p * ox;
                let (a, b) = w(buf[i0], buf[i1]);
                buf[i0] = a;
                buf[i1] = b;
                x += p2;
            }
        }
    }
}

/// Inverse wavelet: same traversal with the levels in reverse order and
/// the vertical pairs undone before the horizontal ones.
fn wav_decode(
    buf: &mut [u16],
    base: usize,
    nx: usize,
    ny: usize,
    ox: usize,
    oy: usize,
    use16: bool,
) {
    if use16 {
        wav_decode_impl(buf, base, nx, ny, ox, oy, wdec16);
    } else {
        wav_decode_impl(buf, base, nx, ny, ox, oy, wdec14);
    }
}

fn wav_decode_impl<W: Fn(u16, u16) -> (u16, u16) + Copy>(
    buf: &mut [u16],
    base: usize,
    nx: usize,
    ny: usize,
    ox: usize,
    oy: usize,
    w: W,
) {
    for &p in level_steps(nx.min(ny)).iter().rev() {
        let p2 = p * 2;
        let mut y = 0usize;
        while y + p2 <= ny {
            let mut x = 0usize;
            while x + p2 <= nx {
                let i00 = base + x * ox + y * oy;
                let i01 = i00 + p * ox;
                let i10 = i00 + p * oy;
                let i11 = i10 + p * ox;
                let (a, b) = w(buf[i00], buf[i10]);
                buf[i00] = a;
                buf[i10] = b;
                let (a, b) = w(buf[i01], buf[i11]);
                buf[i01] = a;
                buf[i11] = b;
                let (a, b) = w(buf[i00], buf[i01]);
                buf[i00] = a;
                buf[i01] = b;
                let (a, b) = w(buf[i10], buf[i11]);
                buf[i10] = a;
                buf[i11] = b;
                x += p2;
            }
            if nx & p != 0 {
                let i0 = base + x * ox + y * oy;
                let i1 = i0 + p * oy;
                let (a, b) = w(buf[i0], buf[i1]);
                buf[i0] = a;
                buf[i1] = b;
            }
            y += p2;
        }
        if ny & p != 0 {
            let mut x = 0usize;
            while x + p2 <= nx {
                let i0 = base + x * ox + y * oy;
                let i1 = i0 + p * ox;
                let (a, b) = w(buf[i0], buf[i1]);
                buf[i0] = a;
                buf[i1] = b;
                x += p2;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Native stream <-> planar word buffer
// ---------------------------------------------------------------------------

/// Gather the native interleaved chunk stream (rows top to bottom,
/// channels within a row in sorted order, little-endian samples) into
/// the back-to-back channel word regions.
fn gather_words(raw: &[u8], shape: &ChunkShape, extents: &[ChannelExtent]) -> Vec<u16> {
    let total_words = raw.len() / 2;
    let mut words = vec![0u16; total_words];
    // Region base per channel.
    let mut bases = Vec::with_capacity(extents.len());
    let mut b = 0usize;
    for e in extents {
        bases.push(b);
        b += e.nx * e.ny * e.words_per_sample;
    }
    let mut rp = 0usize; // byte cursor in the native stream
    let mut row_of_channel = vec![0usize; extents.len()];
    for line in 0..shape.lines_in_block {
        for (ci, ch) in shape.sorted_channels.iter().enumerate() {
            if !shape.row_present(ch, line) {
                continue;
            }
            let e = &extents[ci];
            let row_words = e.nx * e.words_per_sample;
            let dst = bases[ci] + row_of_channel[ci] * row_words;
            for (w, chunk) in words[dst..dst + row_words]
                .iter_mut()
                .zip(raw[rp..rp + row_words * 2].chunks_exact(2))
            {
                *w = u16::from_le_bytes([chunk[0], chunk[1]]);
            }
            rp += row_words * 2;
            row_of_channel[ci] += 1;
        }
    }
    debug_assert_eq!(rp, raw.len());
    words
}

/// Scatter the planar word buffer back into the native interleaved
/// stream.
fn scatter_words(
    words: &[u16],
    shape: &ChunkShape,
    extents: &[ChannelExtent],
    out_len: usize,
) -> Vec<u8> {
    let mut out = vec![0u8; out_len];
    let mut bases = Vec::with_capacity(extents.len());
    let mut b = 0usize;
    for e in extents {
        bases.push(b);
        b += e.nx * e.ny * e.words_per_sample;
    }
    let mut wp = 0usize;
    let mut row_of_channel = vec![0usize; extents.len()];
    for line in 0..shape.lines_in_block {
        for (ci, ch) in shape.sorted_channels.iter().enumerate() {
            if !shape.row_present(ch, line) {
                continue;
            }
            let e = &extents[ci];
            let row_words = e.nx * e.words_per_sample;
            let src = bases[ci] + row_of_channel[ci] * row_words;
            for (chunk, &w) in out[wp..wp + row_words * 2]
                .chunks_exact_mut(2)
                .zip(words[src..src + row_words].iter())
            {
                chunk.copy_from_slice(&w.to_le_bytes());
            }
            wp += row_words * 2;
            row_of_channel[ci] += 1;
        }
    }
    debug_assert_eq!(wp, out_len);
    out
}

// ---------------------------------------------------------------------------
// Chunk encode / decode
// ---------------------------------------------------------------------------

/// Compress one chunk's native byte stream into a PIZ payload. The
/// caller applies the shared raw fallback (store the raw bytes whenever
/// the result is not smaller).
pub(crate) fn piz_compress(raw: &[u8], shape: &ChunkShape) -> Result<Vec<u8>> {
    debug_assert_eq!(raw.len() % 2, 0);
    let extents = shape.extents();
    let mut words = gather_words(raw, shape, &extents);

    // Occupancy bitmap over every word; bit 0 forcibly cleared (code
    // word zero is implicit, §2.2).
    let mut bitmap = [0u8; BITMAP_BYTES];
    for &w in &words {
        bitmap[(w >> 3) as usize] |= 1 << (w & 7);
    }
    bitmap[0] &= !1;

    let (min_b, max_b) = {
        let mut min_b = BITMAP_BYTES - 1;
        let mut max_b = 0usize;
        for (i, &b) in bitmap.iter().enumerate() {
            if b != 0 {
                min_b = min_b.min(i);
                max_b = max_b.max(i);
            }
        }
        (min_b, max_b)
    };

    let (lut, max_value) = forward_lut(&bitmap);
    for w in words.iter_mut() {
        *w = lut[*w as usize];
    }

    let use16 = max_value >= 16384;
    let mut base = 0usize;
    for e in &extents {
        for j in 0..e.words_per_sample {
            wav_encode(
                &mut words,
                base + j,
                e.nx,
                e.ny,
                e.words_per_sample,
                e.words_per_sample * e.nx,
                use16,
            );
        }
        base += e.nx * e.ny * e.words_per_sample;
    }

    let huf = crate::huf::huf_compress(&words)?;

    let bitmap_run = if min_b <= max_b { max_b - min_b + 1 } else { 0 };
    let mut out = Vec::with_capacity(8 + bitmap_run + huf.len());
    out.extend_from_slice(&(min_b as u16).to_le_bytes());
    out.extend_from_slice(&(max_b as u16).to_le_bytes());
    if bitmap_run > 0 {
        out.extend_from_slice(&bitmap[min_b..=max_b]);
    }
    out.extend_from_slice(&(huf.len() as u32).to_le_bytes());
    out.extend_from_slice(&huf);
    Ok(out)
}

/// Decode one PIZ chunk payload into the native interleaved byte stream.
/// The shared raw fallback (`payload.len() == uncompressed_size`) is
/// handled here.
pub(crate) fn decode_piz_payload(
    payload: &[u8],
    shape: &ChunkShape,
    uncompressed_size: usize,
) -> Result<Vec<u8>> {
    if payload.len() == uncompressed_size {
        return Ok(payload.to_vec());
    }
    if uncompressed_size % 2 != 0 {
        return Err(ExrError::invalid(format!(
            "PIZ: odd uncompressed chunk size {uncompressed_size}"
        )));
    }
    if payload.len() < 4 {
        return Err(ExrError::invalid("PIZ: chunk header truncated".to_string()));
    }
    let min_b = u16::from_le_bytes(payload[0..2].try_into().unwrap()) as usize;
    let max_b = u16::from_le_bytes(payload[2..4].try_into().unwrap()) as usize;
    let mut p = 4usize;
    let mut bitmap = [0u8; BITMAP_BYTES];
    if min_b <= max_b {
        // §2.1: maxNonZeroByte must stay inside the 8192-byte bitmap.
        if max_b >= BITMAP_BYTES {
            return Err(ExrError::invalid(format!(
                "PIZ: bitmap byte range {min_b}..{max_b} exceeds 8192"
            )));
        }
        let run = max_b - min_b + 1;
        let src = payload
            .get(p..p + run)
            .ok_or_else(|| ExrError::invalid("PIZ: bitmap run past end of chunk".to_string()))?;
        bitmap[min_b..=max_b].copy_from_slice(src);
        p += run;
    }
    let cnt_bytes = payload
        .get(p..p + 4)
        .ok_or_else(|| ExrError::invalid("PIZ: Huffman byte count truncated".to_string()))?;
    let huf_len = u32::from_le_bytes(cnt_bytes.try_into().unwrap()) as usize;
    p += 4;
    let huf_payload = payload.get(p..p + huf_len).ok_or_else(|| {
        ExrError::invalid(format!(
            "PIZ: Huffman payload of {huf_len} bytes past end of chunk"
        ))
    })?;

    let (lut, max_value) = inverse_lut(&bitmap);
    let n_words = uncompressed_size / 2;
    let mut words = crate::huf::huf_decompress(huf_payload, n_words)?;

    let extents = shape.extents();
    let region_words: usize = extents
        .iter()
        .map(|e| e.nx * e.ny * e.words_per_sample)
        .sum();
    if region_words != n_words {
        return Err(ExrError::invalid(format!(
            "PIZ: channel regions cover {region_words} words but chunk holds {n_words}"
        )));
    }

    let use16 = max_value >= 16384;
    let mut base = 0usize;
    for e in &extents {
        for j in 0..e.words_per_sample {
            wav_decode(
                &mut words,
                base + j,
                e.nx,
                e.ny,
                e.words_per_sample,
                e.words_per_sample * e.nx,
                use16,
            );
        }
        base += e.nx * e.ny * e.words_per_sample;
    }

    for w in words.iter_mut() {
        *w = lut[*w as usize];
    }

    Ok(scatter_words(&words, shape, &extents, uncompressed_size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Channel, PixelType};

    fn ch(name: &str, pt: PixelType) -> Channel {
        Channel {
            name: name.to_string(),
            pixel_type: pt,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        }
    }

    fn native_len(shape: &ChunkShape) -> usize {
        shape
            .extents()
            .iter()
            .map(|e| e.nx * e.ny * e.words_per_sample * 2)
            .sum()
    }

    fn roundtrip(raw: &[u8], shape: &ChunkShape) {
        let compressed = piz_compress(raw, shape).unwrap();
        assert_ne!(compressed.len(), raw.len(), "ambiguous with raw fallback");
        let back = decode_piz_payload(&compressed, shape, raw.len()).unwrap();
        assert_eq!(back, raw, "PIZ round-trip mismatch");
    }

    #[test]
    fn wavelet_14_invertible() {
        for a in [0u16, 1, 2, 100, 8191, 16383] {
            for b in [0u16, 1, 3, 200, 8191, 16383] {
                let (l, h) = wenc14(a, b);
                let (a2, b2) = wdec14(l, h);
                assert_eq!((a, b), (a2, b2), "wenc14({a},{b})");
            }
        }
    }

    #[test]
    fn wavelet_16_invertible() {
        for a in [0u16, 1, 100, 16384, 32767, 40000, 65535] {
            for b in [0u16, 5, 12345, 16384, 54321, 65535] {
                let (l, h) = wenc16(a, b);
                let (a2, b2) = wdec16(l, h);
                assert_eq!((a, b), (a2, b2), "wenc16({a},{b})");
            }
        }
    }

    #[test]
    fn half_channels_roundtrip_odd_dims() {
        let channels = vec![ch("G", PixelType::Half), ch("R", PixelType::Half)];
        let shape = ChunkShape {
            sorted_channels: &channels,
            width: 13,
            block_y0: 0,
            lines_in_block: 9,
        };
        let n = native_len(&shape);
        let raw: Vec<u8> = (0..n).map(|i| ((i * 7 + i / 5) % 251) as u8).collect();
        roundtrip(&raw, &shape);
    }

    #[test]
    fn float_uint_channels_roundtrip() {
        let channels = vec![
            ch("F", PixelType::Float),
            ch("G", PixelType::Half),
            ch("U", PixelType::Uint),
        ];
        let shape = ChunkShape {
            sorted_channels: &channels,
            width: 21,
            block_y0: 0,
            lines_in_block: 32,
        };
        let n = native_len(&shape);
        let raw: Vec<u8> = (0..n).map(|i| ((i * 13 + i / 3) % 253) as u8).collect();
        roundtrip(&raw, &shape);
    }

    #[test]
    fn all_zero_chunk_has_empty_bitmap() {
        let channels = vec![ch("Y", PixelType::Half)];
        let shape = ChunkShape {
            sorted_channels: &channels,
            width: 16,
            block_y0: 0,
            lines_in_block: 8,
        };
        let raw = vec![0u8; 16 * 8 * 2];
        let compressed = piz_compress(&raw, &shape).unwrap();
        // min (8191) > max (0): no bitmap bytes transmitted.
        let min_b = u16::from_le_bytes(compressed[0..2].try_into().unwrap());
        let max_b = u16::from_le_bytes(compressed[2..4].try_into().unwrap());
        assert!(min_b > max_b, "expected empty bitmap, got {min_b}..{max_b}");
        let back = decode_piz_payload(&compressed, &shape, raw.len()).unwrap();
        assert_eq!(back, raw);
    }

    #[test]
    fn dense_alphabet_takes_mod16_variant() {
        // >16384 distinct code words forces maxValue >= 16384 and the
        // modulo-2^16 wavelet.
        let channels = vec![ch("Y", PixelType::Half)];
        let shape = ChunkShape {
            sorted_channels: &channels,
            width: 200,
            block_y0: 0,
            lines_in_block: 100,
        };
        let mut raw = Vec::with_capacity(200 * 100 * 2);
        for i in 0..(200 * 100) as u32 {
            raw.extend_from_slice(&(((i * 3) & 0xFFFF) as u16).to_le_bytes());
        }
        roundtrip(&raw, &shape);
    }

    #[test]
    fn subsampled_channels_roundtrip() {
        let mut c1 = ch("BY", PixelType::Half);
        c1.x_sampling = 2;
        c1.y_sampling = 2;
        let c2 = ch("Y", PixelType::Half);
        let channels = vec![c1, c2];
        let shape = ChunkShape {
            sorted_channels: &channels,
            width: 14,
            block_y0: 0,
            lines_in_block: 10,
        };
        let n = native_len(&shape);
        let raw: Vec<u8> = (0..n).map(|i| ((i * 11) % 199) as u8).collect();
        roundtrip(&raw, &shape);
    }

    #[test]
    fn single_pixel_chunk() {
        let channels = vec![ch("Y", PixelType::Half)];
        let shape = ChunkShape {
            sorted_channels: &channels,
            width: 1,
            block_y0: 0,
            lines_in_block: 1,
        };
        let raw = vec![0x12u8, 0x34];
        let compressed = piz_compress(&raw, &shape).unwrap();
        let back = decode_piz_payload(&compressed, &shape, 2).unwrap();
        assert_eq!(back, raw);
    }

    #[test]
    fn rejects_hostile_bitmap_range() {
        let channels = vec![ch("Y", PixelType::Half)];
        let shape = ChunkShape {
            sorted_channels: &channels,
            width: 4,
            block_y0: 0,
            lines_in_block: 4,
        };
        // min=0, max=9000 (>= 8192) must be rejected before any read.
        let mut payload = Vec::new();
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&9000u16.to_le_bytes());
        payload.extend_from_slice(&[0u8; 16]);
        assert!(decode_piz_payload(&payload, &shape, 32).is_err());
    }

    #[test]
    fn rejects_truncated_huffman_count() {
        let channels = vec![ch("Y", PixelType::Half)];
        let shape = ChunkShape {
            sorted_channels: &channels,
            width: 4,
            block_y0: 0,
            lines_in_block: 4,
        };
        // Empty bitmap then a huffman count claiming more bytes than exist.
        let mut payload = Vec::new();
        payload.extend_from_slice(&8191u16.to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&1000u32.to_le_bytes());
        payload.push(0);
        assert!(decode_piz_payload(&payload, &shape, 32).is_err());
    }
}
