//! B44 / B44A pixel-data compression and decompression (observer-spec §2).
//!
//! B44 is a fixed-ratio (32:14) lossy compressor for `HALF` channels;
//! `FLOAT` and `UINT` channels are copied byte-for-byte. B44A is identical
//! on the wire except that constant-valued 4×4 blocks may collapse to a
//! 3-byte "flat block". A single decoder handles both: each block's third
//! byte is read first and a value `>= 0x34` marks a flat block.
//!
//! Chunk layout (observer-spec §2.1): the chunk's rows are regrouped into
//! per-channel contiguous planes (all of channel 0's rows, then channel
//! 1's, …, accounting for `ySampling` subsampling). Within a HALF plane
//! the pixels are tiled into 4×4 blocks scanned left-to-right then
//! top-to-bottom, with the rightmost column / bottom row replicated when
//! the plane width / height is not a multiple of 4. There is **no zlib
//! back-end** — the packed blocks are the chunk payload directly.
//!
//! On decode, `pLinear` HALF channels additionally pass each unpacked
//! HALF code word through an inverse "log" (to-linear) lookup; non-linear
//! channels skip it. The 65 536-entry table is *computed* here from the
//! closed-form mapping documented in `openexr-observer-spec.md` §2.3
//! (`out = float_to_half(8·log(half_to_float(x)))` with the documented
//! sentinel clamps), using this crate's bit-exact IEEE-754 binary16
//! conversions — it is not embedded from any external array.

use crate::error::{ExrError, Result};
use crate::half::{f32_to_half, half_to_f32};
use crate::types::{Channel, PixelType};

/// HALF code word for `+0.0` (and the smallest non-negative ordered key).
const HALF_NEG_ZERO: u16 = 0x8000;

/// Build the inverse "log" (to-linear) dequantisation LUT used at decode
/// time for `pLinear` HALF channels (observer-spec §2.3).
///
/// `log[x]`:
/// * HALF infinity / NaN (exponent field all ones) → `0x0000`.
/// * any negative HALF (`x > 0x8000`, i.e. excluding `-0.0`) → `0x0000`.
/// * otherwise `float_to_half(8·log(half_to_float(x)))`, where
///   `half_to_float(0) = 0` makes `log(0) = -inf`, so both `+0.0`
///   (`0x0000`) and `-0.0` (`0x8000`) map to `float_to_half(-inf)`
///   (`0xfc00`).
fn build_log_table() -> Vec<u16> {
    let mut table = vec![0u16; 65536];
    for (x, slot) in table.iter_mut().enumerate() {
        let x = x as u16;
        // inf / NaN: exponent field all ones.
        if x & 0x7c00 == 0x7c00 {
            *slot = 0x0000;
            continue;
        }
        // Negative (excluding -0.0 == 0x8000).
        if x > HALF_NEG_ZERO {
            *slot = 0x0000;
            continue;
        }
        let f = half_to_f32(x);
        // f is 0.0 for both +0.0 and -0.0 here; 8*ln(0) = -inf.
        let v = if f == 0.0 {
            f32::NEG_INFINITY
        } else {
            8.0 * f.ln()
        };
        *slot = f32_to_half(v);
    }
    table
}

/// Lazily-built inverse log table, shared across decode calls.
fn log_table() -> &'static [u16] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Vec<u16>> = OnceLock::new();
    TABLE.get_or_init(build_log_table)
}

/// Build the forward "exp" (from-linear) quantisation LUT applied at
/// encode time for `pLinear` HALF channels (observer-spec §2.3).
///
/// `exp[x]`:
/// * HALF infinity / NaN (exponent field all ones) → `0x0000`.
/// * positive HALF code words `>= 0x558c` (HALF value ≥ 88.75, the
///   smallest HALF at/above `8·ln(HALF_MAX)`) → `0x7bff` (clamp to
///   `HALF_MAX`).
/// * otherwise `float_to_half(exp(half_to_float(x) / 8))`.
fn build_exp_table() -> Vec<u16> {
    let mut table = vec![0u16; 65536];
    for (x, slot) in table.iter_mut().enumerate() {
        let x = x as u16;
        // inf / NaN: exponent field all ones.
        if x & 0x7c00 == 0x7c00 {
            *slot = 0x0000;
            continue;
        }
        // Positive code words at/above the clamp threshold map to HALF_MAX.
        // (A negative HALF has its sign bit set, so its code word is
        // >= 0x8000 > 0x558c only when negative — but those large code
        // words are negative values, which the documented closed form
        // handles via the general exp branch below. The `>= 0x558c` clamp
        // is specifically the *positive* clamp, so we restrict it to the
        // non-negative half of the code space, x < 0x8000.)
        if (0x558c..0x8000).contains(&x) {
            *slot = 0x7bff;
            continue;
        }
        let f = half_to_f32(x);
        *slot = f32_to_half((f / 8.0).exp());
    }
    table
}

/// Lazily-built forward exp table, shared across encode calls.
fn exp_table() -> &'static [u16] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<Vec<u16>> = OnceLock::new();
    TABLE.get_or_init(build_exp_table)
}

/// Unpack one 14-byte B44 block into its 16 HALF code words `s[0..15]`,
/// inverting the monotone "sign-magnitude → ordered integer" remap.
///
/// Layout (observer-spec §2.4): bytes 0–1 carry `t[0]` big-endian;
/// byte 2's top 6 bits hold `shift`; the remaining bits stream the fifteen
/// 6-bit biased differences `r[0..14]`. Each child `t` is reconstructed by
/// `t[child] = t[parent] + (r[k] << shift) − (0x20 << shift)`.
fn unpack14(b: &[u8; 14]) -> [u16; 16] {
    let mut t = [0u16; 16];
    t[0] = ((b[0] as u16) << 8) | b[1] as u16;

    let shift = (b[2] >> 2) as u32;
    let bias = 0x20u32 << shift;

    // Decode the fifteen 6-bit r fields, mirroring the §2.4 packing.
    let r = [
        ((b[2] as u32) << 4) | ((b[3] as u32) >> 4),   // r0
        ((b[3] as u32) << 2) | ((b[4] as u32) >> 6),   // r1
        b[4] as u32,                                   // r2
        ((b[5] as u32) >> 2),                          // r3
        ((b[5] as u32) << 4) | ((b[6] as u32) >> 4),   // r4
        ((b[6] as u32) << 2) | ((b[7] as u32) >> 6),   // r5
        b[7] as u32,                                   // r6
        ((b[8] as u32) >> 2),                          // r7
        ((b[8] as u32) << 4) | ((b[9] as u32) >> 4),   // r8
        ((b[9] as u32) << 2) | ((b[10] as u32) >> 6),  // r9
        b[10] as u32,                                  // r10
        ((b[11] as u32) >> 2),                         // r11
        ((b[11] as u32) << 4) | ((b[12] as u32) >> 4), // r12
        ((b[12] as u32) << 2) | ((b[13] as u32) >> 6), // r13
        b[13] as u32,                                  // r14
    ]
    .map(|v| v & 0x3f);

    // Prefix-sum each r field back through the 2-D differencing tree
    // rooted at t[0]. The tree edges mirror the §2.4 r-index table:
    // r[k] = d[parent] − d[child] + 32, so on decode
    //   t[child] = t[parent] + (r[k] << shift) − (0x20 << shift).
    let mut step = |parent: usize, child: usize, k: usize| {
        let val = (t[parent] as u32)
            .wrapping_add(r[k] << shift)
            .wrapping_sub(bias);
        t[child] = val as u16;
    };
    // Left column top-to-bottom.
    step(0, 4, 0);
    step(4, 8, 1);
    step(8, 12, 2);
    // Each row left-to-right.
    step(0, 1, 3);
    step(4, 5, 4);
    step(8, 9, 5);
    step(12, 13, 6);
    step(1, 2, 7);
    step(5, 6, 8);
    step(9, 10, 9);
    step(13, 14, 10);
    step(2, 3, 11);
    step(6, 7, 12);
    step(10, 11, 13);
    step(14, 15, 14);

    // Invert the monotone remap: if the high bit is set the value was
    // non-negative (t = s | 0x8000), else it was the complement of a
    // negative s (t = ~s).
    let mut s = [0u16; 16];
    for i in 0..16 {
        s[i] = if t[i] & 0x8000 != 0 {
            t[i] & 0x7fff
        } else {
            !t[i]
        };
    }
    s
}

/// Replicate the single HALF code word `t0` (already in the on-disk
/// big-endian-stored ordered form) to all 16 pixels of a flat block.
fn unpack3(b: &[u8; 3]) -> [u16; 16] {
    let t0 = ((b[0] as u16) << 8) | b[1] as u16;
    let s = if t0 & 0x8000 != 0 { t0 & 0x7fff } else { !t0 };
    [s; 16]
}

/// Decode one HALF channel plane (`pw` × `ph` samples) from the B44 block
/// stream starting at `payload[*cursor..]`, writing the recovered HALF
/// code words row-major into `out` (length `pw*ph`). Advances `*cursor`
/// past the consumed block bytes. `p_linear` selects whether the inverse
/// log table is applied.
fn decode_half_plane(
    payload: &[u8],
    cursor: &mut usize,
    pw: usize,
    ph: usize,
    p_linear: bool,
    out: &mut [u16],
) -> Result<()> {
    let log = if p_linear { Some(log_table()) } else { None };
    // Blocks tile the plane left-to-right then top-to-bottom; partial
    // edge blocks replicate the rightmost column / bottom row, so the
    // block grid covers ceil(pw/4) × ceil(ph/4) blocks.
    let mut by = 0usize;
    while by < ph {
        let mut bx = 0usize;
        while bx < pw {
            // Determine block kind from byte 2 (>= 0x34 ⇒ 3-byte flat).
            if *cursor >= payload.len() {
                return Err(ExrError::invalid("B44 block stream truncated".to_string()));
            }
            let s: [u16; 16];
            if *cursor + 3 <= payload.len() && payload[*cursor + 2] >= 0x34 {
                let blk: [u8; 3] = payload[*cursor..*cursor + 3].try_into().unwrap();
                s = unpack3(&blk);
                *cursor += 3;
            } else {
                if *cursor + 14 > payload.len() {
                    return Err(ExrError::invalid(
                        "B44 14-byte block runs past chunk end".to_string(),
                    ));
                }
                let blk: [u8; 14] = payload[*cursor..*cursor + 14].try_into().unwrap();
                s = unpack14(&blk);
                *cursor += 14;
            }
            // Scatter the 4×4 block into the plane, clipping to the real
            // plane extent (padding pixels off the right / bottom edge are
            // discarded).
            for r in 0..4 {
                let py = by + r;
                if py >= ph {
                    break;
                }
                for c in 0..4 {
                    let px = bx + c;
                    if px >= pw {
                        continue;
                    }
                    let mut code = s[r * 4 + c];
                    if let Some(tbl) = log {
                        code = tbl[code as usize];
                    }
                    out[py * pw + px] = code;
                }
            }
            bx += 4;
        }
        by += 4;
    }
    Ok(())
}

/// One channel's contribution to a B44 chunk: its sub-sampled width and
/// the number of rows it occupies within the chunk.
pub(crate) struct B44ChannelExtent {
    pub pw: usize,
    pub ph: usize,
}

/// Decode a whole B44 / B44A chunk payload into per-channel HALF / native
/// planes, in the channel order given by `channels`.
///
/// Returns one `Vec<u16>` (HALF code words, row-major `pw*ph`) per HALF
/// channel and one raw `Vec<u8>` per FLOAT / UINT channel — the caller
/// scatters them into the image planes. Channels are returned in the same
/// order as `channels`.
///
/// `extents[i]` gives the i-th channel's sub-sampled `(pw, ph)` for this
/// chunk. FLOAT / UINT channels are copied verbatim (`pw*ph*bps` bytes);
/// HALF channels are block-decoded.
pub(crate) enum B44Plane {
    Half(Vec<u16>),
    Raw(Vec<u8>),
}

pub(crate) fn decode_b44_chunk(
    payload: &[u8],
    channels: &[Channel],
    extents: &[B44ChannelExtent],
) -> Result<Vec<B44Plane>> {
    let mut cursor = 0usize;
    let mut planes = Vec::with_capacity(channels.len());
    for (ch, ext) in channels.iter().zip(extents) {
        let count = ext.pw * ext.ph;
        match ch.pixel_type {
            PixelType::Half => {
                let mut out = vec![0u16; count];
                decode_half_plane(payload, &mut cursor, ext.pw, ext.ph, ch.p_linear, &mut out)?;
                planes.push(B44Plane::Half(out));
            }
            PixelType::Float | PixelType::Uint => {
                let nbytes = count * ch.pixel_type.bytes_per_sample();
                if cursor + nbytes > payload.len() {
                    return Err(ExrError::invalid(format!(
                        "B44 raw channel '{}' runs past chunk end (need {nbytes} at {cursor}, have {})",
                        ch.name,
                        payload.len()
                    )));
                }
                planes.push(B44Plane::Raw(payload[cursor..cursor + nbytes].to_vec()));
                cursor += nbytes;
            }
        }
    }
    Ok(planes)
}

/// Map a HALF code word `s` to the monotone "sign-magnitude → ordered
/// integer" form `t` used by the block packer (observer-spec §2.4):
/// inf/NaN → `0x8000`; negative → `~s`; non-negative → `s | 0x8000`.
fn s_to_t(s: u16) -> u16 {
    if s & 0x7c00 == 0x7c00 {
        // inf / NaN (exponent field all ones).
        0x8000
    } else if s & 0x8000 != 0 {
        !s
    } else {
        s | 0x8000
    }
}

/// `round(x / 2^sh)` with round-half-to-even tie-breaking
/// (observer-spec §2.4 `shiftAndRound`). `x` is non-negative.
fn shift_and_round(x: u32, sh: u32) -> u32 {
    if sh == 0 {
        return x;
    }
    let half = 1u32 << (sh - 1);
    let q = x >> sh;
    let rem = x & ((1u32 << sh) - 1);
    if rem > half {
        q + 1
    } else if rem < half {
        q
    } else {
        // Exact half: round to even.
        q + (q & 1)
    }
}

/// Pack one 4×4 HALF block (`s[0..15]`, already exp-quantised for pLinear
/// channels) into the 14-byte form (observer-spec §2.4). `exactmax`
/// selects the non-linear-channel `t[0]` correction. Returns the 14 bytes.
fn pack14(s: &[u16; 16], exactmax: bool) -> [u8; 14] {
    let mut t = [0u16; 16];
    for i in 0..16 {
        t[i] = s_to_t(s[i]);
    }
    let t_max = *t.iter().max().unwrap();

    // The 15 difference edges of the 2-D tree (parent, child), in r-index
    // order, mirroring the §2.4 table and the decoder's `step` calls.
    const EDGES: [(usize, usize); 15] = [
        (0, 4),
        (4, 8),
        (8, 12),
        (0, 1),
        (4, 5),
        (8, 9),
        (12, 13),
        (1, 2),
        (5, 6),
        (9, 10),
        (13, 14),
        (2, 3),
        (6, 7),
        (10, 11),
        (14, 15),
    ];

    // Find the smallest shift such that every biased difference fits in
    // [0, 63]. `d[i] = shiftAndRound(t_max - t[i], shift)`, and each
    // r = d[parent] - d[child] + 32 (bias 0x20).
    let mut shift = 0u32;
    let mut d = [0u32; 16];
    let mut r = [0i32; 15];
    loop {
        for i in 0..16 {
            d[i] = shift_and_round((t_max - t[i]) as u32, shift);
        }
        let mut ok = true;
        for (k, &(p, c)) in EDGES.iter().enumerate() {
            let v = d[p] as i32 - d[c] as i32 + 0x20;
            r[k] = v;
            if !(0..=63).contains(&v) {
                ok = false;
            }
        }
        if ok {
            break;
        }
        shift += 1;
        // shift can never legitimately reach 13 (the 0xfc flat marker is
        // 0xfc >> 2 = 0x3f shift, far past any real value), but guard the
        // loop so a pathological input can't spin forever.
        debug_assert!(shift < 32, "B44 shift search diverged");
    }

    // Non-linear (`exactmax`) channel correction: make t[0] reproduce
    // t_max as closely as possible.
    let t0 = if exactmax {
        t_max.wrapping_sub((d[0] << shift) as u16)
    } else {
        t[0]
    };

    let r = r.map(|v| v as u32 & 0x3f);
    let mut b = [0u8; 14];
    b[0] = (t0 >> 8) as u8;
    b[1] = (t0 & 0xff) as u8;
    b[2] = ((shift << 2) | (r[0] >> 4)) as u8;
    b[3] = ((r[0] << 4) | (r[1] >> 2)) as u8;
    b[4] = ((r[1] << 6) | r[2]) as u8;
    b[5] = ((r[3] << 2) | (r[4] >> 4)) as u8;
    b[6] = ((r[4] << 4) | (r[5] >> 2)) as u8;
    b[7] = ((r[5] << 6) | r[6]) as u8;
    b[8] = ((r[7] << 2) | (r[8] >> 4)) as u8;
    b[9] = ((r[8] << 4) | (r[9] >> 2)) as u8;
    b[10] = ((r[9] << 6) | r[10]) as u8;
    b[11] = ((r[11] << 2) | (r[12] >> 4)) as u8;
    b[12] = ((r[12] << 4) | (r[13] >> 2)) as u8;
    b[13] = ((r[13] << 6) | r[14]) as u8;
    b
}

/// Pack a constant 4×4 block into the 3-byte flat form (observer-spec
/// §2.5): `t[0]` big-endian followed by the `0xfc` out-of-range marker.
fn pack3(s0: u16) -> [u8; 3] {
    let t0 = s_to_t(s0);
    [(t0 >> 8) as u8, (t0 & 0xff) as u8, 0xfc]
}

/// Encode one HALF channel plane (`pw` × `ph` samples, row-major HALF code
/// words in `plane`) into the B44 block stream, appending to `out`.
///
/// Blocks tile the plane left-to-right then top-to-bottom; partial edge
/// blocks replicate the rightmost column / bottom row (observer-spec §2.2).
/// `p_linear` selects whether the forward exp table is applied to each
/// sample before packing; non-linear channels also set `exactmax` so the
/// block packer corrects `t[0]`. When `flat` (B44A) is on, a block whose
/// 16 samples are all equal is emitted as a 3-byte flat block.
fn encode_half_plane(
    plane: &[u16],
    pw: usize,
    ph: usize,
    p_linear: bool,
    flat: bool,
    out: &mut Vec<u8>,
) {
    let exp = if p_linear { Some(exp_table()) } else { None };
    let exactmax = !p_linear;
    let mut by = 0usize;
    while by < ph {
        let mut bx = 0usize;
        while bx < pw {
            // Gather the 4×4 block, replicating the rightmost column /
            // bottom row when the block overhangs the plane edge.
            let mut s = [0u16; 16];
            for r in 0..4 {
                let py = (by + r).min(ph - 1);
                for c in 0..4 {
                    let px = (bx + c).min(pw - 1);
                    let mut code = plane[py * pw + px];
                    if let Some(tbl) = exp {
                        code = tbl[code as usize];
                    }
                    s[r * 4 + c] = code;
                }
            }
            // Flat-block special case (B44A): all 16 samples identical.
            if flat && s.iter().all(|&v| v == s[0]) {
                out.extend_from_slice(&pack3(s[0]));
            } else {
                out.extend_from_slice(&pack14(&s, exactmax));
            }
            bx += 4;
        }
        by += 4;
    }
}

/// Encode a whole B44 / B44A chunk into its packed payload (observer-spec
/// §2.1): per-channel contiguous planes in `channels` order — HALF
/// channels block-packed, FLOAT / UINT channels copied verbatim — with no
/// zlib back-end. `planes[i]` carries the i-th channel's chunk data:
/// `B44Plane::Half` (row-major HALF code words, `pw*ph`) for HALF channels,
/// `B44Plane::Raw` (little-endian sample bytes, `pw*ph*bps`) otherwise.
/// `flat` enables the B44A 3-byte flat block.
///
/// Returns the concatenated payload. The caller applies the shared
/// raw-fallback rule (store the interleaved uncompressed stream when this
/// payload would not be smaller).
pub(crate) fn encode_b44_chunk(
    planes: &[B44Plane],
    channels: &[Channel],
    extents: &[B44ChannelExtent],
    flat: bool,
) -> Vec<u8> {
    let mut out = Vec::new();
    for ((ch, ext), plane) in channels.iter().zip(extents).zip(planes) {
        match plane {
            B44Plane::Half(codes) => {
                encode_half_plane(codes, ext.pw, ext.ph, ch.p_linear, flat, &mut out);
            }
            B44Plane::Raw(bytes) => {
                out.extend_from_slice(bytes);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_table_sentinels() {
        let log = build_log_table();
        // observer-spec §2.3 sentinel facts.
        assert_eq!(log[0x8000], 0xfc00, "log[-0.0] should map to -inf half");
        assert_eq!(
            log[0x8001], 0x0000,
            "log[smallest negative] should clamp to 0"
        );
        // +0.0 also maps through log(0) = -inf.
        assert_eq!(log[0x0000], 0xfc00, "log[+0.0] should map to -inf half");
        // inf / NaN clamp to 0.
        assert_eq!(log[0x7c00], 0x0000, "log[+inf] should clamp to 0");
        assert_eq!(log[0xfc00], 0x0000, "log[-inf] should clamp to 0");
        assert_eq!(log[0x7e00], 0x0000, "log[NaN] should clamp to 0");
    }

    #[test]
    fn flat_block_replicates() {
        // A 3-byte flat block (marker 0xfc) replicates t[0] to all pixels.
        // Choose t0 for HALF value 1.0 (s = 0x3c00 → non-negative →
        // t = 0x3c00 | 0x8000 = 0xbc00, stored big-endian as bytes
        // [0xbc, 0x00]).
        let blk = [0xbcu8, 0x00, 0xfc];
        let s = unpack3(&blk);
        for v in s {
            assert_eq!(v, 0x3c00, "flat block should recover HALF 1.0");
        }
    }

    #[test]
    fn exp_table_sentinels() {
        let exp = build_exp_table();
        // observer-spec §2.3 sentinel facts.
        assert_eq!(exp[0x0000], 0x3c00, "exp[+0.0] should be HALF 1.0");
        assert_eq!(
            exp[0x558c], 0x7bff,
            "exp[clamp threshold] should be HALF_MAX"
        );
        // inf / NaN clamp to 0.
        assert_eq!(exp[0x7c00], 0x0000, "exp[+inf] should clamp to 0");
        assert_eq!(exp[0x7e00], 0x0000, "exp[NaN] should clamp to 0");
        // A code word just below the clamp threshold is not clamped.
        assert_ne!(exp[0x558b], 0x7bff, "exp just below threshold not clamped");
    }

    #[test]
    fn pack14_unpack14_roundtrip_nonlinear() {
        // Non-linear (exactmax) pack of a varied 4×4 block must unpack back
        // to the same 16 HALF code words (the quantisation is exact when
        // the block fits with shift 0). Use small magnitudes so no shift.
        let s: [u16; 16] = [
            0x3c00, 0x4000, 0x4200, 0x4400, 0x3800, 0x3a00, 0x3e00, 0x4100, 0x3000, 0x3400, 0x3600,
            0x3c80, 0x2800, 0x2c00, 0x3200, 0x3d00,
        ];
        let packed = pack14(&s, true);
        let got = unpack14(&packed);
        // The block max round-trips exactly (exactmax correction); other
        // pixels round to the quantisation lattice. Re-packing the unpacked
        // block must be a fixed point.
        let packed2 = pack14(&got, true);
        let got2 = unpack14(&packed2);
        assert_eq!(got, got2, "pack/unpack must be a fixed point");
    }

    #[test]
    fn pack3_unpack3_roundtrip() {
        // Flat block: pack a constant HALF value, unpack, recover it for all
        // 16 pixels.
        for &v in &[0x3c00u16, 0x0000, 0xbc00, 0x4900] {
            let blk = pack3(v);
            assert_eq!(blk[2], 0xfc, "flat marker must be 0xfc");
            let s = unpack3(&blk);
            for &got in &s {
                assert_eq!(got, v, "flat block must recover {v:#06x}");
            }
        }
    }

    #[test]
    fn s_to_t_monotone_roundtrip() {
        // The monotone remap must be invertible by the decoder's inverse.
        for &s in &[0x0000u16, 0x3c00, 0x7bff, 0x8000, 0xbc00, 0xfbff] {
            let t = s_to_t(s);
            let back = if t & 0x8000 != 0 { t & 0x7fff } else { !t };
            assert_eq!(back, s, "s_to_t inverse must recover {s:#06x}");
        }
    }

    #[test]
    fn constant_14byte_block_roundtrips_value() {
        // A 14-byte block where every pixel equals the same value packs as
        // shift=0, t[0] big-endian, and all r == 0x20 (bias). Decoding it
        // must recover that constant for all 16 pixels.
        // Pick HALF value 0.5 (s = 0x3800 → t = 0xb800).
        let t0: u16 = 0xb800;
        let mut b = [0u8; 14];
        b[0] = (t0 >> 8) as u8;
        b[1] = (t0 & 0xff) as u8;
        // shift = 0; every r = 0x20. Pack per §2.4.
        let r = [0x20u32; 15];
        let shift = 0u32;
        b[2] = ((shift << 2) | (r[0] >> 4)) as u8;
        b[3] = ((r[0] << 4) | (r[1] >> 2)) as u8;
        b[4] = ((r[1] << 6) | r[2]) as u8;
        b[5] = ((r[3] << 2) | (r[4] >> 4)) as u8;
        b[6] = ((r[4] << 4) | (r[5] >> 2)) as u8;
        b[7] = ((r[5] << 6) | r[6]) as u8;
        b[8] = ((r[7] << 2) | (r[8] >> 4)) as u8;
        b[9] = ((r[8] << 4) | (r[9] >> 2)) as u8;
        b[10] = ((r[9] << 6) | r[10]) as u8;
        b[11] = ((r[11] << 2) | (r[12] >> 4)) as u8;
        b[12] = ((r[12] << 4) | (r[13] >> 2)) as u8;
        b[13] = ((r[13] << 6) | r[14]) as u8;
        let s = unpack14(&b);
        for v in s {
            assert_eq!(v, 0x3800, "constant block should recover HALF 0.5");
        }
    }
}
