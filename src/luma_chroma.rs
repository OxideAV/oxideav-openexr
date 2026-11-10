//! Luminance / chroma (`Y` + `RY` + `BY`) ↔ RGB conversion.
//!
//! OpenEXR files may store colour as a full-resolution luminance
//! channel `Y` plus two 2×2 sub-sampled chroma channels `RY` and `BY`
//! (and an independent `A`). The container-level readers and writers
//! carry those planes bit-exactly as stored (reachable through
//! [`crate::ExrPart`]); this module turns them into RGB and back,
//! reproducing the reference RGBA interface sample-for-sample.
//!
//! # Luminance weights
//!
//! `Y = w_r·R + w_g·G + w_b·B` where `(w_r, w_g, w_b)` is the `Y` row of
//! the RGB → CIE XYZ matrix of the image's RGB space, normalised so the
//! white point has `Y = 1`: with `P` the matrix whose columns are the
//! primaries' `(x, y, 1 − x − y)` and `W = (x_w / y_w, 1, (1 − x_w −
//! y_w) / y_w)`, solve `P · S = W` and take `w_i = y_i · S_i`. The
//! primaries come from the `chromaticities` attribute, else the
//! Rec. ITU-R BT.709 primaries with D65 white
//! ([`BT709_CHROMATICITIES`]). The solve runs in binary64 on the binary32
//! attribute values and each weight is rounded once to binary32 — the
//! BT.709 weights are therefore `0.212639019 / 0.715168655 /
//! 0.0721923187`, not the four-decimal luma coefficients.
//!
//! # Decoding ([`luma_chroma_to_rgb`])
//!
//! 1. **Reconstruction.** Chroma sample `(i, j)` is co-sited with luma
//!    pixel `(2i, 2j)`. Each chroma plane is brought to full resolution
//!    separably, **horizontal pass first**: even positions copy their
//!    sample, odd positions sum the 14 samples at luma offsets ±1, ±3,
//!    …, ±13 weighted by [`RECONSTRUCTION_TAPS`] (binary32 products and
//!    running sum, ascending offset order); a sample index beyond either
//!    end is clamped to the end sample. The result of each pass is
//!    rounded to binary16.
//! 2. **Conversion.** Per pixel in binary32: `R = (RY + 1)·Y`,
//!    `B = (BY + 1)·Y`, `G = (Y − R·w_r − B·w_b) / w_g`, each rounded to
//!    binary16 (a zero `Y` gives black, with the zeros' signs following
//!    the products).
//! 3. **Saturation limiting.** With `sat = 1 − min/max` (`0` when no
//!    component is positive), a pixel whose saturation exceeds `t =
//!    min(1, 0.75 + 0.25·avg(sat of its four diagonal neighbours))` is
//!    pulled toward its own maximum component until its saturation is
//!    `t` — `q = c·f + max·(1 − f)` with `f = t / sat` in binary32 and the
//!    interpolation evaluated in binary64, floored at 0, rounded to
//!    binary16 — then rescaled by `Y(p) / Y(q)` (binary32 luminances) to
//!    its original luminance (binary16). Diagonal neighbours outside the
//!    image come from a one-pixel ring: the row above row 0 takes row
//!    0's luminance with the vertical reconstruction evaluated at row
//!    −1, the row below the last takes row `H − 2`, and the columns
//!    replicate the edge column.
//!
//! The output values are binary16-representable `f32`s.
//!
//! # Encoding ([`rgb_to_luma_chroma`])
//!
//! `Y` is computed in binary32 and rounded to binary16; `RY = (R − Y)/Y`
//! and `BY = (B − Y)/Y` use that binary16 `Y` and are rounded to
//! binary16 per pixel (`0` where `Y == 0`). The 2×2 decimation keeps the
//! even positions of a separable low-pass ([`DECIMATION_TAPS`], offset 0
//! and ±1 … ±13 odd), horizontal pass first with binary16 rounding after
//! each pass; a sample index below the start clamps to 0 and one past the
//! end takes index `n − 2`. With mantissa rounding on, the stored `Y`
//! keeps 7 mantissa bits and `RY` / `BY` keep 5 ([`round_half_mantissa`],
//! round half up in magnitude on the binary16 code word).
//!
//! Chroma at full resolution (1×1) skips reconstruction / decimation;
//! any other sampling factor (outside the 2×2 layout conforming readers
//! accept) falls back to [`upsample_bilinear`] / [`downsample_tent`].

use crate::decoder::subsampled_dim;
use crate::error::{ExrError, Result};
use crate::half::{f32_to_half, half_to_f32};
use crate::types::{Attribute, AttributeValue, Chromaticities};

/// Rec. ITU-R BT.709 primaries with the D65 white point — the colour
/// space assumed when a file carries no `chromaticities` attribute.
pub const BT709_CHROMATICITIES: Chromaticities = Chromaticities {
    red_x: 0.64,
    red_y: 0.33,
    green_x: 0.3,
    green_y: 0.6,
    blue_x: 0.15,
    blue_y: 0.06,
    white_x: 0.3127,
    white_y: 0.329,
};

/// Chroma reconstruction (2:1 → 1:1) taps for the luma offsets ±1, ±3,
/// …, ±13 (index `k` ↔ offset `±(2k + 1)`), binary32.
pub const RECONSTRUCTION_TAPS: [f32; 7] = [
    f32::from_bits(0x3f20_8b22),
    f32::from_bits(0xbe3e_8af8),
    f32::from_bits(0x3db4_141f),
    f32::from_bits(0xbd30_c77e),
    f32::from_bits(0x3ca0_89e3),
    f32::from_bits(0xbbf7_121b),
    f32::from_bits(0x3b0b_75ea),
];

/// Chroma decimation (1:1 → 2:1) taps: index 0 ↔ offset 0, index `k ≥ 1`
/// ↔ offset `±(2k − 1)`; even non-zero offsets weigh 0. Binary32.
pub const DECIMATION_TAPS: [f32; 8] = [
    f32::from_bits(0x3eff_ebd1),
    f32::from_bits(0x3ea0_97ea),
    f32::from_bits(0xbdbe_99e9),
    f32::from_bits(0x3d34_2246),
    f32::from_bits(0xbcb0_d520),
    f32::from_bits(0x3c20_9460),
    f32::from_bits(0xbb77_22e2),
    f32::from_bits(0x3a8b_75ea),
];

/// Mantissa bits the luminance/chroma writer keeps in a stored `Y`.
pub const Y_MANTISSA_BITS: u32 = 7;
/// Mantissa bits the luminance/chroma writer keeps in a stored `RY` /
/// `BY`.
pub const CHROMA_MANTISSA_BITS: u32 = 5;

/// Round to the nearest binary16 value (ties to even), as `f32`.
#[inline]
fn h(v: f32) -> f32 {
    half_to_f32(f32_to_half(v))
}

/// Round the binary16 code word `bits` to `keep` (≤ 10) mantissa bits,
/// half up in magnitude; the sign is kept, a carry moves into the
/// exponent, infinities / NaNs pass through, and a finite value that
/// would round up to infinity stays at the largest finite value with
/// that many mantissa bits.
pub fn round_half_mantissa(bits: u16, keep: u32) -> u16 {
    if keep >= 10 || bits & 0x7c00 == 0x7c00 {
        return bits;
    }
    let drop = 10 - keep;
    let sign = bits & 0x8000;
    let mask = (1u16 << drop) - 1;
    let mut mag = ((bits & 0x7fff) + (1 << (drop - 1))) & !mask;
    if mag >= 0x7c00 {
        mag = 0x7c00 - (1 << drop);
    }
    sign | mag
}

/// `v` rounded to binary16 and then to `keep` mantissa bits
/// ([`round_half_mantissa`]), as `f32`.
fn h_keep(v: f32, keep: u32) -> f32 {
    half_to_f32(round_half_mantissa(f32_to_half(v), keep))
}

/// Luminance weights `(w_r, w_g, w_b)` of the RGB space described by
/// `c` (module docs, *Luminance weights*): solved in binary64 from the
/// binary32 attribute values, rounded once to binary32. Returns the
/// BT.709 weights when the primaries are degenerate (a zero / non-finite
/// `y` coordinate or a singular primary matrix), which cannot describe a
/// colour space anyway.
pub fn luminance_weights(c: &Chromaticities) -> [f32; 3] {
    let solve = || -> Option<[f32; 3]> {
        let col = |x: f32, y: f32| {
            let (x, y) = (f64::from(x), f64::from(y));
            [x, y, 1.0 - x - y]
        };
        let (r, g, b) = (
            col(c.red_x, c.red_y),
            col(c.green_x, c.green_y),
            col(c.blue_x, c.blue_y),
        );
        let (xw, yw) = (f64::from(c.white_x), f64::from(c.white_y));
        if yw == 0.0 || !yw.is_finite() || !xw.is_finite() {
            return None;
        }
        let w = [xw / yw, 1.0, (1.0 - xw - yw) / yw];
        // Solve [r g b] · s = w by Cramer's rule.
        let det = |a: [f64; 3], b: [f64; 3], c: [f64; 3]| -> f64 {
            a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1])
                + c[0] * (a[1] * b[2] - a[2] * b[1])
        };
        let d = det(r, g, b);
        if d == 0.0 || !d.is_finite() {
            return None;
        }
        let s = [det(w, g, b) / d, det(r, w, b) / d, det(r, g, w) / d];
        let out = [r[1] * s[0], g[1] * s[1], b[1] * s[2]];
        if out.iter().any(|v| !v.is_finite()) || out == [0.0; 3] {
            return None;
        }
        if r[1] == 0.0 || g[1] == 0.0 || b[1] == 0.0 {
            return None;
        }
        Some([out[0] as f32, out[1] as f32, out[2] as f32])
    };
    solve().unwrap_or_else(|| {
        // BT.709 is never degenerate, so this recursion terminates.
        luminance_weights(&BT709_CHROMATICITIES)
    })
}

/// The `chromaticities` attribute of a header, if present and typed.
pub fn chromaticities_of(attrs: &[Attribute]) -> Option<Chromaticities> {
    attrs.iter().find_map(|a| match (&a.name[..], &a.value) {
        ("chromaticities", AttributeValue::Chromaticities(c)) => Some(*c),
        _ => None,
    })
}

/// Luminance weights for an image with the given header attributes:
/// [`luminance_weights`] of its `chromaticities` attribute, or of
/// [`BT709_CHROMATICITIES`] when the attribute is absent.
pub fn luminance_weights_of(attrs: &[Attribute]) -> [f32; 3] {
    luminance_weights(&chromaticities_of(attrs).unwrap_or(BT709_CHROMATICITIES))
}

/// One reconstruction pass value at full-resolution position `x` (may
/// be `-1`) of a line of chroma samples `c` (module docs, step 1),
/// before rounding.
fn reconstruct_at(c: &[f32], x: isize) -> f32 {
    let last = c.len() as isize - 1;
    if x.rem_euclid(2) == 0 {
        return c[(x / 2).clamp(0, last) as usize];
    }
    let mut sum = 0.0f32;
    for o in (-13isize..=13).step_by(2) {
        let idx = (x + o).div_euclid(2).clamp(0, last) as usize;
        sum += c[idx] * RECONSTRUCTION_TAPS[(o.unsigned_abs() - 1) / 2];
    }
    sum
}

/// 2×2 chroma brought to full resolution (module docs, step 1),
/// returning rows `-1 ..= height - 1` (`height + 1` rows of `width`
/// samples; the first is the row-`-1` evaluation the saturation ring
/// uses). `sub` holds `ceil(width/2) × ceil(height/2)` samples.
fn reconstruct_2x2_with_ring(sub: &[f32], width: usize, height: usize) -> Vec<f32> {
    let pw = width.div_ceil(2);
    let ph = height.div_ceil(2);
    // Horizontal pass: every chroma row → `width` values, binary16.
    let mut horiz = vec![0.0f32; ph * width];
    for j in 0..ph {
        let row = &sub[j * pw..(j + 1) * pw];
        for x in 0..width {
            horiz[j * width + x] = h(reconstruct_at(row, x as isize));
        }
    }
    // Vertical pass over every column, rows -1 ..= height - 1.
    let mut out = vec![0.0f32; (height + 1) * width];
    let mut col = vec![0.0f32; ph];
    for x in 0..width {
        for (j, c) in col.iter_mut().enumerate() {
            *c = horiz[j * width + x];
        }
        for y in -1..height as isize {
            out[(y + 1) as usize * width + x] = h(reconstruct_at(&col, y));
        }
    }
    out
}

/// Reconstruct a full-resolution plane from a 2×2 sub-sampled chroma
/// plane with the reconstruction filter (module docs, step 1). `sub`
/// must hold `ceil(width / 2) × ceil(height / 2)` samples. The result
/// is binary16-representable.
pub fn reconstruct_chroma_2x2(sub: &[f32], width: u32, height: u32) -> Result<Vec<f32>> {
    let (w, hgt) = (width as usize, height as usize);
    check_sub_len(sub, width, height, 2, 2)?;
    if w == 0 || hgt == 0 {
        return Ok(Vec::new());
    }
    let mut ring = reconstruct_2x2_with_ring(sub, w, hgt);
    ring.drain(..w);
    Ok(ring)
}

fn check_sub_len(sub: &[f32], width: u32, height: u32, sx: u32, sy: u32) -> Result<()> {
    let pw = subsampled_dim(width, sx) as usize;
    let ph = subsampled_dim(height, sy) as usize;
    if sub.len() != pw * ph {
        return Err(ExrError::invalid(format!(
            "sub-sampled plane holds {} samples, expected {pw}×{ph} for {width}×{height} at \
             {sx}×{sy}",
            sub.len()
        )));
    }
    Ok(())
}

/// Decimate a full-resolution chroma plane to 2×2 sampling with the
/// decimation filter (module docs, *Encoding*): sample `(i, j)` is the
/// filtered value at `(2i, 2j)`. Returns `ceil(width / 2) × ceil(height
/// / 2)` binary16-representable samples.
pub fn decimate_chroma_2x2(full: &[f32], width: u32, height: u32) -> Result<Vec<f32>> {
    let (w, hgt) = (width as usize, height as usize);
    if full.len() != w * hgt {
        return Err(ExrError::invalid(format!(
            "plane holds {} samples, expected {w}×{hgt}",
            full.len()
        )));
    }
    let (pw, ph) = (w.div_ceil(2), hgt.div_ceil(2));
    // Index rule: below the start → 0, past the end → n − 2 (n − 1 for
    // a one-sample line).
    let filter = |line: &dyn Fn(usize) -> f32, n: usize, pos: usize| -> f32 {
        let last = n as isize - 1;
        let past = (n as isize - 2).max(0);
        let at = |k: isize| -> f32 {
            let k = if k < 0 {
                0
            } else if k > last {
                past
            } else {
                k
            };
            line(k as usize)
        };
        let mut sum = 0.0f32;
        for o in -13isize..=13 {
            let tap = match o.unsigned_abs() {
                0 => DECIMATION_TAPS[0],
                a if a % 2 == 1 => DECIMATION_TAPS[a.div_ceil(2)],
                _ => continue,
            };
            sum += at(pos as isize + o) * tap;
        }
        sum
    };
    let mut horiz = vec![0.0f32; hgt * pw];
    for y in 0..hgt {
        let row = &full[y * w..(y + 1) * w];
        for i in 0..pw {
            horiz[y * pw + i] = h(filter(&|k| row[k], w, 2 * i));
        }
    }
    let mut out = vec![0.0f32; ph * pw];
    for i in 0..pw {
        for j in 0..ph {
            out[j * pw + i] = h(filter(&|k| horiz[k * pw + i], hgt, 2 * j));
        }
    }
    Ok(out)
}

/// Reconstruct a full-resolution plane from a sub-sampled one by
/// bilinear interpolation between sample positions (the fallback for
/// sampling factors other than 1×1 / 2×2). `sub` must hold `ceil(width
/// / sx) × ceil(height / sy)` samples; `sx == sy == 1` returns a copy.
pub fn upsample_bilinear(
    sub: &[f32],
    width: u32,
    height: u32,
    sx: u32,
    sy: u32,
) -> Result<Vec<f32>> {
    let (sx, sy) = (sx.max(1), sy.max(1));
    check_sub_len(sub, width, height, sx, sy)?;
    if sx == 1 && sy == 1 {
        return Ok(sub.to_vec());
    }
    let pw = subsampled_dim(width, sx) as usize;
    let ph = subsampled_dim(height, sy) as usize;
    let (w, hgt) = (width as usize, height as usize);
    let mut out = vec![0.0f32; w * hgt];
    let (sxf, syf) = (sx as f32, sy as f32);
    // Horizontal lerp of every sample row into a `w`-wide scratch row.
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(ph);
    for jy in 0..ph {
        let src = &sub[jy * pw..(jy + 1) * pw];
        let mut row = vec![0.0f32; w];
        for (x, v) in row.iter_mut().enumerate() {
            let i0 = x / sx as usize;
            let i1 = (i0 + 1).min(pw - 1);
            let t = (x - i0 * sx as usize) as f32 / sxf;
            *v = src[i0] + (src[i1] - src[i0]) * t;
        }
        rows.push(row);
    }
    for y in 0..hgt {
        let j0 = y / sy as usize;
        let j1 = (j0 + 1).min(ph - 1);
        let t = (y - j0 * sy as usize) as f32 / syf;
        let (r0, r1) = (&rows[j0], &rows[j1]);
        let dst = &mut out[y * w..(y + 1) * w];
        for x in 0..w {
            dst[x] = r0[x] + (r1[x] - r0[x]) * t;
        }
    }
    Ok(out)
}

/// Reduce a full-resolution plane to `(sx, sy)` sampling with a
/// separable tent filter centred on each sample position (the fallback
/// for sampling factors other than 1×1 / 2×2). Returns `ceil(width /
/// sx) × ceil(height / sy)` samples; `sx == sy == 1` returns a copy.
pub fn downsample_tent(
    full: &[f32],
    width: u32,
    height: u32,
    sx: u32,
    sy: u32,
) -> Result<Vec<f32>> {
    let (sx, sy) = (sx.max(1), sy.max(1));
    let (w, hgt) = (width as usize, height as usize);
    if full.len() != w * hgt {
        return Err(ExrError::invalid(format!(
            "plane holds {} samples, expected {w}×{hgt}",
            full.len()
        )));
    }
    if sx == 1 && sy == 1 {
        return Ok(full.to_vec());
    }
    let pw = subsampled_dim(width, sx) as usize;
    let ph = subsampled_dim(height, sy) as usize;
    let (sxi, syi) = (sx as isize, sy as isize);
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(hgt);
    for y in 0..hgt {
        let src = &full[y * w..(y + 1) * w];
        let mut row = Vec::with_capacity(pw);
        for jx in 0..pw {
            let cx = (jx * sx as usize) as isize;
            let (mut acc, mut wsum) = (0.0f64, 0.0f64);
            for dx in (1 - sxi)..sxi {
                let x = cx + dx;
                if x < 0 || x >= w as isize {
                    continue;
                }
                let wt = (sxi - dx.abs()) as f64;
                acc += f64::from(src[x as usize]) * wt;
                wsum += wt;
            }
            row.push((acc / wsum) as f32);
        }
        rows.push(row);
    }
    let mut out = Vec::with_capacity(pw * ph);
    for jy in 0..ph {
        let cy = (jy * sy as usize) as isize;
        let mut acc = vec![0.0f64; pw];
        let mut wsum = 0.0f64;
        for dy in (1 - syi)..syi {
            let y = cy + dy;
            if y < 0 || y >= hgt as isize {
                continue;
            }
            let wt = (syi - dy.abs()) as f64;
            for (a, &v) in acc.iter_mut().zip(&rows[y as usize]) {
                *a += f64::from(v) * wt;
            }
            wsum += wt;
        }
        out.extend(acc.iter().map(|&a| (a / wsum) as f32));
    }
    Ok(out)
}

/// One chroma plane with its sampling factors, as stored in the file.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ChromaPlane<'a> {
    /// `ceil(width / x_sampling) × ceil(height / y_sampling)` samples.
    pub samples: &'a [f32],
    pub x_sampling: u32,
    pub y_sampling: u32,
}

impl<'a> ChromaPlane<'a> {
    /// Build `ChromaPlane` from every field, in declaration order.
    pub fn new(samples: &'a [f32], x_sampling: u32, y_sampling: u32) -> Self {
        Self {
            samples,
            x_sampling,
            y_sampling,
        }
    }
}

/// Full-resolution RGB planes reconstructed from luminance/chroma.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RgbPlanes {
    pub r: Vec<f32>,
    pub g: Vec<f32>,
    pub b: Vec<f32>,
}

impl RgbPlanes {
    /// Build `RgbPlanes` from every field, in declaration order.
    pub fn new(r: Vec<f32>, g: Vec<f32>, b: Vec<f32>) -> Self {
        Self { r, g, b }
    }
}

/// Full-resolution chroma for rows `-1 ..= height - 1` (module docs,
/// step 1 and the saturation ring).
fn chroma_with_ring(p: ChromaPlane<'_>, width: u32, height: u32) -> Result<Vec<f32>> {
    let (w, hgt) = (width as usize, height as usize);
    let (sx, sy) = (p.x_sampling.max(1), p.y_sampling.max(1));
    if (sx, sy) == (2, 2) {
        check_sub_len(p.samples, width, height, 2, 2)?;
        return Ok(reconstruct_2x2_with_ring(p.samples, w, hgt));
    }
    // 1×1 (a copy) or another factor (bilinear); row -1 repeats row 0.
    let full = upsample_bilinear(p.samples, width, height, sx, sy)?;
    let mut out = Vec::with_capacity((hgt + 1) * w);
    out.extend_from_slice(&full[..w]);
    out.extend_from_slice(&full);
    Ok(out)
}

/// Conversion of one pixel (module docs, step 2). A zero `Y` gives
/// zeros whose signs follow the factors (`(RY + 1)·±0`), as stored.
#[inline]
fn to_rgb(y: f32, ry: f32, by: f32, [wr, wg, wb]: [f32; 3]) -> [f32; 3] {
    let r = (ry + 1.0) * y;
    let b = (by + 1.0) * y;
    let g = (y - r * wr - b * wb) / wg;
    [h(r), h(g), h(b)]
}

#[inline]
fn max3(p: [f32; 3]) -> f32 {
    p[0].max(p[1]).max(p[2])
}

#[inline]
fn saturation(p: [f32; 3]) -> f32 {
    let mx = max3(p);
    let mn = p[0].min(p[1]).min(p[2]);
    if mx > 0.0 { 1.0 - mn / mx } else { 0.0 }
}

#[inline]
fn luminance([r, g, b]: [f32; 3], [wr, wg, wb]: [f32; 3]) -> f32 {
    r * wr + g * wg + b * wb
}

/// Saturation limiting of one pixel against its four diagonal
/// neighbours' saturations (module docs, step 3).
#[inline]
fn limit_saturation(p: [f32; 3], diag: [f32; 4], w: [f32; 3]) -> [f32; 3] {
    let s = saturation(p);
    let avg = (diag[0] + diag[1] + diag[2] + diag[3]) * 0.25;
    let t = (1.0 - (1.0 - avg) * 0.25).min(1.0);
    if s <= t {
        return p;
    }
    // The interpolation toward the maximum runs in binary64 on the
    // binary32 operands (`f` itself is binary32) and is rounded once.
    let f = f64::from(t / s);
    let mx = f64::from(max3(p));
    let lift = mx * (1.0 - f);
    let q = p.map(|c| h(((f64::from(c) * f + lift) as f32).max(0.0)));
    let k = luminance(p, w) / luminance(q, w);
    if !k.is_finite() {
        return q;
    }
    q.map(|c| h(c * k))
}

/// Reconstruct full-resolution `R`, `G`, `B` planes from a
/// full-resolution `y` plane and the `ry` / `by` chroma planes (module
/// docs, *Decoding*). `y` must hold `width × height` samples.
pub fn luma_chroma_to_rgb(
    width: u32,
    height: u32,
    y: &[f32],
    ry: ChromaPlane<'_>,
    by: ChromaPlane<'_>,
    weights: [f32; 3],
) -> Result<RgbPlanes> {
    let (w, hgt) = (width as usize, height as usize);
    let pixels = w * hgt;
    if y.len() != pixels {
        return Err(ExrError::invalid(format!(
            "Y plane holds {} samples, expected {width}×{height}",
            y.len()
        )));
    }
    if pixels == 0 {
        return Ok(RgbPlanes::new(Vec::new(), Vec::new(), Vec::new()));
    }
    let ry_ring = chroma_with_ring(ry, width, height)?;
    let by_ring = chroma_with_ring(by, width, height)?;
    // Converted rows -1 ..= height - 1 (row -1: row 0's luminance with
    // the row -1 chroma).
    let mut rgb = vec![[0.0f32; 3]; (hgt + 1) * w];
    for ring_row in 0..=hgt {
        let ly = ring_row.saturating_sub(1);
        for x in 0..w {
            let i = ring_row * w + x;
            rgb[i] = to_rgb(y[ly * w + x], ry_ring[i], by_ring[i], weights);
        }
    }
    let sat: Vec<f32> = rgb.iter().map(|&p| saturation(p)).collect();
    // Saturation of the (extended) pixel at row `yy` ∈ [-1, H], column
    // `xx` ∈ [-1, W]: columns replicate the edge, row H is row H − 2.
    let sat_at = |yy: isize, xx: isize| -> f32 {
        let yy = if yy >= hgt as isize {
            (hgt as isize - 2).max(0)
        } else {
            yy
        };
        let xx = xx.clamp(0, w as isize - 1);
        sat[(yy + 1) as usize * w + xx as usize]
    };
    let mut r = Vec::with_capacity(pixels);
    let mut g = Vec::with_capacity(pixels);
    let mut b = Vec::with_capacity(pixels);
    for yy in 0..hgt as isize {
        for xx in 0..w as isize {
            let p = rgb[(yy + 1) as usize * w + xx as usize];
            let diag = [
                sat_at(yy - 1, xx - 1),
                sat_at(yy - 1, xx + 1),
                sat_at(yy + 1, xx - 1),
                sat_at(yy + 1, xx + 1),
            ];
            let [pr, pg, pb] = limit_saturation(p, diag, weights);
            r.push(pr);
            g.push(pg);
            b.push(pb);
        }
    }
    Ok(RgbPlanes { r, g, b })
}

/// Luminance/chroma planes produced from RGB.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct LumaChromaPlanes {
    /// Full-resolution luminance, `width × height`.
    pub y: Vec<f32>,
    /// `RY` at `(x_sampling, y_sampling)`.
    pub ry: Vec<f32>,
    /// `BY` at `(x_sampling, y_sampling)`.
    pub by: Vec<f32>,
    pub x_sampling: u32,
    pub y_sampling: u32,
}

impl LumaChromaPlanes {
    /// Build `LumaChromaPlanes` from every field, in declaration order.
    pub fn new(y: Vec<f32>, ry: Vec<f32>, by: Vec<f32>, x_sampling: u32, y_sampling: u32) -> Self {
        Self {
            y,
            ry,
            by,
            x_sampling,
            y_sampling,
        }
    }
}

/// Convert full-resolution `rgb = [r, g, b]` planes to a full-resolution
/// `Y` plane plus `RY` / `BY` chroma reduced to `sampling = (x_sampling,
/// y_sampling)`, with the stored-value mantissa rounding (7-bit `Y`,
/// 5-bit `RY` / `BY`) on — the module docs' *Encoding*. Each input must
/// hold `width × height` samples. See [`rgb_to_luma_chroma_with`] to
/// turn the mantissa rounding off.
pub fn rgb_to_luma_chroma(
    width: u32,
    height: u32,
    rgb: [&[f32]; 3],
    weights: [f32; 3],
    sampling: (u32, u32),
) -> Result<LumaChromaPlanes> {
    rgb_to_luma_chroma_with(width, height, rgb, weights, sampling, true)
}

/// [`rgb_to_luma_chroma`] with the stored-value mantissa rounding
/// selectable: `round_mantissas = false` keeps the full binary16
/// mantissa of every stored sample (larger files, a slightly more
/// accurate round trip).
pub fn rgb_to_luma_chroma_with(
    width: u32,
    height: u32,
    rgb: [&[f32]; 3],
    weights: [f32; 3],
    sampling: (u32, u32),
    round_mantissas: bool,
) -> Result<LumaChromaPlanes> {
    let [r, g, b] = rgb;
    let (sx, sy) = (sampling.0.max(1), sampling.1.max(1));
    let pixels = (width as usize) * (height as usize);
    for (name, p) in [("R", r), ("G", g), ("B", b)] {
        if p.len() != pixels {
            return Err(ExrError::invalid(format!(
                "{name} plane holds {} samples, expected {width}×{height}",
                p.len()
            )));
        }
    }
    let [wr, wg, wb] = weights;
    let mut y = Vec::with_capacity(pixels);
    let mut ry_full = Vec::with_capacity(pixels);
    let mut by_full = Vec::with_capacity(pixels);
    for i in 0..pixels {
        let yy = h(r[i] * wr + g[i] * wg + b[i] * wb);
        y.push(yy);
        if yy == 0.0 || !yy.is_finite() {
            ry_full.push(0.0);
            by_full.push(0.0);
        } else {
            ry_full.push(h((r[i] - yy) / yy));
            by_full.push(h((b[i] - yy) / yy));
        }
    }
    let (mut ry, mut by) = match (sx, sy) {
        (1, 1) => (ry_full, by_full),
        (2, 2) => (
            decimate_chroma_2x2(&ry_full, width, height)?,
            decimate_chroma_2x2(&by_full, width, height)?,
        ),
        _ => (
            downsample_tent(&ry_full, width, height, sx, sy)?,
            downsample_tent(&by_full, width, height, sx, sy)?,
        ),
    };
    if round_mantissas {
        for v in &mut y {
            *v = h_keep(*v, Y_MANTISSA_BITS);
        }
        for v in ry.iter_mut().chain(by.iter_mut()) {
            *v = h_keep(*v, CHROMA_MANTISSA_BITS);
        }
    }
    Ok(LumaChromaPlanes {
        y,
        ry,
        by,
        x_sampling: sx,
        y_sampling: sy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_match_the_tabulated_binary32_bits() {
        let w = luminance_weights(&BT709_CHROMATICITIES);
        assert_eq!(w.map(f32::to_bits), [0x3e59_be0b, 0x3f37_154b, 0x3d93_d991]);
        let ap0 = Chromaticities {
            red_x: 0.7347,
            red_y: 0.2653,
            green_x: 0.0,
            green_y: 1.0,
            blue_x: 0.0001,
            blue_y: -0.077,
            white_x: 0.32168,
            white_y: 0.33767,
        };
        assert_eq!(
            luminance_weights(&ap0).map(f32::to_bits),
            [0x3eb0_1c5f, 0x3f3a_6918, 0xbd93_ba3b]
        );
        let attrs = vec![Attribute {
            name: "chromaticities".to_string(),
            value: AttributeValue::Chromaticities(ap0),
        }];
        assert_eq!(luminance_weights_of(&attrs), luminance_weights(&ap0));
        assert_eq!(luminance_weights_of(&[]), w);
    }

    #[test]
    fn degenerate_chromaticities_fall_back_to_bt709() {
        let bt709 = luminance_weights(&BT709_CHROMATICITIES);
        let bad = Chromaticities {
            white_y: 0.0,
            ..BT709_CHROMATICITIES
        };
        assert_eq!(luminance_weights(&bad), bt709);
        let collinear = Chromaticities {
            red_x: 0.3,
            red_y: 0.6,
            ..BT709_CHROMATICITIES
        };
        assert_eq!(luminance_weights(&collinear), bt709);
    }

    #[test]
    fn mantissa_rounding_is_half_up_in_magnitude() {
        // 1.0 + 3/1024 → 7 bits: the dropped 3 bits are 0b011 < half.
        assert_eq!(round_half_mantissa(0x3c03, 7), 0x3c00);
        // 0b100 is the tie: rounds away from zero, either sign.
        assert_eq!(round_half_mantissa(0x3c04, 7), 0x3c08);
        assert_eq!(round_half_mantissa(0xbc04, 7), 0xbc08);
        // A carry moves into the exponent.
        assert_eq!(round_half_mantissa(0x3ffc, 7), 0x4000);
        assert_eq!(round_half_mantissa(0x3ff0, 5), 0x4000);
        // Infinity / NaN untouched; no rounding up into infinity.
        assert_eq!(round_half_mantissa(0x7c00, 5), 0x7c00);
        assert_eq!(round_half_mantissa(0x7e01, 5), 0x7e01);
        assert_eq!(round_half_mantissa(0x7bff, 7), 0x7bf8);
        assert_eq!(round_half_mantissa(0x1234, 10), 0x1234);
    }

    #[test]
    fn reconstruction_copies_even_positions_and_keeps_constants() {
        let sub = vec![0.375f32; 6 * 5];
        let full = reconstruct_chroma_2x2(&sub, 12, 10).unwrap();
        assert!(full.iter().all(|&v| v == 0.375), "{full:?}");
        // An impulse lands unattenuated on its co-sited luma position
        // and nowhere else on even rows / columns.
        let mut sub = vec![0.0f32; 16 * 16];
        sub[8 * 16 + 8] = 1.0;
        let full = reconstruct_chroma_2x2(&sub, 32, 32).unwrap();
        for y in (0..32).step_by(2) {
            for x in (0..32).step_by(2) {
                let expect = if (x, y) == (16, 16) { 1.0 } else { 0.0 };
                assert_eq!(full[y * 32 + x], expect, "({x},{y})");
            }
        }
        // Odd neighbours carry the first tap (to binary16).
        assert_eq!(full[16 * 32 + 17], h(RECONSTRUCTION_TAPS[0]));
        assert!(reconstruct_chroma_2x2(&sub, 34, 32).is_err());
    }

    #[test]
    fn decimation_of_an_even_impulse_is_the_squared_centre_tap() {
        let mut full = vec![0.0f32; 32 * 32];
        full[16 * 32 + 16] = 1.0;
        let sub = decimate_chroma_2x2(&full, 32, 32).unwrap();
        assert_eq!(sub.len(), 16 * 16);
        let centre = h(h(DECIMATION_TAPS[0]) * DECIMATION_TAPS[0]);
        assert_eq!(sub[8 * 16 + 8], centre);
        assert_eq!(sub.iter().filter(|&&v| v != 0.0).count(), 1);
        assert!(decimate_chroma_2x2(&full, 31, 32).is_err());
    }

    #[test]
    fn grey_and_black_convert_without_saturation_changes() {
        let wts = luminance_weights(&BT709_CHROMATICITIES);
        let (w, hgt) = (4u32, 4u32);
        let y = vec![0.5f32; 16];
        let zero = vec![0.0f32; 4];
        let rgb = luma_chroma_to_rgb(
            w,
            hgt,
            &y,
            ChromaPlane::new(&zero, 2, 2),
            ChromaPlane::new(&zero, 2, 2),
            wts,
        )
        .unwrap();
        assert!(rgb.r.iter().all(|&v| v == 0.5));
        assert!(rgb.b.iter().all(|&v| v == 0.5));
        assert!(rgb.g.iter().all(|&v| (v - 0.5).abs() < 1e-3));
        // Y = 0 is black.
        let rgb = luma_chroma_to_rgb(
            w,
            hgt,
            &[0.0; 16],
            ChromaPlane::new(&[0.5; 4], 2, 2),
            ChromaPlane::new(&[-0.25; 4], 2, 2),
            wts,
        )
        .unwrap();
        assert!(rgb.r.iter().chain(&rgb.g).chain(&rgb.b).all(|&v| v == 0.0));
    }

    #[test]
    fn negative_components_are_clamped_by_the_limiter() {
        // Y = 1, RY = -2 everywhere → R = -1: saturation > 1 → t = 1,
        // the smallest component is driven to 0 at constant luminance.
        let wts = luminance_weights(&BT709_CHROMATICITIES);
        let rgb = luma_chroma_to_rgb(
            2,
            2,
            &[1.0; 4],
            ChromaPlane::new(&[-2.0], 2, 2),
            ChromaPlane::new(&[0.0], 2, 2),
            wts,
        )
        .unwrap();
        for i in 0..4 {
            // Driven to zero up to a binary16 sub-normal residue.
            assert!(rgb.r[i].abs() < 1e-4, "{}", rgb.r[i]);
            assert!((rgb.g[i] - 1.298).abs() < 2e-3, "{}", rgb.g[i]);
            assert_eq!(rgb.b[i], 1.0);
        }
    }

    #[test]
    fn encode_then_decode_a_constant_colour_is_close() {
        let wts = luminance_weights(&BT709_CHROMATICITIES);
        let (w, hgt) = (16u32, 8u32);
        let n = (w * hgt) as usize;
        let (r, g, b) = (vec![0.8f32; n], vec![0.4f32; n], vec![0.2f32; n]);
        let yc = rgb_to_luma_chroma(w, hgt, [&r, &g, &b], wts, (2, 2)).unwrap();
        assert_eq!((yc.ry.len(), yc.x_sampling), (n / 4, 2));
        // Stored values carry the reduced mantissas.
        for &v in &yc.y {
            assert_eq!(f32_to_half(v) & 0x7, 0);
        }
        for &v in yc.ry.iter().chain(&yc.by) {
            assert_eq!(f32_to_half(v) & 0x1f, 0);
        }
        let back = luma_chroma_to_rgb(
            w,
            hgt,
            &yc.y,
            ChromaPlane::new(&yc.ry, 2, 2),
            ChromaPlane::new(&yc.by, 2, 2),
            wts,
        )
        .unwrap();
        for i in 0..n {
            for (got, want) in [(back.r[i], 0.8), (back.g[i], 0.4), (back.b[i], 0.2)] {
                assert!((got - want).abs() / want < 0.01, "{got} vs {want}");
            }
        }
        // Without the mantissa rounding: full binary16 storage.
        let ramp: Vec<f32> = (0..n).map(|i| 0.1 + i as f32 * 0.0123).collect();
        let yc = rgb_to_luma_chroma_with(w, hgt, [&ramp, &g, &b], wts, (1, 1), false).unwrap();
        assert_eq!(yc.ry.len(), n);
        assert!(yc.y.iter().any(|&v| f32_to_half(v) & 0x7 != 0));
    }

    #[test]
    fn other_samplings_use_the_bilinear_and_tent_fallbacks() {
        let sub = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let up = upsample_bilinear(&sub, 5, 3, 2, 2).unwrap();
        assert_eq!(up.len(), 15);
        assert_eq!(
            (up[0], up[1], up[4], up[5], up[14]),
            (1.0, 1.5, 3.0, 2.5, 6.0)
        );
        assert_eq!(upsample_bilinear(&sub, 3, 2, 1, 1).unwrap(), sub.to_vec());
        assert!(upsample_bilinear(&sub, 5, 3, 3, 3).is_err());
        assert!(downsample_tent(&sub, 5, 3, 2, 2).is_err());
        let wts = luminance_weights(&BT709_CHROMATICITIES);
        let p = vec![0.5f32; 64];
        let yc = rgb_to_luma_chroma(8, 8, [&p, &p, &p], wts, (4, 4)).unwrap();
        assert_eq!(yc.ry.len(), 4);
        let back = luma_chroma_to_rgb(
            8,
            8,
            &yc.y,
            ChromaPlane::new(&yc.ry, 4, 4),
            ChromaPlane::new(&yc.by, 4, 4),
            wts,
        )
        .unwrap();
        assert!(back.r.iter().all(|&v| (v - 0.5).abs() < 5e-3));
    }
}
