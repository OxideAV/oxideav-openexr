//! Luminance / chroma (`Y` + `RY` + `BY`) ↔ RGB conversion.
//!
//! OpenEXR files may store colour as a full-resolution luminance
//! channel `Y` plus two (usually 2×2 sub-sampled) chroma channels `RY`
//! and `BY`. The container-level readers and writers in this crate
//! have handled that layout bit-exactly for a long time; this module
//! adds the *colour* reconstruction so the framework path (and any
//! standalone caller) can get RGB out of such a file and write one
//! from RGB.
//!
//! # Definition
//!
//! With luminance weights `Yw = (Yw.r, Yw.g, Yw.b)`:
//!
//! ```text
//! Y  = Yw.r · R + Yw.g · G + Yw.b · B
//! RY = (R − Y) / Y
//! BY = (B − Y) / Y
//! ```
//!
//! and the exact inverse
//!
//! ```text
//! R = (RY + 1) · Y
//! B = (BY + 1) · Y
//! G = (Y − Yw.r · R − Yw.b · B) / Yw.g
//! ```
//!
//! The weights are the `Y` row of the RGB → CIE XYZ matrix built from
//! the image's `chromaticities` attribute (the primaries and white
//! point of its RGB space); when the attribute is absent the Rec. ITU-R
//! BT.709 primaries and D65 white apply, which give the familiar
//! `(0.2126, 0.7152, 0.0722)`. The derivation is standard colorimetry:
//! for primaries `(x_i, y_i)` the unscaled XYZ columns are
//! `(x_i / y_i, 1, (1 − x_i − y_i) / y_i)`; solving `P · s = W` for the
//! white point `W = (x_w / y_w, 1, (1 − x_w − y_w) / y_w)` yields the
//! per-primary scale `s`, and because every column has unit Y the
//! luminance row is exactly `s`.
//!
//! When `Y == 0` the chroma ratios are undefined; the encoder writes
//! `RY = BY = 0` there (the pixel is black whatever the chroma) and the
//! decoder reconstructs black.
//!
//! # Sub-sampling
//!
//! A chroma channel with sampling `(sx, sy)` carries one sample per
//! `sx × sy` block, positioned at the block's top-left pixel
//! `(jx · sx, jy · sy)` — the same convention the container-level
//! chunk layout uses (a sub-sampled row is present only on scanlines
//! whose `y` is a multiple of `sy`). Reconstruction interpolates
//! **bilinearly** between the four surrounding sample positions,
//! clamping at the right and bottom edges. Reduction applies a
//! separable **tent** filter centred on the sample position (weights
//! `sx − |dx|` horizontally, `sy − |dy|` vertically, taps outside the
//! image dropped and the remaining weights renormalised): it is
//! symmetric about the sample, so a locally linear image reduces and
//! reconstructs exactly away from the edges.
//!
//! The container-level chunk layout is bit-exact by construction; the
//! colour-level reconstruction is validated against a reference EXR
//! tool as an opaque process (see `tests/luma_chroma_validation.rs`).
//! That reference applies its own reconstruction filter, so the
//! cross-check is a tolerance comparison, not a bit match.

use crate::decoder::subsampled_dim;
use crate::error::{ExrError, Result};
use crate::types::{Attribute, AttributeValue, Chromaticities};

/// Rec. ITU-R BT.709 primaries with the D65 white point — the colour
/// space assumed when a file carries no `chromaticities` attribute.
pub const BT709_CHROMATICITIES: Chromaticities = Chromaticities {
    red_x: 0.6400,
    red_y: 0.3300,
    green_x: 0.3000,
    green_y: 0.6000,
    blue_x: 0.1500,
    blue_y: 0.0600,
    white_x: 0.3127,
    white_y: 0.3290,
};

/// Luminance weights `(Yw.r, Yw.g, Yw.b)` of the RGB space described by
/// `c` — the `Y` row of its RGB → XYZ matrix (module docs). Computed in
/// `f64` and rounded once. Returns the BT.709 weights when the
/// primaries are degenerate (a zero `y` coordinate or a singular
/// primary triangle), which cannot describe a colour space anyway.
pub fn luminance_weights(c: &Chromaticities) -> [f32; 3] {
    fn column(x: f32, y: f32) -> Option<[f64; 3]> {
        let (x, y) = (f64::from(x), f64::from(y));
        if y == 0.0 || !y.is_finite() || !x.is_finite() {
            return None;
        }
        Some([x / y, 1.0, (1.0 - x - y) / y])
    }
    let solve = || -> Option<[f32; 3]> {
        let r = column(c.red_x, c.red_y)?;
        let g = column(c.green_x, c.green_y)?;
        let b = column(c.blue_x, c.blue_y)?;
        let w = column(c.white_x, c.white_y)?;
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
        if s.iter().any(|v| !v.is_finite()) {
            return None;
        }
        Some([s[0] as f32, s[1] as f32, s[2] as f32])
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

/// Reconstruct a full-resolution plane from a sub-sampled one by
/// bilinear interpolation between sample positions (module docs,
/// *Sub-sampling*). `sub` must hold `ceil(width / sx) × ceil(height /
/// sy)` samples; `sx == sy == 1` returns a copy.
pub fn upsample_bilinear(
    sub: &[f32],
    width: u32,
    height: u32,
    sx: u32,
    sy: u32,
) -> Result<Vec<f32>> {
    let (sx, sy) = (sx.max(1), sy.max(1));
    let pw = subsampled_dim(width, sx) as usize;
    let ph = subsampled_dim(height, sy) as usize;
    if sub.len() != pw * ph {
        return Err(ExrError::invalid(format!(
            "sub-sampled plane holds {} samples, expected {pw}×{ph} for {width}×{height} at \
             {sx}×{sy}",
            sub.len()
        )));
    }
    if sx == 1 && sy == 1 {
        return Ok(sub.to_vec());
    }
    let (w, h) = (width as usize, height as usize);
    let mut out = vec![0.0f32; w * h];
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
    for y in 0..h {
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
/// separable tent filter centred on each sample position (module docs,
/// *Sub-sampling*). Returns `ceil(width / sx) × ceil(height / sy)`
/// samples; `sx == sy == 1` returns a copy.
pub fn downsample_tent(
    full: &[f32],
    width: u32,
    height: u32,
    sx: u32,
    sy: u32,
) -> Result<Vec<f32>> {
    let (sx, sy) = (sx.max(1), sy.max(1));
    let (w, h) = (width as usize, height as usize);
    if full.len() != w * h {
        return Err(ExrError::invalid(format!(
            "plane holds {} samples, expected {w}×{h}",
            full.len()
        )));
    }
    if sx == 1 && sy == 1 {
        return Ok(full.to_vec());
    }
    let pw = subsampled_dim(width, sx) as usize;
    let ph = subsampled_dim(height, sy) as usize;
    let (sxi, syi) = (sx as isize, sy as isize);
    // Horizontal pass: every image row → `pw` samples.
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(h);
    for y in 0..h {
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
    // Vertical pass over the reduced rows.
    let mut out = Vec::with_capacity(pw * ph);
    for jy in 0..ph {
        let cy = (jy * sy as usize) as isize;
        let mut acc = vec![0.0f64; pw];
        let mut wsum = 0.0f64;
        for dy in (1 - syi)..syi {
            let y = cy + dy;
            if y < 0 || y >= h as isize {
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
pub struct ChromaPlane<'a> {
    /// `ceil(width / x_sampling) × ceil(height / y_sampling)` samples.
    pub samples: &'a [f32],
    pub x_sampling: u32,
    pub y_sampling: u32,
}

/// Full-resolution RGB planes reconstructed from luminance/chroma.
#[derive(Debug, Clone, PartialEq)]
pub struct RgbPlanes {
    pub r: Vec<f32>,
    pub g: Vec<f32>,
    pub b: Vec<f32>,
}

/// Reconstruct full-resolution `R`, `G`, `B` planes from a
/// full-resolution `y` plane and the `ry` / `by` chroma planes (module
/// docs). `y` must hold `width × height` samples.
pub fn luma_chroma_to_rgb(
    width: u32,
    height: u32,
    y: &[f32],
    ry: ChromaPlane<'_>,
    by: ChromaPlane<'_>,
    weights: [f32; 3],
) -> Result<RgbPlanes> {
    let pixels = (width as usize) * (height as usize);
    if y.len() != pixels {
        return Err(ExrError::invalid(format!(
            "Y plane holds {} samples, expected {width}×{height}",
            y.len()
        )));
    }
    let ry_full = upsample_bilinear(ry.samples, width, height, ry.x_sampling, ry.y_sampling)?;
    let by_full = upsample_bilinear(by.samples, width, height, by.x_sampling, by.y_sampling)?;
    let [wr, wg, wb] = weights;
    let mut r = Vec::with_capacity(pixels);
    let mut g = Vec::with_capacity(pixels);
    let mut b = Vec::with_capacity(pixels);
    for i in 0..pixels {
        let yy = y[i];
        let rr = (ry_full[i] + 1.0) * yy;
        let bb = (by_full[i] + 1.0) * yy;
        r.push(rr);
        b.push(bb);
        g.push((yy - wr * rr - wb * bb) / wg);
    }
    Ok(RgbPlanes { r, g, b })
}

/// Luminance/chroma planes produced from RGB.
#[derive(Debug, Clone, PartialEq)]
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

/// Convert full-resolution `rgb = [r, g, b]` planes to a full-resolution
/// `Y` plane plus `RY` / `BY` chroma reduced to `sampling = (x_sampling,
/// y_sampling)` (module docs). Each input must hold `width × height`
/// samples.
pub fn rgb_to_luma_chroma(
    width: u32,
    height: u32,
    rgb: [&[f32]; 3],
    weights: [f32; 3],
    sampling: (u32, u32),
) -> Result<LumaChromaPlanes> {
    let [r, g, b] = rgb;
    let (x_sampling, y_sampling) = sampling;
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
        let yy = wr * r[i] + wg * g[i] + wb * b[i];
        y.push(yy);
        if yy == 0.0 || !yy.is_finite() {
            ry_full.push(0.0);
            by_full.push(0.0);
        } else {
            ry_full.push((r[i] - yy) / yy);
            by_full.push((b[i] - yy) / yy);
        }
    }
    let ry = downsample_tent(&ry_full, width, height, x_sampling, y_sampling)?;
    let by = downsample_tent(&by_full, width, height, x_sampling, y_sampling)?;
    Ok(LumaChromaPlanes {
        y,
        ry,
        by,
        x_sampling: x_sampling.max(1),
        y_sampling: y_sampling.max(1),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn bt709_weights_match_the_published_constants() {
        let w = luminance_weights(&BT709_CHROMATICITIES);
        assert!(close(w[0], 0.2126, 2e-4), "{w:?}");
        assert!(close(w[1], 0.7152, 2e-4), "{w:?}");
        assert!(close(w[2], 0.0722, 2e-4), "{w:?}");
        assert!(close(w[0] + w[1] + w[2], 1.0, 1e-6), "{w:?}");
    }

    #[test]
    fn weights_follow_the_chromaticities_attribute() {
        // Wide-gamut primaries: a negative blue weight is legitimate.
        let wide = Chromaticities {
            red_x: 0.7347,
            red_y: 0.2653,
            green_x: 0.0,
            green_y: 1.0,
            blue_x: 0.0001,
            blue_y: -0.077,
            white_x: 0.32168,
            white_y: 0.33767,
        };
        let w = luminance_weights(&wide);
        assert!(close(w[0], 0.34397, 1e-4), "{w:?}");
        assert!(close(w[1], 0.72817, 1e-4), "{w:?}");
        assert!(close(w[2], -0.07213, 1e-4), "{w:?}");
        let attrs = vec![Attribute {
            name: "chromaticities".to_string(),
            value: AttributeValue::Chromaticities(wide),
        }];
        assert_eq!(luminance_weights_of(&attrs), w);
        assert_eq!(
            luminance_weights_of(&[]),
            luminance_weights(&BT709_CHROMATICITIES)
        );
    }

    #[test]
    fn degenerate_chromaticities_fall_back_to_bt709() {
        let bad = Chromaticities {
            red_y: 0.0,
            ..BT709_CHROMATICITIES
        };
        assert_eq!(
            luminance_weights(&bad),
            luminance_weights(&BT709_CHROMATICITIES)
        );
        let collinear = Chromaticities {
            red_x: 0.3,
            red_y: 0.6,
            ..BT709_CHROMATICITIES
        };
        assert_eq!(
            luminance_weights(&collinear),
            luminance_weights(&BT709_CHROMATICITIES)
        );
    }

    #[test]
    fn full_resolution_round_trip_is_tight() {
        let (w, h) = (7u32, 5u32);
        let n = (w * h) as usize;
        let r: Vec<f32> = (0..n).map(|i| 0.05 + (i as f32) * 0.37 % 3.0).collect();
        let g: Vec<f32> = (0..n).map(|i| 1.5 - (i as f32) * 0.11 % 1.4).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.23 % 0.9).collect();
        let wts = luminance_weights(&BT709_CHROMATICITIES);
        let yc = rgb_to_luma_chroma(w, h, [&r, &g, &b], wts, (1, 1)).unwrap();
        assert_eq!(yc.ry.len(), n);
        let back = luma_chroma_to_rgb(
            w,
            h,
            &yc.y,
            ChromaPlane {
                samples: &yc.ry,
                x_sampling: 1,
                y_sampling: 1,
            },
            ChromaPlane {
                samples: &yc.by,
                x_sampling: 1,
                y_sampling: 1,
            },
            wts,
        )
        .unwrap();
        for i in 0..n {
            let tol = 1e-5 * (1.0 + r[i].abs() + g[i].abs() + b[i].abs());
            assert!(
                close(back.r[i], r[i], tol),
                "R px{i} {} vs {}",
                back.r[i],
                r[i]
            );
            assert!(
                close(back.g[i], g[i], tol),
                "G px{i} {} vs {}",
                back.g[i],
                g[i]
            );
            assert!(
                close(back.b[i], b[i], tol),
                "B px{i} {} vs {}",
                back.b[i],
                b[i]
            );
        }
    }

    #[test]
    fn black_pixels_produce_zero_chroma_and_reconstruct_black() {
        let wts = luminance_weights(&BT709_CHROMATICITIES);
        let z = [0.0f32; 4];
        let yc = rgb_to_luma_chroma(2, 2, [&z, &z, &z], wts, (2, 2)).unwrap();
        assert_eq!(yc.ry, vec![0.0]);
        assert_eq!(yc.by, vec![0.0]);
        let back = luma_chroma_to_rgb(
            2,
            2,
            &yc.y,
            ChromaPlane {
                samples: &yc.ry,
                x_sampling: 2,
                y_sampling: 2,
            },
            ChromaPlane {
                samples: &yc.by,
                x_sampling: 2,
                y_sampling: 2,
            },
            wts,
        )
        .unwrap();
        assert!(back
            .r
            .iter()
            .chain(&back.g)
            .chain(&back.b)
            .all(|&v| v == 0.0));
    }

    #[test]
    fn tent_then_bilinear_reproduces_a_linear_ramp_in_the_interior() {
        let (w, h) = (16u32, 12u32);
        let full: Vec<f32> = (0..h)
            .flat_map(|y| (0..w).map(move |x| 0.25 * x as f32 - 0.125 * y as f32 + 3.0))
            .collect();
        let sub = downsample_tent(&full, w, h, 2, 2).unwrap();
        assert_eq!(sub.len(), 8 * 6);
        // Interior sample positions are exact (symmetric taps).
        assert!(close(sub[8 + 1], full[2 * 16 + 2], 1e-6));
        assert!(close(sub[3 * 8 + 5], full[6 * 16 + 10], 1e-6));
        let back = upsample_bilinear(&sub, w, h, 2, 2).unwrap();
        for y in 2..(h as usize - 2) {
            for x in 2..(w as usize - 2) {
                let i = y * w as usize + x;
                assert!(
                    close(back[i], full[i], 1e-5),
                    "px({x},{y}) {} vs {}",
                    back[i],
                    full[i]
                );
            }
        }
    }

    #[test]
    fn upsample_handles_odd_extents_and_unit_sampling() {
        // 5×3 at 2×2 sampling → 3×2 samples; the last column/row clamp.
        let sub = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let up = upsample_bilinear(&sub, 5, 3, 2, 2).unwrap();
        assert_eq!(up.len(), 15);
        assert_eq!(up[0], 1.0);
        assert_eq!(up[1], 1.5);
        assert_eq!(up[4], 3.0);
        assert_eq!(up[5], 2.5); // halfway between rows 0 and 1
        assert_eq!(up[14], 6.0);
        assert_eq!(upsample_bilinear(&sub, 3, 2, 1, 1).unwrap(), sub.to_vec());
        assert!(upsample_bilinear(&sub, 5, 3, 3, 3).is_err());
        assert!(downsample_tent(&sub, 5, 3, 2, 2).is_err());
    }
}
