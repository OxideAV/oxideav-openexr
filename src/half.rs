//! IEEE 754-2008 binary16 (`half`) <-> `f32` conversion.
//!
//! Bit layout (s1.e5.m10):
//! * bit 15: sign
//! * bits 14..10: biased 5-bit exponent (bias 15)
//! * bits 9..0: 10-bit fraction
//!
//! Special encodings:
//! * exponent == 0     => subnormal (or zero if fraction == 0)
//! * exponent == 0x1F  => infinity (fraction == 0) or NaN (fraction != 0)
//!
//! Used for the EXR `HALF` channel pixel type.
//!
//! These two functions are bit-exact mirrors: round-tripping every
//! representable `half` through `half_to_f32` followed by `f32_to_half`
//! returns the original 16-bit pattern (NaN payload bits aside, which
//! are not architecturally guaranteed by the IEEE spec). The unit
//! tests at the bottom assert this for all 65536 patterns.

/// Decode a binary16 bit pattern to `f32`.
pub fn half_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 0x1;
    let exp = (h >> 10) & 0x1F;
    let frac = h & 0x3FF;

    let s32: u32 = (sign as u32) << 31;

    if exp == 0 {
        if frac == 0 {
            // signed zero
            return f32::from_bits(s32);
        }
        // Subnormal: value = (-1)^s * 2^-14 * (frac / 1024)
        // Re-normalise into f32.
        let mut m = frac as u32;
        let mut e: i32 = -14;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let exp_f32 = (e + 127) as u32;
        return f32::from_bits(s32 | (exp_f32 << 23) | (m << 13));
    }
    if exp == 0x1F {
        // Inf or NaN. Propagate fraction to f32 so a quiet-NaN stays quiet.
        let mantissa = (frac as u32) << 13;
        return f32::from_bits(s32 | (0xFFu32 << 23) | mantissa);
    }
    // Normalised. f32 exponent = (h_exp - 15) + 127.
    let exp_f32 = (exp as u32 + (127 - 15)) << 23;
    let mantissa = (frac as u32) << 13;
    f32::from_bits(s32 | exp_f32 | mantissa)
}

/// Encode `f32` to binary16 bit pattern with round-half-to-even.
///
/// Branch-light form (the DWA decoder converts every texel through
/// this, and the round-457 profile put ~38% of DWA decode time in the
/// previous cascade of range tests): the classification is done on
/// the magnitude bits, rounding adds `0x0FFF + lsb` before the 13-bit
/// shift so a mantissa carry rolls into the exponent by itself, and
/// the subnormal path uses the same add-then-shift rounding at its
/// wider shift. Bit-identical to [`f32_to_half_reference`] for every
/// `f32` pattern (pinned by `fast_f32_to_half_matches_the_reference`).
pub fn f32_to_half(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let abs = bits & 0x7fff_ffff;
    if abs >= 0x7f80_0000 {
        // Inf or NaN — keep the top mantissa bits, force a NaN to stay
        // a NaN.
        if abs == 0x7f80_0000 {
            return sign | 0x7c00;
        }
        let mut m = ((abs >> 13) & 0x3ff) as u16;
        if m == 0 {
            m = 1;
        }
        return sign | 0x7c00 | m;
    }
    if abs >= 0x4780_0000 {
        // Exponent above the half range: overflow to infinity.
        return sign | 0x7c00;
    }
    if abs >= 0x3880_0000 {
        // Normal half: round the 13 dropped bits to nearest-even and
        // rebias the exponent (127 → 15). A rounding carry propagates
        // into the exponent, and past the top exponent lands exactly
        // on the infinity pattern.
        let lsb = (abs >> 13) & 1;
        let rounded = (abs + 0x0fff + lsb) >> 13;
        return sign | (rounded - (112 << 10)) as u16;
    }
    if abs < 0x3380_0000 {
        // Below 2^-24: underflow to signed zero.
        return sign;
    }
    // Subnormal half: insert the implicit one and shift out
    // `126 - exponent` bits (13 plus the extra exponent deficit),
    // rounding to nearest-even; a carry to 0x400 is the smallest normal.
    let m = (abs & 0x007f_ffff) | 0x0080_0000;
    let shift = 126 - (abs >> 23);
    let lsb = (m >> shift) & 1;
    let rounded = (m + (1u32 << (shift - 1)) - 1 + lsb) >> shift;
    sign | rounded as u16
}

/// Straight-line reference encoder (the crate's original
/// implementation): explicit range tests and [`round_to_nearest_even`].
/// Kept as the oracle for the fast path.
#[cfg(test)]
pub(crate) fn f32_to_half_reference(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 0x1) as u16;
    let exp_f32 = ((bits >> 23) & 0xFF) as i32;
    let frac_f32 = bits & 0x007F_FFFF;

    // NaN / Inf
    if exp_f32 == 0xFF {
        if frac_f32 != 0 {
            // NaN — keep top mantissa bits, force at least one bit set.
            let mut m = (frac_f32 >> 13) as u16 & 0x3FF;
            if m == 0 {
                m = 1;
            }
            return (sign << 15) | (0x1F << 10) | m;
        }
        // Infinity
        return (sign << 15) | (0x1F << 10);
    }

    // Unbias f32 exponent.
    let unbiased = exp_f32 - 127;

    if unbiased > 15 {
        // Overflow to infinity.
        return (sign << 15) | (0x1F << 10);
    }
    if unbiased >= -14 {
        // Normal half.
        let exp_h = (unbiased + 15) as u16;
        // Round to nearest, ties to even on the dropped low 13 bits.
        let mant = round_to_nearest_even(frac_f32, 13) as u16;
        // Rounding may overflow into the exponent.
        if mant == 0x400 {
            // bumped to next exponent
            let exp_h2 = exp_h + 1;
            if exp_h2 >= 0x1F {
                return (sign << 15) | (0x1F << 10); // -> infinity
            }
            return (sign << 15) | (exp_h2 << 10);
        }
        return (sign << 15) | (exp_h << 10) | mant;
    }
    // Subnormal half (or zero).
    if unbiased < -24 {
        // Underflow to signed zero.
        return sign << 15;
    }
    // Insert implicit leading one and shift right (-14 - unbiased) extra
    // bits, on top of the standard 13-bit drop.
    let mant_with_implicit = frac_f32 | 0x0080_0000;
    let shift = (13 + (-14 - unbiased)) as u32;
    let mant = round_to_nearest_even(mant_with_implicit, shift) as u16;
    // mant is at most 0x400 here too (rounding can produce a normal).
    if mant == 0x400 {
        return (sign << 15) | (1 << 10);
    }
    (sign << 15) | mant
}

/// Round `value` right-shifted by `shift` bits to nearest, ties-to-even.
#[cfg(test)]
fn round_to_nearest_even(value: u32, shift: u32) -> u32 {
    if shift == 0 {
        return value;
    }
    let half = 1u32 << (shift - 1);
    let mask = (1u32 << shift) - 1;
    let dropped = value & mask;
    let kept = value >> shift;
    if dropped > half {
        kept + 1
    } else if dropped < half {
        kept
    } else {
        // Exactly half: round to even.
        kept + (kept & 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_f32_to_half_matches_the_reference() {
        // Every pattern with the low 8 bits clear (16.7M values) covers
        // all exponents and every rounding neighbourhood; the dense
        // sweeps below cover the exact tie / carry / boundary bits.
        let mut bits = 0u32;
        loop {
            assert_eq!(
                f32_to_half(f32::from_bits(bits)),
                f32_to_half_reference(f32::from_bits(bits)),
                "{bits:#010x}"
            );
            bits = bits.wrapping_add(256);
            if bits == 0 {
                break;
            }
        }
        let dense = [
            0x3380_0000u32, // 2^-24: smallest subnormal boundary
            0x3300_0000,    // 2^-25: below it
            0x3880_0000,    // 2^-14: smallest normal
            0x387f_ffff,
            0x477f_e000, // largest finite half neighbourhood
            0x477f_f000,
            0x4780_0000, // 2^16: overflow
            0x3f80_0000, // 1.0
            0x7f7f_ffff, // f32::MAX
            0x7f80_0000, // inf
            0x7f80_0001, // NaN with tiny payload
            0x7fc0_0000, // quiet NaN
        ];
        for &base in &dense {
            for d in 0..0x4000u32 {
                for b in [
                    base.wrapping_add(d),
                    base.wrapping_sub(d),
                    base.wrapping_add(d) | 0x8000_0000,
                ] {
                    assert_eq!(
                        f32_to_half(f32::from_bits(b)),
                        f32_to_half_reference(f32::from_bits(b)),
                        "{b:#010x}"
                    );
                }
            }
        }
    }

    #[test]
    fn roundtrip_zero() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert!(half_to_f32(0x8000).is_sign_negative());
        assert_eq!(f32_to_half(0.0), 0x0000);
        assert_eq!(f32_to_half(-0.0), 0x8000);
    }

    #[test]
    fn roundtrip_one() {
        // half(1.0) = 0x3C00 (sign 0, exp 15 = bias, frac 0)
        assert_eq!(half_to_f32(0x3C00), 1.0);
        assert_eq!(f32_to_half(1.0), 0x3C00);
    }

    #[test]
    fn roundtrip_neg_one() {
        assert_eq!(half_to_f32(0xBC00), -1.0);
        assert_eq!(f32_to_half(-1.0), 0xBC00);
    }

    #[test]
    fn roundtrip_inf() {
        assert!(half_to_f32(0x7C00).is_infinite() && half_to_f32(0x7C00).is_sign_positive());
        assert!(half_to_f32(0xFC00).is_infinite() && half_to_f32(0xFC00).is_sign_negative());
        assert_eq!(f32_to_half(f32::INFINITY), 0x7C00);
        assert_eq!(f32_to_half(f32::NEG_INFINITY), 0xFC00);
    }

    #[test]
    fn roundtrip_nan() {
        assert!(half_to_f32(0x7E00).is_nan());
        let h = f32_to_half(f32::NAN);
        let exp = (h >> 10) & 0x1F;
        let frac = h & 0x3FF;
        assert_eq!(exp, 0x1F);
        assert_ne!(frac, 0);
    }

    #[test]
    fn smallest_subnormal() {
        // half(min subnormal) = 0x0001 = 2^-24
        let v = half_to_f32(0x0001);
        assert!((v - 2f32.powi(-24)).abs() < 1e-30);
        assert_eq!(f32_to_half(2f32.powi(-24)), 0x0001);
    }

    #[test]
    fn largest_normal() {
        // half(largest finite) = 0x7BFF = 65504
        assert_eq!(half_to_f32(0x7BFF), 65504.0);
        assert_eq!(f32_to_half(65504.0), 0x7BFF);
    }

    #[test]
    fn overflow_to_inf() {
        // 70000 > 65504, must overflow.
        assert_eq!(f32_to_half(70000.0), 0x7C00);
    }

    #[test]
    fn underflow_to_zero() {
        // 1e-30 is below 2^-24 so it underflows.
        assert_eq!(f32_to_half(1e-30), 0x0000);
    }

    #[test]
    fn roundtrip_all_finite_halves() {
        // Every finite/zero/subnormal half pattern (i.e. excluding NaN
        // payloads, which the IEEE spec doesn't guarantee preserved
        // through narrow->wide->narrow) must round-trip exactly.
        for h in 0u16..=0xFFFF {
            let exp = (h >> 10) & 0x1F;
            let frac = h & 0x3FF;
            // Skip NaN payloads (exp == 0x1F && frac != 0). Their
            // payload survives our impl but the spec-guarantee is just
            // "is_nan" so don't assert pattern equality.
            if exp == 0x1F && frac != 0 {
                let f = half_to_f32(h);
                assert!(f.is_nan());
                continue;
            }
            let f = half_to_f32(h);
            let h2 = f32_to_half(f);
            assert_eq!(h, h2, "h={h:#06x} -> f={f} -> {h2:#06x}");
        }
    }
}
