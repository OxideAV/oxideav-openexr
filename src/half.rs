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
pub fn f32_to_half(f: f32) -> u16 {
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
