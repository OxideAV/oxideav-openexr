//! OpenEXR RLE compression (compression code 1).
//!
//! Layout per the openexr.com "Pixel Data Compression" appendix:
//!
//! Each compressed scanline block is the output of the following pipeline:
//!   1. Channel-then-row-major-byte-flat raw byte stream.
//!   2. ZIP-style byte-half interleave (low halves first, then high
//!      halves).
//!   3. ZIP-style byte predictor (`b[i] -= b[i-1]` mod 256).
//!   4. Byte-oriented RLE.
//!
//! On decode the steps run in reverse: RLE expand, then unpredictor,
//! then uninterleave. The interleave and predictor passes are shared
//! with ZIP / ZIPS (see [`crate::decoder`]).
//!
//! Byte-oriented RLE control byte `c` (signed):
//!   * `c >= 0` (in `0..=127`): the next `c + 1` bytes are a literal
//!     run.
//!   * `c < 0` (in `-127..=-1`, `-128` reserved): the next byte is
//!     repeated `1 - c` times (so `c = -1` → 2 repeats, `c = -127` →
//!     128 repeats).

use crate::error::{ExrError, Result};

/// Byte-oriented RLE compressor. Always emits the smaller of literal
/// vs repeat for ambiguous (length-2) runs. Rounds-trip-pairs with
/// [`rle_decompress`].
pub fn rle_compress(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len());
    let mut i = 0;
    while i < src.len() {
        // How long is the current byte run (capped at 128, the largest
        // run a single repeat header can encode)?
        let b = src[i];
        let mut run = 1;
        while i + run < src.len() && run < 128 && src[i + run] == b {
            run += 1;
        }
        if run >= 3 {
            // Repeat shape: 2 bytes encode `run` source bytes (>=3).
            // c = 1 - run yields range [-127..=-2] for run in [3..=128].
            out.push((1i32 - run as i32) as i8 as u8);
            out.push(b);
            i += run;
        } else {
            // Literal run: extend until either we hit 128 bytes or we
            // see a 3+ byte repeat ahead that would beat continuing the
            // literal.
            let lit_start = i;
            i += run;
            while i < src.len() && i - lit_start < 128 {
                let b2 = src[i];
                let mut run2 = 1;
                while i + run2 < src.len() && run2 < 128 && src[i + run2] == b2 {
                    run2 += 1;
                }
                if run2 >= 3 {
                    break;
                }
                i += run2;
            }
            let lit_len = i - lit_start;
            out.push((lit_len as i32 - 1) as u8);
            out.extend_from_slice(&src[lit_start..i]);
        }
    }
    out
}

/// Decompress a byte-oriented RLE stream produced by [`rle_compress`]
/// or by an OpenEXR reference encoder.
pub fn rle_decompress(src: &[u8], expected_size: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(expected_size);
    let mut i = 0;
    while i < src.len() {
        let c = src[i] as i8;
        i += 1;
        if c == -128 {
            return Err(ExrError::invalid(
                "RLE stream contains reserved control byte -128".to_string(),
            ));
        }
        if c >= 0 {
            let n = c as usize + 1;
            if i + n > src.len() {
                return Err(ExrError::invalid(format!(
                    "RLE literal run of {n} bytes truncated at offset {i}"
                )));
            }
            out.extend_from_slice(&src[i..i + n]);
            i += n;
        } else {
            let n = (1 - c as i32) as usize;
            if i >= src.len() {
                return Err(ExrError::invalid(format!(
                    "RLE repeat-marker missing payload byte at offset {i}"
                )));
            }
            let b = src[i];
            i += 1;
            for _ in 0..n {
                out.push(b);
            }
        }
        if out.len() > expected_size {
            return Err(ExrError::invalid(format!(
                "RLE expansion exceeded expected size {expected_size} (got {})",
                out.len()
            )));
        }
    }
    if out.len() != expected_size {
        return Err(ExrError::invalid(format!(
            "RLE expansion produced {} bytes, expected {expected_size}",
            out.len()
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_roundtrip() {
        let out = rle_compress(&[]);
        assert!(out.is_empty());
        let back = rle_decompress(&out, 0).unwrap();
        assert!(back.is_empty());
    }

    #[test]
    fn single_byte() {
        let src = [0x42u8];
        let out = rle_compress(&src);
        assert_eq!(out, vec![0, 0x42]);
        let back = rle_decompress(&out, 1).unwrap();
        assert_eq!(back, src);
    }

    #[test]
    fn long_repeat() {
        let src = vec![0x55u8; 10];
        let out = rle_compress(&src);
        assert_eq!(out.len(), 2);
        let back = rle_decompress(&out, 10).unwrap();
        assert_eq!(back, src);
    }

    #[test]
    fn alternating_then_repeat() {
        let mut src = vec![1u8, 2, 3, 4, 5];
        src.extend(std::iter::repeat_n(0xAA, 8));
        let out = rle_compress(&src);
        let back = rle_decompress(&out, src.len()).unwrap();
        assert_eq!(back, src);
    }

    #[test]
    fn long_literal_split_at_128() {
        let src: Vec<u8> = (0..200).map(|i| (i as u8).wrapping_mul(17)).collect();
        let out = rle_compress(&src);
        let back = rle_decompress(&out, src.len()).unwrap();
        assert_eq!(back, src);
    }

    #[test]
    fn long_repeat_split_at_128() {
        let src = vec![0xCCu8; 300];
        let out = rle_compress(&src);
        let back = rle_decompress(&out, src.len()).unwrap();
        assert_eq!(back, src);
    }

    #[test]
    fn rejects_reserved_control() {
        let bad = [0x80u8, 0x00];
        assert!(rle_decompress(&bad, 1).is_err());
    }

    #[test]
    fn rejects_truncated_literal() {
        let bad = [0x05u8, 0, 0, 0];
        assert!(rle_decompress(&bad, 6).is_err());
    }

    #[test]
    fn rejects_truncated_repeat() {
        let bad = [0xFBu8];
        assert!(rle_decompress(&bad, 6).is_err());
    }

    #[test]
    fn rejects_size_mismatch() {
        let stream = [0x00u8, 0x42];
        assert!(rle_decompress(&stream, 5).is_err());
    }
}
