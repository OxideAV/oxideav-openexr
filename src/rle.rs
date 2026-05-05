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
//!   * `c >= 0` (in `0..=127`): the next byte is repeated `c + 1` times.
//!   * `c < 0` (in `-1..=-127`): the next `−c` bytes are a literal run.
//!   * `c == −128` (0x80): the next byte is repeated 129 times.
//!
//! Note: OpenEXR's RLE uses the opposite sign convention from TIFF PackBits —
//! non-negative control bytes encode REPEAT runs, negative control bytes encode
//! LITERAL runs. The value −128 (0x80) follows the "no special treatment" rule:
//! repeat 1−(−128) = 129 times.

use crate::error::{ExrError, Result};

/// Byte-oriented RLE compressor. Round-trips with [`rle_decompress`].
///
/// Encoding rules (OpenEXR convention):
///   * `c >= 0`: repeat next byte `c + 1` times.
///   * `c < 0`: literal run of `−c` bytes follow.
pub fn rle_compress(src: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(src.len());
    let mut i = 0;
    while i < src.len() {
        let b = src[i];
        // Count consecutive identical bytes (repeat run), capped at 128.
        let mut run = 1;
        while i + run < src.len() && run < 128 && src[i + run] == b {
            run += 1;
        }
        if run >= 2 {
            // Repeat run: control byte = run - 1 (>= 0, since run >= 2).
            out.push((run as i32 - 1) as u8);
            out.push(b);
            i += run;
        } else {
            // Literal run: scan forward collecting non-repeating bytes.
            // Stop when we see 2+ identical bytes ahead or hit 127 literals.
            let lit_start = i;
            i += 1;
            while i < src.len() && i - lit_start < 127 {
                let b2 = src[i];
                // Peek: is the next byte a duplicate?
                if i + 1 < src.len() && src[i + 1] == b2 {
                    break; // a repeat run starts here; end the literal run
                }
                i += 1;
            }
            let lit_len = i - lit_start;
            // control byte = -lit_len (< 0 for lit_len >= 1).
            out.push(-(lit_len as i32) as i8 as u8);
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
        if c >= 0 {
            // Repeat run: next byte repeated (c + 1) times.
            // c == 127 (i8 max): 128 repeats; c == -128: 129 repeats (no special case).
            let n = c as usize + 1;
            if i >= src.len() {
                return Err(ExrError::invalid(format!(
                    "RLE repeat-marker missing payload byte at offset {i}"
                )));
            }
            let b = src[i];
            i += 1;
            if out.len() + n > expected_size {
                return Err(ExrError::invalid(format!(
                    "RLE expansion exceeded expected size {expected_size} (got {}+{n})",
                    out.len()
                )));
            }
            for _ in 0..n {
                out.push(b);
            }
        } else {
            // Literal run: the next (-c) bytes are literal.
            // c == -128 (i8::MIN): 128 literal bytes (since -(-128) = 128 in two's complement;
            // but i8 negation overflows, so we use `0i32 - c as i32`).
            let n = (0i32 - c as i32) as usize;
            if i + n > src.len() {
                return Err(ExrError::invalid(format!(
                    "RLE literal run of {n} bytes truncated at offset {i}"
                )));
            }
            if out.len() + n > expected_size {
                return Err(ExrError::invalid(format!(
                    "RLE expansion exceeded expected size {expected_size} (got {}+{n})",
                    out.len()
                )));
            }
            out.extend_from_slice(&src[i..i + n]);
            i += n;
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
        // Single byte: repeat 1 → c = 0, then 0x42. OR literal 1 → c = -1, then 0x42.
        // Our encoder uses repeat for any run >= 2; single byte uses literal.
        // c = -1 (0xFF), then 0x42.
        let back = rle_decompress(&out, 1).unwrap();
        assert_eq!(back, src);
    }

    #[test]
    fn long_repeat() {
        let src = vec![0x55u8; 10];
        let out = rle_compress(&src);
        // 10 repeats → c = 9 (0x09), then 0x55
        assert_eq!(out.len(), 2);
        let back = rle_decompress(&out, 10).unwrap();
        assert_eq!(back, src);
    }

    #[test]
    fn repeat_encodes_as_nonneg_control() {
        // Verify the sign convention: repeat uses c >= 0.
        let src = vec![0xAAu8; 5];
        let compressed = rle_compress(&src);
        // Expect: c = 4 (repeat 5), then 0xAA
        assert_eq!(compressed, vec![4u8, 0xAA]);
        let back = rle_decompress(&compressed, 5).unwrap();
        assert_eq!(back, src);
    }

    #[test]
    fn literal_encodes_as_neg_control() {
        // Verify the sign convention: literal uses c < 0.
        let src = vec![1u8, 2, 3, 4, 5];
        let compressed = rle_compress(&src);
        // All distinct → one literal run of 5: c = -5 (0xFB), then 1 2 3 4 5
        assert_eq!(compressed[0] as i8, -5);
        let back = rle_decompress(&compressed, src.len()).unwrap();
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
    fn long_literal_split_at_127() {
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
    fn rejects_truncated_literal() {
        // c = -5 (0xFB) but only 3 bytes follow instead of 5
        let bad = [0xFBu8, 0, 0, 0];
        assert!(rle_decompress(&bad, 5).is_err());
    }

    #[test]
    fn rejects_truncated_repeat() {
        // c = 5 (repeat 6 times) but no next byte
        let bad = [0x05u8];
        assert!(rle_decompress(&bad, 6).is_err());
    }

    #[test]
    fn rejects_size_mismatch() {
        // Produces 1 byte but expected 5
        let stream = rle_compress(&[0x42u8]);
        assert!(rle_decompress(&stream, 5).is_err());
    }

    #[test]
    fn repeat_128_via_control_0x7f() {
        // c = 127 (0x7F, i8 max) → repeat 128 times
        let data = vec![0xBBu8; 128];
        let compressed = rle_compress(&data);
        let back = rle_decompress(&compressed, 128).unwrap();
        assert_eq!(back, data);
    }

    #[test]
    fn repeat_run_capped_at_128() {
        // 200 repeating bytes → two repeat runs (128 + 72)
        let src = vec![0xDDu8; 200];
        let out = rle_compress(&src);
        // First chunk: c=127 (128 reps), then c=71 (72 reps)
        assert!(
            out.len() <= 6,
            "expected at most 6 bytes for two repeat runs"
        );
        let back = rle_decompress(&out, 200).unwrap();
        assert_eq!(back, src);
    }
}
