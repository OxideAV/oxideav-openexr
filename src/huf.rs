//! Shared static-Huffman payload container used by PIZ chunks and by the
//! DWAA / DWAB AC sub-stream in its `acCompression == 0` mode
//! (observer-spec `openexr-piz-dwa-observer-spec.md` §2.5).
//!
//! Wire layout: a 20-byte header of five little-endian `u32` —
//! `im` (lowest symbol index with a non-zero code length), `iM` (highest
//! such index, which is *also* the run-length escape symbol), the byte
//! length of the packed code-length table (advisory: the unpacker
//! consumes symbols `im..=iM` and the entropy data begins wherever it
//! stops), the entropy-coded bit count, and a reserved zero — followed
//! by the packed code lengths, followed by the entropy-coded data.
//!
//! The alphabet has 65 537 symbols (the 16-bit values plus one escape
//! slot above the highest value present). Codes are canonical: only
//! lengths travel, max length 58, and both sides rebuild the same codes
//! by sweeping lengths 58 down to 1 with `first(i) = c` then
//! `c = (c + n[i]) >> 1`. Bits are packed most-significant-first with
//! the final partial byte left-aligned.

use crate::error::{ExrError, Result};

/// Total alphabet size: 65 536 possible 16-bit values + 1 escape slot.
const ALPHABET: usize = 65537;
/// Maximum canonical code length in bits (observer-spec §2.5).
const MAX_CODE_LEN: usize = 58;
/// Fast decode-table width: every code of length <= 14 resolves in one
/// lookup (observer-spec §2.5 "decode table shape" note).
const FAST_BITS: usize = 14;

// ---------------------------------------------------------------------------
// Bit I/O (MSB-first, left-aligned final byte)
// ---------------------------------------------------------------------------

struct BitWriter {
    out: Vec<u8>,
    /// Pending bits, left-aligned (the next bit to emit is bit 63).
    acc: u64,
    /// Valid bits currently held in `acc` (< 64 between calls).
    nacc: u32,
    bits_written: u64,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            out: Vec::new(),
            acc: 0,
            nacc: 0,
            bits_written: 0,
        }
    }

    /// Append the low `n` bits of `v`, most-significant-first. `n` is
    /// always in 1..=58 here (6-bit table symbols, 8-bit run counts,
    /// codes up to the 58-bit maximum). Whole 8-byte accumulators are
    /// flushed in one extend instead of per-byte pushes.
    #[inline]
    fn put(&mut self, v: u64, n: u32) {
        debug_assert!((1..=58).contains(&n));
        self.bits_written += n as u64;
        let mut n = n;
        let mut v = v & ((1u64 << n) - 1);
        let room = 64 - self.nacc;
        if n > room {
            // Top up the accumulator to exactly 64 bits and flush it.
            let spill = n - room; // bits that don't fit (< 64)
            if room > 0 {
                self.acc |= v >> spill;
            }
            self.out.extend_from_slice(&self.acc.to_be_bytes());
            self.acc = 0;
            self.nacc = 0;
            n = spill;
            v &= (1u64 << spill) - 1;
        }
        if n > 0 {
            self.acc |= v << (64 - self.nacc - n);
            self.nacc += n;
        }
        if self.nacc == 64 {
            self.out.extend_from_slice(&self.acc.to_be_bytes());
            self.acc = 0;
            self.nacc = 0;
        }
    }

    /// Flush the trailing bytes, the final partial byte left-aligned
    /// (high end filled).
    fn finish(mut self) -> (Vec<u8>, u64) {
        while self.nacc > 0 {
            self.out.push((self.acc >> 56) as u8);
            self.acc <<= 8;
            self.nacc = self.nacc.saturating_sub(8);
        }
        (self.out, self.bits_written)
    }
}

struct BitReader<'a> {
    data: &'a [u8],
    /// Next byte of `data` not yet loaded into `acc`.
    byte_pos: usize,
    /// Upcoming bits, left-aligned (the next bit to read is bit 63).
    acc: u64,
    /// Valid bits currently held in `acc`.
    nacc: u32,
    /// Bits consumed so far (MSB of byte 0 is bit 0).
    pos: u64,
    /// Total available bits.
    limit: u64,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], limit_bits: u64) -> Self {
        BitReader {
            data,
            byte_pos: 0,
            acc: 0,
            nacc: 0,
            pos: 0,
            limit: limit_bits.min(data.len() as u64 * 8),
        }
    }

    /// Top up the accumulator to at least 56 valid bits (or to the end
    /// of the data). Bulk path loads up to seven bytes from one
    /// unaligned eight-byte read; the tail falls back to per-byte
    /// loads.
    #[inline]
    fn refill(&mut self) {
        if self.nacc <= 55 {
            if let Some(chunk) = self.data.get(self.byte_pos..self.byte_pos + 8) {
                let w = u64::from_be_bytes(chunk.try_into().unwrap());
                let fill = (63 - self.nacc) >> 3; // whole bytes that fit (>= 1)
                self.acc |= (w & (!0u64 << (64 - fill * 8))) >> self.nacc;
                self.byte_pos += fill as usize;
                self.nacc += fill * 8;
            } else {
                while self.nacc <= 55 && self.byte_pos < self.data.len() {
                    self.acc |= (self.data[self.byte_pos] as u64) << (56 - self.nacc);
                    self.byte_pos += 1;
                    self.nacc += 8;
                }
            }
        }
    }

    /// Read `n` bits (1..=56) MSB-first. Errors past the declared bit
    /// limit.
    #[inline]
    fn get(&mut self, n: u32) -> Result<u64> {
        debug_assert!((1..=56).contains(&n));
        if self.pos + n as u64 > self.limit {
            return Err(ExrError::invalid(
                "Huffman payload: bit stream exhausted".to_string(),
            ));
        }
        self.refill();
        debug_assert!(self.nacc >= n);
        let v = self.acc >> (64 - n);
        self.acc <<= n;
        self.nacc -= n;
        self.pos += n as u64;
        Ok(v)
    }

    /// Peek `n` bits (1..=56) without consuming; bits past the limit
    /// read as zero.
    #[inline]
    fn peek(&mut self, n: u32) -> u64 {
        debug_assert!((1..=56).contains(&n));
        self.refill();
        let raw = self.acc >> (64 - n);
        let avail = (self.limit - self.pos).min(n as u64) as u32;
        if avail >= n {
            raw
        } else if avail == 0 {
            0
        } else {
            // The final loaded byte may carry padding past the declared
            // bit count; zero it so table lookups see the same window
            // the bit-exact reader would.
            (raw >> (n - avail)) << (n - avail)
        }
    }

    #[inline]
    fn skip(&mut self, n: u32) {
        debug_assert!(self.nacc >= n);
        self.acc <<= n;
        self.nacc -= n;
        self.pos += n as u64;
    }
}

// ---------------------------------------------------------------------------
// Canonical code construction (shared by encode and decode)
// ---------------------------------------------------------------------------

/// Given per-symbol code lengths (index space `im..=iM`, sparse), build
/// the canonical code for each coded symbol. Returns `(codes, first)`
/// where `codes[sym]` is the code value (right-aligned) for symbols with
/// non-zero length, and `first[len]` is the smallest code value of each
/// length (used by the decoder's slow path).
fn canonical_codes(lengths: &[u8], syms: &[u32]) -> ([u64; MAX_CODE_LEN + 1], Vec<u64>) {
    let mut n = [0u64; MAX_CODE_LEN + 1];
    for &s in syms {
        n[lengths[s as usize] as usize] += 1;
    }
    let mut first = [0u64; MAX_CODE_LEN + 1];
    let mut c = 0u64;
    for i in (1..=MAX_CODE_LEN).rev() {
        first[i] = c;
        c = (c + n[i]) >> 1;
    }
    // Assign codes in increasing symbol order per length.
    let mut next = first;
    let mut codes = vec![0u64; syms.len()];
    for (i, &s) in syms.iter().enumerate() {
        let l = lengths[s as usize] as usize;
        codes[i] = next[l];
        next[l] += 1;
    }
    (first, codes)
}

// ---------------------------------------------------------------------------
// Code-length table packing (6-bit alphabet, observer-spec §2.5 +
// tables/piz-huf-lengthcode-alphabet.csv)
// ---------------------------------------------------------------------------

const SHORT_ZERO_RUN_BASE: u64 = 59; // symbol 59 = run of 2 zeros .. 62 = run of 5
const LONG_ZERO_RUN: u64 = 63; // + 8 raw bits, run of 6..=261

fn pack_code_lengths(lengths: &[u8], im: usize, i_m: usize) -> Vec<u8> {
    let mut w = BitWriter::new();
    let mut sym = im;
    while sym <= i_m {
        let l = lengths[sym];
        if l == 0 {
            // Count the zero run.
            let mut run = 1usize;
            while sym + run <= i_m && lengths[sym + run] == 0 && run < 261 {
                run += 1;
            }
            if run >= 6 {
                w.put(LONG_ZERO_RUN, 6);
                w.put((run - 6) as u64, 8);
            } else if run >= 2 {
                w.put(SHORT_ZERO_RUN_BASE + (run as u64 - 2), 6);
            } else {
                w.put(0, 6);
            }
            sym += run;
        } else {
            w.put(l as u64, 6);
            sym += 1;
        }
    }
    w.finish().0
}

/// Unpack the code-length stream for symbols `im..=iM`. Returns the
/// sparse length array (full alphabet size) plus the byte count consumed
/// (which is where the entropy data begins — the header's advisory
/// `tableLength` is deliberately not trusted, observer-spec §2.5).
fn unpack_code_lengths(data: &[u8], im: usize, i_m: usize) -> Result<(Vec<u8>, usize)> {
    let mut lengths = vec![0u8; ALPHABET];
    let mut r = BitReader::new(data, data.len() as u64 * 8);
    let mut sym = im;
    while sym <= i_m {
        let s = r.get(6)?;
        if s <= MAX_CODE_LEN as u64 {
            lengths[sym] = s as u8;
            sym += 1;
        } else if s < LONG_ZERO_RUN {
            let run = (s - SHORT_ZERO_RUN_BASE + 2) as usize;
            if sym + run > i_m + 1 {
                return Err(ExrError::invalid(
                    "Huffman payload: zero-length run past iM".to_string(),
                ));
            }
            sym += run;
        } else {
            let run = r.get(8)? as usize + 6;
            if sym + run > i_m + 1 {
                return Err(ExrError::invalid(
                    "Huffman payload: long zero-length run past iM".to_string(),
                ));
            }
            sym += run;
        }
    }
    Ok((lengths, r.pos.div_ceil(8) as usize))
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Decode a Huffman payload into exactly `expected` 16-bit values.
pub(crate) fn huf_decompress(payload: &[u8], expected: usize) -> Result<Vec<u16>> {
    if expected == 0 {
        return Ok(Vec::new());
    }
    if payload.len() < 20 {
        return Err(ExrError::invalid(format!(
            "Huffman payload header truncated ({} bytes)",
            payload.len()
        )));
    }
    let im = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    let i_m = u32::from_le_bytes(payload[4..8].try_into().unwrap()) as usize;
    let _table_len = u32::from_le_bytes(payload[8..12].try_into().unwrap());
    let n_bits = u32::from_le_bytes(payload[12..16].try_into().unwrap()) as u64;
    if im >= ALPHABET || i_m >= ALPHABET || im > i_m {
        return Err(ExrError::invalid(format!(
            "Huffman payload: symbol range {im}..{i_m} out of bounds"
        )));
    }
    let body = &payload[20..];
    let (lengths, table_bytes) = unpack_code_lengths(body, im, i_m)?;
    let data = &body[table_bytes..];
    if (data.len() as u64) * 8 < n_bits {
        return Err(ExrError::invalid(format!(
            "Huffman payload: {} data bytes cannot hold {n_bits} bits",
            data.len()
        )));
    }

    // Coded symbols in increasing index order.
    let syms: Vec<u32> = (im..=i_m)
        .filter(|&s| lengths[s] != 0)
        .map(|s| s as u32)
        .collect();
    if syms.is_empty() {
        return Err(ExrError::invalid(
            "Huffman payload: no coded symbols".to_string(),
        ));
    }
    let (first, codes) = canonical_codes(&lengths, &syms);

    // Per-length symbol ranges for the slow (>14-bit) path: the coded
    // symbol with rank r among length-l symbols (index order) has code
    // first[l] + r.
    let mut n_per_len = [0u32; MAX_CODE_LEN + 1];
    let mut per_len_syms: Vec<Vec<u32>> = vec![Vec::new(); MAX_CODE_LEN + 1];
    for &s in &syms {
        let l = lengths[s as usize] as usize;
        n_per_len[l] += 1;
        per_len_syms[l].push(s);
    }

    // Fast table: (symbol_index_u32 << 6) | length for codes <= FAST_BITS.
    // Escape is stored as the sentinel symbol value ALPHABET-1 marker via
    // a parallel "is escape" bit: encode entry as (sym << 7) | (is_esc << 6)?
    // Simpler: store u32 = (sym << 6) | len where sym is the true index
    // (fits in 26 bits).
    let mut fast = vec![0u32; 1 << FAST_BITS];
    for (i, &s) in syms.iter().enumerate() {
        let l = lengths[s as usize] as usize;
        if l <= FAST_BITS {
            let code = codes[i];
            let lo = (code << (FAST_BITS - l)) as usize;
            let hi = lo + (1usize << (FAST_BITS - l));
            for slot in &mut fast[lo..hi] {
                *slot = ((s << 6) | l as u32) + 1; // +1 so 0 = invalid
            }
        }
    }

    let mut r = BitReader::new(data, n_bits);
    let mut out: Vec<u16> = Vec::with_capacity(expected);
    let escape = i_m as u32;
    while out.len() < expected {
        let peeked = r.peek(FAST_BITS as u32);
        let entry = fast[peeked as usize];
        let sym: u32;
        if entry != 0 {
            let e = entry - 1;
            let l = (e & 63) as u32;
            if r.pos + l as u64 > r.limit {
                return Err(ExrError::invalid(
                    "Huffman payload: bit stream exhausted".to_string(),
                ));
            }
            r.skip(l);
            sym = e >> 6;
        } else {
            // Slow path: accumulate bits beyond FAST_BITS until a
            // per-length range matches.
            let mut acc = r.get(FAST_BITS as u32)?;
            let mut l = FAST_BITS;
            loop {
                if l >= MAX_CODE_LEN {
                    return Err(ExrError::invalid(
                        "Huffman payload: invalid code (no symbol within 58 bits)".to_string(),
                    ));
                }
                acc = (acc << 1) | r.get(1)?;
                l += 1;
                let n = n_per_len[l] as u64;
                if n != 0 && acc >= first[l] && acc < first[l] + n {
                    let rank = (acc - first[l]) as usize;
                    sym = per_len_syms[l][rank];
                    break;
                }
            }
        }
        if sym == escape {
            // Run escape: 8 raw bits = additional repeats of the previous
            // output value. Never legal as the first symbol.
            let cnt = r.get(8)? as usize;
            let prev = *out.last().ok_or_else(|| {
                ExrError::invalid("Huffman payload: run escape before any value".to_string())
            })?;
            if out.len() + cnt > expected {
                return Err(ExrError::invalid(
                    "Huffman payload: run overruns expected output".to_string(),
                ));
            }
            out.extend(std::iter::repeat(prev).take(cnt));
        } else {
            out.push((sym & 0xFFFF) as u16);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Build optimal prefix-code lengths for the given frequencies
/// (index-sparse; zero frequency = uncoded), capped at 58 bits. Standard
/// two-queue Huffman over a sorted leaf list; if the depth cap is ever
/// exceeded the frequencies are halved (rounding up) and the tree is
/// rebuilt, which strictly flattens it.
fn build_code_lengths(freqs: &[u64], lengths: &mut [u8]) {
    #[derive(Clone, Copy)]
    struct Node {
        weight: u64,
        // Leaf: symbol index. Internal: children indices into `nodes`.
        left: i32,
        right: i32,
        sym: i32,
    }

    let coded: Vec<usize> = (0..freqs.len()).filter(|&i| freqs[i] != 0).collect();
    if coded.is_empty() {
        return;
    }
    if coded.len() == 1 {
        lengths[coded[0]] = 1;
        return;
    }
    let mut fs: Vec<u64> = coded.iter().map(|&i| freqs[i]).collect();
    loop {
        // Heap of (weight, node index); tie-break on node index for
        // determinism.
        let mut nodes: Vec<Node> = coded
            .iter()
            .zip(fs.iter())
            .map(|(&s, &w)| Node {
                weight: w,
                left: -1,
                right: -1,
                sym: s as i32,
            })
            .collect();
        let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u64, usize)>> = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| std::cmp::Reverse((n.weight, i)))
            .collect();
        while heap.len() > 1 {
            let std::cmp::Reverse((wa, a)) = heap.pop().unwrap();
            let std::cmp::Reverse((wb, b)) = heap.pop().unwrap();
            let idx = nodes.len();
            nodes.push(Node {
                weight: wa + wb,
                left: a as i32,
                right: b as i32,
                sym: -1,
            });
            heap.push(std::cmp::Reverse((wa + wb, idx)));
        }
        // Depth-assign iteratively.
        let root = heap.pop().unwrap().0 .1;
        let mut stack = vec![(root, 0u8)];
        let mut max_depth = 0u8;
        let mut depths: Vec<(usize, u8)> = Vec::with_capacity(coded.len());
        while let Some((n, d)) = stack.pop() {
            let node = nodes[n];
            if node.sym >= 0 {
                depths.push((node.sym as usize, d.max(1)));
                max_depth = max_depth.max(d.max(1));
            } else {
                stack.push((node.left as usize, d + 1));
                stack.push((node.right as usize, d + 1));
            }
        }
        if max_depth as usize <= MAX_CODE_LEN {
            for (s, d) in depths {
                lengths[s] = d;
            }
            return;
        }
        // Flatten: halve frequencies (round up) and rebuild.
        for f in fs.iter_mut() {
            *f = (*f >> 1).max(1);
        }
    }
}

/// Encode 16-bit values into a self-contained Huffman payload
/// (observer-spec §2.5). Returns the payload bytes.
pub(crate) fn huf_compress(values: &[u16]) -> Result<Vec<u8>> {
    if values.is_empty() {
        return Err(ExrError::invalid(
            "Huffman payload: nothing to encode".to_string(),
        ));
    }
    let mut freqs = vec![0u64; ALPHABET];
    for &v in values {
        freqs[v as usize] += 1;
    }
    let im = freqs.iter().position(|&f| f != 0).unwrap();
    let max_val = freqs.iter().rposition(|&f| f != 0).unwrap();
    // The escape symbol sits one index above the highest value present
    // and always gets a code (frequency 1); iM is that escape index.
    let i_m = max_val + 1;
    freqs[i_m] = 1;

    let mut lengths = vec![0u8; ALPHABET];
    build_code_lengths(&freqs, &mut lengths);

    let syms: Vec<u32> = (im..=i_m)
        .filter(|&s| lengths[s] != 0)
        .map(|s| s as u32)
        .collect();
    let (_first, codes) = canonical_codes(&lengths, &syms);
    // Dense lookup: symbol -> (code, len).
    let mut code_of = vec![(0u64, 0u8); ALPHABET];
    for (i, &s) in syms.iter().enumerate() {
        code_of[s as usize] = (codes[i], lengths[s as usize]);
    }

    let table = pack_code_lengths(&lengths, im, i_m);

    // Entropy-code the values with the run-length escape.
    let (esc_code, esc_len) = code_of[i_m];
    let mut w = BitWriter::new();
    let mut i = 0usize;
    while i < values.len() {
        let v = values[i];
        // Count additional repeats, saturating at 255.
        let mut c = 0usize;
        while c < 255 && i + 1 + c < values.len() && values[i + 1 + c] == v {
            c += 1;
        }
        let (code, len) = code_of[v as usize];
        debug_assert!(len != 0);
        if (len as usize) + (esc_len as usize) + 8 < (len as usize) * c {
            w.put(code, len as u32);
            w.put(esc_code, esc_len as u32);
            w.put(c as u64, 8);
        } else {
            for _ in 0..=c {
                w.put(code, len as u32);
            }
        }
        i += 1 + c;
    }
    let (data, n_bits) = w.finish();
    if n_bits > u32::MAX as u64 {
        return Err(ExrError::invalid(
            "Huffman payload: bit count overflows u32".to_string(),
        ));
    }

    let mut out = Vec::with_capacity(20 + table.len() + data.len());
    out.extend_from_slice(&(im as u32).to_le_bytes());
    out.extend_from_slice(&(i_m as u32).to_le_bytes());
    out.extend_from_slice(&(table.len() as u32).to_le_bytes());
    out.extend_from_slice(&(n_bits as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&table);
    out.extend_from_slice(&data);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(values: &[u16]) {
        let payload = huf_compress(values).unwrap();
        let back = huf_decompress(&payload, values.len()).unwrap();
        assert_eq!(back, values, "round-trip mismatch");
    }

    #[test]
    fn single_value_stream() {
        roundtrip(&[7]);
        roundtrip(&[0]);
        roundtrip(&[65535]);
    }

    #[test]
    fn constant_run_uses_escape() {
        let vals = vec![42u16; 1000];
        let payload = huf_compress(&vals).unwrap();
        // A 1000-long constant run must compress far below one code per
        // value (runs of 256 via the 8-bit repeat field).
        assert!(
            payload.len() < 60,
            "run coding ineffective: {} bytes",
            payload.len()
        );
        let back = huf_decompress(&payload, vals.len()).unwrap();
        assert_eq!(back, vals);
    }

    #[test]
    fn mixed_values() {
        let vals: Vec<u16> = (0..5000u32)
            .map(|i| ((i * 7 + (i / 13) * 3) % 300) as u16)
            .collect();
        roundtrip(&vals);
    }

    #[test]
    fn sparse_extreme_alphabet() {
        // Values at both alphabet ends force a wide im..iM span with long
        // zero runs in the code-length table.
        let mut vals = vec![0u16; 100];
        vals.extend(std::iter::repeat(65535u16).take(100));
        vals.extend((0..50).map(|i| (i * 1000) as u16));
        roundtrip(&vals);
    }

    #[test]
    fn runs_at_saturation_boundary() {
        // Exactly 256 repeats (c saturates at 255) and 257.
        for n in [255usize, 256, 257, 512, 513] {
            let vals = vec![9u16; n];
            roundtrip(&vals);
        }
    }

    #[test]
    fn escape_value_collision() {
        // Highest value present is 65535, so the escape lands on index
        // 65536 (the u16-truncation-ambiguous slot).
        let mut vals: Vec<u16> = (65000..=65535).collect();
        vals.extend(std::iter::repeat(65535u16).take(300));
        roundtrip(&vals);
    }

    #[test]
    fn rejects_truncated_header() {
        assert!(huf_decompress(&[0u8; 10], 5).is_err());
    }

    #[test]
    fn rejects_bad_symbol_range() {
        let mut p = vec![0u8; 24];
        p[0..4].copy_from_slice(&70000u32.to_le_bytes());
        assert!(huf_decompress(&p, 5).is_err());
    }

    #[test]
    fn rejects_truncated_data() {
        let vals: Vec<u16> = (0..100).collect();
        let payload = huf_compress(&vals).unwrap();
        // Chop the entropy data in half.
        let cut = payload.len() - (payload.len() - 20) / 4;
        assert!(huf_decompress(&payload[..cut], vals.len()).is_err());
    }

    #[test]
    fn hostile_zero_run_past_im() {
        // Header claiming im=0, iM=1 but a long zero run of 261 symbols.
        let mut p = Vec::new();
        p.extend_from_slice(&0u32.to_le_bytes());
        p.extend_from_slice(&1u32.to_le_bytes());
        p.extend_from_slice(&2u32.to_le_bytes());
        p.extend_from_slice(&0u32.to_le_bytes());
        p.extend_from_slice(&0u32.to_le_bytes());
        p.push(0b1111_1111); // symbol 63 then high bits of the 8-bit run
        p.push(0b1111_1100);
        assert!(huf_decompress(&p, 5).is_err());
    }
}
