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
/// Widest decode table, used for large chunks with long codes.
const WIDE_FAST_BITS: usize = 16;
/// Values a chunk must hold before the wide table is worth filling.
const WIDE_TABLE_MIN_VALUES: usize = 1 << 16;

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
    /// A writer for about `values` coded 16-bit values (typical codes
    /// average under a byte; the buffer grows if they do not).
    fn with_capacity(values: usize) -> Self {
        BitWriter {
            out: Vec::with_capacity(values + 16),
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
    let mut w = BitWriter::with_capacity(0);
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
    huf_decompress_with(payload, expected, None)
}

/// [`huf_decompress`] with the decode-table width forced (tests: every
/// width must decode alike).
fn huf_decompress_with(
    payload: &[u8],
    expected: usize,
    table_bits: Option<usize>,
) -> Result<Vec<u16>> {
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

    // Coded symbols in increasing index order (a plain loop over the
    // declared range: this runs once per chunk over up to 65 537
    // entries, so it is kept allocation-tight).
    let mut syms: Vec<u32> = Vec::new();
    let mut n_per_len = [0u32; MAX_CODE_LEN + 1];
    for (s, &l) in lengths[im..=i_m].iter().enumerate() {
        if l != 0 {
            syms.push((im + s) as u32);
            n_per_len[l as usize] += 1;
        }
    }
    if syms.is_empty() {
        return Err(ExrError::invalid(
            "Huffman payload: no coded symbols".to_string(),
        ));
    }
    let (first, codes) = canonical_codes(&lengths, &syms);

    // Per-length symbol ranges for the slow (>14-bit) path: the coded
    // symbol with rank r among length-l symbols (index order) has code
    // first[l] + r. Stored flat — `per_len_syms[l]` is the slice
    // `flat[len_start[l] .. len_start[l + 1]]` — so the setup is one
    // allocation rather than one per length.
    let mut len_start = [0usize; MAX_CODE_LEN + 2];
    for l in 1..=MAX_CODE_LEN {
        len_start[l + 1] = len_start[l] + n_per_len[l] as usize;
    }
    let mut flat_syms = vec![0u32; syms.len()];
    let mut len_fill = len_start;
    for &s in &syms {
        let l = lengths[s as usize] as usize;
        flat_syms[len_fill[l]] = s;
        len_fill[l] += 1;
    }
    let per_len_sym = |l: usize, rank: usize| flat_syms[len_start[l] + rank];

    // Validate the code-length distribution before it indexes anything.
    // The lengths arrive over the wire and are only meaningful as a
    // canonical prefix code if they do not over-subscribe the code
    // space: for each length `l`, the highest code assigned is
    // `first[l] + n[l] - 1`, which must still fit in `l` bits. An
    // over-subscribed table (Kraft sum above one) otherwise produces a
    // code `>= 2^l`, which would run the fast-table fill past its end
    // and, more fundamentally, is not a decodable code at all.
    for l in 1..=MAX_CODE_LEN {
        let n = n_per_len[l] as u64;
        if n != 0 && first[l] + n > (1u64 << l) {
            return Err(ExrError::invalid(format!(
                "Huffman payload: code lengths over-subscribe the code space at length {l}"
            )));
        }
    }

    // Width of the direct-lookup table: 14 bits, or up to 16 when the
    // chunk is large enough to pay for the bigger table and its codes
    // are that long (FLOAT data often codes near 16 bits per word; a
    // code the table does not hold costs a search over lengths). The
    // width changes speed only: a code of at most `fast_bits` bits is
    // resolved here exactly as the slow path would resolve it.
    let longest = (1..=MAX_CODE_LEN)
        .rev()
        .find(|&l| n_per_len[l] != 0)
        .unwrap_or(1);
    let fast_bits: usize = match table_bits {
        Some(bits) => bits.clamp(FAST_BITS, WIDE_FAST_BITS),
        None if expected >= WIDE_TABLE_MIN_VALUES => longest.clamp(FAST_BITS, WIDE_FAST_BITS),
        None => FAST_BITS,
    };
    // Codes resolved per refill in the burst below (56 buffered bits).
    let burst = 56 / fast_bits;

    // Fast table: (symbol_index_u32 << 6) | length for codes <= fast_bits.
    // Escape is stored as the sentinel symbol value ALPHABET-1 marker via
    // a parallel "is escape" bit: encode entry as (sym << 7) | (is_esc << 6)?
    // Simpler: store u32 = (sym << 6) | len where sym is the true index
    // (fits in 26 bits).
    let mut fast = vec![0u32; 1 << fast_bits];
    let mut fill = |i: usize, s: u32, l: usize| {
        let lo = (codes[i] << (fast_bits - l)) as usize;
        let hi = lo + (1usize << (fast_bits - l));
        for slot in &mut fast[lo..hi] {
            *slot = ((s << 6) | l as u32) + 1; // +1 so 0 = invalid
        }
    };
    // Fill order = match priority, so a table of any width resolves a
    // (malformed) non-prefix-free code set the way the 14-bit table
    // plus the length search do: a code of at most 14 bits wins over
    // any longer one, and among longer codes the shortest wins.
    for wide in (FAST_BITS + 1..=fast_bits).rev() {
        for (i, &s) in syms.iter().enumerate() {
            if lengths[s as usize] as usize == wide {
                fill(i, s, wide);
            }
        }
    }
    for (i, &s) in syms.iter().enumerate() {
        let l = lengths[s as usize] as usize;
        if l <= FAST_BITS {
            fill(i, s, l);
        }
    }

    // `expected` is bounded by `n_bits` above, but `n_bits` can itself be
    // large; reserve modestly and let the buffer grow to what the stream
    // actually decodes rather than to the header's claim.
    const RESERVE_CAP: usize = 1 << 20;
    let mut out: Vec<u16> = Vec::with_capacity(expected.min(RESERVE_CAP));
    let escape = i_m as u32;

    // The bit reader state lives in plain locals for the decode loop so
    // the compiler keeps it in registers (the method-based reader spilled
    // every field to the stack once per symbol — the round-457 profile
    // put ~60% of PIZ decode time in those loads and stores). The
    // semantics are exactly those of `BitReader::peek` / `skip` / `get`:
    // MSB-first, `limit` bits available, bits past the limit peek as
    // zero and can never be consumed.
    let limit: u64 = n_bits.min(data.len() as u64 * 8);
    let mut acc: u64 = 0; // upcoming bits, left-aligned
    let mut nacc: u32 = 0; // valid bits in `acc`
    let mut bp: usize = 0; // next byte of `data` to load
    let mut pos: u64 = 0; // bits consumed
    let exhausted = || ExrError::invalid("Huffman payload: bit stream exhausted".to_string());

    macro_rules! refill {
        () => {
            if nacc <= 55 {
                if bp + 8 <= data.len() {
                    let w = u64::from_be_bytes(data[bp..bp + 8].try_into().unwrap());
                    let fill = (63 - nacc) >> 3;
                    acc |= (w & (!0u64 << (64 - fill * 8))) >> nacc;
                    bp += fill as usize;
                    nacc += fill * 8;
                } else {
                    while nacc <= 55 && bp < data.len() {
                        acc |= (data[bp] as u64) << (56 - nacc);
                        bp += 1;
                        nacc += 8;
                    }
                }
            }
        };
    }
    // Consume `n` (1..=56) bits, MSB-first; errors past the limit.
    macro_rules! get {
        ($n:expr_2021) => {{
            let n: u32 = $n;
            if pos + n as u64 > limit {
                return Err(exhausted());
            }
            refill!();
            let v = acc >> (64 - n);
            acc <<= n;
            nacc -= n;
            pos += n as u64;
            v
        }};
    }

    while out.len() < expected {
        refill!();
        // Burst: with 56 buffered bits inside the limit and at least
        // `burst` values still owed, `burst` fast-table codes (at most
        // `fast_bits` bits each) need no per-symbol bounds checks. A long code or
        // the run escape ends the burst and takes the general path
        // below, which does exactly what the burst does for a short
        // code — so the output and every error are unchanged.
        if nacc >= 56 && limit - pos >= 56 && expected - out.len() >= burst {
            let before = out.len();
            for _ in 0..burst {
                let entry = fast[(acc >> (64 - fast_bits)) as usize];
                if entry == 0 {
                    break;
                }
                let e = entry - 1;
                if e >> 6 == escape {
                    break;
                }
                let l = e & 63;
                acc <<= l;
                nacc -= l;
                pos += l as u64;
                out.push((e >> 6) as u16);
            }
            if out.len() != before {
                continue;
            }
        }
        // Peek fast_bits; bits past the declared limit read as zero.
        let raw = acc >> (64 - fast_bits);
        let remaining = limit - pos;
        let peeked = if remaining >= fast_bits as u64 {
            raw
        } else if remaining == 0 {
            0
        } else {
            let pad = fast_bits as u32 - remaining as u32;
            (raw >> pad) << pad
        };
        let entry = fast[peeked as usize];
        let sym: u32;
        if entry != 0 {
            let e = entry - 1;
            let l = e & 63;
            if pos + l as u64 > limit {
                return Err(exhausted());
            }
            acc <<= l;
            nacc -= l;
            pos += l as u64;
            sym = e >> 6;
        } else {
            // Slow path, first attempt: the accumulator already holds up
            // to 56 upcoming bits, so test every longer length against
            // its canonical range directly instead of growing the code
            // one bit at a time. This finds exactly the match the
            // incremental scan below would find first (lengths are
            // tried in increasing order) whenever the whole code is
            // legitimately available; otherwise fall through to the
            // incremental scan, which also produces the exact
            // exhaustion error.
            let avail = (nacc as u64).min(remaining).min(56) as usize;
            let mut found: Option<(u32, u32)> = None;
            for l in (fast_bits + 1)..=avail {
                let n = n_per_len[l] as u64;
                if n != 0 {
                    let code = acc >> (64 - l);
                    if code >= first[l] && code < first[l] + n {
                        let rank = (code - first[l]) as usize;
                        found = Some((per_len_sym(l, rank), l as u32));
                        break;
                    }
                }
            }
            if let Some((s, l)) = found {
                acc <<= l;
                nacc -= l;
                pos += l as u64;
                sym = s;
            } else {
                // Slow path: accumulate bits beyond fast_bits until a
                // per-length range matches.
                let mut acc_code = get!(fast_bits as u32);
                let mut l = fast_bits;
                loop {
                    if l >= MAX_CODE_LEN {
                        return Err(ExrError::invalid(
                            "Huffman payload: invalid code (no symbol within 58 bits)".to_string(),
                        ));
                    }
                    acc_code = (acc_code << 1) | get!(1);
                    l += 1;
                    let n = n_per_len[l] as u64;
                    if n != 0 && acc_code >= first[l] && acc_code < first[l] + n {
                        let rank = (acc_code - first[l]) as usize;
                        sym = per_len_sym(l, rank);
                        break;
                    }
                }
            }
        }
        if sym == escape {
            // Run escape: 8 raw bits = additional repeats of the previous
            // output value. Never legal as the first symbol.
            let cnt = get!(8) as usize;
            let prev = *out.last().ok_or_else(|| {
                ExrError::invalid("Huffman payload: run escape before any value".to_string())
            })?;
            if out.len() + cnt > expected {
                return Err(ExrError::invalid(
                    "Huffman payload: run overruns expected output".to_string(),
                ));
            }
            out.extend(std::iter::repeat_n(prev, cnt));
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
/// (index-sparse; zero frequency = uncoded), capped at 58 bits. If the
/// depth cap is ever exceeded the frequencies are halved (rounding up)
/// and the tree is rebuilt, which strictly flattens it.
///
/// Two-queue construction: the leaves sorted by (weight, symbol) form
/// one queue, the internal nodes — created in non-decreasing weight
/// order — the other; each step merges the two lightest heads, a leaf
/// before an internal node of equal weight. That is the order a
/// priority queue keyed on (weight, node index) pops them in, without
/// the queue.
fn build_code_lengths(freqs: &[u32], lengths: &mut [u8]) {
    // (weight, symbol), in symbol order.
    let mut leaves: Vec<(u64, u32)> = freqs
        .iter()
        .enumerate()
        .filter(|&(_, &f)| f != 0)
        .map(|(s, &f)| (f as u64, s as u32))
        .collect();
    let n = leaves.len();
    if n == 0 {
        return;
    }
    if n == 1 {
        lengths[leaves[0].1 as usize] = 1;
        return;
    }
    // Node ids: leaves 0..n in sorted order, internal nodes n..2n-1 in
    // creation order (the last one is the root).
    let mut parent = vec![0u32; 2 * n - 1];
    let mut internal = vec![0u64; n - 1];
    let mut depth = vec![0u32; 2 * n - 1];
    loop {
        // Stable: equal weights stay in symbol order.
        leaves.sort_by_key(|&(w, _)| w);
        let (mut leaf, mut node) = (0usize, 0usize);
        for k in 0..n - 1 {
            let take = |leaf: &mut usize, node: &mut usize| -> (u64, usize) {
                if *leaf < n && (*node >= k || leaves[*leaf].0 <= internal[*node]) {
                    *leaf += 1;
                    (leaves[*leaf - 1].0, *leaf - 1)
                } else {
                    *node += 1;
                    (internal[*node - 1], n + *node - 1)
                }
            };
            let (wa, a) = take(&mut leaf, &mut node);
            let (wb, b) = take(&mut leaf, &mut node);
            internal[k] = wa + wb;
            parent[a] = (n + k) as u32;
            parent[b] = (n + k) as u32;
        }
        // A parent is always created after its children: walk down
        // from the root.
        depth[2 * n - 2] = 0;
        for id in (0..2 * n - 2).rev() {
            depth[id] = depth[parent[id] as usize] + 1;
        }
        if depth[..n].iter().all(|&d| d as usize <= MAX_CODE_LEN) {
            for (i, &(_, sym)) in leaves.iter().enumerate() {
                lengths[sym as usize] = depth[i] as u8;
            }
            return;
        }
        // Flatten: halve the frequencies (round up) and rebuild, the
        // leaves back in symbol order first.
        leaves.sort_unstable_by_key(|&(_, sym)| sym);
        for leaf in leaves.iter_mut() {
            leaf.0 = (leaf.0 >> 1).max(1);
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
    // A chunk holds far fewer than 2^32 words; saturate regardless (a
    // saturated count only makes the code marginally less tight).
    let mut freqs = vec![0u32; ALPHABET];
    for &v in values {
        let f = &mut freqs[v as usize];
        *f = f.saturating_add(1);
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
    // Dense lookup: symbol -> code << 6 | length (codes are at most
    // 58 bits). Only the coded range is touched.
    let mut code_of = vec![0u64; i_m + 1];
    for (i, &s) in syms.iter().enumerate() {
        code_of[s as usize] = (codes[i] << 6) | lengths[s as usize] as u64;
    }

    let table = pack_code_lengths(&lengths, im, i_m);

    // Entropy-code the values with the run-length escape.
    let (esc_code, esc_len) = (code_of[i_m] >> 6, (code_of[i_m] & 63) as u32);
    let mut w = BitWriter::with_capacity(values.len());
    let mut i = 0usize;
    while i < values.len() {
        let v = values[i];
        let entry = code_of[v as usize];
        let (code, len) = (entry >> 6, (entry & 63) as u32);
        debug_assert!(len != 0);
        // Most values do not repeat: one compare, one code.
        if i + 1 == values.len() || values[i + 1] != v {
            w.put(code, len);
            i += 1;
            continue;
        }
        // Count additional repeats, saturating at 255.
        let window = &values[i + 1..(i + 256).min(values.len())];
        let c = window.iter().position(|&x| x != v).unwrap_or(window.len());
        if (len as usize) + (esc_len as usize) + 8 < (len as usize) * c {
            w.put(code, len);
            w.put(esc_code, esc_len);
            w.put(c as u64, 8);
        } else {
            for _ in 0..=c {
                w.put(code, len);
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

    /// The priority-queue construction `build_code_lengths` replaced:
    /// pops the two lightest nodes by (weight, node index). Kept as the
    /// reference the two-queue construction must agree with, length for
    /// length.
    fn reference_code_lengths(freqs: &[u64], lengths: &mut [u8]) {
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
            let root = heap.pop().unwrap().0.1;
            let mut stack = vec![(root, 0u32)];
            let mut max_depth = 0u32;
            let mut depths: Vec<(usize, u32)> = Vec::with_capacity(coded.len());
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
                    lengths[s] = d as u8;
                }
                return;
            }
            // Flatten: halve frequencies (round up) and rebuild.
            for f in fs.iter_mut() {
                *f = (*f >> 1).max(1);
            }
        }
    }

    /// Decode at every table width; outcomes (values or error text)
    /// must be the same.
    fn decode_all_widths(payload: &[u8], expected: usize) -> Result<Vec<u16>> {
        let narrow = huf_decompress_with(payload, expected, Some(FAST_BITS));
        for bits in [15, WIDE_FAST_BITS] {
            let wide = huf_decompress_with(payload, expected, Some(bits));
            match (&narrow, &wide) {
                (Ok(a), Ok(b)) => assert!(a == b, "values differ at {bits} bits"),
                (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string(), "{bits} bits"),
                _ => panic!("outcome differs at {bits} bits"),
            }
        }
        let auto = huf_decompress(payload, expected);
        assert_eq!(auto.is_ok(), narrow.is_ok());
        narrow
    }

    #[test]
    fn every_table_width_decodes_alike() {
        let mut state = 0x0F0F_1234_5678_9ABCu64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..40 {
            // Alphabets from a handful of symbols (short codes, long
            // runs) to the full 16-bit range (codes past 16 bits).
            let spread = [4u64, 300, 5000, 65536][round % 4];
            let n = 1 + (next() % 200_000) as usize;
            let values: Vec<u16> = (0..n)
                .map(|i| {
                    let v = next() % spread;
                    // Skew plus occasional runs.
                    if i % 97 < 5 {
                        7
                    } else {
                        (v * (next() % 3)) as u16
                    }
                })
                .collect();
            let payload = huf_compress(&values).unwrap();
            assert!(decode_all_widths(&payload, n).unwrap() == values);
            // Too many / too few values asked for, truncated data,
            // damaged bytes: same outcome at every width.
            let _ = decode_all_widths(&payload, n + 3);
            let _ = decode_all_widths(&payload, n.saturating_sub(2).max(1));
            for cut in [1usize, 2, 9, 40] {
                let _ = decode_all_widths(&payload[..payload.len().saturating_sub(cut).max(20)], n);
            }
            let mut bad = payload.clone();
            for _ in 0..12 {
                let at = (next() % bad.len() as u64) as usize;
                bad[at] ^= 1 << (next() % 8);
                let _ = decode_all_widths(&bad, n);
            }
        }
    }

    fn agree(freqs: &[u32]) {
        let mut fast = vec![0u8; freqs.len()];
        build_code_lengths(freqs, &mut fast);
        let wide: Vec<u64> = freqs.iter().map(|&f| f as u64).collect();
        let mut reference = vec![0u8; freqs.len()];
        reference_code_lengths(&wide, &mut reference);
        assert!(
            fast == reference,
            "code lengths differ for {} symbols",
            freqs.len()
        );
    }

    #[test]
    fn two_queue_lengths_match_the_priority_queue_construction() {
        let mut state = 0x1234_5678_9ABC_DEF1u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // Random shapes: dense and sparse alphabets, flat and skewed
        // weights, many ties.
        for round in 0..300 {
            let size = 1 + (next() % 3000) as usize;
            let skew = (next() % 20) as u32;
            let freqs: Vec<u32> = (0..size)
                .map(|_| match round % 4 {
                    0 => (next() % 4) as u32,
                    1 => 1,
                    2 => ((next() % 1000) as u32) >> (next() % (skew + 1) as u64).min(9),
                    _ => (1u32 << (next() % 24)) * (next() % 2) as u32,
                })
                .collect();
            agree(&freqs);
        }
        // Full alphabet, as a FLOAT chunk of noise produces.
        let full: Vec<u32> = (0..ALPHABET).map(|_| 1 + (next() % 9) as u32).collect();
        agree(&full);
        // Degenerate cases.
        agree(&[0, 0, 0]);
        agree(&[0, 7, 0]);
        agree(&[5, 5]);
        // Fibonacci weights: the deepest tree a total allows — past the
        // 58-bit cap, so the halving path runs.
        let mut fib = vec![1u32, 1];
        while fib.len() < 46 {
            let n = fib.len();
            fib.push(fib[n - 1] + fib[n - 2]);
        }
        agree(&fib);
        let mut deep = vec![0u8; 80];
        let mut weights = vec![1u32; 80];
        for i in 2..80 {
            weights[i] = weights[i - 1]
                .saturating_add(weights[i - 2])
                .min(u32::MAX / 2);
        }
        build_code_lengths(&weights, &mut deep);
        assert!(deep.iter().all(|&l| (1..=MAX_CODE_LEN as u8).contains(&l)));
        agree(&weights);
    }

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
        vals.extend(std::iter::repeat_n(65535u16, 100));
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
        vals.extend(std::iter::repeat_n(65535u16, 300));
        roundtrip(&vals);
    }

    #[test]
    fn rejects_truncated_header() {
        assert!(huf_decompress(&[0u8; 10], 5).is_err());
    }

    #[test]
    fn rejects_over_subscribed_code_lengths() {
        // Hand-build a payload whose code-length table assigns three
        // symbols a length of 1 bit. Only two distinct 1-bit codes
        // exist, so the table over-subscribes the code space; the
        // canonical assignment would need a code of 2 in one bit, which
        // must be rejected rather than run the fast table past its end.
        let im = 0usize;
        let i_m = 2usize;
        let mut body = BitWriter::with_capacity(0);
        // Three explicit 6-bit lengths, all 1.
        for _ in 0..3 {
            body.put(1, 6);
        }
        let (table, _) = body.finish();
        // Header: im, iM, table byte length, a nonzero bit count, zero.
        let mut payload = Vec::new();
        payload.extend_from_slice(&(im as u32).to_le_bytes());
        payload.extend_from_slice(&(i_m as u32).to_le_bytes());
        payload.extend_from_slice(&(table.len() as u32).to_le_bytes());
        payload.extend_from_slice(&8u32.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&table);
        payload.push(0); // one byte of entropy data so the bit check passes
        let err = huf_decompress(&payload, 4).unwrap_err();
        assert!(format!("{err}").contains("over-subscribe"));
    }

    #[test]
    fn huge_symbol_count_does_not_over_reserve() {
        // A well-formed payload for 100 symbols, but the caller declares
        // billions (as a hostile DWA AC count can). The reservation must
        // stay bounded and the decode must terminate with an ordinary
        // error when the short bit stream runs out — never OOM on the
        // up-front allocation.
        let vals: Vec<u16> = (0..100).collect();
        let payload = huf_compress(&vals).unwrap();
        assert!(huf_decompress(&payload, 4 << 30).is_err());
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
