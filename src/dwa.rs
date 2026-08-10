//! DWAA / DWAB compression (compression codes 8 and 9) —
//! observer-spec `openexr-piz-dwa-observer-spec.md` §3 plus the staged
//! tables under `docs/image/openexr/tables/` (perceptual LUT closed
//! forms and their binary32 patch indices, channel-rule sets, zig-zag,
//! IDCT butterfly constants, quantisation matrices).
//!
//! DWAA and DWAB are the same compressor; the only wire difference is
//! the chunk height (32 vs 256 scanlines — handled by
//! `Compression::scanlines_per_block`). A chunk carries an 88-byte
//! header of eleven little-endian `u64`, an optional channel-rule block
//! (version 2), then four concatenated sub-streams: verbatim
//! (deflated), AC (static-Huffman payload or deflate), DC (deflated
//! after the ZIP byte preconditioning), RLE (deflated after byte-plane
//! split + byte RLE).
//!
//! Channels are classified by (name-suffix, pixel type) rules into
//! VERBATIM / LOSSY_DCT / RLE schemes; an R/G/B triple sharing a name
//! prefix is colour-transformed together (BT.709). LOSSY_DCT data is
//! coded in 8×8 half-precision DCT blocks with a perceptual half → half
//! LUT for channels not flagged perceptually linear.
//!
//! Stream orderings not printed in the observer-spec were derived by
//! black-box observation of reference-produced chunks (wire bytes
//! only): AC and DC walk channel-sets with every CSC **triple first**
//! (in first-member order), then the remaining lossy channels in
//! sorted-channel order; within a triple the AC stream interleaves per
//! block (Y, C1, C2 for each block in row-major order) while the DC
//! stream stays plane-major; the RLE byte-plane split runs over the
//! channel's whole chunk region (not per row); verbatim regions are
//! channel-major. See `tests/dwa_observer_notes.md`.

use crate::error::{ExrError, Result};
use crate::piz::ChunkShape;
use crate::types::{Channel, PixelType};

// ---------------------------------------------------------------------------
// Constants from the staged tables
// ---------------------------------------------------------------------------

/// Zig-zag scan: scan position -> raster index
/// (`tables/dct-zigzag.csv`, ITU-T T.81 order).
const ZIGZAG_RASTER: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// The seven IDCT butterfly multipliers `0.5 · cos(k·π/16)` for
/// k = 4, 1, 2, 3, 5, 6, 7, in **binary32 evaluated with the truncated
/// literal π = 3.14159** — the staged bit patterns from
/// `tables/dwa-idct-butterfly-constants.csv`, which is what the wire
/// format's reference decoder computes with.
// Decimal spellings round to exactly the staged bit patterns
// (0x3eb504fb, 0x3efb14bf, 0x3eec8360, 0x3ed4db36, 0x3e8e39e5,
// 0x3e43ef33, 0x3dc7c60b — asserted by `dct_constants_match_staged_bits`).
const DCT_A: f32 = 0.353_553_62; // k = 4
const DCT_B: f32 = 0.490_392_65; // k = 1
const DCT_C: f32 = 0.461_939_8; // k = 2
const DCT_D: f32 = 0.415_734_95; // k = 3
const DCT_E: f32 = 0.277_785_45; // k = 5
const DCT_F: f32 = 0.191_342_16; // k = 6
const DCT_G: f32 = 0.097_545_706; // k = 7

/// Encoder-side luma quantisation-sensitivity matrix
/// (`tables/dwa-jpeg-quant-y.csv`, raster order; ITU-T T.81 Annex K
/// sample table; never transmitted). Normalisation divisor = 10.
const QUANT_Y: [u16; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];
const QUANT_Y_MIN: f32 = 10.0;

/// Encoder-side chroma matrix (`tables/dwa-jpeg-quant-cbcr.csv`).
/// Normalisation divisor = 17.
const QUANT_CBCR: [u16; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];
const QUANT_CBCR_MIN: f32 = 17.0;

/// Default DWA compression level; `errorTarget = level / 100000`
/// (observer-spec §3.5). The `dwaCompressionLevel` header attribute can
/// override it at encode time.
pub(crate) const DEFAULT_DWA_LEVEL: f32 = 45.0;

// ---------------------------------------------------------------------------
// Perceptual LUTs (observer-spec §3.5 + tables/dwa-to{non,}linear-table)
// ---------------------------------------------------------------------------

/// Compute the two 65 536-entry perceptual LUTs. Closed form per the
/// staged tables: gamma 1/2.2 below unity blended into a logarithm
/// above it (base literal 9.02501329156, NOT exp(2.2)), sign preserved;
/// code word 0 maps to itself and any all-ones-exponent code word
/// (inf / NaN) maps to 0. The reference evaluates in binary32; we
/// evaluate the closed form in binary64 (well-defined to well below the
/// half rounding step) and patch the six entries the `.meta` sidecars
/// document as binary32-vs-binary64 sensitive, giving the exact staged
/// binary32 tables (asserted against the CSVs by the fixture-gated
/// tests).
fn compute_luts() -> (Vec<u16>, Vec<u16>) {
    let mut to_nonlinear = vec![0u16; 65536];
    let mut to_linear = vec![0u16; 65536];
    for code in 0u32..65536 {
        let h = code as u16;
        if h == 0 {
            continue;
        }
        if (h & 0x7C00) == 0x7C00 {
            // inf / NaN guard: maps to 0 in both directions.
            continue;
        }
        let v = crate::half::half_to_f32(h) as f64;
        let (sign, mag) = (v.signum(), v.abs());

        let nl = if mag <= 1.0 {
            mag.powf(1.0 / 2.2)
        } else {
            mag.ln() / 2.2 + 1.0
        };
        to_nonlinear[code as usize] = crate::half::f32_to_half((sign * nl) as f32);

        let lin = if mag <= 1.0 {
            mag.powf(2.2)
        } else {
            9.025_013_291_56_f64.powf(mag - 1.0)
        };
        to_linear[code as usize] = crate::half::f32_to_half((sign * lin) as f32);
    }
    // binary32 patch entries (values from the staged CSVs; indices from
    // the .meta sidecars).
    to_linear[0x24f8] = 0x099c;
    to_linear[0xa4f8] = 0x899c;
    to_nonlinear[0x1919] = 0x2c32;
    to_nonlinear[0x422c] = 0x3e0c;
    to_nonlinear[0x9919] = 0xac32;
    to_nonlinear[0xc22c] = 0xbe0c;
    (to_nonlinear, to_linear)
}

fn luts() -> &'static (Vec<u16>, Vec<u16>) {
    static LUTS: std::sync::OnceLock<(Vec<u16>, Vec<u16>)> = std::sync::OnceLock::new();
    LUTS.get_or_init(compute_luts)
}

/// Forward perceptual mapping (encoder side).
pub(crate) fn to_nonlinear(code: u16) -> u16 {
    luts().0[code as usize]
}

/// Inverse perceptual mapping (decoder side).
pub(crate) fn to_linear(code: u16) -> u16 {
    luts().1[code as usize]
}

// ---------------------------------------------------------------------------
// Channel classification (observer-spec §3.2 +
// tables/dwa-channel-rules-{default,legacy}.csv)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DwaScheme {
    Verbatim,
    LossyDct,
    Rle,
}

#[derive(Debug, Clone)]
struct DwaRule {
    suffix: &'static str,
    pixel_type: PixelType,
    scheme: DwaScheme,
    /// -1 = not part of a colour-transform triple; 0/1/2 = R/G/B slot.
    csc_slot: i8,
    case_insensitive: bool,
}

/// A rule parsed from a version-2 chunk (owned suffix).
#[derive(Debug, Clone)]
struct WireRule {
    suffix: String,
    pixel_type: PixelType,
    scheme: DwaScheme,
    csc_slot: i8,
    case_insensitive: bool,
}

const fn rule(
    suffix: &'static str,
    pixel_type: PixelType,
    scheme: DwaScheme,
    csc_slot: i8,
    case_insensitive: bool,
) -> DwaRule {
    DwaRule {
        suffix,
        pixel_type,
        scheme,
        csc_slot,
        case_insensitive,
    }
}

/// The rule set a version-2 encoder uses and writes into the chunk
/// (`tables/dwa-channel-rules-default.csv`; case-sensitive).
const DEFAULT_RULES: [DwaRule; 15] = [
    rule("R", PixelType::Half, DwaScheme::LossyDct, 0, false),
    rule("R", PixelType::Float, DwaScheme::LossyDct, 0, false),
    rule("G", PixelType::Half, DwaScheme::LossyDct, 1, false),
    rule("G", PixelType::Float, DwaScheme::LossyDct, 1, false),
    rule("B", PixelType::Half, DwaScheme::LossyDct, 2, false),
    rule("B", PixelType::Float, DwaScheme::LossyDct, 2, false),
    rule("Y", PixelType::Half, DwaScheme::LossyDct, -1, false),
    rule("Y", PixelType::Float, DwaScheme::LossyDct, -1, false),
    rule("BY", PixelType::Half, DwaScheme::LossyDct, -1, false),
    rule("BY", PixelType::Float, DwaScheme::LossyDct, -1, false),
    rule("RY", PixelType::Half, DwaScheme::LossyDct, -1, false),
    rule("RY", PixelType::Float, DwaScheme::LossyDct, -1, false),
    rule("A", PixelType::Uint, DwaScheme::Rle, -1, false),
    rule("A", PixelType::Half, DwaScheme::Rle, -1, false),
    rule("A", PixelType::Float, DwaScheme::Rle, -1, false),
];

/// The rule set a decoder MUST assume for version-0/1 chunks, which
/// transmit no rules (`tables/dwa-channel-rules-legacy.csv`;
/// case-insensitive, lower-case spellings incl. long colour names).
const LEGACY_RULES: [DwaRule; 25] = [
    rule("r", PixelType::Half, DwaScheme::LossyDct, 0, true),
    rule("r", PixelType::Float, DwaScheme::LossyDct, 0, true),
    rule("red", PixelType::Half, DwaScheme::LossyDct, 0, true),
    rule("red", PixelType::Float, DwaScheme::LossyDct, 0, true),
    rule("g", PixelType::Half, DwaScheme::LossyDct, 1, true),
    rule("g", PixelType::Float, DwaScheme::LossyDct, 1, true),
    rule("grn", PixelType::Half, DwaScheme::LossyDct, 1, true),
    rule("grn", PixelType::Float, DwaScheme::LossyDct, 1, true),
    rule("green", PixelType::Half, DwaScheme::LossyDct, 1, true),
    rule("green", PixelType::Float, DwaScheme::LossyDct, 1, true),
    rule("b", PixelType::Half, DwaScheme::LossyDct, 2, true),
    rule("b", PixelType::Float, DwaScheme::LossyDct, 2, true),
    rule("blu", PixelType::Half, DwaScheme::LossyDct, 2, true),
    rule("blu", PixelType::Float, DwaScheme::LossyDct, 2, true),
    rule("blue", PixelType::Half, DwaScheme::LossyDct, 2, true),
    rule("blue", PixelType::Float, DwaScheme::LossyDct, 2, true),
    rule("y", PixelType::Half, DwaScheme::LossyDct, -1, true),
    rule("y", PixelType::Float, DwaScheme::LossyDct, -1, true),
    rule("by", PixelType::Half, DwaScheme::LossyDct, -1, true),
    rule("by", PixelType::Float, DwaScheme::LossyDct, -1, true),
    rule("ry", PixelType::Half, DwaScheme::LossyDct, -1, true),
    rule("ry", PixelType::Float, DwaScheme::LossyDct, -1, true),
    rule("a", PixelType::Uint, DwaScheme::Rle, -1, true),
    rule("a", PixelType::Half, DwaScheme::Rle, -1, true),
    rule("a", PixelType::Float, DwaScheme::Rle, -1, true),
];

/// A channel's suffix: the text after the last `.`, or the whole name.
fn channel_suffix(name: &str) -> &str {
    match name.rfind('.') {
        Some(i) => &name[i + 1..],
        None => name,
    }
}

/// A channel's prefix: everything up to and including the last `.`, or
/// empty. Used only for CSC triple grouping.
fn channel_prefix(name: &str) -> &str {
    match name.rfind('.') {
        Some(i) => &name[..=i],
        None => "",
    }
}

fn suffix_matches(rule_suffix: &str, suffix: &str, case_insensitive: bool) -> bool {
    if case_insensitive {
        rule_suffix.eq_ignore_ascii_case(suffix)
    } else {
        rule_suffix == suffix
    }
}

/// Classify one channel against a rule list: first (suffix, pixel type)
/// match wins; unmatched channels are verbatim (observer-spec §3.2).
fn classify_wire(ch: &Channel, rules: &[WireRule]) -> (DwaScheme, i8) {
    let suffix = channel_suffix(&ch.name);
    for r in rules {
        if r.pixel_type == ch.pixel_type && suffix_matches(&r.suffix, suffix, r.case_insensitive) {
            return (r.scheme, r.csc_slot);
        }
    }
    (DwaScheme::Verbatim, -1)
}

fn static_rules_to_wire(rules: &[DwaRule]) -> Vec<WireRule> {
    rules
        .iter()
        .map(|r| WireRule {
            suffix: r.suffix.to_string(),
            pixel_type: r.pixel_type,
            scheme: r.scheme,
            csc_slot: r.csc_slot,
            case_insensitive: r.case_insensitive,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Channel-set construction
// ---------------------------------------------------------------------------

/// One lossy channel-set: either a CSC triple (channel indices in
/// R, G, B slot order) or a single channel.
enum LossySet {
    Triple([usize; 3]),
    Single(usize),
}

impl LossySet {
    fn component_count(&self) -> usize {
        match self {
            LossySet::Triple(_) => 3,
            LossySet::Single(_) => 1,
        }
    }
}

/// Build the lossy channel-set processing order (derivation record in
/// `tests/dwa_observer_notes.md`): every complete CSC triple first,
/// ordered by the sorted-channel index of its first member, then the
/// remaining LOSSY_DCT channels in sorted-channel order.
fn build_lossy_sets(channels: &[Channel], classes: &[(DwaScheme, i8)]) -> Vec<LossySet> {
    let mut in_triple = vec![false; channels.len()];
    let mut triples: Vec<(usize, [usize; 3])> = Vec::new();

    // Group by prefix, look for complete R/G/B slot triples.
    let mut prefixes_seen: Vec<&str> = Vec::new();
    for (i, ch) in channels.iter().enumerate() {
        if classes[i].0 != DwaScheme::LossyDct || classes[i].1 < 0 {
            continue;
        }
        let prefix = channel_prefix(&ch.name);
        if prefixes_seen.contains(&prefix) {
            continue;
        }
        prefixes_seen.push(prefix);
        let mut slots: [Option<usize>; 3] = [None, None, None];
        for (j, cj) in channels.iter().enumerate() {
            if classes[j].0 == DwaScheme::LossyDct
                && classes[j].1 >= 0
                && channel_prefix(&cj.name) == prefix
            {
                let s = classes[j].1 as usize;
                if s < 3 && slots[s].is_none() {
                    slots[s] = Some(j);
                }
            }
        }
        if let (Some(r), Some(g), Some(b)) = (slots[0], slots[1], slots[2]) {
            // Same pixel type and sampling required for a joint transform.
            let (cr, cg, cb) = (&channels[r], &channels[g], &channels[b]);
            if cr.pixel_type == cg.pixel_type
                && cg.pixel_type == cb.pixel_type
                && cr.x_sampling == cg.x_sampling
                && cg.x_sampling == cb.x_sampling
                && cr.y_sampling == cg.y_sampling
                && cg.y_sampling == cb.y_sampling
            {
                let first = r.min(g).min(b);
                triples.push((first, [r, g, b]));
                in_triple[r] = true;
                in_triple[g] = true;
                in_triple[b] = true;
            }
        }
    }
    triples.sort_by_key(|(first, _)| *first);

    let mut sets: Vec<LossySet> = triples
        .into_iter()
        .map(|(_, t)| LossySet::Triple(t))
        .collect();
    for (i, _) in channels.iter().enumerate() {
        if classes[i].0 == DwaScheme::LossyDct && !in_triple[i] {
            sets.push(LossySet::Single(i));
        }
    }
    sets
}

// ---------------------------------------------------------------------------
// 8-point DCT butterflies (binary32, staged constants)
// ---------------------------------------------------------------------------

/// Inverse 8-point transform over a stride-`s` slice view.
#[inline]
fn idct8(v: &mut [f32; 64], base: usize, s: usize) {
    let c0 = v[base];
    let c1 = v[base + s];
    let c2 = v[base + 2 * s];
    let c3 = v[base + 3 * s];
    let c4 = v[base + 4 * s];
    let c5 = v[base + 5 * s];
    let c6 = v[base + 6 * s];
    let c7 = v[base + 7 * s];

    // Wire-exactness note: the DC pair folds its sum/difference BEFORE
    // the multiplier (s = a·(c0 ± c4)); multiplying each term first
    // rounds differently and diverges from reference-decoded pixels by
    // one half-ULP on boundary values (empirically pinned in
    // tests/dwa_observer_notes.md).
    let s0 = DCT_A * (c0 + c4);
    let s1 = DCT_A * (c0 - c4);
    let r0 = DCT_C * c2 + DCT_F * c6;
    let r1 = DCT_F * c2 - DCT_C * c6;
    let g0 = s0 + r0;
    let g3 = s0 - r0;
    let g1 = s1 + r1;
    let g2 = s1 - r1;

    let o0 = DCT_B * c1 + DCT_D * c3 + DCT_E * c5 + DCT_G * c7;
    let o1 = DCT_D * c1 - DCT_G * c3 - DCT_B * c5 - DCT_E * c7;
    let o2 = DCT_E * c1 - DCT_B * c3 + DCT_G * c5 + DCT_D * c7;
    let o3 = DCT_G * c1 - DCT_E * c3 + DCT_D * c5 - DCT_B * c7;

    v[base] = g0 + o0;
    v[base + 7 * s] = g0 - o0;
    v[base + s] = g1 + o1;
    v[base + 6 * s] = g1 - o1;
    v[base + 2 * s] = g2 + o2;
    v[base + 5 * s] = g2 - o2;
    v[base + 3 * s] = g3 + o3;
    v[base + 4 * s] = g3 - o3;
}

/// Forward 8-point transform (encoder side; the mathematical transpose
/// of [`idct8`], so the pair round-trips up to float rounding).
#[inline]
fn fdct8(v: &mut [f32; 64], base: usize, s: usize) {
    let x0 = v[base];
    let x1 = v[base + s];
    let x2 = v[base + 2 * s];
    let x3 = v[base + 3 * s];
    let x4 = v[base + 4 * s];
    let x5 = v[base + 5 * s];
    let x6 = v[base + 6 * s];
    let x7 = v[base + 7 * s];

    let t0 = x0 + x7;
    let t1 = x1 + x6;
    let t2 = x2 + x5;
    let t3 = x3 + x4;
    let u0 = x0 - x7;
    let u1 = x1 - x6;
    let u2 = x2 - x5;
    let u3 = x3 - x4;

    let s0 = t0 + t3;
    let s1 = t1 + t2;
    let r0 = t0 - t3;
    let r1 = t1 - t2;

    v[base] = DCT_A * (s0 + s1);
    v[base + 4 * s] = DCT_A * (s0 - s1);
    v[base + 2 * s] = DCT_C * r0 + DCT_F * r1;
    v[base + 6 * s] = DCT_F * r0 - DCT_C * r1;
    v[base + s] = DCT_B * u0 + DCT_D * u1 + DCT_E * u2 + DCT_G * u3;
    v[base + 3 * s] = DCT_D * u0 - DCT_G * u1 - DCT_B * u2 - DCT_E * u3;
    v[base + 5 * s] = DCT_E * u0 - DCT_B * u1 + DCT_G * u2 + DCT_D * u3;
    v[base + 7 * s] = DCT_G * u0 - DCT_E * u1 + DCT_D * u2 - DCT_B * u3;
}

/// Separable inverse 8×8 DCT: rows then columns (observer-spec §3.5).
fn idct8x8(block: &mut [f32; 64]) {
    for row in 0..8 {
        idct8(block, row * 8, 1);
    }
    for col in 0..8 {
        idct8(block, col, 8);
    }
}

/// Separable forward 8×8 DCT (encoder side): rows then columns.
fn fdct8x8(block: &mut [f32; 64]) {
    for row in 0..8 {
        fdct8(block, row * 8, 1);
    }
    for col in 0..8 {
        fdct8(block, col, 8);
    }
}

// ---------------------------------------------------------------------------
// Colour transform (BT.709; observer-spec §3.5)
// ---------------------------------------------------------------------------

#[inline]
fn csc_forward(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let c1 = -0.1146 * r - 0.3854 * g + 0.5 * b;
    let c2 = 0.5 * r - 0.4542 * g - 0.0458 * b;
    (y, c1, c2)
}

/// Inverse colour transform with the staged binary32 BT.709 constants
/// (1.5747 / 1.8556 / −0.1873 / −0.4682 — see the GAP-TRACKER's Round-A
/// inventory of `internal_dwa_simd.h`).
#[inline]
fn csc_inverse(y: f32, c1: f32, c2: f32) -> (f32, f32, f32) {
    let r = y + 1.5747 * c2;
    let g = y - 0.1873 * c1 - 0.4682 * c2;
    let b = y + 1.8556 * c1;
    (r, g, b)
}

// ---------------------------------------------------------------------------
// Chunk header
// ---------------------------------------------------------------------------

const HEADER_BYTES: usize = 88;

struct DwaHeader {
    version: u64,
    unknown_uncompressed_size: usize,
    unknown_compressed_size: usize,
    ac_compressed_size: usize,
    dc_compressed_size: usize,
    rle_compressed_size: usize,
    rle_uncompressed_size: usize,
    rle_raw_size: usize,
    ac_count: usize,
    dc_count: usize,
    ac_compression: u64,
}

fn read_u64_le(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap())
}

fn u64_to_usize(v: u64, what: &str) -> Result<usize> {
    usize::try_from(v)
        .ok()
        .filter(|&s| s <= u32::MAX as usize * 16)
        .ok_or_else(|| ExrError::invalid(format!("DWA: {what} {v} out of range")))
}

impl DwaHeader {
    fn parse(payload: &[u8]) -> Result<DwaHeader> {
        if payload.len() < HEADER_BYTES {
            return Err(ExrError::invalid(format!(
                "DWA: chunk of {} bytes cannot hold the 88-byte header",
                payload.len()
            )));
        }
        let h = DwaHeader {
            version: read_u64_le(payload, 0),
            unknown_uncompressed_size: u64_to_usize(read_u64_le(payload, 1), "verbatim size")?,
            unknown_compressed_size: u64_to_usize(read_u64_le(payload, 2), "verbatim stream")?,
            ac_compressed_size: u64_to_usize(read_u64_le(payload, 3), "AC stream")?,
            dc_compressed_size: u64_to_usize(read_u64_le(payload, 4), "DC stream")?,
            rle_compressed_size: u64_to_usize(read_u64_le(payload, 5), "RLE stream")?,
            rle_uncompressed_size: u64_to_usize(read_u64_le(payload, 6), "RLE size")?,
            rle_raw_size: u64_to_usize(read_u64_le(payload, 7), "RLE raw size")?,
            ac_count: u64_to_usize(read_u64_le(payload, 8), "AC count")?,
            dc_count: u64_to_usize(read_u64_le(payload, 9), "DC count")?,
            ac_compression: read_u64_le(payload, 10),
        };
        if h.version > 2 {
            return Err(ExrError::invalid(format!(
                "DWA: chunk version {} > 2 must be rejected",
                h.version
            )));
        }
        if h.ac_compression > 1 {
            return Err(ExrError::invalid(format!(
                "DWA: unknown AC compression {}",
                h.ac_compression
            )));
        }
        if h.dc_count > 0 && h.dc_compressed_size == 0 {
            return Err(ExrError::invalid(
                "DWA: dcCount > 0 with an empty DC stream".to_string(),
            ));
        }
        Ok(h)
    }
}

/// Parse the version-2 channel-rule block. Returns the rules plus the
/// block's total byte length.
fn parse_rule_block(data: &[u8]) -> Result<(Vec<WireRule>, usize)> {
    if data.len() < 2 {
        return Err(ExrError::invalid(
            "DWA: rule block size field truncated".to_string(),
        ));
    }
    let block_size = u16::from_le_bytes([data[0], data[1]]) as usize;
    if block_size < 2 || block_size > data.len() {
        return Err(ExrError::invalid(format!(
            "DWA: rule block size {block_size} out of range"
        )));
    }
    let mut rules = Vec::new();
    let mut p = 2usize;
    while p < block_size {
        let rest = &data[p..block_size];
        let nul = rest
            .iter()
            .take(129)
            .position(|&b| b == 0)
            .ok_or_else(|| ExrError::invalid("DWA: rule suffix unterminated".to_string()))?;
        let suffix = String::from_utf8_lossy(&rest[..nul]).into_owned();
        p += nul + 1;
        if p + 2 > block_size {
            return Err(ExrError::invalid(
                "DWA: rule flag/type bytes truncated".to_string(),
            ));
        }
        let flag = data[p];
        let ptype = data[p + 1];
        p += 2;
        let pixel_type = PixelType::from_int(ptype as i32)
            .ok_or_else(|| ExrError::invalid(format!("DWA: rule pixel type {ptype} invalid")))?;
        let scheme = match (flag >> 2) & 0x3 {
            0 => DwaScheme::Verbatim,
            1 => DwaScheme::LossyDct,
            2 => DwaScheme::Rle,
            other => {
                return Err(ExrError::invalid(format!(
                    "DWA: rule scheme {other} invalid"
                )))
            }
        };
        let csc_slot = ((flag >> 4) & 0xF) as i8 - 1;
        if csc_slot > 2 {
            return Err(ExrError::invalid(format!(
                "DWA: rule CSC slot {csc_slot} invalid"
            )));
        }
        rules.push(WireRule {
            suffix,
            pixel_type,
            scheme,
            csc_slot,
            case_insensitive: flag & 1 != 0,
        });
    }
    Ok((rules, block_size))
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Decode one DWAA / DWAB chunk payload into the native interleaved
/// byte stream. The shared raw fallback
/// (`payload.len() == uncompressed_size`) is handled here.
pub(crate) fn decode_dwa_payload(
    payload: &[u8],
    shape: &ChunkShape,
    uncompressed_size: usize,
) -> Result<Vec<u8>> {
    if payload.len() == uncompressed_size {
        return Ok(payload.to_vec());
    }
    let header = DwaHeader::parse(payload)?;
    let mut p = HEADER_BYTES;

    let rules: Vec<WireRule> = if header.version >= 2 {
        let (rules, used) = parse_rule_block(&payload[p..])?;
        p += used;
        rules
    } else {
        static_rules_to_wire(&LEGACY_RULES)
    };

    // Slice the four sub-streams with running-sum bounds checks.
    let take = |p: &mut usize, len: usize, what: &str| -> Result<std::ops::Range<usize>> {
        let start = *p;
        let end = start
            .checked_add(len)
            .ok_or_else(|| ExrError::invalid(format!("DWA: {what} stream length overflows")))?;
        if end > payload.len() {
            return Err(ExrError::invalid(format!(
                "DWA: {what} stream ({len} bytes at {start}) runs past chunk end"
            )));
        }
        *p = end;
        Ok(start..end)
    };
    let verbatim_r = take(&mut p, header.unknown_compressed_size, "verbatim")?;
    let ac_r = take(&mut p, header.ac_compressed_size, "AC")?;
    let dc_r = take(&mut p, header.dc_compressed_size, "DC")?;
    let rle_r = take(&mut p, header.rle_compressed_size, "RLE")?;

    // Classify channels and size each scheme's expectations.
    let extents = shape.extents();
    let classes: Vec<(DwaScheme, i8)> = shape
        .sorted_channels
        .iter()
        .map(|ch| classify_wire(ch, &rules))
        .collect();
    // UINT channels cannot take the half-based lossy path.
    for (ch, cl) in shape.sorted_channels.iter().zip(classes.iter()) {
        if cl.0 == DwaScheme::LossyDct && ch.pixel_type == PixelType::Uint {
            return Err(ExrError::invalid(format!(
                "DWA: UINT channel '{}' classified LOSSY_DCT",
                ch.name
            )));
        }
    }

    // Inflate the verbatim stream.
    let verbatim_expected: usize = shape
        .sorted_channels
        .iter()
        .zip(extents.iter())
        .zip(classes.iter())
        .filter(|(_, cl)| cl.0 == DwaScheme::Verbatim)
        .map(|((ch, e), _)| e.nx * e.ny * ch.pixel_type.bytes_per_sample())
        .sum();
    if verbatim_expected != header.unknown_uncompressed_size {
        return Err(ExrError::invalid(format!(
            "DWA: verbatim channels need {verbatim_expected} bytes but header declares {}",
            header.unknown_uncompressed_size
        )));
    }
    let verbatim = if verbatim_expected > 0 {
        crate::decoder::zlib_inflate_pub(&payload[verbatim_r], verbatim_expected)?
    } else {
        Vec::new()
    };
    if verbatim.len() != verbatim_expected {
        return Err(ExrError::invalid(format!(
            "DWA: verbatim stream inflated to {} bytes, expected {verbatim_expected}",
            verbatim.len()
        )));
    }

    // AC elements.
    let ac: Vec<u16> = if header.ac_count > 0 {
        match header.ac_compression {
            0 => crate::huf::huf_decompress(&payload[ac_r], header.ac_count)?,
            _ => {
                let bytes = crate::decoder::zlib_inflate_pub(&payload[ac_r], header.ac_count * 2)?;
                if bytes.len() != header.ac_count * 2 {
                    return Err(ExrError::invalid(format!(
                        "DWA: AC stream inflated to {} bytes, expected {}",
                        bytes.len(),
                        header.ac_count * 2
                    )));
                }
                bytes
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect()
            }
        }
    } else {
        Vec::new()
    };

    // DC elements: inflate then undo the ZIP byte preconditioning.
    let dc: Vec<u16> = if header.dc_count > 0 {
        let raw = crate::decoder::zlib_inflate_pub(&payload[dc_r], header.dc_count * 2)?;
        if raw.len() != header.dc_count * 2 {
            return Err(ExrError::invalid(format!(
                "DWA: DC stream inflated to {} bytes, expected {}",
                raw.len(),
                header.dc_count * 2
            )));
        }
        let native = crate::decoder::undo_zip_pipeline_pub(raw);
        native
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    } else {
        Vec::new()
    };

    // RLE bytes: inflate, un-RLE, then byte-plane unsplit per channel.
    let rle_expected: usize = shape
        .sorted_channels
        .iter()
        .zip(extents.iter())
        .zip(classes.iter())
        .filter(|(_, cl)| cl.0 == DwaScheme::Rle)
        .map(|((ch, e), _)| e.nx * e.ny * ch.pixel_type.bytes_per_sample())
        .sum();
    if rle_expected != header.rle_raw_size {
        return Err(ExrError::invalid(format!(
            "DWA: RLE channels need {rle_expected} bytes but header declares {}",
            header.rle_raw_size
        )));
    }
    let rle_raw = if rle_expected > 0 {
        let inflated =
            crate::decoder::zlib_inflate_pub(&payload[rle_r], header.rle_uncompressed_size)?;
        if inflated.len() != header.rle_uncompressed_size {
            return Err(ExrError::invalid(format!(
                "DWA: RLE stream inflated to {} bytes, expected {}",
                inflated.len(),
                header.rle_uncompressed_size
            )));
        }
        crate::rle::rle_decompress(&inflated, header.rle_raw_size)?
    } else {
        Vec::new()
    };

    // Lossy channel-sets, then per-channel decoded half planes.
    let sets = build_lossy_sets(shape.sorted_channels, &classes);
    let expected_dc: usize = sets
        .iter()
        .map(|s| {
            let idx = match s {
                LossySet::Triple(t) => t[0],
                LossySet::Single(i) => *i,
            };
            let e = &extents[idx];
            e.nx.div_ceil(8) * e.ny.div_ceil(8) * s.component_count()
        })
        .sum();
    if expected_dc != header.dc_count {
        return Err(ExrError::invalid(format!(
            "DWA: lossy sets need {expected_dc} DC values but header declares {}",
            header.dc_count
        )));
    }

    let mut half_planes: Vec<Option<Vec<u16>>> = vec![None; shape.sorted_channels.len()];
    let mut ac_pos = 0usize;
    let mut dc_pos = 0usize;
    for set in &sets {
        let ncomp = set.component_count();
        let first = match set {
            LossySet::Triple(t) => t[0],
            LossySet::Single(i) => *i,
        };
        let e = &extents[first];
        let (nx, ny) = (e.nx, e.ny);
        let bx = nx.div_ceil(8);
        let by = ny.div_ceil(8);
        let nblocks = bx * by;

        let mut planes: Vec<Vec<u16>> = vec![vec![0u16; nx * ny]; ncomp];
        let mut blocks: Vec<[f32; 64]> = vec![[0.0f32; 64]; ncomp];
        for blk in 0..nblocks {
            let bx0 = (blk % bx) * 8;
            let by0 = (blk / bx) * 8;
            for (comp, block) in blocks.iter_mut().enumerate() {
                // Scan-order coefficients: DC from the plane-major DC
                // stream, AC un-RLE'd from the block-interleaved AC
                // stream.
                let mut scan = [0u16; 64];
                scan[0] = *dc
                    .get(dc_pos + comp * nblocks + blk)
                    .ok_or_else(|| ExrError::invalid("DWA: DC stream exhausted".to_string()))?;
                let mut pos = 1usize;
                while pos < 64 {
                    let elem = *ac
                        .get(ac_pos)
                        .ok_or_else(|| ExrError::invalid("DWA: AC stream exhausted".to_string()))?;
                    ac_pos += 1;
                    if elem & 0xFF00 == 0xFF00 {
                        let run = (elem & 0xFF) as usize;
                        if run == 0 {
                            // End of block: the rest is zero.
                            pos = 64;
                        } else {
                            pos = pos.checked_add(run).filter(|&p| p <= 64).ok_or_else(|| {
                                ExrError::invalid("DWA: AC zero run overruns the block".to_string())
                            })?;
                        }
                    } else {
                        scan[pos] = elem;
                        pos += 1;
                    }
                }
                // Inverse zig-zag into raster order, half -> f32.
                let mut raster = [0.0f32; 64];
                for (k, &code) in scan.iter().enumerate() {
                    raster[ZIGZAG_RASTER[k]] = crate::half::half_to_f32(code);
                }
                idct8x8(&mut raster);
                *block = raster;
            }
            // Inverse colour transform for triples, then half + inverse
            // perceptual LUT per component channel, cropping mirrored
            // edge texels.
            for yy in 0..8usize {
                let py = by0 + yy;
                if py >= ny {
                    continue;
                }
                for xx in 0..8usize {
                    let px = bx0 + xx;
                    if px >= nx {
                        continue;
                    }
                    let idx = yy * 8 + xx;
                    match set {
                        LossySet::Triple(t) => {
                            let (r, g, b) =
                                csc_inverse(blocks[0][idx], blocks[1][idx], blocks[2][idx]);
                            for (comp, (&chi, v)) in t.iter().zip([r, g, b]).enumerate() {
                                let mut code = crate::half::f32_to_half(v);
                                if !shape.sorted_channels[chi].p_linear {
                                    code = to_linear(code);
                                }
                                planes[comp][py * nx + px] = code;
                            }
                        }
                        LossySet::Single(chi) => {
                            let mut code = crate::half::f32_to_half(blocks[0][idx]);
                            if !shape.sorted_channels[*chi].p_linear {
                                code = to_linear(code);
                            }
                            planes[0][py * nx + px] = code;
                        }
                    }
                }
            }
        }
        dc_pos += ncomp * nblocks;
        match set {
            LossySet::Triple(t) => {
                for (comp, &chi) in t.iter().enumerate() {
                    half_planes[chi] = Some(std::mem::take(&mut planes[comp]));
                }
            }
            LossySet::Single(chi) => {
                half_planes[*chi] = Some(std::mem::take(&mut planes[0]));
            }
        }
    }
    if ac_pos != header.ac_count {
        return Err(ExrError::invalid(format!(
            "DWA: consumed {ac_pos} of {} AC elements",
            header.ac_count
        )));
    }

    // Assemble the native interleaved stream. Verbatim and RLE regions
    // are channel-major; compute each channel's region base first.
    let mut verbatim_base = vec![0usize; shape.sorted_channels.len()];
    let mut rle_base = vec![0usize; shape.sorted_channels.len()];
    {
        let mut vb = 0usize;
        let mut rb = 0usize;
        for (i, (ch, e)) in shape.sorted_channels.iter().zip(extents.iter()).enumerate() {
            let span = e.nx * e.ny * ch.pixel_type.bytes_per_sample();
            match classes[i].0 {
                DwaScheme::Verbatim => {
                    verbatim_base[i] = vb;
                    vb += span;
                }
                DwaScheme::Rle => {
                    rle_base[i] = rb;
                    rb += span;
                }
                DwaScheme::LossyDct => {}
            }
        }
    }

    let mut out = vec![0u8; uncompressed_size];
    let mut wp = 0usize;
    let mut row_of_channel = vec![0usize; shape.sorted_channels.len()];
    for line in 0..shape.lines_in_block {
        for (i, ch) in shape.sorted_channels.iter().enumerate() {
            if !shape.row_present(ch, line) {
                continue;
            }
            let e = &extents[i];
            let bps = ch.pixel_type.bytes_per_sample();
            let row = row_of_channel[i];
            row_of_channel[i] += 1;
            let span = e.nx * bps;
            let dst = out.get_mut(wp..wp + span).ok_or_else(|| {
                ExrError::invalid(format!(
                    "DWA: chunk output overrun at channel '{}' row {row}",
                    ch.name
                ))
            })?;
            wp += span;
            match classes[i].0 {
                DwaScheme::Verbatim => {
                    let src = verbatim_base[i] + row * span;
                    dst.copy_from_slice(&verbatim[src..src + span]);
                }
                DwaScheme::Rle => {
                    // Byte-plane unsplit over the channel's whole region:
                    // byte j of sample s lives at plane j offset s.
                    let nsamples = e.nx * e.ny;
                    let region = &rle_raw[rle_base[i]..rle_base[i] + nsamples * bps];
                    for x in 0..e.nx {
                        let s = row * e.nx + x;
                        for j in 0..bps {
                            dst[x * bps + j] = region[j * nsamples + s];
                        }
                    }
                }
                DwaScheme::LossyDct => {
                    let plane = half_planes[i].as_ref().expect("lossy plane decoded");
                    match ch.pixel_type {
                        PixelType::Half => {
                            for (d, &code) in dst
                                .chunks_exact_mut(2)
                                .zip(plane[row * e.nx..(row + 1) * e.nx].iter())
                            {
                                d.copy_from_slice(&code.to_le_bytes());
                            }
                        }
                        PixelType::Float => {
                            for (d, &code) in dst
                                .chunks_exact_mut(4)
                                .zip(plane[row * e.nx..(row + 1) * e.nx].iter())
                            {
                                d.copy_from_slice(&crate::half::half_to_f32(code).to_le_bytes());
                            }
                        }
                        PixelType::Uint => unreachable!("rejected above"),
                    }
                }
            }
        }
    }
    if wp != uncompressed_size {
        return Err(ExrError::invalid(format!(
            "DWA: assembled {wp} of {uncompressed_size} native bytes"
        )));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Quantise one DCT coefficient to a half code word within the given
/// absolute tolerance, preferring representations with fewer set
/// mantissa bits (patent-described objective; the exact choice is
/// encoder-side only and never signalled — observer-spec §3.5).
fn quantize_coeff(c: f32, tol: f32) -> u16 {
    if c.abs() <= tol {
        return 0;
    }
    let base = crate::half::f32_to_half(c);
    let mut best = base;
    for k in 1..=10u16 {
        let cand = base & !((1u16 << k) - 1);
        if (crate::half::half_to_f32(cand) - c).abs() <= tol {
            best = cand;
        } else {
            break;
        }
    }
    best
}

/// Clamp a FLOAT sample into the finite half range before the DCT
/// (observer-spec §3.5 precision rule) and reduce to half.
fn f32_to_half_clamped(v: f32) -> u16 {
    if v.is_nan() {
        return crate::half::f32_to_half(v);
    }
    crate::half::f32_to_half(v.clamp(-65504.0, 65504.0))
}

/// Compress one chunk's native interleaved byte stream into a DWA
/// (version 2) payload. The caller applies the shared raw fallback.
/// `level` is the DWA compression level (`errorTarget = level / 100000`).
pub(crate) fn dwa_compress(raw: &[u8], shape: &ChunkShape, level: f32) -> Result<Vec<u8>> {
    let extents = shape.extents();
    let classes: Vec<(DwaScheme, i8)> = shape
        .sorted_channels
        .iter()
        .map(|ch| {
            let suffix = channel_suffix(&ch.name);
            for r in DEFAULT_RULES.iter() {
                if r.pixel_type == ch.pixel_type
                    && suffix_matches(r.suffix, suffix, r.case_insensitive)
                {
                    return (r.scheme, r.csc_slot);
                }
            }
            (DwaScheme::Verbatim, -1)
        })
        .collect();

    // Split the native stream into channel-major regions.
    let mut regions: Vec<Vec<u8>> = shape
        .sorted_channels
        .iter()
        .zip(extents.iter())
        .map(|(ch, e)| Vec::with_capacity(e.nx * e.ny * ch.pixel_type.bytes_per_sample()))
        .collect();
    {
        let mut rp = 0usize;
        for line in 0..shape.lines_in_block {
            for (i, ch) in shape.sorted_channels.iter().enumerate() {
                if !shape.row_present(ch, line) {
                    continue;
                }
                let span = extents[i].nx * ch.pixel_type.bytes_per_sample();
                regions[i].extend_from_slice(&raw[rp..rp + span]);
                rp += span;
            }
        }
        debug_assert_eq!(rp, raw.len());
    }

    // --- Verbatim stream ---
    let mut verbatim_raw = Vec::new();
    for (i, _) in shape.sorted_channels.iter().enumerate() {
        if classes[i].0 == DwaScheme::Verbatim {
            verbatim_raw.extend_from_slice(&regions[i]);
        }
    }
    let verbatim_deflated = if verbatim_raw.is_empty() {
        Vec::new()
    } else {
        crate::encoder::zlib_deflate_pub(&verbatim_raw)?
    };

    // --- RLE stream: whole-channel byte-plane split, byte RLE, deflate ---
    let mut rle_split = Vec::new();
    for (i, ch) in shape.sorted_channels.iter().enumerate() {
        if classes[i].0 != DwaScheme::Rle {
            continue;
        }
        let bps = ch.pixel_type.bytes_per_sample();
        let region = &regions[i];
        let nsamples = region.len() / bps;
        for j in 0..bps {
            for s in 0..nsamples {
                rle_split.push(region[s * bps + j]);
            }
        }
    }
    let rle_raw_size = rle_split.len();
    let (rle_rle, rle_deflated) = if rle_raw_size > 0 {
        let r = crate::rle::rle_compress(&rle_split);
        let d = crate::encoder::zlib_deflate_pub(&r)?;
        (r, d)
    } else {
        (Vec::new(), Vec::new())
    };

    // --- Lossy sets: perceptual LUT, CSC, DCT, quantise, AC/DC coding ---
    let sets = build_lossy_sets(shape.sorted_channels, &classes);
    let error_target = level / 100_000.0;
    let mut ac_elems: Vec<u16> = Vec::new();
    let mut dc_elems: Vec<u16> = Vec::new();
    for set in &sets {
        let ncomp = set.component_count();
        let members: Vec<usize> = match set {
            LossySet::Triple(t) => t.to_vec(),
            LossySet::Single(i) => vec![*i],
        };
        let e = &extents[members[0]];
        let (nx, ny) = (e.nx, e.ny);
        if nx == 0 || ny == 0 {
            continue;
        }
        let bx = nx.div_ceil(8);
        let by = ny.div_ceil(8);
        let nblocks = bx * by;

        // Per-component f32 planes after LUT (and clamped half reduction
        // for FLOAT channels), pre-CSC.
        let mut planes: Vec<Vec<f32>> = Vec::with_capacity(ncomp);
        for &chi in &members {
            let ch = &shape.sorted_channels[chi];
            let region = &regions[chi];
            let mut plane = Vec::with_capacity(nx * ny);
            match ch.pixel_type {
                PixelType::Half => {
                    for c in region.chunks_exact(2) {
                        let mut code = u16::from_le_bytes([c[0], c[1]]);
                        if !ch.p_linear {
                            code = to_nonlinear(code);
                        }
                        plane.push(crate::half::half_to_f32(code));
                    }
                }
                PixelType::Float => {
                    for c in region.chunks_exact(4) {
                        let v = f32::from_le_bytes(c.try_into().unwrap());
                        let mut code = f32_to_half_clamped(v);
                        if !ch.p_linear {
                            code = to_nonlinear(code);
                        }
                        plane.push(crate::half::half_to_f32(code));
                    }
                }
                PixelType::Uint => {
                    return Err(ExrError::invalid(format!(
                        "DWA: UINT channel '{}' cannot take the lossy path",
                        ch.name
                    )))
                }
            }
            planes.push(plane);
        }
        // Colour transform for triples (in place, pre-DCT).
        if ncomp == 3 {
            let (p0, rest) = planes.split_at_mut(1);
            let (p1, p2) = rest.split_at_mut(1);
            for ((a, b2), c) in p0[0].iter_mut().zip(p1[0].iter_mut()).zip(p2[0].iter_mut()) {
                let (y, c1, c2) = csc_forward(*a, *b2, *c);
                *a = y;
                *b2 = c1;
                *c = c2;
            }
        }

        // DC values are plane-major; AC interleaves components per block.
        let mut set_dc: Vec<Vec<u16>> = vec![Vec::with_capacity(nblocks); ncomp];
        for blk in 0..nblocks {
            let bx0 = (blk % bx) * 8;
            let by0 = (blk / bx) * 8;
            for (comp, plane) in planes.iter().enumerate() {
                // Gather the 8x8 block with mirrored edges
                // (v -> 2L - v, clamped; no edge repeat).
                let mut block = [0.0f32; 64];
                for yy in 0..8usize {
                    let mut sy = by0 + yy;
                    if sy >= ny {
                        sy = (2 * (ny - 1)).wrapping_sub(sy);
                        if sy >= ny {
                            sy = ny - 1;
                        }
                    }
                    for xx in 0..8usize {
                        let mut sx = bx0 + xx;
                        if sx >= nx {
                            sx = (2 * (nx - 1)).wrapping_sub(sx);
                            if sx >= nx {
                                sx = nx - 1;
                            }
                        }
                        block[yy * 8 + xx] = plane[sy * nx + sx];
                    }
                }
                fdct8x8(&mut block);

                // Quantise under the per-coefficient tolerance: luma
                // matrix for the first component of a triple (and lone
                // channels), chroma for the rest (observer-spec §3.5).
                let (quant, quant_min): (&[u16; 64], f32) = if comp == 0 {
                    (&QUANT_Y, QUANT_Y_MIN)
                } else {
                    (&QUANT_CBCR, QUANT_CBCR_MIN)
                };
                let mut scan = [0u16; 64];
                for (k, &r_idx) in ZIGZAG_RASTER.iter().enumerate() {
                    let tol = error_target * quant[r_idx] as f32 / quant_min;
                    scan[k] = quantize_coeff(block[r_idx], tol);
                }

                set_dc[comp].push(scan[0]);
                // AC run coding over zeros with the end-of-block form
                // (we emit version 2 chunks, so it is available).
                let mut pos = 1usize;
                while pos < 64 {
                    if scan[pos] == 0 {
                        let mut run = 1usize;
                        while pos + run < 64 && scan[pos + run] == 0 {
                            run += 1;
                        }
                        if pos + run == 64 {
                            ac_elems.push(0xFF00); // end of block
                        } else {
                            ac_elems.push(0xFF00 | run as u16);
                        }
                        pos += run;
                    } else {
                        ac_elems.push(scan[pos]);
                        pos += 1;
                    }
                }
            }
        }
        for comp_dc in set_dc {
            dc_elems.extend_from_slice(&comp_dc);
        }
    }

    // --- AC entropy stage (static Huffman, acCompression = 0) ---
    let ac_stream = if ac_elems.is_empty() {
        Vec::new()
    } else {
        crate::huf::huf_compress(&ac_elems)?
    };

    // --- DC stage: ZIP byte preconditioning + deflate ---
    let dc_stream = if dc_elems.is_empty() {
        Vec::new()
    } else {
        let mut bytes = Vec::with_capacity(dc_elems.len() * 2);
        for &v in &dc_elems {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let mut pre = vec![0u8; bytes.len()];
        crate::decoder::apply_zip_interleave(&bytes, &mut pre);
        crate::decoder::apply_zip_predictor(&mut pre);
        crate::encoder::zlib_deflate_pub(&pre)?
    };

    // --- Rule block (version 2): the default rules matched by at least
    // one present channel, in rule order ---
    let mut rule_block = Vec::new();
    for r in DEFAULT_RULES.iter() {
        let used = shape.sorted_channels.iter().any(|ch| {
            ch.pixel_type == r.pixel_type
                && suffix_matches(r.suffix, channel_suffix(&ch.name), r.case_insensitive)
        });
        if !used {
            continue;
        }
        rule_block.extend_from_slice(r.suffix.as_bytes());
        rule_block.push(0);
        let flag = (((r.csc_slot + 1) as u8) << 4)
            | (match r.scheme {
                DwaScheme::Verbatim => 0u8,
                DwaScheme::LossyDct => 1,
                DwaScheme::Rle => 2,
            } << 2)
            | u8::from(r.case_insensitive);
        rule_block.push(flag);
        rule_block.push(match r.pixel_type {
            PixelType::Uint => 0,
            PixelType::Half => 1,
            PixelType::Float => 2,
        });
    }
    let rule_block_size = rule_block.len() + 2;

    // --- Assemble ---
    let mut out = Vec::with_capacity(
        HEADER_BYTES
            + rule_block_size
            + verbatim_deflated.len()
            + ac_stream.len()
            + dc_stream.len()
            + rle_deflated.len(),
    );
    for v in [
        2u64,
        verbatim_raw.len() as u64,
        verbatim_deflated.len() as u64,
        ac_stream.len() as u64,
        dc_stream.len() as u64,
        rle_deflated.len() as u64,
        rle_rle.len() as u64,
        rle_raw_size as u64,
        ac_elems.len() as u64,
        dc_elems.len() as u64,
        0u64, // acCompression: static Huffman
    ] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&(rule_block_size as u16).to_le_bytes());
    out.extend_from_slice(&rule_block);
    out.extend_from_slice(&verbatim_deflated);
    out.extend_from_slice(&ac_stream);
    out.extend_from_slice(&dc_stream);
    out.extend_from_slice(&rle_deflated);
    Ok(out)
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

    #[test]
    fn lut_guards_and_anchors() {
        // Guards: 0 and every all-ones-exponent code word map to 0.
        assert_eq!(to_nonlinear(0), 0);
        assert_eq!(to_linear(0), 0);
        for code in [0x7C00u16, 0x7C01, 0x7FFF, 0xFC00, 0xFFFF] {
            assert_eq!(to_nonlinear(code), 0, "code {code:#06x}");
            assert_eq!(to_linear(code), 0, "code {code:#06x}");
        }
        // 1.0 is a fixed point of both mappings.
        assert_eq!(to_nonlinear(0x3C00), 0x3C00);
        assert_eq!(to_linear(0x3C00), 0x3C00);
        // Anchors read from the staged CSVs.
        assert_eq!(to_nonlinear(0x3800), 0x39D7); // 0.5 -> 0.5^(1/2.2)
        assert_eq!(to_linear(0x3800), 0x32F7); // 0.5 -> 0.5^2.2
        assert_eq!(to_nonlinear(0x4000), 0x3D43); // 2 -> ln2/2.2 + 1
        assert_eq!(to_linear(0x4000), 0x4883); // 2 -> 9.025^(2-1)
        assert_eq!(to_nonlinear(0x0001), 0x1043);
        assert_eq!(to_linear(0x0001), 0x0000);
        // Sign symmetry.
        assert_eq!(to_nonlinear(0x8001), 0x9043);
        assert_eq!(to_linear(0x8000 | 0x3800), 0x8000 | 0x32F7);
    }

    #[test]
    fn dct_constants_match_staged_bits() {
        // The staged butterfly table pins the binary32 bit patterns of
        // 0.5*cos(k*pi/16) with the truncated literal pi = 3.14159.
        assert_eq!(DCT_A.to_bits(), 0x3eb5_04fb);
        assert_eq!(DCT_B.to_bits(), 0x3efb_14bf);
        assert_eq!(DCT_C.to_bits(), 0x3eec_8360);
        assert_eq!(DCT_D.to_bits(), 0x3ed4_db36);
        assert_eq!(DCT_E.to_bits(), 0x3e8e_39e5);
        assert_eq!(DCT_F.to_bits(), 0x3e43_ef33);
        assert_eq!(DCT_G.to_bits(), 0x3dc7_c60b);
    }

    #[test]
    fn zigzag_is_a_permutation() {
        let mut seen = [false; 64];
        for &r in ZIGZAG_RASTER.iter() {
            assert!(!seen[r]);
            seen[r] = true;
        }
        assert!(seen.iter().all(|&s| s));
        // First few entries of the T.81 order.
        assert_eq!(&ZIGZAG_RASTER[..8], &[0, 1, 8, 16, 9, 2, 3, 10]);
    }

    #[test]
    fn dct_roundtrip_is_close() {
        let mut block = [0.0f32; 64];
        for (i, v) in block.iter_mut().enumerate() {
            *v = ((i * 7) % 13) as f32 * 0.25 - 1.0;
        }
        let orig = block;
        fdct8x8(&mut block);
        idct8x8(&mut block);
        for (a, b) in block.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn dct_dc_of_constant_block() {
        let mut block = [1.0f32; 64];
        fdct8x8(&mut block);
        // 2D DC gain is 8: F(0,0) = 8 for a constant 1.0 block.
        assert!((block[0] - 8.0).abs() < 1e-4, "DC = {}", block[0]);
        for (i, &c) in block.iter().enumerate().skip(1) {
            assert!(c.abs() < 1e-4, "AC {i} = {c}");
        }
    }

    #[test]
    fn csc_roundtrip_is_close() {
        for (r, g, b) in [(0.25f32, 0.5f32, 0.75f32), (1.0, 0.0, 0.0), (0.9, 0.9, 0.1)] {
            let (y, c1, c2) = csc_forward(r, g, b);
            let (r2, g2, b2) = csc_inverse(y, c1, c2);
            assert!((r - r2).abs() < 2e-3, "{r} vs {r2}");
            assert!((g - g2).abs() < 2e-3, "{g} vs {g2}");
            assert!((b - b2).abs() < 2e-3, "{b} vs {b2}");
        }
    }

    #[test]
    fn classification_and_sets() {
        let channels = vec![
            ch("A", PixelType::Half),
            ch("B", PixelType::Half),
            ch("G", PixelType::Half),
            ch("R", PixelType::Half),
            ch("Z", PixelType::Float),
        ];
        let rules = static_rules_to_wire(&DEFAULT_RULES);
        let classes: Vec<(DwaScheme, i8)> =
            channels.iter().map(|c| classify_wire(c, &rules)).collect();
        assert_eq!(classes[0], (DwaScheme::Rle, -1)); // A
        assert_eq!(classes[1], (DwaScheme::LossyDct, 2)); // B
        assert_eq!(classes[2], (DwaScheme::LossyDct, 1)); // G
        assert_eq!(classes[3], (DwaScheme::LossyDct, 0)); // R
        assert_eq!(classes[4], (DwaScheme::Verbatim, -1)); // Z
        let sets = build_lossy_sets(&channels, &classes);
        assert_eq!(sets.len(), 1);
        match &sets[0] {
            LossySet::Triple(t) => assert_eq!(*t, [3, 2, 1]), // R, G, B indices
            _ => panic!("expected a triple"),
        }
    }

    #[test]
    fn legacy_rules_match_lowercase_names() {
        let rules = static_rules_to_wire(&LEGACY_RULES);
        let c = ch("green", PixelType::Half);
        assert_eq!(classify_wire(&c, &rules), (DwaScheme::LossyDct, 1));
        let c = ch("layer.RED", PixelType::Float);
        assert_eq!(classify_wire(&c, &rules), (DwaScheme::LossyDct, 0));
        let c = ch("A", PixelType::Half);
        assert_eq!(classify_wire(&c, &rules), (DwaScheme::Rle, -1));
    }

    #[test]
    fn rule_block_roundtrip() {
        // Encode a chunk with a known channel mix, re-parse its rule
        // block, and check the classification survives.
        let channels = vec![
            ch("A", PixelType::Half),
            ch("B", PixelType::Half),
            ch("G", PixelType::Half),
            ch("R", PixelType::Half),
        ];
        let shape = ChunkShape {
            sorted_channels: &channels,
            width: 16,
            block_y0: 0,
            lines_in_block: 8,
        };
        let raw = vec![0u8; 16 * 8 * 2 * 4];
        let payload = dwa_compress(&raw, &shape, DEFAULT_DWA_LEVEL).unwrap();
        let (rules, _) = parse_rule_block(&payload[HEADER_BYTES..]).unwrap();
        assert_eq!(rules.len(), 4); // R, G, B, A (HALF entries only)
        assert_eq!(rules[0].suffix, "R");
        assert_eq!(rules[0].csc_slot, 0);
        assert_eq!(rules[3].suffix, "A");
        assert_eq!(rules[3].scheme, DwaScheme::Rle);
    }

    fn roundtrip_close(
        channels: &[Channel],
        width: u32,
        lines: usize,
        raw: &[u8],
        tol: f32,
    ) -> Vec<u8> {
        let shape = ChunkShape {
            sorted_channels: channels,
            width,
            block_y0: 0,
            lines_in_block: lines,
        };
        let payload = dwa_compress(raw, &shape, DEFAULT_DWA_LEVEL).unwrap();
        assert_ne!(payload.len(), raw.len(), "ambiguous with raw fallback");
        let back = decode_dwa_payload(&payload, &shape, raw.len()).unwrap();
        assert_eq!(back.len(), raw.len());
        // Per-sample comparison in f32 space for HALF channels.
        let mut rp = 0usize;
        for line in 0..lines {
            for ch in channels {
                let _ = line;
                let nx = width as usize;
                match ch.pixel_type {
                    PixelType::Half => {
                        for x in 0..nx {
                            let a = crate::half::half_to_f32(u16::from_le_bytes([
                                raw[rp + 2 * x],
                                raw[rp + 2 * x + 1],
                            ]));
                            let b = crate::half::half_to_f32(u16::from_le_bytes([
                                back[rp + 2 * x],
                                back[rp + 2 * x + 1],
                            ]));
                            assert!(
                                (a - b).abs() <= tol * a.abs().max(1.0),
                                "channel {} x {x}: {a} vs {b}",
                                ch.name
                            );
                        }
                        rp += nx * 2;
                    }
                    _ => {
                        rp += nx * ch.pixel_type.bytes_per_sample();
                    }
                }
            }
        }
        back
    }

    #[test]
    fn lossy_singleton_roundtrip_within_tolerance() {
        let channels = vec![ch("Y", PixelType::Half)];
        let (w, h) = (24usize, 16usize);
        let mut raw = Vec::with_capacity(w * h * 2);
        for i in 0..w * h {
            let v = 0.25 + ((i % 40) as f32) * 0.01;
            raw.extend_from_slice(&crate::half::f32_to_half(v).to_le_bytes());
        }
        roundtrip_close(&channels, w as u32, h, &raw, 0.02);
    }

    #[test]
    fn rgb_triple_and_rle_alpha_roundtrip() {
        let channels = vec![
            ch("A", PixelType::Half),
            ch("B", PixelType::Half),
            ch("G", PixelType::Half),
            ch("R", PixelType::Half),
        ];
        let (w, h) = (16usize, 16usize);
        let mut raw = Vec::new();
        for row in 0..h {
            for c in 0..4 {
                for x in 0..w {
                    let v = match c {
                        0 => (row * w + x) as f32 / (w * h) as f32, // A
                        1 => 0.75,                                  // B
                        2 => 0.5 + 0.2 * (x as f32 / w as f32),     // G
                        _ => 0.25,                                  // R
                    };
                    raw.extend_from_slice(&crate::half::f32_to_half(v).to_le_bytes());
                }
            }
        }
        let back = roundtrip_close(&channels, w as u32, h, &raw, 0.05);
        // The RLE-classified A channel is lossless: bit-compare its rows.
        for row in 0..h {
            let row_span = w * 2 * 4;
            let a0 = row * row_span;
            assert_eq!(
                &back[a0..a0 + w * 2],
                &raw[a0..a0 + w * 2],
                "A channel row {row} not lossless"
            );
        }
    }

    #[test]
    fn verbatim_channels_are_lossless() {
        let channels = vec![ch("P", PixelType::Float), ch("Q", PixelType::Uint)];
        let (w, h) = (20usize, 12usize);
        let mut raw = Vec::new();
        for i in 0..w * h * 2 {
            raw.extend_from_slice(&((i * 2654435761) as u32).to_le_bytes());
        }
        let shape = ChunkShape {
            sorted_channels: &channels,
            width: w as u32,
            block_y0: 0,
            lines_in_block: h,
        };
        let payload = dwa_compress(&raw, &shape, DEFAULT_DWA_LEVEL).unwrap();
        let back = decode_dwa_payload(&payload, &shape, raw.len()).unwrap();
        assert_eq!(back, raw, "verbatim channels must be byte-exact");
    }

    #[test]
    fn rejects_bad_version_and_hostile_sizes() {
        let channels = vec![ch("Y", PixelType::Half)];
        let shape = ChunkShape {
            sorted_channels: &channels,
            width: 8,
            block_y0: 0,
            lines_in_block: 8,
        };
        // version 3.
        let mut p = vec![0u8; 88];
        p[0] = 3;
        assert!(decode_dwa_payload(&p, &shape, 999).is_err());
        // dcCount > 0 with empty DC stream.
        let mut p = vec![0u8; 88];
        p[0] = 0;
        p[72..80].copy_from_slice(&5u64.to_le_bytes()); // dcCount
        assert!(decode_dwa_payload(&p, &shape, 999).is_err());
        // Sub-stream length past chunk end.
        let mut p = vec![0u8; 88];
        p[16..24].copy_from_slice(&1000u64.to_le_bytes()); // unknownCompressedSize
        assert!(decode_dwa_payload(&p, &shape, 999).is_err());
    }
}
