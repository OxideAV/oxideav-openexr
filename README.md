# oxideav-openexr

[![CI](https://github.com/OxideAV/oxideav-openexr/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-openexr/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-openexr.svg)](https://crates.io/crates/oxideav-openexr) [![docs.rs](https://docs.rs/oxideav-openexr/badge.svg)](https://docs.rs/oxideav-openexr) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust OpenEXR (HDR scanline + tiled image) reader/writer for [`oxideav`].

Clean-room from the public OpenEXR file-format specification.

## Capability matrix

| Capability                          | Status                                           |
| ----------------------------------- | ------------------------------------------------ |
| Magic + version field               | parse + write (format-version 2)                 |
| Attribute table                     | parse + write; eight required attributes typed, plus typed inspectors for `int` / `double` / `string` / `v2i` / `v2d` / `v3i` / `v3f` / `v3d` / `m33f` / `m44f` / `m33d` / `m44d` / `chromaticities` / `box2f` / `tiledesc` / `rational` / `timecode` (BCD time accessors) / `keycode` / `stringvector` / `envmap` / `preview` / `floatvector` / `deepImageState` |
| Channel list (`chlist`)             | parse + write — `HALF`, `FLOAT`, `UINT`          |
| `lineOrder` (INCREASING_Y / DECREASING_Y / RANDOM_Y) | parse + write — observer-derived (r410): the chunk offset table is ALWAYS keyed canonically (top-first / ty-outer-tx-inner walk) and `lineOrder` governs only physical chunk storage order; RANDOM_Y is invalid for scanline images (writer rejects; reader stays lenient since chunks self-describe coordinates). `*_with_line_order` writer variants for scanline, tiled ONE_LEVEL, MIPMAP and RIPMAP; DECREASING_Y stores rows bottom-first, RANDOM_Y (tiled only) a deterministic shuffle. Reference-validated (header echo + independent reader + convert pixel-exact); derivation record in `tests/line_order_observer_notes.md` |
| Compression: `NONE`                 | parse + write                                    |
| Compression: `ZIP`  (16 lines/blk)  | parse + write (zlib)                             |
| Compression: `ZIPS` (1 line/blk)    | parse + write (zlib)                             |
| Compression: `RLE`                  | parse + write (byte-RLE + spec preprocessing)    |
| Compression: `PXR24` (16 lines/blk) | **parse + write** (scanline + tiled + multi-part) — encode: FLOAT→24-bit reduction (round mantissa to 15 bits) + byte-plane horizontal-delta + zlib deflate with raw fallback; decode: zlib inflate + prefix-sum + 24-bit reconstruction. HALF/UINT lossless. Decode validated bit-exact against the staged observer-spec's 24-bit reduction; encode round-trips through our decoder AND is accepted + decoded identically by a reference EXR validator binary. r410: the raw fallback now stores/detects the NATIVE chunk bytes (`compressed_len == uncompressed_len`), matching conforming readers — incompressible chunks carry FLOAT at full precision |
| Compression: `PIZ` (32 lines/blk)   | **parse + write** (scanline + tiled — ONE_LEVEL / MIPMAP / RIPMAP — + multi-part + mixed) — lossless: occupancy bitmap + chunk-derived range-compaction LUT + hierarchical 2D wavelet (14-bit and modulo-2^16 variants, selection recomputed from the bitmap) + canonical static Huffman (58-bit max codes, run-length escape at `iM`); HALF/FLOAT/UINT + sub-sampled channels; shared raw fallback. Landed r439 from the staged trace `openexr-piz-dwa-observer-spec.md` §2. Validated **bit-exact both directions** against a reference EXR binary (opaque process): reference-encoded PIZ decodes identically, and our PIZ files are accepted + decoded identically |
| Compression: `DWAA` (32 lines/blk) / `DWAB` (256 lines/blk) | **parse + write** (scanline + tiled — ONE_LEVEL / MIPMAP / RIPMAP — + multi-part + mixed) — 88-byte eleven-slot chunk header, version-2 rule block (+ staged legacy rule set for v0/v1), verbatim / AC / DC / RLE sub-streams, (suffix, pixel-type) channel classification, BT.709 CSC triples, binary32 perceptual half LUTs, staged truncated-π IDCT butterfly, plane-major DC + per-block AC with half-NaN run escapes, whole-region byte-plane RLE split; encoder emits v2 + static-Huffman AC, honours `dwaCompressionLevel` (default 45). Landed r439 (trace §3 + nine staged tables; implicit stream orderings observer-derived, record in `tests/dwa_observer_notes.md`). Decode validated **bit-exact** vs the reference's own decode of reference-encoded files; our chunks accepted + decoded bit-identically by the reference |
| Compression: `B44` / `B44A` (32 lines/blk) | **parse + write** (scanline + tiled + multi-part) — per-channel planes; HALF 4×4 blocks (14-byte packed + B44A 3-byte flat), edge replication, optional pLinear exp/log quantisation (tables computed bit-exact vs staged 65 536-entry CSVs); FLOAT/UINT copied raw; shared raw fallback. Encode searches the smallest 6-bit shift, applies the non-linear `exactmax` `t[0]` correction, and emits 3-byte flat blocks for B44A. Decode validated bit-exact against the staged observer-spec's B44 reduction; encode round-trips through our decoder AND is accepted + decoded identically by a reference EXR validator binary (b44/b44a) |
| Single-part scanline                | parse + write                                    |
| Single-part tiled (`ONE_LEVEL`)     | parse + write                                    |
| Tiled `MIPMAP_LEVELS`               | parse + write — full pyramid via `parse_exr_tiled_multilevel`; NONE / ZIP / ZIPS / RLE / PXR24 / B44 / B44A (r410: the single-part writer now carries the lossy schemes too, reference-validated incl. bit-exact reference-decode vs our-decode). `parse_exr` returns level-0 only |
| Tiled `RIPMAP_LEVELS`               | parse + write — full 2-D reduction grid; NONE / ZIP / ZIPS / RLE / PXR24 / B44 / B44A (r410: single-part writer incl. lossy) |
| Multi-part EXR (scanline parts)     | parse + write                                    |
| Multi-part EXR (flat tiled parts)   | parse + write — ONE_LEVEL + MIPMAP_LEVELS + RIPMAP_LEVELS, edge-tile aware |
| Sub-sampled channels (`xSampling` / `ySampling != 1`) | parse + write — lossless AND lossy (PXR24 / B44 / B44A) scanline paths; luminance/chroma (`Y` + 2×2 `BY`/`RY`) layouts validated bit-exact against a reference EXR validator binary. Note: the reference reader requires sub-sampled data-window extents divisible by the sampling factor; our reader additionally accepts ceil-sized odd extents |
| Layered / multi-view channel names (`diffuse.R`, `left.R` / `right.R`, `a.b.c.Y` …) | **typed enumeration + framework mapping** — `layers` module groups the channel list by prefix (arbitrary depth), classifies each layer (RGBA / RGB / luma-chroma / gray / depth / other) and tags views from `multiView`; the registry decoder's `layer` option selects a layer (or the default view) and the encoder's `layer` option writes prefixed names. Validated against reference-produced multi-view files |
| Luminance/chroma colour (`Y` + `RY` + `BY` ↔ RGB) | **decode + encode** (`luma_chroma` module + registry decoder) — `RY = (R − Y) / Y`, `BY = (B − Y) / Y` with luminance weights derived from the `chromaticities` attribute (BT.709 when absent); chroma reconstructed bilinearly from any `(xSampling, ySampling)`, reduced with a centred tent filter. Validated against a reference EXR tool (opaque process): chroma ratios and luminance match to HALF precision on constant-chroma images (filter-independent), smooth gradients agree to a colour-level tolerance; our files are accepted by the reference |
| Deep scanline (`deepscanline`)      | parse + write — NONE / RLE / ZIPS; single- and multi-part |
| Deep tiled (`deeptile`)             | parse + write — ONE_LEVEL + MIPMAP_LEVELS + RIPMAP_LEVELS, edge-tile aware; single- and multi-part |
| Multi-part **mixed** flat + deep    | parse + write — one file may freely mix `scanlineimage`, `tiledimage` (ONE_LEVEL / MIPMAP / RIPMAP), `deepscanline`, and `deeptile` (ONE_LEVEL / MIPMAP / RIPMAP) in any order. Multi-level flat **and deep** tiled parts now carry their full pyramid/grid inline (`MultipartMixedPart::DeepTiledMipmap` / `DeepTiledRipmap`, surfaced as `MultipartMixedImage::DeepTiledMipmap` / `DeepTiledRipmap`). Flat `scanlineimage` and `tiledimage` parts (ONE_LEVEL, MIPMAP, RIPMAP) also carry `PXR24` / `B44` / `B44A` (alongside NONE / ZIP / ZIPS / RLE), reusing the shared block builders + decoders; **deep** parts (scanline, ONE_LEVEL / MIPMAP / RIPMAP tiled) stay NONE / ZIPS / RLE |
| `HALF` (binary16)                   | round-trips every representable pattern (65 536) |
| `UINT` pixel type                   | parse + write (f32 view, bit-exact `< 2^24`)     |

## What this crate does NOT yet cover

* (Resolved r439.) `PIZ`, `DWAA` and `DWAB` — the last blocked
  compression schemes — now decode AND encode across every flat
  surface; the ten-code compression matrix is complete for flat
  images. Deep parts deliberately stay NONE / ZIPS / RLE (the spec
  text forbids PIZ for deep data and the validators reject deep ZIP).
* A reference EXR B44A decoder zeroes pLinear channels (its
  plain-B44 decoder of identical data does not); our codec follows the
  observer-spec, so pLinear validation runs on the self-consistent
  plain-B44 path.
* `ZIP_COMPRESSION` is rejected for deep data (the format validators
  reject deep ZIP files even though the spec page text lists ZIP as
  permitted).
* (Resolved r382.) Mixed multi-part files may now include multi-level
  (MIPMAP / RIPMAP) **deep** tiled parts alongside every other part type
  — see the capability matrix. The dedicated
  `parse_exr_multipart_deep_tiled_mipmap` /
  `parse_exr_multipart_deep_tiled_ripmap` readers remain available for
  homogeneous deep multi-level files.
* Lossy `PXR24` / `B44` / `B44A` for **deep** parts (deep scanline and
  deep tiled) — deep parts stay NONE / ZIP / ZIPS / RLE. (All **flat**
  mixed parts — scanline + ONE_LEVEL / MIPMAP / RIPMAP tiled — now carry
  the lossy schemes; see the capability matrix.)
* Framework frames for channel sets outside RGB(A) / `Y` / `Y RY BY`:
  depth-only (`Z`) and AOV-only layers have no `PixelFormat` mapping
  and decode `Unsupported` through the registry — the standalone
  `parse_exr` API still returns every channel. Deep parts likewise
  (variable samples per pixel); use `parse_exr_deep_*`.
* One layer per decode: the framework fixes a stream's pixel format
  once, so the registry decoder cannot emit every layer of a layered /
  multi-view image as separate frames — select each with the `layer`
  option (enumerate them with `ExrImage::layers`).
* The luminance/chroma colour reconstruction uses bilinear chroma
  interpolation (decode) and a centred tent reduction (encode); a
  reference EXR reader applies a different filter, so colour-level
  agreement is to a tolerance (tight in the interior, loosest at the
  image edges) while the container level stays bit-exact.

## Standalone vs registry-integrated

The default `registry` Cargo feature pulls in `oxideav-core` and
exposes the framework `Decoder` / `Encoder` trait surface plus a
`registry::register` entry point. The framework path is true HDR
(`oxideav-core` 0.1.35+): the decoder emits scene-referred linear
`RgbaF32Le` / `RgbF32Le` / `GrayF32Le` frames — HALF widened exactly,
FLOAT copied bit-for-bit, UINT converted, never clamped or tone-mapped —
choosing the format from the part's channel set (`R G B A` → RGBA,
`R G B` → RGB, `Y RY BY` → RGB reconstructed from luminance/chroma,
`Y` → gray, `Y A` → RGBA with `Y` replicated). The
`part` decoder option picks a part in multi-part files (multi-level
parts contribute level 0; deep parts are `Unsupported`) and the `layer`
option a channel-name prefix (`diffuse`, `right`, …) or the default
view. The encoder
accepts the same three formats and writes `A B G R` / `B G R` / `Y`
scanline channels — or, with `colour=luma_chroma`, `A BY RY Y` /
`BY RY Y` with the chroma sub-sampled by `chroma_sampling` (default
2×2); `pixel_type` selects `float` (default, lossless round trip) or
`half`, and `compression` any of `none rle zips zip piz pxr24 b44 b44a
dwaa dwab`. See `src/registry.rs` for the full rules.

For image-library callers that don't want the framework dependency,
build with `default-features = false`:

```toml
oxideav-openexr = { version = "0.0", default-features = false }
```

The standalone API stays available either way:

```rust
use oxideav_openexr::{parse_exr, encode_exr_scanline_rgba_float};

let bytes = encode_exr_scanline_rgba_float(width, height, &rgba_f32).unwrap();
let img = parse_exr(&bytes).unwrap();
```

## Benchmarks

A Criterion harness (`benches/codec_benchmarks.rs`) measures decode /
encode throughput across every compression scheme for HALF and FLOAT,
scanline and tiled — see [`BENCHMARKS.md`](BENCHMARKS.md) for the
current numbers (scanline HALF NONE decode 2.6 GiB/s, FLOAT NONE
19.7 GiB/s, PXR24 1.1–2.4 GiB/s, PIZ 0.27–0.35 GiB/s and DWAA/DWAB
0.21–0.44 GiB/s after the round-446 optimisation pass, on Apple
Silicon).

## Fuzzing

Three coverage-guided `cargo-fuzz` targets live under `fuzz/`:

```sh
cargo +nightly fuzz run parse_flat
cargo +nightly fuzz run parse_deep_scanline
cargo +nightly fuzz run parse_multipart_mixed
```

`parse_flat` attacks the single-part flat readers — `parse_exr`
(scanline + tiled ONE_LEVEL) and `parse_exr_tiled_multilevel`
(MIPMAP / RIPMAP) — the only route into the PXR24 byte-plane/delta,
B44/B44A 4×4-block, PIZ (bitmap / range LUT / wavelet /
canonical-Huffman) and DWAA/DWAB (rule block / sub-streams / AC run
coding / IDCT) decode arithmetic; its overlay mode splices fuzz bytes
over the offset-table + chunk region of writer-produced valid files
across shape × compression × lineOrder combinations (all ten
compression codes since r439).
`parse_deep_scanline` attacks the deep scanline chunk walk.
`parse_multipart_mixed` attacks the mixed multi-part reader — the
per-part chunk-shape dispatch (flat scanline / flat + deep tiled at
every level mode), the concatenated offset tables, the i32 tile/level
coordinates, and the u64 deep chunk sizes — both raw and by splicing
fuzz bytes over the chunk region of a writer-produced two-part file.

The decode contract is that every byte slice returns `Ok` or `Err`,
never panicking, integer-overflowing (debug build), indexing out of
bounds, or allocating an attacker-claimed length the input can't back.
Offset-table entries (absolute `u64` byte positions read off the wire)
are bounds-checked with overflow-safe arithmetic so a near-`usize::MAX`
entry yields an error rather than wrapping past its EOF guard — see
`tests/offset_table_overflow_hardening.rs` and
`tests/multipart_mixed_hardening.rs`.

Sustained fuzzing of the DWA decode path surfaced three defects, all
fixed and pinned by unit tests: two reservation out-of-memories where a
chunk header's declared inflated size / symbol count sized an
allocation before any bytes were produced (the shared inflate helper
and the static-Huffman decoder now cap the eager reservation and let
the buffer grow only to what the stream yields), and an out-of-bounds
panic where an over-subscribed code-length table produced a Huffman
code that did not fit its bit length (the length distribution is now
validated as a proper prefix code before it indexes the decode table).

## License

MIT — see `LICENSE`.

[`oxideav`]: https://github.com/OxideAV/oxideav-workspace
