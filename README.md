# oxideav-openexr

Pure-Rust OpenEXR (HDR scanline + tiled image) reader/writer for [`oxideav`].

Clean-room from the public OpenEXR file-format specification.

## Capability matrix

| Capability                          | Status                                           |
| ----------------------------------- | ------------------------------------------------ |
| Magic + version field               | parse + write (format-version 2)                 |
| Attribute table                     | parse + write; eight required attributes typed, plus typed inspectors for `int` / `double` / `string` / `v2i` / `v2d` / `v3i` / `v3f` / `v3d` / `m33f` / `m44f` / `m33d` / `m44d` / `chromaticities` / `box2f` / `tiledesc` / `rational` / `timecode` (BCD time accessors) / `keycode` / `stringvector` |
| Channel list (`chlist`)             | parse + write — `HALF`, `FLOAT`, `UINT`          |
| Compression: `NONE`                 | parse + write                                    |
| Compression: `ZIP`  (16 lines/blk)  | parse + write (zlib)                             |
| Compression: `ZIPS` (1 line/blk)    | parse + write (zlib)                             |
| Compression: `RLE`                  | parse + write (byte-RLE + spec preprocessing)    |
| Compression: `PXR24` (16 lines/blk) | **parse + write** (scanline + tiled + multi-part) — encode: FLOAT→24-bit reduction (round mantissa to 15 bits) + byte-plane horizontal-delta + zlib deflate with raw fallback; decode: zlib inflate + prefix-sum + 24-bit reconstruction. HALF/UINT lossless. Decode validated bit-exact against the staged observer-spec's 24-bit reduction; encode round-trips through our decoder AND is accepted + decoded identically by a reference EXR validator binary |
| Compression: `B44` / `B44A` (32 lines/blk) | **parse + write** (scanline + tiled + multi-part) — per-channel planes; HALF 4×4 blocks (14-byte packed + B44A 3-byte flat), edge replication, optional pLinear exp/log quantisation (tables computed bit-exact vs staged 65 536-entry CSVs); FLOAT/UINT copied raw; shared raw fallback. Encode searches the smallest 6-bit shift, applies the non-linear `exactmax` `t[0]` correction, and emits 3-byte flat blocks for B44A. Decode validated bit-exact against the staged observer-spec's B44 reduction; encode round-trips through our decoder AND is accepted + decoded identically by a reference EXR validator binary (b44/b44a) |
| Single-part scanline                | parse + write                                    |
| Single-part tiled (`ONE_LEVEL`)     | parse + write                                    |
| Tiled `MIPMAP_LEVELS`               | parse + write — full pyramid via `parse_exr_tiled_multilevel`; NONE / ZIP / ZIPS / RLE / PXR24 / B44 / B44A. `parse_exr` returns level-0 only |
| Tiled `RIPMAP_LEVELS`               | parse + write — full 2-D reduction grid; NONE / ZIP / ZIPS / RLE / PXR24 / B44 / B44A |
| Multi-part EXR (scanline parts)     | parse + write                                    |
| Multi-part EXR (flat tiled parts)   | parse + write — ONE_LEVEL + MIPMAP_LEVELS + RIPMAP_LEVELS, edge-tile aware |
| Sub-sampled channels (`xSampling` / `ySampling != 1`) | parse + write                  |
| Deep scanline (`deepscanline`)      | parse + write — NONE / RLE / ZIPS; single- and multi-part |
| Deep tiled (`deeptile`)             | parse + write — ONE_LEVEL + MIPMAP_LEVELS + RIPMAP_LEVELS, edge-tile aware; single- and multi-part |
| Multi-part **mixed** flat + deep    | parse + write — one file may freely mix `scanlineimage`, `tiledimage` (ONE_LEVEL / MIPMAP / RIPMAP), `deepscanline`, and `deeptile` (ONE_LEVEL / MIPMAP / RIPMAP) in any order. Multi-level flat **and deep** tiled parts now carry their full pyramid/grid inline (`MultipartMixedPart::DeepTiledMipmap` / `DeepTiledRipmap`, surfaced as `MultipartMixedImage::DeepTiledMipmap` / `DeepTiledRipmap`). Flat `scanlineimage` and `tiledimage` parts (ONE_LEVEL, MIPMAP, RIPMAP) also carry `PXR24` / `B44` / `B44A` (alongside NONE / ZIP / ZIPS / RLE), reusing the shared block builders + decoders; **deep** parts (scanline, ONE_LEVEL / MIPMAP / RIPMAP tiled) stay NONE / ZIPS / RLE |
| `HALF` (binary16)                   | round-trips every representable pattern (65 536) |
| `UINT` pixel type                   | parse + write (f32 view, bit-exact `< 2^24`)     |

## What this crate does NOT yet cover

* Compression types `PIZ`, `DWAA`, `DWAB` — recognised in the type enum
  but rejected on parse. (`PXR24` and `B44`/`B44A` decode + encode now
  cover scanline, tiled — ONE_LEVEL / MIPMAP / RIPMAP, single- and
  multi-part — and multi-part scanline. PIZ/DWAA/DWAB remain
  DOCS-GAPPED: the staged observer-spec covers only PXR24 + B44/B44A.)
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
* True-HDR pixel-format integration with `oxideav-core` — the
  `Decoder` / `Encoder` shims now clamp to `Rgba64Le` (16-bit per
  channel) for previews, which keeps far more tonal precision than the
  earlier 8-bit path but still tone-maps to [0, 1]. Full floating-point
  HDR awaits an `Rgba128Float`-style pixel format in core.

## Standalone vs registry-integrated

The default `registry` Cargo feature pulls in `oxideav-core` and
exposes the framework `Decoder` / `Encoder` trait surface plus a
`registry::register` entry point.

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

## Fuzzing

A coverage-guided `cargo-fuzz` target lives under `fuzz/`:

```sh
cargo +nightly fuzz run parse_deep_scanline
```

The decode contract is that every byte slice returns `Ok` or `Err`,
never panicking, integer-overflowing (debug build), indexing out of
bounds, or allocating an attacker-claimed length the input can't back.
Offset-table entries (absolute `u64` byte positions read off the wire)
are bounds-checked with overflow-safe arithmetic so a near-`usize::MAX`
entry yields an error rather than wrapping past its EOF guard — see
`tests/offset_table_overflow_hardening.rs`.

## License

MIT — see `LICENSE`.

[`oxideav`]: https://github.com/OxideAV/oxideav-workspace
