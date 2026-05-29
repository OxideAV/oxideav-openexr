# oxideav-openexr

Pure-Rust OpenEXR (HDR scanline + tiled image) reader/writer for [`oxideav`].

Clean-room from the OpenEXR file format spec at
<https://openexr.com/en/latest/OpenEXRFileLayout.html>. No ILM/Academy
source, no `exr` / `openexr-rs` Rust crates, and no Imath consulted.
Empirical validation against the `exrheader` / `exrinfo` / `exrmetrics`
/ `exrmaketiled` binaries (used as opaque oracles only).

## Capability matrix

| Capability                          | Status                                           |
| ----------------------------------- | ------------------------------------------------ |
| Magic + version field               | parse + write (format-version 2)                 |
| Attribute table                     | parse + write, eight required attributes typed   |
| Channel list (`chlist`)             | parse + write — `HALF`, `FLOAT`, `UINT`          |
| Compression: `NONE`                 | parse + write                                    |
| Compression: `ZIP`  (16 lines/blk)  | parse + write (zlib via `flate2`)                |
| Compression: `ZIPS` (1 line/blk)    | parse + write (zlib via `flate2`)                |
| Compression: `RLE`                  | parse + write (byte-RLE + spec preprocessing)    |
| Single-part scanline                | parse + write                                    |
| Single-part tiled (`ONE_LEVEL`)     | parse + write (validated against `exrmetrics`)   |
| Tiled `MIPMAP_LEVELS` (read)        | full pyramid via `parse_exr_tiled_multilevel` (level-0..N-1) — round-trips encoder bit-exactly; `parse_exr` still returns level-0 only |
| Tiled `MIPMAP_LEVELS` (write)       | full pyramid encode (NONE / ZIP / ZIPS / RLE) — validated against `exrmetrics --convert` and `exrheader` |
| Tiled `RIPMAP_LEVELS` (read)        | full 2-D grid via `parse_exr_tiled_multilevel` (every `(lvlx, lvly)` cell) — round-trips encoder bit-exactly |
| Tiled `RIPMAP_LEVELS` (write)       | full 2-D reduction grid encode (NONE / ZIP / ZIPS / RLE) — validated against `exrmetrics --convert` + `exrheader`, decoder pinned vs `exrmaketiled -r` |
| Multi-part EXR (scanline parts)     | parse + write (validated against `exrmultipart -separate`) |
| Sub-sampled channels (`xSampling`/`ySampling != 1`) | parse + write (validated against `exrmetrics --convert`) |
| Deep scanline (`deepscanline`)      | parse + write — NONE/RLE/ZIPS (validated against `exrheader` + `exrmetrics --convert -z none`) |
| Multi-part deep scanline (read)     | parse — NONE/RLE/ZIPS, per-part `Vec<DeepScanlinePart>` (validated against `exrmultipart -combine`) |
| Multi-part deep scanline (write)    | encode — NONE/RLE/ZIPS, per-part `MultipartDeepScanlinePart` (validated against `exrheader` + `exrmultipart -separate` round-trip back through `parse_exr_deep_scanline`) |
| Single-part deep tiled (`deeptile`) | parse + encode — ONE_LEVEL, NONE/RLE/ZIPS, edge-tile aware (validated against `exrheader` + `exrmetrics --convert` round-trip back through `parse_exr_deep_tiled`) |
| Multi-part deep tiled (`deeptile`)  | parse + encode — ONE_LEVEL per part, NONE/RLE/ZIPS, edge-tile aware (self-roundtrip on 2- and 3-part mixed-compression layouts) |
| `HALF` (binary16)                   | round-trips every representable pattern (65 536) |
| `UINT` pixel type                   | parse + write (f32 view, bit-exact <2^24)        |
| Spec predictor + interleave         | bit-exact against `exrmetrics`-produced files    |

Cross-validation: `exrmetrics --convert -z none` decodes each compressed
output bit-exactly back to the input pixels (see
`tests/exrmetrics_validation.rs`). Mipmap / ripmap levels validated
against `exrmaketiled`; multi-part validated against `exrmultipart`
(see `tests/multilevel_validation.rs`).

## What this crate does NOT yet cover

* Compression types: `PIZ`, `PXR24`, `B44`, `B44A`, `DWAA`, `DWAB`.
  Recognised in the type enum but rejected on parse. PIZ requires a
  wavelet + Huffman coder for which we don't yet have a clean-room
  trace doc; the openexr.com public spec page gives only a one-line
  summary. B44/Pxr24 are mentioned in the Technical Introduction at a
  high level only — exact 14-byte block layout is left to the
  reference source, which we don't consult.
* `ZIP_COMPRESSION` is rejected for deep data (matching the openexr.com
  reference `exrinfo`, which returns `EXR_ERR_INVALID_ATTR` on deep ZIP
  files even though the spec page text lists ZIP as permitted).
* Tiled-output encode now covers `ONE_LEVEL`, `MIPMAP_LEVELS` (full
  pyramid) and `RIPMAP_LEVELS` (full 2-D reduction grid) — all ROUND_DOWN,
  NONE / ZIP / ZIPS / RLE.
* Multipart-output encode covers scanline parts, deep-scanline parts,
  and deep-tiled parts (`encode_exr_multipart`,
  `encode_exr_multipart_deep_scanline`,
  `encode_exr_multipart_deep_tiled`); flat tiled parts are not yet
  emitted in multipart form.
* Deep-tiled support is **ONE_LEVEL only** (single-part *and*
  multi-part) — MIPMAP/RIPMAP-level deep tiled is a followup.
* HDR pixel-format integration with `oxideav-core` (the
  `Decoder`/`Encoder` shims clamp to `Rgba` 8-bit pending an
  `Rgba128Float`-style pixel format addition to core).

## Standalone vs registry-integrated

The default `registry` Cargo feature pulls in `oxideav-core` and exposes
the framework `Decoder` / `Encoder` trait surface plus a
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

## License

MIT — see `LICENSE`.

[`oxideav`]: https://github.com/OxideAV/oxideav-workspace
