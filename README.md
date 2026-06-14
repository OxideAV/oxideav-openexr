# oxideav-openexr

Pure-Rust OpenEXR (HDR scanline + tiled image) reader/writer for [`oxideav`].

Clean-room from the OpenEXR file format spec (public format
documentation). Empirical validation against the `exrheader` /
`exrinfo` / `exrmetrics` / `exrmaketiled` binaries, invoked as opaque
processes (input bytes in, stdout/stderr text out).

## Capability matrix

| Capability                          | Status                                           |
| ----------------------------------- | ------------------------------------------------ |
| Magic + version field               | parse + write (format-version 2)                 |
| Attribute table                     | parse + write, eight required attributes typed plus typed inspectors for `int` / `double` / `string` / `v2i` / `v2d` / `v3i` / `v3f` / `v3d` / `m33f` / `m44f` / `chromaticities` / `box2f` / `tiledesc` / `rational` / `timecode` (BCD time accessors) / `keycode` / `stringvector` (round-trip + validated by `exrheader`) |
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
| Multi-part EXR (flat tiled parts)   | parse + write — ONE_LEVEL, NONE/ZIP/ZIPS/RLE, edge-tile aware (validated against `exrheader` + `exrmultipart -separate` round-trip back through `parse_exr`) |
| Multi-part flat tiled MIPMAP_LEVELS | parse + write (`encode_exr_multipart_tiled_mipmap` / `parse_exr_multipart_tiled_multilevel`) — full ROUND_DOWN pyramid per part, NONE/ZIP/ZIPS/RLE per part, edge-tile aware. Validated against `exrheader` ("mip-map") + `exrmultipart -separate` (each split = a single-part MIPMAP file our `parse_exr_tiled_multilevel` decodes bit-exactly back to the source pyramid) |
| Multi-part flat tiled RIPMAP_LEVELS | parse + write (`encode_exr_multipart_tiled_ripmap` / `parse_exr_multipart_tiled_multilevel`) — full 2-D ROUND_DOWN reduction grid per part (`lvly`-outer `lvlx`-inner), NONE/ZIP/ZIPS/RLE per part, edge-tile aware. Validated against `exrheader` ("rip-map") + `exrmultipart -separate` (each split = a single-part RIPMAP file our `parse_exr_tiled_multilevel` decodes bit-exactly back to the source grid) |
| Sub-sampled channels (`xSampling`/`ySampling != 1`) | parse + write (validated against `exrmetrics --convert`) |
| Deep scanline (`deepscanline`)      | parse + write — NONE/RLE/ZIPS (validated against `exrheader` + `exrmetrics --convert -z none`) |
| Multi-part deep scanline (read)     | parse — NONE/RLE/ZIPS, per-part `Vec<DeepScanlinePart>` (validated against `exrmultipart -combine`) |
| Multi-part deep scanline (write)    | encode — NONE/RLE/ZIPS, per-part `MultipartDeepScanlinePart` (validated against `exrheader` + `exrmultipart -separate` round-trip back through `parse_exr_deep_scanline`) |
| Single-part deep tiled (`deeptile`) | parse + encode — ONE_LEVEL, NONE/RLE/ZIPS, edge-tile aware (validated against `exrheader` + `exrmetrics --convert` round-trip back through `parse_exr_deep_tiled`) |
| Single-part deep tiled MIPMAP_LEVELS | parse + encode (`encode_exr_deep_tiled_mipmap` / `parse_exr_deep_tiled_mipmap`) — full ROUND_DOWN pyramid, NONE/RLE/ZIPS, edge-tile aware. Validated against `exrheader` ("mip-map" + "deeptile") + pure-Rust pyramid-roundtrip across power-of-two and non-power-of-two 24×16 with ZIPS |
| Single-part deep tiled RIPMAP_LEVELS | parse + encode (`encode_exr_deep_tiled_ripmap` / `parse_exr_deep_tiled_ripmap`) — full 2-D ROUND_DOWN reduction grid (`lvly`-outer `lvlx`-inner) with explicit `(lvlx, lvly)` per chunk, NONE/RLE/ZIPS, edge-tile aware. Validated against `exrheader` ("rip-map" + "deeptile") + pure-Rust grid-roundtrip across power-of-two and non-power-of-two 24×16 with ZIPS |
| Multi-part deep tiled (`deeptile`)  | parse + encode — ONE_LEVEL per part, NONE/RLE/ZIPS, edge-tile aware (self-roundtrip on 2- and 3-part mixed-compression layouts) |
| Multi-part deep tiled MIPMAP_LEVELS  | parse + encode (`encode_exr_multipart_deep_tiled_mipmap` / `parse_exr_multipart_deep_tiled_mipmap`) — full ROUND_DOWN pyramid per part, NONE/RLE/ZIPS per part, edge-tile aware, supports parts with distinct level-0 dimensions. Self-roundtrips 2- and 3-part mixed-compression layouts plus 13×9 non-power-of-two edge-tile cases. ONE_LEVEL multi-part files dispatched to `parse_exr_multipart_deep_tiled`; MIPMAP multi-part files (tiledesc mode=0x01) dispatched here |
| Multi-part deep tiled RIPMAP_LEVELS  | parse + encode (`encode_exr_multipart_deep_tiled_ripmap` / `parse_exr_multipart_deep_tiled_ripmap`) — full 2-D ROUND_DOWN reduction grid per part (`lvly`-outer `lvlx`-inner), NONE/RLE/ZIPS per part, edge-tile aware, supports parts with distinct level-(0,0) dimensions. Self-roundtrips 2- and 3-part mixed-compression layouts plus 13×9 non-power-of-two edge-tile cases. The ONE_LEVEL multi-part reader (`parse_exr_multipart_deep_tiled`) and MIPMAP multi-part reader (`parse_exr_multipart_deep_tiled_mipmap`) both redirect RIPMAP multi-part files (tiledesc mode=0x02) here; the single-part deep RIPMAP reader (`parse_exr_deep_tiled_ripmap`) redirects multi-part RIPMAP files here too |
| Multi-part **mixed** flat + deep | parse + encode (`encode_exr_multipart_mixed` / `parse_exr_multipart_mixed`) — a single multi-part file may freely mix all four part types: `type="scanlineimage"`, `type="tiledimage"` (ONE_LEVEL), `type="deepscanline"`, and `type="deeptile"` (ONE_LEVEL), in arbitrary order. Compression per part: NONE/ZIP/ZIPS/RLE for flat parts, NONE/ZIPS/RLE for deep parts. The reader walks chunks linearly and dispatches each chunk-body shape (scanline `i32 Y, i32 size, payload`; tiled `i32 tx, i32 ty, i32 lvlx, i32 lvly, i32 size, payload`; deep scanline `i32 Y, 3×u64 sizes, table, data`; deep tiled `4×i32 coords, 3×u64 sizes, table, data`) via the part's declared `type`. The version field sets `non_image` (0x800) alongside `multipart` (0x1000) when any part is deep; one shared `displayWindow` (union of part data windows) spans all parts as the reference validator requires. Self-roundtrips 2-, 3-, and 4-part layouts with mixed per-part compression, distinct per-part dimensions, edge-tile cases, empty deep pixels, and HALF / FLOAT / UINT pixel-type mixes. Validated against `exrheader` (prints all four part types) + `exrmultipart -separate` (each of the 4 splits decodes bit-exactly through `parse_exr` / `parse_exr_deep_scanline` / `parse_exr_deep_tiled`) |
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
  trace doc; the public format-spec page gives only a one-line
  summary. B44/Pxr24 are mentioned in the Technical Introduction at a
  high level only — exact 14-byte block layout is left to the
  reference source, which we don't consult.
* `ZIP_COMPRESSION` is rejected for deep data (matching the reference
  `exrinfo` validator, which returns `EXR_ERR_INVALID_ATTR` on deep
  ZIP files even though the spec page text lists ZIP as permitted).
* Tiled-output encode now covers `ONE_LEVEL`, `MIPMAP_LEVELS` (full
  pyramid) and `RIPMAP_LEVELS` (full 2-D reduction grid) — all ROUND_DOWN,
  NONE / ZIP / ZIPS / RLE.
* Multipart-output encode covers scanline parts, flat tiled parts
  (ONE_LEVEL + MIPMAP_LEVELS + RIPMAP_LEVELS), deep-scanline parts, and
  deep-tiled parts at every level mode (ONE_LEVEL + MIPMAP_LEVELS +
  RIPMAP_LEVELS), plus mixed multi-part files combining any of
  `scanlineimage`, `tiledimage` (ONE_LEVEL), `deepscanline`, and
  `deeptile` (ONE_LEVEL) — `encode_exr_multipart`,
  `encode_exr_multipart_tiled`, `encode_exr_multipart_tiled_mipmap`,
  `encode_exr_multipart_tiled_ripmap`,
  `encode_exr_multipart_deep_scanline`, `encode_exr_multipart_deep_tiled`,
  `encode_exr_multipart_deep_tiled_mipmap`,
  `encode_exr_multipart_deep_tiled_ripmap`,
  `encode_exr_multipart_mixed`. Mixed multi-part files that include
  multi-level (MIPMAP/RIPMAP) tiled parts alongside other types are a
  followup.
* The deep-tiled matrix is closed: single-part and multi-part deep
  tiled both support ONE_LEVEL, MIPMAP_LEVELS, and RIPMAP_LEVELS.
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

## Fuzzing

A coverage-guided `cargo-fuzz` target lives under `fuzz/`:

* `parse_deep_scanline` — drives `parse_exr_deep_scanline` over both raw
  fuzz bytes and a structurally valid deep file whose offset table and
  per-block headers (`packed_table` / `packed_data` / `unpacked_data`)
  are overlaid with fuzz-controlled bytes, exercising the deep chunk
  arithmetic without first rediscovering a valid header.

```sh
cargo +nightly fuzz run parse_deep_scanline
```

The decode contract is that every byte slice returns `Ok` or `Err`,
never panicking, integer-overflowing (debug build), indexing out of
bounds, or allocating an attacker-claimed length the input can't back.
The target surfaced and fixed two hostile-input defects: a 64-bit
per-block size that overflowed the `table_start + packed_table` offset
sum, and a `dataWindow` pairing `x_min = i32::MIN` with
`x_max = i32::MAX` that overflowed `Box2i::width()` / drove an unbacked
pixel-grid allocation.

## License

MIT — see `LICENSE`.

[`oxideav`]: https://github.com/OxideAV/oxideav-workspace
