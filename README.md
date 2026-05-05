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
| Single-part scanline                | yes                                              |
| Single-part tiled (`ONE_LEVEL`)     | parse-only (decode validated against `exrmaketiled`) |
| Sub-sampled channels (`xSampling`/`ySampling != 1`) | parse-only (round-2 followup for encode) |
| `HALF` (binary16)                   | round-trips every representable pattern (65 536) |
| `UINT` pixel type                   | parse + write (f32 view, bit-exact <2^24)        |
| Spec predictor + interleave         | bit-exact against `exrmetrics`-produced files    |

Cross-validation: `exrmetrics --convert -z none` decodes each compressed
output bit-exactly back to the input pixels (see
`tests/exrmetrics_validation.rs`).

## What this round (round 2) does NOT cover

* Compression types: `PIZ`, `PXR24`, `B44`, `B44A`, `DWAA`, `DWAB`.
  Recognised in the type enum but rejected on parse.
* Tiled multi-level (`MIPMAP_LEVELS`, `RIPMAP_LEVELS`) files.
* Sub-sampled channel encoding (decode supports it).
* Multi-part files.
* Deep-data scanlines.
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
