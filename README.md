# oxideav-openexr

Pure-Rust OpenEXR (HDR scanline image) reader/writer for [`oxideav`].

Clean-room from the OpenEXR file format spec at
<https://openexr.com/en/latest/OpenEXRFileLayout.html>. No ILM/Academy
source, no `exr` / `openexr-rs` Rust crates, and no Imath consulted.
Empirical validation against the `exrheader` / `exrinfo` / `exrenvmap`
binaries (used as opaque oracles only).

## Round-1 surface

| Capability             | Status                                                 |
| ---------------------- | ------------------------------------------------------ |
| Magic + version field  | parse + write (format-version 2, no flag bits)         |
| Attribute table        | parse + write, eight required attributes typed         |
| Channel list (`chlist`)| parse + write, `HALF` and `FLOAT` pixel types          |
| Compression: NONE      | parse + write                                          |
| Compression: ZIP       | parse + write (16 lines/block, zlib via `flate2`)      |
| Single-part scanline   | yes                                                    |
| HALF (binary16)        | round-trips every representable pattern (65536 cases)  |

## What round 1 does NOT cover

* Other compression types: `RLE`, `ZIPS`, `PIZ`, `PXR24`, `B44`, `B44A`,
  `DWAA`, `DWAB`. Recognised in the type enum but rejected on parse.
* Tiled format (single-tile bit set in the version field).
* Multi-part files.
* Deep-data scanlines.
* `UINT` channel pixel type.
* Sub-sampled channels (`xSampling != 1` or `ySampling != 1`).
* HDR pixel-format integration with `oxideav-core` (the `Decoder`/`Encoder`
  shims clamp to `Rgba` 8-bit pending an `Rgba128Float`-style pixel
  format addition to core).

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
