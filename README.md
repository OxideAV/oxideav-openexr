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
| Single-part scanline                | parse + write                                    |
| Single-part tiled (`ONE_LEVEL`)     | parse + write                                    |
| Tiled `MIPMAP_LEVELS`               | parse + write — full pyramid via `parse_exr_tiled_multilevel`; NONE / ZIP / ZIPS / RLE. `parse_exr` returns level-0 only |
| Tiled `RIPMAP_LEVELS`               | parse + write — full 2-D reduction grid; NONE / ZIP / ZIPS / RLE |
| Multi-part EXR (scanline parts)     | parse + write                                    |
| Multi-part EXR (flat tiled parts)   | parse + write — ONE_LEVEL + MIPMAP_LEVELS + RIPMAP_LEVELS, edge-tile aware |
| Sub-sampled channels (`xSampling` / `ySampling != 1`) | parse + write                  |
| Deep scanline (`deepscanline`)      | parse + write — NONE / RLE / ZIPS; single- and multi-part |
| Deep tiled (`deeptile`)             | parse + write — ONE_LEVEL + MIPMAP_LEVELS + RIPMAP_LEVELS, edge-tile aware; single- and multi-part |
| Multi-part **mixed** flat + deep    | parse + write — one file may freely mix `scanlineimage`, `tiledimage` (ONE_LEVEL), `deepscanline`, and `deeptile` (ONE_LEVEL) in any order |
| `HALF` (binary16)                   | round-trips every representable pattern (65 536) |
| `UINT` pixel type                   | parse + write (f32 view, bit-exact `< 2^24`)     |

## What this crate does NOT yet cover

* Compression types `PIZ`, `PXR24`, `B44`, `B44A`, `DWAA`, `DWAB` —
  recognised in the type enum but rejected on parse. The public
  format-spec page gives only one-line summaries for these; the exact
  block layouts are not pinned by clean-room material.
* `ZIP_COMPRESSION` is rejected for deep data (the format validators
  reject deep ZIP files even though the spec page text lists ZIP as
  permitted).
* Mixed multi-part files that include multi-level (MIPMAP / RIPMAP)
  tiled parts alongside other types.
* HDR pixel-format integration with `oxideav-core` — the
  `Decoder` / `Encoder` shims clamp to `Rgba` 8-bit pending an
  `Rgba128Float`-style pixel format addition to core.

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
