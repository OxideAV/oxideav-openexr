# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- CI shim wired to the OxideAV org-level reusable workflows
  (`crate-ci.yml` + `crate-release.yml`) plus an inline
  `ci-standalone` job that builds + tests with `--no-default-features`.
- Round-2 compression coverage: `RLE` (byte-RLE on top of the spec's
  predictor + interleave preprocessing) and `ZIPS` (per-scanline
  zlib) — full encode + decode round-trip.
- `UINT` pixel type — parse + write (f32 view; bit-exact for values
  below 2^24).
- Sub-sampled channels (`xSampling != 1` or `ySampling != 1`) — decode
  side now produces per-channel f32 planes sized to each channel's
  effective sub-sampled dimensions. Encode side still requires 1×1.
- Tiled single-part EXR files (`single_tile` bit, `ONE_LEVEL` mode) —
  decode side handles per-tile `Y(i32) | size(i32)`-equivalent chunk
  framing with `tx,ty,lvlx,lvly,size` headers and the same compression
  pipeline as scanline blocks. Multi-resolution `MIPMAP_LEVELS` /
  `RIPMAP_LEVELS` deferred to round 3.
- Cross-validation tests against the `exrmetrics --convert -z none` and
  `exrmaketiled` reference binaries (auto-skip when the OpenEXR CLI
  tools are missing).

### Fixed

- ZIP / ZIPS / RLE byte predictor previously used the naive
  `out[i] = raw[i] - raw[i-1]` form. The openexr.com spec mandates the
  centred form `out[i] = (raw[i] - raw[i-1] + 128) & 0xFF` (decoder
  inverse `raw[i] = (in[i] + raw[i-1] - 128) & 0xFF`). Self-roundtrip
  worked but the bytes were not actually spec-compliant; external
  decoders saw garbage. Fixed and validated against `exrmetrics`.

### Changed

- `parse_exr` now also handles single-part tiled files (header parser
  no longer rejects the `single_tile` flag bit). The crate-level
  `parse_exr` doc updated accordingly.

## [0.0.1] - 2026-05-05

### Added

- Initial release: pure-Rust OpenEXR scanline reader/writer, clean-room
  from the openexr.com file format spec.
- Magic + version field (format-version 2, no flag bits).
- Attribute table parser/encoder with typed values for the eight
  required attributes.
- Channel list (`chlist`) parser/encoder for `HALF` and `FLOAT` pixel
  types.
- Compression: `NO_COMPRESSION` and `ZIP` (16 scanlines per block, zlib
  via `flate2` with the spec's interleave + predictor transforms).
- IEEE 754-2008 binary16 (`half`) <-> `f32` codec — round-trips every
  representable pattern (65536 cases).
- Public standalone API: `parse_exr`, `encode_exr_scanline_rgba_float`,
  `encode_exr_scanline_rgba_float_with`, `encode_exr_scanline`.
- Default-on `registry` Cargo feature wires the codec into
  `oxideav-core` via the framework `Decoder`/`Encoder` trait surface
  and registers the `.exr` extension.
- Auto-registration into `oxideav_core::REGISTRARS` via the
  `oxideav_core::register!` macro (linkme distributed slice).
