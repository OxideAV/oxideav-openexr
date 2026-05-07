# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Round-40 tiled-output encoder (`encode_exr_tiled_rgba_float_with`,
  `encode_exr_tiled`) — single-part `ONE_LEVEL` tiled files with
  NONE / ZIP / ZIPS / RLE compression. Sets the version-field
  `single_tile` bit, emits `tiles` (tiledesc) + `chunkCount` + `type`
  attributes, builds the tile offset table in INCREASING_Y row-major
  order, and writes per-tile `tx | ty | lvlx | lvly | size | payload`
  chunks. Edge tiles (right column / bottom row partial tiles)
  validated against `exrmetrics --convert -z none`.
- Round-40 multipart-output encoder (`encode_exr_multipart`,
  `encode_exr_multipart_rgba_float_with`, `MultipartScanlinePart`) —
  multipart files with one or more independent scanline parts. Sets
  the version-field `multipart` bit, emits per-part headers (each with
  required `name` / `type=scanlineimage` / `chunkCount` attributes plus
  the standard required attributes) terminated by a double NUL, then
  per-part offset tables, then chunks each prefixed with the
  `part_number` integer. Supports NONE / ZIP / ZIPS / RLE per part.
  Verified via `parse_exr_multipart` self-roundtrip and via the
  `exrmultipart -separate` reference binary.
- New integration suite `tests/round40_encoder_validation.rs` —
  cross-validates tiled and multipart encoder output against
  `exrheader`, `exrmetrics`, and `exrmultipart -separate`. Auto-skips
  on hosts without the OpenEXR CLI tools.
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
- Round-3: multi-level tiled read — `parse_exr` now handles
  `MIPMAP_LEVELS` and `RIPMAP_LEVELS` tiled files; the full-resolution
  level (lvlx=0, lvly=0) is decoded, reduction levels are skipped.
  Offset table is correctly sized via `compute_total_tiles`.
- Round-3: multi-part EXR read — `parse_exr_multipart` decodes files
  with version-field bit 12 set; it parses per-part headers, skips the
  (possibly zero-filled) concatenated offset tables, and routes chunks
  by embedded part number via sequential scan.
- Public helpers `mipmap_level_count` and `mipmap_level_dim` for
  ROUND_DOWN / ROUND_UP level-dimension arithmetic.
- `parse_multipart_headers` (public) for callers that only need header
  metadata from multi-part files.
- Integration tests in `tests/multilevel_validation.rs`: mipmap (ZIP /
  ZIPS / RLE / NONE / PIZ / B44 / B44A) and ripmap (ZIP / RLE) via
  `exrmaketiled`; multi-part via `exrmultipart`; unit tests for the
  level-count / level-dim helpers (all auto-skip if tools absent).

### Fixed

- ZIP / ZIPS / RLE byte predictor previously used the naive
  `out[i] = raw[i] - raw[i-1]` form. The openexr.com spec mandates the
  centred form `out[i] = (raw[i] - raw[i-1] + 128) & 0xFF` (decoder
  inverse `raw[i] = (in[i] + raw[i-1] - 128) & 0xFF`). Self-roundtrip
  worked but the bytes were not actually spec-compliant; external
  decoders saw garbage. Fixed and validated against `exrmetrics`.
- RLE control-byte sign convention was inverted relative to the OpenEXR
  reference implementation. The spec documentation is ambiguous, but
  empirical validation against `exrmetrics` and `exrmaketiled` output
  confirms: `c >= 0` = repeat `(c+1)` times; `c < 0` = literal `(-c)`
  bytes. The previous implementation had these backwards, producing
  correct self-roundtrips but failing to decode external RLE files.

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
