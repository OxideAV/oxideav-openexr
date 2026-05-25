# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Round-127 multi-part deep scanline WRITE. New public API:
  `encode_exr_multipart_deep_scanline`, `MultipartDeepScanlinePart`.
  Emits files with version-field bits 0x1800 (multipart + non_image) set,
  per-part `type = "deepscanline"` + `name` + `chunkCount` + `version=1`
  + `maxSamplesPerPixel`, concatenated per-part offset tables (one
  `u64` per chunk, populated with real chunk offsets), then chunks each
  prefixed with `i32 part_number` followed by the standard deep chunk
  record `i32 Y, u64 packed_table, u64 packed_data, u64 unpacked_data,
  table_bytes, data_bytes`. Compression NONE / RLE / ZIPS (deep ZIP
  continues to be rejected — matches `exrinfo`). Self round-trips
  through `parse_exr_deep_multipart`; cross-validated against
  `exrheader` (file accepted, each part dump mentions its name +
  `deepscanline`) and against `exrmultipart -separate`, which splits
  our file into per-part single-part deep .exrs each readable by
  `parse_exr_deep_scanline` with bit-exact channel data. Tests in
  `src/deep.rs` (7 unit tests) + `tests/deep_validation.rs` (4 cross-
  validation tests). Deep-tiled WRITE (`type = "deeptile"`) remains a
  followup.
- Round-124 `RIPMAP_LEVELS` tiled-output encoder. New public API:
  `encode_exr_tiled_rgba_float_ripmap_box_filter`,
  `encode_exr_tiled_ripmap`, `build_box_filter_ripmap`,
  `ripmap_level_counts_round_down`, plus the `RipmapPyramid` /
  `RipmapLevel` types. Writes single-part tiled EXR files with the full
  2-D reduction grid (`tiledesc.level_mode = 2`, ROUND_DOWN): x-levels
  reduce width only, y-levels reduce height only, so cell `(lvlx, lvly)`
  is `mipmap_level_dim(w, lvlx) × mipmap_level_dim(h, lvly)`. The offset
  table and chunk stream walk `lvly` outer, `lvlx` inner, INCREASING_Y
  row-major within each level (matching the decoder's existing RIPMAP
  `compute_total_tiles` ordering). NONE / ZIP / ZIPS / RLE compression.
  `build_box_filter_ripmap` generates a default separable 2× box-filter
  grid; callers needing custom filtering supply their own `RipmapPyramid`.
  Cross-validated against `exrmetrics --convert -z none` (decodes our
  grid back to a scanline file pixel-exactly at level (0,0)) and
  `exrheader` (reports `ripmap`); our decoder is additionally pinned
  against an `exrmaketiled -r` reference file — see
  `tests/ripmap_encoder_validation.rs`.
- Round-92 multi-part deep scanline READ. New public API:
  `parse_exr_deep_multipart`, `DeepScanlinePart`. Walks files with
  version-field bits `0x1800` (multipart + non_image) set; per-part
  `type = "deepscanline"` + `name = "<partName>"` + `chunkCount` +
  `maxSamplesPerPixel` + `version=1`. Chunks read via a linear scan
  with `i32 part_number` prefix on each record followed by the
  standard deep chunk body (`i32 Y, u64 packed_table, u64 packed_data,
  u64 unpacked_data, table_bytes, data_bytes`). Compression NONE /
  RLE / ZIPS — `ZIP_COMPRESSION` continues to be rejected for deep
  data per the reference `exrinfo` convention. The flat
  `parse_exr_multipart` now explicitly rejects deep parts with a
  message pointing at the new entry. Multi-part deep WRITE remains a
  followup. Cross-validated against `exrmultipart -combine`-built
  fixtures with two-part (ZIPS + NONE), three-part (ZIPS + NONE +
  RLE), and many-chunk (12×10 ZIPS) layouts — see
  `tests/deep_validation.rs`.
- Round-78 `MIPMAP_LEVELS` tiled-output encoder. New public API:
  `encode_exr_tiled_rgba_float_mipmap_box_filter`, `encode_exr_tiled_mipmap`,
  `build_box_filter_pyramid`, `mipmap_level_count_round_down`, plus the
  `MipmapLevel` struct. Writes single-part tiled EXR files with
  `tiledesc.level_mode = 1` (MIPMAP_LEVELS, ROUND_DOWN), full-pyramid
  chunk count, and tile chunks emitted in the spec's iteration order
  (levels 0..N-1, INCREASING_Y row-major within each level, `lvlx ==
  lvly == level` per the openexr.com Technical Introduction). Supports
  NONE / ZIP / ZIPS / RLE compression. Cross-validated against
  `exrmetrics --convert -z none` (which decodes our pyramid back to an
  uncompressed scanline file pixel-exactly at level 0) and `exrheader`
  (which reports the file as tiled mipmap). See
  `tests/mipmap_encoder_validation.rs`. The `build_box_filter_pyramid`
  helper synthesises a ROUND_DOWN 2×2 box-filter pyramid for callers
  who don't need to control filtering; callers needing custom filtering
  build the `Vec<MipmapLevel>` themselves and call `encode_exr_tiled_mipmap`.
- Round-73 sub-sampled channel **encoder**. `encode_exr_scanline` and
  `encode_exr_multipart` now honour `xSampling != 1` / `ySampling != 1`
  per the openexr.com spec, matching the per-line "channels whose
  ySampling divides this row contribute samples; each channel writes
  sub-sampled width samples" rule the decoder already uses. The
  earlier explicit "(sub-sampled encode is round 3; decode supports
  it)" rejection is gone. Cross-validated against
  `exrmetrics --convert -z none` on a 4:2:0-style chroma layout
  (Y at 1×1, U/V at 2×2) — see `tests/subsampled_encoder.rs`.
- Round-73 deep-scanline read + write scaffold. New public API:
  `parse_exr_deep_scanline`, `encode_exr_deep_scanline`, `DeepExrImage`,
  `DeepScanlineInput`. Single-part deep files (version-field bit 11,
  `type = "deepscanline"`, required `chunkCount` + `maxSamplesPerPixel`
  + `version` attributes) round-trip through both our reader and the
  reference `exrmetrics --convert -z none` pipeline. The pixel offset
  table (cumulative-inclusive `int` per column) plus non-interleaved
  per-channel sample data layout follows the openexr.com File Layout
  page §Deep scanline part verbatim. Compression set: `NONE` / `RLE` /
  `ZIPS`. `ZIP_COMPRESSION` is intentionally rejected because the
  reference `exrinfo` returns `EXR_ERR_INVALID_ATTR: Invalid compression
  for deep data` on deep ZIP files — the spec page lists ZIP but the
  reference disagrees, and we side with the reference. Validated via
  `exrheader` + `exrinfo` + `exrmetrics --convert` in
  `tests/deep_validation.rs`.

### Changed

- Lib top-level docs updated to describe the round-73 surface and to
  note B44 / Pxr24 cannot be implemented from the openexr.com public
  documentation alone (algorithm sketched in the Technical
  Introduction but exact byte layout is left to the reference source,
  which is off-limits).


## [0.0.2](https://github.com/OxideAV/oxideav-openexr/compare/v0.0.1...v0.0.2) - 2026-05-07

### Other

- round 40: tiled-output encoder + multipart-output encoder
- drop stale REGISTRARS / with_all_features intra-doc links
- drop dead `linkme` dep
- re-export __oxideav_entry from registry sub-module
- round 3: multi-level tiled, multi-part, RLE sign-convention fix
- round 2: RLE/ZIPS, tiled (read), UINT, sub-sampled chroma + spec predictor fix

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
