# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Round-189 docs scrub. Removed every citation of the tainted
  `openexr.com` project-shipped docs URL (Task #1240) from `README.md`,
  `CHANGELOG.md`, `Cargo.toml` (crate description), and all `src/` +
  `tests/` doc-comments. 57 occurrences replaced with neutral phrasing
  — "the OpenEXR file format spec", "the OpenEXR Technical
  Introduction", "the reference `exrinfo` validator", "the reference
  encoder", etc. The format name "OpenEXR" itself is just a format
  identifier and stays. The `exrheader` / `exrinfo` / `exrmetrics` /
  `exrmaketiled` binaries remain referenced as opaque black-box
  validator processes, which is permitted by the round allow-list.
  No functional code changes; all 178 tests still pass; standalone
  (no-default-features) build still green. Rationale: per
  `docs/image/openexr/README.md` policy notice (2026-05-24), the
  `openexr.com` documentation site is rendered from project-shipped
  `.rst` sources, which creates a derivative-work relationship with
  the reference implementation regardless of the BSD-3-Clause licence
  and therefore taints clean-room reimplementations' independent
  copyright.

### Added

- Round-181 multi-part deep TILED WRITE + READ. New public API:
  `encode_exr_multipart_deep_tiled`, `parse_exr_multipart_deep_tiled`,
  `MultipartDeepTiledPart`, `DeepTiledPart`. Composes the round-127
  multi-part deep-scanline envelope (version-field bits 0x1800 =
  multipart + non_image; concatenated per-part headers terminated by a
  double NUL; concatenated per-part offset tables) with the round-130
  single-part deep-tiled chunk shape, prefixed per chunk by
  `i32 part_number`. Each tile chunk on disk is
  `i32 part_number, i32 tx, i32 ty, i32 lvlx, i32 lvly, u64 packed_table,
  u64 packed_data, u64 unpacked_data, packed_table_bytes,
  packed_sample_bytes` (44 bytes of header + the two byte blobs).
  Per-part attributes mirror the single-part deep-tiled writer plus the
  mandatory `name` attribute: `channels`, `chunkCount[int]`,
  `compression`, `dataWindow`, `displayWindow`, `lineOrder`,
  `maxSamplesPerPixel[int]`, `name[string]`, `pixelAspectRatio`,
  `screenWindowCenter`, `screenWindowWidth`, `tiles[tiledesc]` (ONE_LEVEL
  + ROUND_DOWN), `type[string="deeptile"]`, `version[int=1]`. The reader
  uses the same linear-scan strategy as `parse_exr_multipart` and
  `parse_exr_deep_multipart` to remain robust against zero-filled
  offset tables. Compression NONE / RLE / ZIPS (deep ZIP rejected to
  match the reference `exrinfo` validator and the single-part
  deep-tiled discipline). NONE-compressed pixel-offset tables accept
  both the canonical `tw * th * 4`-byte size and the `tile_x * tile_y *
  4`-byte padded size emitted by the reference encoder (mirrors
  the single-part deep-tiled reader). Edge tiles are trimmed to their
  valid pixel rectangle in both encoder and decoder. 8 new unit tests
  in `src/deep.rs` cover: 2-part ZIPS roundtrip + version-field bit
  invariants (multipart=0x1000, non_image=0x800, single_tile MUST NOT
  be set), 3-part NONE/RLE/ZIPS mixed-compression roundtrip, 13×9-in-
  4×3 edge-tile roundtrip, rejection of empty parts list, duplicate
  names, deep ZIP, single-part deep-tiled bytes, and multi-part deep-
  scanline bytes — plus an all-zero-samples-in-one-part edge case. The
  remaining followups in this codepath (multi-level deep tiled
  MIPMAP/RIPMAP, single- and multi-part) are tracked in
  `lib.rs`.
- Round-174 full-pyramid READ for tiled `MIPMAP_LEVELS` / `RIPMAP_LEVELS`
  files. New public API: `parse_exr_tiled_multilevel`, `MultilevelTiledImage`,
  `TiledLevel`. The existing `parse_exr` continues to return only the
  full-resolution level (no behaviour change for callers that already
  used it on a multi-level file). The new entry point decodes every
  level: ONE_LEVEL files surface as a single-entry `levels` vector;
  MIPMAP files surface levels `0..N-1` with `level_x == level_y`; RIPMAP
  files surface the full 2-D grid in the spec's iteration order (`lvly`
  outer, `lvlx` inner), with each cell sized to `mipmap_level_dim(w,
  lvlx) × mipmap_level_dim(h, lvly)` and its own per-channel f32 plane.
  Compression NONE / ZIP / ZIPS / RLE supported (others rejected with a
  clear `Unsupported` error pointing at the same compression-rejection
  set the rest of the crate uses). The existing `compute_total_tiles`
  helper drives offset-table sizing; per-tile chunk lookup is by
  `(lvlx, lvly)` not by index, so a malformed offset table that carries
  an unknown level produces an actionable error. Pure-Rust round-trip
  tests in `tests/multilevel_read_validation.rs` encode pyramids via
  the existing `encode_exr_tiled_mipmap` / `encode_exr_tiled_ripmap`
  writers (6-level MIPMAP at 32×32, 5×4 = 20 cell RIPMAP at 16×8, plus
  a non-power-of-two MIPMAP at 16×12) and confirm every sample of every
  level matches the input bit-exactly. Backward-compat test pins
  `parse_exr` level-0 against `parse_exr_tiled_multilevel` level
  `(0, 0)`. The encoder side keeps its existing `exrmetrics --convert`
  + `exrmaketiled -r` cross-validation in
  `tests/{mipmap,ripmap}_encoder_validation.rs`.
- Round-130 single-part deep TILED WRITE + READ. New public API:
  `encode_exr_deep_tiled`, `parse_exr_deep_tiled`, `DeepTiledInput`,
  `DeepTiledImage`. Emits / consumes single-part deep-tiled files with
  `type = "deeptile"`, the `tiles[tiledesc]` attribute (ONE_LEVEL +
  ROUND_DOWN), `chunkCount = tx_count * ty_count`, `version = 1`, and
  `maxSamplesPerPixel`. Empirical version-field discovery: single-part
  deep tiled sets the `non_image` (0x800) bit ONLY — `exrheader`
  rejects files that also set `single_tile` (0x200) here. Each tile
  chunk on disk: `i32 tx, i32 ty, i32 lvlx, i32 lvly, u64 packed_table,
  u64 packed_data, u64 unpacked_data, packed_table_bytes,
  packed_sample_bytes`. Per-tile offset table is `tw * th * 4` bytes of
  cumulative i32 entries (row-major within the tile's valid rectangle)
  for compressed chunks; the reader additionally accepts the reference
  encoder's NONE-compression convention of padding to
  `tile_x * tile_y * 4` bytes so files produced by `exrmetrics --convert
  -z none` round-trip cleanly. Sample data is non-interleaved
  (channel-major within each tile). Compression NONE / RLE / ZIPS (deep
  ZIP rejected to match the reference `exrinfo` validator). The
  reader trims edge tiles to their valid pixel rectangle and reassembles
  channel samples into pixel-scan row-major order before return, so
  callers don't have to know the file was tiled. Tests in `src/deep.rs`
  (9 unit tests covering NONE/ZIPS/RLE self-roundtrip, 13×9-in-4×4 edge
  tiles, all-zero samples, ZIP rejection, sub-sampled rejection,
  rejection of flat / scanline-deep files) + `tests/deep_validation.rs`
  (6 cross-validation tests: `exrheader` accepts the file with the
  expected `deeptile` / `tiles` dump, `exrmetrics --convert -z none`
  round-trips NONE / ZIPS / RLE / edge-tile cases bit-exactly back
  through `parse_exr_deep_tiled`, plus a 23×17-in-6×5 pure-Rust
  roundtrip). MIPMAP/RIPMAP deep tiled + multi-part deep tiled are
  followups.
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
  lvly == level` per the OpenEXR Technical Introduction). Supports
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
  per the OpenEXR spec, matching the per-line "channels whose
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
  per-channel sample data layout follows the OpenEXR File Layout
  page §Deep scanline part verbatim. Compression set: `NONE` / `RLE` /
  `ZIPS`. `ZIP_COMPRESSION` is intentionally rejected because the
  reference `exrinfo` returns `EXR_ERR_INVALID_ATTR: Invalid compression
  for deep data` on deep ZIP files — the spec page lists ZIP but the
  reference disagrees, and we side with the reference. Validated via
  `exrheader` + `exrinfo` + `exrmetrics --convert` in
  `tests/deep_validation.rs`.

### Changed

- Lib top-level docs updated to describe the round-73 surface and to
  note B44 / Pxr24 cannot be implemented from the the public OpenEXR
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
  `out[i] = raw[i] - raw[i-1]` form. The OpenEXR spec mandates the
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
  from the OpenEXR file format spec.
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
