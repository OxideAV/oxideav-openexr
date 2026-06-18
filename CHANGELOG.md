# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Round-335 **B44 / B44A compression / encode (single-part scanline)**:
  the scanline encoder now emits `B44` and `B44A` (`src/b44.rs`
  `encode_b44_chunk` + `src/encoder.rs` `build_b44_block_payload`). A
  chunk is regrouped into per-channel contiguous planes; each HALF plane
  is tiled into 4×4 blocks with right-column / bottom-row edge
  replication. `pLinear` HALF channels pass through a forward "exp"
  (from-linear) quantisation LUT — *computed* from the documented closed
  form (`float_to_half(exp(half_to_float(x)/8))` with the §2.3 clamp at
  code word `0x558c → 0x7bff`) using this crate's bit-exact binary16
  conversions, not embedded from any array. Each block is packed by the
  monotone sign-magnitude remap, a smallest-6-bit-`shift` search with
  round-half-to-even `shiftAndRound`, the 2-D differencing tree (bias
  `0x20`), and — for non-linear channels — the `exactmax` `t[0]`
  correction (observer-spec §2.4). B44A additionally emits the 3-byte
  flat block (marker `0xfc`) for constant 4×4 regions (§2.5). The shared
  raw-fallback stores the interleaved uncompressed stream when the packed
  chunk would not be smaller. Validated by a self round-trip through our
  decoder (pixel-level fixed-point on non-linear channels), a reference
  cross-check (`exrmetrics` transcodes our B44/B44A bytes to NONE and the
  reference decode bit-matches ours, pLinear included), and ratio sanity
  checks (`tests/b44_encode_validation.rs`). The encoder's supported set
  is now NONE / ZIP / ZIPS / RLE / PXR24 / B44 / B44A.
- Round-331 **B44 / B44A decompression (single-part scanline)**: the
  fixed-ratio lossy HALF compressor and its flat-block variant are now
  decoded (`src/b44.rs`). A B44 chunk is regrouped into per-channel
  contiguous planes; each HALF plane is tiled into 4×4 blocks (14-byte
  packed, or B44A 3-byte flat marked by `b[2] >= 0x34`) with right-column
  / bottom-row edge replication, then the 2-D differencing tree rooted at
  `t[0]` is prefix-summed back and the monotone sign-magnitude remap
  inverted (observer-spec §2.2–2.5). `pLinear` HALF channels additionally
  pass through an inverse "log" (to-linear) dequantisation LUT, which is
  *computed* from the documented closed form
  (`float_to_half(8·log(half_to_float(x)))` with the §2.3 sentinel
  clamps) using this crate's bit-exact binary16 conversions — verified
  bit-exact against all 65 536 entries of the staged
  `tables/b44-log-table.csv`. FLOAT / UINT channels are copied raw and
  the shared raw-fallback (compressed length == uncompressed length) is
  honoured. Validated bit-exact against the reference's own B44 decode
  via `exrmetrics -z b44`/`b44a` re-conversion across single-block,
  odd-size edge-replication, flat-block, and non-linear cases
  (`tests/b44_decode_validation.rs`). Note OpenEXR 3.4.x's B44A decoder
  zeroes pLinear channels (its plain-B44 decoder of identical data does
  not), so pLinear validation runs on the self-consistent plain-B44 path;
  our decoder follows the observer-spec uniformly. Tiled / multi-part B44
  decode and all B44 encode remain follow-ups.
- Round-326 **PXR24 compression / encode (single-part scanline)**: the
  lossy 24-bit FLOAT compressor can now be written, completing the PXR24
  read+write pair for scanline images. `encode_exr_scanline` /
  `encode_exr_scanline_rgba_float_with` accept `Compression::Pxr24`. Each
  FLOAT sample is reduced to its 24-bit code (round the mantissa to 15
  bits, drop the low byte; inf/NaN handled per observer-spec §1.1), HALF
  and UINT pass through losslessly; the block is reorganised into
  per-row, per-channel, byte-plane-major form with most-significant-first
  horizontal deltas, then zlib-deflated. The universal raw-fallback rule
  is honoured: if deflate does not shrink the reorganised stream the raw
  bytes are stored and the decoder detects this by length. Validated by a
  self round-trip through our PXR24 decoder, an incompressible-data
  raw-fallback round-trip, and a reference cross-check where `exrmetrics`
  reads our PXR24 file and transcodes it to ZIP yielding the same spec-
  reduced pixels (`tests/pxr24_encode_validation.rs`). Tiled / multi-part
  PXR24 encode remain follow-ups.
- Round-321 **PXR24 decompression (single-part scanline)**: the lossy
  24-bit FLOAT compressor is now decoded. The block payload is
  zlib-inflated to a per-row, per-channel, byte-plane-major stream;
  each plane carries the most-significant-first horizontal deltas of the
  channel's samples, which are prefix-summed back into per-sample integer
  codes. FLOAT codes are reconstructed from their 24-bit form (the
  dropped low byte is implicitly zero), HALF and UINT are preserved
  losslessly. Multi-block images and odd widths are handled; an encoder
  raw-fallback (compressed == reorganised size) is detected. Validated
  bit-exact against the reference `exrmetrics -z pxr24` transcode plus
  the observer-spec §1.1 worked-fact reduction table
  (`tests/pxr24_decode_validation.rs`). PXR24 encode and tiled /
  multi-part PXR24 decode remain follow-ups.

### Fixed

- Round-318 **offset-table overflow hardening (untrusted input)**: each
  EXR data chunk is located through an offset table whose entries are
  absolute `u64` byte positions read straight off the wire. The decoder
  cast each entry to `usize` and then bounds-checked the chunk header
  with `block_off + 8 > bytes.len()` (scanline) or `tile_off + 20 >
  bytes.len()` (single-tile + multi-level tile). A hostile entry near
  `usize::MAX` overflowed the `+ N` *inside* the bounds check itself —
  a panic in debug builds, a wrap into an out-of-bounds slice (also a
  panic) in release — so a malformed file could crash the decoder before
  the past-EOF guard ever ran. All three offset-table dereference sites
  in `decoder.rs` (`parse_exr` scanline loop, `parse_exr` single-tile
  loop, `parse_exr_tiled_multilevel` loop) now compute the header end
  with `checked_add(..).filter(|e| *e <= len)` and the payload end with
  `checked_add(payload_size)`, returning a clean `ExrError::invalid`
  instead of unwinding. Well-formed files decode unchanged. New
  `tests/offset_table_overflow_hardening.rs` (5 tests) corrupts the
  first offset-table entry of a valid scanline / tiled / mipmapped file
  with `u64::MAX` (and a just-past-EOF value) and asserts each parser
  returns `Err` without panicking, plus a happy-path regression guard.
  fmt + clippy clean; all 184 library/integration tests pass.

- Round-309 **library-lockfile hygiene**: stop tracking the crate-root
  `Cargo.lock` (this is a library, so the resolved lockfile must not live
  in version control) and add a `.gitignore` carrying `/target` plus a
  `Cargo.lock` line. The previously-committed root lock pinned
  `oxideav-core` at a stale crates.io patch (`0.1.26`) while the umbrella
  patch table resolves the local path crate (`0.1.28`); an untracked lock
  removes that dual-version hazard and lets each consumer resolve the
  intended single version. The `fuzz/` binary crate keeps its own
  `Cargo.lock` (binaries pin their dependency graph; `fuzz/.gitignore`
  already excludes only build artefacts). No source changes; all 179
  library tests + every integration suite still pass; fmt + clippy clean.

### Added

- Round-303 **typed `m33d` / `m44d` double-precision matrix attributes**:
  the header attribute parser + encoder now model `m33d` (nine
  little-endian `f64` in row-major order, 72 bytes) and `m44d` (sixteen
  little-endian `f64`, 128 bytes) as the typed `AttributeValue::M33d` /
  `AttributeValue::M44d` variants — the double-precision companions of
  the existing `M33f` / `M44f`, completing the float/double matrix
  pairing alongside `v2f`/`v2d` and `v3f`/`v3d`. Both round-trip
  bit-exactly through `encode_attribute_value` / `parse_attribute_value`
  and through a full scanline header via `encode_exr_scanline` /
  `parse_exr`; payloads with the wrong byte length are rejected. Both
  are validated against the `exrheader` binary (invoked as an opaque
  process) in `tests/typed_attribute_roundtrip.rs`.

- Round-296 **`parse_deep_scanline` cargo-fuzz target** (`fuzz/`): a
  coverage-guided libfuzzer harness over `parse_exr_deep_scanline`,
  driving both raw fuzz bytes and a structurally valid deep file whose
  offset table + per-block headers are overlaid with fuzz-controlled
  bytes so the deep chunk arithmetic is reached without first
  rediscovering a valid header. ~265k execs/run, crash-free after the
  fixes below.

### Fixed

- Round-296 **deep-scanline hostile-input hardening** (found by the new
  fuzz target). A 64-bit per-block `packed_table` / `packed_data` size
  read off the wire overflowed the `table_start + packed_table` /
  `+ packed_data` offset sums (debug-build panic, silent wrap past the
  EOF bound in release); the chunk-header offsets now use checked
  arithmetic and reject out-of-range blocks. A `dataWindow` pairing
  `x_min = i32::MIN` with `x_max = i32::MAX` overflowed
  `Box2i::width()` / `height()`; the extent is now computed in `i64` and
  clamped into `u32`. `parse_exr_deep_scanline` additionally rejects a
  pixel grid larger than the input could possibly back before allocating
  the per-pixel sample-count buffer. 3 new regression tests.

- Round-282 **deep parts inside mixed multi-part files**:
  `encode_exr_multipart_mixed` / `parse_exr_multipart_mixed` now accept
  any combination of all four part types — `scanlineimage`,
  `tiledimage` (ONE_LEVEL), `deepscanline`, and `deeptile` (ONE_LEVEL)
  — in arbitrary order in one file. `MultipartMixedPart` gains
  `DeepScanline` / `DeepTiled` input variants (per-pixel sample counts
  + channel-major sample lists, mirroring the homogeneous deep
  multi-part writers); `MultipartMixedImage` gains matching output
  variants wrapping `DeepScanlinePart` / `DeepTiledPart`, plus
  `deep_scanline()` / `deep_tiled()` / `is_deep_scanline()` /
  `is_deep_tiled()` accessors. Compression per part: NONE/ZIP/ZIPS/RLE
  for flat parts, NONE/ZIPS/RLE for deep parts. The version field sets
  the `non_image` (0x800) bit alongside `multipart` (0x1000) when at
  least one part is deep. Multi-level tiled parts in a mixed file are
  still rejected with a redirect at the dedicated multi-level readers.
  Validated against `exrheader` (accepts the 4-type file and prints
  every part) and `exrmultipart -separate` (the 4-way split decodes
  bit-exactly back through `parse_exr`, `parse_exr_deep_scanline`, and
  `parse_exr_deep_tiled`). 7 new tests.

### Changed

- `MultipartMixedImage::image()` / `into_image()` now return
  `Option<&ExrImage>` / `Option<ExrImage>` (`None` for deep parts).
- The mixed multi-part writer now emits one shared `displayWindow` (the
  union of all part data windows) across every part instead of
  duplicating each part's `dataWindow`. The reference `exrheader`
  validator refuses multi-part files whose parts carry different
  `displayWindow` values, which previously made mixed files with
  distinct per-part dimensions unreadable by the reference tools (the
  pure-Rust reader accepted them either way).

### Added

- Round-273 **optional standard header-attribute types**: six previously
  `Other`-passthrough attribute types now parse + encode as typed
  `AttributeValue` variants. `v2d` (two LE `f64`, 16 bytes), `v3d`
  (three LE `f64`, 24 bytes), `rational` (`i32` numerator + `u32`
  denominator, 8 bytes — used by `framesPerSecond`), `timecode`
  (`Timecode { time_and_flags, user_data }`, two LE `u32`, 8 bytes; the
  `time_and_flags` word is stored verbatim and BCD `hours`/`minutes`/
  `seconds`/`frames` accessors decode the four time nibbles),
  `keycode` (`Keycode` — seven LE `i32` in the order film-mfc-code,
  film-type, prefix, count, perf-offset, perfs-per-frame,
  perfs-per-count; 28 bytes), and `stringvector` (a sequence of
  `i32`-length-prefixed UTF-8 entries, count implied by the outer
  attribute size). Each type's exact on-disk layout (field order, byte
  widths, BCD packing, keycode field ranges) was derived empirically
  from the opaque `exrheader` validator's text rendering. New
  `tests/optional_attribute_roundtrip.rs` adds algebraic round-trips,
  malformed-size rejection, a full-header round-trip through
  `parse_exr`, and an `exrheader` interop check (auto-skipped when the
  binary is absent). 9 new tests.

- Round-265 **typed `tiledesc` attribute inspector**. New
  `AttributeValue::TileDesc(TileDesc)` variant routes the 9-byte `tiles`
  attribute payload through the existing `crate::tiled::TileDesc` struct
  (`x_size`, `y_size` as `u32`; `level_mode`, `round_mode` as 4-bit
  nibbles). `parse_attribute_value` now decodes the `"tiledesc"` type
  name into the typed variant instead of falling through to
  `AttributeValue::Other { type_name: "tiledesc", .. }`. New
  `TileDesc::to_bytes() -> [u8; 9]` returns the on-disk packing (two LE
  `u32` then the packed mode byte = `(round_mode << 4) | level_mode`).
  `encode_attribute_value` emits the same 9 bytes under type-name
  `"tiledesc"`. The four call sites in `src/deep.rs` that previously
  inspected `Other { type_name: "tiledesc", data }` inline (the
  `parse_exr_deep_tiled` ONE_LEVEL routing, the
  `parse_exr_deep_tiled_mipmap` MIPMAP routing, the
  `parse_exr_deep_tiled_ripmap` RIPMAP routing, and their three
  multi-part siblings) all route through a new
  `crate::tiled::tiledesc_raw_from_attribute` helper that accepts BOTH
  the typed variant AND the legacy `Other` shape, so every encoder site
  that still writes `Other { type_name: "tiledesc", .. }` (round-40 +
  round-78 + round-124 + round-130 + round-181 + round-192 + round-196 +
  round-202 + round-208 + round-214 + round-220 + round-227 + round-232)
  keeps working unchanged. `tiledesc_from_attribute` similarly accepts
  both shapes. New test file `tests/tiledesc_attribute_roundtrip.rs`
  (11 tests) covers: (a) algebraic round-trip across every permitted
  `level_mode` ∈ {0,1,2} × `round_mode` ∈ {0,1} pair plus extreme
  `x_size`/`y_size` cases (1×1 minimum, large power of two, asymmetric
  non-power-of-two); (b) on-disk byte-layout pin (two LE `u32` followed
  by the packed mode byte 0x11 for MIPMAP+ROUND_UP); (c) error paths
  for short and oversize payloads; (d) full-file round-trip — generate
  a real tiled file via `encode_exr_tiled_rgba_float_with` and confirm
  the `tiles` attribute surfaces as the typed
  `AttributeValue::TileDesc(_)` after `parse_header`, repeated for
  NONE/ZIP/ZIPS compression to confirm the variant is independent of
  payload encoding; and (e) `exrheader` interop — opaque-process
  invocation on a generated tiled file asserts zero exit and the
  presence of the `tiles` attribute name in the emitted text
  (auto-skipped when `exrheader` is missing from `$PATH`). Three
  additional unit tests in `src/tiled.rs` cover the new `to_bytes`
  helper. Test count: 322 (+14).

- Round-247 **typed `box2f` attribute inspector**. New `AttributeValue::Box2f`
  variant + public `Box2f` struct (`x_min`, `y_min`, `x_max`, `y_max` as
  `f32`). `parse_attribute_value` accepts the `"box2f"` type-name with a
  16-byte payload (four little-endian `f32` packed in declaration order —
  identical field shape to `box2i` with `i32` swapped for `f32`); short
  or oversize payloads error. `encode_attribute_value` emits the same
  16-byte layout under type-name `"box2f"`. New test file
  `tests/box2f_attribute_roundtrip.rs` (11 tests) covers:
  (a) algebraic round-trip including signed extremes (`f32::{MIN, MAX,
  MIN_POSITIVE}`), `INFINITY` / `NEG_INFINITY`, `-0.0`, sub-normal
  patterns, and NaN bit-pattern preservation;
  (b) payload-size + LE byte-order pin against a hand-built
  `1.0 / 2.0 / 4.0 / 8.0` expected-bytes vector;
  (c) error paths for short and oversize payloads;
  (d) distinction from `box2i` — same 16-byte width but the two
  type-names parse into and encode out of distinct typed variants;
  (e) full scanline-file round-trip through `encode_exr_scanline` +
  `parse_exr` confirming a `renderRegion: box2f` attribute survives as
  the typed variant (asserts it doesn't fall through to `Other`); and
  (f) `exrheader` interop — the binary is invoked as an opaque process
  on a generated scanline file carrying the `box2f` attribute, asserting
  zero exit and the presence of the attribute name in the emitted text.
  Auto-skipped when `exrheader` is absent from `$PATH`. Test count: 308
  (+11).

- Round-238 **typed attribute inspectors** for nine additional EXR header
  attribute payload types: `int`, `double`, `string`, `v2i`, `v3i`, `v3f`,
  `m33f`, `m44f`, `chromaticities`. New `AttributeValue` variants `Int(i32)`,
  `Double(f64)`, `String(String)`, `V2i(i32, i32)`, `V3i(i32, i32, i32)`,
  `V3f(f32, f32, f32)`, `M33f([f32; 9])`, `M44f([f32; 16])`,
  `Chromaticities(Chromaticities)` (new public struct with CIE-xy `red_*`,
  `green_*`, `blue_*`, `white_*` fields). `parse_attribute_value` decodes
  these from on-disk payloads of the spec-table sizes
  (4 / 8 / 12 / 36 / 64 / 32 bytes + variable-length `string`) and
  `encode_attribute_value` round-trips them back to bit-identical bytes
  with the matching type-name string. The variable-length `string` payload
  uses the same on-disk shape this crate's multi-part writers already
  emit and cross-validate against `exrmetrics` (round 40): raw bytes,
  length carried by the outer attribute size field, no NUL terminator
  inside the payload. Existing call sites
  that consumed `AttributeValue::Other { type_name: "string", .. }` /
  `AttributeValue::Other { type_name: "int", .. }` (the multi-part
  `find_part_type` / `find_chunk_count` helpers and the deep-file
  `find_string_attr` / `find_int_attr` helpers) accept both the typed
  variant and the legacy `Other` shape, so producer code that still emits
  the `Other` form keeps working unchanged. New test file
  `tests/typed_attribute_roundtrip.rs` covers (a) algebraic round-trips
  for every new variant including `i32::{MIN,MAX}`, `f64::{MIN,MAX}`,
  NaN bit-pattern preservation, multi-byte UTF-8 strings, and
  alphabetic/identity matrices; (b) full scanline-file round-trips
  through `encode_exr_scanline` + `parse_exr` confirming the typed
  variant survives a full encode-then-parse with the bit-exact value;
  and (c) `exrheader` interop — the binary is invoked as an opaque
  process (input bytes in, stdout text out) on a generated scanline
  file carrying every new attribute type, and the test asserts a
  zero exit status plus the presence of every attribute name in the
  emitted text. Test count: 297 (+12).

- Round-232 multi-part **mixed** flat scanline + flat tiled WRITE + READ.
  New public API: `encode_exr_multipart_mixed`,
  `parse_exr_multipart_mixed`, `MultipartMixedPart`, `MultipartMixedImage`.
  A single multi-part file may now freely combine `type="scanlineimage"`
  and `type="tiledimage"` (ONE_LEVEL + ROUND_DOWN) parts in arbitrary
  order. Composes the round-40 multi-part scanline chunk shape
  (`i32 part_number, i32 Y, i32 size, payload`) with the round-192
  multi-part flat-tiled chunk shape (`i32 part_number, i32 tx, i32 ty,
  i32 lvlx, i32 lvly, i32 size, payload`) inside one file; the reader
  walks the offset tables linearly and dispatches the chunk-body shape
  via each part's declared `type` attribute. Version-field bits stay at
  `0x1000` (`multipart` only; `non_image` is for deep parts, `single_tile`
  is never set on multi-part files — the per-part `type` + `tiles`
  attribute carry the tile-ness signal). Per-part `chunkCount` matches
  the homogeneous writers: scanline = `ceil(height / scanlines_per_block)`,
  tiled = `tx_count * ty_count`. Per-tile payload layout is identical to
  the round-192 single-part / multi-part tiled writers (row-major within
  the tile, channels in alphabetical order, edge tiles store only the
  valid pixel rectangle). Sub-sampled channels are accepted on scanline
  parts but rejected on tiled parts (the chunk-body assumes 1×1 sampling,
  matching round-40 and round-192). Compression: NONE / ZIP / ZIPS / RLE
  per part. 11 new self-roundtrip tests cover 2-part `Scanline` +
  `Tiled` (both orderings) under ZIPS and NONE, 3-part RLE + ZIP + ZIP
  mixed-compression layouts, 13×9 non-power-of-two edge-tile cases
  paired with a matching scanline part, layouts where the two parts
  carry distinct dimensions (16×16 scanline + 24×16 tiled), and a
  UINT + HALF scanline part alongside a FLOAT tiled part. Reject paths
  cover empty parts, duplicate names, empty names, zero tile sizes,
  and sub-sampled channels on tiled parts. Multi-level tiled parts in a
  mixed file remain a followup (pure multi-level multi-part files keep
  using `parse_exr_multipart_tiled_multilevel`); mixed files with deep
  parts are a separate followup since the deep chunk shape carries three
  `u64` sizes plus a per-tile offset table rather than a single `i32`
  size + payload.

- Round-227 multi-part deep tiled **RIPMAP_LEVELS** WRITE + READ. New
  public API: `encode_exr_multipart_deep_tiled_ripmap`,
  `parse_exr_multipart_deep_tiled_ripmap`,
  `MultipartDeepRipmapTiledPart`, `DeepRipmapTiledPart`. Composes the
  round-181 multi-part deep-tiled chunk shape (`i32 part_number` prefix
  + `tx, ty, lvlx, lvly` + 3 u64 sizes + per-tile cumulative-inclusive
  offset table + non-interleaved channel-major sample data) with the
  round-214 single-part deep-tiled RIPMAP iteration order: per part,
  chunks walk the `(nx × ny)` grid `lvly`-outer / `lvlx`-inner, and
  within each cell INCREASING_Y row-major (ty outer, tx inner). The
  chunk header carries the explicit `(lvlx, lvly)` pair (axes
  independent per RIPMAP). Each part carries its own grid with possibly
  distinct level-(0,0) dimensions; per-part `chunkCount` is the sum
  over `nx * ny` cells of `ceil(cell_w / tile_x) *
  ceil(cell_h / tile_y)`. Version-field bits are `0x800 | 0x1000`
  (`non_image | multipart`); `single_tile` (0x200) MUST NOT be set —
  the `tiles[tiledesc, mode=0x02]` attribute + `type="deeptile"` carry
  the RIPMAP-deep signal. Compression: NONE / RLE / ZIPS per part (deep
  ZIP rejected to match the reference `exrinfo` validator and the
  round-130 / 181 / 208 / 214 / 220 deep-tiled writers). Concatenated
  per-part offset tables in part-order followed by chunk records in
  part-order. The reader uses a linear chunk scan (matching the
  multi-part deep-tiled ONE_LEVEL / MIPMAP readers) for robustness
  against zero-filled offset tables produced by `exrmultipart -combine`.
  Per-tile pixel-offset table holds `tile_h * tile_w` cumulative-
  inclusive i32 entries row-major within the tile's valid pixel
  rectangle (edge tiles trim to their valid extent); for NONE
  compression the reader also tolerates files that pad the table to
  the full `tile_x * tile_y * 4` bytes (matching the round-130 /
  round-208 / round-214 / round-220 deep-tiled discipline). 15 new
  self-roundtrip tests cover 2-part ZIPS, 3-part mixed NONE/RLE/ZIPS,
  13×9 non-power-of-two edge-tile RLE+ZIPS, and a layout where the two
  parts carry distinct level-(0,0) dimensions (16×16 + 24×16 with
  different tile sizes per part). Reject paths cover empty parts,
  duplicate names, ZIP compression, sub-sampled channels, and pyramid-
  row / column-length mismatches. The existing single-part deep
  RIPMAP entry (`parse_exr_deep_tiled_ripmap`), the ONE_LEVEL
  multi-part deep entry (`parse_exr_multipart_deep_tiled`), and the
  MIPMAP multi-part deep entry (`parse_exr_multipart_deep_tiled_mipmap`)
  all redirect their respective miscategorised files to the new entry
  instead of returning a generic unsupported / followup message.
  ROUND_DOWN only. This closes the deep-tiled matrix: every combination
  of {single-part, multi-part} × {ONE_LEVEL, MIPMAP_LEVELS,
  RIPMAP_LEVELS} now has dedicated WRITE + READ entries.

- Round-220 multi-part deep tiled **MIPMAP_LEVELS** WRITE + READ. New
  public API: `encode_exr_multipart_deep_tiled_mipmap`,
  `parse_exr_multipart_deep_tiled_mipmap`,
  `MultipartDeepMipmapTiledPart`, `DeepMipmapTiledPart`. Composes the
  round-181 multi-part deep-tiled chunk shape (`i32 part_number` prefix
  + `tx, ty, lvlx, lvly` + 3 u64 sizes + per-tile cumulative-inclusive
  offset table + non-interleaved channel-major sample data) with the
  round-208 single-part deep-tiled MIPMAP iteration order: per part,
  the chunks walk levels 0..N-1 ascending and within each level
  INCREASING_Y row-major (ty outer, tx inner) with `lvlx == lvly ==
  level` for the MIPMAP diagonal. Each part carries its own pyramid
  with possibly distinct level-0 dimensions; `chunkCount` is the
  per-part sum over levels of `ceil(level_w / tile_x) *
  ceil(level_h / tile_y)`. Version-field bits are `0x800 | 0x1000`
  (`non_image | multipart`); `single_tile` (0x200) MUST NOT be set —
  the `tiles[tiledesc, mode=0x01]` attribute + `type="deeptile"` carry
  the MIPMAP-deep signal. Compression: NONE / RLE / ZIPS per part
  (deep ZIP rejected to match the reference `exrinfo` validator and
  the round-130 / 181 / 208 deep-tiled writers). Concatenated offset
  tables in part-order followed by chunk records in part-order. The
  reader uses a linear chunk scan (matching the multi-part deep-tiled
  ONE_LEVEL reader) for robustness against zero-filled offset tables
  produced by `exrmultipart -combine`. Per-tile pixel-offset table
  holds `tile_h * tile_w` cumulative-inclusive i32 entries row-major
  within the tile's valid pixel rectangle (edge tiles trim to their
  valid extent); for NONE compression the reader also tolerates files
  that pad the table to the full `tile_x * tile_y * 4` bytes (matching
  the round-130 single-part deep-tiled discipline). 11 new self-
  roundtrip tests cover 2-part ZIPS, 3-part mixed NONE/RLE/ZIPS,
  13×9 non-power-of-two edge-tile RLE+ZIPS, and a layout where the two
  parts carry distinct level-0 dimensions (16×16 + 24×16 with different
  tile sizes per part). Reject paths cover empty parts, duplicate
  names, ZIP compression, and pyramid-length mismatches. The
  single-part deep MIPMAP entry (`parse_exr_deep_tiled_mipmap`) and
  the ONE_LEVEL multi-part deep entry (`parse_exr_multipart_deep_tiled`)
  both now redirect their respective miscategorised files to the new
  entry instead of returning the generic unsupported-multipart /
  unsupported-MIPMAP message. ROUND_DOWN only; multi-part deep tiled
  RIPMAP_LEVELS is the only remaining followup on the deep-tiled
  matrix.
- Round-214 single-part deep tiled **RIPMAP_LEVELS** WRITE + READ. New
  public API: `encode_exr_deep_tiled_ripmap`, `parse_exr_deep_tiled_ripmap`,
  `DeepRipmapTiledInput`, `DeepRipmapTiledLevelInput`,
  `DeepRipmapTiledImage`, `DeepTiledRipmapCell`. Composes the round-130
  single-part deep-tiled chunk shape (`tx, ty, lvlx, lvly` + 3 u64 sizes +
  per-tile cumulative-inclusive offset table + non-interleaved
  channel-major sample data) with the round-124 single-part flat RIPMAP
  iteration order: the offset table walks the grid `lvly`-outer
  `lvlx`-inner across `(nx × ny)` cells and within each cell
  INCREASING_Y row-major (ty outer, tx inner). Cell `(lvlx, lvly)` has
  dimensions `(mipmap_level_dim(w, lvlx), mipmap_level_dim(h, lvly))`
  and the chunk header carries the explicit `(lvlx, lvly)` pair (the two
  axes are independent for RIPMAP, unlike the MIPMAP diagonal).
  Version-field convention follows the round-130 / round-208 single-part
  deep-tiled discipline: only the `non_image` (0x800) bit is set (the
  `tiles[tiledesc, mode=0x02]` attribute + `type="deeptile"` string
  attribute carry the 2-D-reduction-grid signal; `single_tile` 0x200
  MUST NOT be set). Per-tile pixel-offset table holds `tile_h * tile_w`
  cumulative-inclusive i32 entries row-major within the tile's valid
  pixel rectangle. Edge tiles trim to their valid extent in both encoder
  and decoder; for NONE compression the reader also accepts the
  reference encoder's `tile_x * tile_y * 4` padded-table convention.
  Per-file `chunkCount` = sum over the `nx * ny` cells of
  `ceil(cell_w / tile_x) * ceil(cell_h / tile_y)`. Compression NONE /
  RLE / ZIPS (deep ZIP rejected to match the round-130 / round-208
  single-part deep-tiled discipline and the reference `exrinfo`
  validator). ROUND_DOWN only. The round-130 `parse_exr_deep_tiled`
  reader now redirects RIPMAP files to the new entry alongside its
  MIPMAP redirect, and `parse_exr_deep_tiled_mipmap` rejects RIPMAP
  files with a pointer to the new entry alongside its existing
  ONE_LEVEL guard. 16 new unit tests in `src/deep.rs` cover NONE 16×16
  in 8×8 / ZIPS 24×16 in 8×4 (edge tiles in many cells) / RLE 13×9 in
  4×4 (multi-axis edge tiles) / version-field-bit invariants /
  rejection of ZIP, wrong y-level / x-level grid shapes, sub-sampled
  channels, zero tile size, empty grid / the four
  `parse_exr_deep_tiled[_mipmap|_ripmap]` cross-redirect pointers /
  a non-power-of-two 16×12 grid. 2 new integration tests in
  `tests/deep_validation.rs`: `exrheader` accepts our deep-tiled RIPMAP
  16×16-in-8×8 file with `deeptile` + rip-map level-mode output, and a
  pure-Rust 24×16-in-8×4 ZIPS full-grid round-trip exercising the
  public-API import path.

- Round-208 single-part deep tiled **MIPMAP_LEVELS** WRITE + READ. New
  public API: `encode_exr_deep_tiled_mipmap`, `parse_exr_deep_tiled_mipmap`,
  `DeepMipmapTiledInput`, `DeepMipmapTiledLevelInput`,
  `DeepMipmapTiledImage`, `DeepTiledMipmapLevel`. Composes the round-130
  single-part deep-tiled chunk shape (`tx, ty, lvlx, lvly` + 3 u64 sizes +
  per-tile cumulative-inclusive offset table + non-interleaved
  channel-major sample data) with the round-78 single-part flat
  MIPMAP_LEVELS iteration order: the offset table walks levels `0..N-1`
  ascending and within each level INCREASING_Y row-major (ty outer, tx
  inner); the chunk header carries `lvlx == lvly == level` (the MIPMAP
  diagonal). Version-field convention mirrors the round-130 single-part
  deep-tiled writer: only the `non_image` (0x800) bit is set (the
  `tiles[tiledesc, mode=0x01]` attribute + `type="deeptile"` string
  attribute carry the multi-level deep-tile signal; `single_tile` 0x200
  MUST NOT be set). Per-tile pixel-offset table holds `tile_h * tile_w`
  cumulative-inclusive i32 entries row-major within the tile's valid
  pixel rectangle. Edge tiles trim to their valid extent in both encoder
  and decoder; for NONE compression the reader also accepts the
  reference encoder's `tile_x * tile_y * 4` padded-table convention.
  Per-file `chunkCount` = sum over levels of
  `ceil(level_w / tile_x) * ceil(level_h / tile_y)`. Compression NONE /
  RLE / ZIPS (deep ZIP rejected to match the round-130 single-part
  deep-tiled discipline and the reference `exrinfo` validator).
  ROUND_DOWN only. The round-130 `parse_exr_deep_tiled` reader now
  redirects MIPMAP files to the new entry rather than rejecting them
  outright (`(mode & 0x0F) == 0x01` surfaces a pointer message). 10 new
  unit tests in `src/deep.rs` cover NONE 16×16 in 8×8 / ZIPS 24×16 in
  8×4 (edge tiles at every level) / RLE 13×9 in 4×4 (multi-axis edge
  tiles) / version-field-bit invariants (non_image MUST be set;
  multipart, single_tile MUST NOT be set) / rejection of ZIP, wrong
  pyramid length, sub-sampled channels, zero tile size, empty pyramid /
  the `parse_exr_deep_tiled` → `parse_exr_deep_tiled_mipmap` redirect
  pointer / the symmetric ONE_LEVEL-rejection guard in
  `parse_exr_deep_tiled_mipmap` / a non-power-of-two 16×12 pyramid.
  2 new integration tests in `tests/deep_validation.rs`: `exrheader`
  accepts our deep-tiled MIPMAP 16×16 in 8×8 file with `deeptile` +
  mip-map level-mode output, and a pure-Rust 24×16-in-8×4 ZIPS full
  pyramid round-trip exercising the public-API import path.

- Round-202 multi-part flat (non-deep) **RIPMAP_LEVELS** TILED WRITE +
  READ. New public API: `encode_exr_multipart_tiled_ripmap`,
  `MultipartRipmapTiledPart`. Composes the round-124 single-part
  RIPMAP_LEVELS encoder with the round-192 multi-part flat-tiled
  envelope: per-part `tiles[tiledesc, level_mode=2]` +
  `type="tiledimage"` carry the 2-D-reduction-grid signal; only the
  multipart (0x1000) version-field bit is set (the `single_tile` 0x200
  bit is NOT set, mirroring the round-192 multi-part flat-tiled and
  round-196 MIPMAP multi-part writers). Each chunk on disk is `i32
  part_number, i32 tx, i32 ty, i32 lvlx, i32 lvly, i32 size,
  payload[size]` (24 bytes of chunk header). RIPMAP convention: `lvlx`
  and `lvly` are independent; iteration walks `lvly`-outer
  `lvlx`-inner across the grid, then ty-outer tx-inner (INCREASING_Y
  row-major) within each `(lvlx, lvly)` cell — matching the single-part
  RIPMAP writer's order and the decoder's `compute_total_tiles` RIPMAP
  branch. Per-part `chunkCount` = sum over the `nx * ny` cells of
  `ceil(cell_w / tile_x) * ceil(cell_h / tile_y)`. Compression NONE /
  ZIP / ZIPS / RLE per part. ROUND_DOWN only. The companion reader is
  the existing `parse_exr_multipart_tiled_multilevel`, whose former
  `level_mode == 2` rejection is gone and which now enumerates the full
  2-D RIPMAP grid alongside ONE_LEVEL and MIPMAP_LEVELS parts using the
  same linear chunk-scan strategy. `parse_exr_multipart_tiled` also now
  redirects RIPMAP multi-part files to
  `parse_exr_multipart_tiled_multilevel` alongside the existing MIPMAP
  redirect. 10 new unit tests in `src/multipart_ripmap_encoder.rs` cover
  2-part NONE / 3-part mixed-compression / 13×9-tile-4 edge tiles ZIPS /
  version-field-bit invariants (multipart=0x1000, single_tile MUST NOT
  be set, non_image MUST NOT be set) / rejection of empty parts list /
  duplicate names / bad grid shape / unsupported compression (PIZ) /
  sub-sampled channels / ONE_LEVEL-reader redirect. 3 new integration
  tests in `tests/multipart_ripmap_validation.rs`: `exrheader` accepts
  the file with the expected "rip-map" + "tiledimage" + per-part names
  dump; `exrmultipart -separate` splits each part into a single-part
  RIPMAP file that our `parse_exr_tiled_multilevel` decodes back to the
  source grid sample-for-sample; pure-Rust 3-part mixed-compression
  self-roundtrip on `(24×16 ZIP, 16×16 ZIPS, 13×9 RLE)` exercising the
  public-API import path.

- Round-196 multi-part flat (non-deep) **multi-level** TILED WRITE + READ.
  New public API: `encode_exr_multipart_tiled_mipmap`,
  `parse_exr_multipart_tiled_multilevel`, `MultipartMipmapTiledPart`,
  `MultilevelTiledPart`. Composes the round-78 single-part MIPMAP_LEVELS
  encoder with the round-192 multi-part flat-tiled envelope: per-part
  `tiles[tiledesc, level_mode=1]` + `type="tiledimage"` carry the
  multi-level tile-ness signal; only the multipart (0x1000) version-field
  bit is set (the `single_tile` 0x200 bit is NOT set, mirroring the
  multi-part deep-tiled discipline). Each chunk on disk is
  `i32 part_number, i32 tx, i32 ty, i32 lvlx, i32 lvly, i32 size,
  payload[size]` (24 bytes of chunk header + the compressed/raw
  payload). MIPMAP convention `lvlx == lvly == level` (the diagonal of
  the `(lvlx, lvly)` grid). Per-part `chunkCount` = sum over levels of
  `ceil(level_w / tile_x) * ceil(level_h / tile_y)`. The reader uses
  the same linear-scan strategy as `parse_exr_multipart_tiled` /
  `parse_exr_multipart_deep_tiled` to remain robust against zero-filled
  offset tables. Compression NONE / ZIP / ZIPS / RLE per part.
  ROUND_DOWN only. 8 new unit tests in `src/multipart_mipmap_encoder.rs`
  cover 2-part NONE / 3-part mixed-compression / 13×9-tile-4 edge tiles
  ZIPS / version-field-bit invariants (multipart=0x1000, single_tile MUST
  NOT be set, non_image MUST NOT be set) / rejection of empty parts list
  / duplicate names / wrong pyramid length / unsupported compression
  (PIZ) / sub-sampled channels. 3 new integration tests in
  `tests/multipart_mipmap_validation.rs`: `exrheader` accepts the file
  with the expected "mip-map" + "tiledimage" + per-part names dump;
  `exrmultipart -separate` splits each part into a single-part MIPMAP
  file that our `parse_exr_tiled_multilevel` decodes back to the source
  pyramid sample-for-sample; pure-Rust 3-part mixed-compression
  self-roundtrip on `(24×16 ZIP, 16×16 ZIPS, 13×9 RLE)` exercising the
  public-API import path. `parse_exr_multipart_tiled` now points
  multi-level multi-part tiled files at the new entry rather than
  rejecting them outright (both the header-level `level_mode != 0`
  branch and the chunk-level `lvlx != 0 || lvly != 0` branch surface
  pointer messages).


## [0.0.3](https://github.com/OxideAV/oxideav-openexr/compare/v0.0.2...v0.0.3) - 2026-05-30

### Other

- round 192: multi-part flat tiled WRITE + READ (type="tiledimage")
- scrub openexr.com citations to neutral phrasing (Task #1240)
- point parse_exr_deep_tiled multipart-rejection at the new entry
- round 181: multi-part deep TILED WRITE + READ (type="deeptile")
- round 174: full-pyramid READ for tiled MIPMAP_LEVELS / RIPMAP_LEVELS
- round 130: single-part deep tiled WRITE + READ (type="deeptile")
- round 127: multi-part deep scanline WRITE
- round 124: RIPMAP_LEVELS tiled-output encoder
- round 92: multi-part deep scanline READ
- round 78: MIPMAP_LEVELS tiled-output encoder
- round 73: sub-sampled channel encoder + deep scanline read/write scaffold

### Added

- Round-192 multi-part flat (non-deep) TILED WRITE + READ. New public
  API: `encode_exr_multipart_tiled`, `parse_exr_multipart_tiled`,
  `MultipartTiledPart`. Each part is `type="tiledimage"` ONE_LEVEL +
  ROUND_DOWN tiled with NONE/ZIP/ZIPS/RLE compression. File envelope
  mirrors the round-40 multi-part scanline writer (version-field bit
  0x1000 only — the `single_tile` 0x200 bit is NOT set; per-part
  `tiles[tiledesc]` + `type="tiledimage"` carry the tile-ness signal,
  matching the round-181 multi-part deep-tiled discipline). On disk
  each tile chunk is `i32 part_number, i32 tx, i32 ty, i32 lvlx,
  i32 lvly, i32 size, payload[size]` (24 bytes of chunk header + the
  compressed/raw payload). Per-tile payload layout — row-major within
  the tile, channels in alphabetical order, edge tiles trimmed to
  their valid pixel rectangle — is byte-identical to the single-part
  tiled writer, so each split-out part is a normal flat tiled file
  the existing `parse_exr` reader handles unchanged. Reader uses a
  linear chunk scan (same robustness pattern as `parse_exr_multipart`
  and `parse_exr_multipart_deep_tiled`) so zero-filled offset tables
  still decode correctly. Per-part attribute set: standard required
  attributes + the mandatory `name` + `tiles[tiledesc]` (ONE_LEVEL +
  ROUND_DOWN) + `type[string="tiledimage"]` + `chunkCount`.
  `parse_exr_multipart` now points `tiledimage` parts at the new
  entry rather than mis-parsing them as scanline chunks. 7 new
  unit tests in `src/multipart_tiled_encoder.rs` cover 2-part NONE
  + 3-part NONE/ZIPS/RLE mixed-compression round-trip, 13×9-in-4×3
  ZIP edge-tile round-trip, rejection of empty parts list / duplicate
  names / sub-sampled channels, and the routing assertion that
  `parse_exr_multipart` redirects `tiledimage` parts to
  `parse_exr_multipart_tiled`. 3 new integration tests in
  `tests/multipart_tiled_validation.rs`: `exrheader` accepts the
  multi-part tiled file (validates `type="tiledimage"` + part-name
  output), `exrmultipart -separate` splits it into per-part tiled
  files that round-trip pixel-exact through `parse_exr`, and a pure
  self-roundtrip on a 3-part mixed-compression 24×16-in-8×8 layout
  exercising the public-API import path.

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
