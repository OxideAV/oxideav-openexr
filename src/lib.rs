//! Pure-Rust OpenEXR reader/writer (clean-room from the OpenEXR file
//! format spec).
//!
//! Round 1 surface — single-part scanline files only:
//!
//! * Magic + version field (no flag bits supported on read other than
//!   plain format-version 2).
//! * Attribute table parser with typed values for the eight required
//!   attributes (`channels`, `compression`, `dataWindow`,
//!   `displayWindow`, `lineOrder`, `pixelAspectRatio`,
//!   `screenWindowCenter`, `screenWindowWidth`); other attributes are
//!   preserved verbatim as `AttributeValue::Other`.
//! * Channel-list (`chlist`) parser/encoder for `HALF` and `FLOAT`
//!   pixel types (UINT is a round-2 followup).
//! * Compression: `NO_COMPRESSION` and `ZIP` (16 scanlines per block,
//!   zlib via `flate2` + the spec's interleave + predictor transforms).
//! * IEEE 754-2008 binary16 (`half`) <-> `f32` codec.
//! * Public functions:
//!     - [`parse_exr`] — bytes -> [`ExrImage`]
//!     - [`encode_exr_scanline_rgba_float`] — RGBA `f32` interleaved
//!       slice -> bytes (ZIP-compressed by default)
//!     - [`encode_exr_scanline_rgba_float_with`] — same with explicit
//!       `Compression`
//!     - [`encode_exr_scanline`] — general per-channel encoder
//!
//! Standalone vs registry-integrated:
//!
//! The crate's default `registry` Cargo feature pulls in `oxideav-core`
//! and exposes the framework `Decoder` / `Encoder` trait surface plus a
//! [`registry::register`] entry point. Disable the feature
//! (`default-features = false`) for an `oxideav-core`-free build that
//! still exposes the standalone [`parse_exr`] /
//! [`encode_exr_scanline_rgba_float`] API plus crate-local [`ExrImage`]
//! / [`ExrError`] / [`ExrPlane`] types.
//!
//! Round-2 surface (this crate, this round):
//! * Compression: NONE, ZIP, ZIPS (per-scanline ZIP), RLE — full
//!   round-trip (encode + decode).
//! * UINT pixel type alongside HALF/FLOAT — read + write.
//! * Sub-sampled channels (`xSampling != 1` or `ySampling != 1`) —
//!   read; the encoder still requires 1×1 sampling.
//! * Tiled scanline files — read-only support for `single_tile` files
//!   in `ONE_LEVEL` mode; multi-resolution mip/rip-map levels are
//!   deferred to round 3.
//!
//! Round-40 surface (this crate, this round):
//! * Tiled-output encoder ([`encode_exr_tiled_rgba_float_with`] /
//!   [`encode_exr_tiled`]) — single-part `ONE_LEVEL` tiled files with
//!   NONE / ZIP / ZIPS / RLE compression. Validated against
//!   `exrmetrics --convert -z none` (the OpenEXR reference impl
//!   re-decodes our tile chunks bit-exactly).
//! * Multi-part output encoder ([`encode_exr_multipart`] /
//!   [`encode_exr_multipart_rgba_float_with`]) — scanline parts with
//!   `name` + `type=scanlineimage` + `chunkCount` per-part. Validated
//!   against `exrmultipart -separate`.
//!
//! Round-73 surface (this crate, this round):
//! * Sub-sampled channel ENCODE — [`encode_exr_scanline`] (and the
//!   multipart variant) now honour `xSampling != 1` / `ySampling != 1`
//!   per the OpenEXR spec, matching the decoder's existing sub-
//!   sampled scatter layout. The earlier "round-3 followup" guard is
//!   gone; round-trip + `exrmetrics --convert` cross-checked.
//! * Deep scanline READ + WRITE scaffold (single-part) —
//!   [`parse_exr_deep_scanline`] / [`encode_exr_deep_scanline`] +
//!   [`DeepExrImage`] / [`DeepScanlineInput`]. NONE / RLE / ZIPS only
//!   (the spec page lists ZIP too but `exrinfo` rejects deep ZIP with
//!   EXR_ERR_INVALID_ATTR, so we follow the reference). Pixel offset
//!   table + non-interleaved sample data layout per the OpenEXR
//!   File Layout spec; cross-validated against `exrheader`, `exrinfo`,
//!   and `exrmetrics --convert -z none`. Multi-part deep + deep-tiled
//!   are followups.
//!
//! Round-78 surface (this crate, this round):
//! * `MIPMAP_LEVELS` tiled-output encoder
//!   ([`encode_exr_tiled_rgba_float_mipmap_box_filter`] /
//!   [`encode_exr_tiled_mipmap`]) — single-part tiled files with full
//!   ROUND_DOWN mipmap pyramid (`tiledesc.level_mode = 1`). Tile chunks
//!   emit in spec iteration order (levels 0..N-1, INCREASING_Y row-major
//!   within each level, `lvlx == lvly == level` for the MIPMAP diagonal
//!   convention from the OpenEXR Technical Introduction). NONE / ZIP
//!   / ZIPS / RLE compression. Cross-validated against
//!   `exrmetrics --convert` (which decodes our pyramid back to a
//!   scanline file pixel-exactly at level 0) and `exrheader`.
//!   [`build_box_filter_pyramid`] gives a default ROUND_DOWN 2×2
//!   box-filter pyramid; callers needing custom filtering supply their
//!   own `Vec<MipmapLevel>`.
//!
//! Round-92 surface (this crate, this round):
//! * Multi-part deep scanline READ — [`parse_exr_deep_multipart`] +
//!   [`DeepScanlinePart`]. Walks files with version-field bits 0x1800
//!   (multipart + non_image) set, each part `type = "deepscanline"`,
//!   `name = <partName>`. Chunks are linearly scanned (matching the
//!   robust strategy already used by [`parse_exr_multipart`] for
//!   zero-filled offset tables emitted by `exrmultipart -combine`).
//!   Each chunk record is `i32 part_number, i32 Y, u64 packed_table,
//!   u64 packed_data, u64 unpacked_data, table_bytes, data_bytes`.
//!   Compression NONE / RLE / ZIPS. Cross-validated against
//!   `exrmultipart -combine`-built fixtures composed of two and three
//!   distinct deep parts (different compressions, different pixel
//!   patterns) — see `tests/deep_validation.rs`. Multi-part deep WRITE
//!   remains a followup.
//!
//! Round-124 surface (this crate, this round):
//! * `RIPMAP_LEVELS` tiled-output encoder
//!   ([`encode_exr_tiled_rgba_float_ripmap_box_filter`] /
//!   [`encode_exr_tiled_ripmap`]) — single-part tiled files carrying the
//!   full 2-D reduction grid (`tiledesc.level_mode = 2`). x-levels reduce
//!   width only, y-levels reduce height only, so cell `(lvlx, lvly)` is
//!   `mipmap_level_dim(w, lvlx) × mipmap_level_dim(h, lvly)`. The offset
//!   table / chunk order walks `lvly` outer, `lvlx` inner (matching the
//!   decoder's `compute_total_tiles` RIPMAP branch), INCREASING_Y
//!   row-major within each level. NONE / ZIP / ZIPS / RLE. Cross-validated
//!   against `exrmetrics --convert` + `exrheader`, and our decoder is
//!   pinned against an `exrmaketiled -r` reference file (see
//!   `tests/ripmap_encoder_validation.rs`). [`build_box_filter_ripmap`]
//!   gives a default separable 2× box-filter grid; callers needing custom
//!   filtering supply their own [`RipmapPyramid`].
//!
//! Round-127 surface (this crate, this round):
//! * Multi-part deep scanline WRITE
//!   ([`encode_exr_multipart_deep_scanline`] + [`MultipartDeepScanlinePart`]).
//!   Emits files with version-field bits 0x1800 (multipart + non_image)
//!   set, per-part `type = "deepscanline"` + `name` + `chunkCount` +
//!   `version=1` + `maxSamplesPerPixel`, concatenated per-part offset
//!   tables, then chunks each prefixed with `i32 part_number` followed by
//!   the standard deep chunk body `i32 Y, u64 packed_table, u64
//!   packed_data, u64 unpacked_data, table_bytes, data_bytes`. Self
//!   round-trips through [`parse_exr_deep_multipart`]; cross-validated
//!   against `exrheader` + the `exrmultipart -separate` reference flow.
//!   NONE / RLE / ZIPS compression. Deep-tiled WRITE (`type =
//!   "deeptile"`) still a followup.
//!
//! Round-130 surface (this crate, this round):
//! * Single-part deep TILED WRITE + READ
//!   ([`encode_exr_deep_tiled`] / [`parse_exr_deep_tiled`] +
//!   [`DeepTiledInput`] / [`DeepTiledImage`]). Emits files with the
//!   `non_image` (0x800) version-field bit set ONLY (single-part deep
//!   tiled files do NOT set `single_tile` — the `tiles[tiledesc]`
//!   attribute + `type="deeptile"` string attribute are the
//!   discriminators; `exrheader` rejects files that set both bits).
//!   Headers carry `type="deeptile"`, `version=1`, `maxSamplesPerPixel`,
//!   `tiles` tiledesc (ONE_LEVEL + ROUND_DOWN), `chunkCount = tx_count *
//!   ty_count`. Each tile chunk on disk is `i32 tx, i32 ty, i32 lvlx,
//!   i32 lvly, u64 packed_table, u64 packed_data, u64 unpacked_data,
//!   packed_table_bytes, packed_sample_bytes`. Per-tile offset table
//!   holds `tile_h * tile_w` cumulative i32 entries (per-row within the
//!   tile rectangle). Edge tiles store only their valid pixel area.
//!   Sample data is non-interleaved (channel-major within each tile).
//!   Compression NONE / RLE / ZIPS (deep ZIP rejected, matching the
//!   single-part deep scanline encoder). Self-roundtrip + cross-validated
//!   against `exrheader` (header dump + tiledesc + type=deeptile).
//!   MIPMAP/RIPMAP deep tiled + multi-part deep tiled are followups.
//!
//! Round-174 surface (this crate, this round): full-pyramid READ for
//! tiled `MIPMAP_LEVELS` / `RIPMAP_LEVELS` files via
//! [`parse_exr_tiled_multilevel`] + [`MultilevelTiledImage`] +
//! [`TiledLevel`]. Returns every decoded level (ONE_LEVEL: single entry;
//! MIPMAP: `0..N-1` with `level_x == level_y`; RIPMAP: full 2-D grid in
//! `lvly` outer, `lvlx` inner order). The existing [`parse_exr`] entry
//! point is unchanged. Validated by encoding pyramids through
//! [`encode_exr_tiled_mipmap`] / [`encode_exr_tiled_ripmap`] and
//! confirming every sample of every level matches the input.
//!
//! Round-181 surface (this crate, this round): multi-part deep TILED
//! WRITE + READ ([`encode_exr_multipart_deep_tiled`] /
//! [`parse_exr_multipart_deep_tiled`] + [`MultipartDeepTiledPart`] +
//! [`DeepTiledPart`]). Composes the round-127 multipart deep-scanline
//! envelope (version-field bits 0x1800, concatenated per-part headers +
//! offset tables) with the round-130 single-part deep-tiled chunk shape
//! (`tx, ty, lvlx, lvly, packed_table, packed_data, unpacked_data` +
//! payload), prefixed by `i32 part_number` per chunk. Per-part attrs
//! mirror the single-part `type="deeptile"` writer plus the mandatory
//! `name` attribute. Linear-scan reader for robustness against
//! zero-filled offset tables. ONE_LEVEL + ROUND_DOWN; NONE / RLE / ZIPS
//! compression (deep ZIP rejected, matching the single-part deep-tiled
//! discipline and the reference `exrinfo` validator). Self-roundtrips
//! at every supported compression on multi-part 2- and 3-part layouts.
//!
//! Round-4+ followups still open: PIZ / B44 / B44A / DWAA / DWAB / Pxr24
//! compression (PIZ blocked on a clean-room wavelet+Huffman trace doc;
//! B44 / Pxr24 documented at high-level only, byte layout not in the
//! public spec); multi-part WRITE for tiled parts (flat); multi-level
//! deep tiled (MIPMAP/RIPMAP, single-part and multi-part); HDR
//! pixel-format integration with `oxideav-core`.

pub mod decoder;
pub mod deep;
pub mod encoder;
pub mod error;
pub mod half;
pub mod header;
pub mod image;
pub mod mipmap_encoder;
pub mod multipart_encoder;
#[cfg(feature = "registry")]
pub mod registry;
pub mod rle;
pub mod tile_encoder;
pub mod tiled;
pub mod types;

/// Codec id for OpenEXR image frames.
pub const CODEC_ID_STR: &str = "openexr";

pub use decoder::{
    mipmap_level_count, mipmap_level_dim, parse_exr, parse_exr_multipart,
    parse_exr_tiled_multilevel, MultilevelTiledImage, TiledLevel,
};
pub use deep::{
    encode_exr_deep_scanline, encode_exr_deep_tiled, encode_exr_multipart_deep_scanline,
    encode_exr_multipart_deep_tiled, parse_exr_deep_multipart, parse_exr_deep_scanline,
    parse_exr_deep_tiled, parse_exr_multipart_deep_tiled, DeepExrImage, DeepScanlineInput,
    DeepScanlinePart, DeepTiledImage, DeepTiledInput, DeepTiledPart, MultipartDeepScanlinePart,
    MultipartDeepTiledPart,
};
pub use encoder::{
    encode_exr_scanline, encode_exr_scanline_rgba_float, encode_exr_scanline_rgba_float_with,
};
pub use error::{ExrError, Result};
pub use header::{
    encode_header, parse_header, parse_multipart_headers, ParsedHeader, VersionField,
};
pub use image::{ExrImage, ExrPlane};
pub use mipmap_encoder::{
    build_box_filter_pyramid, build_box_filter_ripmap, encode_exr_tiled_mipmap,
    encode_exr_tiled_rgba_float_mipmap_box_filter, encode_exr_tiled_rgba_float_ripmap_box_filter,
    encode_exr_tiled_ripmap, mipmap_level_count_round_down, ripmap_level_counts_round_down,
    MipmapLevel, RipmapLevel, RipmapPyramid,
};
pub use multipart_encoder::{
    encode_exr_multipart, encode_exr_multipart_rgba_float_with, MultipartScanlinePart,
};
pub use tile_encoder::{encode_exr_tiled, encode_exr_tiled_rgba_float_with};
pub use types::{
    Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType, EXR_MAGIC,
};

#[cfg(feature = "registry")]
pub use registry::{__oxideav_entry, register, register_codecs, register_containers};
