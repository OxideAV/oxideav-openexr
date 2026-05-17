//! Pure-Rust OpenEXR reader/writer (clean-room from the openexr.com file
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
//!   per the openexr.com spec, matching the decoder's existing sub-
//!   sampled scatter layout. The earlier "round-3 followup" guard is
//!   gone; round-trip + `exrmetrics --convert` cross-checked.
//! * Deep scanline READ + WRITE scaffold (single-part) —
//!   [`parse_exr_deep_scanline`] / [`encode_exr_deep_scanline`] +
//!   [`DeepExrImage`] / [`DeepScanlineInput`]. NONE / RLE / ZIPS only
//!   (the spec page lists ZIP too but `exrinfo` rejects deep ZIP with
//!   EXR_ERR_INVALID_ATTR, so we follow the reference). Pixel offset
//!   table + non-interleaved sample data layout per the openexr.com
//!   File Layout page; cross-validated against `exrheader`, `exrinfo`,
//!   and `exrmetrics --convert -z none`. Multi-part deep + deep-tiled
//!   are followups.
//!
//! Round-4+ followups still open: PIZ / B44 / B44A / DWAA / DWAB / Pxr24
//! compression (PIZ blocked on a clean-room wavelet+Huffman trace doc;
//! B44 / Pxr24 documented at high-level only, byte layout not in the
//! public spec); multi-resolution tiled-output writes (`MIPMAP_LEVELS` /
//! `RIPMAP_LEVELS`); tiled or deep parts inside multi-part files;
//! deep-tile data; HDR pixel-format integration with `oxideav-core`.

pub mod decoder;
pub mod deep;
pub mod encoder;
pub mod error;
pub mod half;
pub mod header;
pub mod image;
pub mod multipart_encoder;
#[cfg(feature = "registry")]
pub mod registry;
pub mod rle;
pub mod tile_encoder;
pub mod tiled;
pub mod types;

/// Codec id for OpenEXR image frames.
pub const CODEC_ID_STR: &str = "openexr";

pub use decoder::{mipmap_level_count, mipmap_level_dim, parse_exr, parse_exr_multipart};
pub use deep::{
    encode_exr_deep_scanline, parse_exr_deep_scanline, DeepExrImage, DeepScanlineInput,
};
pub use encoder::{
    encode_exr_scanline, encode_exr_scanline_rgba_float, encode_exr_scanline_rgba_float_with,
};
pub use error::{ExrError, Result};
pub use header::{
    encode_header, parse_header, parse_multipart_headers, ParsedHeader, VersionField,
};
pub use image::{ExrImage, ExrPlane};
pub use multipart_encoder::{
    encode_exr_multipart, encode_exr_multipart_rgba_float_with, MultipartScanlinePart,
};
pub use tile_encoder::{encode_exr_tiled, encode_exr_tiled_rgba_float_with};
pub use types::{
    Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType, EXR_MAGIC,
};

#[cfg(feature = "registry")]
pub use registry::{__oxideav_entry, register, register_codecs, register_containers};
