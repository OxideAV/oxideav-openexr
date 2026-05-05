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
//! Round-3 followups: PIZ / B44 / B44A / DWAA / DWAB / Pxr24
//! compression; multi-part files; deep-data scanlines; HDR
//! pixel-format integration with `oxideav-core`.

pub mod decoder;
pub mod encoder;
pub mod error;
pub mod half;
pub mod header;
pub mod image;
#[cfg(feature = "registry")]
pub mod registry;
pub mod rle;
pub mod tiled;
pub mod types;

/// Codec id for OpenEXR image frames.
pub const CODEC_ID_STR: &str = "openexr";

pub use decoder::{mipmap_level_count, mipmap_level_dim, parse_exr, parse_exr_multipart};
pub use encoder::{
    encode_exr_scanline, encode_exr_scanline_rgba_float, encode_exr_scanline_rgba_float_with,
};
pub use error::{ExrError, Result};
pub use header::{
    encode_header, parse_header, parse_multipart_headers, ParsedHeader, VersionField,
};
pub use image::{ExrImage, ExrPlane};
pub use types::{
    Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType, EXR_MAGIC,
};

#[cfg(feature = "registry")]
pub use registry::{register, register_codecs, register_containers};
