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
//! Round-2 followups: PIZ / RLE / B44 / B44A / DWAA / DWAB / Pxr24
//! compression; tiled format; multi-part files; deep-data scanlines;
//! UINT pixel type; sub-sampled channels; HDR pixel-format integration
//! with `oxideav-core`.

pub mod decoder;
pub mod encoder;
pub mod error;
pub mod half;
pub mod header;
pub mod image;
#[cfg(feature = "registry")]
pub mod registry;
pub mod types;

/// Codec id for OpenEXR image frames.
pub const CODEC_ID_STR: &str = "openexr";

pub use decoder::parse_exr;
pub use encoder::{
    encode_exr_scanline, encode_exr_scanline_rgba_float, encode_exr_scanline_rgba_float_with,
};
pub use error::{ExrError, Result};
pub use header::{encode_header, parse_header, ParsedHeader, VersionField};
pub use image::{ExrImage, ExrPlane};
pub use types::{
    Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType, EXR_MAGIC,
};

#[cfg(feature = "registry")]
pub use registry::{register, register_codecs, register_containers};
