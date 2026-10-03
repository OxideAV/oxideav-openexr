//! Pure-Rust OpenEXR reader and writer, clean-room from the public
//! OpenEXR file-format documentation.
//!
//! The crate root follows the OxideAV image-crate API contract
//! (`IMAGE_CRATE_API`): [`probe`], [`info`], [`decode`] / [`decode_with`]
//! → [`ExrImage`] (one packed little-endian `f32` plane: `RgbF32Le`,
//! `RgbaF32Le` or `GrayF32Le`), [`decode_rgb8`] / [`decode_rgba8`] →
//! [`RgbImage`] / [`RgbaImage`], [`decode_all`] (one [`Frame`] per part),
//! [`decode_from`], [`encode`] / [`encode_rgb8`] / [`encode_rgba8`] /
//! [`encode_to`] / [`encode_all`] with [`EncodeOptions`] and
//! [`DecodeOptions`], and [`ExrError`] (= [`Error`]).
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let bytes = std::fs::read("in.exr")?;
//! if oxideav_openexr::probe(&bytes) {
//!     let info = oxideav_openexr::info(&bytes)?;       // header only
//!     let img = oxideav_openexr::decode(&bytes)?;      // ExrImage, linear f32
//!     let rgba8: Vec<u8> = img.to_rgba8();             // clamp [0, 1] × 255
//!     let floats: Vec<f32> = img.pixels();             // scene-referred samples
//!     assert_eq!((img.width(), img.height()), (info.width, info.height));
//!     let opts = oxideav_openexr::EncodeOptions::default()
//!         .with_compression(oxideav_openexr::Compression::Piz);
//!     std::fs::write("out.exr", oxideav_openexr::encode(&img, &opts)?)?;
//! }
//! # Ok(()) }
//! ```
//!
//! # The colour view and the depth API
//!
//! An OpenEXR part is an arbitrarily named channel set. The contract
//! image is the part's *colour view* — `R G B (A)`, `Y (A)` or the
//! luminance/chroma triple `Y RY BY (A)` reconstructed to RGB (see
//! [`view`](crate::ExrImage) docs on [`ExrImage`]); HALF channels widen
//! to `f32` exactly, FLOAT channels copy bit-for-bit, UINT channels
//! convert (exact below 2^24), and nothing is clamped or tone-mapped on
//! decode. A channel set without such a view, and deep data, are
//! [`ExrError::Unsupported`] at the root; they stay reachable through
//! the depth API, which keeps its names: [`parse_exr`] → [`ExrPart`]
//! (every channel as a named `f32` plane; before the contract this type
//! was called `ExrImage`), [`parse_exr_multipart_mixed`], the
//! `parse_exr_deep_*` readers, [`parse_header`], and the per-channel
//! writers [`encode_exr_scanline`], [`encode_exr_tiled`],
//! [`encode_exr_multipart`], `encode_exr_deep_*` …
//!
//! # Features
//!
//! The default `registry` feature pulls in `oxideav-core` and adds
//! [`register`] / [`register_codecs`] / [`register_containers`], the
//! [`make_decoder`] / [`make_encoder`] factories, and the frame bridge
//! (`From<ExrImage> for VideoFrame`, [`ExrImage::from_video_frame`]).
//! `default-features = false` keeps the whole standalone layer and the
//! depth API with no framework dependency.
//!
//! # Format coverage
//!
//! Scanline, tiled (ONE_LEVEL / MIPMAP / RIPMAP), multi-part and deep
//! files; every compression scheme (NONE, RLE, ZIPS, ZIP, PIZ, PXR24,
//! B44, B44A, DWAA, DWAB) for flat parts; HALF / FLOAT / UINT channels;
//! sub-sampled channels; layered and multi-view channel names;
//! `lineOrder` storage orders. See the README's format-specifics
//! section and the CHANGELOG for the history.

// Internal B44/B44A plumbing (no public items) — hidden from
// rustdoc/semver (fleet rule 2026-09-01).
#[doc(hidden)]
pub mod b44;
// Chunk-level decode entry points for the fuzz harness / profiling —
// hidden from rustdoc/semver.
mod api;
#[doc(hidden)]
pub mod chunk_api;
pub mod decoder;
pub mod deep;
pub(crate) mod dwa;
pub mod encoder;
pub mod error;
pub mod half;
pub mod header;
pub(crate) mod huf;
pub mod image;
pub mod layers;
pub mod luma_chroma;
pub mod mipmap_encoder;
pub mod multipart_encoder;
pub mod multipart_mipmap_encoder;
pub mod multipart_mixed_encoder;
pub mod multipart_ripmap_encoder;
pub mod multipart_tiled_encoder;
pub mod options;
pub mod part;
pub(crate) mod piz;
#[cfg(feature = "registry")]
pub mod registry;
// Internal RLE compression plumbing (like `piz`/`huf`/`dwa`) — hidden
// from rustdoc/semver (fleet rule 2026-09-01).
#[doc(hidden)]
pub mod rle;
pub mod tile_encoder;
pub mod tiled;
pub mod types;
pub(crate) mod view;

/// Codec id for OpenEXR image frames.
pub const CODEC_ID_STR: &str = "openexr";

pub use api::{
    decode, decode_all, decode_all_with, decode_from, decode_rgb8, decode_rgba8, decode_with,
    encode, encode_all, encode_rgb8, encode_rgba8, encode_to, info, probe,
};
pub use decoder::{
    mipmap_level_count, mipmap_level_dim, parse_exr, parse_exr_multipart,
    parse_exr_multipart_tiled, parse_exr_multipart_tiled_multilevel, parse_exr_tiled_multilevel,
    MultilevelTiledImage, MultilevelTiledPart, TiledLevel,
};
pub use deep::{
    encode_exr_deep_scanline, encode_exr_deep_tiled, encode_exr_deep_tiled_mipmap,
    encode_exr_deep_tiled_ripmap, encode_exr_multipart_deep_scanline,
    encode_exr_multipart_deep_tiled, encode_exr_multipart_deep_tiled_mipmap,
    encode_exr_multipart_deep_tiled_ripmap, parse_exr_deep_multipart, parse_exr_deep_scanline,
    parse_exr_deep_tiled, parse_exr_deep_tiled_mipmap, parse_exr_deep_tiled_ripmap,
    parse_exr_multipart_deep_tiled, parse_exr_multipart_deep_tiled_mipmap,
    parse_exr_multipart_deep_tiled_ripmap, DeepExrImage, DeepMipmapTiledImage,
    DeepMipmapTiledInput, DeepMipmapTiledLevelInput, DeepMipmapTiledPart, DeepRipmapTiledImage,
    DeepRipmapTiledInput, DeepRipmapTiledLevelInput, DeepRipmapTiledPart, DeepScanlineInput,
    DeepScanlinePart, DeepTiledImage, DeepTiledInput, DeepTiledMipmapLevel, DeepTiledPart,
    DeepTiledRipmapCell, MultipartDeepMipmapTiledPart, MultipartDeepRipmapTiledPart,
    MultipartDeepScanlinePart, MultipartDeepTiledPart,
};
#[allow(deprecated)]
pub use encoder::{
    encode_exr_scanline, encode_exr_scanline_rgba_float, encode_exr_scanline_rgba_float_with,
    encode_exr_scanline_rgba_float_with_line_order,
};
pub use error::{Error, ExrError, Result};
pub use header::{
    encode_header, parse_header, parse_multipart_headers, ParsedHeader, VersionField,
};
pub use image::{
    ColorInfo, ColorRange, ExrImage, ExrPixelFormat, Frame, ImageInfo, Metadata, PixelFormat,
    Plane, RgbImage, RgbaImage,
};
pub use layers::{
    enumerate_layers, find_layer, multi_view, split_channel_name, ExrLayer, LayerKind,
};
pub use luma_chroma::{
    chromaticities_of, downsample_tent, luma_chroma_to_rgb, luminance_weights,
    luminance_weights_of, rgb_to_luma_chroma, upsample_bilinear, ChromaPlane, LumaChromaPlanes,
    RgbPlanes, BT709_CHROMATICITIES,
};
#[allow(deprecated)]
pub use mipmap_encoder::{
    build_box_filter_pyramid, build_box_filter_ripmap, encode_exr_tiled_mipmap,
    encode_exr_tiled_mipmap_with_line_order, encode_exr_tiled_rgba_float_mipmap_box_filter,
    encode_exr_tiled_rgba_float_ripmap_box_filter, encode_exr_tiled_ripmap,
    encode_exr_tiled_ripmap_with_line_order, mipmap_level_count_round_down,
    ripmap_level_counts_round_down, MipmapLevel, RipmapLevel, RipmapPyramid,
};
#[allow(deprecated)]
pub use multipart_encoder::{
    encode_exr_multipart, encode_exr_multipart_rgba_float_with, MultipartScanlinePart,
};
pub use multipart_mipmap_encoder::{encode_exr_multipart_tiled_mipmap, MultipartMipmapTiledPart};
pub use multipart_mixed_encoder::{
    encode_exr_multipart_mixed, parse_exr_multipart_mixed, MultipartMixedImage, MultipartMixedPart,
};
pub use multipart_ripmap_encoder::{encode_exr_multipart_tiled_ripmap, MultipartRipmapTiledPart};
pub use multipart_tiled_encoder::{encode_exr_multipart_tiled, MultipartTiledPart};
pub use options::{ColourLayout, DecodeOptions, EncodeOptions, LevelMode};
pub use part::{ExrPart, ExrPlane};
#[allow(deprecated)]
pub use tile_encoder::{
    encode_exr_tiled, encode_exr_tiled_rgba_float_with,
    encode_exr_tiled_rgba_float_with_line_order, encode_exr_tiled_with_line_order,
};
pub use types::{
    Attribute, AttributeValue, Box2f, Box2i, Channel, Chromaticities, Compression, EnvMap, Keycode,
    LineOrder, PixelType, Preview, Timecode, EXR_MAGIC,
};

#[cfg(feature = "registry")]
#[doc(hidden)]
pub use registry::__oxideav_entry;
#[cfg(feature = "registry")]
pub use registry::{
    make_decoder, make_encoder, register, register_codecs, register_containers, register_registries,
};
