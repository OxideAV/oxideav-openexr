//! Multi-part EXR encoder + reader for **mixed** part types — a single
//! file carrying any combination of `type="scanlineimage"` (flat
//! scanline), `type="tiledimage"` (flat tiled — ONE_LEVEL, MIPMAP, or
//! RIPMAP), `type="deepscanline"` (deep scanline) and `type="deeptile"`
//! (deep tiled, ONE_LEVEL) parts.
//!
//! Round-232 surface, generalised to deep parts in round 282. Until
//! round 232 every multi-part entry point ([`crate::encode_exr_multipart`]
//! / [`crate::parse_exr_multipart`] for flat scanline,
//! [`crate::encode_exr_multipart_tiled`] /
//! [`crate::parse_exr_multipart_tiled`] for flat tiled ONE_LEVEL, and the
//! deep cousins) required every part to share the same type. The format
//! itself does not enforce that — a multi-part file is simply a sequence
//! of independent parts each declaring its own `type`, `chunkCount`, and
//! tile-ness (via the `tiles[tiledesc]` attribute). This module exposes
//! the general mixed case behind a single encode + parse pair.
//!
//! Binary layout (version-field bit 0x1000 = multipart always set;
//! `non_image` 0x800 set IFF at least one part is deep; `single_tile`
//! 0x200 NOT set — per-part `type` carries the discrimination):
//!
//! ```text
//! magic(4) | version(4 with multipart=0x1000 [| non_image=0x800])
//! | header_0 ... NUL | header_1 ... NUL | NUL          (extra NUL = end-of-headers)
//! | offset_table_0(chunkCount_0×u64) | offset_table_1(...) | ...
//! | chunks: each starts with i32 part_number, then one of
//!     scanline:      i32 Y | i32 size | payload[size]
//!     tiled:         i32 tx | i32 ty | i32 lvlx | i32 lvly | i32 size | payload[size]
//!     deep scanline: i32 Y | u64 packed_table | u64 packed_data
//!                    | u64 unpacked_data | table_bytes | data_bytes
//!     deep tiled:    i32 tx | i32 ty | i32 lvlx | i32 lvly
//!                    | u64 packed_table | u64 packed_data
//!                    | u64 unpacked_data | table_bytes | data_bytes
//!   The reader dispatches the chunk-body shape via the part's declared
//!   `type` attribute, exactly as a homogeneous reader would.
//! ```
//!
//! Flat tiled parts may be ONE_LEVEL, MIPMAP, or RIPMAP (the level mode
//! travels in each part's `tiles[tiledesc]` byte; multi-level tile
//! chunks carry their real `(lvlx, lvly)` indices in the 24-byte tiled
//! chunk header). Deep-tiled parts remain ONE_LEVEL only in a mixed
//! file — deep multi-level files keep using the dedicated deep readers
//! ([`crate::parse_exr_multipart_deep_tiled_mipmap`] /
//! [`crate::parse_exr_multipart_deep_tiled_ripmap`]). Compression per
//! part: NONE / ZIP /
//! ZIPS / RLE for flat parts, NONE / ZIPS / RLE for deep parts (deep
//! ZIP is rejected to match the deep writers elsewhere in this crate).
//! Per-part payload layouts are identical to the homogeneous writers —
//! only the dispatch on `type` is new.
//!
//! Companion reader: [`parse_exr_multipart_mixed`]. Self-roundtrips at
//! every supported compression on layouts mixing all four part types
//! in arbitrary order.

use crate::decoder::{
    apply_zip_interleave, apply_zip_predictor, decode_pxr24_payload, extract_required,
    find_chunk_count, find_part_type, mipmap_level_count, mipmap_level_dim,
    scatter_b44_block_into_planes, scatter_tile_into_planes, subsampled_dim, MultilevelTiledPart,
    Pxr24RowSpec, RequiredAttrs, TiledLevel,
};
use crate::deep::{
    compress_buffer, cumulative_inclusive, decompress_buffer, find_string_attr,
    per_pixel_from_cumulative, DeepMipmapTiledLevelInput, DeepMipmapTiledPart,
    DeepRipmapTiledLevelInput, DeepRipmapTiledPart, DeepScanlinePart, DeepTiledMipmapLevel,
    DeepTiledPart, DeepTiledRipmapCell,
};
use crate::error::{ExrError, Result};
use crate::header::{encode_attribute_value, parse_multipart_headers, VersionField};
use crate::image::{ExrImage, ExrPlane};
use crate::mipmap_encoder::{
    mipmap_level_count_round_down, ripmap_level_counts_round_down, MipmapLevel,
};
use crate::tiled::tiledesc_from_attribute;
use crate::types::{
    Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType, EXR_MAGIC,
};

/// One part for [`encode_exr_multipart_mixed`]. The flat variants carry
/// the same field-set as [`crate::MultipartScanlinePart`] and
/// [`crate::MultipartTiledPart`]; the deep variants mirror
/// [`crate::MultipartDeepScanlinePart`] and
/// [`crate::MultipartDeepTiledPart`]. The writer dispatches on the
/// variant to emit the matching per-part header + chunk shape.
pub enum MultipartMixedPart<'a> {
    /// Flat scanline part (`type="scanlineimage"`). Honours every
    /// rule the single-typed scanline writer does.
    Scanline {
        /// Unique non-empty part name.
        name: String,
        /// dataWindow / displayWindow dimensions (both > 0).
        width: u32,
        height: u32,
        /// Channels in alphabetical order; sub-sampling allowed.
        channels: Vec<Channel>,
        /// One per-channel f32 slice (sub-sampled length per the
        /// channel's `x_sampling` / `y_sampling`).
        planes: Vec<&'a [f32]>,
        compression: Compression,
    },
    /// Flat tiled part (`type="tiledimage"`, ONE_LEVEL + ROUND_DOWN).
    /// Channels must use 1×1 sampling (tiled files require this).
    Tiled {
        /// Unique non-empty part name.
        name: String,
        /// dataWindow / displayWindow dimensions (both > 0).
        width: u32,
        height: u32,
        /// Tile size (both > 0). Edge tiles store only their valid
        /// pixel rectangle.
        tile_x: u32,
        tile_y: u32,
        /// Channels in alphabetical order (1×1 sampling).
        channels: Vec<Channel>,
        /// One `width * height` f32 slice per channel.
        planes: Vec<&'a [f32]>,
        compression: Compression,
    },
    /// Multi-level **MIPMAP** flat tiled part (`type="tiledimage"`,
    /// `tiles[tiledesc, level_mode=1]`, ROUND_DOWN). Carries a full
    /// ROUND_DOWN mipmap pyramid (one entry per level in level-index
    /// order; pyramid length must equal
    /// `mipmap_level_count_round_down(level0_w, level0_h)`). Channels
    /// must be alphabetical with 1×1 sampling. Each level's
    /// `planes.len()` equals `channels.len()`; each plane is
    /// `level_w * level_h` long. The full-resolution dimensions are
    /// taken from `pyramid[0]`.
    TiledMipmap {
        /// Unique non-empty part name.
        name: String,
        /// Tile size (both > 0). Edge tiles at every level store only
        /// their valid pixel rectangle.
        tile_x: u32,
        tile_y: u32,
        /// Channels in alphabetical order (1×1 sampling).
        channels: Vec<Channel>,
        /// Full ROUND_DOWN mipmap pyramid (`pyramid[0]` = full res).
        pyramid: Vec<MipmapLevel>,
        compression: Compression,
    },
    /// Multi-level **RIPMAP** flat tiled part (`type="tiledimage"`,
    /// `tiles[tiledesc, level_mode=2]`, ROUND_DOWN). Carries the full
    /// 2-D ROUND_DOWN reduction grid (`grid[lvly][lvlx]`); the grid
    /// must be `ripmap_level_counts_round_down(level0_w, level0_h)`
    /// shaped, where the full-resolution dimensions come from
    /// `grid[0][0]`. Channels must be alphabetical with 1×1 sampling.
    TiledRipmap {
        /// Unique non-empty part name.
        name: String,
        /// Tile size (both > 0).
        tile_x: u32,
        tile_y: u32,
        /// Channels in alphabetical order (1×1 sampling).
        channels: Vec<Channel>,
        /// Full ROUND_DOWN ripmap grid (`grid[lvly][lvlx]`).
        grid: Vec<Vec<MipmapLevel>>,
        compression: Compression,
    },
    /// Deep scanline part (`type="deepscanline"`). Compression NONE /
    /// ZIPS / RLE only; channels must use 1×1 sampling.
    DeepScanline {
        /// Unique non-empty part name.
        name: String,
        /// dataWindow / displayWindow dimensions (both > 0).
        width: u32,
        height: u32,
        /// Channels in alphabetical order (1×1 sampling).
        channels: Vec<Channel>,
        /// One u32 per pixel (`width * height` long) — how many samples
        /// this pixel carries.
        samples_per_pixel: &'a [u32],
        /// One f32 slice per channel, each `samples_per_pixel.iter().sum()`
        /// long, in pixel-scan order. UINT stored as the u32 bits cast
        /// to f32 (matching the [`crate::DeepExrImage`] convention).
        channel_samples: Vec<&'a [f32]>,
        compression: Compression,
    },
    /// Deep tiled part (`type="deeptile"`, ONE_LEVEL + ROUND_DOWN).
    /// Compression NONE / ZIPS / RLE only; channels must use 1×1
    /// sampling. Edge tiles store only their valid pixel rectangle.
    DeepTiled {
        /// Unique non-empty part name.
        name: String,
        /// dataWindow / displayWindow dimensions (both > 0).
        width: u32,
        height: u32,
        /// Tile size (both > 0).
        tile_x: u32,
        tile_y: u32,
        /// Channels in alphabetical order (1×1 sampling).
        channels: Vec<Channel>,
        /// One u32 per pixel (`width * height` long).
        samples_per_pixel: &'a [u32],
        /// One f32 slice per channel, each `samples_per_pixel.iter().sum()`
        /// long, in pixel-scan order.
        channel_samples: Vec<&'a [f32]>,
        compression: Compression,
    },
    /// Multi-level **MIPMAP** deep tiled part (`type="deeptile"`,
    /// `tiles[tiledesc, level_mode=1]`, ROUND_DOWN). Carries a full
    /// ROUND_DOWN mipmap pyramid of deep levels (one
    /// [`DeepMipmapTiledLevelInput`] per level, level-index order;
    /// pyramid length must equal
    /// `mipmap_level_count_round_down(level0_w, level0_h)`). Full-
    /// resolution dimensions come from `pyramid[0]`. Compression NONE /
    /// ZIPS / RLE only; channels must use 1×1 sampling. Edge tiles at
    /// every level store only their valid pixel rectangle.
    DeepTiledMipmap {
        /// Unique non-empty part name.
        name: String,
        /// Tile size (both > 0).
        tile_x: u32,
        tile_y: u32,
        /// Channels in alphabetical order (1×1 sampling).
        channels: Vec<Channel>,
        /// Full ROUND_DOWN deep mipmap pyramid (`pyramid[0]` = full res).
        pyramid: Vec<DeepMipmapTiledLevelInput<'a>>,
        compression: Compression,
    },
    /// Multi-level **RIPMAP** deep tiled part (`type="deeptile"`,
    /// `tiles[tiledesc, level_mode=2]`, ROUND_DOWN). Carries the full
    /// 2-D ROUND_DOWN reduction grid (`grid[lvly][lvlx]`); the grid must
    /// be `ripmap_level_counts_round_down(level0_w, level0_h)` shaped,
    /// with full-resolution dimensions from `grid[0][0]`. Compression
    /// NONE / ZIPS / RLE only; channels must use 1×1 sampling.
    DeepTiledRipmap {
        /// Unique non-empty part name.
        name: String,
        /// Tile size (both > 0).
        tile_x: u32,
        tile_y: u32,
        /// Channels in alphabetical order (1×1 sampling).
        channels: Vec<Channel>,
        /// Full ROUND_DOWN deep ripmap grid (`grid[lvly][lvlx]`).
        grid: Vec<Vec<DeepRipmapTiledLevelInput<'a>>>,
        compression: Compression,
    },
}

impl MultipartMixedPart<'_> {
    fn name(&self) -> &str {
        match self {
            Self::Scanline { name, .. }
            | Self::Tiled { name, .. }
            | Self::TiledMipmap { name, .. }
            | Self::TiledRipmap { name, .. }
            | Self::DeepScanline { name, .. }
            | Self::DeepTiled { name, .. }
            | Self::DeepTiledMipmap { name, .. }
            | Self::DeepTiledRipmap { name, .. } => name,
        }
    }
    fn is_deep(&self) -> bool {
        matches!(
            self,
            Self::DeepScanline { .. }
                | Self::DeepTiled { .. }
                | Self::DeepTiledMipmap { .. }
                | Self::DeepTiledRipmap { .. }
        )
    }
}

/// One image surfaced by [`parse_exr_multipart_mixed`]. Variants mirror
/// the per-part `type` attribute; flat variants wrap a fully-decoded
/// [`ExrImage`], deep variants wrap the same part payloads the
/// homogeneous deep multi-part readers return.
#[derive(Debug, Clone)]
pub enum MultipartMixedImage {
    Scanline(ExrImage),
    Tiled(ExrImage),
    /// Multi-level MIPMAP flat tiled part; carries every decoded
    /// pyramid level (`level_mode == 1`).
    TiledMipmap(MultilevelTiledPart),
    /// Multi-level RIPMAP flat tiled part; carries the full decoded
    /// reduction grid (`level_mode == 2`).
    TiledRipmap(MultilevelTiledPart),
    DeepScanline(DeepScanlinePart),
    DeepTiled(DeepTiledPart),
    /// Multi-level MIPMAP deep tiled part; carries every decoded pyramid
    /// level (`type="deeptile"`, `level_mode == 1`).
    DeepTiledMipmap(DeepMipmapTiledPart),
    /// Multi-level RIPMAP deep tiled part; carries the full decoded
    /// reduction grid (`type="deeptile"`, `level_mode == 2`).
    DeepTiledRipmap(DeepRipmapTiledPart),
}

impl MultipartMixedImage {
    /// Borrow the underlying decoded flat image (`None` for deep parts).
    pub fn image(&self) -> Option<&ExrImage> {
        match self {
            Self::Scanline(img) | Self::Tiled(img) => Some(img),
            _ => None,
        }
    }
    /// Consume and return the underlying decoded flat image (`None` for
    /// deep parts).
    pub fn into_image(self) -> Option<ExrImage> {
        match self {
            Self::Scanline(img) | Self::Tiled(img) => Some(img),
            _ => None,
        }
    }
    /// Borrow the deep scanline payload (`None` for other part types).
    pub fn deep_scanline(&self) -> Option<&DeepScanlinePart> {
        match self {
            Self::DeepScanline(p) => Some(p),
            _ => None,
        }
    }
    /// Borrow the deep tiled payload (`None` for other part types).
    pub fn deep_tiled(&self) -> Option<&DeepTiledPart> {
        match self {
            Self::DeepTiled(p) => Some(p),
            _ => None,
        }
    }
    /// True for flat scanline parts (`type="scanlineimage"`).
    pub fn is_scanline(&self) -> bool {
        matches!(self, Self::Scanline(_))
    }
    /// True for flat tiled parts (`type="tiledimage"`).
    pub fn is_tiled(&self) -> bool {
        matches!(self, Self::Tiled(_))
    }
    /// True for deep scanline parts (`type="deepscanline"`).
    pub fn is_deep_scanline(&self) -> bool {
        matches!(self, Self::DeepScanline(_))
    }
    /// True for deep tiled parts (`type="deeptile"`).
    pub fn is_deep_tiled(&self) -> bool {
        matches!(self, Self::DeepTiled(_))
    }
    /// Borrow the decoded multi-level (MIPMAP or RIPMAP) flat tiled
    /// part (`None` for every other part type).
    pub fn multilevel_tiled(&self) -> Option<&MultilevelTiledPart> {
        match self {
            Self::TiledMipmap(p) | Self::TiledRipmap(p) => Some(p),
            _ => None,
        }
    }
    /// True for multi-level MIPMAP flat tiled parts.
    pub fn is_tiled_mipmap(&self) -> bool {
        matches!(self, Self::TiledMipmap(_))
    }
    /// True for multi-level RIPMAP flat tiled parts.
    pub fn is_tiled_ripmap(&self) -> bool {
        matches!(self, Self::TiledRipmap(_))
    }
    /// Borrow the decoded multi-level MIPMAP deep tiled part (`None` for
    /// every other part type).
    pub fn deep_tiled_mipmap(&self) -> Option<&DeepMipmapTiledPart> {
        match self {
            Self::DeepTiledMipmap(p) => Some(p),
            _ => None,
        }
    }
    /// Borrow the decoded multi-level RIPMAP deep tiled part (`None` for
    /// every other part type).
    pub fn deep_tiled_ripmap(&self) -> Option<&DeepRipmapTiledPart> {
        match self {
            Self::DeepTiledRipmap(p) => Some(p),
            _ => None,
        }
    }
    /// True for multi-level MIPMAP deep tiled parts.
    pub fn is_deep_tiled_mipmap(&self) -> bool {
        matches!(self, Self::DeepTiledMipmap(_))
    }
    /// True for multi-level RIPMAP deep tiled parts.
    pub fn is_deep_tiled_ripmap(&self) -> bool {
        matches!(self, Self::DeepTiledRipmap(_))
    }
}

/// Validate the deep-part field set shared by both deep variants.
#[allow(clippy::too_many_arguments)]
fn validate_deep_part(
    name: &str,
    width: u32,
    height: u32,
    channels: &[Channel],
    samples_per_pixel: &[u32],
    channel_samples: &[&[f32]],
    compression: Compression,
) -> Result<()> {
    if !matches!(
        compression,
        Compression::None | Compression::Rle | Compression::Zips
    ) {
        return Err(ExrError::unsupported(format!(
            "mixed multi-part deep part '{name}': compression {compression:?} \
             (deep parts accept only NONE/RLE/ZIPS)"
        )));
    }
    if width == 0 || height == 0 {
        return Err(ExrError::invalid(format!(
            "mixed multi-part deep part '{name}': dataWindow {width}×{height} must be > 0"
        )));
    }
    let pixels = (width as usize) * (height as usize);
    if samples_per_pixel.len() != pixels {
        return Err(ExrError::invalid(format!(
            "mixed multi-part deep part '{name}': samples_per_pixel len {} != \
             width*height = {pixels}",
            samples_per_pixel.len()
        )));
    }
    if channels.len() != channel_samples.len() {
        return Err(ExrError::invalid(format!(
            "mixed multi-part deep part '{name}': channels.len()={} != \
             channel_samples.len()={}",
            channels.len(),
            channel_samples.len()
        )));
    }
    for win in channels.windows(2) {
        if win[0].name >= win[1].name {
            return Err(ExrError::invalid(format!(
                "mixed multi-part deep part '{name}': channels not alphabetical: \
                 '{}' >= '{}'",
                win[0].name, win[1].name
            )));
        }
    }
    let total_samples: u64 = samples_per_pixel.iter().map(|&n| n as u64).sum();
    for (ch, slc) in channels.iter().zip(channel_samples.iter()) {
        if ch.x_sampling != 1 || ch.y_sampling != 1 {
            return Err(ExrError::unsupported(format!(
                "mixed multi-part deep part '{name}': sub-sampled channel '{}' \
                 (deep parts require 1×1 sampling)",
                ch.name
            )));
        }
        if slc.len() != total_samples as usize {
            return Err(ExrError::invalid(format!(
                "mixed multi-part deep part '{name}': channel '{}' sample slice \
                 len {} != total_samples {total_samples}",
                ch.name,
                slc.len()
            )));
        }
    }
    Ok(())
}

/// Validate one flat tiled level's plane shape against `level_w ×
/// level_h` (the shape rule shared by MIPMAP and RIPMAP parts).
fn validate_tiled_level(
    name: &str,
    label: &str,
    channels: &[Channel],
    lvl: &MipmapLevel,
    want_w: u32,
    want_h: u32,
) -> Result<()> {
    if lvl.width != want_w || lvl.height != want_h {
        return Err(ExrError::invalid(format!(
            "mixed multi-part {label} part '{name}': level is {}×{}, ROUND_DOWN spec \
             requires {want_w}×{want_h}",
            lvl.width, lvl.height
        )));
    }
    if lvl.planes.len() != channels.len() {
        return Err(ExrError::invalid(format!(
            "mixed multi-part {label} part '{name}': level has {} planes but {} channels \
             declared",
            lvl.planes.len(),
            channels.len()
        )));
    }
    let need = (lvl.width as usize) * (lvl.height as usize);
    for (ch, plane) in channels.iter().zip(lvl.planes.iter()) {
        if plane.len() != need {
            return Err(ExrError::invalid(format!(
                "mixed multi-part {label} part '{name}': level channel '{}' plane length \
                 {} != {}×{} = {need}",
                ch.name,
                plane.len(),
                lvl.width,
                lvl.height
            )));
        }
    }
    Ok(())
}

/// Validate the channel + compression + tile-size rules common to every
/// flat tiled part variant (ONE_LEVEL, MIPMAP, RIPMAP).
fn validate_tiled_common(
    name: &str,
    label: &str,
    tile_x: u32,
    tile_y: u32,
    channels: &[Channel],
    compression: Compression,
) -> Result<()> {
    if !matches!(
        compression,
        Compression::None
            | Compression::Zip
            | Compression::Zips
            | Compression::Rle
            | Compression::Pxr24
            | Compression::B44
            | Compression::B44a
    ) {
        return Err(ExrError::unsupported(format!(
            "mixed multi-part {label} part '{name}': compression {compression:?} \
             (encoder supports NONE/ZIP/ZIPS/RLE/PXR24/B44/B44A)"
        )));
    }
    if tile_x == 0 || tile_y == 0 {
        return Err(ExrError::invalid(format!(
            "mixed multi-part {label} part '{name}': tile size {tile_x}×{tile_y} must both be > 0"
        )));
    }
    for win in channels.windows(2) {
        if win[0].name >= win[1].name {
            return Err(ExrError::invalid(format!(
                "mixed multi-part {label} part '{name}': channels not alphabetical: \
                 '{}' >= '{}'",
                win[0].name, win[1].name
            )));
        }
    }
    for ch in channels {
        if ch.x_sampling != 1 || ch.y_sampling != 1 {
            return Err(ExrError::unsupported(format!(
                "mixed multi-part {label} part '{name}': sub-sampled channel '{}' \
                 (tiled parts require 1×1 sampling)",
                ch.name
            )));
        }
    }
    Ok(())
}

/// Validate the channel + compression + tile-size rules common to the
/// multi-level **deep** tiled variants (MIPMAP + RIPMAP). Deep parts
/// accept only NONE / ZIPS / RLE.
fn validate_deep_multilevel_common(
    name: &str,
    label: &str,
    tile_x: u32,
    tile_y: u32,
    channels: &[Channel],
    compression: Compression,
) -> Result<()> {
    if !matches!(
        compression,
        Compression::None | Compression::Rle | Compression::Zips
    ) {
        return Err(ExrError::unsupported(format!(
            "mixed multi-part {label} part '{name}': compression {compression:?} \
             (deep parts accept only NONE/ZIPS/RLE)"
        )));
    }
    if tile_x == 0 || tile_y == 0 {
        return Err(ExrError::invalid(format!(
            "mixed multi-part {label} part '{name}': tile size {tile_x}×{tile_y} must both be > 0"
        )));
    }
    for win in channels.windows(2) {
        if win[0].name >= win[1].name {
            return Err(ExrError::invalid(format!(
                "mixed multi-part {label} part '{name}': channels not alphabetical: \
                 '{}' >= '{}'",
                win[0].name, win[1].name
            )));
        }
    }
    for ch in channels {
        if ch.x_sampling != 1 || ch.y_sampling != 1 {
            return Err(ExrError::unsupported(format!(
                "mixed multi-part {label} part '{name}': sub-sampled channel '{}' \
                 (deep tiled parts require 1×1 sampling)",
                ch.name
            )));
        }
    }
    Ok(())
}

/// Validate one deep tiled level's shape: dimensions match the
/// ROUND_DOWN expectation, `samples_per_pixel` has one entry per pixel,
/// and each channel plane length equals the total sample count.
#[allow(clippy::too_many_arguments)]
fn validate_deep_level(
    name: &str,
    label: &str,
    channels: &[Channel],
    width: u32,
    height: u32,
    samples_per_pixel: &[u32],
    channel_samples: &[&[f32]],
    want_w: u32,
    want_h: u32,
) -> Result<()> {
    if width != want_w || height != want_h {
        return Err(ExrError::invalid(format!(
            "mixed multi-part {label} part '{name}': level is {width}×{height}, ROUND_DOWN \
             spec requires {want_w}×{want_h}"
        )));
    }
    let pixels = (width as usize) * (height as usize);
    if samples_per_pixel.len() != pixels {
        return Err(ExrError::invalid(format!(
            "mixed multi-part {label} part '{name}': level samples_per_pixel len {} != \
             {width}×{height} = {pixels}",
            samples_per_pixel.len()
        )));
    }
    if channels.len() != channel_samples.len() {
        return Err(ExrError::invalid(format!(
            "mixed multi-part {label} part '{name}': level has {} planes but {} channels declared",
            channel_samples.len(),
            channels.len()
        )));
    }
    let total: u64 = samples_per_pixel.iter().map(|&n| n as u64).sum();
    for (ch, slc) in channels.iter().zip(channel_samples.iter()) {
        if slc.len() as u64 != total {
            return Err(ExrError::invalid(format!(
                "mixed multi-part {label} part '{name}': level channel '{}' sample slice \
                 len {} != total_samples {total}",
                ch.name,
                slc.len()
            )));
        }
    }
    Ok(())
}

/// Build one deep tile's compressed (offset-table, sample-data) payload
/// plus the uncompressed sample-data length, from one level's full deep
/// data. `lvl_w`/`lvl_h` are the level's pixel dimensions;
/// `pixel_sample_starts` is the cumulative-EXCLUSIVE per-pixel sample
/// offset table over that level (length `lvl_w*lvl_h + 1`). The tile is
/// the rectangle `(tx*tile_x, ty*tile_y)` clipped to the level bounds;
/// its per-row offset table is cumulative-inclusive restarting per row,
/// and its sample bytes are channel-major within the tile in pixel-scan
/// order. This is the shared per-tile builder used by both the ONE_LEVEL
/// and the multi-level (MIPMAP / RIPMAP) deep tiled emission arms.
#[allow(clippy::too_many_arguments)]
fn build_deep_tile_payload(
    name: &str,
    lvl_w: u32,
    lvl_h: u32,
    tile_x: u32,
    tile_y: u32,
    tx: u32,
    ty: u32,
    channels: &[Channel],
    samples_per_pixel: &[u32],
    channel_samples: &[&[f32]],
    pixel_sample_starts: &[u64],
    compression: Compression,
) -> Result<(Vec<u8>, Vec<u8>, u64)> {
    let w = lvl_w as usize;
    let x0 = tx * tile_x;
    let y0 = ty * tile_y;
    let x1 = (x0 + tile_x).min(lvl_w);
    let y1 = (y0 + tile_y).min(lvl_h);
    let tw = (x1 - x0) as usize;
    let th = (y1 - y0) as usize;
    // Per-row cumulative-inclusive offsets, restarting per row within
    // the tile rectangle.
    let mut table_bytes = Vec::with_capacity(tw * th * 4);
    for r in 0..th {
        let dst_y = y0 as usize + r;
        let mut row_acc: i32 = 0;
        for c in 0..tw {
            let dst_x = x0 as usize + c;
            let n = samples_per_pixel[dst_y * w + dst_x];
            row_acc = row_acc.checked_add(n as i32).ok_or_else(|| {
                ExrError::invalid(format!(
                    "mixed multi-part deep tiled part '{name}' tile ({tx},{ty}) row {r}: \
                     cumulative offset overflows i32"
                ))
            })?;
            table_bytes.extend_from_slice(&row_acc.to_le_bytes());
        }
    }
    // Channel-major sample bytes for this tile.
    let mut sample_bytes: Vec<u8> = Vec::new();
    for (ch_idx, ch) in channels.iter().enumerate() {
        let plane = channel_samples[ch_idx];
        for r in 0..th {
            let dst_y = y0 as usize + r;
            for c in 0..tw {
                let dst_x = x0 as usize + c;
                let pi = dst_y * w + dst_x;
                let s_start = pixel_sample_starts[pi] as usize;
                let s_end = pixel_sample_starts[pi + 1] as usize;
                for &v in &plane[s_start..s_end] {
                    push_pixel(&mut sample_bytes, v, ch.pixel_type);
                }
            }
        }
    }
    let unpacked_len = sample_bytes.len() as u64;
    let packed_table = compress_buffer(&table_bytes, compression)?;
    let packed_data = compress_buffer(&sample_bytes, compression)?;
    Ok((packed_table, packed_data, unpacked_len))
}

/// Cumulative-EXCLUSIVE per-pixel sample offset table over one level's
/// `samples_per_pixel` (length `spp.len() + 1`).
fn pixel_sample_starts(samples_per_pixel: &[u32]) -> Vec<u64> {
    let mut starts = Vec::with_capacity(samples_per_pixel.len() + 1);
    starts.push(0u64);
    let mut acc: u64 = 0;
    for &n in samples_per_pixel {
        acc += n as u64;
        starts.push(acc);
    }
    starts
}

/// Encode a multi-part EXR file whose parts may freely mix
/// `type="scanlineimage"`, `type="tiledimage"` (ONE_LEVEL, MIPMAP,
/// RIPMAP), `type="deepscanline"` and `type="deeptile"` (ONE_LEVEL).
/// Validation,
/// attribute layout, and chunk-body emission per part mirror the
/// homogeneous writers exactly; this entry only adds the dispatch on
/// [`MultipartMixedPart`] variant. The version field sets the
/// `non_image` (deep) bit when at least one part is deep.
///
/// Self-roundtrips through [`parse_exr_multipart_mixed`].
pub fn encode_exr_multipart_mixed(parts: &[MultipartMixedPart]) -> Result<Vec<u8>> {
    if parts.is_empty() {
        return Err(ExrError::invalid(
            "encode_exr_multipart_mixed: at least one part required".to_string(),
        ));
    }

    // ---- Per-part validation (shape rules identical to homogeneous writers). ----
    for (i, p) in parts.iter().enumerate() {
        let name = p.name();
        if name.is_empty() {
            return Err(ExrError::invalid(format!(
                "mixed multi-part part {i}: empty name"
            )));
        }
        for (j, other) in parts.iter().enumerate() {
            if j != i && other.name() == name {
                return Err(ExrError::invalid(format!(
                    "duplicate mixed multi-part name '{name}' (parts {i} and {j})"
                )));
            }
        }
        match p {
            MultipartMixedPart::Scanline {
                width,
                height,
                channels,
                planes,
                compression,
                ..
            } => {
                if !matches!(
                    compression,
                    Compression::None
                        | Compression::Zip
                        | Compression::Zips
                        | Compression::Rle
                        | Compression::Pxr24
                        | Compression::B44
                        | Compression::B44a
                ) {
                    return Err(ExrError::unsupported(format!(
                        "mixed multi-part part '{name}': compression {compression:?} \
                         (scanline supports NONE/ZIP/ZIPS/RLE/PXR24/B44/B44A)"
                    )));
                }
                if *width == 0 || *height == 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part part '{name}': dataWindow {width}×{height} must be > 0"
                    )));
                }
                if channels.len() != planes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part part '{name}': channels.len()={} != planes.len()={}",
                        channels.len(),
                        planes.len()
                    )));
                }
                for win in channels.windows(2) {
                    if win[0].name >= win[1].name {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part part '{name}': channels not alphabetical: \
                             '{}' >= '{}'",
                            win[0].name, win[1].name
                        )));
                    }
                }
                for (ch, plane) in channels.iter().zip(planes.iter()) {
                    if ch.x_sampling <= 0 || ch.y_sampling <= 0 {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part part '{name}': channel '{}' \
                             x_sampling={} y_sampling={} (must be positive)",
                            ch.name, ch.x_sampling, ch.y_sampling
                        )));
                    }
                    let pw = subsampled_dim(*width, ch.x_sampling as u32) as usize;
                    let ph = subsampled_dim(*height, ch.y_sampling as u32) as usize;
                    let need = pw * ph;
                    if plane.len() != need {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part part '{name}': channel '{}' plane length {} \
                             != subsampled {pw}×{ph} = {need}",
                            ch.name,
                            plane.len()
                        )));
                    }
                }
            }
            MultipartMixedPart::Tiled {
                width,
                height,
                tile_x,
                tile_y,
                channels,
                planes,
                compression,
                ..
            } => {
                if !matches!(
                    compression,
                    Compression::None
                        | Compression::Zip
                        | Compression::Zips
                        | Compression::Rle
                        | Compression::Pxr24
                        | Compression::B44
                        | Compression::B44a
                ) {
                    return Err(ExrError::unsupported(format!(
                        "mixed multi-part part '{name}': compression {compression:?} \
                         (ONE_LEVEL tiled supports NONE/ZIP/ZIPS/RLE/PXR24/B44/B44A)"
                    )));
                }
                if *width == 0 || *height == 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part part '{name}': dataWindow {width}×{height} must be > 0"
                    )));
                }
                if *tile_x == 0 || *tile_y == 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part part '{name}': tile size {tile_x}×{tile_y} \
                         must both be > 0"
                    )));
                }
                if channels.len() != planes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part part '{name}': channels.len()={} != planes.len()={}",
                        channels.len(),
                        planes.len()
                    )));
                }
                for win in channels.windows(2) {
                    if win[0].name >= win[1].name {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part part '{name}': channels not alphabetical: \
                             '{}' >= '{}'",
                            win[0].name, win[1].name
                        )));
                    }
                }
                for (ch, plane) in channels.iter().zip(planes.iter()) {
                    if ch.x_sampling != 1 || ch.y_sampling != 1 {
                        return Err(ExrError::unsupported(format!(
                            "mixed multi-part part '{name}': sub-sampled channel '{}' \
                             (tiled parts require 1×1 sampling)",
                            ch.name
                        )));
                    }
                    let need = (*width as usize) * (*height as usize);
                    if plane.len() != need {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part part '{name}': channel '{}' plane length {} \
                             != width*height = {need}",
                            ch.name,
                            plane.len()
                        )));
                    }
                }
            }
            MultipartMixedPart::TiledMipmap {
                tile_x,
                tile_y,
                channels,
                pyramid,
                compression,
                ..
            } => {
                validate_tiled_common(
                    name,
                    "mipmap tiled",
                    *tile_x,
                    *tile_y,
                    channels,
                    *compression,
                )?;
                if pyramid.is_empty() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part mipmap tiled part '{name}': empty pyramid"
                    )));
                }
                let width = pyramid[0].width;
                let height = pyramid[0].height;
                if width == 0 || height == 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part mipmap tiled part '{name}': level-0 dataWindow \
                         {width}×{height} must be > 0"
                    )));
                }
                let want_levels = mipmap_level_count_round_down(width, height);
                if pyramid.len() as u32 != want_levels {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part mipmap tiled part '{name}': pyramid has {} levels, \
                         expected {want_levels} for {width}×{height} ROUND_DOWN",
                        pyramid.len()
                    )));
                }
                for (l, lvl) in pyramid.iter().enumerate() {
                    let want_w = mipmap_level_dim(width, l as u32, false);
                    let want_h = mipmap_level_dim(height, l as u32, false);
                    validate_tiled_level(name, "mipmap tiled", channels, lvl, want_w, want_h)?;
                }
            }
            MultipartMixedPart::TiledRipmap {
                tile_x,
                tile_y,
                channels,
                grid,
                compression,
                ..
            } => {
                validate_tiled_common(
                    name,
                    "ripmap tiled",
                    *tile_x,
                    *tile_y,
                    channels,
                    *compression,
                )?;
                if grid.is_empty() || grid[0].is_empty() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part ripmap tiled part '{name}': empty grid"
                    )));
                }
                let width = grid[0][0].width;
                let height = grid[0][0].height;
                if width == 0 || height == 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part ripmap tiled part '{name}': level-0 dataWindow \
                         {width}×{height} must be > 0"
                    )));
                }
                let (nx, ny) = ripmap_level_counts_round_down(width, height);
                if grid.len() as u32 != ny {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part ripmap tiled part '{name}': grid has {} rows, \
                         expected {ny} y-levels for {width}×{height} ROUND_DOWN",
                        grid.len()
                    )));
                }
                for (ly, row) in grid.iter().enumerate() {
                    if row.len() as u32 != nx {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part ripmap tiled part '{name}': grid row {ly} has {} \
                             cells, expected {nx} x-levels",
                            row.len()
                        )));
                    }
                    let want_h = mipmap_level_dim(height, ly as u32, false);
                    for (lx, lvl) in row.iter().enumerate() {
                        let want_w = mipmap_level_dim(width, lx as u32, false);
                        validate_tiled_level(name, "ripmap tiled", channels, lvl, want_w, want_h)?;
                    }
                }
            }
            MultipartMixedPart::DeepScanline {
                width,
                height,
                channels,
                samples_per_pixel,
                channel_samples,
                compression,
                ..
            } => {
                validate_deep_part(
                    name,
                    *width,
                    *height,
                    channels,
                    samples_per_pixel,
                    channel_samples,
                    *compression,
                )?;
            }
            MultipartMixedPart::DeepTiled {
                width,
                height,
                tile_x,
                tile_y,
                channels,
                samples_per_pixel,
                channel_samples,
                compression,
                ..
            } => {
                if *tile_x == 0 || *tile_y == 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part part '{name}': tile size {tile_x}×{tile_y} \
                         must both be > 0"
                    )));
                }
                validate_deep_part(
                    name,
                    *width,
                    *height,
                    channels,
                    samples_per_pixel,
                    channel_samples,
                    *compression,
                )?;
            }
            MultipartMixedPart::DeepTiledMipmap {
                tile_x,
                tile_y,
                channels,
                pyramid,
                compression,
                ..
            } => {
                validate_deep_multilevel_common(
                    name,
                    "deep mipmap tiled",
                    *tile_x,
                    *tile_y,
                    channels,
                    *compression,
                )?;
                if pyramid.is_empty() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part deep mipmap tiled part '{name}': empty pyramid"
                    )));
                }
                let width = pyramid[0].width;
                let height = pyramid[0].height;
                if width == 0 || height == 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part deep mipmap tiled part '{name}': level-0 dataWindow \
                         {width}×{height} must be > 0"
                    )));
                }
                let want_levels = mipmap_level_count_round_down(width, height);
                if pyramid.len() as u32 != want_levels {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part deep mipmap tiled part '{name}': pyramid has {} levels, \
                         expected {want_levels} for {width}×{height} ROUND_DOWN",
                        pyramid.len()
                    )));
                }
                for (l, lvl) in pyramid.iter().enumerate() {
                    let want_w = mipmap_level_dim(width, l as u32, false);
                    let want_h = mipmap_level_dim(height, l as u32, false);
                    validate_deep_level(
                        name,
                        "deep mipmap tiled",
                        channels,
                        lvl.width,
                        lvl.height,
                        lvl.samples_per_pixel,
                        &lvl.channel_samples,
                        want_w,
                        want_h,
                    )?;
                }
            }
            MultipartMixedPart::DeepTiledRipmap {
                tile_x,
                tile_y,
                channels,
                grid,
                compression,
                ..
            } => {
                validate_deep_multilevel_common(
                    name,
                    "deep ripmap tiled",
                    *tile_x,
                    *tile_y,
                    channels,
                    *compression,
                )?;
                if grid.is_empty() || grid[0].is_empty() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part deep ripmap tiled part '{name}': empty grid"
                    )));
                }
                let width = grid[0][0].width;
                let height = grid[0][0].height;
                if width == 0 || height == 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part deep ripmap tiled part '{name}': level-0 dataWindow \
                         {width}×{height} must be > 0"
                    )));
                }
                let (nx, ny) = ripmap_level_counts_round_down(width, height);
                if grid.len() as u32 != ny {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part deep ripmap tiled part '{name}': grid has {} rows, \
                         expected {ny} y-levels for {width}×{height} ROUND_DOWN",
                        grid.len()
                    )));
                }
                for (ly, row) in grid.iter().enumerate() {
                    if row.len() as u32 != nx {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part deep ripmap tiled part '{name}': grid row {ly} has \
                             {} cells, expected {nx} x-levels",
                            row.len()
                        )));
                    }
                    let want_h = mipmap_level_dim(height, ly as u32, false);
                    for (lx, lvl) in row.iter().enumerate() {
                        let want_w = mipmap_level_dim(width, lx as u32, false);
                        validate_deep_level(
                            name,
                            "deep ripmap tiled",
                            channels,
                            lvl.width,
                            lvl.height,
                            lvl.samples_per_pixel,
                            &lvl.channel_samples,
                            want_w,
                            want_h,
                        )?;
                    }
                }
            }
        }
    }

    // ---- Per-part chunk counts. ----
    let mut chunk_counts: Vec<u32> = Vec::with_capacity(parts.len());
    for p in parts {
        let cc = match p {
            MultipartMixedPart::Scanline {
                height,
                compression,
                ..
            }
            | MultipartMixedPart::DeepScanline {
                height,
                compression,
                ..
            } => {
                let block_h = compression.scanlines_per_block();
                height.div_ceil(block_h)
            }
            MultipartMixedPart::Tiled {
                width,
                height,
                tile_x,
                tile_y,
                ..
            }
            | MultipartMixedPart::DeepTiled {
                width,
                height,
                tile_x,
                tile_y,
                ..
            } => width.div_ceil(*tile_x) * height.div_ceil(*tile_y),
            MultipartMixedPart::TiledMipmap {
                tile_x,
                tile_y,
                pyramid,
                ..
            } => pyramid
                .iter()
                .map(|lvl| lvl.width.div_ceil(*tile_x) * lvl.height.div_ceil(*tile_y))
                .sum(),
            MultipartMixedPart::TiledRipmap {
                tile_x,
                tile_y,
                grid,
                ..
            } => grid
                .iter()
                .flat_map(|row| row.iter())
                .map(|lvl| lvl.width.div_ceil(*tile_x) * lvl.height.div_ceil(*tile_y))
                .sum(),
            MultipartMixedPart::DeepTiledMipmap {
                tile_x,
                tile_y,
                pyramid,
                ..
            } => pyramid
                .iter()
                .map(|lvl| lvl.width.div_ceil(*tile_x) * lvl.height.div_ceil(*tile_y))
                .sum(),
            MultipartMixedPart::DeepTiledRipmap {
                tile_x,
                tile_y,
                grid,
                ..
            } => grid
                .iter()
                .flat_map(|row| row.iter())
                .map(|lvl| lvl.width.div_ceil(*tile_x) * lvl.height.div_ceil(*tile_y))
                .sum(),
        };
        chunk_counts.push(cc);
    }

    // ---- Per-part header byte blocks. ----
    // Every part of a multi-part file must carry the SAME displayWindow
    // (the reference `exrheader` refuses files whose parts disagree);
    // dataWindow stays per-part. Use the union of the part data windows.
    let part_dims = |p: &MultipartMixedPart| -> (u32, u32) {
        match p {
            MultipartMixedPart::Scanline { width, height, .. }
            | MultipartMixedPart::Tiled { width, height, .. }
            | MultipartMixedPart::DeepScanline { width, height, .. }
            | MultipartMixedPart::DeepTiled { width, height, .. } => (*width, *height),
            MultipartMixedPart::TiledMipmap { pyramid, .. } => {
                (pyramid[0].width, pyramid[0].height)
            }
            MultipartMixedPart::TiledRipmap { grid, .. } => (grid[0][0].width, grid[0][0].height),
            MultipartMixedPart::DeepTiledMipmap { pyramid, .. } => {
                (pyramid[0].width, pyramid[0].height)
            }
            MultipartMixedPart::DeepTiledRipmap { grid, .. } => {
                (grid[0][0].width, grid[0][0].height)
            }
        }
    };
    let disp_w = parts.iter().map(|p| part_dims(p).0).max().unwrap();
    let disp_h = parts.iter().map(|p| part_dims(p).1).max().unwrap();
    let display_window = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (disp_w - 1) as i32,
        y_max: (disp_h - 1) as i32,
    };
    let mut header_byte_blocks: Vec<Vec<u8>> = Vec::with_capacity(parts.len());
    for (i, p) in parts.iter().enumerate() {
        let attrs = match p {
            MultipartMixedPart::Scanline {
                name,
                width,
                height,
                channels,
                compression,
                ..
            } => build_scanline_part_attrs(
                name,
                *width,
                *height,
                display_window,
                channels,
                *compression,
                chunk_counts[i] as i32,
            ),
            MultipartMixedPart::Tiled {
                name,
                width,
                height,
                tile_x,
                tile_y,
                channels,
                compression,
                ..
            } => build_tiled_part_attrs(
                name,
                *width,
                *height,
                display_window,
                *tile_x,
                *tile_y,
                channels,
                *compression,
                chunk_counts[i] as i32,
            ),
            MultipartMixedPart::TiledMipmap {
                name,
                tile_x,
                tile_y,
                channels,
                pyramid,
                compression,
            } => build_multilevel_tiled_part_attrs(
                name,
                pyramid[0].width,
                pyramid[0].height,
                display_window,
                *tile_x,
                *tile_y,
                1, // MIPMAP_LEVELS
                channels,
                *compression,
                chunk_counts[i] as i32,
            ),
            MultipartMixedPart::TiledRipmap {
                name,
                tile_x,
                tile_y,
                channels,
                grid,
                compression,
            } => build_multilevel_tiled_part_attrs(
                name,
                grid[0][0].width,
                grid[0][0].height,
                display_window,
                *tile_x,
                *tile_y,
                2, // RIPMAP_LEVELS
                channels,
                *compression,
                chunk_counts[i] as i32,
            ),
            MultipartMixedPart::DeepScanline {
                name,
                width,
                height,
                channels,
                samples_per_pixel,
                compression,
                ..
            } => {
                let max_samples = samples_per_pixel.iter().copied().max().unwrap_or(0) as i32;
                build_deep_part_attrs(
                    name,
                    *width,
                    *height,
                    display_window,
                    None,
                    channels,
                    *compression,
                    chunk_counts[i] as i32,
                    max_samples,
                )
            }
            MultipartMixedPart::DeepTiled {
                name,
                width,
                height,
                tile_x,
                tile_y,
                channels,
                samples_per_pixel,
                compression,
                ..
            } => {
                let max_samples = samples_per_pixel.iter().copied().max().unwrap_or(0) as i32;
                build_deep_part_attrs(
                    name,
                    *width,
                    *height,
                    display_window,
                    Some((*tile_x, *tile_y, 0)),
                    channels,
                    *compression,
                    chunk_counts[i] as i32,
                    max_samples,
                )
            }
            MultipartMixedPart::DeepTiledMipmap {
                name,
                tile_x,
                tile_y,
                channels,
                pyramid,
                compression,
            } => {
                let max_samples = pyramid
                    .iter()
                    .flat_map(|lvl| lvl.samples_per_pixel.iter().copied())
                    .max()
                    .unwrap_or(0) as i32;
                build_deep_part_attrs(
                    name,
                    pyramid[0].width,
                    pyramid[0].height,
                    display_window,
                    Some((*tile_x, *tile_y, 1)), // MIPMAP_LEVELS
                    channels,
                    *compression,
                    chunk_counts[i] as i32,
                    max_samples,
                )
            }
            MultipartMixedPart::DeepTiledRipmap {
                name,
                tile_x,
                tile_y,
                channels,
                grid,
                compression,
            } => {
                let max_samples = grid
                    .iter()
                    .flat_map(|row| row.iter())
                    .flat_map(|lvl| lvl.samples_per_pixel.iter().copied())
                    .max()
                    .unwrap_or(0) as i32;
                build_deep_part_attrs(
                    name,
                    grid[0][0].width,
                    grid[0][0].height,
                    display_window,
                    Some((*tile_x, *tile_y, 2)), // RIPMAP_LEVELS
                    channels,
                    *compression,
                    chunk_counts[i] as i32,
                    max_samples,
                )
            }
        };
        header_byte_blocks.push(encode_part_header_attributes(&attrs));
    }

    // ---- Stitch magic + version + headers + double-NUL. ----
    // multipart (0x1000) always; `non_image` (0x800) when at least one
    // part is deep; `single_tile` (0x200) is never set on multi-part
    // files (the per-part `tiles[tiledesc]` attribute + `type` string
    // carry the tile-ness signal).
    let any_deep = parts.iter().any(|p| p.is_deep());
    let version_bits = 2 | 0x1000 | if any_deep { 0x800 } else { 0 };
    let version = VersionField::from_u32(version_bits);
    let mut out: Vec<u8> = Vec::with_capacity(2048);
    out.extend_from_slice(&EXR_MAGIC.to_le_bytes());
    out.extend_from_slice(&version.to_u32().to_le_bytes());
    for hb in &header_byte_blocks {
        out.extend_from_slice(hb);
        out.push(0); // per-part header terminator
    }
    out.push(0); // double-NUL = end-of-all-headers

    // ---- Build per-part chunk payloads. ----
    // One record per chunk keeps the four chunk-body shapes distinct so
    // the offset / emission pass can dispatch per part.
    enum ChunkPayload {
        Scanline {
            y: u32,
            payload: Vec<u8>,
        },
        Tile {
            tx: u32,
            ty: u32,
            lvl_x: u32,
            lvl_y: u32,
            payload: Vec<u8>,
        },
        DeepScanline {
            y: u32,
            packed_table: Vec<u8>,
            packed_data: Vec<u8>,
            unpacked_len: u64,
        },
        DeepTile {
            tx: u32,
            ty: u32,
            lvl_x: u32,
            lvl_y: u32,
            packed_table: Vec<u8>,
            packed_data: Vec<u8>,
            unpacked_len: u64,
        },
    }
    let mut all_chunks: Vec<(u32, ChunkPayload)> = Vec::new();
    for (part_idx, p) in parts.iter().enumerate() {
        match p {
            MultipartMixedPart::Scanline {
                width,
                height,
                channels,
                planes,
                compression,
                ..
            } => {
                let block_h = compression.scanlines_per_block();
                let cc = chunk_counts[part_idx] as usize;
                let plane_refs: Vec<&[f32]> = planes.to_vec();
                for block_idx in 0..cc {
                    let row0 = block_idx as u32 * block_h;
                    let lines_in_block = (height - row0).min(block_h) as usize;
                    // PXR24 / B44 / B44A reorganise the whole chunk (byte
                    // planes / per-channel 4×4 blocks) directly from the
                    // f32 source planes; the shared block builders apply the
                    // §0 raw-fallback internally.
                    let payload = match *compression {
                        Compression::Pxr24 => crate::encoder::build_pxr24_block_payload(
                            channels,
                            &plane_refs,
                            *width,
                            row0,
                            lines_in_block,
                        )?,
                        Compression::B44 | Compression::B44a => {
                            let flat = matches!(*compression, Compression::B44a);
                            let raw_len =
                                scanline_block_raw_len(channels, *width, row0, lines_in_block);
                            let packed = crate::encoder::build_b44_block_payload(
                                channels,
                                &plane_refs,
                                *width,
                                row0,
                                lines_in_block,
                                flat,
                            );
                            if packed.len() >= raw_len {
                                // §0 raw fallback: store the interleaved
                                // native chunk uncompressed.
                                scanline_block_raw(
                                    channels,
                                    &plane_refs,
                                    *width,
                                    row0,
                                    lines_in_block,
                                )
                            } else {
                                packed
                            }
                        }
                        _ => {
                            let raw = scanline_block_raw(
                                channels,
                                &plane_refs,
                                *width,
                                row0,
                                lines_in_block,
                            );
                            compress_block(raw, *compression)?
                        }
                    };
                    all_chunks.push((part_idx as u32, ChunkPayload::Scanline { y: row0, payload }));
                }
            }
            MultipartMixedPart::Tiled {
                width,
                height,
                tile_x,
                tile_y,
                channels,
                planes,
                compression,
                ..
            } => {
                let txc = width.div_ceil(*tile_x);
                let tyc = height.div_ceil(*tile_y);
                for ty in 0..tyc {
                    for tx in 0..txc {
                        let x0 = tx * tile_x;
                        let y0 = ty * tile_y;
                        let x1 = (x0 + tile_x).min(*width);
                        let y1 = (y0 + tile_y).min(*height);
                        let tw = (x1 - x0) as usize;
                        let th = (y1 - y0) as usize;
                        let payload = compress_one_level_tile(
                            channels,
                            planes,
                            *width,
                            x0,
                            y0,
                            tw,
                            th,
                            *compression,
                        )?;
                        all_chunks.push((
                            part_idx as u32,
                            ChunkPayload::Tile {
                                tx,
                                ty,
                                lvl_x: 0,
                                lvl_y: 0,
                                payload,
                            },
                        ));
                    }
                }
            }
            MultipartMixedPart::TiledMipmap {
                tile_x,
                tile_y,
                channels,
                pyramid,
                compression,
                ..
            } => {
                // Levels outer (diagonal lvlx==lvly==level), ty-outer
                // tx-inner within each level (INCREASING_Y row-major).
                for (l, lvl) in pyramid.iter().enumerate() {
                    let lvl_idx = l as u32;
                    let txc = lvl.width.div_ceil(*tile_x);
                    let tyc = lvl.height.div_ceil(*tile_y);
                    for ty in 0..tyc {
                        for tx in 0..txc {
                            let payload = compress_tiled_level_tile(
                                lvl,
                                channels,
                                *tile_x,
                                *tile_y,
                                tx,
                                ty,
                                *compression,
                            )?;
                            all_chunks.push((
                                part_idx as u32,
                                ChunkPayload::Tile {
                                    tx,
                                    ty,
                                    lvl_x: lvl_idx,
                                    lvl_y: lvl_idx,
                                    payload,
                                },
                            ));
                        }
                    }
                }
            }
            MultipartMixedPart::TiledRipmap {
                tile_x,
                tile_y,
                channels,
                grid,
                compression,
                ..
            } => {
                // lvly outer, lvlx inner; within each cell ty-outer
                // tx-inner (INCREASING_Y row-major).
                for (ly, row) in grid.iter().enumerate() {
                    for (lx, lvl) in row.iter().enumerate() {
                        let txc = lvl.width.div_ceil(*tile_x);
                        let tyc = lvl.height.div_ceil(*tile_y);
                        for ty in 0..tyc {
                            for tx in 0..txc {
                                let payload = compress_tiled_level_tile(
                                    lvl,
                                    channels,
                                    *tile_x,
                                    *tile_y,
                                    tx,
                                    ty,
                                    *compression,
                                )?;
                                all_chunks.push((
                                    part_idx as u32,
                                    ChunkPayload::Tile {
                                        tx,
                                        ty,
                                        lvl_x: lx as u32,
                                        lvl_y: ly as u32,
                                        payload,
                                    },
                                ));
                            }
                        }
                    }
                }
            }
            MultipartMixedPart::DeepScanline {
                width,
                height,
                channels,
                samples_per_pixel,
                channel_samples,
                compression,
                ..
            } => {
                let block_h = compression.scanlines_per_block();
                let cc = chunk_counts[part_idx] as usize;
                let w = *width as usize;
                // Cumulative-EXCLUSIVE per-row sample offsets, for slicing
                // each channel's samples by row range.
                let mut row_sample_starts: Vec<u64> = Vec::with_capacity(*height as usize + 1);
                row_sample_starts.push(0);
                for r in 0..*height as usize {
                    let row_sum: u64 = samples_per_pixel[r * w..(r + 1) * w]
                        .iter()
                        .map(|&n| n as u64)
                        .sum();
                    let last = *row_sample_starts.last().unwrap();
                    row_sample_starts.push(last + row_sum);
                }
                for block_idx in 0..cc {
                    let row0 = block_idx as u32 * block_h;
                    let rows_in_block = (*height - row0).min(block_h) as usize;
                    // Per-row cumulative-inclusive offset table.
                    let mut table_bytes = Vec::with_capacity(rows_in_block * w * 4);
                    for r in 0..rows_in_block {
                        let dst_row = row0 as usize + r;
                        let cumulative = cumulative_inclusive(
                            &samples_per_pixel[dst_row * w..(dst_row + 1) * w],
                        );
                        for c in cumulative {
                            table_bytes.extend_from_slice(&c.to_le_bytes());
                        }
                    }
                    // Channel-major sample bytes for this block.
                    let s0 = row_sample_starts[row0 as usize] as usize;
                    let s1 = row_sample_starts[row0 as usize + rows_in_block] as usize;
                    let mut sample_bytes: Vec<u8> = Vec::new();
                    for (ch_idx, ch) in channels.iter().enumerate() {
                        for &v in &channel_samples[ch_idx][s0..s1] {
                            push_pixel(&mut sample_bytes, v, ch.pixel_type);
                        }
                    }
                    let packed_table = compress_buffer(&table_bytes, *compression)?;
                    let packed_data = compress_buffer(&sample_bytes, *compression)?;
                    all_chunks.push((
                        part_idx as u32,
                        ChunkPayload::DeepScanline {
                            y: row0,
                            packed_table,
                            packed_data,
                            unpacked_len: sample_bytes.len() as u64,
                        },
                    ));
                }
            }
            MultipartMixedPart::DeepTiled {
                name,
                width,
                height,
                tile_x,
                tile_y,
                channels,
                samples_per_pixel,
                channel_samples,
                compression,
            } => {
                let txc = width.div_ceil(*tile_x);
                let tyc = height.div_ceil(*tile_y);
                let starts = pixel_sample_starts(samples_per_pixel);
                for ty in 0..tyc {
                    for tx in 0..txc {
                        let (packed_table, packed_data, unpacked_len) = build_deep_tile_payload(
                            name,
                            *width,
                            *height,
                            *tile_x,
                            *tile_y,
                            tx,
                            ty,
                            channels,
                            samples_per_pixel,
                            channel_samples,
                            &starts,
                            *compression,
                        )?;
                        all_chunks.push((
                            part_idx as u32,
                            ChunkPayload::DeepTile {
                                tx,
                                ty,
                                lvl_x: 0,
                                lvl_y: 0,
                                packed_table,
                                packed_data,
                                unpacked_len,
                            },
                        ));
                    }
                }
            }
            MultipartMixedPart::DeepTiledMipmap {
                name,
                tile_x,
                tile_y,
                channels,
                pyramid,
                compression,
            } => {
                // Levels outer (diagonal lvlx==lvly==level), ty-outer
                // tx-inner within each level (INCREASING_Y row-major).
                for (l, lvl) in pyramid.iter().enumerate() {
                    let lvl_idx = l as u32;
                    let starts = pixel_sample_starts(lvl.samples_per_pixel);
                    let txc = lvl.width.div_ceil(*tile_x);
                    let tyc = lvl.height.div_ceil(*tile_y);
                    for ty in 0..tyc {
                        for tx in 0..txc {
                            let (packed_table, packed_data, unpacked_len) =
                                build_deep_tile_payload(
                                    name,
                                    lvl.width,
                                    lvl.height,
                                    *tile_x,
                                    *tile_y,
                                    tx,
                                    ty,
                                    channels,
                                    lvl.samples_per_pixel,
                                    &lvl.channel_samples,
                                    &starts,
                                    *compression,
                                )?;
                            all_chunks.push((
                                part_idx as u32,
                                ChunkPayload::DeepTile {
                                    tx,
                                    ty,
                                    lvl_x: lvl_idx,
                                    lvl_y: lvl_idx,
                                    packed_table,
                                    packed_data,
                                    unpacked_len,
                                },
                            ));
                        }
                    }
                }
            }
            MultipartMixedPart::DeepTiledRipmap {
                name,
                tile_x,
                tile_y,
                channels,
                grid,
                compression,
            } => {
                // lvly outer, lvlx inner; within each cell ty-outer
                // tx-inner (INCREASING_Y row-major).
                for (ly, row) in grid.iter().enumerate() {
                    for (lx, lvl) in row.iter().enumerate() {
                        let starts = pixel_sample_starts(lvl.samples_per_pixel);
                        let txc = lvl.width.div_ceil(*tile_x);
                        let tyc = lvl.height.div_ceil(*tile_y);
                        for ty in 0..tyc {
                            for tx in 0..txc {
                                let (packed_table, packed_data, unpacked_len) =
                                    build_deep_tile_payload(
                                        name,
                                        lvl.width,
                                        lvl.height,
                                        *tile_x,
                                        *tile_y,
                                        tx,
                                        ty,
                                        channels,
                                        lvl.samples_per_pixel,
                                        &lvl.channel_samples,
                                        &starts,
                                        *compression,
                                    )?;
                                all_chunks.push((
                                    part_idx as u32,
                                    ChunkPayload::DeepTile {
                                        tx,
                                        ty,
                                        lvl_x: lx as u32,
                                        lvl_y: ly as u32,
                                        packed_table,
                                        packed_data,
                                        unpacked_len,
                                    },
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    // ---- Compute absolute chunk offsets after concatenated offset tables. ----
    let header_bytes_so_far = out.len();
    let total_chunks: usize = chunk_counts.iter().map(|&c| c as usize).sum();
    let offset_table_bytes = total_chunks * 8;
    let chunks_start = header_bytes_so_far + offset_table_bytes;

    let mut per_part_table: Vec<Vec<u64>> = vec![Vec::new(); parts.len()];
    let mut running = chunks_start;
    for (pi, payload) in &all_chunks {
        per_part_table[*pi as usize].push(running as u64);
        // Per-chunk on-disk size:
        //   scanline:      i32 part + i32 Y + i32 size                  = 12 B + payload
        //   tiled:         i32 part + 4×i32 coords + i32 size           = 24 B + payload
        //   deep scanline: i32 part + i32 Y + 3×u64 sizes               = 32 B + table + data
        //   deep tiled:    i32 part + 4×i32 coords + 3×u64 sizes        = 44 B + table + data
        running += match payload {
            ChunkPayload::Scanline { payload, .. } => 12 + payload.len(),
            ChunkPayload::Tile { payload, .. } => 24 + payload.len(),
            ChunkPayload::DeepScanline {
                packed_table,
                packed_data,
                ..
            } => 32 + packed_table.len() + packed_data.len(),
            ChunkPayload::DeepTile {
                packed_table,
                packed_data,
                ..
            } => 44 + packed_table.len() + packed_data.len(),
        };
    }

    // Concatenated offset tables, part-order.
    for table in &per_part_table {
        for &o in table {
            out.extend_from_slice(&o.to_le_bytes());
        }
    }

    // Emit chunks in the same flat order they were built.
    for (part_idx, payload) in all_chunks {
        out.extend_from_slice(&(part_idx as i32).to_le_bytes());
        match payload {
            ChunkPayload::Scanline { y, payload } => {
                out.extend_from_slice(&(y as i32).to_le_bytes());
                out.extend_from_slice(&(payload.len() as i32).to_le_bytes());
                out.extend_from_slice(&payload);
            }
            ChunkPayload::Tile {
                tx,
                ty,
                lvl_x,
                lvl_y,
                payload,
            } => {
                out.extend_from_slice(&(tx as i32).to_le_bytes());
                out.extend_from_slice(&(ty as i32).to_le_bytes());
                out.extend_from_slice(&(lvl_x as i32).to_le_bytes());
                out.extend_from_slice(&(lvl_y as i32).to_le_bytes());
                out.extend_from_slice(&(payload.len() as i32).to_le_bytes());
                out.extend_from_slice(&payload);
            }
            ChunkPayload::DeepScanline {
                y,
                packed_table,
                packed_data,
                unpacked_len,
            } => {
                out.extend_from_slice(&(y as i32).to_le_bytes());
                out.extend_from_slice(&(packed_table.len() as u64).to_le_bytes());
                out.extend_from_slice(&(packed_data.len() as u64).to_le_bytes());
                out.extend_from_slice(&unpacked_len.to_le_bytes());
                out.extend_from_slice(&packed_table);
                out.extend_from_slice(&packed_data);
            }
            ChunkPayload::DeepTile {
                tx,
                ty,
                lvl_x,
                lvl_y,
                packed_table,
                packed_data,
                unpacked_len,
            } => {
                out.extend_from_slice(&(tx as i32).to_le_bytes());
                out.extend_from_slice(&(ty as i32).to_le_bytes());
                out.extend_from_slice(&(lvl_x as i32).to_le_bytes());
                out.extend_from_slice(&(lvl_y as i32).to_le_bytes());
                out.extend_from_slice(&(packed_table.len() as u64).to_le_bytes());
                out.extend_from_slice(&(packed_data.len() as u64).to_le_bytes());
                out.extend_from_slice(&unpacked_len.to_le_bytes());
                out.extend_from_slice(&packed_table);
                out.extend_from_slice(&packed_data);
            }
        }
    }

    Ok(out)
}

/// Overflow-safe end offsets for a deep chunk's (offset-table, sample
/// data) payloads. The two packed sizes come off the wire as `u64`
/// fields; a hostile value near `u64::MAX` must yield an error, not a
/// debug-build add-overflow panic. Returns `(table_end, data_end)` only
/// when both sums fit and the payload ends within the file.
fn deep_payload_bounds(
    table_start: usize,
    packed_table: usize,
    packed_data: usize,
    file_len: usize,
    what: &str,
) -> Result<(usize, usize)> {
    let table_end = table_start.checked_add(packed_table);
    let data_end = table_end.and_then(|te| te.checked_add(packed_data));
    match (table_end, data_end) {
        (Some(te), Some(de)) if de <= file_len => Ok((te, de)),
        _ => Err(ExrError::invalid(format!("{what}: payload runs past EOF"))),
    }
}

/// Per-tile decoded deep channel samples (tile extent + one `Vec<f32>`
/// per channel, channel-major within the tile in pixel-scan order).
struct TileDecoded {
    tw: u32,
    th: u32,
    channel_samples: Vec<Vec<f32>>,
}

/// One decode-target level of a multi-level **deep** tiled part. Deep
/// tile chunks scatter into the matching `(level_x, level_y)` slot; the
/// per-pixel sample counts are accumulated as each tile is decoded and
/// the per-tile decoded slabs held until the whole level is assembled.
struct DeepTileLevelState {
    level_x: u32,
    level_y: u32,
    width: u32,
    height: u32,
    tx_count: u32,
    ty_count: u32,
    samples_per_pixel: Vec<u32>,
    /// Indexed by `ty * tx_count + tx`.
    tile_decoded: Vec<Option<TileDecoded>>,
}

/// One fully-materialised deep tiled level (pixel-scan-order sample
/// lists), an intermediate used when reshaping a decoded multi-level
/// deep tiled part into its MIPMAP / RIPMAP output shape.
struct AssembledDeepLevel {
    level_x: u32,
    level_y: u32,
    width: u32,
    height: u32,
    samples_per_pixel: Vec<u32>,
    channel_samples: Vec<Vec<f32>>,
}

/// Per-part decode state for [`parse_exr_multipart_mixed`]. One variant
/// per part `type`; flat variants scatter into [`ExrPlane`]s, deep
/// variants accumulate sample lists.
enum PartState {
    Scanline {
        req: RequiredAttrs,
        sorted_channels: Vec<Channel>,
        planes: Vec<ExrPlane>,
    },
    Tiled {
        req: RequiredAttrs,
        sorted_channels: Vec<Channel>,
        planes: Vec<ExrPlane>,
        tile_x: u32,
        tile_y: u32,
        tx_count: u32,
        ty_count: u32,
    },
    /// Multi-level (MIPMAP / RIPMAP) flat tiled part. Holds one
    /// [`TiledLevel`] slot per expected level in spec iteration order;
    /// tile chunks scatter into the matching `(lvlx, lvly)` slot.
    MultilevelTiled {
        req: RequiredAttrs,
        sorted_channels: Vec<Channel>,
        tile_x: u32,
        tile_y: u32,
        level_mode: u8,
        round_mode: u8,
        levels: Vec<TiledLevel>,
    },
    DeepScanline {
        name: String,
        req: RequiredAttrs,
        sorted_channels: Vec<Channel>,
        samples_per_pixel: Vec<u32>,
        channel_samples: Vec<Vec<f32>>,
    },
    DeepTiled {
        name: String,
        req: RequiredAttrs,
        sorted_channels: Vec<Channel>,
        tile_x: u32,
        tile_y: u32,
        tx_count: u32,
        ty_count: u32,
        samples_per_pixel: Vec<u32>,
        /// Indexed by `ty * tx_count + tx`.
        tile_decoded: Vec<Option<TileDecoded>>,
    },
    /// Multi-level (MIPMAP / RIPMAP) **deep** tiled part. Holds one
    /// [`DeepTileLevelState`] slot per expected level in spec iteration
    /// order; deep tile chunks scatter into the matching
    /// `(level_x, level_y)` slot.
    DeepMultilevelTiled {
        name: String,
        req: RequiredAttrs,
        sorted_channels: Vec<Channel>,
        tile_x: u32,
        tile_y: u32,
        level_mode: u8,
        levels: Vec<DeepTileLevelState>,
    },
}

/// Parse a multi-part EXR whose parts may freely mix
/// `type="scanlineimage"`, `type="tiledimage"` (ONE_LEVEL, MIPMAP, or
/// RIPMAP), `type="deepscanline"` and `type="deeptile"` (ONE_LEVEL).
///
/// Companion to [`encode_exr_multipart_mixed`]. Flat multi-level tiled
/// parts are decoded inline (surfaced as
/// [`MultipartMixedImage::TiledMipmap`] / `TiledRipmap`). Multi-level
/// *deep* tiled parts are still rejected with a pointer at the
/// dedicated entries — call
/// [`crate::parse_exr_multipart_deep_tiled_mipmap`] or
/// [`crate::parse_exr_multipart_deep_tiled_ripmap`] for those shapes.
///
/// Like the other multi-part readers we walk chunks by linear scan
/// rather than offset-table lookup so that zero-filled tables produced
/// by some reference flows still decode correctly. Deep scanline chunks
/// are expected in increasing-Y order per part (the order every writer
/// in this crate emits).
pub fn parse_exr_multipart_mixed(bytes: &[u8]) -> Result<Vec<MultipartMixedImage>> {
    let parts = parse_multipart_headers(bytes)?;
    if parts.is_empty() {
        return Err(ExrError::invalid(
            "mixed multi-part file has no parts".to_string(),
        ));
    }

    // Per-part chunkCount (mandatory in multi-part files).
    let mut chunk_counts: Vec<usize> = Vec::with_capacity(parts.len());
    for (i, part) in parts.iter().enumerate() {
        let cc = find_chunk_count(&part.attributes).ok_or_else(|| {
            ExrError::invalid(format!(
                "mixed multi-part part {i} missing required chunkCount attribute"
            ))
        })?;
        chunk_counts.push(cc);
    }

    // Classify each part by its declared `type` attribute and build the
    // per-part decode state. Multi-level tiled parts are rejected up
    // front.
    let mut state: Vec<PartState> = Vec::with_capacity(parts.len());
    for (part_idx, part) in parts.iter().enumerate() {
        let part_type = find_part_type(&part.attributes).ok_or_else(|| {
            ExrError::invalid(format!(
                "mixed multi-part: part {part_idx} missing required 'type' attribute"
            ))
        })?;
        let req = extract_required(&part.attributes)?;
        let width = req.data_window.width();
        let height = req.data_window.height();
        if width == 0 || height == 0 {
            return Err(ExrError::invalid(format!(
                "mixed multi-part part {part_idx}: dataWindow {width}×{height} must be > 0"
            )));
        }
        let is_deep = matches!(part_type.as_str(), "deepscanline" | "deeptile");
        if is_deep {
            if !matches!(
                req.compression,
                Compression::None | Compression::Rle | Compression::Zips
            ) {
                return Err(ExrError::invalid(format!(
                    "mixed multi-part deep part {part_idx}: compression {:?} \
                     (deep parts accept only NONE/RLE/ZIPS)",
                    req.compression
                )));
            }
        } else if !matches!(
            req.compression,
            Compression::None
                | Compression::Zip
                | Compression::Zips
                | Compression::Rle
                | Compression::Pxr24
                | Compression::B44
                | Compression::B44a
        ) {
            // PXR24 / B44 / B44A are accepted for flat scanline + ONE_LEVEL
            // tiled parts; the multi-level tiled reader below rejects them
            // separately (its per-level tile decoder is NONE/ZIP/ZIPS/RLE).
            return Err(ExrError::unsupported(format!(
                "mixed multi-part part {part_idx}: compression {:?} not yet implemented",
                req.compression
            )));
        }
        let mut sorted_channels = req.channels.clone();
        sorted_channels.sort_by(|a, b| a.name.cmp(&b.name));

        // Deep parts require 1×1 channel sampling.
        let check_deep_sampling = || -> Result<()> {
            for ch in &sorted_channels {
                if ch.x_sampling != 1 || ch.y_sampling != 1 {
                    return Err(ExrError::unsupported(format!(
                        "mixed multi-part deep part {part_idx}: sub-sampled channel \
                         '{}' (deep parts require 1×1 sampling)",
                        ch.name
                    )));
                }
            }
            Ok(())
        };
        // The `name` attribute (mandatory in multi-part files) is
        // surfaced on the deep payload structs.
        let part_name = || -> Result<String> {
            find_string_attr(&part.attributes, "name").ok_or_else(|| {
                ExrError::invalid(format!(
                    "mixed multi-part part {part_idx} missing required 'name' attribute"
                ))
            })
        };

        match part_type.as_str() {
            "scanlineimage" => {
                // Scanline parts accept sub-sampled channels, but a
                // sampling factor of zero off the wire would divide by
                // zero in the row/size math — reject it here.
                for ch in &sorted_channels {
                    if ch.x_sampling == 0 || ch.y_sampling == 0 {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part scanline part {part_idx}: channel '{}' \
                             sampling {}×{} (both factors must be >= 1)",
                            ch.name, ch.x_sampling, ch.y_sampling
                        )));
                    }
                }
                let planes = make_flat_planes(&sorted_channels, width, height);
                state.push(PartState::Scanline {
                    req,
                    sorted_channels,
                    planes,
                });
            }
            "tiledimage" => {
                for ch in &sorted_channels {
                    if ch.x_sampling != 1 || ch.y_sampling != 1 {
                        return Err(ExrError::unsupported(format!(
                            "mixed multi-part tiled part {part_idx}: sub-sampled channel \
                             '{}' (tiled parts require 1×1 sampling)",
                            ch.name
                        )));
                    }
                }
                // Inspect the tiledesc level mode directly so flat
                // multi-level (MIPMAP / RIPMAP) parts are decoded inline
                // rather than rejected.
                let tdesc_attr = part
                    .attributes
                    .iter()
                    .find(|a| a.name == "tiles")
                    .ok_or_else(|| {
                        ExrError::invalid(format!(
                            "mixed multi-part tiled part {part_idx} missing required \
                             'tiles' attribute"
                        ))
                    })?;
                let tdesc = tiledesc_from_attribute(&tdesc_attr.value)?;
                if tdesc.x_size == 0 || tdesc.y_size == 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part tiled part {part_idx}: tile size {}×{} \
                         must both be > 0",
                        tdesc.x_size, tdesc.y_size
                    )));
                }
                if tdesc.level_mode == 0 {
                    let txc = width.div_ceil(tdesc.x_size);
                    let tyc = height.div_ceil(tdesc.y_size);
                    let expected = (txc as usize) * (tyc as usize);
                    if chunk_counts[part_idx] != expected {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part tiled part {part_idx}: chunkCount={} but \
                             tile grid {txc}×{tyc} expects {expected}",
                            chunk_counts[part_idx]
                        )));
                    }
                    let planes = make_flat_planes(&sorted_channels, width, height);
                    state.push(PartState::Tiled {
                        req,
                        sorted_channels,
                        planes,
                        tile_x: tdesc.x_size,
                        tile_y: tdesc.y_size,
                        tx_count: txc,
                        ty_count: tyc,
                    });
                } else if tdesc.level_mode <= 2 {
                    // MIPMAP (1) or RIPMAP (2): enumerate the expected
                    // levels in spec iteration order and allocate planes.
                    // The per-level tile decoder is the shared
                    // `scatter_tile_into_planes`, which handles
                    // NONE/ZIP/ZIPS/RLE/PXR24/B44/B44A.
                    let round_up = tdesc.round_mode != 0;
                    let levels = enumerate_tiled_levels(
                        tdesc.level_mode,
                        width,
                        height,
                        round_up,
                        &sorted_channels,
                    );
                    let expected: usize = levels
                        .iter()
                        .map(|l| {
                            l.width.div_ceil(tdesc.x_size) as usize
                                * l.height.div_ceil(tdesc.y_size) as usize
                        })
                        .sum();
                    if chunk_counts[part_idx] != expected {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part multi-level tiled part {part_idx}: \
                             chunkCount={} but level grid expects {expected}",
                            chunk_counts[part_idx]
                        )));
                    }
                    state.push(PartState::MultilevelTiled {
                        req,
                        sorted_channels,
                        tile_x: tdesc.x_size,
                        tile_y: tdesc.y_size,
                        level_mode: tdesc.level_mode,
                        round_mode: tdesc.round_mode,
                        levels,
                    });
                } else {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part tiled part {part_idx}: tiledesc level_mode={} \
                         unknown (expected 0/1/2)",
                        tdesc.level_mode
                    )));
                }
            }
            "deepscanline" => {
                check_deep_sampling()?;
                let block_h = req.compression.scanlines_per_block();
                let expected = height.div_ceil(block_h) as usize;
                if chunk_counts[part_idx] != expected {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part deep scanline part {part_idx}: \
                         chunkCount={} disagrees with height/block math ({expected})",
                        chunk_counts[part_idx]
                    )));
                }
                let pixels = (width as usize) * (height as usize);
                let n_channels = sorted_channels.len();
                state.push(PartState::DeepScanline {
                    name: part_name()?,
                    req,
                    sorted_channels,
                    samples_per_pixel: vec![0u32; pixels],
                    channel_samples: (0..n_channels).map(|_| Vec::new()).collect(),
                });
            }
            "deeptile" => {
                check_deep_sampling()?;
                // Inspect the tiledesc level mode directly so multi-level
                // (MIPMAP / RIPMAP) deep tiled parts are decoded inline
                // rather than rejected.
                let tdesc_attr = part
                    .attributes
                    .iter()
                    .find(|a| a.name == "tiles")
                    .ok_or_else(|| {
                        ExrError::invalid(format!(
                            "mixed multi-part deep tiled part {part_idx} missing required \
                             'tiles' attribute"
                        ))
                    })?;
                let tdesc = tiledesc_from_attribute(&tdesc_attr.value)?;
                if tdesc.x_size == 0 || tdesc.y_size == 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part deep tiled part {part_idx}: tile size {}×{} \
                         must both be > 0",
                        tdesc.x_size, tdesc.y_size
                    )));
                }
                if tdesc.level_mode == 0 {
                    let tx_count = width.div_ceil(tdesc.x_size);
                    let ty_count = height.div_ceil(tdesc.y_size);
                    let expected = (tx_count as usize) * (ty_count as usize);
                    if chunk_counts[part_idx] != expected {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part deep tiled part {part_idx}: chunkCount={} but \
                             tile grid {tx_count}×{ty_count} expects {expected}",
                            chunk_counts[part_idx]
                        )));
                    }
                    let pixels = (width as usize) * (height as usize);
                    state.push(PartState::DeepTiled {
                        name: part_name()?,
                        req,
                        sorted_channels,
                        tile_x: tdesc.x_size,
                        tile_y: tdesc.y_size,
                        tx_count,
                        ty_count,
                        samples_per_pixel: vec![0u32; pixels],
                        tile_decoded: (0..chunk_counts[part_idx]).map(|_| None).collect(),
                    });
                } else if tdesc.level_mode <= 2 {
                    // MIPMAP (1) or RIPMAP (2): enumerate the expected deep
                    // levels in spec iteration order and allocate per-level
                    // tile grids.
                    let round_up = tdesc.round_mode != 0;
                    let levels = enumerate_deep_tiled_levels(
                        tdesc.level_mode,
                        width,
                        height,
                        round_up,
                        tdesc.x_size,
                        tdesc.y_size,
                    );
                    let expected: usize = levels
                        .iter()
                        .map(|l| (l.tx_count as usize) * (l.ty_count as usize))
                        .sum();
                    if chunk_counts[part_idx] != expected {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part multi-level deep tiled part {part_idx}: \
                             chunkCount={} but level grid expects {expected}",
                            chunk_counts[part_idx]
                        )));
                    }
                    state.push(PartState::DeepMultilevelTiled {
                        name: part_name()?,
                        req,
                        sorted_channels,
                        tile_x: tdesc.x_size,
                        tile_y: tdesc.y_size,
                        level_mode: tdesc.level_mode,
                        levels,
                    });
                } else {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part deep tiled part {part_idx}: tiledesc level_mode={} \
                         unknown (expected 0/1/2)",
                        tdesc.level_mode
                    )));
                }
            }
            other => {
                return Err(ExrError::unsupported(format!(
                    "mixed multi-part part {part_idx} type='{other}' (only \
                     'scanlineimage', 'tiledimage', 'deepscanline', and \
                     'deeptile' supported)"
                )));
            }
        }
    }

    // Linear chunk scan — dispatch chunk shape on the part's declared kind.
    let total_chunks: usize = chunk_counts.iter().sum();
    let tables_start = parts.last().unwrap().end_offset;
    let chunk_scan_start = tables_start + total_chunks * 8;
    if chunk_scan_start > bytes.len() {
        return Err(ExrError::invalid(format!(
            "mixed multi-part offset tables run past EOF (need {}, have {})",
            chunk_scan_start,
            bytes.len()
        )));
    }
    let mut scan_pos = chunk_scan_start;
    for _chunk_global_idx in 0..total_chunks {
        if scan_pos + 4 > bytes.len() {
            return Err(ExrError::invalid(format!(
                "mixed multi-part: unexpected EOF at chunk scan position {scan_pos}"
            )));
        }
        let part_num = i32::from_le_bytes(bytes[scan_pos..scan_pos + 4].try_into().unwrap());
        if part_num < 0 || part_num as usize >= parts.len() {
            return Err(ExrError::invalid(format!(
                "mixed multi-part chunk at {scan_pos}: part_number={part_num} out of \
                 range 0..{}",
                parts.len()
            )));
        }
        let part_idx = part_num as usize;
        match &mut state[part_idx] {
            PartState::Scanline {
                req,
                sorted_channels,
                planes,
            } => {
                if scan_pos + 12 > bytes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part scanline chunk at {scan_pos}: header runs past EOF"
                    )));
                }
                let y_coord =
                    i32::from_le_bytes(bytes[scan_pos + 4..scan_pos + 8].try_into().unwrap());
                let payload_size =
                    i32::from_le_bytes(bytes[scan_pos + 8..scan_pos + 12].try_into().unwrap());
                if payload_size < 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part scanline chunk at {scan_pos}: \
                         negative payload size {payload_size}"
                    )));
                }
                let pl_start = scan_pos + 12;
                let pl_end = pl_start + payload_size as usize;
                if pl_end > bytes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part scanline chunk at {scan_pos}: payload runs past EOF"
                    )));
                }
                let width = req.data_window.width();
                let height = req.data_window.height();
                let row_in_image = (y_coord - req.data_window.y_min) as i64;
                if row_in_image < 0 || row_in_image as u32 >= height {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part scanline part {part_idx} chunk Y={y_coord} \
                         outside dataWindow"
                    )));
                }
                let block_y0 = row_in_image as u32;
                let block_h = req.compression.scanlines_per_block();
                let lines_in_block = ((height - block_y0).min(block_h)) as usize;
                let uncompressed_size: usize = sorted_channels
                    .iter()
                    .map(|ch| {
                        let ys = ch.y_sampling as u32;
                        let lines = (0..lines_in_block as u32)
                            .filter(|&l| (block_y0 + l) % ys == 0)
                            .count();
                        let xs = ch.x_sampling as u32;
                        let pw = subsampled_dim(width, xs) as usize;
                        lines * ch.pixel_type.bytes_per_sample() * pw
                    })
                    .sum();
                let payload = &bytes[pl_start..pl_end];
                match req.compression {
                    Compression::B44 | Compression::B44a => {
                        // B44 regroups the chunk into per-channel planes and
                        // scatters directly (observer-spec §2).
                        scatter_b44_block_into_planes(
                            payload,
                            sorted_channels,
                            planes,
                            width,
                            block_y0,
                            lines_in_block,
                            uncompressed_size,
                        )?;
                    }
                    Compression::Pxr24 => {
                        // PXR24 inflates + reverses byte-plane delta to the
                        // interleaved native stream (or the raw fallback),
                        // then scatters like the other interleaved schemes.
                        let uncompressed = decode_pxr24_payload(
                            payload,
                            &Pxr24RowSpec {
                                sorted_channels,
                                width,
                                block_y0,
                                lines_in_block,
                            },
                            uncompressed_size,
                        )?;
                        scatter_scanline_block_into_planes(
                            &uncompressed,
                            sorted_channels,
                            planes,
                            width,
                            block_y0,
                            lines_in_block,
                        )?;
                    }
                    _ => {
                        let uncompressed =
                            decompress_block(payload, uncompressed_size, req.compression)?;
                        scatter_scanline_block_into_planes(
                            &uncompressed,
                            sorted_channels,
                            planes,
                            width,
                            block_y0,
                            lines_in_block,
                        )?;
                    }
                }
                scan_pos = pl_end;
            }
            PartState::Tiled {
                req,
                sorted_channels,
                planes,
                tile_x,
                tile_y,
                tx_count,
                ty_count,
            } => {
                if scan_pos + 24 > bytes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part tiled chunk at {scan_pos}: header runs past EOF"
                    )));
                }
                let h_tx =
                    i32::from_le_bytes(bytes[scan_pos + 4..scan_pos + 8].try_into().unwrap());
                let h_ty =
                    i32::from_le_bytes(bytes[scan_pos + 8..scan_pos + 12].try_into().unwrap());
                let lvl_x =
                    i32::from_le_bytes(bytes[scan_pos + 12..scan_pos + 16].try_into().unwrap());
                let lvl_y =
                    i32::from_le_bytes(bytes[scan_pos + 16..scan_pos + 20].try_into().unwrap());
                let payload_size =
                    i32::from_le_bytes(bytes[scan_pos + 20..scan_pos + 24].try_into().unwrap());
                if payload_size < 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part tiled chunk at {scan_pos}: \
                         negative payload size {payload_size}"
                    )));
                }
                if lvl_x != 0 || lvl_y != 0 {
                    return Err(ExrError::unsupported(format!(
                        "mixed multi-part tiled chunk at {scan_pos}: lvlx={lvl_x} \
                         lvly={lvl_y} (parse_exr_multipart_mixed is ONE_LEVEL only)"
                    )));
                }
                let pl_start = scan_pos + 24;
                let pl_end = pl_start + payload_size as usize;
                if pl_end > bytes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part tiled chunk at {scan_pos}: payload runs past EOF"
                    )));
                }
                let width = req.data_window.width();
                let height = req.data_window.height();
                let tx = h_tx as u32;
                let ty = h_ty as u32;
                if h_tx < 0 || h_ty < 0 || tx >= *tx_count || ty >= *ty_count {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part tiled chunk at {scan_pos}: tile ({h_tx},{h_ty}) \
                         out of grid {tx_count}×{ty_count}"
                    )));
                }
                let x0 = tx * *tile_x;
                let y0 = ty * *tile_y;
                let x1 = (x0 + *tile_x).min(width);
                let y1 = (y0 + *tile_y).min(height);
                let tw = (x1 - x0) as usize;
                let th = (y1 - y0) as usize;
                let payload = &bytes[pl_start..pl_end];
                scatter_tile_into_planes(
                    payload,
                    sorted_channels,
                    planes,
                    width,
                    x0,
                    y0,
                    tw,
                    th,
                    req.compression,
                    (ty * *tx_count + tx) as usize,
                )?;
                scan_pos = pl_end;
            }
            PartState::MultilevelTiled {
                req,
                sorted_channels,
                tile_x,
                tile_y,
                levels,
                ..
            } => {
                // 24-byte tiled chunk header: tx, ty, lvlx, lvly, size.
                if scan_pos + 24 > bytes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part multi-level tiled chunk at {scan_pos}: \
                         header runs past EOF"
                    )));
                }
                let h_tx =
                    i32::from_le_bytes(bytes[scan_pos + 4..scan_pos + 8].try_into().unwrap());
                let h_ty =
                    i32::from_le_bytes(bytes[scan_pos + 8..scan_pos + 12].try_into().unwrap());
                let lvl_x =
                    i32::from_le_bytes(bytes[scan_pos + 12..scan_pos + 16].try_into().unwrap());
                let lvl_y =
                    i32::from_le_bytes(bytes[scan_pos + 16..scan_pos + 20].try_into().unwrap());
                let payload_size =
                    i32::from_le_bytes(bytes[scan_pos + 20..scan_pos + 24].try_into().unwrap());
                if payload_size < 0 || h_tx < 0 || h_ty < 0 || lvl_x < 0 || lvl_y < 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part multi-level tiled chunk at {scan_pos}: bad header \
                         tx={h_tx} ty={h_ty} lvlx={lvl_x} lvly={lvl_y} size={payload_size}"
                    )));
                }
                let pl_start = scan_pos + 24;
                let pl_end = pl_start + payload_size as usize;
                if pl_end > bytes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part multi-level tiled chunk at {scan_pos}: \
                         payload runs past EOF"
                    )));
                }
                let level = levels
                    .iter_mut()
                    .find(|l| l.level_x as i32 == lvl_x && l.level_y as i32 == lvl_y)
                    .ok_or_else(|| {
                        ExrError::invalid(format!(
                            "mixed multi-part multi-level tiled chunk at {scan_pos}: \
                             unknown level ({lvl_x},{lvl_y}) on part {part_idx}"
                        ))
                    })?;
                let tx = h_tx as u32;
                let ty = h_ty as u32;
                let x0 = tx * *tile_x;
                let y0 = ty * *tile_y;
                if x0 >= level.width || y0 >= level.height {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part multi-level tiled chunk at {scan_pos}: tile \
                         ({tx},{ty}) outside level ({lvl_x},{lvl_y}) dims {}×{}",
                        level.width, level.height
                    )));
                }
                let x1 = (x0 + *tile_x).min(level.width);
                let y1 = (y0 + *tile_y).min(level.height);
                let tw = (x1 - x0) as usize;
                let th = (y1 - y0) as usize;
                let payload = &bytes[pl_start..pl_end];
                scatter_tile_into_planes(
                    payload,
                    sorted_channels,
                    &mut level.planes,
                    level.width,
                    x0,
                    y0,
                    tw,
                    th,
                    req.compression,
                    0,
                )?;
                scan_pos = pl_end;
            }
            PartState::DeepScanline {
                name,
                req,
                sorted_channels,
                samples_per_pixel,
                channel_samples,
            } => {
                // i32 part + i32 Y + 3×u64 sizes = 32 bytes of header.
                if scan_pos + 32 > bytes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part deep scanline chunk at {scan_pos}: \
                         header runs past EOF"
                    )));
                }
                let y_coord =
                    i32::from_le_bytes(bytes[scan_pos + 4..scan_pos + 8].try_into().unwrap());
                let packed_table =
                    u64::from_le_bytes(bytes[scan_pos + 8..scan_pos + 16].try_into().unwrap())
                        as usize;
                let packed_data =
                    u64::from_le_bytes(bytes[scan_pos + 16..scan_pos + 24].try_into().unwrap())
                        as usize;
                let unpacked_data =
                    u64::from_le_bytes(bytes[scan_pos + 24..scan_pos + 32].try_into().unwrap())
                        as usize;
                let table_start = scan_pos + 32;
                let (table_end, data_end) = deep_payload_bounds(
                    table_start,
                    packed_table,
                    packed_data,
                    bytes.len(),
                    &format!("mixed multi-part deep scanline chunk at {scan_pos}"),
                )?;
                let width = req.data_window.width();
                let height = req.data_window.height();
                let row_in_image = (y_coord - req.data_window.y_min) as i64;
                if row_in_image < 0 || row_in_image as u32 >= height {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part deep scanline part {part_idx} ('{name}'): \
                         chunk Y={y_coord} outside dataWindow"
                    )));
                }
                let block_y0 = row_in_image as u32;
                let block_h = req.compression.scanlines_per_block();
                let rows_in_block = ((height - block_y0).min(block_h)) as usize;
                let entries = rows_in_block * width as usize;

                let table_bytes = decompress_buffer(
                    &bytes[table_start..table_end],
                    entries * 4,
                    req.compression,
                )?;
                let mut cumulative_flat: Vec<i32> = Vec::with_capacity(entries);
                for ch in table_bytes.chunks_exact(4) {
                    cumulative_flat.push(i32::from_le_bytes(ch.try_into().unwrap()));
                }
                let mut block_samples_total: u64 = 0;
                for r in 0..rows_in_block {
                    let row_slice = &cumulative_flat[r * width as usize..(r + 1) * width as usize];
                    let per_pixel = per_pixel_from_cumulative(row_slice)?;
                    let dst_base = (block_y0 as usize + r) * width as usize;
                    for (i, &n) in per_pixel.iter().enumerate() {
                        samples_per_pixel[dst_base + i] = n;
                        block_samples_total += n as u64;
                    }
                }
                let block_bpp: usize = sorted_channels
                    .iter()
                    .map(|c| c.pixel_type.bytes_per_sample())
                    .sum();
                let expected_unpacked = block_samples_total as usize * block_bpp;
                if expected_unpacked != unpacked_data {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part deep scanline part {part_idx} ('{name}'): \
                         derived unpacked_data={expected_unpacked} disagrees with \
                         header unpacked_data={unpacked_data}"
                    )));
                }
                let sample_bytes =
                    decompress_buffer(&bytes[table_end..data_end], unpacked_data, req.compression)?;
                decode_deep_sample_block(
                    &sample_bytes,
                    sorted_channels,
                    block_samples_total as usize,
                    channel_samples,
                )
                .map_err(|e| {
                    ExrError::invalid(format!(
                        "mixed multi-part deep scanline part {part_idx} ('{name}'): {e}"
                    ))
                })?;
                scan_pos = data_end;
            }
            PartState::DeepTiled {
                name,
                req,
                sorted_channels,
                tile_x,
                tile_y,
                tx_count,
                ty_count,
                samples_per_pixel,
                tile_decoded,
            } => {
                // i32 part + 4×i32 coords + 3×u64 sizes = 44 bytes of header.
                if scan_pos + 44 > bytes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part deep tiled chunk at {scan_pos}: \
                         header runs past EOF"
                    )));
                }
                let h_tx =
                    i32::from_le_bytes(bytes[scan_pos + 4..scan_pos + 8].try_into().unwrap());
                let h_ty =
                    i32::from_le_bytes(bytes[scan_pos + 8..scan_pos + 12].try_into().unwrap());
                let lvl_x =
                    i32::from_le_bytes(bytes[scan_pos + 12..scan_pos + 16].try_into().unwrap());
                let lvl_y =
                    i32::from_le_bytes(bytes[scan_pos + 16..scan_pos + 20].try_into().unwrap());
                let packed_table =
                    u64::from_le_bytes(bytes[scan_pos + 20..scan_pos + 28].try_into().unwrap())
                        as usize;
                let packed_data =
                    u64::from_le_bytes(bytes[scan_pos + 28..scan_pos + 36].try_into().unwrap())
                        as usize;
                let unpacked_data =
                    u64::from_le_bytes(bytes[scan_pos + 36..scan_pos + 44].try_into().unwrap())
                        as usize;
                if lvl_x != 0 || lvl_y != 0 {
                    return Err(ExrError::unsupported(format!(
                        "mixed multi-part deep tiled chunk at {scan_pos}: lvlx={lvl_x} \
                         lvly={lvl_y} (parse_exr_multipart_mixed is ONE_LEVEL only)"
                    )));
                }
                if h_tx < 0 || h_ty < 0 || (h_tx as u32) >= *tx_count || (h_ty as u32) >= *ty_count
                {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part deep tiled part {part_idx} ('{name}'): \
                         tx={h_tx} ty={h_ty} outside grid {tx_count}×{ty_count}"
                    )));
                }
                let tx = h_tx as u32;
                let ty = h_ty as u32;
                let width = req.data_window.width();
                let height = req.data_window.height();
                let x0 = tx * *tile_x;
                let y0 = ty * *tile_y;
                let x1 = (x0 + *tile_x).min(width);
                let y1 = (y0 + *tile_y).min(height);
                let tw = x1 - x0;
                let th = y1 - y0;
                let part_label = format!("mixed multi-part deep tiled part {part_idx} ('{name}')");
                let table_start = scan_pos + 44;
                let (table_end, data_end) = deep_payload_bounds(
                    table_start,
                    packed_table,
                    packed_data,
                    bytes.len(),
                    &part_label,
                )?;
                let decoded = decode_deep_tile_body(
                    &bytes[table_start..table_end],
                    &bytes[table_end..data_end],
                    unpacked_data,
                    width,
                    *tile_x,
                    *tile_y,
                    tx,
                    ty,
                    tw,
                    th,
                    sorted_channels,
                    req.compression,
                    &part_label,
                    samples_per_pixel,
                )?;
                let tile_grid_idx = (ty * *tx_count + tx) as usize;
                if tile_decoded[tile_grid_idx].is_some() {
                    return Err(ExrError::invalid(format!(
                        "{part_label}: tile ({tx},{ty}) appears more than once"
                    )));
                }
                tile_decoded[tile_grid_idx] = Some(decoded);
                scan_pos = data_end;
            }
            PartState::DeepMultilevelTiled {
                name,
                req,
                sorted_channels,
                tile_x,
                tile_y,
                levels,
                ..
            } => {
                // i32 part + 4×i32 coords + 3×u64 sizes = 44 bytes of header.
                if scan_pos + 44 > bytes.len() {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part multi-level deep tiled chunk at {scan_pos}: \
                         header runs past EOF"
                    )));
                }
                let h_tx =
                    i32::from_le_bytes(bytes[scan_pos + 4..scan_pos + 8].try_into().unwrap());
                let h_ty =
                    i32::from_le_bytes(bytes[scan_pos + 8..scan_pos + 12].try_into().unwrap());
                let lvl_x =
                    i32::from_le_bytes(bytes[scan_pos + 12..scan_pos + 16].try_into().unwrap());
                let lvl_y =
                    i32::from_le_bytes(bytes[scan_pos + 16..scan_pos + 20].try_into().unwrap());
                let packed_table =
                    u64::from_le_bytes(bytes[scan_pos + 20..scan_pos + 28].try_into().unwrap())
                        as usize;
                let packed_data =
                    u64::from_le_bytes(bytes[scan_pos + 28..scan_pos + 36].try_into().unwrap())
                        as usize;
                let unpacked_data =
                    u64::from_le_bytes(bytes[scan_pos + 36..scan_pos + 44].try_into().unwrap())
                        as usize;
                if h_tx < 0 || h_ty < 0 || lvl_x < 0 || lvl_y < 0 {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part multi-level deep tiled part {part_idx} ('{name}'): \
                         bad header tx={h_tx} ty={h_ty} lvlx={lvl_x} lvly={lvl_y}"
                    )));
                }
                let level = levels
                    .iter_mut()
                    .find(|l| l.level_x as i32 == lvl_x && l.level_y as i32 == lvl_y)
                    .ok_or_else(|| {
                        ExrError::invalid(format!(
                            "mixed multi-part multi-level deep tiled part {part_idx} \
                             ('{name}'): unknown level ({lvl_x},{lvl_y})"
                        ))
                    })?;
                let tx = h_tx as u32;
                let ty = h_ty as u32;
                if tx >= level.tx_count || ty >= level.ty_count {
                    return Err(ExrError::invalid(format!(
                        "mixed multi-part multi-level deep tiled part {part_idx} ('{name}'): \
                         tile ({tx},{ty}) outside level ({lvl_x},{lvl_y}) grid \
                         {}×{}",
                        level.tx_count, level.ty_count
                    )));
                }
                let x0 = tx * *tile_x;
                let y0 = ty * *tile_y;
                let x1 = (x0 + *tile_x).min(level.width);
                let y1 = (y0 + *tile_y).min(level.height);
                let tw = x1 - x0;
                let th = y1 - y0;
                let part_label = format!(
                    "mixed multi-part multi-level deep tiled part {part_idx} ('{name}') \
                     level ({lvl_x},{lvl_y})"
                );
                let table_start = scan_pos + 44;
                let (table_end, data_end) = deep_payload_bounds(
                    table_start,
                    packed_table,
                    packed_data,
                    bytes.len(),
                    &part_label,
                )?;
                let decoded = decode_deep_tile_body(
                    &bytes[table_start..table_end],
                    &bytes[table_end..data_end],
                    unpacked_data,
                    level.width,
                    *tile_x,
                    *tile_y,
                    tx,
                    ty,
                    tw,
                    th,
                    sorted_channels,
                    req.compression,
                    &part_label,
                    &mut level.samples_per_pixel,
                )?;
                let tile_grid_idx = (ty * level.tx_count + tx) as usize;
                if level.tile_decoded[tile_grid_idx].is_some() {
                    return Err(ExrError::invalid(format!(
                        "{part_label}: tile ({tx},{ty}) appears more than once"
                    )));
                }
                level.tile_decoded[tile_grid_idx] = Some(decoded);
                scan_pos = data_end;
            }
        }
    }

    // Assemble per-part outputs.
    let mut images: Vec<MultipartMixedImage> = Vec::with_capacity(parts.len());
    for part in parts.iter() {
        let ps = state.remove(0);
        images.push(match ps {
            PartState::Scanline {
                req,
                sorted_channels,
                planes,
            } => MultipartMixedImage::Scanline(make_exr_image(
                req,
                sorted_channels,
                planes,
                part.attributes.clone(),
            )),
            PartState::Tiled {
                req,
                sorted_channels,
                planes,
                ..
            } => MultipartMixedImage::Tiled(make_exr_image(
                req,
                sorted_channels,
                planes,
                part.attributes.clone(),
            )),
            PartState::MultilevelTiled {
                req,
                sorted_channels,
                tile_x,
                tile_y,
                level_mode,
                round_mode,
                levels,
            } => {
                let mlt = MultilevelTiledPart {
                    level_mode,
                    round_mode,
                    tile_x,
                    tile_y,
                    data_window: req.data_window,
                    display_window: req.display_window,
                    channels: sorted_channels,
                    compression: req.compression,
                    levels,
                    attributes: part.attributes.clone(),
                };
                if level_mode == 2 {
                    MultipartMixedImage::TiledRipmap(mlt)
                } else {
                    MultipartMixedImage::TiledMipmap(mlt)
                }
            }
            PartState::DeepScanline {
                name,
                req,
                sorted_channels,
                samples_per_pixel,
                channel_samples,
            } => MultipartMixedImage::DeepScanline(DeepScanlinePart {
                name,
                data_window: req.data_window,
                display_window: req.display_window,
                line_order: req.line_order,
                compression: req.compression,
                channels: sorted_channels,
                samples_per_pixel,
                channel_samples,
                attributes: part.attributes.clone(),
            }),
            PartState::DeepTiled {
                name,
                req,
                sorted_channels,
                tile_x,
                tile_y,
                tx_count,
                samples_per_pixel,
                tile_decoded,
                ..
            } => {
                // Re-emit channel samples in pixel-scan order from the
                // per-tile slabs.
                let width = req.data_window.width();
                let height = req.data_window.height();
                for (idx, slot) in tile_decoded.iter().enumerate() {
                    if slot.is_none() {
                        return Err(ExrError::invalid(format!(
                            "mixed multi-part deep tiled part '{name}': tile grid \
                             missing entry {idx}"
                        )));
                    }
                }
                let channel_samples = assemble_deep_tiled_level(
                    width,
                    height,
                    tile_x,
                    tile_y,
                    tx_count,
                    sorted_channels.len(),
                    &samples_per_pixel,
                    &tile_decoded,
                );
                MultipartMixedImage::DeepTiled(DeepTiledPart {
                    name,
                    data_window: req.data_window,
                    display_window: req.display_window,
                    line_order: req.line_order,
                    compression: req.compression,
                    tile_x,
                    tile_y,
                    channels: sorted_channels,
                    samples_per_pixel,
                    channel_samples,
                    attributes: part.attributes.clone(),
                })
            }
            PartState::DeepMultilevelTiled {
                name,
                req,
                sorted_channels,
                tile_x,
                tile_y,
                level_mode,
                levels,
            } => {
                let n_channels = sorted_channels.len();
                // Materialise each level's per-tile slabs into pixel-scan
                // order, checking every tile slot is present.
                let mut decoded_levels: Vec<AssembledDeepLevel> = Vec::with_capacity(levels.len());
                for lvl in &levels {
                    for (idx, slot) in lvl.tile_decoded.iter().enumerate() {
                        if slot.is_none() {
                            return Err(ExrError::invalid(format!(
                                "mixed multi-part multi-level deep tiled part '{name}': \
                                 level ({},{}) tile grid missing entry {idx}",
                                lvl.level_x, lvl.level_y
                            )));
                        }
                    }
                    let channel_samples = assemble_deep_tiled_level(
                        lvl.width,
                        lvl.height,
                        tile_x,
                        tile_y,
                        lvl.tx_count,
                        n_channels,
                        &lvl.samples_per_pixel,
                        &lvl.tile_decoded,
                    );
                    decoded_levels.push(AssembledDeepLevel {
                        level_x: lvl.level_x,
                        level_y: lvl.level_y,
                        width: lvl.width,
                        height: lvl.height,
                        samples_per_pixel: lvl.samples_per_pixel.clone(),
                        channel_samples,
                    });
                }
                if level_mode == 1 {
                    // MIPMAP: one entry per level in level-index order.
                    let levels_out: Vec<DeepTiledMipmapLevel> = decoded_levels
                        .into_iter()
                        .map(|lvl| DeepTiledMipmapLevel {
                            width: lvl.width,
                            height: lvl.height,
                            samples_per_pixel: lvl.samples_per_pixel,
                            channel_samples: lvl.channel_samples,
                        })
                        .collect();
                    MultipartMixedImage::DeepTiledMipmap(DeepMipmapTiledPart {
                        name,
                        data_window: req.data_window,
                        display_window: req.display_window,
                        line_order: req.line_order,
                        compression: req.compression,
                        tile_x,
                        tile_y,
                        channels: sorted_channels,
                        levels: levels_out,
                        attributes: part.attributes.clone(),
                    })
                } else {
                    // RIPMAP: reshape into grid[lvly][lvlx]. Derive the
                    // grid shape from the enumerated levels themselves so
                    // ROUND_UP files (whose level counts differ from the
                    // ROUND_DOWN math) reshape correctly too.
                    let ny = decoded_levels
                        .iter()
                        .map(|l| l.level_y + 1)
                        .max()
                        .unwrap_or(0);
                    let nx = decoded_levels
                        .iter()
                        .map(|l| l.level_x + 1)
                        .max()
                        .unwrap_or(0);
                    let mut grid: Vec<Vec<DeepTiledRipmapCell>> =
                        (0..ny).map(|_| Vec::with_capacity(nx as usize)).collect();
                    // decoded_levels are in (lvly outer, lvlx inner) order.
                    for lvl in decoded_levels {
                        grid[lvl.level_y as usize].push(DeepTiledRipmapCell {
                            level_x: lvl.level_x,
                            level_y: lvl.level_y,
                            width: lvl.width,
                            height: lvl.height,
                            samples_per_pixel: lvl.samples_per_pixel,
                            channel_samples: lvl.channel_samples,
                        });
                    }
                    MultipartMixedImage::DeepTiledRipmap(DeepRipmapTiledPart {
                        name,
                        data_window: req.data_window,
                        display_window: req.display_window,
                        line_order: req.line_order,
                        compression: req.compression,
                        tile_x,
                        tile_y,
                        channels: sorted_channels,
                        grid,
                        attributes: part.attributes.clone(),
                    })
                }
            }
        });
    }
    Ok(images)
}

// ---------------- Helpers ----------------

/// Allocate zeroed per-channel planes for a flat part.
fn make_flat_planes(sorted_channels: &[Channel], width: u32, height: u32) -> Vec<ExrPlane> {
    sorted_channels
        .iter()
        .map(|c| {
            let pw = subsampled_dim(width, c.x_sampling as u32) as usize;
            let ph = subsampled_dim(height, c.y_sampling as u32) as usize;
            ExrPlane {
                name: c.name.clone(),
                samples: vec![0.0; pw * ph],
            }
        })
        .collect()
}

/// Enumerate the expected decode-target levels (allocated zeroed) for a
/// flat multi-level tiled part, in the spec's chunk iteration order:
/// MIPMAP = diagonal `lvlx == lvly == n`; RIPMAP = `lvly` outer, `lvlx`
/// inner.
fn enumerate_tiled_levels(
    level_mode: u8,
    width: u32,
    height: u32,
    round_up: bool,
    sorted_channels: &[Channel],
) -> Vec<TiledLevel> {
    match level_mode {
        1 => {
            let n = mipmap_level_count(width.max(height), round_up);
            (0..n)
                .map(|l| {
                    let lw = mipmap_level_dim(width, l, round_up);
                    let lh = mipmap_level_dim(height, l, round_up);
                    TiledLevel {
                        level_x: l,
                        level_y: l,
                        width: lw,
                        height: lh,
                        planes: make_flat_planes(sorted_channels, lw, lh),
                    }
                })
                .collect()
        }
        2 => {
            let nx = mipmap_level_count(width, round_up);
            let ny = mipmap_level_count(height, round_up);
            let mut v = Vec::with_capacity((nx * ny) as usize);
            for ly in 0..ny {
                let lh = mipmap_level_dim(height, ly, round_up);
                for lx in 0..nx {
                    let lw = mipmap_level_dim(width, lx, round_up);
                    v.push(TiledLevel {
                        level_x: lx,
                        level_y: ly,
                        width: lw,
                        height: lh,
                        planes: make_flat_planes(sorted_channels, lw, lh),
                    });
                }
            }
            v
        }
        _ => Vec::new(),
    }
}

/// Enumerate the expected decode-target levels for a multi-level
/// **deep** tiled part, in the spec's chunk iteration order: MIPMAP =
/// diagonal `lvlx == lvly == n`; RIPMAP = `lvly` outer, `lvlx` inner.
/// Each level gets a zeroed `samples_per_pixel` and an empty per-tile
/// decode grid sized to that level's tile counts.
fn enumerate_deep_tiled_levels(
    level_mode: u8,
    width: u32,
    height: u32,
    round_up: bool,
    tile_x: u32,
    tile_y: u32,
) -> Vec<DeepTileLevelState> {
    let make_level = |level_x: u32, level_y: u32, lw: u32, lh: u32| -> DeepTileLevelState {
        let tx_count = lw.div_ceil(tile_x);
        let ty_count = lh.div_ceil(tile_y);
        DeepTileLevelState {
            level_x,
            level_y,
            width: lw,
            height: lh,
            tx_count,
            ty_count,
            samples_per_pixel: vec![0u32; (lw as usize) * (lh as usize)],
            tile_decoded: (0..(tx_count as usize) * (ty_count as usize))
                .map(|_| None)
                .collect(),
        }
    };
    match level_mode {
        1 => {
            let n = mipmap_level_count(width.max(height), round_up);
            (0..n)
                .map(|l| {
                    let lw = mipmap_level_dim(width, l, round_up);
                    let lh = mipmap_level_dim(height, l, round_up);
                    make_level(l, l, lw, lh)
                })
                .collect()
        }
        2 => {
            let nx = mipmap_level_count(width, round_up);
            let ny = mipmap_level_count(height, round_up);
            let mut v = Vec::with_capacity((nx * ny) as usize);
            for ly in 0..ny {
                let lh = mipmap_level_dim(height, ly, round_up);
                for lx in 0..nx {
                    let lw = mipmap_level_dim(width, lx, round_up);
                    v.push(make_level(lx, ly, lw, lh));
                }
            }
            v
        }
        _ => Vec::new(),
    }
}

/// Decode one deep tile chunk body (offset table + sample data payloads
/// already located within the file) into a per-tile decoded slab and
/// write its per-pixel sample counts into `samples_per_pixel` (indexed
/// by the level's full-width row-major coords). Shared by the ONE_LEVEL
/// and the multi-level deep tiled decode arms.
#[allow(clippy::too_many_arguments)]
fn decode_deep_tile_body(
    table_payload: &[u8],
    data_payload: &[u8],
    unpacked_data: usize,
    lvl_w: u32,
    tile_x: u32,
    tile_y: u32,
    tx: u32,
    ty: u32,
    tw: u32,
    th: u32,
    sorted_channels: &[Channel],
    compression: Compression,
    part_label: &str,
    samples_per_pixel: &mut [u32],
) -> Result<TileDecoded> {
    let x0 = tx * tile_x;
    let y0 = ty * tile_y;
    let width = lvl_w as usize;
    // NONE-padding accommodation: some writers pad the per-tile offset
    // table to the full tile rectangle.
    let entries = (tw * th) as usize;
    let full_entries = (tile_x as usize) * (tile_y as usize);
    let (unpacked_table_size, row_stride) =
        if compression == Compression::None && table_payload.len() == full_entries * 4 {
            (full_entries * 4, tile_x as usize)
        } else {
            (entries * 4, tw as usize)
        };
    let table_bytes = decompress_buffer(table_payload, unpacked_table_size, compression)?;
    let mut cumulative_flat: Vec<i32> = Vec::with_capacity(unpacked_table_size / 4);
    for ch in table_bytes.chunks_exact(4) {
        cumulative_flat.push(i32::from_le_bytes(ch.try_into().unwrap()));
    }
    let mut tile_total_samples: u64 = 0;
    for r in 0..th as usize {
        let row_base = r * row_stride;
        let row_slice = &cumulative_flat[row_base..row_base + tw as usize];
        let per_pixel = per_pixel_from_cumulative(row_slice)?;
        let dst_base = (y0 as usize + r) * width + x0 as usize;
        for (i, &n) in per_pixel.iter().enumerate() {
            samples_per_pixel[dst_base + i] = n;
            tile_total_samples += n as u64;
        }
    }
    let block_bpp: usize = sorted_channels
        .iter()
        .map(|c| c.pixel_type.bytes_per_sample())
        .sum();
    let expected_unpacked = tile_total_samples as usize * block_bpp;
    if expected_unpacked != unpacked_data {
        return Err(ExrError::invalid(format!(
            "{part_label}: derived unpacked_data={expected_unpacked} disagrees with \
             header unpacked_data={unpacked_data}"
        )));
    }
    let sample_bytes = decompress_buffer(data_payload, unpacked_data, compression)?;
    let mut per_channel: Vec<Vec<f32>> = (0..sorted_channels.len()).map(|_| Vec::new()).collect();
    decode_deep_sample_block(
        &sample_bytes,
        sorted_channels,
        tile_total_samples as usize,
        &mut per_channel,
    )
    .map_err(|e| ExrError::invalid(format!("{part_label}: {e}")))?;
    Ok(TileDecoded {
        tw,
        th,
        channel_samples: per_channel,
    })
}

/// Reassemble one deep tiled level's per-tile decoded slabs into
/// pixel-scan-order per-channel sample lists. `samples_per_pixel` is the
/// level's full per-pixel count table (row-major over `lvl_w × lvl_h`);
/// every tile slot must be `Some`.
#[allow(clippy::too_many_arguments)]
fn assemble_deep_tiled_level(
    lvl_w: u32,
    lvl_h: u32,
    tile_x: u32,
    tile_y: u32,
    tx_count: u32,
    n_channels: usize,
    samples_per_pixel: &[u32],
    tile_decoded: &[Option<TileDecoded>],
) -> Vec<Vec<f32>> {
    let width = lvl_w as usize;
    let total_samples: u64 = samples_per_pixel.iter().map(|&n| n as u64).sum();
    let mut channel_samples: Vec<Vec<f32>> = (0..n_channels)
        .map(|_| Vec::with_capacity(total_samples as usize))
        .collect();
    // Per-tile pixel-start tables, for slicing per-tile channel slabs.
    let mut tile_pixel_starts: Vec<Vec<u64>> = Vec::with_capacity(tile_decoded.len());
    for (idx, slot) in tile_decoded.iter().enumerate() {
        let td = slot.as_ref().unwrap();
        let x0 = ((idx as u32) % tx_count) * tile_x;
        let y0 = ((idx as u32) / tx_count) * tile_y;
        let mut starts = Vec::with_capacity((td.tw * td.th) as usize + 1);
        starts.push(0u64);
        let mut acc: u64 = 0;
        for r in 0..td.th as usize {
            for c in 0..td.tw as usize {
                let dst_y = y0 as usize + r;
                let dst_x = x0 as usize + c;
                acc += samples_per_pixel[dst_y * width + dst_x] as u64;
                starts.push(acc);
            }
        }
        tile_pixel_starts.push(starts);
    }
    for y in 0..lvl_h as usize {
        let ty = (y / tile_y as usize) as u32;
        let y_in_tile = y - (ty as usize) * tile_y as usize;
        for x in 0..lvl_w as usize {
            let tx = (x / tile_x as usize) as u32;
            let x_in_tile = x - (tx as usize) * tile_x as usize;
            let tile_grid_idx = (ty * tx_count + tx) as usize;
            let td = tile_decoded[tile_grid_idx].as_ref().unwrap();
            let pixel_within_tile = y_in_tile * td.tw as usize + x_in_tile;
            let s_start = tile_pixel_starts[tile_grid_idx][pixel_within_tile] as usize;
            let s_end = tile_pixel_starts[tile_grid_idx][pixel_within_tile + 1] as usize;
            for (ch_idx, dst) in channel_samples.iter_mut().enumerate() {
                dst.extend_from_slice(&td.channel_samples[ch_idx][s_start..s_end]);
            }
        }
    }
    channel_samples
}

/// Wrap decoded flat planes into an [`ExrImage`].
fn make_exr_image(
    req: RequiredAttrs,
    sorted_channels: Vec<Channel>,
    planes: Vec<ExrPlane>,
    attributes: Vec<Attribute>,
) -> ExrImage {
    ExrImage {
        data_window: req.data_window,
        display_window: req.display_window,
        line_order: req.line_order,
        compression: req.compression,
        pixel_aspect_ratio: req.pixel_aspect_ratio,
        screen_window_center: req.screen_window_center,
        screen_window_width: req.screen_window_width,
        channels: sorted_channels,
        planes,
        attributes,
    }
}

/// Decode one channel-major deep sample slab (`n_samples` values per
/// channel, channels in `sorted_channels` order) and append each
/// channel's values to `out[ch]`.
fn decode_deep_sample_block(
    sample_bytes: &[u8],
    sorted_channels: &[Channel],
    n_samples: usize,
    out: &mut [Vec<f32>],
) -> Result<()> {
    let mut p = 0usize;
    for (ch_idx, ch) in sorted_channels.iter().enumerate() {
        let bps = ch.pixel_type.bytes_per_sample();
        let need = n_samples * bps;
        if p + need > sample_bytes.len() {
            return Err(ExrError::invalid(format!(
                "channel '{}' bytes past payload end",
                ch.name
            )));
        }
        for s in 0..n_samples {
            let off = p + s * bps;
            let v = match ch.pixel_type {
                PixelType::Half => crate::half::half_to_f32(u16::from_le_bytes(
                    sample_bytes[off..off + 2].try_into().unwrap(),
                )),
                PixelType::Float => {
                    f32::from_le_bytes(sample_bytes[off..off + 4].try_into().unwrap())
                }
                PixelType::Uint => {
                    let bits = u32::from_le_bytes(sample_bytes[off..off + 4].try_into().unwrap());
                    bits as f32
                }
            };
            out[ch_idx].push(v);
        }
        p += need;
    }
    if p != sample_bytes.len() {
        return Err(ExrError::invalid(format!(
            "consumed {p} of {} payload bytes",
            sample_bytes.len()
        )));
    }
    Ok(())
}

fn push_pixel(raw: &mut Vec<u8>, v: f32, pixel_type: PixelType) {
    match pixel_type {
        PixelType::Half => raw.extend_from_slice(&crate::half::f32_to_half(v).to_le_bytes()),
        PixelType::Float => raw.extend_from_slice(&v.to_le_bytes()),
        PixelType::Uint => {
            let u = if v.is_nan() || v < 0.0 {
                0u32
            } else if v >= (u32::MAX as f32) {
                u32::MAX
            } else {
                (v + 0.5) as u32
            };
            raw.extend_from_slice(&u.to_le_bytes());
        }
    }
}

/// Build the interleaved native byte stream of one scanline chunk (rows
/// top-to-bottom; within a row each present channel's sub-sampled samples
/// in channel order). This is the form ZIP / ZIPS / RLE preprocess and
/// the form the B44 raw-fallback stores.
fn scanline_block_raw(
    channels: &[Channel],
    plane_refs: &[&[f32]],
    width: u32,
    row0: u32,
    lines_in_block: usize,
) -> Vec<u8> {
    let mut raw: Vec<u8> = Vec::new();
    for line in 0..lines_in_block {
        let y = row0 as usize + line;
        for (ch_idx, ch) in channels.iter().enumerate() {
            let ys = ch.y_sampling as u32;
            if (y as u32) % ys != 0 {
                continue;
            }
            let xs = ch.x_sampling as u32;
            let pw = subsampled_dim(width, xs) as usize;
            let plane_y = y / ys as usize;
            let plane = plane_refs[ch_idx];
            for x in 0..pw {
                let v = plane[plane_y * pw + x];
                push_pixel(&mut raw, v, ch.pixel_type);
            }
        }
    }
    raw
}

/// Uncompressed byte length of one scanline chunk (the §0 raw-fallback
/// threshold for B44, computed without materialising the bytes).
fn scanline_block_raw_len(
    channels: &[Channel],
    width: u32,
    row0: u32,
    lines_in_block: usize,
) -> usize {
    let mut total = 0usize;
    for line in 0..lines_in_block as u32 {
        let y = row0 + line;
        for ch in channels {
            let ys = ch.y_sampling as u32;
            if y % ys != 0 {
                continue;
            }
            let pw = subsampled_dim(width, ch.x_sampling as u32) as usize;
            total += ch.pixel_type.bytes_per_sample() * pw;
        }
    }
    total
}

/// Compress one ONE_LEVEL flat tile. NONE/ZIP/ZIPS/RLE consume the
/// interleaved native tile stream (per row: every channel's row segment);
/// PXR24/B44/B44A treat the tile as a self-contained `tw × th` block
/// (origin row 0, 1×1 sampling) and reorganise it from tile-local f32
/// sub-planes via the shared block builders (observer-spec §§1–2).
#[allow(clippy::too_many_arguments)]
fn compress_one_level_tile(
    channels: &[Channel],
    planes: &[&[f32]],
    width: u32,
    x0: u32,
    y0: u32,
    tw: usize,
    th: usize,
    compression: Compression,
) -> Result<Vec<u8>> {
    // Interleaved native tile stream (also the §0 raw-fallback form).
    let mut raw: Vec<u8> = Vec::new();
    for line in 0..th {
        let dst_y = y0 as usize + line;
        for (ch_idx, ch) in channels.iter().enumerate() {
            let plane = planes[ch_idx];
            for xx in 0..tw {
                let dst_x = x0 as usize + xx;
                let v = plane[dst_y * (width as usize) + dst_x];
                push_pixel(&mut raw, v, ch.pixel_type);
            }
        }
    }

    if matches!(
        compression,
        Compression::Pxr24 | Compression::B44 | Compression::B44a
    ) {
        // Gather each channel's tile-local `tw × th` sub-plane row-major.
        let mut sub: Vec<Vec<f32>> = Vec::with_capacity(channels.len());
        for &plane in planes.iter().take(channels.len()) {
            let mut s = Vec::with_capacity(tw * th);
            for line in 0..th {
                let dst_y = y0 as usize + line;
                let base = dst_y * (width as usize) + x0 as usize;
                s.extend_from_slice(&plane[base..base + tw]);
            }
            sub.push(s);
        }
        let sub_refs: Vec<&[f32]> = sub.iter().map(|p| p.as_slice()).collect();
        return Ok(match compression {
            Compression::Pxr24 => {
                crate::encoder::build_pxr24_block_payload(channels, &sub_refs, tw as u32, 0, th)?
            }
            Compression::B44 | Compression::B44a => {
                let flat = matches!(compression, Compression::B44a);
                let packed = crate::encoder::build_b44_block_payload(
                    channels, &sub_refs, tw as u32, 0, th, flat,
                );
                if packed.len() >= raw.len() {
                    raw
                } else {
                    packed
                }
            }
            _ => unreachable!(),
        });
    }

    compress_block(raw, compression)
}

fn compress_block(raw: Vec<u8>, compression: Compression) -> Result<Vec<u8>> {
    Ok(match compression {
        Compression::None => raw,
        Compression::Zip | Compression::Zips => {
            let mut interleaved = vec![0u8; raw.len()];
            apply_zip_interleave(&raw, &mut interleaved);
            apply_zip_predictor(&mut interleaved);
            let compressed = crate::encoder::zlib_deflate_pub(&interleaved)?;
            if compressed.len() >= raw.len() {
                raw
            } else {
                compressed
            }
        }
        Compression::Rle => {
            let mut interleaved = vec![0u8; raw.len()];
            apply_zip_interleave(&raw, &mut interleaved);
            apply_zip_predictor(&mut interleaved);
            let compressed = crate::rle::rle_compress(&interleaved);
            if compressed.len() >= raw.len() {
                raw
            } else {
                compressed
            }
        }
        _ => unreachable!("filtered above"),
    })
}

/// Gather + compress one tile of a flat multi-level [`MipmapLevel`].
/// Edge tiles emit only their valid pixel rectangle. Delegates to
/// [`compress_one_level_tile`] against the level's own `width × height`
/// planes — so NONE/ZIP/ZIPS/RLE consume the interleaved native tile and
/// PXR24/B44/B44A reorganise the tile as a self-contained block, exactly
/// as the ONE_LEVEL tiled path does.
fn compress_tiled_level_tile(
    lvl: &MipmapLevel,
    channels: &[Channel],
    tile_x: u32,
    tile_y: u32,
    tx: u32,
    ty: u32,
    compression: Compression,
) -> Result<Vec<u8>> {
    let x0 = tx * tile_x;
    let y0 = ty * tile_y;
    let x1 = (x0 + tile_x).min(lvl.width);
    let y1 = (y0 + tile_y).min(lvl.height);
    let tw = (x1 - x0) as usize;
    let th = (y1 - y0) as usize;
    let plane_refs: Vec<&[f32]> = lvl.planes.iter().map(|p| p.as_slice()).collect();
    compress_one_level_tile(
        channels,
        &plane_refs,
        lvl.width,
        x0,
        y0,
        tw,
        th,
        compression,
    )
}

/// Decompress one scanline block (mirrors the helper used inside
/// [`crate::parse_exr_multipart`]; the chunk-body decompress + reverse
/// preprocessing pipeline is shared across every flat-scanline reader).
fn decompress_block(
    payload: &[u8],
    uncompressed_size: usize,
    compression: Compression,
) -> Result<Vec<u8>> {
    if matches!(compression, Compression::None) {
        if payload.len() != uncompressed_size {
            return Err(ExrError::invalid(format!(
                "mixed multi-part scanline NONE block: size mismatch (have {}, want {uncompressed_size})",
                payload.len()
            )));
        }
        return Ok(payload.to_vec());
    }
    if payload.len() == uncompressed_size {
        // Encoder may have stored raw bytes when compression didn't shrink.
        return Ok(payload.to_vec());
    }
    let raw = match compression {
        Compression::Zip | Compression::Zips => {
            crate::decoder::zlib_inflate_pub(payload, uncompressed_size)?
        }
        Compression::Rle => crate::rle::rle_decompress(payload, uncompressed_size)?,
        _ => unreachable!("filtered above"),
    };
    if raw.len() != uncompressed_size {
        return Err(ExrError::invalid(format!(
            "mixed multi-part scanline block size mismatch after decode \
             (have {}, want {uncompressed_size})",
            raw.len()
        )));
    }
    Ok(crate::decoder::undo_zip_pipeline_pub(raw))
}

/// Scatter one decompressed scanline-block byte stream into the per-channel
/// f32 planes. Mirrors the private helper inside `decoder.rs`; replicated
/// here so the mixed reader can keep its chunk-body branches local.
fn scatter_scanline_block_into_planes(
    uncompressed: &[u8],
    sorted_channels: &[Channel],
    planes: &mut [ExrPlane],
    width: u32,
    block_y0: u32,
    lines_in_block: usize,
) -> Result<()> {
    let mut p = 0usize;
    for line in 0..lines_in_block {
        let dst_y = block_y0 as usize + line;
        for (ch_idx, ch) in sorted_channels.iter().enumerate() {
            let ys = ch.y_sampling as u32;
            if (dst_y as u32) % ys != 0 {
                continue;
            }
            let xs = ch.x_sampling as u32;
            let pw = subsampled_dim(width, xs) as usize;
            let dst_y_sub = dst_y / ys as usize;
            let plane = &mut planes[ch_idx].samples;
            for x in 0..pw {
                let v = match ch.pixel_type {
                    PixelType::Half => {
                        let bits = u16::from_le_bytes(uncompressed[p..p + 2].try_into().unwrap());
                        crate::half::half_to_f32(bits)
                    }
                    PixelType::Float => {
                        f32::from_le_bytes(uncompressed[p..p + 4].try_into().unwrap())
                    }
                    PixelType::Uint => {
                        let bits = u32::from_le_bytes(uncompressed[p..p + 4].try_into().unwrap());
                        bits as f32
                    }
                };
                plane[dst_y_sub * pw + x] = v;
                p += ch.pixel_type.bytes_per_sample();
            }
        }
    }
    if p != uncompressed.len() {
        return Err(ExrError::invalid(format!(
            "mixed multi-part scanline block consumed {p} of {} payload bytes",
            uncompressed.len()
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_scanline_part_attrs(
    name: &str,
    width: u32,
    height: u32,
    display: Box2i,
    channels: &[Channel],
    compression: Compression,
    chunk_count: i32,
) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (width - 1) as i32,
        y_max: (height - 1) as i32,
    };
    vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(channels.to_vec()),
        },
        Attribute {
            name: "chunkCount".to_string(),
            value: AttributeValue::Other {
                type_name: "int".to_string(),
                data: chunk_count.to_le_bytes().to_vec(),
            },
        },
        Attribute {
            name: "compression".to_string(),
            value: AttributeValue::Compression(compression),
        },
        Attribute {
            name: "dataWindow".to_string(),
            value: AttributeValue::Box2i(win),
        },
        Attribute {
            name: "displayWindow".to_string(),
            value: AttributeValue::Box2i(display),
        },
        Attribute {
            name: "lineOrder".to_string(),
            value: AttributeValue::LineOrder(LineOrder::IncreasingY),
        },
        Attribute {
            name: "name".to_string(),
            value: AttributeValue::Other {
                type_name: "string".to_string(),
                data: name.as_bytes().to_vec(),
            },
        },
        Attribute {
            name: "pixelAspectRatio".to_string(),
            value: AttributeValue::Float(1.0),
        },
        Attribute {
            name: "screenWindowCenter".to_string(),
            value: AttributeValue::V2f(0.0, 0.0),
        },
        Attribute {
            name: "screenWindowWidth".to_string(),
            value: AttributeValue::Float(1.0),
        },
        Attribute {
            name: "type".to_string(),
            value: AttributeValue::Other {
                type_name: "string".to_string(),
                data: b"scanlineimage".to_vec(),
            },
        },
    ]
}

#[allow(clippy::too_many_arguments)]
fn build_tiled_part_attrs(
    name: &str,
    width: u32,
    height: u32,
    display: Box2i,
    tile_x: u32,
    tile_y: u32,
    channels: &[Channel],
    compression: Compression,
    chunk_count: i32,
) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (width - 1) as i32,
        y_max: (height - 1) as i32,
    };
    let mut tiledesc = Vec::with_capacity(9);
    tiledesc.extend_from_slice(&tile_x.to_le_bytes());
    tiledesc.extend_from_slice(&tile_y.to_le_bytes());
    tiledesc.push(0x00); // ONE_LEVEL + ROUND_DOWN

    vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(channels.to_vec()),
        },
        Attribute {
            name: "chunkCount".to_string(),
            value: AttributeValue::Other {
                type_name: "int".to_string(),
                data: chunk_count.to_le_bytes().to_vec(),
            },
        },
        Attribute {
            name: "compression".to_string(),
            value: AttributeValue::Compression(compression),
        },
        Attribute {
            name: "dataWindow".to_string(),
            value: AttributeValue::Box2i(win),
        },
        Attribute {
            name: "displayWindow".to_string(),
            value: AttributeValue::Box2i(display),
        },
        Attribute {
            name: "lineOrder".to_string(),
            value: AttributeValue::LineOrder(LineOrder::IncreasingY),
        },
        Attribute {
            name: "name".to_string(),
            value: AttributeValue::Other {
                type_name: "string".to_string(),
                data: name.as_bytes().to_vec(),
            },
        },
        Attribute {
            name: "pixelAspectRatio".to_string(),
            value: AttributeValue::Float(1.0),
        },
        Attribute {
            name: "screenWindowCenter".to_string(),
            value: AttributeValue::V2f(0.0, 0.0),
        },
        Attribute {
            name: "screenWindowWidth".to_string(),
            value: AttributeValue::Float(1.0),
        },
        Attribute {
            name: "tiles".to_string(),
            value: AttributeValue::Other {
                type_name: "tiledesc".to_string(),
                data: tiledesc,
            },
        },
        Attribute {
            name: "type".to_string(),
            value: AttributeValue::Other {
                type_name: "string".to_string(),
                data: b"tiledimage".to_vec(),
            },
        },
    ]
}

/// Per-part attribute set for a multi-level flat tiled part. Identical
/// to [`build_tiled_part_attrs`] except the `tiles[tiledesc]` mode byte
/// carries the level mode (`1 = MIPMAP_LEVELS`, `2 = RIPMAP_LEVELS`) in
/// its low nibble, ROUND_DOWN (high nibble 0) as the rest of this crate
/// emits. `type` is `tiledimage`.
#[allow(clippy::too_many_arguments)]
fn build_multilevel_tiled_part_attrs(
    name: &str,
    width: u32,
    height: u32,
    display: Box2i,
    tile_x: u32,
    tile_y: u32,
    level_mode: u8,
    channels: &[Channel],
    compression: Compression,
    chunk_count: i32,
) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (width - 1) as i32,
        y_max: (height - 1) as i32,
    };
    let mut tiledesc = Vec::with_capacity(9);
    tiledesc.extend_from_slice(&tile_x.to_le_bytes());
    tiledesc.extend_from_slice(&tile_y.to_le_bytes());
    tiledesc.push(level_mode & 0x0F); // ROUND_DOWN (high nibble 0) + level_mode

    vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(channels.to_vec()),
        },
        Attribute {
            name: "chunkCount".to_string(),
            value: AttributeValue::Other {
                type_name: "int".to_string(),
                data: chunk_count.to_le_bytes().to_vec(),
            },
        },
        Attribute {
            name: "compression".to_string(),
            value: AttributeValue::Compression(compression),
        },
        Attribute {
            name: "dataWindow".to_string(),
            value: AttributeValue::Box2i(win),
        },
        Attribute {
            name: "displayWindow".to_string(),
            value: AttributeValue::Box2i(display),
        },
        Attribute {
            name: "lineOrder".to_string(),
            value: AttributeValue::LineOrder(LineOrder::IncreasingY),
        },
        Attribute {
            name: "name".to_string(),
            value: AttributeValue::Other {
                type_name: "string".to_string(),
                data: name.as_bytes().to_vec(),
            },
        },
        Attribute {
            name: "pixelAspectRatio".to_string(),
            value: AttributeValue::Float(1.0),
        },
        Attribute {
            name: "screenWindowCenter".to_string(),
            value: AttributeValue::V2f(0.0, 0.0),
        },
        Attribute {
            name: "screenWindowWidth".to_string(),
            value: AttributeValue::Float(1.0),
        },
        Attribute {
            name: "tiles".to_string(),
            value: AttributeValue::Other {
                type_name: "tiledesc".to_string(),
                data: tiledesc,
            },
        },
        Attribute {
            name: "type".to_string(),
            value: AttributeValue::Other {
                type_name: "string".to_string(),
                data: b"tiledimage".to_vec(),
            },
        },
    ]
}

/// Per-part attribute set for a deep part — strict superset of the flat
/// per-part required attrs (adds `maxSamplesPerPixel`, `version = 1`,
/// and for deep tiled parts the `tiles[tiledesc]` attribute). `tiles`
/// being `Some` selects `type="deeptile"`, `None` selects
/// `type="deepscanline"`. Attributes stay in lexicographic order.
#[allow(clippy::too_many_arguments)]
fn build_deep_part_attrs(
    name: &str,
    width: u32,
    height: u32,
    display: Box2i,
    tiles: Option<(u32, u32, u8)>,
    channels: &[Channel],
    compression: Compression,
    chunk_count: i32,
    max_samples: i32,
) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (width - 1) as i32,
        y_max: (height - 1) as i32,
    };
    let mut attrs = vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(channels.to_vec()),
        },
        Attribute {
            name: "chunkCount".to_string(),
            value: AttributeValue::Other {
                type_name: "int".to_string(),
                data: chunk_count.to_le_bytes().to_vec(),
            },
        },
        Attribute {
            name: "compression".to_string(),
            value: AttributeValue::Compression(compression),
        },
        Attribute {
            name: "dataWindow".to_string(),
            value: AttributeValue::Box2i(win),
        },
        Attribute {
            name: "displayWindow".to_string(),
            value: AttributeValue::Box2i(display),
        },
        Attribute {
            name: "lineOrder".to_string(),
            value: AttributeValue::LineOrder(LineOrder::IncreasingY),
        },
        Attribute {
            name: "maxSamplesPerPixel".to_string(),
            value: AttributeValue::Other {
                type_name: "int".to_string(),
                data: max_samples.to_le_bytes().to_vec(),
            },
        },
        Attribute {
            name: "name".to_string(),
            value: AttributeValue::Other {
                type_name: "string".to_string(),
                data: name.as_bytes().to_vec(),
            },
        },
        Attribute {
            name: "pixelAspectRatio".to_string(),
            value: AttributeValue::Float(1.0),
        },
        Attribute {
            name: "screenWindowCenter".to_string(),
            value: AttributeValue::V2f(0.0, 0.0),
        },
        Attribute {
            name: "screenWindowWidth".to_string(),
            value: AttributeValue::Float(1.0),
        },
    ];
    if let Some((tile_x, tile_y, level_mode)) = tiles {
        let mut tiledesc = Vec::with_capacity(9);
        tiledesc.extend_from_slice(&tile_x.to_le_bytes());
        tiledesc.extend_from_slice(&tile_y.to_le_bytes());
        tiledesc.push(level_mode & 0x0F); // ROUND_DOWN (high nibble 0) + level_mode
        attrs.push(Attribute {
            name: "tiles".to_string(),
            value: AttributeValue::Other {
                type_name: "tiledesc".to_string(),
                data: tiledesc,
            },
        });
    }
    attrs.push(Attribute {
        name: "type".to_string(),
        value: AttributeValue::Other {
            type_name: "string".to_string(),
            data: if tiles.is_some() {
                b"deeptile".to_vec()
            } else {
                b"deepscanline".to_vec()
            },
        },
    });
    attrs.push(Attribute {
        name: "version".to_string(),
        value: AttributeValue::Other {
            type_name: "int".to_string(),
            data: 1i32.to_le_bytes().to_vec(),
        },
    });
    attrs
}

fn encode_part_header_attributes(attrs: &[Attribute]) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    for a in attrs {
        out.extend_from_slice(a.name.as_bytes());
        out.push(0);
        let (type_name, payload) = encode_attribute_value(&a.value);
        out.extend_from_slice(type_name.as_bytes());
        out.push(0);
        out.extend_from_slice(&(payload.len() as i32).to_le_bytes());
        out.extend_from_slice(&payload);
    }
    out
}
