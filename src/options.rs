//! Decode-side limits, strictness and part / layer selection
//! ([`DecodeOptions`]) and the encode-side knobs ([`EncodeOptions`]) of
//! the standalone API. Framework-free; the `registry` feature adds the
//! `CodecOptionsStruct` schemas on top.

use crate::error::{ExrError, Result};
use crate::types::{Box2i, Compression, LineOrder, PixelType};

/// Limits, strictness and selection for [`crate::decode_with`].
///
/// Every limit is checked against the selected part's header **before**
/// any pixel plane is allocated, so a hostile `dataWindow` fails with
/// [`ExrError::LimitExceeded`] instead of committing memory. `max_bytes`
/// bounds the `f32` channel planes the decoder allocates for the part
/// (every channel, 4 bytes per sample — at least the size of the
/// returned view). The defaults: `max_width` / `max_height` of
/// [`DecodeOptions::DEFAULT_MAX_DIMENSION`] (65 535), no pixel-count
/// limit, [`DecodeOptions::DEFAULT_MAX_BYTES`] (1 GiB), `strict = false`.
///
/// `strict` rejects what the lenient decoder tolerates: a sub-sampled
/// channel whose data window is not aligned to and divisible by its
/// sampling factors (conforming readers refuse such files; the lenient
/// path reads ceil-sized planes), and in [`crate::decode_all`] a part
/// without a colour view (lenient skips it).
///
/// `part` / `part_name` pick the part of a multi-part file (`part_name`
/// wins when non-empty; the default is part 0) and `layer` the
/// channel-name prefix (`diffuse`, `right`, …) or default-view name whose
/// channels form the view (`""` = the unprefixed base layer).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DecodeOptions {
    /// Reject parts wider than this (data-window pixels).
    pub max_width: Option<u32>,
    /// Reject parts taller than this (data-window pixels).
    pub max_height: Option<u32>,
    /// Reject parts with more than this many pixels (`width × height`).
    pub max_pixels: Option<u64>,
    /// Reject parts whose decoded channel planes would exceed this many
    /// bytes (`width × height × channels × 4`).
    pub max_bytes: Option<u64>,
    /// Reject the recoverable irregularities listed in the type docs.
    pub strict: bool,
    /// Zero-based part index to decode from a multi-part file (must be
    /// `0` for single-part files).
    pub part: u32,
    /// Part to decode by its `name` attribute; overrides `part` when
    /// non-empty.
    pub part_name: String,
    /// Layer whose channels form the view (type docs).
    pub layer: String,
}

impl DecodeOptions {
    /// Default [`Self::max_width`] / [`Self::max_height`]: 65 535.
    pub const DEFAULT_MAX_DIMENSION: u32 = 65_535;
    /// Default [`Self::max_bytes`]: 1 GiB of decoded planes.
    pub const DEFAULT_MAX_BYTES: u64 = 1 << 30;

    /// The defaults (see the type docs).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set (or lift with `None`) the width limit.
    pub fn with_max_width(mut self, max_width: impl Into<Option<u32>>) -> Self {
        self.max_width = max_width.into();
        self
    }

    /// Set (or lift with `None`) the height limit.
    pub fn with_max_height(mut self, max_height: impl Into<Option<u32>>) -> Self {
        self.max_height = max_height.into();
        self
    }

    /// Set (or lift with `None`) the pixel-count limit.
    pub fn with_max_pixels(mut self, max_pixels: impl Into<Option<u64>>) -> Self {
        self.max_pixels = max_pixels.into();
        self
    }

    /// Set (or lift with `None`) the decoded-bytes limit.
    pub fn with_max_bytes(mut self, max_bytes: impl Into<Option<u64>>) -> Self {
        self.max_bytes = max_bytes.into();
        self
    }

    /// Set strict mode (see the type docs).
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// Select a part by index.
    pub fn with_part(mut self, part: u32) -> Self {
        self.part = part;
        self
    }

    /// Select a part by its `name` attribute.
    pub fn with_part_name(mut self, part_name: impl Into<String>) -> Self {
        self.part_name = part_name.into();
        self
    }

    /// Select the layer whose channels form the view.
    pub fn with_layer(mut self, layer: impl Into<String>) -> Self {
        self.layer = layer.into();
        self
    }

    /// Lift every limit (`max_*` all `None`). Trusted local input
    /// only — a hostile header is then bounded by the allocator alone.
    pub fn unlimited(mut self) -> Self {
        self.max_width = None;
        self.max_height = None;
        self.max_pixels = None;
        self.max_bytes = None;
        self
    }

    /// Check a part's geometry against the limits. `bytes` is the size of
    /// the `f32` planes the decoder will allocate for it.
    pub(crate) fn check(&self, width: u32, height: u32, bytes: u64) -> Result<()> {
        if let Some(m) = self.max_width {
            if width > m {
                return Err(ExrError::limit(format!(
                    "OpenEXR: dataWindow width {width} exceeds max_width {m}"
                )));
            }
        }
        if let Some(m) = self.max_height {
            if height > m {
                return Err(ExrError::limit(format!(
                    "OpenEXR: dataWindow height {height} exceeds max_height {m}"
                )));
            }
        }
        let pixels = u64::from(width) * u64::from(height);
        if let Some(m) = self.max_pixels {
            if pixels > m {
                return Err(ExrError::limit(format!(
                    "OpenEXR: {pixels} pixels exceed max_pixels {m}"
                )));
            }
        }
        if let Some(m) = self.max_bytes {
            if bytes > m {
                return Err(ExrError::limit(format!(
                    "OpenEXR: decoded planes of {bytes} bytes exceed max_bytes {m}"
                )));
            }
        }
        Ok(())
    }
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            max_width: Some(Self::DEFAULT_MAX_DIMENSION),
            max_height: Some(Self::DEFAULT_MAX_DIMENSION),
            max_pixels: None,
            max_bytes: Some(Self::DEFAULT_MAX_BYTES),
            strict: false,
            part: 0,
            part_name: String::new(),
            layer: String::new(),
        }
    }
}

/// Colour channel layout the encoder writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ColourLayout {
    /// `R`, `G`, `B` (+ `A`) channels; gray images write `Y`.
    #[default]
    Rgb,
    /// `Y`, `RY`, `BY` (+ `A`) luminance/chroma channels with the
    /// chroma sub-sampled by [`EncodeOptions::chroma_sampling`]; gray
    /// images write `Y`.
    LumaChroma,
}

/// Level mode of a tiled file written by the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum LevelMode {
    /// ONE_LEVEL.
    #[default]
    One,
    /// MIPMAP_LEVELS, box-filtered from the image.
    Mipmap,
    /// RIPMAP_LEVELS, separably box-filtered from the image.
    Ripmap,
}

/// Encoder knobs for [`crate::encode`] / [`crate::encode_rgb8`] /
/// [`crate::encode_rgba8`] / [`crate::encode_all`].
///
/// The on-wire choices are fields: the channel [`PixelType`] (`Float`
/// by default — binary32, lossless; `Half` rounds to binary16, nearest
/// even), the [`Compression`] scheme (`Zip` by default; every scanline
/// codec the crate has), the [`ColourLayout`] (`Rgb` by default,
/// `LumaChroma` writes `Y RY BY` with `chroma_sampling`), the `layer`
/// prefix for the written channel names, the part shape (`tile_size`
/// `0` = scanline file, else `N × N` tiles with `levels`), the chunk
/// storage [`LineOrder`], and `data_window` / `display_window`
/// overrides (`None` = the image's own windows). `input_gamma` only
/// affects the raw 8-bit paths: `None` treats bytes as linear (`b /
/// 255`), `Some(g)` linearises colour bytes as `(b / 255) ^ g`.
///
/// Tiled output (`tile_size > 0`) requires full-resolution channels
/// (`chroma_sampling = 1` under `LumaChroma`) and a data window at the
/// origin.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct EncodeOptions {
    /// Channel pixel type written to the file.
    pub pixel_type: PixelType,
    /// Compression scheme.
    pub compression: Compression,
    /// Colour channel layout.
    pub colour: ColourLayout,
    /// `RY` / `BY` sampling factor (both axes) for
    /// [`ColourLayout::LumaChroma`]; `1` keeps the chroma at full
    /// resolution. Ignored for [`ColourLayout::Rgb`].
    pub chroma_sampling: u32,
    /// Layer prefix for every written channel (`diffuse` → `diffuse.R`
    /// …); empty writes unprefixed names.
    pub layer: String,
    /// Tile edge in pixels; `0` writes a scanline file.
    pub tile_size: u32,
    /// Level mode for tiled files.
    pub levels: LevelMode,
    /// Chunk storage order.
    pub line_order: LineOrder,
    /// `dataWindow` override; `None` writes the image's.
    pub data_window: Option<Box2i>,
    /// `displayWindow` override; `None` writes the image's.
    pub display_window: Option<Box2i>,
    /// Linearisation exponent for 8-bit input (raw paths only).
    pub input_gamma: Option<f32>,
}

impl Default for EncodeOptions {
    fn default() -> Self {
        Self {
            pixel_type: PixelType::Float,
            compression: Compression::Zip,
            colour: ColourLayout::Rgb,
            chroma_sampling: 2,
            layer: String::new(),
            tile_size: 0,
            levels: LevelMode::One,
            line_order: LineOrder::IncreasingY,
            data_window: None,
            display_window: None,
            input_gamma: None,
        }
    }
}

impl EncodeOptions {
    /// The defaults (FLOAT, ZIP, RGB channels, scanline, INCREASING_Y).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the channel pixel type.
    pub fn with_pixel_type(mut self, pixel_type: PixelType) -> Self {
        self.pixel_type = pixel_type;
        self
    }

    /// Set the compression scheme.
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Set the colour channel layout.
    pub fn with_colour(mut self, colour: ColourLayout) -> Self {
        self.colour = colour;
        self
    }

    /// Set the chroma sampling factor (≥ 1).
    pub fn with_chroma_sampling(mut self, chroma_sampling: u32) -> Self {
        self.chroma_sampling = chroma_sampling;
        self
    }

    /// Set the layer prefix.
    pub fn with_layer(mut self, layer: impl Into<String>) -> Self {
        self.layer = layer.into();
        self
    }

    /// Set the tile edge (`0` = scanline).
    pub fn with_tile_size(mut self, tile_size: u32) -> Self {
        self.tile_size = tile_size;
        self
    }

    /// Set the tiled level mode.
    pub fn with_levels(mut self, levels: LevelMode) -> Self {
        self.levels = levels;
        self
    }

    /// Set the chunk storage order.
    pub fn with_line_order(mut self, line_order: LineOrder) -> Self {
        self.line_order = line_order;
        self
    }

    /// Set (or clear) the `dataWindow` override.
    pub fn with_data_window(mut self, data_window: impl Into<Option<Box2i>>) -> Self {
        self.data_window = data_window.into();
        self
    }

    /// Set (or clear) the `displayWindow` override.
    pub fn with_display_window(mut self, display_window: impl Into<Option<Box2i>>) -> Self {
        self.display_window = display_window.into();
        self
    }

    /// Set (or clear) the 8-bit input linearisation exponent.
    pub fn with_input_gamma(mut self, input_gamma: impl Into<Option<f32>>) -> Self {
        self.input_gamma = input_gamma.into();
        self
    }

    /// Reject option combinations no writer accepts.
    pub(crate) fn validate(&self) -> Result<()> {
        if self.chroma_sampling == 0 {
            return Err(ExrError::invalid(
                "OpenEXR encoder: chroma_sampling must be >= 1",
            ));
        }
        let l = &self.layer;
        if l.starts_with('.') || l.ends_with('.') || l.contains("..") {
            return Err(ExrError::invalid(format!(
                "OpenEXR encoder: layer '{l}' must not start or end with '.' or contain an \
                 empty component"
            )));
        }
        if self.tile_size == 0 && self.levels != LevelMode::One {
            return Err(ExrError::invalid(format!(
                "OpenEXR encoder: levels={:?} needs a tiled file (tile_size > 0)",
                self.levels
            )));
        }
        if self.tile_size == 0 && self.line_order == LineOrder::RandomY {
            return Err(ExrError::invalid(
                "OpenEXR encoder: line_order=random_y is only valid for tiled files",
            ));
        }
        if self.pixel_type == PixelType::Uint {
            return Err(ExrError::unsupported(
                "OpenEXR encoder: pixel_type UINT has no float-view mapping; use \
                 encode_exr_scanline with explicit channels",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limits_fire_in_order() {
        let o = DecodeOptions::default()
            .with_max_width(10u32)
            .with_max_height(10u32)
            .with_max_pixels(50u64)
            .with_max_bytes(100u64);
        assert!(o.check(5, 5, 75).is_ok());
        assert!(matches!(o.check(11, 1, 1), Err(ExrError::LimitExceeded(_))));
        assert!(matches!(o.check(1, 11, 1), Err(ExrError::LimitExceeded(_))));
        assert!(matches!(o.check(8, 8, 1), Err(ExrError::LimitExceeded(_))));
        assert!(matches!(
            o.check(5, 5, 101),
            Err(ExrError::LimitExceeded(_))
        ));
        assert!(o.unlimited().check(u32::MAX, u32::MAX, u64::MAX).is_ok());
    }

    #[test]
    fn defaults_are_finite_and_lenient() {
        let d = DecodeOptions::default();
        assert_eq!(d.max_width, Some(65_535));
        assert_eq!(d.max_height, Some(65_535));
        assert_eq!(d.max_pixels, None);
        assert_eq!(d.max_bytes, Some(1 << 30));
        assert!(!d.strict);
        assert_eq!(d.part, 0);
        assert!(d.part_name.is_empty() && d.layer.is_empty());
        // A 4K RGBA float render (3840 × 2160 × 16 ≈ 127 MiB) fits.
        assert!(d.check(3840, 2160, 3840 * 2160 * 16).is_ok());
        assert!(d.check(65_535, 65_535, 65_535 * 65_535 * 16).is_err());
    }

    #[test]
    fn encode_defaults_and_validation() {
        let e = EncodeOptions::default();
        assert_eq!(e.pixel_type, PixelType::Float);
        assert_eq!(e.compression, Compression::Zip);
        assert_eq!(e.colour, ColourLayout::Rgb);
        assert_eq!(e.chroma_sampling, 2);
        assert_eq!(e.tile_size, 0);
        assert_eq!(e.levels, LevelMode::One);
        assert_eq!(e.line_order, LineOrder::IncreasingY);
        assert!(e.data_window.is_none() && e.display_window.is_none());
        assert!(e.input_gamma.is_none());
        assert!(e.validate().is_ok());
        assert!(e.clone().with_chroma_sampling(0).validate().is_err());
        assert!(e.clone().with_layer(".x").validate().is_err());
        assert!(e.clone().with_layer("a..b").validate().is_err());
        assert!(e.clone().with_levels(LevelMode::Mipmap).validate().is_err());
        assert!(e
            .clone()
            .with_line_order(LineOrder::RandomY)
            .validate()
            .is_err());
        assert!(e
            .clone()
            .with_tile_size(16)
            .with_levels(LevelMode::Ripmap)
            .with_line_order(LineOrder::RandomY)
            .validate()
            .is_ok());
        assert!(matches!(
            e.with_pixel_type(PixelType::Uint).validate(),
            Err(ExrError::Unsupported(_))
        ));
    }
}
