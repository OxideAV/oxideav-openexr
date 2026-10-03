//! The image-crate API contract types (`IMAGE_CRATE_API`): [`ExrImage`]
//! and its companions [`Plane`], [`ColorInfo`], [`ColorRange`],
//! [`Metadata`], [`RgbImage`], [`RgbaImage`], [`ImageInfo`], [`Frame`]
//! and the [`ExrPixelFormat`] / [`PixelFormat`] layout tag.
//!
//! An [`ExrImage`] is the *colour view* of one flat OpenEXR part: the
//! part's `R G B (A)` channels, its `Y (A)` channel, or its `Y RY BY
//! (A)` luminance/chroma triple reconstructed to RGB, as **one packed
//! little-endian `f32` plane** (`RgbF32Le` / `RgbaF32Le` / `GrayF32Le`).
//! Samples are the file's scene-referred linear values: HALF channels
//! are widened exactly, FLOAT channels copied bit-for-bit, UINT
//! channels converted (exact below 2^24). No clamp and no tone mapping
//! happen on decode; `to_rgb8` / `to_rgba8` clamp to `[0, 1]` × 255.
//!
//! Every other channel set (depth-only parts, AOV layers, deep data,
//! arbitrary names) stays in the depth API — [`crate::ExrPart`] /
//! [`crate::parse_exr`] / [`crate::parse_exr_multipart_mixed`] and the
//! `parse_exr_deep_*` readers keep every channel by name.
//!
//! Everything here is framework-free (builds with
//! `default-features = false`).

use std::time::Duration;

use crate::error::{ExrError, Result};
use crate::types::{Attribute, AttributeValue, Box2i, Channel, Chromaticities, Compression};

// ---------------------------------------------------------------------------
// Pixel formats
// ---------------------------------------------------------------------------

/// Native layouts an [`ExrImage`] can carry. Variant names mirror
/// `oxideav_core::PixelFormat`. Each is one packed plane of little-endian
/// IEEE 754 binary32 samples: 4, 12 or 16 bytes per pixel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ExrPixelFormat {
    /// One `f32` per pixel (the part's `Y` channel).
    GrayF32Le,
    /// Packed `R, G, B` `f32` triples, 12 bytes per pixel.
    RgbF32Le,
    /// Packed `R, G, B, A` `f32` quads, 16 bytes per pixel.
    RgbaF32Le,
}

/// Contract alias for [`ExrPixelFormat`].
pub type PixelFormat = ExrPixelFormat;

impl ExrPixelFormat {
    /// Interleaved components per pixel (`1`, `3` or `4`).
    pub const fn components(self) -> usize {
        match self {
            Self::GrayF32Le => 1,
            Self::RgbF32Le => 3,
            Self::RgbaF32Le => 4,
        }
    }

    /// Bytes per pixel of a packed row (`4 × components`).
    pub const fn bytes_per_pixel(self) -> usize {
        self.components() * 4
    }

    /// Bits per sample — always `32`.
    pub const fn bits_per_sample(self) -> u8 {
        32
    }

    /// `true` for [`ExrPixelFormat::RgbaF32Le`].
    pub const fn has_alpha(self) -> bool {
        matches!(self, Self::RgbaF32Le)
    }
}

// ---------------------------------------------------------------------------
// Plane / colour / metadata
// ---------------------------------------------------------------------------

/// One pixel plane: `stride` bytes per row, `data` holding at least
/// `stride × (height − 1) + width × bytes_per_pixel` bytes (rows may
/// carry padding past the visible width). Every OpenEXR view is packed,
/// so an [`ExrImage`] has exactly one plane.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct Plane {
    /// Bytes per row.
    pub stride: usize,
    /// Row-major bytes.
    pub data: Vec<u8>,
}

impl Plane {
    /// Wrap a plane buffer with its row stride.
    pub fn new(stride: usize, data: Vec<u8>) -> Self {
        Self { stride, data }
    }
}

/// Nominal sample range (H.273 `VideoFullRangeFlag`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum ColorRange {
    /// No range was signalled.
    #[default]
    Unspecified,
    /// Limited (video / studio) range: `VideoFullRangeFlag == 0`.
    Limited,
    /// Full (PC) range: `VideoFullRangeFlag == 1`.
    Full,
}

/// Colour signalling of an image: the sample range plus the H.273
/// `ColourPrimaries` / `TransferCharacteristics` /
/// `MatrixCoefficients` code points (`2` = unspecified).
///
/// OpenEXR stores scene-referred linear light, so every image is `Full`
/// range, `transfer` 8 (linear) and `matrix` 0 (RGB / identity).
/// `primaries` comes from the part's `chromaticities` attribute when its
/// eight coordinates match an H.273 primaries set within `1e-3`
/// (BT.709 → 1, BT.470 M → 4, BT.470 B/G → 5, BT.601 525 / ST 240 → 6,
/// generic film → 8, BT.2020 → 9, ST 428 XYZ → 10, P3 DCI → 11, P3 D65 →
/// 12, EBU 3213 → 22); any other set reports `2` with the exact
/// coordinates kept in [`ExrImage::chromaticities`]. A part without the
/// attribute uses the format's documented default — Rec. ITU-R BT.709
/// primaries with D65 white — and reports `1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct ColorInfo {
    /// Sample range.
    pub range: ColorRange,
    /// H.273 `ColourPrimaries` code point.
    pub primaries: u8,
    /// H.273 `TransferCharacteristics` code point (`8` = linear).
    pub transfer: u8,
    /// H.273 `MatrixCoefficients` code point (`0` = identity / RGB).
    pub matrix: u8,
}

impl ColorInfo {
    /// H.273 "unspecified" code point.
    pub const UNSPECIFIED: u8 = 2;
    /// H.273 `MatrixCoefficients` identity (RGB / GBR) code point.
    pub const MATRIX_IDENTITY: u8 = 0;
    /// H.273 `ColourPrimaries` BT.709 / sRGB code point.
    pub const PRIMARIES_BT709: u8 = 1;
    /// H.273 `ColourPrimaries` BT.2020 / BT.2100 code point.
    pub const PRIMARIES_BT2020: u8 = 9;
    /// H.273 `ColourPrimaries` SMPTE RP 431-2 (P3, DCI white) code point.
    pub const PRIMARIES_P3_DCI: u8 = 11;
    /// H.273 `ColourPrimaries` SMPTE EG 432-1 (Display P3, D65) code
    /// point.
    pub const PRIMARIES_P3_D65: u8 = 12;
    /// H.273 `TransferCharacteristics` linear code point.
    pub const TRANSFER_LINEAR: u8 = 8;

    /// Build a description from its four parts.
    pub const fn new(range: ColorRange, primaries: u8, transfer: u8, matrix: u8) -> Self {
        Self {
            range,
            primaries,
            transfer,
            matrix,
        }
    }

    /// Every field unspecified.
    pub const fn unspecified() -> Self {
        Self::new(
            ColorRange::Unspecified,
            Self::UNSPECIFIED,
            Self::UNSPECIFIED,
            Self::UNSPECIFIED,
        )
    }

    /// The documented default for a part without a `chromaticities`
    /// attribute: full-range linear RGB with BT.709 primaries (the
    /// format's own default primaries).
    pub const fn exr_default() -> Self {
        Self::new(
            ColorRange::Full,
            Self::PRIMARIES_BT709,
            Self::TRANSFER_LINEAR,
            Self::MATRIX_IDENTITY,
        )
    }

    /// Linear light with the primaries code point matching `c` (see the
    /// type docs), or `2` when no H.273 set matches.
    pub fn from_chromaticities(c: &Chromaticities) -> Self {
        Self::new(
            ColorRange::Full,
            primaries_code_point(c).unwrap_or(Self::UNSPECIFIED),
            Self::TRANSFER_LINEAR,
            Self::MATRIX_IDENTITY,
        )
    }

    /// Derive the colour signalling from a part's header attributes:
    /// [`Self::from_chromaticities`] of its `chromaticities` attribute,
    /// else [`Self::exr_default`].
    pub fn from_attributes(attrs: &[Attribute]) -> Self {
        match crate::luma_chroma::chromaticities_of(attrs) {
            Some(c) => Self::from_chromaticities(&c),
            None => Self::exr_default(),
        }
    }

    /// Set the range.
    pub fn with_range(mut self, range: ColorRange) -> Self {
        self.range = range;
        self
    }

    /// Set the primaries code point.
    pub fn with_primaries(mut self, primaries: u8) -> Self {
        self.primaries = primaries;
        self
    }

    /// Set the transfer code point.
    pub fn with_transfer(mut self, transfer: u8) -> Self {
        self.transfer = transfer;
        self
    }

    /// Set the matrix code point.
    pub fn with_matrix(mut self, matrix: u8) -> Self {
        self.matrix = matrix;
        self
    }

    /// `true` when both primaries and transfer are specified (`!= 2`).
    pub fn is_specified(&self) -> bool {
        self.primaries != Self::UNSPECIFIED && self.transfer != Self::UNSPECIFIED
    }
}

impl Default for ColorInfo {
    /// [`ColorInfo::exr_default`].
    fn default() -> Self {
        Self::exr_default()
    }
}

/// `(code point, [rx, ry, gx, gy, bx, by, wx, wy])` for every H.273
/// Table 2 primaries set with published coordinates.
const H273_PRIMARIES: [(u8, [f32; 8]); 10] = [
    (
        1,
        [0.640, 0.330, 0.300, 0.600, 0.150, 0.060, 0.3127, 0.3290],
    ),
    (
        9,
        [0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290],
    ),
    (
        12,
        [0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290],
    ),
    (11, [0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.314, 0.351]),
    (10, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 / 3.0, 1.0 / 3.0]),
    (5, [0.64, 0.33, 0.29, 0.60, 0.15, 0.06, 0.3127, 0.3290]),
    (
        6,
        [0.630, 0.340, 0.310, 0.595, 0.155, 0.070, 0.3127, 0.3290],
    ),
    (4, [0.67, 0.33, 0.21, 0.71, 0.14, 0.08, 0.310, 0.316]),
    (8, [0.681, 0.319, 0.243, 0.692, 0.145, 0.049, 0.310, 0.316]),
    (
        22,
        [0.630, 0.340, 0.295, 0.605, 0.155, 0.077, 0.3127, 0.3290],
    ),
];

/// The H.273 `ColourPrimaries` code point whose chromaticities match
/// `c` (each coordinate within `1e-3`), if any.
pub(crate) fn primaries_code_point(c: &Chromaticities) -> Option<u8> {
    let have = [
        c.red_x, c.red_y, c.green_x, c.green_y, c.blue_x, c.blue_y, c.white_x, c.white_y,
    ];
    H273_PRIMARIES
        .iter()
        .find(|(_, want)| {
            have.iter()
                .zip(want.iter())
                .all(|(a, b)| (a - b).abs() <= 1e-3)
        })
        .map(|(code, _)| *code)
}

impl ColorInfo {
    /// The `chromaticities` attribute value for an H.273 `ColourPrimaries`
    /// code point this crate recognises (`1`, `4`, `5`, `6`, `8`, `9`,
    /// `10`, `11`, `12`, `22`), `None` otherwise — the inverse of
    /// [`Self::from_chromaticities`].
    pub fn chromaticities_for(primaries: u8) -> Option<Chromaticities> {
        chromaticities_table(primaries)
    }
}

fn chromaticities_table(code: u8) -> Option<Chromaticities> {
    H273_PRIMARIES
        .iter()
        .find(|(c, _)| *c == code)
        .map(|(_, p)| Chromaticities {
            red_x: p[0],
            red_y: p[1],
            green_x: p[2],
            green_y: p[3],
            blue_x: p[4],
            blue_y: p[5],
            white_x: p[6],
            white_y: p[7],
        })
}

/// Embedded metadata. OpenEXR headers have no standard ICC / Exif / XMP
/// attribute and no gamma record (samples are linear), so every field is
/// always `None`; the full attribute list is [`ExrImage::attributes`].
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct Metadata {
    /// ICC profile bytes — never carried by OpenEXR.
    pub icc: Option<Vec<u8>>,
    /// Exif payload — never carried by OpenEXR.
    pub exif: Option<Vec<u8>>,
    /// XMP packet — never carried by OpenEXR.
    pub xmp: Option<Vec<u8>>,
    /// Encoding gamma exponent — never carried (linear light).
    pub gamma: Option<f32>,
}

impl Metadata {
    /// Empty metadata.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ICC profile.
    pub fn with_icc(mut self, icc: impl Into<Option<Vec<u8>>>) -> Self {
        self.icc = icc.into();
        self
    }

    /// Set the Exif payload.
    pub fn with_exif(mut self, exif: impl Into<Option<Vec<u8>>>) -> Self {
        self.exif = exif.into();
        self
    }

    /// Set the XMP packet.
    pub fn with_xmp(mut self, xmp: impl Into<Option<Vec<u8>>>) -> Self {
        self.xmp = xmp.into();
        self
    }

    /// Set the gamma exponent.
    pub fn with_gamma(mut self, gamma: impl Into<Option<f32>>) -> Self {
        self.gamma = gamma.into();
        self
    }

    /// `true` when every field is `None`.
    pub fn is_empty(&self) -> bool {
        self.icc.is_none() && self.exif.is_none() && self.xmp.is_none() && self.gamma.is_none()
    }
}

// ---------------------------------------------------------------------------
// RgbImage / RgbaImage
// ---------------------------------------------------------------------------

/// Tightly packed 8-bit RGB image: `width × height × 3` bytes,
/// row-major, channel order `R, G, B`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RgbImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// `width × height × 3` bytes.
    pub data: Vec<u8>,
}

impl RgbImage {
    /// Wrap a tightly packed `width × height × 3` RGB buffer.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            data,
        }
    }

    /// The pixel bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume the image and return the pixel bytes.
    pub fn into_raw(self) -> Vec<u8> {
        self.data
    }

    /// Stride (bytes per row) — always `width × 3`.
    pub fn stride(&self) -> usize {
        self.width as usize * 3
    }
}

/// Tightly packed 8-bit RGBA image: `width × height × 4` bytes,
/// row-major, channel order `R, G, B, A` (`A = 255` when the source has
/// no alpha).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct RgbaImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// `width × height × 4` bytes.
    pub data: Vec<u8>,
}

impl RgbaImage {
    /// Wrap a tightly packed `width × height × 4` RGBA buffer.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            data,
        }
    }

    /// The pixel bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume the image and return the pixel bytes.
    pub fn into_raw(self) -> Vec<u8> {
        self.data
    }

    /// Stride (bytes per row) — always `width × 4`.
    pub fn stride(&self) -> usize {
        self.width as usize * 4
    }
}

// ---------------------------------------------------------------------------
// ImageInfo
// ---------------------------------------------------------------------------

/// Header-only description of an OpenEXR file ([`crate::info`]): the
/// first part's data-window geometry and colour view plus the file's
/// shape (part count, tiled / deep flags, channel list, compression).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ImageInfo {
    /// Data-window width of the described part.
    pub width: u32,
    /// Data-window height of the described part.
    pub height: u32,
    /// The part's colour view layout.
    pub format: PixelFormat,
    /// Number of parts in the file (`1` for a single-part file).
    pub frames: u32,
    /// `true` when the view carries an `A` channel.
    pub has_alpha: bool,
    /// Colour signalling derived from the `chromaticities` attribute.
    pub color: ColorInfo,
    /// `false` — OpenEXR carries no ICC profile.
    pub has_icc: bool,
    /// `false` — OpenEXR carries no Exif.
    pub has_exif: bool,
    /// `false` — OpenEXR carries no XMP.
    pub has_xmp: bool,
    /// The part's `dataWindow`.
    pub data_window: Box2i,
    /// The part's `displayWindow`.
    pub display_window: Box2i,
    /// The part's complete channel list (file order), including the
    /// channels outside the colour view.
    pub channels: Vec<Channel>,
    /// The part's compression scheme.
    pub compression: Compression,
    /// `true` for a tiled part (`ONE_LEVEL`, `MIPMAP_LEVELS` or
    /// `RIPMAP_LEVELS`).
    pub tiled: bool,
    /// `true` for a deep part (variable samples per pixel — no colour
    /// view; `decode` returns `Unsupported`).
    pub deep: bool,
    /// `true` when the file's version field has the multi-part bit.
    pub multipart: bool,
    /// The part's `name` attribute, when present.
    pub part_name: Option<String>,
    /// The part's `type` attribute (`scanlineimage`, `tiledimage`,
    /// `deepscanline`, `deeptile`), when present.
    pub part_type: Option<String>,
}

impl ImageInfo {
    /// Assemble a description. `color` derives from `attributes`
    /// ([`ColorInfo::from_attributes`]); `part_name` / `part_type`
    /// are read from them too.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        data_window: Box2i,
        display_window: Box2i,
        format: PixelFormat,
        frames: u32,
        channels: Vec<Channel>,
        compression: Compression,
        tiled: bool,
        deep: bool,
        multipart: bool,
        attributes: &[Attribute],
    ) -> Self {
        Self {
            width: data_window.width(),
            height: data_window.height(),
            format,
            frames,
            has_alpha: format.has_alpha(),
            color: ColorInfo::from_attributes(attributes),
            has_icc: false,
            has_exif: false,
            has_xmp: false,
            data_window,
            display_window,
            channels,
            compression,
            tiled,
            deep,
            multipart,
            part_name: string_attribute(attributes, "name"),
            part_type: string_attribute(attributes, "type"),
        }
    }
}

/// The value of a `string` attribute, when present.
pub(crate) fn string_attribute(attrs: &[Attribute], name: &str) -> Option<String> {
    attrs.iter().find_map(|a| match (&a.name[..], &a.value) {
        (n, AttributeValue::String(s)) if n == name => Some(s.clone()),
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// ExrImage
// ---------------------------------------------------------------------------

/// Header attributes the encoder regenerates from the image geometry and
/// the [`crate::EncodeOptions`]; they are never kept in
/// [`ExrImage::attributes`].
pub(crate) const STRUCTURAL_ATTRIBUTES: [&str; 11] = [
    "channels",
    "compression",
    "dataWindow",
    "displayWindow",
    "lineOrder",
    "tiles",
    "chunkCount",
    "version",
    "type",
    "name",
    "maxSamplesPerPixel",
];

/// One flat OpenEXR part's colour view: a single packed little-endian
/// `f32` plane in the [`ExrPixelFormat`] the part's channel set maps to
/// (module docs). `width` / `height` are the data-window extents;
/// [`Self::data_window`] keeps the window's position.
///
/// `attributes` holds the part's header attributes **except** the
/// structural ones the encoder regenerates (`channels`, `compression`,
/// `dataWindow`, `displayWindow`, `lineOrder`, `tiles`, `chunkCount`,
/// `version`, `type`, `name`, `maxSamplesPerPixel`), in file order —
/// `pixelAspectRatio`, `screenWindowCenter`, `screenWindowWidth`,
/// `chromaticities`, `owner`, `comments`, … A freshly built image
/// carries the three required viewing attributes at their defaults.
/// [`crate::encode`] writes them back verbatim, so
/// `decode(encode(img)) == img` holds for every image this crate decoded
/// or built.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ExrImage {
    /// Data-window width in pixels (≥ 1).
    pub width: u32,
    /// Data-window height in pixels (≥ 1).
    pub height: u32,
    /// Native layout of the single plane.
    pub format: PixelFormat,
    /// Pixel planes — exactly one, packed little-endian `f32`.
    pub planes: Vec<Plane>,
    /// Colour signalling ([`ColorInfo::from_attributes`]).
    pub color: ColorInfo,
    /// ICC / Exif / XMP / gamma — never carried by OpenEXR.
    pub metadata: Metadata,
    /// The part's `dataWindow` (`width × height` extents at its
    /// position).
    pub data_window: Box2i,
    /// The part's `displayWindow`.
    pub display_window: Box2i,
    /// Non-structural header attributes, in file order (type docs).
    pub attributes: Vec<Attribute>,
}

impl ExrImage {
    /// The spec's three required viewing attributes at their default
    /// values (`pixelAspectRatio` 1, `screenWindowCenter` (0, 0),
    /// `screenWindowWidth` 1) — what a freshly built image carries.
    pub fn default_attributes() -> Vec<Attribute> {
        vec![
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
        ]
    }

    /// Assemble an image from its geometry, layout and planes (exactly
    /// one). The data and display windows are `[0, width) × [0, height)`,
    /// colour is [`ColorInfo::exr_default`], metadata empty and
    /// `attributes` [`Self::default_attributes`]; the `with_*` builders
    /// fill them in.
    ///
    /// Returns [`ExrError::InvalidData`] when `width` or `height` is
    /// `0`, when there is not exactly one plane, when the plane's
    /// `stride` is below `width × bytes_per_pixel`, or when its `data`
    /// is shorter than `stride × (height − 1) + width × bytes_per_pixel`;
    /// [`ExrError::Unsupported`] when the geometry overflows `usize`.
    pub fn new(width: u32, height: u32, format: PixelFormat, planes: Vec<Plane>) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(ExrError::invalid(format!(
                "OpenEXR: {width}x{height} image (both dimensions must be > 0)"
            )));
        }
        if planes.len() != 1 {
            return Err(ExrError::invalid(format!(
                "OpenEXR: expected exactly one packed plane, got {}",
                planes.len()
            )));
        }
        let row_bytes = (width as usize)
            .checked_mul(format.bytes_per_pixel())
            .ok_or_else(|| ExrError::unsupported("OpenEXR: row size overflows usize"))?;
        let plane = &planes[0];
        if plane.stride < row_bytes {
            return Err(ExrError::invalid(format!(
                "OpenEXR: stride {} below row size {row_bytes}",
                plane.stride
            )));
        }
        let needed = plane
            .stride
            .checked_mul(height as usize - 1)
            .and_then(|n| n.checked_add(row_bytes))
            .ok_or_else(|| ExrError::unsupported("OpenEXR: plane size overflows usize"))?;
        if plane.data.len() < needed {
            return Err(ExrError::invalid(format!(
                "OpenEXR: plane holds {} bytes, geometry needs {needed}",
                plane.data.len()
            )));
        }
        let window = Box2i {
            x_min: 0,
            y_min: 0,
            x_max: (width - 1) as i32,
            y_max: (height - 1) as i32,
        };
        Ok(Self {
            width,
            height,
            format,
            planes,
            color: ColorInfo::exr_default(),
            metadata: Metadata::default(),
            data_window: window,
            display_window: window,
            attributes: Self::default_attributes(),
        })
    }

    /// One packed plane with an explicit row stride (`stride ≥ width ×
    /// bytes_per_pixel`). Same validation as [`Self::new`].
    pub fn packed(
        width: u32,
        height: u32,
        format: PixelFormat,
        stride: usize,
        data: Vec<u8>,
    ) -> Result<Self> {
        Self::new(width, height, format, vec![Plane::new(stride, data)])
    }

    /// Build from `width × height × components` `f32` samples
    /// (row-major, interleaved per `format`), serialised little-endian
    /// into one tight plane. [`ExrError::InvalidData`] on a length
    /// mismatch.
    pub fn from_f32(width: u32, height: u32, format: PixelFormat, samples: &[f32]) -> Result<Self> {
        let need = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(format.components()))
            .ok_or_else(|| ExrError::unsupported("OpenEXR: sample count overflows usize"))?;
        if samples.len() != need {
            return Err(ExrError::invalid(format!(
                "OpenEXR: {} samples for {width}x{height} {format:?} (need {need})",
                samples.len()
            )));
        }
        let mut data = Vec::with_capacity(need * 4);
        for v in samples {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Self::packed(
            width,
            height,
            format,
            width as usize * format.bytes_per_pixel(),
            data,
        )
    }

    /// Build an `RgbF32Le` image from tightly packed 8-bit RGB
    /// (`width × height × 3` bytes): each byte `b` becomes the linear
    /// float `b / 255`. [`ExrError::InvalidData`] on a length mismatch.
    pub fn from_rgb8(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        Self::from_8bit(width, height, &data, PixelFormat::RgbF32Le, None)
    }

    /// Build an `RgbaF32Le` image from tightly packed 8-bit RGBA
    /// (`width × height × 4` bytes), `b / 255` per component including
    /// alpha. [`ExrError::InvalidData`] on a length mismatch.
    pub fn from_rgba8(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        Self::from_8bit(width, height, &data, PixelFormat::RgbaF32Le, None)
    }

    /// Shared 8-bit import: `b / 255`, or `(b / 255) ^ gamma` for the
    /// colour components when `gamma` is `Some` (alpha stays linear).
    pub(crate) fn from_8bit(
        width: u32,
        height: u32,
        bytes: &[u8],
        format: PixelFormat,
        gamma: Option<f32>,
    ) -> Result<Self> {
        let comps = format.components();
        let need = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(comps))
            .ok_or_else(|| ExrError::unsupported("OpenEXR: pixel count overflows usize"))?;
        if bytes.len() != need {
            return Err(ExrError::invalid(format!(
                "OpenEXR: {} bytes for {width}x{height} {comps}-byte pixels (need {need})",
                bytes.len()
            )));
        }
        let mut data = Vec::with_capacity(need * 4);
        for px in bytes.chunks_exact(comps) {
            for (c, &b) in px.iter().enumerate() {
                let mut v = b as f32 / 255.0;
                if let Some(g) = gamma {
                    if c < 3 || comps == 1 {
                        v = v.powf(g);
                    }
                }
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
        Self::packed(width, height, format, width as usize * comps * 4, data)
    }

    /// Replace the colour signalling.
    pub fn with_color(mut self, color: ColorInfo) -> Self {
        self.color = color;
        self
    }

    /// Replace the metadata.
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Replace the non-structural header attributes.
    pub fn with_attributes(mut self, attributes: Vec<Attribute>) -> Self {
        self.attributes = attributes;
        self
    }

    /// Move the data window to `window`. Its extents must equal
    /// `width × height` ([`ExrError::InvalidData`] otherwise).
    pub fn with_data_window(mut self, window: Box2i) -> Result<Self> {
        if window.width() != self.width || window.height() != self.height {
            return Err(ExrError::invalid(format!(
                "OpenEXR: dataWindow {}x{} does not match the {}x{} image",
                window.width(),
                window.height(),
                self.width,
                self.height
            )));
        }
        self.data_window = window;
        Ok(self)
    }

    /// Replace the display window (any extents; it need not contain the
    /// data window).
    pub fn with_display_window(mut self, window: Box2i) -> Self {
        self.display_window = window;
        self
    }

    /// Data-window width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Data-window height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Native layout.
    pub fn format(&self) -> PixelFormat {
        self.format
    }

    /// Interleaved components per pixel.
    pub fn components(&self) -> usize {
        self.format.components()
    }

    /// Bytes per pixel of the plane.
    pub fn bytes_per_pixel(&self) -> usize {
        self.format.bytes_per_pixel()
    }

    /// Row stride of the plane in bytes.
    pub fn stride(&self) -> usize {
        self.planes.first().map_or(0, |p| p.stride)
    }

    /// `true` when the plane has no row padding and no trailing bytes.
    pub fn is_tightly_packed(&self) -> bool {
        let row = self.width as usize * self.bytes_per_pixel();
        self.planes
            .first()
            .is_some_and(|p| p.stride == row && p.data.len() == row * self.height as usize)
    }

    /// The pixel bytes of the single packed plane (little-endian `f32`
    /// samples, `stride` bytes per row).
    pub fn as_bytes(&self) -> Option<&[u8]> {
        self.planes.first().map(|p| p.data.as_slice())
    }

    /// Consume the image and return its plane bytes (planes concatenated
    /// in order, strides as reported — one plane here).
    pub fn into_raw(self) -> Vec<u8> {
        let mut planes = self.planes.into_iter();
        let mut out = planes.next().map(|p| p.data).unwrap_or_default();
        for p in planes {
            out.extend_from_slice(&p.data);
        }
        out
    }

    /// The float samples as a tightly packed `width × height ×
    /// components` `Vec<f32>` (row-major, interleaved, row padding
    /// dropped). Allocates a copy.
    pub fn pixels(&self) -> Vec<f32> {
        let comps = self.components();
        let w = self.width as usize;
        let mut out = Vec::with_capacity(w * self.height as usize * comps);
        self.for_each_row(|row| {
            for s in row.chunks_exact(4) {
                out.push(f32::from_le_bytes([s[0], s[1], s[2], s[3]]));
            }
        });
        out
    }

    /// The components of pixel `(x, y)` (`components()` floats; panics
    /// when out of range).
    pub fn pixel(&self, x: u32, y: u32) -> Vec<f32> {
        assert!(x < self.width && y < self.height, "pixel out of range");
        let plane = &self.planes[0];
        let bpp = self.bytes_per_pixel();
        let off = y as usize * plane.stride + x as usize * bpp;
        plane.data[off..off + bpp]
            .chunks_exact(4)
            .map(|s| f32::from_le_bytes([s[0], s[1], s[2], s[3]]))
            .collect()
    }

    /// Visit the visible bytes of each row in order (padding skipped).
    fn for_each_row(&self, mut f: impl FnMut(&[u8])) {
        let Some(plane) = self.planes.first() else {
            return;
        };
        let row = self.width as usize * self.bytes_per_pixel();
        for y in 0..self.height as usize {
            let start = y * plane.stride;
            f(&plane.data[start..start + row]);
        }
    }

    /// One `f32` plane per interleaved component (component order of
    /// the format), each `width × height` long — the shape the depth
    /// encoders take.
    pub(crate) fn component_planes(&self) -> Vec<Vec<f32>> {
        let comps = self.components();
        let pixels = self.width as usize * self.height as usize;
        let mut out: Vec<Vec<f32>> = (0..comps).map(|_| Vec::with_capacity(pixels)).collect();
        self.for_each_row(|row| {
            for px in row.chunks_exact(comps * 4) {
                for (c, s) in px.chunks_exact(4).enumerate() {
                    out[c].push(f32::from_le_bytes([s[0], s[1], s[2], s[3]]));
                }
            }
        });
        out
    }

    /// Tightly packed 8-bit RGB: every colour sample clamped to
    /// `[0, 1]` and scaled `× 255` (nearest; `NaN` → 0). Gray images
    /// replicate `Y` into R, G and B; alpha is dropped. No exposure,
    /// gamma or tone curve is applied — the samples are linear light.
    pub fn to_rgb8(&self) -> Vec<u8> {
        let comps = self.components();
        let n = self.width as usize * self.height as usize;
        let mut out = Vec::with_capacity(n * 3);
        self.for_each_row(|row| {
            for px in row.chunks_exact(comps * 4) {
                let v = |c: usize| {
                    quantise_unit(f32::from_le_bytes([
                        px[c * 4],
                        px[c * 4 + 1],
                        px[c * 4 + 2],
                        px[c * 4 + 3],
                    ]))
                };
                if comps == 1 {
                    let g = v(0);
                    out.extend_from_slice(&[g, g, g]);
                } else {
                    out.extend_from_slice(&[v(0), v(1), v(2)]);
                }
            }
        });
        out
    }

    /// Tightly packed 8-bit RGBA: [`Self::to_rgb8`] plus the alpha
    /// channel quantised the same way, or `255` when the layout has no
    /// alpha.
    pub fn to_rgba8(&self) -> Vec<u8> {
        let comps = self.components();
        let n = self.width as usize * self.height as usize;
        let mut out = Vec::with_capacity(n * 4);
        self.for_each_row(|row| {
            for px in row.chunks_exact(comps * 4) {
                let v = |c: usize| {
                    quantise_unit(f32::from_le_bytes([
                        px[c * 4],
                        px[c * 4 + 1],
                        px[c * 4 + 2],
                        px[c * 4 + 3],
                    ]))
                };
                match comps {
                    1 => {
                        let g = v(0);
                        out.extend_from_slice(&[g, g, g, 255]);
                    }
                    3 => out.extend_from_slice(&[v(0), v(1), v(2), 255]),
                    _ => out.extend_from_slice(&[v(0), v(1), v(2), v(3)]),
                }
            }
        });
        out
    }

    /// The part's `chromaticities` attribute, when present (the exact
    /// primaries behind [`Self::color`]).
    pub fn chromaticities(&self) -> Option<Chromaticities> {
        crate::luma_chroma::chromaticities_of(&self.attributes)
    }

    /// Replace (or with `None`, remove) the `chromaticities` attribute
    /// and re-derive [`Self::color`] from it.
    pub fn with_chromaticities(mut self, c: impl Into<Option<Chromaticities>>) -> Self {
        self.attributes.retain(|a| a.name != "chromaticities");
        if let Some(c) = c.into() {
            self.attributes.push(Attribute {
                name: "chromaticities".to_string(),
                value: AttributeValue::Chromaticities(c),
            });
        }
        self.color = ColorInfo::from_attributes(&self.attributes);
        self
    }

    /// The value of a `string` attribute by name (`owner`, `comments`,
    /// `capDate`, …).
    pub fn string_attribute(&self, name: &str) -> Option<String> {
        string_attribute(&self.attributes, name)
    }

    /// Strip the structural attributes from a part header
    /// ([`STRUCTURAL_ATTRIBUTES`]), keeping the rest in order.
    pub(crate) fn non_structural(attrs: &[Attribute]) -> Vec<Attribute> {
        attrs
            .iter()
            .filter(|a| !STRUCTURAL_ATTRIBUTES.contains(&a.name.as_str()))
            .cloned()
            .collect()
    }
}

/// `v` clamped to `[0, 1]` and scaled to `u8` (nearest; `NaN` → `0`).
#[inline]
pub(crate) fn quantise_unit(v: f32) -> u8 {
    if v.is_nan() {
        return 0;
    }
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

// ---------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------

/// One part of a multi-part file as returned by [`crate::decode_all`]
/// (and consumed by [`crate::encode_all`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Frame {
    /// The part's colour view.
    pub image: ExrImage,
    /// Always `None` — OpenEXR parts carry no timing.
    pub delay: Option<Duration>,
    /// Zero-based part index in the file.
    pub index: u32,
    /// The part's `name` attribute (required in multi-part files).
    pub name: Option<String>,
    /// The part's `type` attribute (`scanlineimage` / `tiledimage`).
    pub part_type: Option<String>,
}

impl Frame {
    /// Wrap a part's view with its index; `name` / `part_type` `None`.
    pub fn new(image: ExrImage, index: u32) -> Self {
        Self {
            image,
            delay: None,
            index,
            name: None,
            part_type: None,
        }
    }

    /// Set the part name.
    pub fn with_name(mut self, name: impl Into<Option<String>>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the part type.
    pub fn with_part_type(mut self, part_type: impl Into<Option<String>>) -> Self {
        self.part_type = part_type.into();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_validate_geometry() {
        assert!(matches!(
            ExrImage::new(
                0,
                1,
                PixelFormat::RgbF32Le,
                vec![Plane::new(12, vec![0; 12])]
            ),
            Err(ExrError::InvalidData(_))
        ));
        assert!(matches!(
            ExrImage::new(1, 1, PixelFormat::RgbF32Le, vec![]),
            Err(ExrError::InvalidData(_))
        ));
        assert!(matches!(
            ExrImage::new(
                2,
                1,
                PixelFormat::RgbF32Le,
                vec![Plane::new(12, vec![0; 24])]
            ),
            Err(ExrError::InvalidData(_))
        ));
        assert!(matches!(
            ExrImage::new(
                2,
                2,
                PixelFormat::GrayF32Le,
                vec![Plane::new(8, vec![0; 15])]
            ),
            Err(ExrError::InvalidData(_))
        ));
        let ok = ExrImage::new(
            2,
            2,
            PixelFormat::GrayF32Le,
            vec![Plane::new(8, vec![0; 16])],
        )
        .unwrap();
        assert!(ok.is_tightly_packed());
        assert_eq!(ok.attributes, ExrImage::default_attributes());
        assert_eq!(ok.color, ColorInfo::exr_default());
        assert!(ok.metadata.is_empty());
        // Padded stride is accepted.
        let padded = ExrImage::packed(2, 2, PixelFormat::GrayF32Le, 12, vec![0; 20]).unwrap();
        assert!(!padded.is_tightly_packed());
        assert_eq!(padded.pixels().len(), 4);
        assert!(matches!(
            ExrImage::from_rgb8(2, 1, vec![0; 5]),
            Err(ExrError::InvalidData(_))
        ));
        assert!(matches!(
            ExrImage::from_f32(2, 1, PixelFormat::RgbaF32Le, &[0.0; 7]),
            Err(ExrError::InvalidData(_))
        ));
        assert!(matches!(
            ok.clone().with_data_window(Box2i {
                x_min: 0,
                y_min: 0,
                x_max: 2,
                y_max: 1
            }),
            Err(ExrError::InvalidData(_))
        ));
        let moved = ok
            .with_data_window(Box2i {
                x_min: 10,
                y_min: -3,
                x_max: 11,
                y_max: -2,
            })
            .unwrap();
        assert_eq!((moved.width(), moved.height()), (2, 2));
    }

    #[test]
    fn to_rgb8_clamps_and_scales() {
        let img = ExrImage::from_f32(
            3,
            1,
            PixelFormat::RgbaF32Le,
            &[
                -1.0,
                0.0,
                0.5,
                0.25,
                1.0,
                2.0,
                f32::NAN,
                1.0,
                0.2,
                0.4,
                0.6,
                0.0,
            ],
        )
        .unwrap();
        assert_eq!(img.to_rgb8(), vec![0, 0, 128, 255, 255, 0, 51, 102, 153]);
        assert_eq!(
            img.to_rgba8(),
            vec![0, 0, 128, 64, 255, 255, 0, 255, 51, 102, 153, 0]
        );
        let gray = ExrImage::from_f32(2, 1, PixelFormat::GrayF32Le, &[0.5, 3.0]).unwrap();
        assert_eq!(gray.to_rgb8(), vec![128, 128, 128, 255, 255, 255]);
        assert_eq!(
            gray.to_rgba8(),
            vec![128, 128, 128, 255, 255, 255, 255, 255]
        );
        let rgb = ExrImage::from_rgb8(1, 1, vec![0, 128, 255]).unwrap();
        assert_eq!(rgb.to_rgb8(), vec![0, 128, 255]);
        assert_eq!(rgb.pixel(0, 0)[2], 1.0);
        assert_eq!(rgb.to_rgba8(), vec![0, 128, 255, 255]);
        let rgba = ExrImage::from_rgba8(1, 1, vec![10, 20, 30, 40]).unwrap();
        assert_eq!(rgba.to_rgba8(), vec![10, 20, 30, 40]);
    }

    #[test]
    fn primaries_code_points_match_h273() {
        let c = |p: [f32; 8]| Chromaticities {
            red_x: p[0],
            red_y: p[1],
            green_x: p[2],
            green_y: p[3],
            blue_x: p[4],
            blue_y: p[5],
            white_x: p[6],
            white_y: p[7],
        };
        assert_eq!(
            primaries_code_point(&crate::luma_chroma::BT709_CHROMATICITIES),
            Some(1)
        );
        assert_eq!(
            primaries_code_point(&c([
                0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290
            ])),
            Some(9)
        );
        assert_eq!(
            primaries_code_point(&c([
                0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290
            ])),
            Some(12)
        );
        assert_eq!(
            primaries_code_point(&c([0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.314, 0.351])),
            Some(11)
        );
        // ACES AP0 has no H.273 code point.
        let ap0 = c([0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767]);
        assert_eq!(primaries_code_point(&ap0), None);
        assert_eq!(
            ColorInfo::from_chromaticities(&ap0),
            ColorInfo::new(ColorRange::Full, 2, 8, 0)
        );
        for (code, _) in H273_PRIMARIES {
            let back = ColorInfo::chromaticities_for(code).unwrap();
            assert_eq!(primaries_code_point(&back), Some(code));
        }
    }

    #[test]
    fn with_chromaticities_rederives_colour() {
        let img = ExrImage::from_f32(1, 1, PixelFormat::GrayF32Le, &[1.0]).unwrap();
        assert_eq!(img.color.primaries, 1);
        assert!(img.chromaticities().is_none());
        let p3 = ColorInfo::chromaticities_for(12).unwrap();
        let img = img.with_chromaticities(p3);
        assert_eq!(img.color.primaries, 12);
        assert_eq!(img.chromaticities(), Some(p3));
        assert_eq!(img.attributes.len(), 4);
        let img = img.with_chromaticities(None);
        assert_eq!(img.color.primaries, 1);
        assert_eq!(img.attributes.len(), 3);
    }

    #[test]
    fn into_raw_and_component_planes() {
        let img = ExrImage::from_f32(2, 1, PixelFormat::RgbF32Le, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let planes = img.component_planes();
        assert_eq!(planes, vec![vec![1.0, 4.0], vec![2.0, 5.0], vec![3.0, 6.0]]);
        assert_eq!(img.as_bytes().unwrap().len(), 24);
        assert_eq!(img.clone().into_raw(), img.as_bytes().unwrap().to_vec());
    }
}
