//! `oxideav-core` integration layer for `oxideav-openexr`.
//!
//! Gated behind the default-on `registry` feature so image-library
//! consumers can depend on `oxideav-openexr` with `default-features =
//! false` and skip the `oxideav-core` dependency entirely.
//!
//! The module is a thin adapter over the standalone layer: the framework
//! [`Decoder`] calls [`crate::decode_with`] and the [`Encoder`] calls
//! [`crate::encode`] (one implementation). It exposes:
//!
//! * [`register`] — the unified `RuntimeContext` entry point
//!   `oxideav_meta::register_all` calls (via the `register!` macro);
//!   [`register_codecs`] / [`register_containers`] /
//!   [`register_registries`] for the individual registries;
//!   [`make_decoder`] / [`make_encoder`] — the factories.
//! * The frame bridge: `From<ExrImage> for VideoFrame` (one packed float
//!   plane plus the colour-signal side-channel — OpenEXR defines its
//!   colour semantics, linear light with the `chromaticities` attribute
//!   or BT.709 by default, so the signal is always stamped) and
//!   [`ExrImage::from_video_frame`] / `TryFrom<(&VideoFrame,
//!   &CodecParameters)>`, with the [`ExrPixelFormat`] ↔ `PixelFormat`
//!   and [`ColorInfo`] ↔ `ColorSignal` mappings.
//! * `From<ExrError> for oxideav_core::Error` and the
//!   `CodecOptionsStruct` schemas for [`DecodeOptions`] (`part`, `layer`,
//!   `part_name`) and [`EncodeOptions`] (`pixel_type`, `compression`,
//!   `colour`, `chroma_sampling`, `layer`, `tile_size`, `levels`,
//!   `line_order`, `input_gamma`).
//!
//! # Pixel formats
//!
//! The decoder emits the view's native layout — [`PixelFormat::RgbaF32Le`],
//! [`PixelFormat::RgbF32Le`] or [`PixelFormat::GrayF32Le`] — exactly as
//! [`crate::decode`] does (channel mapping, layers and parts: see
//! [`crate::ExrImage`]). There is **no tone-mapping and no clamp**. The
//! encoder accepts the same three formats natively and `Rgb24` / `Rgba`
//! through the raw-path rule (`b / 255`, `input_gamma` to linearise).
//! Deep parts and multi-part output have no frame mapping (one sample
//! per pixel, one image per packet); use [`crate::decode_all`] /
//! [`crate::encode_all`] and the depth API.

use oxideav_core::{
    parse_options, CodecCapabilities, CodecId, CodecInfo, CodecOptionsStruct, CodecParameters,
    CodecRegistry, ColorPrimaries, ColorSignal, ContainerRegistry, Decoder, Encoder, Frame,
    MatrixCoefficients, OptionField, OptionKind, OptionValue, Packet, PixelFormat, RuntimeContext,
    TimeBase, TransferCharacteristics, VideoFrame, VideoPlane,
};

use crate::error::ExrError;
use crate::image::{ColorInfo, ColorRange, ExrImage, ExrPixelFormat};
pub use crate::options::{ColourLayout, LevelMode};
use crate::options::{DecodeOptions, EncodeOptions};
use crate::types::{Compression, LineOrder, PixelType};
use crate::CODEC_ID_STR;

/// The pre-contract name of [`DecodeOptions`].
#[deprecated(note = "use oxideav_openexr::DecodeOptions (IMAGE_CRATE_API)")]
pub type ExrDecoderOptions = DecodeOptions;

/// The pre-contract name of [`EncodeOptions`].
#[deprecated(note = "use oxideav_openexr::EncodeOptions (IMAGE_CRATE_API)")]
pub type ExrEncoderOptions = EncodeOptions;

/// Convert an [`ExrError`] into the framework-shared
/// `oxideav_core::Error` so trait impls can use `?` on errors returned
/// by the framework-free parse/encode functions.
impl From<ExrError> for oxideav_core::Error {
    fn from(e: ExrError) -> Self {
        match e {
            ExrError::InvalidData(s) => oxideav_core::Error::InvalidData(s),
            ExrError::Unsupported(s) => oxideav_core::Error::Unsupported(s),
            ExrError::LimitExceeded(s) => oxideav_core::Error::ResourceExhausted(s),
            ExrError::Io(e) => oxideav_core::Error::Io(e),
        }
    }
}

/// Pixel formats the decoder emits and the encoder accepts natively, in
/// preference order.
const FRAME_FORMATS: [PixelFormat; 3] = [
    PixelFormat::RgbaF32Le,
    PixelFormat::RgbF32Le,
    PixelFormat::GrayF32Le,
];

/// 8-bit formats the encoder additionally accepts (raw-path rule).
const RAW_FORMATS: [PixelFormat; 2] = [PixelFormat::Rgb24, PixelFormat::Rgba];

// ---- pixel formats --------------------------------------------------------

/// The 1:1 name mapping from [`ExrPixelFormat`] to the framework enum.
pub fn to_core_pixel_format(pf: ExrPixelFormat) -> PixelFormat {
    match pf {
        ExrPixelFormat::GrayF32Le => PixelFormat::GrayF32Le,
        ExrPixelFormat::RgbF32Le => PixelFormat::RgbF32Le,
        ExrPixelFormat::RgbaF32Le => PixelFormat::RgbaF32Le,
    }
}

/// The inverse of [`to_core_pixel_format`]; [`ExrError::Unsupported`]
/// for a layout OpenEXR views do not use.
pub fn from_core_pixel_format(pf: PixelFormat) -> Result<ExrPixelFormat, ExrError> {
    match pf {
        PixelFormat::GrayF32Le => Ok(ExrPixelFormat::GrayF32Le),
        PixelFormat::RgbF32Le => Ok(ExrPixelFormat::RgbF32Le),
        PixelFormat::RgbaF32Le => Ok(ExrPixelFormat::RgbaF32Le),
        other => Err(ExrError::unsupported(format!(
            "OpenEXR: pixel format {other:?} is not an OpenEXR view layout (RgbaF32Le / \
             RgbF32Le / GrayF32Le)"
        ))),
    }
}

impl From<ExrPixelFormat> for PixelFormat {
    fn from(pf: ExrPixelFormat) -> Self {
        to_core_pixel_format(pf)
    }
}

impl TryFrom<PixelFormat> for ExrPixelFormat {
    type Error = ExrError;
    fn try_from(pf: PixelFormat) -> Result<Self, ExrError> {
        from_core_pixel_format(pf)
    }
}

// ---- colour signalling ----------------------------------------------------

/// [`ColorInfo`] as the framework's [`ColorSignal`] (code points map
/// 1:1).
pub fn to_color_signal(c: &ColorInfo) -> ColorSignal {
    let range = match c.range {
        ColorRange::Unspecified => oxideav_core::ColorRange::Unspecified,
        ColorRange::Limited => oxideav_core::ColorRange::Limited,
        ColorRange::Full => oxideav_core::ColorRange::Full,
    };
    ColorSignal::new(
        range,
        ColorPrimaries(c.primaries),
        TransferCharacteristics(c.transfer),
        MatrixCoefficients(c.matrix),
    )
}

/// The inverse of [`to_color_signal`].
pub fn from_color_signal(s: &ColorSignal) -> ColorInfo {
    let range = match s.range {
        oxideav_core::ColorRange::Limited => ColorRange::Limited,
        oxideav_core::ColorRange::Full => ColorRange::Full,
        _ => ColorRange::Unspecified,
    };
    ColorInfo::new(range, s.primaries.0, s.transfer.0, s.matrix.0)
}

// ---- frame bridge ---------------------------------------------------------

/// [`ExrImage`] → `VideoFrame`, moving the plane out of the image: one
/// packed float plane plus the colour-signal side-channel.
pub(crate) fn image_into_video_frame(mut image: ExrImage, pts: Option<i64>) -> VideoFrame {
    let stride = image.stride();
    let data = if image.planes.is_empty() {
        Vec::new()
    } else {
        std::mem::take(&mut image.planes[0].data)
    };
    let mut frame = VideoFrame {
        pts,
        planes: vec![VideoPlane { stride, data }],
    };
    frame.set_color_signal(to_color_signal(&image.color));
    frame
}

impl From<ExrImage> for VideoFrame {
    /// The pixel plane (`pts` `None`) plus the colour-signal
    /// side-channel.
    fn from(image: ExrImage) -> Self {
        image_into_video_frame(image, None)
    }
}

impl From<&ExrImage> for VideoFrame {
    fn from(image: &ExrImage) -> Self {
        image_into_video_frame(image.clone(), None)
    }
}

impl ExrImage {
    /// Rebuild an image from a framework frame and the stream parameters
    /// that describe it: `width`, `height` and `pixel_format` are
    /// required. `RgbaF32Le` / `RgbF32Le` / `GrayF32Le` frames become
    /// the plane as is (geometry validated by [`ExrImage::packed`]);
    /// `Rgb24` / `Rgba` frames convert by the raw-path rule (`b / 255`,
    /// see [`ExrImage::from_rgb8`]); any other layout is
    /// [`ExrError::Unsupported`]. The frame's colour-signal side-channel,
    /// refined over `params.color_signal`, becomes `color` when it
    /// specifies anything, and a primaries code point this crate knows
    /// the chromaticities of is also written as the `chromaticities`
    /// attribute so the encoder emits it.
    pub fn from_video_frame(
        frame: &VideoFrame,
        params: &CodecParameters,
    ) -> Result<Self, ExrError> {
        Self::from_video_frame_with_gamma(frame, params, None)
    }

    /// [`Self::from_video_frame`] with the raw-path linearisation
    /// exponent for `Rgb24` / `Rgba` frames.
    pub(crate) fn from_video_frame_with_gamma(
        frame: &VideoFrame,
        params: &CodecParameters,
        input_gamma: Option<f32>,
    ) -> Result<Self, ExrError> {
        let width = params
            .width
            .ok_or_else(|| ExrError::invalid("OpenEXR: width missing in CodecParameters"))?;
        let height = params
            .height
            .ok_or_else(|| ExrError::invalid("OpenEXR: height missing in CodecParameters"))?;
        let format = params
            .pixel_format
            .ok_or_else(|| ExrError::invalid("OpenEXR: pixel_format missing in CodecParameters"))?;
        let plane = frame
            .image_planes()
            .first()
            .ok_or_else(|| ExrError::invalid("OpenEXR: frame has no planes"))?;
        let mut img = match format {
            PixelFormat::RgbaF32Le | PixelFormat::RgbF32Le | PixelFormat::GrayF32Le => {
                ExrImage::packed(
                    width,
                    height,
                    from_core_pixel_format(format)?,
                    plane.stride,
                    plane.data.clone(),
                )?
            }
            PixelFormat::Rgb24 | PixelFormat::Rgba => {
                let (bpp, target) = if format == PixelFormat::Rgb24 {
                    (3, ExrPixelFormat::RgbF32Le)
                } else {
                    (4, ExrPixelFormat::RgbaF32Le)
                };
                let row = width as usize * bpp;
                if plane.stride < row {
                    return Err(ExrError::invalid("OpenEXR: frame stride below row size"));
                }
                let mut tight = Vec::with_capacity(row * height as usize);
                for y in 0..height as usize {
                    let start = y * plane.stride;
                    let src = plane
                        .data
                        .get(start..start + row)
                        .ok_or_else(|| ExrError::invalid("OpenEXR: frame plane too short"))?;
                    tight.extend_from_slice(src);
                }
                ExrImage::from_8bit(width, height, &tight, target, input_gamma)?
            }
            other => {
                return Err(ExrError::unsupported(format!(
                    "OpenEXR: pixel format {other:?} not supported (RgbaF32Le / RgbF32Le / \
                     GrayF32Le / Rgb24 / Rgba)"
                )))
            }
        };
        let sig = frame
            .color_signal()
            .unwrap_or_default()
            .or(params.color_signal);
        if !sig.is_unspecified() {
            let color = from_color_signal(&sig);
            img = match crate::image::ColorInfo::chromaticities_for(color.primaries) {
                Some(c) if color.primaries != ColorInfo::PRIMARIES_BT709 => {
                    img.with_chromaticities(c)
                }
                _ => img,
            };
            img.color = color;
        }
        Ok(img)
    }
}

impl TryFrom<(&VideoFrame, &CodecParameters)> for ExrImage {
    type Error = ExrError;
    fn try_from((frame, params): (&VideoFrame, &CodecParameters)) -> Result<Self, ExrError> {
        ExrImage::from_video_frame(frame, params)
    }
}

// ---- registration ---------------------------------------------------------

/// Register the OpenEXR codec into the supplied [`CodecRegistry`].
pub fn register_codecs(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let mut formats = FRAME_FORMATS.to_vec();
    formats.extend(RAW_FORMATS);
    let caps = CodecCapabilities::video("openexr_sw")
        .with_intra_only(true)
        .with_lossless(true)
        .with_max_size(65535, 65535)
        .with_pixel_formats(formats);
    reg.register(
        CodecInfo::new(cid)
            .capabilities(caps)
            .decoder(make_decoder)
            .decoder_options::<DecodeOptions>()
            .encoder(make_encoder)
            .encoder_options::<EncodeOptions>(),
    );
}

/// Register the `openexr` container ([`crate::container`]): the magic
/// probe, the demuxer (one packet per viewable part, each a single-part
/// file), the muxer (one packet → that file; several → a multi-part
/// file) and the `.exr` extension.
pub fn register_containers(reg: &mut ContainerRegistry) {
    crate::container::register(reg);
}

/// Register codecs and containers into two separate registries.
pub fn register_registries(codecs: &mut CodecRegistry, containers: &mut ContainerRegistry) {
    register_codecs(codecs);
    register_containers(containers);
}

/// Unified entry point: install every codec and container provided by
/// `oxideav-openexr` into a [`RuntimeContext`]. Also wired into
/// `oxideav_meta::register_all` via the [`oxideav_core::register!`]
/// macro below.
pub fn register(ctx: &mut RuntimeContext) {
    register_registries(&mut ctx.codecs, &mut ctx.containers);
}

oxideav_core::register!("openexr", register);

// ---- options schemas ------------------------------------------------------

impl CodecOptionsStruct for DecodeOptions {
    const SCHEMA: &'static [OptionField] = &[
        OptionField {
            name: "part",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "zero-based part index to decode from a multi-part file",
        },
        OptionField {
            name: "layer",
            kind: OptionKind::String,
            default: OptionValue::String(String::new()),
            help: "layer (channel-name prefix such as diffuse or right, or the default view's \
                   name) to decode; empty = the base layer of unprefixed channels",
        },
        OptionField {
            name: "part_name",
            kind: OptionKind::String,
            default: OptionValue::String(String::new()),
            help: "part to decode by its name attribute (multi-part files); overrides part when \
                   set",
        },
    ];
    fn apply(&mut self, key: &str, value: &OptionValue) -> oxideav_core::Result<()> {
        match key {
            "part" => self.part = value.as_u32()?,
            "layer" => self.layer = value.as_str()?.to_string(),
            "part_name" => self.part_name = value.as_str()?.to_string(),
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

const PIXEL_TYPE_NAMES: [&str; 2] = ["float", "half"];
const COLOUR_NAMES: [&str; 2] = ["rgb", "luma_chroma"];
const LEVEL_NAMES: [&str; 3] = ["one", "mipmap", "ripmap"];
const LINE_ORDER_NAMES: [&str; 3] = ["increasing_y", "decreasing_y", "random_y"];
const COMPRESSION_NAMES: [&str; 10] = [
    "none", "rle", "zips", "zip", "piz", "pxr24", "b44", "b44a", "dwaa", "dwab",
];

fn compression_from_name(name: &str) -> Option<Compression> {
    Some(match name {
        "none" => Compression::None,
        "rle" => Compression::Rle,
        "zips" => Compression::Zips,
        "zip" => Compression::Zip,
        "piz" => Compression::Piz,
        "pxr24" => Compression::Pxr24,
        "b44" => Compression::B44,
        "b44a" => Compression::B44a,
        "dwaa" => Compression::Dwaa,
        "dwab" => Compression::Dwab,
        _ => return None,
    })
}

impl CodecOptionsStruct for EncodeOptions {
    const SCHEMA: &'static [OptionField] = &[
        OptionField {
            name: "pixel_type",
            kind: OptionKind::Enum(&PIXEL_TYPE_NAMES),
            default: OptionValue::String(String::new()),
            help: "channel pixel type: float (binary32, lossless) or half (binary16)",
        },
        OptionField {
            name: "compression",
            kind: OptionKind::Enum(&COMPRESSION_NAMES),
            default: OptionValue::String(String::new()),
            help: "scanline compression: none, rle, zips, zip, piz, pxr24, b44, b44a, dwaa, dwab",
        },
        OptionField {
            name: "colour",
            kind: OptionKind::Enum(&COLOUR_NAMES),
            default: OptionValue::String(String::new()),
            help: "colour channel layout: rgb (R G B [A]) or luma_chroma (Y RY BY [A], chroma \
                   sub-sampled by chroma_sampling)",
        },
        OptionField {
            name: "chroma_sampling",
            kind: OptionKind::U32,
            default: OptionValue::U32(2),
            help: "RY/BY sampling factor for colour=luma_chroma (1 = full-resolution chroma); \
                   the frame width and height must be multiples of it",
        },
        OptionField {
            name: "layer",
            kind: OptionKind::String,
            default: OptionValue::String(String::new()),
            help: "layer prefix for the written channel names (diffuse -> diffuse.R ...); empty \
                   = unprefixed",
        },
        OptionField {
            name: "tile_size",
            kind: OptionKind::U32,
            default: OptionValue::U32(0),
            help: "tile edge in pixels for a tiled file; 0 = scanline file",
        },
        OptionField {
            name: "levels",
            kind: OptionKind::Enum(&LEVEL_NAMES),
            default: OptionValue::String(String::new()),
            help: "tiled level mode: one, mipmap or ripmap (reduced levels box-filtered from the \
                   frame); requires tile_size > 0",
        },
        OptionField {
            name: "line_order",
            kind: OptionKind::Enum(&LINE_ORDER_NAMES),
            default: OptionValue::String(String::new()),
            help: "chunk storage order: increasing_y, decreasing_y, or random_y (tiled only)",
        },
        OptionField {
            name: "input_gamma",
            kind: OptionKind::F32,
            default: OptionValue::F32(0.0),
            help: "linearisation exponent for Rgb24 / Rgba input frames ((b / 255) ^ gamma); 0 \
                   = bytes are linear",
        },
    ];
    fn apply(&mut self, key: &str, value: &OptionValue) -> oxideav_core::Result<()> {
        match key {
            "colour" => {
                self.colour = match value.as_str()? {
                    "rgb" => ColourLayout::Rgb,
                    "luma_chroma" => ColourLayout::LumaChroma,
                    other => {
                        return Err(oxideav_core::Error::invalid(format!(
                            "OpenEXR encoder: unknown colour layout '{other}'"
                        )))
                    }
                }
            }
            "tile_size" => self.tile_size = value.as_u32()?,
            "levels" => {
                self.levels = match value.as_str()? {
                    "one" => LevelMode::One,
                    "mipmap" => LevelMode::Mipmap,
                    "ripmap" => LevelMode::Ripmap,
                    other => {
                        return Err(oxideav_core::Error::invalid(format!(
                            "OpenEXR encoder: unknown level mode '{other}'"
                        )))
                    }
                }
            }
            "line_order" => {
                self.line_order = match value.as_str()? {
                    "increasing_y" => LineOrder::IncreasingY,
                    "decreasing_y" => LineOrder::DecreasingY,
                    "random_y" => LineOrder::RandomY,
                    other => {
                        return Err(oxideav_core::Error::invalid(format!(
                            "OpenEXR encoder: unknown line order '{other}'"
                        )))
                    }
                }
            }
            "layer" => {
                let v = value.as_str()?;
                if v.ends_with('.') || v.starts_with('.') || v.contains("..") {
                    return Err(oxideav_core::Error::invalid(format!(
                        "OpenEXR encoder: layer '{v}' must not start or end with '.' or contain \
                         an empty component"
                    )));
                }
                self.layer = v.to_string();
            }
            "chroma_sampling" => {
                let v = value.as_u32()?;
                if v == 0 {
                    return Err(oxideav_core::Error::invalid(
                        "OpenEXR encoder: chroma_sampling must be >= 1",
                    ));
                }
                self.chroma_sampling = v;
            }
            "pixel_type" => {
                self.pixel_type = match value.as_str()? {
                    "float" => PixelType::Float,
                    "half" => PixelType::Half,
                    other => {
                        return Err(oxideav_core::Error::invalid(format!(
                            "OpenEXR encoder: unknown pixel_type '{other}'"
                        )))
                    }
                }
            }
            "compression" => {
                let name = value.as_str()?;
                self.compression = compression_from_name(name).ok_or_else(|| {
                    oxideav_core::Error::invalid(format!(
                        "OpenEXR encoder: unknown compression '{name}'"
                    ))
                })?;
            }
            "input_gamma" => {
                let g = value.as_f32()?;
                self.input_gamma = if g > 0.0 && g.is_finite() {
                    Some(g)
                } else {
                    None
                };
            }
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

// ---- Decoder --------------------------------------------------------------

/// Build the framework decoder (one packet = one OpenEXR file; options:
/// `part`, `part_name`, `layer`).
pub fn make_decoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Decoder>> {
    let opts: DecodeOptions = parse_options(&params.options)?;
    Ok(Box::new(ExrDecoder {
        codec_id: CodecId::new(CODEC_ID_STR),
        opts,
        pending: None,
        eof: false,
    }))
}

struct ExrDecoder {
    codec_id: CodecId,
    opts: DecodeOptions,
    pending: Option<VideoFrame>,
    eof: bool,
}

impl Decoder for ExrDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }
    fn send_packet(&mut self, packet: &Packet) -> oxideav_core::Result<()> {
        let img = crate::decode_with(&packet.data, &self.opts)?;
        self.pending = Some(image_into_video_frame(img, packet.pts));
        Ok(())
    }
    fn receive_frame(&mut self) -> oxideav_core::Result<Frame> {
        match self.pending.take() {
            Some(f) => Ok(Frame::Video(f)),
            None => {
                if self.eof {
                    Err(oxideav_core::Error::Eof)
                } else {
                    Err(oxideav_core::Error::NeedMore)
                }
            }
        }
    }
    fn flush(&mut self) -> oxideav_core::Result<()> {
        self.eof = true;
        Ok(())
    }
}

// ---- Encoder --------------------------------------------------------------

/// Build the framework encoder (`width`, `height`, `pixel_format`
/// required; options per the [`EncodeOptions`] schema).
pub fn make_encoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Encoder>> {
    let opts: EncodeOptions = parse_options(&params.options)?;
    let mut out_params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    out_params.width = params.width;
    out_params.height = params.height;
    out_params.pixel_format = params.pixel_format;
    out_params.color_signal = params.color_signal;
    Ok(Box::new(ExrEncoder {
        codec_id: CodecId::new(CODEC_ID_STR),
        out_params,
        opts,
        pending: None,
        eof: false,
    }))
}

struct ExrEncoder {
    codec_id: CodecId,
    out_params: CodecParameters,
    opts: EncodeOptions,
    pending: Option<Vec<u8>>,
    eof: bool,
}

impl Encoder for ExrEncoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }
    fn output_params(&self) -> &CodecParameters {
        &self.out_params
    }
    fn send_frame(&mut self, frame: &Frame) -> oxideav_core::Result<()> {
        let vf = match frame {
            Frame::Video(v) => v,
            _ => {
                return Err(oxideav_core::Error::invalid(
                    "OpenEXR encoder: expected video frame",
                ))
            }
        };
        let format = self.out_params.pixel_format.ok_or_else(|| {
            oxideav_core::Error::invalid("OpenEXR encoder: pixel_format missing in CodecParameters")
        })?;
        if !FRAME_FORMATS.contains(&format) && !RAW_FORMATS.contains(&format) {
            return Err(oxideav_core::Error::invalid(format!(
                "OpenEXR encoder: unsupported pixel format {format:?} (RgbaF32Le / RgbF32Le / \
                 GrayF32Le / Rgb24 / Rgba)"
            )));
        }
        let img =
            ExrImage::from_video_frame_with_gamma(vf, &self.out_params, self.opts.input_gamma)?;
        self.pending = Some(crate::encode(&img, &self.opts)?);
        Ok(())
    }
    fn receive_packet(&mut self) -> oxideav_core::Result<Packet> {
        match self.pending.take() {
            Some(bytes) => {
                let mut pkt = Packet::new(0, TimeBase::new(1, 1), bytes);
                pkt.flags.keyframe = true;
                Ok(pkt)
            }
            None => {
                if self.eof {
                    Err(oxideav_core::Error::Eof)
                } else {
                    Err(oxideav_core::Error::NeedMore)
                }
            }
        }
    }
    fn flush(&mut self) -> oxideav_core::Result<()> {
        self.eof = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::{
        encode_exr_scanline, required_scanline_attributes as scanline_attributes,
    };
    use crate::half::{f32_to_half, half_to_f32};
    use crate::parse_exr;
    use crate::types::{Attribute, AttributeValue, Channel};

    /// RGBA float scanline file through the contract encoder (the
    /// pre-contract `encode_exr_scanline_rgba_float` shape).
    fn encode_exr_scanline_rgba_float(w: u32, h: u32, samples: &[f32]) -> crate::Result<Vec<u8>> {
        let img = ExrImage::from_f32(w, h, ExrPixelFormat::RgbaF32Le, samples)?;
        crate::encode(&img, &EncodeOptions::default())
    }

    fn decode_frame(bytes: Vec<u8>, part: Option<u32>) -> oxideav_core::Result<VideoFrame> {
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        if let Some(p) = part {
            params.options.insert("part", p.to_string());
        }
        let mut dec = make_decoder(&params)?;
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), bytes))?;
        match dec.receive_frame()? {
            Frame::Video(v) => Ok(v),
            _ => panic!("expected video frame"),
        }
    }

    fn f32_at(vf: &VideoFrame, px: usize, comps: usize, c: usize) -> f32 {
        let b = px * comps * 4 + c * 4;
        f32::from_le_bytes(vf.planes[0].data[b..b + 4].try_into().unwrap())
    }

    fn packed_frame(w: u32, h: u32, comps: usize, f: impl Fn(usize, usize) -> f32) -> VideoFrame {
        let mut data = Vec::with_capacity((w * h) as usize * comps * 4);
        for px in 0..(w * h) as usize {
            for c in 0..comps {
                data.extend_from_slice(&f(px, c).to_le_bytes());
            }
        }
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: w as usize * comps * 4,
                data,
            }],
        }
    }

    fn encode_frame(
        vf: VideoFrame,
        w: u32,
        h: u32,
        format: PixelFormat,
        opts: &[(&str, &str)],
    ) -> oxideav_core::Result<Vec<u8>> {
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(w);
        params.height = Some(h);
        params.pixel_format = Some(format);
        for (k, v) in opts {
            params.options.insert(*k, v.to_string());
        }
        let mut enc = make_encoder(&params)?;
        enc.send_frame(&Frame::Video(vf))?;
        Ok(enc.receive_packet()?.data)
    }

    #[test]
    fn exr_extension_resolves_to_openexr_container() {
        let mut reg = ContainerRegistry::new();
        register_containers(&mut reg);
        assert_eq!(reg.container_for_extension("exr"), Some(CODEC_ID_STR));
    }

    #[test]
    fn exr_extension_lookup_is_case_insensitive() {
        let mut reg = ContainerRegistry::new();
        register_containers(&mut reg);
        assert_eq!(reg.container_for_extension("EXR"), Some(CODEC_ID_STR));
        assert_eq!(reg.container_for_extension("Exr"), Some(CODEC_ID_STR));
        assert_eq!(reg.container_for_extension("eXr"), Some(CODEC_ID_STR));
    }

    #[test]
    fn register_via_runtime_context_installs_factories() {
        let mut ctx = RuntimeContext::new();
        register(&mut ctx);
        assert!(
            ctx.codecs.decoder_ids().next().is_some(),
            "register(ctx) should install codec decoder factories"
        );
        assert_eq!(
            ctx.containers.container_for_extension("exr"),
            Some(CODEC_ID_STR),
            "register(ctx) should install .exr extension hint"
        );
        let cid = CodecId::new(CODEC_ID_STR);
        assert!(ctx.codecs.encoder_options_schema(&cid).is_some());
        assert!(ctx.codecs.decoder_options_schema(&cid).is_some());
    }

    #[test]
    fn capabilities_advertise_the_float_family() {
        let mut reg = CodecRegistry::new();
        register_codecs(&mut reg);
        let impls = reg.implementations(&CodecId::new(CODEC_ID_STR));
        assert_eq!(impls.len(), 1, "openexr registered once");
        let caps = &impls[0].caps;
        assert_eq!(&caps.accepted_pixel_formats[..3], &FRAME_FORMATS[..]);
        assert!(caps.accepted_pixel_formats[..3]
            .iter()
            .all(|f| f.is_float()));
        assert_eq!(&caps.accepted_pixel_formats[3..], &RAW_FORMATS[..]);
    }

    #[test]
    fn decoder_emits_rgba_f32_without_clamping() {
        // HDR values outside [0, 1] and negative excursions must
        // survive: no tone-map, no clamp.
        let (w, h) = (2u32, 2u32);
        let mut samples = vec![0.0f32; (w * h * 4) as usize];
        samples[0] = 12.5; // px0 R — specular above white
        samples[3] = 1.0;
        samples[4 + 1] = -0.25; // px1 G — out-of-gamut negative
        samples[4 + 3] = 0.5;
        samples[8 + 2] = 1e-7; // px2 B — tiny linear value
        samples[8 + 3] = 1.0;
        samples[12] = 65504.0; // px3 R — largest finite half
        samples[12 + 3] = 1.0;
        let bytes = encode_exr_scanline_rgba_float(w, h, &samples).unwrap();

        let vf = decode_frame(bytes, None).unwrap();
        assert_eq!(vf.planes[0].stride, (w as usize) * 16);
        assert_eq!(vf.planes[0].data.len(), (w * h) as usize * 16);
        for px in 0..4 {
            for c in 0..4 {
                assert_eq!(
                    f32_at(&vf, px, 4, c).to_bits(),
                    samples[px * 4 + c].to_bits(),
                    "px{px} c{c}"
                );
            }
        }
    }

    #[test]
    fn decoder_maps_rgb_without_alpha_to_rgb_f32() {
        let (w, h) = (3u32, 1u32);
        let chs: Vec<Channel> = ["B", "G", "R"]
            .iter()
            .map(|n| Channel {
                name: n.to_string(),
                pixel_type: PixelType::Float,
                p_linear: false,
                x_sampling: 1,
                y_sampling: 1,
            })
            .collect();
        let b = [0.1f32, 0.2, 0.3];
        let g = [1.5f32, 2.5, 3.5];
        let r = [-1.0f32, 0.0, 7.0];
        let attrs = scanline_attributes(w, h, &chs, Compression::None);
        let bytes =
            encode_exr_scanline(w, h, &chs, &[&b, &g, &r], Compression::None, attrs).unwrap();
        let vf = decode_frame(bytes, None).unwrap();
        assert_eq!(vf.planes[0].stride, 3 * 12);
        for px in 0..3 {
            assert_eq!(f32_at(&vf, px, 3, 0), r[px]);
            assert_eq!(f32_at(&vf, px, 3, 1), g[px]);
            assert_eq!(f32_at(&vf, px, 3, 2), b[px]);
        }
    }

    #[test]
    fn decoder_maps_y_to_gray_f32_and_y_plus_a_to_rgba() {
        let (w, h) = (2u32, 2u32);
        let y = [0.5f32, 4.0, -0.5, 100.0];
        let mk = |n: &str, t: PixelType| Channel {
            name: n.to_string(),
            pixel_type: t,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        };
        // Y only, stored as HALF → exact widening.
        let chs = vec![mk("Y", PixelType::Half)];
        let attrs = scanline_attributes(w, h, &chs, Compression::Zip);
        let bytes = encode_exr_scanline(w, h, &chs, &[&y], Compression::Zip, attrs).unwrap();
        let vf = decode_frame(bytes, None).unwrap();
        assert_eq!(vf.planes[0].stride, 2 * 4);
        for (px, &expect) in y.iter().enumerate() {
            assert_eq!(f32_at(&vf, px, 1, 0), expect);
        }
        // Y + A → RgbaF32Le with Y replicated.
        let a = [1.0f32, 0.75, 0.5, 0.0];
        let chs = vec![mk("A", PixelType::Float), mk("Y", PixelType::Float)];
        let attrs = scanline_attributes(w, h, &chs, Compression::Zip);
        let bytes = encode_exr_scanline(w, h, &chs, &[&a, &y], Compression::Zip, attrs).unwrap();
        let vf = decode_frame(bytes, None).unwrap();
        assert_eq!(vf.planes[0].stride, 2 * 16);
        for px in 0..4 {
            for c in 0..3 {
                assert_eq!(f32_at(&vf, px, 4, c), y[px]);
            }
            assert_eq!(f32_at(&vf, px, 4, 3), a[px]);
        }
    }

    #[test]
    fn decoder_rejects_unmappable_channel_sets() {
        let (w, h) = (2u32, 1u32);
        let mk = |n: &str, xs: i32| Channel {
            name: n.to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: xs,
            y_sampling: 1,
        };
        // Depth-only part.
        let chs = vec![mk("Z", 1)];
        let z = [1.0f32, 2.0];
        let attrs = scanline_attributes(w, h, &chs, Compression::None);
        let bytes = encode_exr_scanline(w, h, &chs, &[&z], Compression::None, attrs).unwrap();
        let err = decode_frame(bytes, None).unwrap_err();
        assert!(
            matches!(err, oxideav_core::Error::Unsupported(_)),
            "{err:?}"
        );
        // Partial colour triple (R + G, no B).
        let chs = vec![mk("G", 1), mk("R", 1)];
        let attrs = scanline_attributes(w, h, &chs, Compression::None);
        let bytes = encode_exr_scanline(w, h, &chs, &[&z, &z], Compression::None, attrs).unwrap();
        assert!(matches!(
            decode_frame(bytes, None).unwrap_err(),
            oxideav_core::Error::Unsupported(_)
        ));
        // Lone chroma channel without its partner.
        let chs = vec![mk("RY", 1), mk("Y", 1)];
        let attrs = scanline_attributes(w, h, &chs, Compression::None);
        let bytes = encode_exr_scanline(w, h, &chs, &[&z, &z], Compression::None, attrs).unwrap();
        assert!(matches!(
            decode_frame(bytes, None).unwrap_err(),
            oxideav_core::Error::Unsupported(_)
        ));
        // Sub-sampled Y.
        let chs = vec![mk("Y", 2)];
        let ysub = [3.0f32];
        let attrs = scanline_attributes(w, h, &chs, Compression::None);
        let bytes = encode_exr_scanline(w, h, &chs, &[&ysub], Compression::None, attrs).unwrap();
        assert!(matches!(
            decode_frame(bytes, None).unwrap_err(),
            oxideav_core::Error::Unsupported(_)
        ));
    }

    #[test]
    fn decoder_reconstructs_rgb_from_luma_chroma() {
        use crate::luma_chroma::{luminance_weights, rgb_to_luma_chroma, BT709_CHROMATICITIES};
        let (w, h) = (6u32, 4u32);
        let n = (w * h) as usize;
        // Constant chroma so the 2×2 reduction is exact everywhere.
        let k = |i: usize| 0.1 + (i as f32) * 0.2;
        let r: Vec<f32> = (0..n).map(|i| 0.7 * k(i)).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.4 * k(i)).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.2 * k(i)).collect();
        let wts = luminance_weights(&BT709_CHROMATICITIES);
        let yc = rgb_to_luma_chroma(w, h, [&r, &g, &b], wts, (2, 2)).unwrap();
        let mk = |name: &str, s: i32| Channel {
            name: name.to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: s,
            y_sampling: s,
        };
        let chs = vec![mk("BY", 2), mk("RY", 2), mk("Y", 1)];
        let attrs = scanline_attributes(w, h, &chs, Compression::Zip);
        let bytes = encode_exr_scanline(
            w,
            h,
            &chs,
            &[&yc.by, &yc.ry, &yc.y],
            Compression::Zip,
            attrs,
        )
        .unwrap();
        let vf = decode_frame(bytes, None).unwrap();
        assert_eq!(vf.planes[0].stride, w as usize * 12);
        for px in 0..n {
            for (c, expect) in [r[px], g[px], b[px]].iter().enumerate() {
                let got = f32_at(&vf, px, 3, c);
                assert!(
                    (got - expect).abs() <= 1e-5 * (1.0 + expect.abs()),
                    "px{px} c{c} {got} vs {expect}"
                );
            }
        }
        // With alpha → RgbaF32Le; chroma at 1×1 is fine too.
        let a: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32)).collect();
        let yc1 = rgb_to_luma_chroma(w, h, [&r, &g, &b], wts, (1, 1)).unwrap();
        let chs = vec![mk("A", 1), mk("BY", 1), mk("RY", 1), mk("Y", 1)];
        let attrs = scanline_attributes(w, h, &chs, Compression::None);
        let bytes = encode_exr_scanline(
            w,
            h,
            &chs,
            &[&a, &yc1.by, &yc1.ry, &yc1.y],
            Compression::None,
            attrs,
        )
        .unwrap();
        let vf = decode_frame(bytes, None).unwrap();
        assert_eq!(vf.planes[0].stride, w as usize * 16);
        for px in 0..n {
            assert!((f32_at(&vf, px, 4, 0) - r[px]).abs() <= 1e-5 * (1.0 + r[px]));
            assert_eq!(f32_at(&vf, px, 4, 3), a[px]);
        }
    }

    #[test]
    fn decoder_honours_the_chromaticities_attribute_for_luma_chroma() {
        use crate::luma_chroma::{luminance_weights, rgb_to_luma_chroma};
        use crate::types::Chromaticities;
        let wide = Chromaticities {
            red_x: 0.7347,
            red_y: 0.2653,
            green_x: 0.0,
            green_y: 1.0,
            blue_x: 0.0001,
            blue_y: -0.077,
            white_x: 0.32168,
            white_y: 0.33767,
        };
        let (w, h) = (4u32, 2u32);
        let n = (w * h) as usize;
        let r: Vec<f32> = (0..n).map(|i| 0.3 + 0.05 * i as f32).collect();
        let g: Vec<f32> = (0..n).map(|i| 0.9 - 0.04 * i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.1 + 0.08 * i as f32).collect();
        let wts = luminance_weights(&wide);
        let yc = rgb_to_luma_chroma(w, h, [&r, &g, &b], wts, (1, 1)).unwrap();
        let mk = |name: &str| Channel {
            name: name.to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        };
        let chs = vec![mk("BY"), mk("RY"), mk("Y")];
        let mut attrs = scanline_attributes(w, h, &chs, Compression::None);
        attrs.push(Attribute {
            name: "chromaticities".to_string(),
            value: AttributeValue::Chromaticities(wide),
        });
        let bytes = encode_exr_scanline(
            w,
            h,
            &chs,
            &[&yc.by, &yc.ry, &yc.y],
            Compression::None,
            attrs,
        )
        .unwrap();
        let vf = decode_frame(bytes, None).unwrap();
        for (px, &expect) in g.iter().enumerate() {
            // G depends on the weights: BT.709 weights would be off by
            // far more than the tolerance here.
            let got = f32_at(&vf, px, 3, 1);
            assert!(
                (got - expect).abs() <= 1e-5 * (1.0 + expect),
                "px{px} G {got} vs {expect}"
            );
        }
    }

    #[test]
    fn decoder_layer_option_selects_prefixed_channels() {
        let (w, h) = (3u32, 2u32);
        let n = (w * h) as usize;
        let mk = |name: &str| Channel {
            name: name.to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        };
        // Base RGBA + a `diffuse` RGB layer + a `right` gray+alpha
        // layer + a depth-only layer, all in one part.
        let names = [
            "A",
            "B",
            "G",
            "R",
            "depth.Z",
            "diffuse.B",
            "diffuse.G",
            "diffuse.R",
            "right.A",
            "right.Y",
        ];
        let chs: Vec<Channel> = names.iter().map(|n| mk(n)).collect();
        let planes: Vec<Vec<f32>> = (0..names.len())
            .map(|c| (0..n).map(|px| (c * 100 + px) as f32).collect())
            .collect();
        let refs: Vec<&[f32]> = planes.iter().map(|p| p.as_slice()).collect();
        let mut attrs = scanline_attributes(w, h, &chs, Compression::Zips);
        attrs.push(Attribute {
            name: "multiView".to_string(),
            value: AttributeValue::StringVector(vec!["left".to_string(), "right".to_string()]),
        });
        let bytes = encode_exr_scanline(w, h, &chs, &refs, Compression::Zips, attrs).unwrap();

        let decode_layer = |layer: &str| {
            let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
            params.options.insert("layer", layer.to_string());
            let mut dec = make_decoder(&params).unwrap();
            dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), bytes.clone()))?;
            match dec.receive_frame()? {
                Frame::Video(v) => Ok::<_, oxideav_core::Error>(v),
                _ => panic!("expected video frame"),
            }
        };
        // Base layer: RGBA (R is channel 3, A channel 0).
        let vf = decode_layer("").unwrap();
        assert_eq!(vf.planes[0].stride, w as usize * 16);
        assert_eq!(f32_at(&vf, 4, 4, 0), 304.0);
        assert_eq!(f32_at(&vf, 4, 4, 3), 4.0);
        // The default view name resolves to the base layer too.
        let vf = decode_layer("left").unwrap();
        assert_eq!(f32_at(&vf, 1, 4, 1), 201.0);
        // diffuse → RGB from the prefixed channels.
        let vf = decode_layer("diffuse").unwrap();
        assert_eq!(vf.planes[0].stride, w as usize * 12);
        assert_eq!(f32_at(&vf, 5, 3, 0), 705.0);
        assert_eq!(f32_at(&vf, 5, 3, 1), 605.0);
        assert_eq!(f32_at(&vf, 5, 3, 2), 505.0);
        // right → Y + A replicated into RGBA.
        let vf = decode_layer("right").unwrap();
        assert_eq!(vf.planes[0].stride, w as usize * 16);
        assert_eq!(f32_at(&vf, 2, 4, 0), 902.0);
        assert_eq!(f32_at(&vf, 2, 4, 2), 902.0);
        assert_eq!(f32_at(&vf, 2, 4, 3), 802.0);
        // depth → Unsupported (no colour), unknown → InvalidData naming
        // the layers.
        assert!(matches!(
            decode_layer("depth").unwrap_err(),
            oxideav_core::Error::Unsupported(_)
        ));
        let err = decode_layer("beauty").unwrap_err();
        match err {
            oxideav_core::Error::InvalidData(msg) => {
                assert!(msg.contains("diffuse (Rgb)"), "{msg}");
                assert!(msg.contains("right (GrayAlpha, view right)"), "{msg}");
                assert!(msg.contains("depth (Depth)"), "{msg}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn encoder_layer_option_prefixes_channel_names() {
        let (w, h) = (4u32, 2u32);
        let value = |px: usize, c: usize| px as f32 + c as f32 * 0.25;
        let src = packed_frame(w, h, 4, value);
        let bytes = encode_frame(
            src.clone(),
            w,
            h,
            PixelFormat::RgbaF32Le,
            &[("layer", "beauty.spec")],
        )
        .unwrap();
        let img = parse_exr(&bytes).unwrap();
        let names: Vec<&str> = img.channels.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "beauty.spec.A",
                "beauty.spec.B",
                "beauty.spec.G",
                "beauty.spec.R"
            ]
        );
        let layers = img.layers();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].name, "beauty.spec");
        assert_eq!(layers[0].kind, crate::layers::LayerKind::Rgba);
        // The base layer is empty now, so a default decode fails
        // loudly (no default colour view = Unsupported) and the
        // prefixed decode round-trips.
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.options.insert("layer", "beauty.spec");
        let mut dec = make_decoder(&params).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), bytes.clone()))
            .unwrap();
        match dec.receive_frame().unwrap() {
            Frame::Video(v) => assert_eq!(v.planes[0].data, src.planes[0].data),
            _ => panic!(),
        }
        assert!(matches!(
            decode_frame(bytes, None).unwrap_err(),
            oxideav_core::Error::Unsupported(_)
        ));
        // Luma/chroma under a layer prefix.
        let src = packed_frame(w, h, 3, |_, c| [0.5f32, 0.25, 0.125][c]);
        let bytes = encode_frame(
            src,
            w,
            h,
            PixelFormat::RgbF32Le,
            &[("layer", "left"), ("colour", "luma_chroma")],
        )
        .unwrap();
        let img = parse_exr(&bytes).unwrap();
        let names: Vec<&str> = img.channels.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["left.BY", "left.RY", "left.Y"]);
        assert_eq!(img.layers()[0].kind, crate::layers::LayerKind::LumaChroma);
        // Malformed prefixes are rejected at construction.
        for bad in [".x", "x.", "a..b"] {
            let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
            params.options.insert("layer", bad);
            assert!(make_encoder(&params).is_err(), "{bad}");
        }
    }

    #[test]
    fn decoder_rejects_deep_files_as_unsupported() {
        use crate::deep::{encode_exr_deep_scanline, DeepScanlineInput};
        let ch = Channel {
            name: "R".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        };
        let samples = vec![1.0f32, 2.0, 3.0];
        let input = DeepScanlineInput {
            width: 1,
            height: 1,
            channels: vec![ch],
            samples_per_pixel: &[3],
            channel_samples: vec![&samples],
            compression: Compression::None,
        };
        let bytes = encode_exr_deep_scanline(&input).unwrap();
        let err = decode_frame(bytes, None).unwrap_err();
        assert!(
            matches!(err, oxideav_core::Error::Unsupported(_)),
            "{err:?}"
        );
    }

    #[test]
    fn decoder_part_option_selects_multipart_part() {
        use crate::multipart_encoder::{encode_exr_multipart, MultipartScanlinePart};
        let (w, h) = (2u32, 1u32);
        let mk = |n: &str| Channel {
            name: n.to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        };
        let p0 = [0.25f32, 0.5];
        let p1 = [8.0f32, 16.0];
        let parts = vec![
            MultipartScanlinePart {
                name: "left".to_string(),
                width: w,
                height: h,
                channels: vec![mk("Y")],
                planes: vec![&p0],
                compression: Compression::None,
            },
            MultipartScanlinePart {
                name: "right".to_string(),
                width: w,
                height: h,
                channels: vec![mk("Y")],
                planes: vec![&p1],
                compression: Compression::Rle,
            },
        ];
        let bytes = encode_exr_multipart(&parts).unwrap();
        let vf = decode_frame(bytes.clone(), None).unwrap();
        assert_eq!(f32_at(&vf, 1, 1, 0), 0.5);
        let vf = decode_frame(bytes.clone(), Some(1)).unwrap();
        assert_eq!(f32_at(&vf, 0, 1, 0), 8.0);
        assert!(matches!(
            decode_frame(bytes.clone(), Some(2)).unwrap_err(),
            oxideav_core::Error::InvalidData(_)
        ));
        // By name, overriding `part`.
        let by_name = |name: &str| {
            let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
            params.options.insert("part", "0");
            params.options.insert("part_name", name);
            let mut dec = make_decoder(&params)?;
            dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), bytes.clone()))?;
            match dec.receive_frame()? {
                Frame::Video(v) => Ok::<_, oxideav_core::Error>(v),
                _ => panic!("expected video frame"),
            }
        };
        assert_eq!(f32_at(&by_name("right").unwrap(), 1, 1, 0), 16.0);
        assert_eq!(f32_at(&by_name("left").unwrap(), 1, 1, 0), 0.5);
        match by_name("centre").unwrap_err() {
            oxideav_core::Error::InvalidData(msg) => {
                assert!(msg.contains("0: \"left\""), "{msg}");
                assert!(msg.contains("1: \"right\""), "{msg}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn encoder_float_roundtrip_is_bit_exact_for_every_format_and_lossless_codec() {
        let (w, h) = (5u32, 3u32);
        let value = |px: usize, c: usize| ((px as f32) - 4.0) * 3.75 + (c as f32) * 0.125;
        for &codec in &["none", "rle", "zips", "zip", "piz"] {
            for &(format, comps) in &[
                (PixelFormat::RgbaF32Le, 4usize),
                (PixelFormat::RgbF32Le, 3),
                (PixelFormat::GrayF32Le, 1),
            ] {
                let src = packed_frame(w, h, comps, value);
                let bytes = encode_frame(
                    src.clone(),
                    w,
                    h,
                    format,
                    &[("pixel_type", "float"), ("compression", codec)],
                )
                .unwrap();
                let img = parse_exr(&bytes).unwrap();
                assert!(img
                    .channels
                    .iter()
                    .all(|c| c.pixel_type == PixelType::Float));
                assert_eq!(img.channels.len(), comps, "{format:?}/{codec}");
                let vf = decode_frame(bytes, None).unwrap();
                assert_eq!(vf.planes[0].data, src.planes[0].data, "{format:?}/{codec}");
            }
        }
    }

    #[test]
    fn encoder_half_roundtrip_is_half_rounded() {
        let (w, h) = (4u32, 2u32);
        let value = |px: usize, c: usize| 1.0 / (1.0 + px as f32) * 1.0001 + c as f32 * 1.3333;
        for &codec in &["none", "zip", "piz"] {
            let src = packed_frame(w, h, 4, value);
            let bytes = encode_frame(
                src,
                w,
                h,
                PixelFormat::RgbaF32Le,
                &[("pixel_type", "half"), ("compression", codec)],
            )
            .unwrap();
            let img = parse_exr(&bytes).unwrap();
            assert!(img.channels.iter().all(|c| c.pixel_type == PixelType::Half));
            let vf = decode_frame(bytes, None).unwrap();
            for px in 0..(w * h) as usize {
                for c in 0..4 {
                    let expect = half_to_f32(f32_to_half(value(px, c)));
                    assert_eq!(f32_at(&vf, px, 4, c), expect, "{codec} px{px} c{c}");
                }
            }
        }
    }

    #[test]
    fn encoder_accepts_every_lossy_codec_in_half_and_float() {
        // Lossy schemes: only check that the encode succeeds and decodes
        // back to the right geometry/format; exactness is covered by the
        // per-codec validation suites.
        let (w, h) = (16u32, 16u32);
        let value = |px: usize, c: usize| (px % 16) as f32 / 16.0 + c as f32 * 0.01;
        for &codec in &["pxr24", "b44", "b44a", "dwaa", "dwab"] {
            for &pt in &["half", "float"] {
                let src = packed_frame(w, h, 3, value);
                let bytes = encode_frame(
                    src,
                    w,
                    h,
                    PixelFormat::RgbF32Le,
                    &[("pixel_type", pt), ("compression", codec)],
                )
                .unwrap();
                let vf = decode_frame(bytes, None).unwrap();
                assert_eq!(
                    vf.planes[0].data.len(),
                    (w * h) as usize * 12,
                    "{codec}/{pt}"
                );
                // Loose sanity bound on the lossy reconstruction.
                for px in 0..(w * h) as usize {
                    let got = f32_at(&vf, px, 3, 0);
                    assert!(
                        (got - value(px, 0)).abs() < 0.05,
                        "{codec}/{pt} px{px} {got}"
                    );
                }
            }
        }
    }

    #[test]
    fn encoder_luma_chroma_layout_writes_y_ry_by_and_round_trips() {
        let (w, h) = (8u32, 4u32);
        let n = (w * h) as usize;
        // Constant chroma: the 2×2 reduction is exact so the round
        // trip is tight.
        let base = [0.7f32, 0.45, 0.15];
        let value = |px: usize, c: usize| base[c] * (0.2 + px as f32 * 0.1);
        let src = packed_frame(w, h, 3, value);
        let bytes = encode_frame(
            src,
            w,
            h,
            PixelFormat::RgbF32Le,
            &[("colour", "luma_chroma"), ("compression", "piz")],
        )
        .unwrap();
        let img = parse_exr(&bytes).unwrap();
        let names: Vec<(&str, i32, i32)> = img
            .channels
            .iter()
            .map(|c| (c.name.as_str(), c.x_sampling, c.y_sampling))
            .collect();
        assert_eq!(names, [("BY", 2, 2), ("RY", 2, 2), ("Y", 1, 1)]);
        assert_eq!(img.planes[0].samples.len(), n / 4);
        assert_eq!(img.planes[2].samples.len(), n);
        let vf = decode_frame(bytes, None).unwrap();
        for px in 0..n {
            for c in 0..3 {
                let expect = value(px, c);
                let got = f32_at(&vf, px, 3, c);
                assert!(
                    (got - expect).abs() <= 1e-5 * (1.0 + expect),
                    "px{px} c{c} {got} vs {expect}"
                );
            }
        }
        // RGBA keeps the alpha at full resolution ahead of the chroma.
        let value4 = |px: usize, c: usize| {
            if c == 3 {
                px as f32 / n as f32
            } else {
                value(px, c)
            }
        };
        let src = packed_frame(w, h, 4, value4);
        let bytes = encode_frame(
            src,
            w,
            h,
            PixelFormat::RgbaF32Le,
            &[("colour", "luma_chroma"), ("pixel_type", "half")],
        )
        .unwrap();
        let img = parse_exr(&bytes).unwrap();
        let names: Vec<&str> = img.channels.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["A", "BY", "RY", "Y"]);
        assert!(img.channels.iter().all(|c| c.pixel_type == PixelType::Half));
        let vf = decode_frame(bytes, None).unwrap();
        for px in 0..n {
            assert_eq!(
                f32_at(&vf, px, 4, 3),
                half_to_f32(f32_to_half(value4(px, 3)))
            );
            let expect = value(px, 0);
            assert!((f32_at(&vf, px, 4, 0) - expect).abs() <= 2e-3 * (1.0 + expect));
        }
        // Gray frames write Y under either layout.
        let src = packed_frame(w, h, 1, |px, _| px as f32);
        let bytes = encode_frame(
            src,
            w,
            h,
            PixelFormat::GrayF32Le,
            &[("colour", "luma_chroma")],
        )
        .unwrap();
        let img = parse_exr(&bytes).unwrap();
        assert_eq!(img.channels.len(), 1);
        assert_eq!(img.channels[0].name, "Y");
    }

    #[test]
    fn encoder_luma_chroma_sampling_rules() {
        let (w, h) = (5u32, 3u32);
        let n = (w * h) as usize;
        let value = |px: usize, c: usize| 0.1 + (px % 4) as f32 * 0.2 + c as f32 * 0.3;
        // Odd extents at the default 2×2 sampling are rejected …
        let src = packed_frame(w, h, 3, value);
        let err = encode_frame(
            src,
            w,
            h,
            PixelFormat::RgbF32Le,
            &[("colour", "luma_chroma")],
        )
        .unwrap_err();
        assert!(
            matches!(err, oxideav_core::Error::InvalidData(_)),
            "{err:?}"
        );
        // … and chroma_sampling=1 round-trips any image exactly-ish.
        let src = packed_frame(w, h, 3, value);
        let bytes = encode_frame(
            src,
            w,
            h,
            PixelFormat::RgbF32Le,
            &[("colour", "luma_chroma"), ("chroma_sampling", "1")],
        )
        .unwrap();
        let img = parse_exr(&bytes).unwrap();
        assert!(img
            .channels
            .iter()
            .all(|c| c.x_sampling == 1 && c.y_sampling == 1));
        let vf = decode_frame(bytes, None).unwrap();
        for px in 0..n {
            for c in 0..3 {
                let expect = value(px, c);
                assert!((f32_at(&vf, px, 3, c) - expect).abs() <= 1e-5 * (1.0 + expect));
            }
        }
        // 4×4 chroma on a 8×4 frame.
        let (w, h) = (8u32, 4u32);
        let src = packed_frame(w, h, 3, |_, c| [0.5f32, 0.25, 0.125][c]);
        let bytes = encode_frame(
            src,
            w,
            h,
            PixelFormat::RgbF32Le,
            &[("colour", "luma_chroma"), ("chroma_sampling", "4")],
        )
        .unwrap();
        let img = parse_exr(&bytes).unwrap();
        assert_eq!(img.channels[0].x_sampling, 4);
        assert_eq!(img.planes[0].samples.len(), 2);
        // Bad option values.
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.options.insert("chroma_sampling", "0");
        assert!(make_encoder(&params).is_err());
        params.options = Default::default();
        params.options.insert("colour", "ycbcr");
        assert!(make_encoder(&params).is_err());
    }

    #[test]
    fn encoder_writes_tiled_one_level_mipmap_and_ripmap_files() {
        use crate::decoder::parse_exr_tiled_multilevel;
        let (w, h) = (20u32, 12u32);
        let n = (w * h) as usize;
        let value = |px: usize, c: usize| (px % 7) as f32 * 0.5 - 1.0 + c as f32 * 0.125;
        for &(levels, mode, count) in &[("one", 0u8, 1usize), ("mipmap", 1, 5), ("ripmap", 2, 20)] {
            for &lo in &["increasing_y", "decreasing_y", "random_y"] {
                for &codec in &["none", "zip", "piz", "b44"] {
                    let src = packed_frame(w, h, 3, value);
                    let bytes = encode_frame(
                        src.clone(),
                        w,
                        h,
                        PixelFormat::RgbF32Le,
                        &[
                            ("tile_size", "8"),
                            ("levels", levels),
                            ("line_order", lo),
                            ("compression", codec),
                        ],
                    )
                    .unwrap();
                    let ml = parse_exr_tiled_multilevel(&bytes).unwrap();
                    assert_eq!(ml.level_mode, mode, "{levels}/{lo}/{codec}");
                    assert_eq!((ml.tile_x, ml.tile_y), (8, 8));
                    assert_eq!(ml.levels.len(), count, "{levels}/{lo}/{codec}");
                    // Level (0, 0) decodes through the registry.
                    let vf = decode_frame(bytes, None).unwrap();
                    assert_eq!(vf.planes[0].data.len(), n * 12);
                    if codec != "b44" {
                        assert_eq!(
                            vf.planes[0].data, src.planes[0].data,
                            "{levels}/{lo}/{codec}"
                        );
                    }
                    // The coarsest mipmap level is the box-filtered mean.
                    if levels == "mipmap" {
                        let top = ml.levels.last().unwrap();
                        assert_eq!((top.width, top.height), (1, 1));
                    }
                }
            }
        }
        // Scanline files honour decreasing_y too; random_y is rejected.
        let src = packed_frame(w, h, 3, value);
        let bytes = encode_frame(
            src.clone(),
            w,
            h,
            PixelFormat::RgbF32Le,
            &[("line_order", "decreasing_y")],
        )
        .unwrap();
        assert_eq!(
            parse_exr(&bytes).unwrap().line_order,
            LineOrder::DecreasingY
        );
        assert_eq!(
            decode_frame(bytes, None).unwrap().planes[0].data,
            src.planes[0].data
        );
        let src = packed_frame(w, h, 3, value);
        assert!(encode_frame(
            src,
            w,
            h,
            PixelFormat::RgbF32Le,
            &[("line_order", "random_y")]
        )
        .is_err());
        // levels without tiles, and tiles with sub-sampled chroma, are
        // rejected up front.
        let src = packed_frame(w, h, 3, value);
        assert!(encode_frame(src, w, h, PixelFormat::RgbF32Le, &[("levels", "mipmap")]).is_err());
        let src = packed_frame(w, h, 3, value);
        assert!(encode_frame(
            src,
            w,
            h,
            PixelFormat::RgbF32Le,
            &[("tile_size", "8"), ("colour", "luma_chroma")]
        )
        .is_err());
        let src = packed_frame(w, h, 3, value);
        let bytes = encode_frame(
            src,
            w,
            h,
            PixelFormat::RgbF32Le,
            &[
                ("tile_size", "8"),
                ("colour", "luma_chroma"),
                ("chroma_sampling", "1"),
            ],
        )
        .unwrap();
        let names: Vec<String> = parse_exr(&bytes)
            .unwrap()
            .channels
            .into_iter()
            .map(|c| c.name)
            .collect();
        assert_eq!(names, ["BY", "RY", "Y"]);
    }

    #[test]
    fn encoder_honours_padded_stride() {
        let (w, h) = (3u32, 2u32);
        let row = w as usize * 12;
        let stride = row + 20;
        let mut data = vec![0xAAu8; stride * h as usize];
        for y in 0..h as usize {
            for px in 0..w as usize {
                for c in 0..3 {
                    let v = (y * 10 + px) as f32 + c as f32 * 0.5;
                    let off = y * stride + px * 12 + c * 4;
                    data[off..off + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        let vf = VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride, data }],
        };
        let bytes = encode_frame(vf, w, h, PixelFormat::RgbF32Le, &[]).unwrap();
        let out = decode_frame(bytes, None).unwrap();
        assert_eq!(f32_at(&out, 4, 3, 2), 11.0 + 1.0);
    }

    #[test]
    fn encoder_rejects_integer_formats_and_bad_options() {
        let (w, h) = (2u32, 2u32);
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(w);
        params.height = Some(h);
        params.pixel_format = Some(PixelFormat::Rgba64Le);
        let mut enc = make_encoder(&params).unwrap();
        let vf = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w as usize) * 8,
                data: vec![0u8; (w * h) as usize * 8],
            }],
        };
        assert!(enc.send_frame(&Frame::Video(vf)).is_err());

        params.pixel_format = Some(PixelFormat::RgbaF32Le);
        params.options.insert("compression", "lzw");
        assert!(make_encoder(&params).is_err());
        params.options = Default::default();
        params.options.insert("pixel_type", "uint");
        assert!(make_encoder(&params).is_err());

        // Short plane data is rejected rather than panicking.
        params.options = Default::default();
        let mut enc = make_encoder(&params).unwrap();
        let vf = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w as usize) * 16,
                data: vec![0u8; 16],
            }],
        };
        assert!(enc.send_frame(&Frame::Video(vf)).is_err());
    }
}
