//! `oxideav-core` integration layer for `oxideav-openexr`.
//!
//! Gated behind the default-on `registry` feature so image-library
//! consumers can depend on `oxideav-openexr` with `default-features = false`
//! and skip the `oxideav-core` dependency entirely.
//!
//! # Pixel formats
//!
//! The framework shims speak the scene-referred 32-bit float family
//! (`oxideav-core` 0.1.35+): [`PixelFormat::RgbaF32Le`],
//! [`PixelFormat::RgbF32Le`] and [`PixelFormat::GrayF32Le`]. Samples are
//! IEEE 754 binary32 little-endian words carrying linear light exactly
//! as stored in the file — HALF channels are widened (exact), FLOAT
//! channels are copied bit-for-bit, UINT channels are converted to
//! `f32` (exact up to 2^24). There is **no tone-mapping and no clamp**:
//! values above 1.0 and below 0.0 survive the round trip.
//!
//! # Decoder channel mapping
//!
//! An OpenEXR channel list is an arbitrarily-named set (the staged
//! format description only fixes `R` / `G` / `B` / `A` as the
//! conventional colour names). The decoder maps the part's channels to
//! a frame with these rules, applied in order:
//!
//! 1. `R`, `G`, `B` and `A` all present → `RgbaF32Le` (packed R, G, B,
//!    A per pixel).
//! 2. `R`, `G`, `B` present, no `A` → `RgbF32Le`. A missing alpha is
//!    *not* synthesised; the frame format says so instead.
//! 3. `Y`, `RY` and `BY` present (no `R`/`G`/`B`) → luminance/chroma:
//!    RGB is reconstructed with the image's luminance weights (its
//!    `chromaticities` attribute, else BT.709) and the chroma planes are
//!    interpolated up from their declared sampling — see
//!    [`crate::luma_chroma`]. The frame is `RgbF32Le`, or `RgbaF32Le`
//!    when an `A` channel accompanies the triple.
//! 4. `Y` present (no `R`/`G`/`B`, no chroma) → `GrayF32Le` from the
//!    `Y` channel. If an `A` channel accompanies `Y` the frame is
//!    `RgbaF32Le` with `Y` replicated into R, G and B, so the alpha is
//!    not dropped.
//! 5. Anything else — a partial colour triple, a lone `RY` or `BY`,
//!    depth-only (`Z`) or AOV-only parts — is `Error::Unsupported`
//!    naming the channel list. Extra channels alongside a recognised
//!    set (`Z`, motion vectors, ids …) are ignored; they are still
//!    reachable through the standalone [`crate::parse_exr`] API.
//!
//! Every directly-mapped channel (`R G B A Y`) must be at 1×1
//! sampling; only the `RY` / `BY` chroma planes may be sub-sampled.
//!
//! # Parts
//!
//! Single-part flat files (scanline or tiled) decode directly. In a
//! multi-part file the `part` decoder option (default `0`) selects the
//! part to emit; flat scanline / tiled parts decode as above and a
//! multi-level (MIPMAP / RIPMAP) tiled part contributes its level
//! `(0, 0)` full-resolution image. Deep parts (single-part deep files,
//! or a deep part selected in a multi-part file) carry a variable
//! number of samples per pixel and have no `VideoFrame` mapping — the
//! decoder returns `Error::Unsupported`; use [`crate::parse_exr_deep_scanline`]
//! and friends from the standalone API instead.
//!
//! # Encoder
//!
//! The encoder accepts `RgbaF32Le` / `RgbF32Le` / `GrayF32Le` frames
//! and writes a single-part scanline file with channels `A B G R` /
//! `B G R` / `Y` respectively. The `pixel_type` option selects the
//! channel type — `float` (default, lossless) or `half` (binary16 with
//! round-to-nearest-even) — and `compression` picks any of the crate's
//! scanline codecs (`none`, `rle`, `zips`, `zip` (default), `piz`,
//! `pxr24`, `b44`, `b44a`, `dwaa`, `dwab`).
//!
//! `colour=luma_chroma` writes RGB(A) frames as `Y` + `RY` + `BY`
//! (+ `A`) instead, with the chroma planes reduced by `chroma_sampling`
//! (default `2`, i.e. 2×2 — the frame's width and height must be
//! multiples of it; `1` keeps full-resolution chroma and round-trips
//! exactly). Luminance uses the BT.709 weights and no `chromaticities`
//! attribute is written (the frame model carries no primaries), so
//! decoders reconstruct with the same weights. Gray frames write `Y`
//! under either layout.

use oxideav_core::{
    parse_options, CodecCapabilities, CodecId, CodecInfo, CodecOptionsStruct, CodecParameters,
    CodecRegistry, ContainerRegistry, Decoder, Encoder, Frame, OptionField, OptionKind,
    OptionValue, Packet, PixelFormat, RuntimeContext, TimeBase, VideoFrame, VideoPlane,
};

use crate::decoder::parse_exr;
use crate::encoder::encode_exr_scanline;
use crate::error::ExrError;
use crate::header::VersionField;
use crate::image::ExrPlane;
use crate::luma_chroma::{
    luma_chroma_to_rgb, luminance_weights, luminance_weights_of, rgb_to_luma_chroma, ChromaPlane,
    RgbPlanes, BT709_CHROMATICITIES,
};
use crate::multipart_mixed_encoder::{parse_exr_multipart_mixed, MultipartMixedImage};
use crate::types::{Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType};
use crate::CODEC_ID_STR;

/// Convert an [`ExrError`] into the framework-shared
/// `oxideav_core::Error` so trait impls can use `?` on errors returned
/// by the framework-free parse/encode functions.
impl From<ExrError> for oxideav_core::Error {
    fn from(e: ExrError) -> Self {
        match e {
            ExrError::InvalidData(s) => oxideav_core::Error::InvalidData(s),
            ExrError::Unsupported(s) => oxideav_core::Error::Unsupported(s),
        }
    }
}

/// Pixel formats the decoder emits and the encoder accepts, in
/// preference order.
const FRAME_FORMATS: [PixelFormat; 3] = [
    PixelFormat::RgbaF32Le,
    PixelFormat::RgbF32Le,
    PixelFormat::GrayF32Le,
];

/// Register the OpenEXR codec into the supplied [`CodecRegistry`].
pub fn register_codecs(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let caps = CodecCapabilities::video("openexr_sw")
        .with_intra_only(true)
        .with_lossless(true)
        .with_max_size(65535, 65535)
        .with_pixel_formats(FRAME_FORMATS.to_vec());
    reg.register(
        CodecInfo::new(cid)
            .capabilities(caps)
            .decoder(make_decoder)
            .decoder_options::<ExrDecoderOptions>()
            .encoder(make_encoder)
            .encoder_options::<ExrEncoderOptions>(),
    );
}

/// OpenEXR is its own container (single image per file). Demuxer/muxer
/// registration is a round-2 followup — for now we only register the
/// `.exr` extension so cli-convert + the central [`ContainerRegistry`]
/// resolver can route inputs/outputs to the OpenEXR codec by filename.
///
/// The container name matches [`CODEC_ID_STR`] (`"openexr"`) so the
/// extension lookup lines up with the codec id; this mirrors the
/// `oxideav-pict` pattern (single-image format where the container is
/// effectively the codec itself).
pub fn register_containers(reg: &mut ContainerRegistry) {
    reg.register_extension("exr", CODEC_ID_STR);
}

/// Unified entry point: install every codec and container provided by
/// `oxideav-openexr` into a [`RuntimeContext`].
///
/// Also wired into [`oxideav_meta::register_all`] via the
/// [`oxideav_core::register!`] macro below.
pub fn register(ctx: &mut RuntimeContext) {
    register_codecs(&mut ctx.codecs);
    register_containers(&mut ctx.containers);
}

oxideav_core::register!("openexr", register);

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Decoder tuning knobs (see the module docs, *Parts*).
#[derive(Debug, Clone, Default)]
pub struct ExrDecoderOptions {
    /// Zero-based part index to emit from a multi-part file. Ignored
    /// (must be 0) for single-part files.
    pub part: u32,
}

impl CodecOptionsStruct for ExrDecoderOptions {
    const SCHEMA: &'static [OptionField] = &[OptionField {
        name: "part",
        kind: OptionKind::U32,
        default: OptionValue::U32(0),
        help: "zero-based part index to decode from a multi-part file",
    }];
    fn apply(&mut self, key: &str, value: &OptionValue) -> oxideav_core::Result<()> {
        match key {
            "part" => self.part = value.as_u32()?,
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

/// Colour channel layout the encoder writes (see the module docs,
/// *Encoder*).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColourLayout {
    /// `R`, `G`, `B` (+ `A`) channels; gray frames write `Y`.
    Rgb,
    /// `Y`, `RY`, `BY` (+ `A`) luminance/chroma channels with the
    /// chroma sub-sampled by `chroma_sampling`; gray frames write `Y`.
    LumaChroma,
}

/// Encoder tuning knobs (see the module docs, *Encoder*).
#[derive(Debug, Clone)]
pub struct ExrEncoderOptions {
    /// Channel pixel type written to the file.
    pub pixel_type: PixelType,
    /// Scanline compression scheme.
    pub compression: Compression,
    /// Colour channel layout.
    pub colour: ColourLayout,
    /// `RY` / `BY` sampling factor (both axes) for
    /// [`ColourLayout::LumaChroma`]; `1` keeps the chroma at full
    /// resolution. Ignored for [`ColourLayout::Rgb`].
    pub chroma_sampling: u32,
}

impl Default for ExrEncoderOptions {
    fn default() -> Self {
        Self {
            pixel_type: PixelType::Float,
            compression: Compression::Zip,
            colour: ColourLayout::Rgb,
            chroma_sampling: 2,
        }
    }
}

const PIXEL_TYPE_NAMES: [&str; 2] = ["float", "half"];
const COLOUR_NAMES: [&str; 2] = ["rgb", "luma_chroma"];
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

impl CodecOptionsStruct for ExrEncoderOptions {
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
            _ => unreachable!("guarded by SCHEMA"),
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

fn make_decoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Decoder>> {
    let opts: ExrDecoderOptions = parse_options(&params.options)?;
    Ok(Box::new(ExrDecoder {
        codec_id: CodecId::new(CODEC_ID_STR),
        opts,
        pending: None,
        eof: false,
    }))
}

struct ExrDecoder {
    codec_id: CodecId,
    opts: ExrDecoderOptions,
    pending: Option<VideoFrame>,
    eof: bool,
}

impl Decoder for ExrDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }
    fn send_packet(&mut self, packet: &Packet) -> oxideav_core::Result<()> {
        let flat = decode_flat_part(&packet.data, self.opts.part)?;
        let (_format, frame) = flat_to_video_frame(&flat)?;
        self.pending = Some(frame);
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

/// One decoded flat (non-deep) image at full resolution, normalised
/// across the single-part / multi-part / multi-level readers.
struct FlatPixels {
    width: u32,
    height: u32,
    channels: Vec<Channel>,
    planes: Vec<ExrPlane>,
    /// The part's header attributes (chromaticities, multiView, …).
    attributes: Vec<Attribute>,
}

/// Decode part `part` of `bytes` as a flat image.
fn decode_flat_part(bytes: &[u8], part: u32) -> oxideav_core::Result<FlatPixels> {
    if bytes.len() < 8 {
        return Err(oxideav_core::Error::invalid(
            "OpenEXR: packet shorter than the magic + version field",
        ));
    }
    let version = VersionField::from_u32(u32::from_le_bytes(bytes[4..8].try_into().unwrap()));
    if !version.multipart {
        if part != 0 {
            return Err(oxideav_core::Error::invalid(format!(
                "OpenEXR decoder: part {part} requested from a single-part file"
            )));
        }
        if version.non_image {
            return Err(oxideav_core::Error::Unsupported(
                "OpenEXR decoder: deep image (variable samples per pixel) has no VideoFrame \
                 mapping; use the standalone parse_exr_deep_* API"
                    .to_string(),
            ));
        }
        let img = parse_exr(bytes)?;
        return Ok(FlatPixels {
            width: img.width(),
            height: img.height(),
            channels: img.channels,
            planes: img.planes,
            attributes: img.attributes,
        });
    }
    let mut parts = parse_exr_multipart_mixed(bytes)?;
    let count = parts.len();
    let idx = part as usize;
    if idx >= count {
        return Err(oxideav_core::Error::invalid(format!(
            "OpenEXR decoder: part {part} requested but the file has {count} part(s)"
        )));
    }
    match parts.swap_remove(idx) {
        MultipartMixedImage::Scanline(img) | MultipartMixedImage::Tiled(img) => Ok(FlatPixels {
            width: img.width(),
            height: img.height(),
            channels: img.channels,
            planes: img.planes,
            attributes: img.attributes,
        }),
        MultipartMixedImage::TiledMipmap(p) | MultipartMixedImage::TiledRipmap(p) => {
            let level = p
                .levels
                .into_iter()
                .find(|l| l.level_x == 0 && l.level_y == 0)
                .ok_or_else(|| {
                    oxideav_core::Error::invalid(format!(
                        "OpenEXR decoder: multi-level part {part} has no level (0, 0)"
                    ))
                })?;
            Ok(FlatPixels {
                width: level.width,
                height: level.height,
                channels: p.channels,
                planes: level.planes,
                attributes: p.attributes,
            })
        }
        MultipartMixedImage::DeepScanline(_)
        | MultipartMixedImage::DeepTiled(_)
        | MultipartMixedImage::DeepTiledMipmap(_)
        | MultipartMixedImage::DeepTiledRipmap(_) => {
            Err(oxideav_core::Error::Unsupported(format!(
                "OpenEXR decoder: part {part} is a deep part (variable samples per pixel) with no \
             VideoFrame mapping; use the standalone parse_exr_deep_* API"
            )))
        }
    }
}

/// Map a flat image's channel set to a frame (module docs, *Decoder
/// channel mapping*). Returns the chosen pixel format alongside the
/// packed frame.
fn flat_to_video_frame(img: &FlatPixels) -> oxideav_core::Result<(PixelFormat, VideoFrame)> {
    let find = |name: &str| img.planes.iter().position(|p| p.name == name);
    let (r, g, b, a, y) = (find("R"), find("G"), find("B"), find("A"), find("Y"));
    let (ry, by) = (find("RY"), find("BY"));
    let w = img.width;
    let h = img.height;
    let pixels = (w as usize) * (h as usize);

    // Every directly-mapped channel must be full-resolution and sized
    // for the data window.
    let full_res = |idx: usize| -> oxideav_core::Result<&[f32]> {
        let ch = &img.channels[idx];
        if ch.x_sampling != 1 || ch.y_sampling != 1 {
            return Err(oxideav_core::Error::Unsupported(format!(
                "OpenEXR decoder: colour channel '{}' is sub-sampled ({}x{}); frame formats are \
                 full-resolution",
                ch.name, ch.x_sampling, ch.y_sampling
            )));
        }
        let samples = &img.planes[idx].samples;
        if samples.len() != pixels {
            return Err(oxideav_core::Error::invalid(format!(
                "OpenEXR decoder: channel '{}' holds {} samples for {w}x{h}",
                ch.name,
                samples.len()
            )));
        }
        Ok(samples.as_slice())
    };
    // A chroma channel keeps whatever sampling the file declares; the
    // conversion reconstructs full resolution.
    let chroma = |idx: usize| -> oxideav_core::Result<ChromaPlane<'_>> {
        let ch = &img.channels[idx];
        if ch.x_sampling <= 0 || ch.y_sampling <= 0 {
            return Err(oxideav_core::Error::invalid(format!(
                "OpenEXR decoder: chroma channel '{}' declares sampling {}x{}",
                ch.name, ch.x_sampling, ch.y_sampling
            )));
        }
        Ok(ChromaPlane {
            samples: &img.planes[idx].samples,
            x_sampling: ch.x_sampling as u32,
            y_sampling: ch.y_sampling as u32,
        })
    };

    let converted: Option<RgbPlanes>;
    let (format, sources): (PixelFormat, Vec<&[f32]>) = match (r, g, b, a, y, ry, by) {
        (Some(r), Some(g), Some(b), Some(a), ..) => (
            PixelFormat::RgbaF32Le,
            vec![full_res(r)?, full_res(g)?, full_res(b)?, full_res(a)?],
        ),
        (Some(r), Some(g), Some(b), None, ..) => (
            PixelFormat::RgbF32Le,
            vec![full_res(r)?, full_res(g)?, full_res(b)?],
        ),
        (None, None, None, a, Some(y), Some(ry), Some(by)) => {
            // Luminance/chroma: reconstruct RGB with the image's own
            // luminance weights (chromaticities attribute, else BT.709).
            let weights = luminance_weights_of(&img.attributes);
            let rgb = luma_chroma_to_rgb(w, h, full_res(y)?, chroma(ry)?, chroma(by)?, weights)?;
            converted = Some(rgb);
            let rgb = converted.as_ref().unwrap();
            match a {
                Some(a) => (
                    PixelFormat::RgbaF32Le,
                    vec![&rgb.r, &rgb.g, &rgb.b, full_res(a)?],
                ),
                None => (PixelFormat::RgbF32Le, vec![&rgb.r, &rgb.g, &rgb.b]),
            }
        }
        (None, None, None, Some(a), Some(y), None, None) => {
            let y = full_res(y)?;
            (PixelFormat::RgbaF32Le, vec![y, y, y, full_res(a)?])
        }
        (None, None, None, None, Some(y), None, None) => {
            (PixelFormat::GrayF32Le, vec![full_res(y)?])
        }
        _ => {
            let names: Vec<&str> = img.channels.iter().map(|c| c.name.as_str()).collect();
            return Err(oxideav_core::Error::Unsupported(format!(
                "OpenEXR decoder: channel set [{}] has no RGB(A) / Y / Y RY BY frame mapping",
                names.join(", ")
            )));
        }
    };

    let stride = format.plane_row_bytes(0, w).ok_or_else(|| {
        oxideav_core::Error::invalid(format!("OpenEXR decoder: {w}x{h} frame size overflows"))
    })?;
    let mut data = Vec::with_capacity(stride * h as usize);
    for px in 0..pixels {
        for src in &sources {
            data.extend_from_slice(&src[px].to_le_bytes());
        }
    }
    Ok((
        format,
        VideoFrame {
            pts: None,
            planes: vec![VideoPlane { stride, data }],
        },
    ))
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

fn make_encoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Encoder>> {
    let opts: ExrEncoderOptions = parse_options(&params.options)?;
    let mut out_params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    out_params.width = params.width;
    out_params.height = params.height;
    out_params.pixel_format = params.pixel_format;
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
    opts: ExrEncoderOptions,
    pending: Option<Vec<u8>>,
    eof: bool,
}

/// Packed component count per accepted frame format (`None` for
/// formats the encoder does not take).
trait InterleavedComponents {
    fn plane_count_interleaved(self) -> usize;
}
impl InterleavedComponents for PixelFormat {
    fn plane_count_interleaved(self) -> usize {
        match self {
            PixelFormat::RgbaF32Le => 4,
            PixelFormat::RgbF32Le => 3,
            PixelFormat::GrayF32Le => 1,
            _ => 0,
        }
    }
}

fn accepted_format(format: PixelFormat) -> bool {
    FRAME_FORMATS.contains(&format)
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
        if !accepted_format(format) {
            return Err(oxideav_core::Error::invalid(format!(
                "OpenEXR encoder: unsupported pixel format {format:?} (RgbaF32Le / RgbF32Le / \
                 GrayF32Le only)"
            )));
        }
        let width = self.out_params.width.ok_or_else(|| {
            oxideav_core::Error::invalid("OpenEXR encoder: width missing in CodecParameters")
        })?;
        let height = self.out_params.height.ok_or_else(|| {
            oxideav_core::Error::invalid("OpenEXR encoder: height missing in CodecParameters")
        })?;
        if width == 0 || height == 0 {
            return Err(oxideav_core::Error::invalid(format!(
                "OpenEXR encoder: {width}x{height} frame (both dimensions must be > 0)"
            )));
        }
        let plane = vf
            .planes
            .first()
            .ok_or_else(|| oxideav_core::Error::invalid("OpenEXR encoder: empty frame plane"))?;
        let row_bytes = format.plane_row_bytes(0, width).ok_or_else(|| {
            oxideav_core::Error::invalid(format!("OpenEXR encoder: {width}x{height} overflows"))
        })?;
        if plane.stride < row_bytes {
            return Err(oxideav_core::Error::invalid(format!(
                "OpenEXR encoder: {format:?} stride {} too small for width {width} (need \
                 {row_bytes})",
                plane.stride
            )));
        }
        let last_row_end = (height as usize - 1)
            .checked_mul(plane.stride)
            .and_then(|o| o.checked_add(row_bytes))
            .filter(|&end| end <= plane.data.len())
            .ok_or_else(|| {
                oxideav_core::Error::invalid(format!(
                    "OpenEXR encoder: plane data {} bytes too short for {height} rows of stride {}",
                    plane.data.len(),
                    plane.stride
                ))
            })?;
        debug_assert!(last_row_end <= plane.data.len());

        // De-interleave the packed frame into one f32 plane per
        // component (R, G, B, A / R, G, B / Y).
        let components = format.plane_count_interleaved();
        let pixels = (width as usize) * (height as usize);
        let mut comps: Vec<Vec<f32>> = (0..components)
            .map(|_| Vec::with_capacity(pixels))
            .collect();
        for y in 0..height as usize {
            let row = &plane.data[y * plane.stride..y * plane.stride + row_bytes];
            for px in 0..width as usize {
                let base = px * components * 4;
                for (c, dst) in comps.iter_mut().enumerate() {
                    let off = base + c * 4;
                    dst.push(f32::from_le_bytes(row[off..off + 4].try_into().unwrap()));
                }
            }
        }

        let mk = |name: &str, sampling: u32| Channel {
            name: name.to_string(),
            pixel_type: self.opts.pixel_type,
            p_linear: false,
            x_sampling: sampling as i32,
            y_sampling: sampling as i32,
        };
        // (channel, plane) pairs in the alphabetical order the file
        // layout requires.
        let (channels, planes): (Vec<Channel>, Vec<Vec<f32>>) = match (self.opts.colour, format) {
            (ColourLayout::LumaChroma, PixelFormat::RgbaF32Le | PixelFormat::RgbF32Le) => {
                let s = self.opts.chroma_sampling;
                if width % s != 0 || height % s != 0 {
                    return Err(oxideav_core::Error::invalid(format!(
                        "OpenEXR encoder: {width}x{height} frame is not a multiple of \
                             chroma_sampling={s} (conforming readers require sub-sampled \
                             extents divisible by the sampling factor; use chroma_sampling=1)"
                    )));
                }
                let weights = luminance_weights(&BT709_CHROMATICITIES);
                let yc = rgb_to_luma_chroma(
                    width,
                    height,
                    [&comps[0], &comps[1], &comps[2]],
                    weights,
                    (s, s),
                )?;
                let mut chs = Vec::with_capacity(4);
                let mut pls = Vec::with_capacity(4);
                if let Some(a) = comps.get(3) {
                    chs.push(mk("A", 1));
                    pls.push(a.clone());
                }
                chs.extend([mk("BY", s), mk("RY", s), mk("Y", 1)]);
                pls.extend([yc.by, yc.ry, yc.y]);
                (chs, pls)
            }
            (_, PixelFormat::RgbaF32Le) => {
                let mut it = comps.into_iter();
                let (r, g, b, a) = (
                    it.next().unwrap(),
                    it.next().unwrap(),
                    it.next().unwrap(),
                    it.next().unwrap(),
                );
                (
                    vec![mk("A", 1), mk("B", 1), mk("G", 1), mk("R", 1)],
                    vec![a, b, g, r],
                )
            }
            (_, PixelFormat::RgbF32Le) => {
                let mut it = comps.into_iter();
                let (r, g, b) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
                (vec![mk("B", 1), mk("G", 1), mk("R", 1)], vec![b, g, r])
            }
            (_, PixelFormat::GrayF32Le) => (vec![mk("Y", 1)], comps),
            _ => unreachable!("guarded by channel_layout"),
        };

        let attributes = scanline_attributes(width, height, &channels, self.opts.compression);
        let plane_refs: Vec<&[f32]> = planes.iter().map(|p| p.as_slice()).collect();
        let bytes = encode_exr_scanline(
            width,
            height,
            &channels,
            &plane_refs,
            self.opts.compression,
            attributes,
        )?;
        self.pending = Some(bytes);
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

/// The required header attribute set for a single-part scanline image
/// covering `[0, width) × [0, height)`.
fn scanline_attributes(
    width: u32,
    height: u32,
    channels: &[Channel],
    compression: Compression,
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
            name: "compression".to_string(),
            value: AttributeValue::Compression(compression),
        },
        Attribute {
            name: "dataWindow".to_string(),
            value: AttributeValue::Box2i(win),
        },
        Attribute {
            name: "displayWindow".to_string(),
            value: AttributeValue::Box2i(win),
        },
        Attribute {
            name: "lineOrder".to_string(),
            value: AttributeValue::LineOrder(LineOrder::IncreasingY),
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
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::encode_exr_scanline_rgba_float;
    use crate::half::{f32_to_half, half_to_f32};
    use crate::parse_exr;

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
        assert_eq!(caps.accepted_pixel_formats, FRAME_FORMATS.to_vec());
        assert!(caps.accepted_pixel_formats.iter().all(|f| f.is_float()));
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
            decode_frame(bytes, Some(2)).unwrap_err(),
            oxideav_core::Error::InvalidData(_)
        ));
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
