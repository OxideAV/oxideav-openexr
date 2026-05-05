//! `oxideav-core` integration layer for `oxideav-openexr`.
//!
//! Gated behind the default-on `registry` feature so image-library
//! consumers can depend on `oxideav-openexr` with `default-features = false`
//! and skip the `oxideav-core` dependency entirely.
//!
//! Round-1 framework integration is minimal because `oxideav-core` does
//! not yet have a float pixel format. The `Decoder` impl converts the
//! decoded HDR samples to packed 8-bit RGBA via a clamp-to-[0, 1]
//! tone-map; the `Encoder` impl reads packed 8-bit RGBA and inflates
//! it back to FLOAT samples in [0, 1]. This mirrors the "EXR loaded
//! as a thumbnail" use case the framework needs for previews — a
//! proper HDR pipeline lands when `oxideav-core` grows an
//! `Rgba128Float` (or similar) pixel-format variant.

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, ContainerRegistry,
    Decoder, Encoder, Frame, Packet, PixelFormat, TimeBase, VideoFrame, VideoPlane,
};

use crate::decoder::parse_exr;
use crate::encoder::encode_exr_scanline_rgba_float;
use crate::error::ExrError;
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

/// Register the OpenEXR codec into the supplied [`CodecRegistry`].
pub fn register_codecs(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let caps = CodecCapabilities::video("openexr_sw")
        .with_intra_only(true)
        .with_lossless(true)
        .with_max_size(65535, 65535)
        .with_pixel_formats(vec![PixelFormat::Rgba]);
    reg.register(
        CodecInfo::new(cid)
            .capabilities(caps)
            .decoder(make_decoder)
            .encoder(make_encoder),
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

/// Combined registration: codecs + (no-op) containers.
pub fn register(codecs: &mut CodecRegistry, containers: &mut ContainerRegistry) {
    register_codecs(codecs);
    register_containers(containers);
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

fn make_decoder(_params: &CodecParameters) -> oxideav_core::Result<Box<dyn Decoder>> {
    Ok(Box::new(ExrDecoder {
        codec_id: CodecId::new(CODEC_ID_STR),
        pending: None,
        eof: false,
    }))
}

struct ExrDecoder {
    codec_id: CodecId,
    pending: Option<VideoFrame>,
    eof: bool,
}

impl Decoder for ExrDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }
    fn send_packet(&mut self, packet: &Packet) -> oxideav_core::Result<()> {
        let img = parse_exr(&packet.data)?;
        self.pending = Some(exr_image_to_rgba_video_frame(&img));
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

/// Convert an [`crate::ExrImage`] into a packed 8-bit RGBA `VideoFrame`
/// by clamping each sample to [0, 1] then scaling to [0, 255]. Channels
/// not in {R, G, B, A} are ignored; missing channels default to 0
/// (R/G/B) or 255 (A).
fn exr_image_to_rgba_video_frame(img: &crate::ExrImage) -> VideoFrame {
    let w = img.width() as usize;
    let h = img.height() as usize;
    let mut data = vec![0u8; w * h * 4];

    // Find the four canonical channels by name in the alphabetical plane list.
    let r_idx = img.planes.iter().position(|p| p.name == "R");
    let g_idx = img.planes.iter().position(|p| p.name == "G");
    let b_idx = img.planes.iter().position(|p| p.name == "B");
    let a_idx = img.planes.iter().position(|p| p.name == "A");

    for y in 0..h {
        for x in 0..w {
            let off = y * w + x;
            let r = r_idx.map(|i| img.planes[i].samples[off]).unwrap_or(0.0);
            let g = g_idx.map(|i| img.planes[i].samples[off]).unwrap_or(0.0);
            let b = b_idx.map(|i| img.planes[i].samples[off]).unwrap_or(0.0);
            let a = a_idx.map(|i| img.planes[i].samples[off]).unwrap_or(1.0);
            data[off * 4] = clamp_unit_to_u8(r);
            data[off * 4 + 1] = clamp_unit_to_u8(g);
            data[off * 4 + 2] = clamp_unit_to_u8(b);
            data[off * 4 + 3] = clamp_unit_to_u8(a);
        }
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 4,
            data,
        }],
    }
}

fn clamp_unit_to_u8(f: f32) -> u8 {
    let v = f.clamp(0.0, 1.0);
    (v * 255.0 + 0.5) as u8
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

fn make_encoder(params: &CodecParameters) -> oxideav_core::Result<Box<dyn Encoder>> {
    let mut out_params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    out_params.width = params.width;
    out_params.height = params.height;
    out_params.pixel_format = params.pixel_format;
    Ok(Box::new(ExrEncoder {
        codec_id: CodecId::new(CODEC_ID_STR),
        out_params,
        pending: None,
        eof: false,
    }))
}

struct ExrEncoder {
    codec_id: CodecId,
    out_params: CodecParameters,
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
        if format != PixelFormat::Rgba {
            return Err(oxideav_core::Error::invalid(format!(
                "OpenEXR encoder: unsupported pixel format {format:?} (Rgba only in round 1)"
            )));
        }
        let width = self.out_params.width.ok_or_else(|| {
            oxideav_core::Error::invalid("OpenEXR encoder: width missing in CodecParameters")
        })?;
        let height = self.out_params.height.ok_or_else(|| {
            oxideav_core::Error::invalid("OpenEXR encoder: height missing in CodecParameters")
        })?;
        if vf.planes.is_empty() {
            return Err(oxideav_core::Error::invalid(
                "OpenEXR encoder: empty frame plane",
            ));
        }
        let plane = &vf.planes[0];
        // Inflate each 8-bit RGBA sample to a [0, 1] f32. Tone-mapping
        // back from HDR is up to the caller; this just preserves
        // whatever the LDR source had.
        let mut samples = Vec::with_capacity((width as usize) * (height as usize) * 4);
        for y in 0..height as usize {
            let row = &plane.data[y * plane.stride..y * plane.stride + (width as usize) * 4];
            for px in 0..width as usize {
                samples.push(row[px * 4] as f32 / 255.0);
                samples.push(row[px * 4 + 1] as f32 / 255.0);
                samples.push(row[px * 4 + 2] as f32 / 255.0);
                samples.push(row[px * 4 + 3] as f32 / 255.0);
            }
        }
        let bytes = encode_exr_scanline_rgba_float(width, height, &samples)?;
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
