//! `oxideav-core` integration layer for `oxideav-openexr`.
//!
//! Gated behind the default-on `registry` feature so image-library
//! consumers can depend on `oxideav-openexr` with `default-features = false`
//! and skip the `oxideav-core` dependency entirely.
//!
//! Framework integration is preview-oriented because `oxideav-core` does
//! not yet have a float pixel format. The `Decoder` impl converts the
//! decoded HDR samples to packed **16-bit** RGBA (`Rgba64Le`) via a
//! clamp-to-[0, 1] tone-map; the `Encoder` impl reads packed 16-bit RGBA
//! and inflates it back to FLOAT samples in [0, 1]. 16-bit (vs the
//! original 8-bit) keeps far more of the EXR's tonal precision for the
//! "EXR loaded as a preview" use case the framework needs — a full HDR
//! pipeline lands when `oxideav-core` grows an `Rgba128Float` (or
//! similar) pixel-format variant. Each channel is a little-endian `u16`,
//! so a pixel is 8 bytes `RR GG BB AA`.

use oxideav_core::{
    CodecCapabilities, CodecId, CodecInfo, CodecParameters, CodecRegistry, ContainerRegistry,
    Decoder, Encoder, Frame, Packet, PixelFormat, RuntimeContext, TimeBase, VideoFrame, VideoPlane,
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
        .with_pixel_formats(vec![PixelFormat::Rgba64Le]);
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

/// Convert an [`crate::ExrImage`] into a packed 16-bit RGBA (`Rgba64Le`)
/// `VideoFrame` by clamping each sample to [0, 1] then scaling to
/// [0, 65535]. Each channel is a little-endian `u16` (8 bytes/pixel,
/// `RR GG BB AA`). Channels not in {R, G, B, A} are ignored; missing
/// channels default to 0 (R/G/B) or full-scale (A).
fn exr_image_to_rgba_video_frame(img: &crate::ExrImage) -> VideoFrame {
    let w = img.width() as usize;
    let h = img.height() as usize;
    let mut data = vec![0u8; w * h * 8];

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
            let base = off * 8;
            data[base..base + 2].copy_from_slice(&clamp_unit_to_u16(r).to_le_bytes());
            data[base + 2..base + 4].copy_from_slice(&clamp_unit_to_u16(g).to_le_bytes());
            data[base + 4..base + 6].copy_from_slice(&clamp_unit_to_u16(b).to_le_bytes());
            data[base + 6..base + 8].copy_from_slice(&clamp_unit_to_u16(a).to_le_bytes());
        }
    }
    VideoFrame {
        pts: None,
        planes: vec![VideoPlane {
            stride: w * 8,
            data,
        }],
    }
}

fn clamp_unit_to_u16(f: f32) -> u16 {
    let v = f.clamp(0.0, 1.0);
    (v * 65535.0 + 0.5) as u16
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
        if format != PixelFormat::Rgba64Le {
            return Err(oxideav_core::Error::invalid(format!(
                "OpenEXR encoder: unsupported pixel format {format:?} (Rgba64Le only)"
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
        if plane.stride < (width as usize) * 8 {
            return Err(oxideav_core::Error::invalid(format!(
                "OpenEXR encoder: Rgba64Le stride {} too small for width {width} (need {})",
                plane.stride,
                (width as usize) * 8
            )));
        }
        // Inflate each 16-bit (little-endian `u16`) RGBA sample to a [0, 1]
        // f32. Tone-mapping back from HDR is up to the caller; this just
        // preserves whatever the source had at 16-bit precision.
        let mut samples = Vec::with_capacity((width as usize) * (height as usize) * 4);
        for y in 0..height as usize {
            let row = &plane.data[y * plane.stride..y * plane.stride + (width as usize) * 8];
            for px in 0..width as usize {
                let base = px * 8;
                let ch = |i: usize| {
                    u16::from_le_bytes([row[base + i], row[base + i + 1]]) as f32 / 65535.0
                };
                samples.push(ch(0));
                samples.push(ch(2));
                samples.push(ch(4));
                samples.push(ch(6));
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
    }

    #[test]
    fn decoder_emits_16bit_rgba64le_frame() {
        // Encode a small in-[0,1] FLOAT RGBA EXR, decode it through the
        // framework shim, and confirm the frame is packed 16-bit RGBA at
        // 8 bytes/pixel with the expected quantised values.
        let (w, h) = (2u32, 2u32);
        // Pixel (0,0): R=0,G=0,B=0,A=1 ; (1,0): R=1 ; (0,1): G=0.5 ; (1,1): B=0.25
        let mut samples = vec![0.0f32; (w * h * 4) as usize];
        samples[3] = 1.0; // A of px0
        samples[4] = 1.0; // R of px1
        samples[7] = 1.0; // A of px1
        samples[8 + 1] = 0.5; // G of px2
        samples[8 + 3] = 1.0; // A of px2
        samples[12 + 2] = 0.25; // B of px3
        samples[12 + 3] = 1.0; // A of px3
        let bytes = encode_exr_scanline_rgba_float(w, h, &samples).unwrap();

        let mut dec = make_decoder(&CodecParameters::video(CodecId::new(CODEC_ID_STR))).unwrap();
        dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), bytes))
            .unwrap();
        let frame = dec.receive_frame().unwrap();
        let vf = match frame {
            Frame::Video(v) => v,
            _ => panic!("expected video frame"),
        };
        assert_eq!(vf.planes[0].stride, (w as usize) * 8);
        assert_eq!(vf.planes[0].data.len(), (w * h) as usize * 8);
        let rd = |px: usize, ch: usize| {
            let b = px * 8 + ch * 2;
            u16::from_le_bytes([vf.planes[0].data[b], vf.planes[0].data[b + 1]])
        };
        // px1 R = 1.0 -> 65535 ; px2 G = 0.5 -> 32768 ; px3 B = 0.25 -> 16384
        assert_eq!(rd(1, 0), 65535);
        assert_eq!(rd(2, 1), 32768);
        assert_eq!(rd(3, 2), 16384);
        // All A channels were 1.0.
        for px in 0..4 {
            assert_eq!(rd(px, 3), 65535, "alpha px{px}");
        }
    }

    #[test]
    fn encoder_accepts_rgba64le_and_roundtrips() {
        use crate::parse_exr;
        let (w, h) = (3u32, 2u32);
        // Build a 16-bit RGBA frame with a few known values.
        let mut data = vec![0u8; (w * h) as usize * 8];
        let put = |d: &mut [u8], px: usize, ch: usize, v: u16| {
            let b = px * 8 + ch * 2;
            d[b..b + 2].copy_from_slice(&v.to_le_bytes());
        };
        put(&mut data, 0, 0, 65535); // px0 R full
        put(&mut data, 1, 1, 32768); // px1 G half
        put(&mut data, 5, 2, 16384); // px5 B quarter
        for px in 0..6 {
            put(&mut data, px, 3, 65535); // A full
        }
        let vf = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w as usize) * 8,
                data,
            }],
        };

        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(w);
        params.height = Some(h);
        params.pixel_format = Some(PixelFormat::Rgba64Le);
        let mut enc = make_encoder(&params).unwrap();
        enc.send_frame(&Frame::Video(vf)).unwrap();
        let pkt = enc.receive_packet().unwrap();

        let img = parse_exr(&pkt.data).unwrap();
        assert_eq!(img.width(), w);
        assert_eq!(img.height(), h);
        let plane = |name: &str| img.planes.iter().find(|p| p.name == name).unwrap();
        // 65535/65535 == 1.0 ; 32768/65535 ≈ 0.50000763 ; 16384/65535 ≈ 0.25000381
        assert!((plane("R").samples[0] - 1.0).abs() < 1e-6);
        assert!((plane("G").samples[1] - (32768.0 / 65535.0)).abs() < 1e-6);
        assert!((plane("B").samples[5] - (16384.0 / 65535.0)).abs() < 1e-6);
    }

    #[test]
    fn encoder_rejects_8bit_rgba() {
        let (w, h) = (2u32, 2u32);
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(w);
        params.height = Some(h);
        params.pixel_format = Some(PixelFormat::Rgba);
        let mut enc = make_encoder(&params).unwrap();
        let vf = VideoFrame {
            pts: None,
            planes: vec![VideoPlane {
                stride: (w as usize) * 4,
                data: vec![0u8; (w * h) as usize * 4],
            }],
        };
        assert!(enc.send_frame(&Frame::Video(vf)).is_err());
    }
}
