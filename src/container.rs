//! OpenEXR container: demuxer + muxer for the `oxideav-core` container
//! registry (the `openexr` container, `.exr`).
//!
//! An OpenEXR file is its own container, so the framework packet is a
//! complete single-part file and the registry [`crate::make_decoder`]
//! decodes it with [`crate::decode_with`] (one implementation). The
//! demuxer and muxer therefore never touch pixel data:
//!
//! * **Single-part file** → one packet holding the whole file.
//! * **Multi-part file** → one packet per part that has a contract view
//!   (`R G B (A)`, `Y (A)`, `Y RY BY (A)`), in file order, each packet a
//!   valid single-part file *repacked byte for byte* from the part's
//!   header and chunks: the header is re-emitted with the multi-part bit
//!   cleared (and the tiled bit set for a `tiledimage` part), the offset
//!   table recomputed, and the 4-byte part-number prefix stripped from
//!   every chunk. Compressed chunk payloads are copied unchanged, so the
//!   decoded planes are the ones [`crate::decode_all`] produces. Deep
//!   parts and parts whose channel set has no view are skipped exactly
//!   like `decode_all` (the lenient default); a file with no viewable
//!   part at all is `Unsupported`.
//! * **Streams**: one video stream per distinct (width, height, native
//!   layout, colour signal) among the emitted parts, in first-appearance
//!   order; parts that share all four (a stereo `left` / `right` pair) share
//!   a stream, parts that differ (a `RgbaF32Le` beauty next to a
//!   `GrayF32Le` depth) each get their own, so every stream's `params`
//!   describe every packet on it. The layout is `RgbaF32Le` / `RgbF32Le`
//!   / `GrayF32Le` as [`crate::info`] reports for that part; the colour
//!   signal is the part's (OpenEXR defines its colour semantics: linear
//!   light, `chromaticities` or the BT.709 default — see
//!   [`crate::ColorInfo`]). A single-part file has one stream.
//! * **Timing**: OpenEXR parts are not timed. Packets carry `pts` = the
//!   zero-based part index in the file (gaps where parts were skipped,
//!   matching `Frame::index`) in a `1/1` time base and no `duration`.
//! * **Muxer**: accepts one or more video streams. One packet in total →
//!   the file is written verbatim; several packets (on any streams) → a
//!   multi-part file combining the single-part packets in arrival order
//!   with the inverse repack (`name` / `type` / `chunkCount` attributes
//!   added, part-number prefixes inserted, one offset table per part).
//!   Only flat (scanline / tiled) single-part packets combine — what the
//!   registry encoder emits.
//!
//! Gated behind the `registry` feature: every type here comes from
//! `oxideav-core`.

use std::collections::VecDeque;
use std::io::{Read, SeekFrom, Write};

use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, ContainerRegistry, Demuxer, Error, MAX_PROBE_SCORE,
    MediaType, Muxer, PROBE_SCORE_EXTENSION, Packet, ProbeData, ProbeScore, ReadSeek, Result,
    StreamInfo, TimeBase, WriteSeek,
};

use crate::CODEC_ID_STR;
use crate::decoder::extract_required;
use crate::error::ExrError;
use crate::image::{ColorInfo, string_attribute};
use crate::layout::{CombineOptions, FileLayout};
use crate::registry::{to_color_signal, to_core_pixel_format};
use crate::types::Attribute;
use crate::view::plan_view;

/// Registered container name (the same string as the codec id).
pub const CONTAINER_NAME: &str = CODEC_ID_STR;

/// Register the OpenEXR container: demuxer, muxer, `.exr` extension and
/// the magic probe.
pub fn register(reg: &mut ContainerRegistry) {
    reg.register_demuxer(CONTAINER_NAME, open_demuxer);
    reg.register_muxer(CONTAINER_NAME, open_muxer);
    reg.register_extension("exr", CONTAINER_NAME);
    reg.register_probe(CONTAINER_NAME, probe);
}

/// `100` on the OpenEXR magic, `25` on a bare `.exr` extension hint,
/// `0` otherwise.
pub fn probe(data: &ProbeData) -> ProbeScore {
    if crate::probe(data.buf) {
        return MAX_PROBE_SCORE;
    }
    if matches!(data.ext, Some("exr")) {
        PROBE_SCORE_EXTENSION
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Demuxer
// ---------------------------------------------------------------------------

/// One emitted part: its file index, its view's geometry / layout /
/// colour, and the single-part file bytes.
struct PartPacket {
    index: u32,
    name: Option<String>,
    width: u32,
    height: u32,
    format: crate::PixelFormat,
    color: ColorInfo,
    bytes: Vec<u8>,
}

/// What the demuxer learned from the file.
struct Demuxed {
    /// Attributes of the first emitted part (metadata source).
    primary_attributes: Vec<Attribute>,
    part_count: u32,
    parts: Vec<PartPacket>,
}

/// Split a file into its viewable parts (see the module docs).
fn demux_file(bytes: &[u8]) -> crate::Result<Demuxed> {
    if !crate::probe(bytes) {
        return Err(ExrError::invalid("OpenEXR container: bad magic"));
    }
    // The layout rules of the multi-part note apply to every file the
    // demuxer hands on: version flags, `name` / `type`, shared
    // attributes, offset tables, part numbers.
    let layout = FileLayout::read(bytes)?;
    if !layout.is_multipart() {
        return demux_single_part(bytes, &layout);
    }
    demux_multipart(bytes, &layout)
}

fn demux_single_part(bytes: &[u8], layout: &FileLayout) -> crate::Result<Demuxed> {
    let part = &layout.parts[0];
    if part.kind.is_deep() {
        return Err(ExrError::unsupported(
            "OpenEXR container: deep image (variable samples per pixel) has no colour view; \
             use the parse_exr_deep_* API",
        ));
    }
    let req = extract_required(&part.attributes)?;
    let plan = plan_view(&req.channels, &part.attributes, "")?;
    let (width, height) = (req.data_window.width(), req.data_window.height());
    Ok(Demuxed {
        primary_attributes: part.attributes.clone(),
        part_count: 1,
        parts: vec![PartPacket {
            index: 0,
            name: string_attribute(&part.attributes, "name"),
            width,
            height,
            format: plan.format(),
            color: ColorInfo::from_attributes(&part.attributes),
            bytes: bytes.to_vec(),
        }],
    })
}

fn demux_multipart(bytes: &[u8], layout: &FileLayout) -> crate::Result<Demuxed> {
    // Every part as a single-part file: header unchanged, offset table
    // rewritten, chunks located through the tables and copied without
    // their part-number prefix.
    let singles = crate::layout::split_parts(bytes)?;
    let mut parts: Vec<PartPacket> = Vec::new();
    let mut primary_attributes: Option<Vec<Attribute>> = None;
    let mut skipped: Vec<String> = Vec::new();
    for (part, single) in layout.parts.iter().zip(singles) {
        let i = part.index;
        if part.kind.is_deep() {
            skipped.push(format!("part {i}: deep part has no colour view"));
            continue;
        }
        let req = extract_required(&part.attributes)?;
        let plan = match plan_view(&req.channels, &part.attributes, "") {
            Ok(p) => p,
            Err(ExrError::Unsupported(msg)) => {
                skipped.push(format!("part {i}: {msg}"));
                continue;
            }
            Err(e) => return Err(e),
        };
        if primary_attributes.is_none() {
            primary_attributes = Some(part.attributes.clone());
        }
        parts.push(PartPacket {
            index: i,
            name: part.name.clone(),
            width: req.data_window.width(),
            height: req.data_window.height(),
            format: plan.format(),
            color: ColorInfo::from_attributes(&part.attributes),
            bytes: single,
        });
    }
    let Some(primary_attributes) = primary_attributes else {
        return Err(ExrError::unsupported(format!(
            "OpenEXR container: no part has a colour view ({})",
            skipped.join("; ")
        )));
    };
    Ok(Demuxed {
        primary_attributes,
        part_count: layout.parts.len() as u32,
        parts,
    })
}

/// Open an OpenEXR file as a demuxer (see the module docs).
pub fn open_demuxer(
    mut input: Box<dyn ReadSeek>,
    _codecs: &dyn CodecResolver,
) -> Result<Box<dyn Demuxer>> {
    input.seek(SeekFrom::Start(0))?;
    let mut buf = Vec::new();
    input.read_to_end(&mut buf)?;
    drop(input);
    let demuxed = demux_file(&buf)?;
    drop(buf);

    let time_base = TimeBase::new(1, 1);
    // One stream per distinct (geometry, layout, colour) among the parts,
    // first-appearance order; `stream_of[k]` is part k's stream index.
    let mut streams: Vec<StreamInfo> = Vec::new();
    let mut stream_of = Vec::with_capacity(demuxed.parts.len());
    for p in &demuxed.parts {
        let pixel_format = to_core_pixel_format(p.format);
        let color_signal = to_color_signal(&p.color);
        let found = streams.iter().position(|s| {
            s.params.width == Some(p.width)
                && s.params.height == Some(p.height)
                && s.params.pixel_format == Some(pixel_format)
                && s.params.color_signal == color_signal
        });
        let idx = match found {
            Some(i) => i,
            None => {
                let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
                params.width = Some(p.width);
                params.height = Some(p.height);
                params.pixel_format = Some(pixel_format);
                params.color_signal = color_signal;
                streams.push(StreamInfo {
                    index: streams.len() as u32,
                    time_base,
                    duration: None,
                    start_time: Some(p.index as i64),
                    params,
                });
                streams.len() - 1
            }
        };
        stream_of.push(idx as u32);
    }

    let mut metadata: Vec<(String, String)> = Vec::new();
    for (attr, key) in [
        ("comments", "comment"),
        ("owner", "owner"),
        ("capDate", "date"),
    ] {
        if let Some(v) = string_attribute(&demuxed.primary_attributes, attr) {
            metadata.push((key.to_string(), v));
        }
    }
    if demuxed.part_count > 1 {
        metadata.push(("parts".to_string(), demuxed.part_count.to_string()));
        for p in &demuxed.parts {
            if let Some(name) = &p.name {
                metadata.push((format!("part_name:{}", p.index), name.clone()));
            }
        }
    }

    let packets = demuxed
        .parts
        .into_iter()
        .zip(stream_of)
        .map(|(p, stream_index)| {
            let mut pkt = Packet::new(stream_index, time_base, p.bytes);
            pkt.pts = Some(p.index as i64);
            pkt.dts = Some(p.index as i64);
            pkt.flags.keyframe = true;
            pkt
        })
        .collect();

    Ok(Box::new(ExrDemuxer {
        streams,
        packets,
        metadata,
    }))
}

struct ExrDemuxer {
    streams: Vec<StreamInfo>,
    packets: VecDeque<Packet>,
    metadata: Vec<(String, String)>,
}

impl Demuxer for ExrDemuxer {
    fn format_name(&self) -> &str {
        CONTAINER_NAME
    }
    fn streams(&self) -> &[StreamInfo] {
        &self.streams
    }
    fn next_packet(&mut self) -> Result<Packet> {
        self.packets.pop_front().ok_or(Error::Eof)
    }
    fn metadata(&self) -> &[(String, String)] {
        &self.metadata
    }
}

// ---------------------------------------------------------------------------
// Muxer
// ---------------------------------------------------------------------------

/// Open an OpenEXR muxer over `output` for one or more video streams
/// (every packet, whatever its stream, becomes one part; several packets
/// in total make a multi-part file).
pub fn open_muxer(output: Box<dyn WriteSeek>, streams: &[StreamInfo]) -> Result<Box<dyn Muxer>> {
    if streams.is_empty() {
        return Err(Error::invalid(
            "OpenEXR muxer: expected at least one video stream",
        ));
    }
    if let Some(s) = streams
        .iter()
        .find(|s| s.params.media_type != MediaType::Video)
    {
        return Err(Error::invalid(format!(
            "OpenEXR muxer: stream {} must be video",
            s.index
        )));
    }
    Ok(Box::new(ExrMuxer {
        output,
        stream_count: streams.len() as u32,
        packets: Vec::new(),
    }))
}

struct ExrMuxer {
    output: Box<dyn WriteSeek>,
    stream_count: u32,
    packets: Vec<Vec<u8>>,
}

impl Muxer for ExrMuxer {
    fn format_name(&self) -> &str {
        CONTAINER_NAME
    }
    fn write_header(&mut self) -> Result<()> {
        Ok(())
    }
    fn write_packet(&mut self, packet: &Packet) -> Result<()> {
        if packet.stream_index >= self.stream_count {
            return Err(Error::invalid(format!(
                "OpenEXR muxer: packet on stream {} but the muxer has {} stream(s)",
                packet.stream_index, self.stream_count
            )));
        }
        if packet.data.is_empty() {
            return Err(Error::invalid("OpenEXR muxer: empty packet"));
        }
        if !crate::probe(&packet.data) {
            return Err(Error::invalid(
                "OpenEXR muxer: packet is not an OpenEXR file (bad magic)",
            ));
        }
        self.packets.push(packet.data.clone());
        Ok(())
    }
    fn write_trailer(&mut self) -> Result<()> {
        let packets = std::mem::take(&mut self.packets);
        match packets.len() {
            0 => Err(Error::invalid(
                "OpenEXR muxer: no packet written (an OpenEXR file holds at least one part)",
            )),
            1 => {
                self.output.write_all(&packets[0])?;
                Ok(())
            }
            _ => {
                let combined = combine_parts(&packets)?;
                self.output.write_all(&combined)?;
                Ok(())
            }
        }
    }
}

/// Combine single-part flat files into one multi-part file, byte for
/// byte (the inverse of the demuxer's repack): `name` / `type` /
/// `chunkCount` added, duplicate names suffixed, and every part given
/// the union of the packets' display windows (the attribute is shared
/// by all parts of a file — for origin-anchored encoder output the
/// union is the largest extent, the rule `encode_all` applies).
/// `pixelAspectRatio` and `chromaticities` are shared too and are not
/// reconciled: packets that disagree on them are refused.
pub(crate) fn combine_parts(files: &[Vec<u8>]) -> crate::Result<Vec<u8>> {
    if files.len() < 2 {
        return Err(ExrError::invalid(
            "OpenEXR muxer: combining needs at least two packets",
        ));
    }
    for (i, f) in files.iter().enumerate() {
        if FileLayout::read(f)?.parts.iter().any(|p| p.kind.is_deep()) {
            return Err(ExrError::unsupported(format!(
                "OpenEXR muxer: multi-part output combines flat single-part packets only; \
                 packet {i} is a deep image"
            )));
        }
    }
    crate::layout::combine_parts_with(
        files,
        &CombineOptions::new()
            .with_unify_display_window(true)
            .with_rename_duplicates(true),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EncodeOptions, ExrImage, ExrPixelFormat};

    fn image(w: u32, h: u32, format: ExrPixelFormat, salt: f32) -> ExrImage {
        let comps = match format {
            ExrPixelFormat::GrayF32Le => 1,
            ExrPixelFormat::RgbF32Le => 3,
            ExrPixelFormat::RgbaF32Le => 4,
        };
        let samples: Vec<f32> = (0..(w * h) as usize * comps)
            .map(|i| i as f32 * 0.125 + salt)
            .collect();
        ExrImage::from_f32(w, h, format, &samples).unwrap()
    }

    #[test]
    fn probe_scores() {
        let bytes = crate::encode(
            &image(2, 2, ExrPixelFormat::RgbF32Le, 0.0),
            &EncodeOptions::default(),
        )
        .unwrap();
        assert_eq!(
            probe(&ProbeData {
                buf: &bytes,
                ext: None
            }),
            MAX_PROBE_SCORE
        );
        assert_eq!(
            probe(&ProbeData {
                buf: b"farbfeld",
                ext: Some("exr")
            }),
            PROBE_SCORE_EXTENSION
        );
        assert_eq!(
            probe(&ProbeData {
                buf: b"farbfeld",
                ext: Some("ff")
            }),
            0
        );
    }

    #[test]
    fn combine_then_demux_is_the_identity_on_chunks() {
        let a = crate::encode(
            &image(5, 3, ExrPixelFormat::RgbF32Le, 0.0),
            &EncodeOptions::default().with_compression(crate::Compression::Zip),
        )
        .unwrap();
        let b = crate::encode(
            &image(4, 6, ExrPixelFormat::GrayF32Le, 1.0),
            &EncodeOptions::default().with_tile_size(2),
        )
        .unwrap();
        let multi = combine_parts(&[a.clone(), b.clone()]).unwrap();
        let demuxed = demux_file(&multi).unwrap();
        assert_eq!(demuxed.part_count, 2);
        assert_eq!(demuxed.parts.len(), 2);
        assert_eq!(
            (demuxed.parts[0].format, demuxed.parts[1].format),
            (ExrPixelFormat::RgbF32Le, ExrPixelFormat::GrayF32Le)
        );
        assert_eq!((demuxed.parts[1].width, demuxed.parts[1].height), (4, 6));
        assert_eq!(demuxed.parts[0].name.as_deref(), Some("part0"));
        assert_eq!(demuxed.parts[1].name.as_deref(), Some("part1"));
        for (orig, part) in [a, b].iter().zip(&demuxed.parts) {
            // The repacked part decodes to the original planes.
            assert_eq!(
                crate::decode(&part.bytes).unwrap().planes,
                crate::decode(orig).unwrap().planes
            );
        }
    }

    #[test]
    fn hostile_inputs_do_not_panic() {
        let good = crate::encode(
            &image(3, 3, ExrPixelFormat::RgbaF32Le, 0.0),
            &EncodeOptions::default(),
        )
        .unwrap();
        for cut in 0..good.len() {
            let _ = demux_file(&good[..cut]);
        }
        let multi = combine_parts(&[good.clone(), good.clone()]).unwrap();
        for cut in 0..multi.len() {
            let _ = demux_file(&multi[..cut]);
        }
        let mut flipped = multi.clone();
        for i in 8..flipped.len().min(400) {
            flipped[i] ^= 0x5a;
            let _ = demux_file(&flipped);
            flipped[i] ^= 0x5a;
        }
        assert!(combine_parts(&[good.clone(), vec![0x76, 0x2f, 0x31, 0x01]]).is_err());
        assert!(combine_parts(&[good.clone(), multi]).is_err());
    }
}
