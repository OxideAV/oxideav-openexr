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

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{Read, SeekFrom, Write};

use oxideav_core::{
    CodecId, CodecParameters, CodecResolver, ContainerRegistry, Demuxer, Error, MediaType, Muxer,
    Packet, ProbeData, ProbeScore, ReadSeek, Result, StreamInfo, TimeBase, WriteSeek,
    MAX_PROBE_SCORE, PROBE_SCORE_EXTENSION,
};

use crate::decoder::{compute_total_tiles, extract_required, find_chunk_count, find_part_type};
use crate::deep::parse_header_allow_deep;
use crate::error::ExrError;
use crate::header::{
    encode_attribute_value, encode_header, parse_header, parse_multipart_headers, ParsedHeader,
    VersionField,
};
use crate::image::{string_attribute, ColorInfo};
use crate::registry::{to_color_signal, to_core_pixel_format};
use crate::tiled::tiledesc_from_attribute;
use crate::types::{Attribute, AttributeValue, Box2i, EXR_MAGIC};
use crate::view::plan_view;
use crate::CODEC_ID_STR;

/// Registered container name (the same string as the codec id).
pub const CONTAINER_NAME: &str = CODEC_ID_STR;

/// Version-field bit: single-part tiled file.
const VERSION_SINGLE_TILE: u32 = 0x200;
/// Version-field bit: attribute names / type names up to 255 bytes.
const VERSION_LONG_NAMES: u32 = 0x400;
/// Version-field bit: multi-part file.
const VERSION_MULTIPART: u32 = 0x1000;
/// Longest attribute name / type name without the long-names bit.
const SHORT_NAME_MAX: usize = 31;

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
// Byte readers
// ---------------------------------------------------------------------------

fn need(bytes: &[u8], pos: usize, len: usize, what: &str) -> crate::Result<usize> {
    let end = pos
        .checked_add(len)
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| {
            ExrError::invalid(format!(
                "OpenEXR container: {what} at offset {pos} runs past EOF (file size {})",
                bytes.len()
            ))
        })?;
    Ok(end)
}

fn read_i32(bytes: &[u8], pos: usize, what: &str) -> crate::Result<i32> {
    let end = need(bytes, pos, 4, what)?;
    Ok(i32::from_le_bytes(bytes[pos..end].try_into().unwrap()))
}

fn read_u64(bytes: &[u8], pos: usize, what: &str) -> crate::Result<u64> {
    let end = need(bytes, pos, 8, what)?;
    Ok(u64::from_le_bytes(bytes[pos..end].try_into().unwrap()))
}

/// A non-negative `i32` size field as `usize`.
fn read_size(bytes: &[u8], pos: usize, what: &str) -> crate::Result<usize> {
    let v = read_i32(bytes, pos, what)?;
    usize::try_from(v)
        .map_err(|_| ExrError::invalid(format!("OpenEXR container: {what} is negative ({v})")))
}

// ---------------------------------------------------------------------------
// Part shapes and chunk walking
// ---------------------------------------------------------------------------

/// The chunk record shape of one part.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PartShape {
    /// `i32 Y | i32 size | payload`.
    Scanline,
    /// `i32 tx | i32 ty | i32 lx | i32 ly | i32 size | payload`.
    Tiled,
    /// `i32 Y | u64 packed_table | u64 packed_data | u64 unpacked | table | data`.
    DeepScanline,
    /// `4 × i32 coords | 3 × u64 sizes | table | data`.
    DeepTile,
}

impl PartShape {
    fn from_type(part_type: &str, idx: usize) -> crate::Result<Self> {
        Ok(match part_type {
            "scanlineimage" => Self::Scanline,
            "tiledimage" => Self::Tiled,
            "deepscanline" => Self::DeepScanline,
            "deeptile" => Self::DeepTile,
            other => {
                return Err(ExrError::invalid(format!(
                    "OpenEXR container: part {idx} has unknown type '{other}'"
                )))
            }
        })
    }

    fn is_deep(self) -> bool {
        matches!(self, Self::DeepScanline | Self::DeepTile)
    }

    /// Fixed header bytes of one chunk record (part-number prefix excluded).
    fn header_len(self) -> usize {
        match self {
            Self::Scanline => 8,
            Self::Tiled => 20,
            Self::DeepScanline => 4 + 24,
            Self::DeepTile => 16 + 24,
        }
    }

    /// Length of the chunk record starting at `pos` (its header included,
    /// `prefix` bytes of leading part number included), bounds-checked.
    fn chunk_len(self, bytes: &[u8], pos: usize, prefix: usize) -> crate::Result<usize> {
        let header_len = self.header_len();
        // The whole fixed header must be in range before any field is
        // read; `pos` may be an untrusted offset-table entry.
        need(bytes, pos, prefix + header_len, "chunk header")?;
        let body = pos + prefix;
        let payload = match self {
            Self::Scanline => read_size(bytes, body + 4, "scanline chunk size")?,
            Self::Tiled => read_size(bytes, body + 16, "tile chunk size")?,
            Self::DeepScanline | Self::DeepTile => {
                let sizes_at = body + header_len - 24;
                let table = read_u64(bytes, sizes_at, "deep packed table size")?;
                let data = read_u64(bytes, sizes_at + 8, "deep packed data size")?;
                usize::try_from(table)
                    .ok()
                    .zip(usize::try_from(data).ok())
                    .and_then(|(t, d)| t.checked_add(d))
                    .ok_or_else(|| {
                        ExrError::invalid(
                            "OpenEXR container: deep chunk sizes overflow the address space",
                        )
                    })?
            }
        };
        let total = (prefix + header_len).checked_add(payload).ok_or_else(|| {
            ExrError::invalid("OpenEXR container: chunk length overflows the address space")
        })?;
        need(bytes, pos, total, "chunk")?;
        Ok(total)
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

fn file_version(bytes: &[u8]) -> crate::Result<VersionField> {
    if !crate::probe(bytes) {
        return Err(ExrError::invalid("OpenEXR container: bad magic"));
    }
    if bytes.len() < 8 {
        return Err(ExrError::invalid(
            "OpenEXR container: file shorter than the magic + version field",
        ));
    }
    Ok(VersionField::from_u32(u32::from_le_bytes(
        bytes[4..8].try_into().unwrap(),
    )))
}

/// Split a file into its viewable parts (see the module docs).
fn demux_file(bytes: &[u8]) -> crate::Result<Demuxed> {
    let version = file_version(bytes)?;
    if !version.multipart {
        return demux_single_part(bytes, version);
    }
    demux_multipart(bytes, version)
}

fn demux_single_part(bytes: &[u8], version: VersionField) -> crate::Result<Demuxed> {
    let header = parse_header_allow_deep(bytes)?;
    if version.non_image {
        return Err(ExrError::unsupported(
            "OpenEXR container: deep image (variable samples per pixel) has no colour view; \
             use the parse_exr_deep_* API",
        ));
    }
    let req = extract_required(&header.attributes)?;
    let plan = plan_view(&req.channels, &header.attributes, "")?;
    let (width, height) = (req.data_window.width(), req.data_window.height());
    if width == 0 || height == 0 {
        return Err(ExrError::invalid(format!(
            "OpenEXR container: dataWindow {width}×{height} must both be > 0"
        )));
    }
    Ok(Demuxed {
        primary_attributes: header.attributes.clone(),
        part_count: 1,
        parts: vec![PartPacket {
            index: 0,
            name: string_attribute(&header.attributes, "name"),
            width,
            height,
            format: plan.format(),
            color: ColorInfo::from_attributes(&header.attributes),
            bytes: bytes.to_vec(),
        }],
    })
}

/// Per-part layout facts of a multi-part file gathered from the headers.
struct MultipartLayout {
    shapes: Vec<PartShape>,
    counts: Vec<usize>,
    /// Offset of part `i`'s offset table.
    table_offsets: Vec<usize>,
    /// First chunk record.
    chunk_start: usize,
}

fn multipart_layout(bytes: &[u8], headers: &[ParsedHeader]) -> crate::Result<MultipartLayout> {
    let mut shapes = Vec::with_capacity(headers.len());
    let mut counts = Vec::with_capacity(headers.len());
    let mut table_offsets = Vec::with_capacity(headers.len());
    // Every header shares the post-double-NUL offset.
    let mut pos = headers[0].end_offset;
    for (i, h) in headers.iter().enumerate() {
        let part_type = find_part_type(&h.attributes).ok_or_else(|| {
            ExrError::invalid(format!(
                "OpenEXR container: multi-part part {i} is missing the required 'type' attribute"
            ))
        })?;
        shapes.push(PartShape::from_type(&part_type, i)?);
        let count = find_chunk_count(&h.attributes).ok_or_else(|| {
            ExrError::invalid(format!(
                "OpenEXR container: multi-part part {i} is missing the required chunkCount \
                 attribute"
            ))
        })?;
        table_offsets.push(pos);
        let table_len = count.checked_mul(8).ok_or_else(|| {
            ExrError::invalid(format!(
                "OpenEXR container: part {i} chunkCount {count} overflows the address space"
            ))
        })?;
        pos = need(bytes, pos, table_len, "offset table")?;
        counts.push(count);
    }
    Ok(MultipartLayout {
        shapes,
        counts,
        table_offsets,
        chunk_start: pos,
    })
}

/// Walk every chunk record of a multi-part file in file order; returns
/// `(start, len)` per part (record start = the part-number prefix).
fn walk_chunks(bytes: &[u8], layout: &MultipartLayout) -> crate::Result<Vec<Vec<(usize, usize)>>> {
    let mut chunks: Vec<Vec<(usize, usize)>> = vec![Vec::new(); layout.shapes.len()];
    let total: usize = layout.counts.iter().sum();
    let mut pos = layout.chunk_start;
    for n in 0..total {
        let part = read_i32(bytes, pos, "chunk part number")?;
        let part = usize::try_from(part)
            .ok()
            .filter(|&p| p < layout.shapes.len())
            .ok_or_else(|| {
                ExrError::invalid(format!(
                    "OpenEXR container: chunk {n} at offset {pos} names part {part}, file has {} \
                     part(s)",
                    layout.shapes.len()
                ))
            })?;
        if chunks[part].len() >= layout.counts[part] {
            return Err(ExrError::invalid(format!(
                "OpenEXR container: part {part} has more chunks than its chunkCount {}",
                layout.counts[part]
            )));
        }
        let len = layout.shapes[part].chunk_len(bytes, pos, 4)?;
        chunks[part].push((pos, len));
        pos += len;
    }
    for (i, c) in chunks.iter().enumerate() {
        if c.len() != layout.counts[i] {
            return Err(ExrError::invalid(format!(
                "OpenEXR container: part {i} has {} chunk(s), chunkCount says {}",
                c.len(),
                layout.counts[i]
            )));
        }
    }
    Ok(chunks)
}

/// The order in which part `i`'s chunks enter the repacked offset table:
/// the file's own table order when it is fully populated, else file
/// order (some writers zero-fill the tables of parts beyond the first).
fn table_order(
    bytes: &[u8],
    layout: &MultipartLayout,
    part: usize,
    chunks: &[(usize, usize)],
) -> Vec<usize> {
    let by_start: HashMap<usize, usize> = chunks
        .iter()
        .enumerate()
        .map(|(k, &(s, _))| (s, k))
        .collect();
    let mut order = Vec::with_capacity(chunks.len());
    let mut used = vec![false; chunks.len()];
    for k in 0..chunks.len() {
        let Ok(off) = read_u64(bytes, layout.table_offsets[part] + k * 8, "offset table") else {
            return (0..chunks.len()).collect();
        };
        let Some(&idx) = usize::try_from(off).ok().and_then(|o| by_start.get(&o)) else {
            return (0..chunks.len()).collect();
        };
        if used[idx] {
            return (0..chunks.len()).collect();
        }
        used[idx] = true;
        order.push(idx);
    }
    order
}

/// Re-emit part `i` of a multi-part file as a single-part file.
fn repack_single_part(
    bytes: &[u8],
    version: VersionField,
    header: &ParsedHeader,
    shape: PartShape,
    chunks: &[(usize, usize)],
    order: &[usize],
) -> Vec<u8> {
    let mut raw = 2u32;
    if version.long_names {
        raw |= VERSION_LONG_NAMES;
    }
    if shape == PartShape::Tiled {
        raw |= VERSION_SINGLE_TILE;
    }
    let mut out = encode_header(VersionField::from_u32(raw), &header.attributes);
    let table_at = out.len();
    let mut data_at = table_at + chunks.len() * 8;
    out.resize(data_at, 0);
    for (slot, &k) in order.iter().enumerate() {
        let (start, len) = chunks[k];
        out[table_at + slot * 8..table_at + slot * 8 + 8]
            .copy_from_slice(&(data_at as u64).to_le_bytes());
        // Drop the 4-byte part-number prefix.
        out.extend_from_slice(&bytes[start + 4..start + len]);
        data_at += len - 4;
    }
    out
}

fn demux_multipart(bytes: &[u8], version: VersionField) -> crate::Result<Demuxed> {
    let headers = parse_multipart_headers(bytes)?;
    if headers.is_empty() {
        return Err(ExrError::invalid(
            "OpenEXR container: multi-part file declares no parts",
        ));
    }
    let layout = multipart_layout(bytes, &headers)?;
    let chunks = walk_chunks(bytes, &layout)?;

    let mut parts: Vec<PartPacket> = Vec::new();
    let mut primary_attributes: Option<Vec<Attribute>> = None;
    let mut skipped: Vec<String> = Vec::new();
    for (i, header) in headers.iter().enumerate() {
        let shape = layout.shapes[i];
        if shape.is_deep() {
            skipped.push(format!("part {i}: deep part has no colour view"));
            continue;
        }
        let req = extract_required(&header.attributes)?;
        let plan = match plan_view(&req.channels, &header.attributes, "") {
            Ok(p) => p,
            Err(ExrError::Unsupported(msg)) => {
                skipped.push(format!("part {i}: {msg}"));
                continue;
            }
            Err(e) => return Err(e),
        };
        let (width, height) = (req.data_window.width(), req.data_window.height());
        if width == 0 || height == 0 {
            return Err(ExrError::invalid(format!(
                "OpenEXR container: part {i} dataWindow {width}×{height} must both be > 0"
            )));
        }
        let order = table_order(bytes, &layout, i, &chunks[i]);
        let single = repack_single_part(bytes, version, header, shape, &chunks[i], &order);
        if primary_attributes.is_none() {
            primary_attributes = Some(header.attributes.clone());
        }
        parts.push(PartPacket {
            index: i as u32,
            name: string_attribute(&header.attributes, "name"),
            width,
            height,
            format: plan.format(),
            color: ColorInfo::from_attributes(&header.attributes),
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
        part_count: headers.len() as u32,
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

/// One single-part packet prepared for the multi-part combine.
struct FlatPart<'a> {
    file: &'a [u8],
    /// The packet's own `displayWindow` (the combined file shares one).
    display_window: Box2i,
    attributes: Vec<Attribute>,
    /// `(start, len)` of every chunk record in `file`, table order.
    chunks: Vec<(usize, usize)>,
}

/// `chunkCount` of a single-part flat file from its header.
fn single_part_chunk_count(header: &ParsedHeader, idx: usize) -> crate::Result<usize> {
    let req = extract_required(&header.attributes)?;
    let (width, height) = (req.data_window.width(), req.data_window.height());
    if width == 0 || height == 0 {
        return Err(ExrError::invalid(format!(
            "OpenEXR muxer: packet {idx} dataWindow {width}×{height} must both be > 0"
        )));
    }
    if !header.version.single_tile {
        return Ok(height.div_ceil(req.compression.scanlines_per_block()) as usize);
    }
    let tiles = header
        .attributes
        .iter()
        .find(|a| a.name == "tiles")
        .ok_or_else(|| {
            ExrError::invalid(format!(
                "OpenEXR muxer: tiled packet {idx} is missing the required 'tiles' attribute"
            ))
        })?;
    let td = tiledesc_from_attribute(&tiles.value)?;
    if td.x_size == 0 || td.y_size == 0 || td.level_mode > 2 {
        return Err(ExrError::invalid(format!(
            "OpenEXR muxer: tiled packet {idx} has an invalid tiledesc ({}×{}, level mode {})",
            td.x_size, td.y_size, td.level_mode
        )));
    }
    Ok(compute_total_tiles(
        td.level_mode,
        width,
        height,
        td.x_size,
        td.y_size,
        td.round_mode != 0,
    ))
}

fn prepare_flat_part<'a>(
    file: &'a [u8],
    idx: usize,
    used_names: &mut HashSet<String>,
) -> crate::Result<FlatPart<'a>> {
    let header = parse_header(file).map_err(|e| match e {
        ExrError::Unsupported(msg) => ExrError::unsupported(format!(
            "OpenEXR muxer: multi-part output combines flat single-part packets only; packet \
             {idx}: {msg}"
        )),
        other => other,
    })?;
    let tiled = header.version.single_tile;
    let shape = if tiled {
        PartShape::Tiled
    } else {
        PartShape::Scanline
    };
    let count = single_part_chunk_count(&header, idx)?;
    let display_window = extract_required(&header.attributes)?.display_window;
    need(
        file,
        header.end_offset,
        count.checked_mul(8).ok_or_else(|| {
            ExrError::invalid(format!(
                "OpenEXR muxer: packet {idx} chunk count {count} overflows the address space"
            ))
        })?,
        "offset table",
    )?;
    let mut chunks = Vec::with_capacity(count);
    for k in 0..count {
        let off = read_u64(file, header.end_offset + k * 8, "offset table")?;
        let start = usize::try_from(off)
            .ok()
            .filter(|&o| o != 0)
            .ok_or_else(|| {
                ExrError::invalid(format!(
                "OpenEXR muxer: packet {idx} offset table entry {k} is {off} (unpopulated table)"
            ))
            })?;
        let len = shape.chunk_len(file, start, 0)?;
        chunks.push((start, len));
    }

    // Attribute table: everything the packet carries except the three
    // multi-part structural attributes, which are re-derived.
    let mut attributes: Vec<Attribute> = header
        .attributes
        .iter()
        .filter(|a| !matches!(a.name.as_str(), "name" | "type" | "chunkCount"))
        .cloned()
        .collect();
    let mut name = string_attribute(&header.attributes, "name")
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| format!("part{idx}"));
    if used_names.contains(&name) {
        let base = name.clone();
        let mut n = idx;
        loop {
            name = format!("{base}.{n}");
            if !used_names.contains(&name) {
                break;
            }
            n += 1;
        }
    }
    used_names.insert(name.clone());
    attributes.push(Attribute {
        name: "name".to_string(),
        value: AttributeValue::String(name),
    });
    attributes.push(Attribute {
        name: "type".to_string(),
        value: AttributeValue::String(
            if tiled { "tiledimage" } else { "scanlineimage" }.to_string(),
        ),
    });
    attributes.push(Attribute {
        name: "chunkCount".to_string(),
        value: AttributeValue::Int(count as i32),
    });
    Ok(FlatPart {
        file,
        display_window,
        attributes,
        chunks,
    })
}

/// Serialise one part's attribute table (NUL-terminated), reporting
/// whether any name needs the long-names bit.
fn attribute_table(attributes: &[Attribute]) -> (Vec<u8>, bool) {
    let mut out = Vec::new();
    let mut long = false;
    for a in attributes {
        let (type_name, payload) = encode_attribute_value(&a.value);
        long |= a.name.len() > SHORT_NAME_MAX || type_name.len() > SHORT_NAME_MAX;
        out.extend_from_slice(a.name.as_bytes());
        out.push(0);
        out.extend_from_slice(type_name.as_bytes());
        out.push(0);
        out.extend_from_slice(&(payload.len() as i32).to_le_bytes());
        out.extend_from_slice(&payload);
    }
    out.push(0);
    (out, long)
}

/// Combine single-part flat files into one multi-part file, byte for
/// byte (the inverse of the demuxer's repack).
pub(crate) fn combine_parts(files: &[Vec<u8>]) -> crate::Result<Vec<u8>> {
    if files.len() < 2 {
        return Err(ExrError::invalid(
            "OpenEXR muxer: combining needs at least two packets",
        ));
    }
    let mut used = HashSet::new();
    let mut parts = Vec::with_capacity(files.len());
    for (i, f) in files.iter().enumerate() {
        parts.push(prepare_flat_part(f, i, &mut used)?);
    }
    // Every part of a multi-part file carries the SAME displayWindow (a
    // file-global attribute; the reference reader refuses parts that
    // disagree). Share the union of the packets' display windows — for
    // origin-anchored encoder output this is the largest extent, the rule
    // `encode_all` applies.
    let shared = parts
        .iter()
        .map(|p| p.display_window)
        .reduce(|a, b| Box2i {
            x_min: a.x_min.min(b.x_min),
            y_min: a.y_min.min(b.y_min),
            x_max: a.x_max.max(b.x_max),
            y_max: a.y_max.max(b.y_max),
        })
        .expect("at least two parts");
    for p in &mut parts {
        for a in &mut p.attributes {
            if a.name == "displayWindow" {
                a.value = AttributeValue::Box2i(shared);
            }
        }
    }
    let mut tables = Vec::with_capacity(parts.len());
    let mut long_names = false;
    for p in &parts {
        let (t, long) = attribute_table(&p.attributes);
        long_names |= long;
        tables.push(t);
    }
    let mut raw = 2 | VERSION_MULTIPART;
    if long_names {
        raw |= VERSION_LONG_NAMES;
    }
    let mut out = Vec::new();
    out.extend_from_slice(&EXR_MAGIC.to_le_bytes());
    out.extend_from_slice(&raw.to_le_bytes());
    for t in &tables {
        out.extend_from_slice(t);
    }
    out.push(0); // end of all headers
    let total: usize = parts.iter().map(|p| p.chunks.len()).sum();
    let mut data_at = out.len() + total * 8;
    for p in &parts {
        for &(_, len) in &p.chunks {
            out.extend_from_slice(&(data_at as u64).to_le_bytes());
            data_at += 4 + len;
        }
    }
    for (i, p) in parts.iter().enumerate() {
        for &(start, len) in &p.chunks {
            out.extend_from_slice(&(i as i32).to_le_bytes());
            out.extend_from_slice(&p.file[start..start + len]);
        }
    }
    Ok(out)
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
