//! Magic + version field + attribute table parser.
//!
//! Layout per the OpenEXR spec:
//! * bytes 0..4   = magic `0x76 0x2F 0x31 0x01` (little-endian `0x0131_2F76`)
//! * bytes 4..8   = version field (u32 LE):
//!     - low byte = format version (currently 2)
//!     - bit 9    = 0x200  single-tile
//!     - bit 10   = 0x400  long names (max 255 vs 31)
//!     - bit 11   = 0x800  non-image / deep data
//!     - bit 12   = 0x1000 multipart
//! * then a sequence of `(name, type-name, size, payload)` attribute
//!   entries terminated by a single null byte (empty name).
//!
//! `name` and `type-name` are null-terminated ASCII strings; `size` is
//! a 4-byte signed int holding the payload length in bytes.

use crate::error::{ExrError, Result};
use crate::tiled::TileDesc;
use crate::types::{
    Attribute, AttributeValue, Box2f, Box2i, Channel, Chromaticities, Compression, LineOrder,
    PixelType, EXR_MAGIC,
};

/// Decoded version field flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VersionField {
    pub raw: u32,
    pub format_version: u8,
    pub single_tile: bool,
    pub long_names: bool,
    pub non_image: bool,
    pub multipart: bool,
}

impl VersionField {
    pub fn from_u32(raw: u32) -> Self {
        Self {
            raw,
            format_version: (raw & 0xFF) as u8,
            single_tile: (raw & 0x200) != 0,
            long_names: (raw & 0x400) != 0,
            non_image: (raw & 0x800) != 0,
            multipart: (raw & 0x1000) != 0,
        }
    }

    /// Encode the flags back to the on-disk u32. Round-trips
    /// `from_u32(x).to_u32() == x` for any well-formed `x` (any unknown
    /// bits are also preserved via `raw`).
    pub fn to_u32(&self) -> u32 {
        self.raw
    }
}

/// Result of header parsing: version, attributes, and the byte offset
/// in `bytes` immediately past the trailing null that terminates the
/// attribute list (i.e. the start of the line/chunk offset table).
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedHeader {
    pub version: VersionField,
    pub attributes: Vec<Attribute>,
    /// Offset in the source slice past the header's terminating NUL.
    pub end_offset: usize,
}

/// Cursor walking a `&[u8]` left-to-right with bounds-checked little-endian
/// readers. Kept private to header.rs (encoder and decoder copy the small
/// helpers they need).
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn require(&self, n: usize) -> Result<()> {
        if self.pos + n > self.buf.len() {
            Err(ExrError::invalid(format!(
                "unexpected EOF reading {n} bytes at offset {}",
                self.pos
            )))
        } else {
            Ok(())
        }
    }
    fn u8(&mut self) -> Result<u8> {
        self.require(1)?;
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }
    fn u32(&mut self) -> Result<u32> {
        self.require(4)?;
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.require(n)?;
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn null_string(&mut self, max_len: usize) -> Result<String> {
        // Search for the first NUL within the next `max_len + 1` bytes.
        let limit = (self.pos + max_len + 1).min(self.buf.len());
        let mut end = self.pos;
        while end < limit && self.buf[end] != 0 {
            end += 1;
        }
        if end >= limit {
            return Err(ExrError::invalid(format!(
                "null-terminated string longer than {max_len} bytes at offset {}",
                self.pos
            )));
        }
        let s = std::str::from_utf8(&self.buf[self.pos..end])
            .map_err(|e| ExrError::invalid(format!("non-UTF8 string in header: {e}")))?
            .to_string();
        self.pos = end + 1; // skip NUL
        Ok(s)
    }
}

/// Parse the magic, version, and attribute table of a single-part EXR file.
///
/// Returns the version field, the attribute list, and the offset of the
/// next byte past the terminating NUL — i.e. where the line offset
/// table begins for a single-part scanline file.
pub fn parse_header(bytes: &[u8]) -> Result<ParsedHeader> {
    let mut c = Cursor::new(bytes);
    let magic = c.u32()?;
    if magic != EXR_MAGIC {
        return Err(ExrError::invalid(format!(
            "bad magic 0x{magic:08x}, expected 0x{EXR_MAGIC:08x}"
        )));
    }
    let version = VersionField::from_u32(c.u32()?);
    if version.multipart {
        return Err(ExrError::unsupported(
            "multipart EXR: use parse_exr_multipart()".to_string(),
        ));
    }
    if version.non_image {
        return Err(ExrError::unsupported(
            "non-image / deep-data EXR files (deferred)".to_string(),
        ));
    }
    // single_tile is supported (ONE_LEVEL + MIPMAP + RIPMAP in round 3);
    // the per-file tile-decode path verifies the tiledesc attribute and
    // offset table further along.

    let max_name = if version.long_names { 255 } else { 31 };

    let mut attributes = Vec::new();
    loop {
        // Empty name (immediate NUL) terminates the attribute table.
        if c.pos >= bytes.len() {
            return Err(ExrError::invalid(
                "EOF before header NUL terminator".to_string(),
            ));
        }
        if bytes[c.pos] == 0 {
            c.pos += 1;
            break;
        }
        let name = c.null_string(max_name)?;
        let type_name = c.null_string(max_name)?;
        let size = c.i32()?;
        if size < 0 {
            return Err(ExrError::invalid(format!(
                "attribute {name} has negative size {size}"
            )));
        }
        let payload = c.bytes(size as usize)?.to_vec();
        let value = parse_attribute_value(&type_name, &payload)?;
        attributes.push(Attribute { name, value });
    }
    Ok(ParsedHeader {
        version,
        attributes,
        end_offset: c.pos,
    })
}

/// Parse a multi-part EXR file and return one [`ParsedHeader`] per part.
///
/// Binary layout:
/// ```text
/// magic(4) | version(4)
/// | header_0 ... NUL   (NUL = empty-name attribute terminator)
/// | header_1 ... NUL
/// | ...
/// | NUL                (extra NUL = end of all headers)
/// | offset_table_0 | offset_table_1 | ...
/// ```
/// The `end_offset` of the LAST returned header points to where the
/// first offset table begins (the position right after the double-NUL).
///
/// Returns `Err` if the magic is bad, the `multipart` bit is NOT set,
/// or the stream is truncated.
///
/// The `non_image` (deep) bit is accepted — multi-part deep files set
/// `version` to `0x1800` (bits 11 + 12) per the OpenEXR spec.
/// Per-part `type` discrimination ("scanlineimage" vs "deepscanline"
/// vs "tiledimage" vs "deeptile") is left to the caller.
pub fn parse_multipart_headers(bytes: &[u8]) -> Result<Vec<ParsedHeader>> {
    let mut c = Cursor::new(bytes);
    let magic = c.u32()?;
    if magic != EXR_MAGIC {
        return Err(ExrError::invalid(format!(
            "bad magic 0x{magic:08x}, expected 0x{EXR_MAGIC:08x}"
        )));
    }
    let version = VersionField::from_u32(c.u32()?);
    if !version.multipart {
        return Err(ExrError::invalid(
            "parse_multipart_headers called on a non-multipart EXR file".to_string(),
        ));
    }
    // non_image (deep) is now permitted — the caller dispatches per-part
    // by reading the `type` attribute. parse_exr_multipart still rejects
    // any deep part (with a clear pointer to parse_exr_deep_multipart);
    // parse_exr_deep_multipart accepts only deep parts.

    let max_name = if version.long_names { 255 } else { 31 };
    let mut parts: Vec<ParsedHeader> = Vec::new();

    loop {
        // A NUL byte at the start of a header terminates the multi-part
        // header list (the "extra NUL" after all per-part headers).
        if c.pos >= bytes.len() {
            return Err(ExrError::invalid(
                "EOF before multi-part header double-NUL terminator".to_string(),
            ));
        }
        if bytes[c.pos] == 0 {
            c.pos += 1; // consume the final NUL
            break;
        }

        // Parse one per-part header.
        let part_start_version = version;
        let mut attributes = Vec::new();
        loop {
            if c.pos >= bytes.len() {
                return Err(ExrError::invalid(
                    "EOF inside multi-part header".to_string(),
                ));
            }
            if bytes[c.pos] == 0 {
                c.pos += 1; // per-part header terminator
                break;
            }
            let name = c.null_string(max_name)?;
            let type_name = c.null_string(max_name)?;
            let size = c.i32()?;
            if size < 0 {
                return Err(ExrError::invalid(format!(
                    "attribute {name} has negative size {size}"
                )));
            }
            let payload = c.bytes(size as usize)?.to_vec();
            let value = parse_attribute_value(&type_name, &payload)?;
            attributes.push(Attribute { name, value });
        }
        parts.push(ParsedHeader {
            version: part_start_version,
            attributes,
            end_offset: c.pos,
        });
    }

    // Fix up end_offset: all parts share the same post-all-headers offset
    // (the position right after the final NUL). Only the last part's
    // end_offset correctly reflects this; update all parts for consistency.
    for part in &mut parts {
        part.end_offset = c.pos;
    }

    Ok(parts)
}

/// Decode an attribute payload according to its declared type name. For
/// types this crate doesn't need explicitly we keep the bytes as
/// `AttributeValue::Other` so the caller can inspect (or re-emit) them.
pub fn parse_attribute_value(type_name: &str, data: &[u8]) -> Result<AttributeValue> {
    match type_name {
        "chlist" => Ok(AttributeValue::Channels(parse_channel_list(data)?)),
        "compression" => {
            if data.len() != 1 {
                return Err(ExrError::invalid(format!(
                    "compression payload size {} != 1",
                    data.len()
                )));
            }
            let c = Compression::from_byte(data[0]).ok_or_else(|| {
                ExrError::invalid(format!("unknown compression byte {}", data[0]))
            })?;
            Ok(AttributeValue::Compression(c))
        }
        "box2i" => {
            if data.len() != 16 {
                return Err(ExrError::invalid(format!(
                    "box2i payload size {} != 16",
                    data.len()
                )));
            }
            let x_min = i32::from_le_bytes(data[0..4].try_into().unwrap());
            let y_min = i32::from_le_bytes(data[4..8].try_into().unwrap());
            let x_max = i32::from_le_bytes(data[8..12].try_into().unwrap());
            let y_max = i32::from_le_bytes(data[12..16].try_into().unwrap());
            Ok(AttributeValue::Box2i(Box2i {
                x_min,
                y_min,
                x_max,
                y_max,
            }))
        }
        "box2f" => {
            // Same on-disk shape as box2i with the four fields stored as
            // little-endian f32 instead of i32. 16 bytes total.
            if data.len() != 16 {
                return Err(ExrError::invalid(format!(
                    "box2f payload size {} != 16",
                    data.len()
                )));
            }
            let x_min = f32::from_le_bytes(data[0..4].try_into().unwrap());
            let y_min = f32::from_le_bytes(data[4..8].try_into().unwrap());
            let x_max = f32::from_le_bytes(data[8..12].try_into().unwrap());
            let y_max = f32::from_le_bytes(data[12..16].try_into().unwrap());
            Ok(AttributeValue::Box2f(Box2f {
                x_min,
                y_min,
                x_max,
                y_max,
            }))
        }
        "lineOrder" => {
            if data.len() != 1 {
                return Err(ExrError::invalid(format!(
                    "lineOrder payload size {} != 1",
                    data.len()
                )));
            }
            let l = LineOrder::from_byte(data[0])
                .ok_or_else(|| ExrError::invalid(format!("unknown lineOrder byte {}", data[0])))?;
            Ok(AttributeValue::LineOrder(l))
        }
        "float" => {
            if data.len() != 4 {
                return Err(ExrError::invalid(format!(
                    "float payload size {} != 4",
                    data.len()
                )));
            }
            Ok(AttributeValue::Float(f32::from_le_bytes(
                data[0..4].try_into().unwrap(),
            )))
        }
        "v2f" => {
            if data.len() != 8 {
                return Err(ExrError::invalid(format!(
                    "v2f payload size {} != 8",
                    data.len()
                )));
            }
            let x = f32::from_le_bytes(data[0..4].try_into().unwrap());
            let y = f32::from_le_bytes(data[4..8].try_into().unwrap());
            Ok(AttributeValue::V2f(x, y))
        }
        "int" => {
            if data.len() != 4 {
                return Err(ExrError::invalid(format!(
                    "int payload size {} != 4",
                    data.len()
                )));
            }
            Ok(AttributeValue::Int(i32::from_le_bytes(
                data[0..4].try_into().unwrap(),
            )))
        }
        "double" => {
            if data.len() != 8 {
                return Err(ExrError::invalid(format!(
                    "double payload size {} != 8",
                    data.len()
                )));
            }
            Ok(AttributeValue::Double(f64::from_le_bytes(
                data[0..8].try_into().unwrap(),
            )))
        }
        "string" => {
            // The outer attribute size field is the length of the string
            // payload; no NUL terminator is stored inside the payload.
            let s = std::str::from_utf8(data)
                .map_err(|e| ExrError::invalid(format!("non-UTF8 string payload: {e}")))?
                .to_string();
            Ok(AttributeValue::String(s))
        }
        "v2i" => {
            if data.len() != 8 {
                return Err(ExrError::invalid(format!(
                    "v2i payload size {} != 8",
                    data.len()
                )));
            }
            let x = i32::from_le_bytes(data[0..4].try_into().unwrap());
            let y = i32::from_le_bytes(data[4..8].try_into().unwrap());
            Ok(AttributeValue::V2i(x, y))
        }
        "v3i" => {
            if data.len() != 12 {
                return Err(ExrError::invalid(format!(
                    "v3i payload size {} != 12",
                    data.len()
                )));
            }
            let x = i32::from_le_bytes(data[0..4].try_into().unwrap());
            let y = i32::from_le_bytes(data[4..8].try_into().unwrap());
            let z = i32::from_le_bytes(data[8..12].try_into().unwrap());
            Ok(AttributeValue::V3i(x, y, z))
        }
        "v3f" => {
            if data.len() != 12 {
                return Err(ExrError::invalid(format!(
                    "v3f payload size {} != 12",
                    data.len()
                )));
            }
            let x = f32::from_le_bytes(data[0..4].try_into().unwrap());
            let y = f32::from_le_bytes(data[4..8].try_into().unwrap());
            let z = f32::from_le_bytes(data[8..12].try_into().unwrap());
            Ok(AttributeValue::V3f(x, y, z))
        }
        "m33f" => {
            if data.len() != 36 {
                return Err(ExrError::invalid(format!(
                    "m33f payload size {} != 36",
                    data.len()
                )));
            }
            let mut m = [0f32; 9];
            for (i, slot) in m.iter_mut().enumerate() {
                let off = i * 4;
                *slot = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            }
            Ok(AttributeValue::M33f(m))
        }
        "m44f" => {
            if data.len() != 64 {
                return Err(ExrError::invalid(format!(
                    "m44f payload size {} != 64",
                    data.len()
                )));
            }
            let mut m = [0f32; 16];
            for (i, slot) in m.iter_mut().enumerate() {
                let off = i * 4;
                *slot = f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            }
            Ok(AttributeValue::M44f(m))
        }
        "chromaticities" => {
            if data.len() != 32 {
                return Err(ExrError::invalid(format!(
                    "chromaticities payload size {} != 32",
                    data.len()
                )));
            }
            let f = |off: usize| f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
            Ok(AttributeValue::Chromaticities(Chromaticities {
                red_x: f(0),
                red_y: f(4),
                green_x: f(8),
                green_y: f(12),
                blue_x: f(16),
                blue_y: f(20),
                white_x: f(24),
                white_y: f(28),
            }))
        }
        "tiledesc" => Ok(AttributeValue::TileDesc(TileDesc::from_bytes(data)?)),
        _ => Ok(AttributeValue::Other {
            type_name: type_name.to_string(),
            data: data.to_vec(),
        }),
    }
}

/// Parse a `chlist` payload: a sequence of channel descriptors followed
/// by a single NUL byte.
///
/// Per descriptor (the spec says max name length depends on the
/// version-field long-names bit; chlist re-uses the file-wide rule, so
/// we accept up to 255 here and let the caller's name limit catch
/// over-long names if they care):
///
/// ```text
/// name        : null-terminated string
/// pixelType   : i32 LE  (0=UINT 1=HALF 2=FLOAT)
/// pLinear     : u8
/// reserved[3] : 3 bytes (should be zero)
/// xSampling   : i32 LE
/// ySampling   : i32 LE
/// ```
pub fn parse_channel_list(data: &[u8]) -> Result<Vec<Channel>> {
    let mut c = Cursor::new(data);
    let mut channels = Vec::new();
    loop {
        if c.pos >= data.len() {
            return Err(ExrError::invalid(
                "channel list missing NUL terminator".to_string(),
            ));
        }
        if data[c.pos] == 0 {
            c.pos += 1;
            break;
        }
        let name = c.null_string(255)?;
        let pixel_type_int = c.i32()?;
        let pixel_type = PixelType::from_int(pixel_type_int).ok_or_else(|| {
            ExrError::invalid(format!(
                "channel '{name}': unknown pixelType {pixel_type_int}"
            ))
        })?;
        let p_linear = c.u8()? != 0;
        let _reserved = c.bytes(3)?;
        let x_sampling = c.i32()?;
        let y_sampling = c.i32()?;
        channels.push(Channel {
            name,
            pixel_type,
            p_linear,
            x_sampling,
            y_sampling,
        });
    }
    if c.pos != data.len() {
        return Err(ExrError::invalid(format!(
            "trailing bytes after channel list: {} extra",
            data.len() - c.pos
        )));
    }
    Ok(channels)
}

/// Encode a [`Channel`] list back to a `chlist` payload (matches
/// [`parse_channel_list`] inverse).
pub fn encode_channel_list(channels: &[Channel]) -> Vec<u8> {
    let mut out = Vec::with_capacity(channels.len() * 32);
    for ch in channels {
        out.extend_from_slice(ch.name.as_bytes());
        out.push(0);
        out.extend_from_slice(&(ch.pixel_type as i32).to_le_bytes());
        out.push(if ch.p_linear { 1 } else { 0 });
        out.extend_from_slice(&[0u8, 0, 0]);
        out.extend_from_slice(&ch.x_sampling.to_le_bytes());
        out.extend_from_slice(&ch.y_sampling.to_le_bytes());
    }
    out.push(0); // chlist terminator
    out
}

/// Encode an attribute value back to its on-disk payload bytes.
pub fn encode_attribute_value(value: &AttributeValue) -> (String, Vec<u8>) {
    match value {
        AttributeValue::Channels(chs) => ("chlist".to_string(), encode_channel_list(chs)),
        AttributeValue::Compression(c) => ("compression".to_string(), vec![*c as u8]),
        AttributeValue::Box2i(b) => {
            let mut v = Vec::with_capacity(16);
            v.extend_from_slice(&b.x_min.to_le_bytes());
            v.extend_from_slice(&b.y_min.to_le_bytes());
            v.extend_from_slice(&b.x_max.to_le_bytes());
            v.extend_from_slice(&b.y_max.to_le_bytes());
            ("box2i".to_string(), v)
        }
        AttributeValue::Box2f(b) => {
            let mut v = Vec::with_capacity(16);
            v.extend_from_slice(&b.x_min.to_le_bytes());
            v.extend_from_slice(&b.y_min.to_le_bytes());
            v.extend_from_slice(&b.x_max.to_le_bytes());
            v.extend_from_slice(&b.y_max.to_le_bytes());
            ("box2f".to_string(), v)
        }
        AttributeValue::LineOrder(l) => ("lineOrder".to_string(), vec![*l as u8]),
        AttributeValue::Float(f) => ("float".to_string(), f.to_le_bytes().to_vec()),
        AttributeValue::V2f(x, y) => {
            let mut v = Vec::with_capacity(8);
            v.extend_from_slice(&x.to_le_bytes());
            v.extend_from_slice(&y.to_le_bytes());
            ("v2f".to_string(), v)
        }
        AttributeValue::Int(i) => ("int".to_string(), i.to_le_bytes().to_vec()),
        AttributeValue::Double(d) => ("double".to_string(), d.to_le_bytes().to_vec()),
        AttributeValue::String(s) => ("string".to_string(), s.as_bytes().to_vec()),
        AttributeValue::V2i(x, y) => {
            let mut v = Vec::with_capacity(8);
            v.extend_from_slice(&x.to_le_bytes());
            v.extend_from_slice(&y.to_le_bytes());
            ("v2i".to_string(), v)
        }
        AttributeValue::V3i(x, y, z) => {
            let mut v = Vec::with_capacity(12);
            v.extend_from_slice(&x.to_le_bytes());
            v.extend_from_slice(&y.to_le_bytes());
            v.extend_from_slice(&z.to_le_bytes());
            ("v3i".to_string(), v)
        }
        AttributeValue::V3f(x, y, z) => {
            let mut v = Vec::with_capacity(12);
            v.extend_from_slice(&x.to_le_bytes());
            v.extend_from_slice(&y.to_le_bytes());
            v.extend_from_slice(&z.to_le_bytes());
            ("v3f".to_string(), v)
        }
        AttributeValue::M33f(m) => {
            let mut v = Vec::with_capacity(36);
            for f in m {
                v.extend_from_slice(&f.to_le_bytes());
            }
            ("m33f".to_string(), v)
        }
        AttributeValue::M44f(m) => {
            let mut v = Vec::with_capacity(64);
            for f in m {
                v.extend_from_slice(&f.to_le_bytes());
            }
            ("m44f".to_string(), v)
        }
        AttributeValue::Chromaticities(c) => {
            let mut v = Vec::with_capacity(32);
            v.extend_from_slice(&c.red_x.to_le_bytes());
            v.extend_from_slice(&c.red_y.to_le_bytes());
            v.extend_from_slice(&c.green_x.to_le_bytes());
            v.extend_from_slice(&c.green_y.to_le_bytes());
            v.extend_from_slice(&c.blue_x.to_le_bytes());
            v.extend_from_slice(&c.blue_y.to_le_bytes());
            v.extend_from_slice(&c.white_x.to_le_bytes());
            v.extend_from_slice(&c.white_y.to_le_bytes());
            ("chromaticities".to_string(), v)
        }
        AttributeValue::TileDesc(td) => ("tiledesc".to_string(), td.to_bytes().to_vec()),
        AttributeValue::Other { type_name, data } => (type_name.clone(), data.clone()),
    }
}

/// Encode a full header (magic + version + attribute table + NUL).
pub fn encode_header(version: VersionField, attributes: &[Attribute]) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    out.extend_from_slice(&EXR_MAGIC.to_le_bytes());
    out.extend_from_slice(&version.to_u32().to_le_bytes());
    for a in attributes {
        out.extend_from_slice(a.name.as_bytes());
        out.push(0);
        let (type_name, payload) = encode_attribute_value(&a.value);
        out.extend_from_slice(type_name.as_bytes());
        out.push(0);
        out.extend_from_slice(&(payload.len() as i32).to_le_bytes());
        out.extend_from_slice(&payload);
    }
    out.push(0); // header terminator
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_roundtrip() {
        let v = VersionField::from_u32(0x0000_0002);
        assert_eq!(v.format_version, 2);
        assert!(!v.long_names);
        assert!(!v.multipart);
        assert_eq!(v.to_u32(), 0x0000_0002);
    }

    #[test]
    fn channel_list_roundtrip() {
        let chs = vec![
            Channel {
                name: "A".to_string(),
                pixel_type: PixelType::Half,
                p_linear: false,
                x_sampling: 1,
                y_sampling: 1,
            },
            Channel {
                name: "B".to_string(),
                pixel_type: PixelType::Float,
                p_linear: true,
                x_sampling: 1,
                y_sampling: 1,
            },
        ];
        let bytes = encode_channel_list(&chs);
        let chs2 = parse_channel_list(&bytes).unwrap();
        assert_eq!(chs, chs2);
    }

    #[test]
    fn attribute_value_roundtrip_compression_and_box2i() {
        let attrs = vec![
            Attribute {
                name: "compression".into(),
                value: AttributeValue::Compression(Compression::Zip),
            },
            Attribute {
                name: "dataWindow".into(),
                value: AttributeValue::Box2i(Box2i {
                    x_min: 0,
                    y_min: 0,
                    x_max: 7,
                    y_max: 3,
                }),
            },
            Attribute {
                name: "lineOrder".into(),
                value: AttributeValue::LineOrder(LineOrder::IncreasingY),
            },
        ];
        let v = VersionField::from_u32(2);
        let raw = encode_header(v, &attrs);
        let parsed = parse_header(&raw).unwrap();
        assert_eq!(parsed.version.format_version, 2);
        assert_eq!(parsed.attributes, attrs);
    }
}
