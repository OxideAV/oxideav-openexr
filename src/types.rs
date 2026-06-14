//! Shared type definitions for `oxideav-openexr`: pixel/compression/line-order
//! enums, `Box2i`, `Channel`, and the public `Attribute` value enum.

/// EXR magic number `20000630` (little-endian: `0x76 0x2F 0x31 0x01`).
pub const EXR_MAGIC: u32 = 0x0131_2F76;

/// Per-channel pixel storage type. `chlist`'s "pixel type" int.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelType {
    /// Unsigned 32-bit integer.
    Uint = 0,
    /// IEEE 754-2008 binary16.
    Half = 1,
    /// IEEE 754-2008 binary32.
    Float = 2,
}

impl PixelType {
    pub fn from_int(v: i32) -> Option<Self> {
        match v {
            0 => Some(PixelType::Uint),
            1 => Some(PixelType::Half),
            2 => Some(PixelType::Float),
            _ => None,
        }
    }
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            PixelType::Uint => 4,
            PixelType::Half => 2,
            PixelType::Float => 4,
        }
    }
}

/// `compression` attribute value. One byte on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// No compression. 1 scanline per block.
    None = 0,
    /// RLE. 1 scanline per block.
    Rle = 1,
    /// Per-scanline zlib. 1 scanline per block.
    Zips = 2,
    /// 16-scanline zlib. 16 scanlines per block.
    Zip = 3,
    /// PIZ. 32 scanlines per block.
    Piz = 4,
    /// Pxr24. 16 scanlines per block.
    Pxr24 = 5,
    /// B44. 32 scanlines per block.
    B44 = 6,
    /// B44A. 32 scanlines per block.
    B44a = 7,
    /// DWAA. 32 scanlines per block.
    Dwaa = 8,
    /// DWAB. 256 scanlines per block.
    Dwab = 9,
}

impl Compression {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(Compression::None),
            1 => Some(Compression::Rle),
            2 => Some(Compression::Zips),
            3 => Some(Compression::Zip),
            4 => Some(Compression::Piz),
            5 => Some(Compression::Pxr24),
            6 => Some(Compression::B44),
            7 => Some(Compression::B44a),
            8 => Some(Compression::Dwaa),
            9 => Some(Compression::Dwab),
            _ => None,
        }
    }
    pub fn scanlines_per_block(&self) -> u32 {
        match self {
            Compression::None | Compression::Rle | Compression::Zips => 1,
            Compression::Zip | Compression::Pxr24 => 16,
            Compression::Piz | Compression::B44 | Compression::B44a | Compression::Dwaa => 32,
            Compression::Dwab => 256,
        }
    }
}

/// `lineOrder` attribute value. One byte on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineOrder {
    IncreasingY = 0,
    DecreasingY = 1,
    RandomY = 2,
}

impl LineOrder {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0 => Some(LineOrder::IncreasingY),
            1 => Some(LineOrder::DecreasingY),
            2 => Some(LineOrder::RandomY),
            _ => None,
        }
    }
}

/// `box2i` attribute value: `(xMin, yMin, xMax, yMax)`. Bounds are
/// inclusive on both ends, so width == xMax-xMin+1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Box2i {
    pub x_min: i32,
    pub y_min: i32,
    pub x_max: i32,
    pub y_max: i32,
}

impl Box2i {
    /// Box width in pixels. The fields come straight off the wire, so a
    /// hostile `dataWindow` can pair `x_min = i32::MIN` with
    /// `x_max = i32::MAX`; computing `x_max - x_min + 1` in `i32` would
    /// overflow (a debug-build panic, a silent wrap in release). Widen to
    /// `i64` for the subtraction and clamp the result into `u32`. A
    /// well-formed box (`x_max >= x_min`, dimensions inside `u32`) is
    /// unaffected; a degenerate box yields a finite extent that the
    /// caller's downstream EOF / size checks reject.
    pub fn width(&self) -> u32 {
        let w = self.x_max as i64 - self.x_min as i64 + 1;
        w.clamp(0, u32::MAX as i64) as u32
    }
    pub fn height(&self) -> u32 {
        let h = self.y_max as i64 - self.y_min as i64 + 1;
        h.clamp(0, u32::MAX as i64) as u32
    }
}

/// `box2f` attribute value: `(xMin, yMin, xMax, yMax)` as four
/// little-endian `f32` — 16 bytes on disk, identical field layout to
/// [`Box2i`] but with floating-point coordinates. The type name on disk
/// is `"box2f"`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Box2f {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

/// One entry in the `channels` (`chlist`) attribute.
#[derive(Debug, Clone, PartialEq)]
pub struct Channel {
    pub name: String,
    pub pixel_type: PixelType,
    /// 0 or 1 — non-zero means linear color space.
    pub p_linear: bool,
    pub x_sampling: i32,
    pub y_sampling: i32,
}

/// Chromaticities attribute payload: four CIE-xy primaries.
///
/// On disk this is eight consecutive little-endian `f32` values in the
/// order `red.x, red.y, green.x, green.y, blue.x, blue.y, white.x,
/// white.y` — 32 bytes total. The type name is `"chromaticities"`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chromaticities {
    pub red_x: f32,
    pub red_y: f32,
    pub green_x: f32,
    pub green_y: f32,
    pub blue_x: f32,
    pub blue_y: f32,
    pub white_x: f32,
    pub white_y: f32,
}

/// Decoded value of one EXR header attribute.
///
/// The variants typed here are the fixed-size attribute payloads whose
/// on-disk byte length is implied by the type name (4 for `int`, 8 for
/// `double` / `v2i` / `v2f`, 9 for `tiledesc`, 12 for `v3i` / `v3f`,
/// 16 for `box2i`, 32 for `chromaticities`, 36 for `m33f`, 64 for
/// `m44f`) plus the variable-length `String` (raw bytes, length carried
/// by the outer attribute size field — the same shape this crate's
/// multi-part writers already emit and `exrmetrics` round-trips, see
/// round-40 CHANGELOG entry) and the `Channels` payload. Any attribute
/// whose type name doesn't map to one of these variants is preserved
/// verbatim as `Other { type_name, data }` so the header round-trips
/// without losing metadata.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    Channels(Vec<Channel>),
    Compression(Compression),
    Box2i(Box2i),
    /// `box2f` — see [`Box2f`]. Four little-endian `f32`, 16 bytes.
    Box2f(Box2f),
    LineOrder(LineOrder),
    Float(f32),
    /// `int` — single little-endian `i32`, 4 bytes.
    Int(i32),
    /// `double` — single little-endian `f64`, 8 bytes.
    Double(f64),
    /// `string` — raw UTF-8 bytes; the outer attribute size field
    /// carries the length, so no NUL terminator is stored.
    String(String),
    V2f(f32, f32),
    /// `v2i` — two little-endian `i32`, 8 bytes.
    V2i(i32, i32),
    /// `v3i` — three little-endian `i32`, 12 bytes.
    V3i(i32, i32, i32),
    /// `v3f` — three little-endian `f32`, 12 bytes.
    V3f(f32, f32, f32),
    /// `m33f` — nine little-endian `f32` in row-major order, 36 bytes.
    M33f([f32; 9]),
    /// `m44f` — sixteen little-endian `f32` in row-major order, 64 bytes.
    M44f([f32; 16]),
    /// `m33d` — nine little-endian `f64` in row-major order, 72 bytes.
    /// The double-precision companion of [`AttributeValue::M33f`].
    M33d([f64; 9]),
    /// `m44d` — sixteen little-endian `f64` in row-major order, 128 bytes.
    /// The double-precision companion of [`AttributeValue::M44f`].
    M44d([f64; 16]),
    /// `chromaticities` — see [`Chromaticities`].
    Chromaticities(Chromaticities),
    /// `tiledesc` — tile-grid descriptor carried by tiled files in the
    /// `tiles` attribute. Fixed 9-byte payload: two little-endian `u32`
    /// (`x_size`, `y_size`) followed by a single packed mode byte (low
    /// nibble = level mode 0=ONE_LEVEL / 1=MIPMAP / 2=RIPMAP; high
    /// nibble = round mode 0=ROUND_DOWN / 1=ROUND_UP). See
    /// [`crate::tiled::TileDesc`] for the struct definition.
    TileDesc(crate::tiled::TileDesc),
    /// `v2d` — two little-endian `f64`, 16 bytes.
    V2d(f64, f64),
    /// `v3d` — three little-endian `f64`, 24 bytes.
    V3d(f64, f64, f64),
    /// `rational` — an `i32` numerator followed by a `u32` denominator,
    /// 8 bytes. Used by e.g. `framesPerSecond`.
    Rational(i32, u32),
    /// `timecode` — see [`Timecode`]. Two little-endian `u32`, 8 bytes.
    Timecode(Timecode),
    /// `keycode` — see [`Keycode`]. Seven little-endian `i32`, 28 bytes.
    Keycode(Keycode),
    /// `stringvector` — a sequence of length-prefixed strings. Each entry
    /// is a little-endian `i32` byte length followed by that many UTF-8
    /// bytes (no NUL terminator). The entry count is implied by the
    /// outer attribute size field.
    StringVector(Vec<String>),
    /// Anything we don't model as a typed enum yet — preserved verbatim.
    Other {
        type_name: String,
        data: Vec<u8>,
    },
}

/// `timecode` attribute payload.
///
/// On disk this is two consecutive little-endian `u32` words, 8 bytes
/// total. The type name is `"timecode"`.
///
/// * `time_and_flags` — the packed time + flag word (SMPTE 12M layout):
///   the four time components (hours, minutes, seconds, frames) are
///   stored as binary-coded-decimal nibble pairs, interleaved with the
///   drop-frame / colour-frame / field-phase / binary-group flag bits.
///   This crate stores the word verbatim so the encoding round-trips
///   bit-exactly regardless of which flag bits are set; the
///   [`Timecode::hours`] / [`Timecode::minutes`] / [`Timecode::seconds`]
///   / [`Timecode::frames`] accessors decode the BCD time nibbles that
///   the `exrheader` validator renders as `HH:MM:SS:FF`.
/// * `user_data` — the second 32-bit word, carried verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timecode {
    pub time_and_flags: u32,
    pub user_data: u32,
}

impl Timecode {
    /// Decode the BCD `frames` field (low two nibbles of the BCD time
    /// quartet — the `tens` nibble is masked to two bits per SMPTE 12M
    /// since frame counts never exceed 39).
    pub fn frames(&self) -> u8 {
        let b = (self.time_and_flags & 0xFF) as u8;
        ((b >> 4) & 0x3) * 10 + (b & 0xF)
    }
    /// Decode the BCD `seconds` field (tens nibble masked to three bits).
    pub fn seconds(&self) -> u8 {
        let b = ((self.time_and_flags >> 8) & 0xFF) as u8;
        ((b >> 4) & 0x7) * 10 + (b & 0xF)
    }
    /// Decode the BCD `minutes` field (tens nibble masked to three bits).
    pub fn minutes(&self) -> u8 {
        let b = ((self.time_and_flags >> 16) & 0xFF) as u8;
        ((b >> 4) & 0x7) * 10 + (b & 0xF)
    }
    /// Decode the BCD `hours` field (tens nibble masked to two bits).
    pub fn hours(&self) -> u8 {
        let b = ((self.time_and_flags >> 24) & 0xFF) as u8;
        ((b >> 4) & 0x3) * 10 + (b & 0xF)
    }
}

/// `keycode` attribute payload: SMPTE 268M motion-picture-film key code.
///
/// On disk this is seven consecutive little-endian `i32` words, 28 bytes
/// total, in the field order observed below. The type name is
/// `"keycode"`. The validator-enforced value ranges (which this crate
/// does not itself enforce on parse, to keep arbitrary headers
/// round-trippable) are noted per field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Keycode {
    /// Film manufacturer code (0..=99).
    pub film_mfc_code: i32,
    /// Film type code (0..=99).
    pub film_type: i32,
    /// Prefix identifying the film roll (0..=999999).
    pub prefix: i32,
    /// Count of film perforations within the roll (0..=9999).
    pub count: i32,
    /// Perforation offset within the frame (0..=119).
    pub perf_offset: i32,
    /// Number of perforations per frame (1..=15).
    pub perfs_per_frame: i32,
    /// Number of perforations per count (20..=120).
    pub perfs_per_count: i32,
}

/// One header attribute (name + typed value).
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    pub name: String,
    pub value: AttributeValue,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box2i_extent_does_not_overflow_on_extreme_bounds() {
        // A normal box reports its exact extent.
        let b = Box2i {
            x_min: 0,
            y_min: 0,
            x_max: 7,
            y_max: 3,
        };
        assert_eq!(b.width(), 8);
        assert_eq!(b.height(), 4);

        // i32::MIN .. i32::MAX would overflow an i32 subtraction; the
        // widened computation clamps into u32 instead of panicking.
        let huge = Box2i {
            x_min: i32::MIN,
            y_min: i32::MIN,
            x_max: i32::MAX,
            y_max: i32::MAX,
        };
        assert_eq!(huge.width(), u32::MAX);
        assert_eq!(huge.height(), u32::MAX);

        // An inverted box (x_max < x_min) clamps to zero rather than
        // wrapping to a huge value.
        let inverted = Box2i {
            x_min: 10,
            y_min: 10,
            x_max: 0,
            y_max: 0,
        };
        assert_eq!(inverted.width(), 0);
        assert_eq!(inverted.height(), 0);
    }
}
