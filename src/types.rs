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
    pub fn width(&self) -> u32 {
        (self.x_max - self.x_min + 1) as u32
    }
    pub fn height(&self) -> u32 {
        (self.y_max - self.y_min + 1) as u32
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
/// `double` / `v2i` / `v2f`, 12 for `v3i` / `v3f`, 16 for `box2i`, 32
/// for `chromaticities`, 36 for `m33f`, 64 for `m44f`) plus the
/// variable-length `String` (raw bytes, length carried by the outer
/// attribute size field — the same shape this crate's multi-part
/// writers already emit and `exrmetrics` round-trips, see round-40
/// CHANGELOG entry) and the `Channels` payload. Any attribute whose
/// type name doesn't map to one of these variants is preserved verbatim
/// as `Other { type_name, data }` so the header round-trips without
/// losing metadata.
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
    /// `chromaticities` — see [`Chromaticities`].
    Chromaticities(Chromaticities),
    /// Anything we don't model as a typed enum yet — preserved verbatim.
    Other {
        type_name: String,
        data: Vec<u8>,
    },
}

/// One header attribute (name + typed value).
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    pub name: String,
    pub value: AttributeValue,
}
