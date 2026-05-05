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

/// Decoded value of one EXR header attribute.
///
/// Only the round-1-required attribute types are explicitly parsed;
/// everything else is preserved as `Other { type_name, data }` so the
/// header can round-trip without losing decoder-irrelevant metadata.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValue {
    Channels(Vec<Channel>),
    Compression(Compression),
    Box2i(Box2i),
    LineOrder(LineOrder),
    Float(f32),
    V2f(f32, f32),
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
