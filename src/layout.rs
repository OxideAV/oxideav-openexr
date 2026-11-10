//! File layout: version flags, the header list, per-part chunk counts,
//! offset tables and the part-number prefix of chunks — and the byte
//! level repack between a single-part file and a part of a multi-part
//! file.
//!
//! Everything here follows the multi-part observer note
//! (`docs/image/openexr/multipart.md`). The module never looks inside a
//! chunk payload: it reads the headers, sizes the offset tables from
//! the part geometry (§4 of the note), locates and delimits the chunks
//! (§5) and checks the rules a conforming reader enforces.
//!
//! * [`FileLayout::read`] parses the header region and reports, as
//!   [`ExrError::Layout`], everything a conforming reader refuses:
//!   bad version flags, a header list without its empty header, a part
//!   without `name` / `type`, an unknown `type`, a tiled part without
//!   `tiles`, and a part whose `displayWindow`, `pixelAspectRatio` or
//!   `chromaticities` does not agree with part 0.
//! * Irregularities a conforming reader tolerates — a missing or wrong
//!   `chunkCount`, zeroed offset tables, duplicate part names, a name
//!   above 31 bytes without the long-name flag, a deep flag that does
//!   not match the parts — are collected in
//!   [`FileLayout::irregularities`]; [`check_layout`] in strict mode
//!   turns the first of them into the error.
//! * [`FileLayout::chunks`] locates every chunk through the offset
//!   tables (rebuilding zeroed tables from the chunks themselves) and
//!   validates each chunk's part number.
//! * [`split_parts`] / [`combine_parts`] repack parts without touching
//!   a payload byte (§6 of the note).

use core::fmt;
use core::ops::Range;
use std::borrow::Cow;

use crate::decoder::{mipmap_level_count, mipmap_level_dim};
use crate::error::{ExrError, Result};
use crate::header::{
    FLAG_DEEP, FLAG_LONG_NAMES, FLAG_MULTIPART, FLAG_SINGLE_TILE, HeaderList, SHORT_NAME_MAX,
    SpannedAttribute, VersionField, encode_attribute_value, parse_header_list,
};
use crate::tiled::tiledesc_from_attribute;
use crate::types::{Attribute, AttributeValue, Box2i, Compression, EXR_MAGIC};

// ---------------------------------------------------------------------------
// Violations
// ---------------------------------------------------------------------------

/// One broken rule of the file layout.
///
/// [`LayoutViolation::is_tolerated`] separates the irregularities a
/// conforming reader reads through (reported by
/// [`FileLayout::irregularities`], errors only in strict mode) from the
/// ones every reader refuses.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LayoutViolation {
    /// The low byte of the version field is not 2.
    FormatVersion { found: u8 },
    /// A flag bit other than `0x200`, `0x400`, `0x800`, `0x1000` is set.
    UnknownVersionFlags { flags: u32 },
    /// The single-part-tiled flag (`0x200`) together with the deep
    /// (`0x800`) or multi-part (`0x1000`) flag.
    ConflictingVersionFlags { flags: u32 },
    /// The single-part-tiled flag on a file without a `tiles` attribute.
    TiledFlagWithoutTiles,
    /// The header list of a multi-part file does not end in an empty
    /// header (a lone `00`): what follows header `parts_read - 1` is
    /// neither that nor a readable header.
    HeaderListNotTerminated { parts_read: u32, detail: String },
    /// A multi-part file without any header.
    NoParts,
    /// A part of a multi-part file has no `name` attribute.
    MissingName { part: u32 },
    /// The `name` attribute is not of type `string`.
    NameNotString { part: u32 },
    /// A part of a multi-part file, or a deep file, has no `type`.
    MissingType { part: u32 },
    /// `type` is not one of `scanlineimage`, `tiledimage`,
    /// `deepscanline`, `deeptile` (case-sensitive).
    UnknownPartType { part: u32, found: String },
    /// A `tiledimage` / `deeptile` part without a usable `tiles`
    /// attribute (absent, zero tile size or an unknown level mode).
    MissingTiles { part: u32 },
    /// A scanline part of a multi-part file that carries `tiles`.
    UnexpectedTiles { part: u32 },
    /// The `version` attribute of a deep part is not 1.
    DeepDataVersion { part: u32, found: i32 },
    /// The part cannot be laid out: a required attribute is missing or
    /// its data window is empty.
    PartGeometry { part: u32, detail: String },
    /// `attribute` of `part` does not agree with part 0
    /// (`displayWindow`, `pixelAspectRatio` with absent = 1.0, or
    /// `chromaticities`, which may be absent from a later part but not
    /// present there without being identical in part 0).
    SharedAttributeMismatch { part: u32, attribute: String },
    /// The offset tables (sized from the part geometry) do not fit in
    /// the file.
    OffsetTableTruncated {
        part: u32,
        needed: u64,
        available: u64,
    },
    /// An offset-table entry points outside the chunk region, or the
    /// chunk it points at runs past the end of the file.
    ChunkOutOfBounds { part: u32, chunk: u32, offset: u64 },
    /// The chunk an offset-table entry of `part` leads to carries the
    /// part number `found`.
    ChunkPartNumber { part: u32, chunk: u32, found: i32 },
    /// The part declares more sample data than its chunks can hold at
    /// the best ratio its compression reaches (`declared` uncompressed
    /// bytes against `stored` chunk bytes): the header is not backed by
    /// the file.
    PixelDataNotBacked {
        part: u32,
        declared: u64,
        stored: u64,
    },
    /// The offset tables are unusable and the chunks in the file do not
    /// make up `part`'s table (`found` of `expected` located).
    ChunksMissing {
        part: u32,
        found: u32,
        expected: u32,
    },

    // ---- tolerated by conforming readers --------------------------------
    /// No `chunkCount` where a writer always stores one (every part of
    /// a multi-part file, a deep file).
    ChunkCountMissing { part: u32, expected: u64 },
    /// `chunkCount` is not the number of offset-table entries the part
    /// geometry gives (or is not an `int`).
    ChunkCountMismatch {
        part: u32,
        declared: i64,
        expected: u64,
    },
    /// The part's offset table has zero entries; the chunks were found
    /// by walking the file.
    OffsetTableZero { part: u32 },
    /// Two parts share a name (`first` is the earlier one).
    DuplicatePartName { part: u32, first: u32, name: String },
    /// An attribute or channel name above 31 bytes in a file whose
    /// long-name flag (`0x400`) is clear.
    LongNameFlagClear { part: u32, name: String },
    /// The deep flag (`0x800`) on a multi-part file without a deep part.
    DeepFlagWithoutDeepPart,
    /// A deep part in a multi-part file whose deep flag is clear.
    DeepPartWithoutDeepFlag { part: u32 },
}

impl LayoutViolation {
    /// `true` for the irregularities a conforming reader reads through;
    /// they fail only [`check_layout`] in strict mode.
    pub fn is_tolerated(&self) -> bool {
        matches!(
            self,
            Self::ChunkCountMissing { .. }
                | Self::ChunkCountMismatch { .. }
                | Self::OffsetTableZero { .. }
                | Self::DuplicatePartName { .. }
                | Self::LongNameFlagClear { .. }
                | Self::DeepFlagWithoutDeepPart
                | Self::DeepPartWithoutDeepFlag { .. }
        )
    }

    /// The part the violation is about, when it concerns one.
    pub fn part(&self) -> Option<u32> {
        match self {
            Self::MissingName { part }
            | Self::NameNotString { part }
            | Self::MissingType { part }
            | Self::UnknownPartType { part, .. }
            | Self::MissingTiles { part }
            | Self::UnexpectedTiles { part }
            | Self::DeepDataVersion { part, .. }
            | Self::PartGeometry { part, .. }
            | Self::SharedAttributeMismatch { part, .. }
            | Self::OffsetTableTruncated { part, .. }
            | Self::ChunkOutOfBounds { part, .. }
            | Self::ChunkPartNumber { part, .. }
            | Self::ChunksMissing { part, .. }
            | Self::PixelDataNotBacked { part, .. }
            | Self::ChunkCountMissing { part, .. }
            | Self::ChunkCountMismatch { part, .. }
            | Self::OffsetTableZero { part }
            | Self::DuplicatePartName { part, .. }
            | Self::LongNameFlagClear { part, .. }
            | Self::DeepPartWithoutDeepFlag { part } => Some(*part),
            _ => None,
        }
    }
}

impl fmt::Display for LayoutViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FormatVersion { found } => {
                write!(f, "format version {found} (only version 2 is defined)")
            }
            Self::UnknownVersionFlags { flags } => {
                write!(f, "version flags 0x{flags:x} set an unassigned bit")
            }
            Self::ConflictingVersionFlags { flags } => write!(
                f,
                "version flags 0x{flags:x} combine the single-part-tiled flag with the deep or \
                 multi-part flag"
            ),
            Self::TiledFlagWithoutTiles => write!(
                f,
                "the single-part-tiled flag is set but the header has no 'tiles' attribute"
            ),
            Self::HeaderListNotTerminated { parts_read, detail } => write!(
                f,
                "the header list is not ended by an empty header after {parts_read} part(s): \
                 {detail}"
            ),
            Self::NoParts => write!(f, "the multi-part file has no part"),
            Self::MissingName { part } => {
                write!(f, "part {part} has no 'name' attribute")
            }
            Self::NameNotString { part } => {
                write!(f, "part {part}: 'name' is not a string attribute")
            }
            Self::MissingType { part } => {
                write!(f, "part {part} has no 'type' attribute")
            }
            Self::UnknownPartType { part, found } => write!(
                f,
                "part {part} has type {found:?} (expected scanlineimage, tiledimage, \
                 deepscanline or deeptile)"
            ),
            Self::MissingTiles { part } => {
                write!(f, "tiled part {part} has no usable 'tiles' attribute")
            }
            Self::UnexpectedTiles { part } => {
                write!(f, "scanline part {part} carries a 'tiles' attribute")
            }
            Self::DeepDataVersion { part, found } => {
                write!(f, "deep part {part} has data version {found} (expected 1)")
            }
            Self::PartGeometry { part, detail } => write!(f, "part {part}: {detail}"),
            Self::SharedAttributeMismatch { part, attribute } => write!(
                f,
                "part {part}: '{attribute}' does not agree with part 0 (the attribute is shared \
                 by all parts)"
            ),
            Self::OffsetTableTruncated {
                part,
                needed,
                available,
            } => write!(
                f,
                "the offset table of part {part} ends at byte {needed} but the file has \
                 {available}"
            ),
            Self::ChunkOutOfBounds {
                part,
                chunk,
                offset,
            } => write!(
                f,
                "part {part} chunk {chunk}: offset {offset} is outside the chunk region or the \
                 chunk runs past the end of the file"
            ),
            Self::ChunkPartNumber { part, chunk, found } => write!(
                f,
                "part {part} chunk {chunk}: the chunk carries part number {found}"
            ),
            Self::PixelDataNotBacked {
                part,
                declared,
                stored,
            } => write!(
                f,
                "part {part} declares {declared} bytes of sample data but its chunks hold \
                 {stored} bytes"
            ),
            Self::ChunksMissing {
                part,
                found,
                expected,
            } => write!(
                f,
                "part {part}: {found} of {expected} chunks found while rebuilding its offset \
                 table"
            ),
            Self::ChunkCountMissing { part, expected } => write!(
                f,
                "part {part} has no 'chunkCount' attribute (its offset table has {expected} \
                 entries)"
            ),
            Self::ChunkCountMismatch {
                part,
                declared,
                expected,
            } => write!(
                f,
                "part {part}: chunkCount {declared} but its offset table has {expected} entries"
            ),
            Self::OffsetTableZero { part } => {
                write!(f, "part {part}: the offset table is not filled in")
            }
            Self::DuplicatePartName { part, first, name } => {
                write!(f, "duplicate part name {name:?} (parts {first} and {part})")
            }
            Self::LongNameFlagClear { part, name } => write!(
                f,
                "part {part}: name {name:?} is longer than 31 bytes but the long-name flag is \
                 clear"
            ),
            Self::DeepFlagWithoutDeepPart => {
                write!(f, "the deep flag is set but no part is deep")
            }
            Self::DeepPartWithoutDeepFlag { part } => {
                write!(f, "part {part} is deep but the deep flag is clear")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Part geometry
// ---------------------------------------------------------------------------

/// The four part types; each has its own chunk layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartKind {
    /// `scanlineimage`: `y · size · data`.
    Scanline,
    /// `tiledimage`: `tile x · tile y · level x · level y · size · data`.
    Tiled,
    /// `deepscanline`: `y · three u64 sizes · table · data`.
    DeepScanline,
    /// `deeptile`: four tile coordinates, three u64 sizes, table, data.
    DeepTiled,
}

impl PartKind {
    /// The `type` attribute value of this kind.
    pub fn type_name(self) -> &'static str {
        match self {
            Self::Scanline => "scanlineimage",
            Self::Tiled => "tiledimage",
            Self::DeepScanline => "deepscanline",
            Self::DeepTiled => "deeptile",
        }
    }

    /// The kind a `type` attribute value names (case-sensitive).
    pub fn from_type_name(name: &str) -> Option<Self> {
        Some(match name {
            "scanlineimage" => Self::Scanline,
            "tiledimage" => Self::Tiled,
            "deepscanline" => Self::DeepScanline,
            "deeptile" => Self::DeepTiled,
            _ => return None,
        })
    }

    /// `true` for the two deep kinds.
    pub fn is_deep(self) -> bool {
        matches!(self, Self::DeepScanline | Self::DeepTiled)
    }

    /// `true` for the two tiled kinds.
    pub fn is_tiled(self) -> bool {
        matches!(self, Self::Tiled | Self::DeepTiled)
    }

    /// Bytes of chunk framing ahead of the data (coordinates and size
    /// fields; the part-number prefix excluded).
    fn frame_len(self) -> usize {
        match self {
            Self::Scanline => 4 + 4,
            Self::Tiled => 16 + 4,
            Self::DeepScanline => 4 + 24,
            Self::DeepTiled => 16 + 24,
        }
    }

    /// Length of the chunk at `pos` (`prefix` bytes of part number
    /// included), or `None` when it does not fit in `bytes`.
    fn chunk_len(self, bytes: &[u8], pos: usize, prefix: usize) -> Option<usize> {
        let frame = prefix + self.frame_len();
        let frame_end = pos.checked_add(frame).filter(|&e| e <= bytes.len())?;
        let data = if self.is_deep() {
            let at = frame_end - 24;
            let table = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
            let samples = u64::from_le_bytes(bytes[at + 8..at + 16].try_into().unwrap());
            usize::try_from(table)
                .ok()?
                .checked_add(usize::try_from(samples).ok()?)?
        } else {
            let at = frame_end - 4;
            let size = i32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
            usize::try_from(size).ok()?
        };
        let total = frame.checked_add(data)?;
        pos.checked_add(total).filter(|&e| e <= bytes.len())?;
        Some(total)
    }
}

/// One resolution level of a tiled part: its tile grid and the index of
/// its first chunk in the part's offset table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LevelGrid {
    level_x: u32,
    level_y: u32,
    tiles_x: u64,
    tiles_y: u64,
    first: u64,
}

/// What sizes and orders a part's offset table.
#[derive(Debug, Clone, PartialEq, Eq)]
enum TableShape {
    Lines {
        y_min: i32,
        lines_per_chunk: u32,
    },
    Tiles {
        /// 0 one level, 1 mip-map, 2 rip-map.
        level_mode: u8,
        levels: Vec<LevelGrid>,
    },
}

impl TableShape {
    /// The table index of the chunk whose framing starts at `frame`.
    fn index_of(&self, kind: PartKind, frame: &[u8]) -> Option<u64> {
        let int = |i: usize| i32::from_le_bytes(frame[i * 4..i * 4 + 4].try_into().unwrap());
        match self {
            Self::Lines {
                y_min,
                lines_per_chunk,
            } => {
                debug_assert!(!kind.is_tiled());
                let rel = int(0) as i64 - *y_min as i64;
                if rel < 0 || rel % *lines_per_chunk as i64 != 0 {
                    return None;
                }
                Some((rel / *lines_per_chunk as i64) as u64)
            }
            Self::Tiles { level_mode, levels } => {
                let (tx, ty, lx, ly) = (int(0), int(1), int(2), int(3));
                if tx < 0 || ty < 0 || lx < 0 || ly < 0 {
                    return None;
                }
                if *level_mode == 1 && lx != ly {
                    return None;
                }
                let level = levels
                    .iter()
                    .find(|l| l.level_x == lx as u32 && l.level_y == ly as u32)?;
                if tx as u64 >= level.tiles_x || ty as u64 >= level.tiles_y {
                    return None;
                }
                Some(level.first + ty as u64 * level.tiles_x + tx as u64)
            }
        }
    }
}

fn find<'a>(attrs: &'a [SpannedAttribute], name: &str) -> Option<&'a SpannedAttribute> {
    attrs.iter().find(|a| a.attribute.name == name)
}

/// A `string` attribute's text; `Err(())` when the attribute exists
/// with another type.
fn string_value(
    attrs: &[SpannedAttribute],
    name: &str,
) -> core::result::Result<Option<String>, ()> {
    match find(attrs, name).map(|a| &a.attribute.value) {
        None => Ok(None),
        Some(AttributeValue::String(s)) => Ok(Some(s.clone())),
        Some(AttributeValue::Other { type_name, data }) if type_name == "string" => {
            Ok(Some(String::from_utf8_lossy(data).into_owned()))
        }
        Some(_) => Err(()),
    }
}

/// Number of offset-table entries of a part (§4 of the note) and the
/// shape that orders them.
fn table_shape(part: u32, kind: PartKind, attrs: &[SpannedAttribute]) -> Result<(u64, TableShape)> {
    let geometry = |detail: String| ExrError::from(LayoutViolation::PartGeometry { part, detail });
    let window = match find(attrs, "dataWindow").map(|a| &a.attribute.value) {
        Some(AttributeValue::Box2i(b)) => *b,
        _ => return Err(geometry("missing required attribute 'dataWindow'".into())),
    };
    let compression = match find(attrs, "compression").map(|a| &a.attribute.value) {
        Some(AttributeValue::Compression(c)) => *c,
        _ => return Err(geometry("missing required attribute 'compression'".into())),
    };
    let (width, height) = (window.width(), window.height());
    if width == 0 || height == 0 {
        return Err(geometry(format!(
            "dataWindow {width}x{height} must not be empty"
        )));
    }
    if !kind.is_tiled() {
        let lines = lines_per_chunk(kind, compression);
        return Ok((
            height.div_ceil(lines) as u64,
            TableShape::Lines {
                y_min: window.y_min,
                lines_per_chunk: lines,
            },
        ));
    }
    let tiles = find(attrs, "tiles")
        .and_then(|a| tiledesc_from_attribute(&a.attribute.value).ok())
        .filter(|t| t.x_size != 0 && t.y_size != 0 && t.level_mode <= 2)
        .ok_or(LayoutViolation::MissingTiles { part })?;
    let round_up = tiles.round_mode != 0;
    let mut levels = Vec::new();
    let mut total: u64 = 0;
    let mut push = |level_x: u32, level_y: u32, w: u32, h: u32| -> Result<()> {
        let tiles_x = w.div_ceil(tiles.x_size) as u64;
        let tiles_y = h.div_ceil(tiles.y_size) as u64;
        levels.push(LevelGrid {
            level_x,
            level_y,
            tiles_x,
            tiles_y,
            first: total,
        });
        total = tiles_x
            .checked_mul(tiles_y)
            .and_then(|n| total.checked_add(n))
            .ok_or_else(|| geometry("tile count overflows".into()))?;
        Ok(())
    };
    match tiles.level_mode {
        0 => push(0, 0, width, height)?,
        1 => {
            for l in 0..mipmap_level_count(width.max(height), round_up) {
                push(
                    l,
                    l,
                    mipmap_level_dim(width, l, round_up),
                    mipmap_level_dim(height, l, round_up),
                )?;
            }
        }
        _ => {
            let nx = mipmap_level_count(width, round_up);
            let ny = mipmap_level_count(height, round_up);
            for ly in 0..ny {
                for lx in 0..nx {
                    push(
                        lx,
                        ly,
                        mipmap_level_dim(width, lx, round_up),
                        mipmap_level_dim(height, ly, round_up),
                    )?;
                }
            }
        }
    }
    Ok((
        total,
        TableShape::Tiles {
            level_mode: tiles.level_mode,
            levels,
        },
    ))
}

/// Uncompressed size of the data a reader sizes from the header alone:
/// every channel's samples over every level of a flat part, the
/// per-pixel sample-count entries of a deep part. Saturating.
fn declared_bytes(kind: PartKind, attrs: &[SpannedAttribute]) -> u64 {
    let Some(AttributeValue::Box2i(window)) = find(attrs, "dataWindow").map(|a| &a.attribute.value)
    else {
        return 0;
    };
    let (width, height) = (window.width() as u64, window.height() as u64);
    let pixels = if kind.is_tiled() {
        let Some(tiles) =
            find(attrs, "tiles").and_then(|a| tiledesc_from_attribute(&a.attribute.value).ok())
        else {
            return 0;
        };
        let round_up = tiles.round_mode != 0;
        let (w, h) = (window.width(), window.height());
        let dim = |n: u32, l: u32| mipmap_level_dim(n, l, round_up) as u64;
        match tiles.level_mode {
            0 => width.saturating_mul(height),
            1 => (0..mipmap_level_count(w.max(h), round_up))
                .map(|l| dim(w, l).saturating_mul(dim(h, l)))
                .fold(0u64, u64::saturating_add),
            _ => {
                let xs: u64 = (0..mipmap_level_count(w, round_up))
                    .map(|l| dim(w, l))
                    .sum();
                let ys: u64 = (0..mipmap_level_count(h, round_up))
                    .map(|l| dim(h, l))
                    .sum();
                xs.saturating_mul(ys)
            }
        }
    } else {
        width.saturating_mul(height)
    };
    if kind.is_deep() {
        return pixels.saturating_mul(4);
    }
    let Some(AttributeValue::Channels(channels)) =
        find(attrs, "channels").map(|a| &a.attribute.value)
    else {
        return 0;
    };
    channels
        .iter()
        .map(|c| {
            let sub = (c.x_sampling.max(1) as u64).saturating_mul(c.y_sampling.max(1) as u64);
            (pixels / sub).saturating_mul(c.pixel_type.bytes_per_sample() as u64)
        })
        .fold(0u64, u64::saturating_add)
}

/// Most uncompressed bytes one stored byte can stand for under a
/// compression scheme — a ceiling, not a typical ratio: 1 uncompressed;
/// a run of 128 per two bytes for RLE; the deflate maximum for the zlib
/// schemes (PXR24 additionally drops a quarter of each FLOAT); 16
/// samples per three bytes for B44A; a 255-sample run per nine bits of
/// PIZ entropy code; and for DWA a whole 8x8 block per two-byte code,
/// deflated again.
fn expansion_ceiling(compression: Compression) -> u64 {
    match compression {
        Compression::None => 1,
        Compression::Rle => 128,
        Compression::Zips | Compression::Zip | Compression::Piz => 1040,
        Compression::Pxr24 => 1400,
        Compression::B44 | Compression::B44a => 16,
        _ => 1 << 17,
    }
}

/// Scanlines per chunk of a scanline part (§4.2 of the note). Deep
/// scanline chunks hold one scanline.
fn lines_per_chunk(kind: PartKind, compression: Compression) -> u32 {
    if kind.is_deep() {
        1
    } else {
        compression.scanlines_per_block()
    }
}

/// Number of chunks — offset-table entries — of a part described by
/// its attribute list: `ceil(height / lines per chunk)` for a scanline
/// part, the tile count over every level for a tiled one. `kind`
/// selects the layout (in a single-part flat file the `type` attribute
/// is not authoritative; the version flags are).
pub fn chunk_count(kind: PartKind, attributes: &[Attribute]) -> Result<u64> {
    let spanned: Vec<SpannedAttribute> = attributes
        .iter()
        .map(|a| SpannedAttribute {
            attribute: a.clone(),
            span: 0..0,
        })
        .collect();
    table_shape(0, kind, &spanned).map(|(n, _)| n)
}

// ---------------------------------------------------------------------------
// FileLayout
// ---------------------------------------------------------------------------

/// Layout facts of one part.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PartLayout {
    /// Position in the header list = the part number chunks carry.
    pub index: u32,
    /// The `name` attribute (always present in a multi-part file).
    pub name: Option<String>,
    /// The chunk layout of the part.
    pub kind: PartKind,
    /// Offset-table entries, computed from the part geometry.
    pub chunk_count: u64,
    /// The `chunkCount` attribute as stored, when it is an `int`.
    pub declared_chunk_count: Option<i32>,
    /// File offset of the part's offset table.
    pub table_offset: usize,
    /// The part's attributes, in file order.
    pub attributes: Vec<Attribute>,
    shape: TableShape,
    spans: Vec<Range<usize>>,
    /// Uncompressed bytes the header declares, and the most one stored
    /// byte can expand to.
    declared_bytes: u64,
    expansion: u64,
}

/// The chunks of a file, located and delimited.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ChunkMap {
    /// Per part, in offset-table order: the byte range of each chunk
    /// (part-number prefix included in a multi-part file).
    pub parts: Vec<Vec<Range<usize>>>,
    /// `true` when zeroed offset tables were rebuilt from the chunks.
    pub rebuilt: bool,
    /// First part whose table held a zero entry.
    zero_part: u32,
}

/// The layout of a file: version flags, parts, table positions.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FileLayout {
    /// The version field.
    pub version: VersionField,
    /// One entry per header, in file order.
    pub parts: Vec<PartLayout>,
    /// Offset of the first offset table.
    pub tables_start: usize,
    /// First byte after the last offset table.
    pub tables_end: usize,
    /// Tolerated irregularities found in the header region.
    pub irregularities: Vec<LayoutViolation>,
}

/// The bytes that decide whether two parts agree on an attribute.
fn encoded(a: &SpannedAttribute) -> (String, Vec<u8>) {
    encode_attribute_value(&a.attribute.value)
}

impl FileLayout {
    /// Read the header region of `bytes` and lay the file out.
    ///
    /// Fails with [`ExrError::Layout`] on everything a conforming
    /// reader refuses at this stage and collects what it tolerates in
    /// [`FileLayout::irregularities`]. Only the headers are read; the
    /// offset tables are sized (and must fit in the file) but not
    /// followed — [`FileLayout::chunks`] does that.
    pub fn read(bytes: &[u8]) -> Result<Self> {
        Self::from_list(parse_header_list(bytes)?, Some(bytes.len()))
    }

    /// [`FileLayout::read`] for a buffer that may end right after the
    /// header region: the offset tables are sized but not required to
    /// fit ([`FileLayout::chunks`] must not be called on the result).
    pub fn read_headers(bytes: &[u8]) -> Result<Self> {
        Self::from_list(parse_header_list(bytes)?, None)
    }

    fn from_list(list: HeaderList, file_len: Option<usize>) -> Result<Self> {
        let HeaderList {
            version,
            headers,
            tables_start,
        } = list;
        let multipart = version.multipart;
        if headers.is_empty() {
            return Err(LayoutViolation::NoParts.into());
        }
        let mut irregularities = Vec::new();
        let mut parts: Vec<PartLayout> = Vec::with_capacity(headers.len());
        let mut table_offset = tables_start;
        let mut any_deep = false;
        for (i, attrs) in headers.iter().enumerate() {
            let part = i as u32;
            let name = string_value(attrs, "name")
                .map_err(|()| LayoutViolation::NameNotString { part })?;
            if multipart && name.is_none() {
                return Err(LayoutViolation::MissingName { part }.into());
            }
            let type_name = string_value(attrs, "type").unwrap_or(None);
            let has_tiles = find(attrs, "tiles").is_some();
            let kind = if multipart || version.non_image {
                let t = type_name.ok_or(LayoutViolation::MissingType { part })?;
                PartKind::from_type_name(&t)
                    .ok_or(LayoutViolation::UnknownPartType { part, found: t })?
            } else if version.single_tile {
                if !has_tiles {
                    return Err(LayoutViolation::TiledFlagWithoutTiles.into());
                }
                PartKind::Tiled
            } else {
                PartKind::Scanline
            };
            if multipart && !kind.is_tiled() && has_tiles {
                return Err(LayoutViolation::UnexpectedTiles { part }.into());
            }
            if kind.is_deep() {
                any_deep = true;
                if let Some(AttributeValue::Int(v)) =
                    find(attrs, "version").map(|a| &a.attribute.value)
                    && *v != 1
                {
                    return Err(LayoutViolation::DeepDataVersion { part, found: *v }.into());
                }
            }
            let (count, shape) = table_shape(part, kind, attrs)?;
            // The table must fit in the file before anything is sized
            // from the count.
            let needed = (table_offset as u64).saturating_add(count.saturating_mul(8));
            let end = match file_len {
                Some(len) if needed > len as u64 => {
                    return Err(LayoutViolation::OffsetTableTruncated {
                        part,
                        needed,
                        available: len as u64,
                    }
                    .into());
                }
                _ => usize::try_from(needed).unwrap_or(usize::MAX),
            };

            // chunkCount: written in every part of a multi-part file and
            // in deep files; never relied on by a conforming reader.
            let declared = match find(attrs, "chunkCount").map(|a| &a.attribute.value) {
                Some(AttributeValue::Int(v)) => Some(Some(*v)),
                Some(AttributeValue::Other { type_name, data })
                    if type_name == "int" && data.len() == 4 =>
                {
                    Some(Some(i32::from_le_bytes(data[..4].try_into().unwrap())))
                }
                Some(_) => Some(None),
                None => None,
            };
            match declared {
                None if multipart || kind.is_deep() => {
                    irregularities.push(LayoutViolation::ChunkCountMissing {
                        part,
                        expected: count,
                    });
                }
                Some(d) if d.map(|v| v as i64) != Some(count as i64) => {
                    irregularities.push(LayoutViolation::ChunkCountMismatch {
                        part,
                        declared: d.map_or(-1, |v| v as i64),
                        expected: count,
                    });
                }
                _ => {}
            }

            if !version.long_names {
                let long_attr = attrs
                    .iter()
                    .map(|a| a.attribute.name.as_str())
                    .find(|n| n.len() > SHORT_NAME_MAX);
                let long_channel = attrs.iter().find_map(|a| match &a.attribute.value {
                    AttributeValue::Channels(chs) => chs
                        .iter()
                        .map(|c| c.name.as_str())
                        .find(|n| n.len() > SHORT_NAME_MAX),
                    _ => None,
                });
                if let Some(n) = long_attr.or(long_channel) {
                    irregularities.push(LayoutViolation::LongNameFlagClear {
                        part,
                        name: n.to_string(),
                    });
                }
            }
            if let Some(n) = &name
                && let Some(first) = parts.iter().find(|p| p.name.as_ref() == Some(n))
            {
                irregularities.push(LayoutViolation::DuplicatePartName {
                    part,
                    first: first.index,
                    name: n.clone(),
                });
            }
            if multipart && kind.is_deep() && !version.non_image {
                irregularities.push(LayoutViolation::DeepPartWithoutDeepFlag { part });
            }

            // Shared attributes: every part must agree with part 0.
            if i > 0 {
                let first = &headers[0];
                let mismatch = |attribute: &str| {
                    ExrError::from(LayoutViolation::SharedAttributeMismatch {
                        part,
                        attribute: attribute.to_string(),
                    })
                };
                if find(attrs, "displayWindow").map(encoded)
                    != find(first, "displayWindow").map(encoded)
                {
                    return Err(mismatch("displayWindow"));
                }
                let unit = || ("float".to_string(), 1.0f32.to_le_bytes().to_vec());
                if find(attrs, "pixelAspectRatio").map_or_else(unit, encoded)
                    != find(first, "pixelAspectRatio").map_or_else(unit, encoded)
                {
                    return Err(mismatch("pixelAspectRatio"));
                }
                if let Some(c) = find(attrs, "chromaticities")
                    && find(first, "chromaticities").map(encoded) != Some(encoded(c))
                {
                    return Err(mismatch("chromaticities"));
                }
            }

            parts.push(PartLayout {
                index: part,
                name,
                kind,
                chunk_count: count,
                declared_chunk_count: declared.flatten(),
                table_offset,
                attributes: attrs.iter().map(|a| a.attribute.clone()).collect(),
                shape,
                spans: attrs.iter().map(|a| a.span.clone()).collect(),
                declared_bytes: declared_bytes(kind, attrs),
                expansion: match find(attrs, "compression").map(|a| &a.attribute.value) {
                    Some(AttributeValue::Compression(c)) => expansion_ceiling(*c),
                    _ => 1,
                },
            });
            table_offset = end;
        }
        if multipart && version.non_image && !any_deep {
            irregularities.push(LayoutViolation::DeepFlagWithoutDeepPart);
        }
        Ok(Self {
            version,
            parts,
            tables_start,
            tables_end: table_offset,
            irregularities,
        })
    }

    /// `true` for a multi-part file (flag `0x1000`).
    pub fn is_multipart(&self) -> bool {
        self.version.multipart
    }

    /// Bytes of part number in front of each chunk: 4 in a multi-part
    /// file, 0 otherwise.
    fn prefix(&self) -> usize {
        if self.version.multipart { 4 } else { 0 }
    }

    /// Locate every chunk of `bytes` (the file this layout was read
    /// from) through the offset tables.
    ///
    /// Every entry must point into the chunk region at a chunk that
    /// fits in the file and, in a multi-part file, carries the part
    /// number of the table that led to it
    /// ([`LayoutViolation::ChunkOutOfBounds`],
    /// [`LayoutViolation::ChunkPartNumber`]). When any table holds a
    /// zero entry the tables are rebuilt by walking the chunk region,
    /// placing each chunk by its part number and coordinates;
    /// [`ChunkMap::rebuilt`] reports it.
    ///
    /// Last, each part's declared sample data must be something its
    /// chunks could hold at the best ratio of its compression scheme
    /// ([`LayoutViolation::PixelDataNotBacked`]), so a reader can size
    /// its planes from the header of a file that passed.
    pub fn chunks(&self, bytes: &[u8]) -> Result<ChunkMap> {
        let map = self.locate_chunks(bytes)?;
        for (p, ranges) in self.parts.iter().zip(&map.parts) {
            // Data bytes only (no part number, no chunk framing), so the
            // verdict does not change when a part is repacked.
            let framing = self.prefix() + p.kind.frame_len();
            let stored: u64 = ranges
                .iter()
                .map(|r| r.len().saturating_sub(framing) as u64)
                .sum();
            if p.declared_bytes > stored.saturating_mul(p.expansion) {
                return Err(LayoutViolation::PixelDataNotBacked {
                    part: p.index,
                    declared: p.declared_bytes,
                    stored,
                }
                .into());
            }
        }
        Ok(map)
    }

    fn locate_chunks(&self, bytes: &[u8]) -> Result<ChunkMap> {
        let prefix = self.prefix();
        let mut any_zero = None;
        let mut parts = Vec::with_capacity(self.parts.len());
        'tables: for p in &self.parts {
            // `chunk_count * 8` fits in the file (checked in `read`).
            let mut ranges = Vec::with_capacity(p.chunk_count as usize);
            for k in 0..p.chunk_count as usize {
                let at = p.table_offset + k * 8;
                let offset = u64::from_le_bytes(
                    bytes
                        .get(at..at + 8)
                        .ok_or(LayoutViolation::OffsetTableTruncated {
                            part: p.index,
                            needed: at as u64 + 8,
                            available: bytes.len() as u64,
                        })?
                        .try_into()
                        .unwrap(),
                );
                if offset == 0 {
                    any_zero = Some(p.index);
                    break 'tables;
                }
                let out_of_bounds = LayoutViolation::ChunkOutOfBounds {
                    part: p.index,
                    chunk: k as u32,
                    offset,
                };
                let start = usize::try_from(offset)
                    .ok()
                    .filter(|&o| o >= self.tables_end)
                    .ok_or_else(|| out_of_bounds.clone())?;
                let len = p
                    .kind
                    .chunk_len(bytes, start, prefix)
                    .ok_or(out_of_bounds)?;
                if prefix != 0 {
                    let found = i32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
                    if found != p.index as i32 {
                        return Err(LayoutViolation::ChunkPartNumber {
                            part: p.index,
                            chunk: k as u32,
                            found,
                        }
                        .into());
                    }
                }
                ranges.push(start..start + len);
            }
            parts.push(ranges);
        }
        let Some(zero_part) = any_zero else {
            return Ok(ChunkMap {
                parts,
                rebuilt: false,
                zero_part: 0,
            });
        };
        let mut map = self.rebuild_tables(bytes)?;
        map.zero_part = zero_part;
        Ok(map)
    }

    /// Walk the chunk region and place each chunk in its part's table
    /// by its coordinates.
    fn rebuild_tables(&self, bytes: &[u8]) -> Result<ChunkMap> {
        let prefix = self.prefix();
        let mut slots: Vec<Vec<Option<Range<usize>>>> = self
            .parts
            .iter()
            .map(|p| vec![None; p.chunk_count as usize])
            .collect();
        let total: u64 = self.parts.iter().map(|p| p.chunk_count).sum();
        let mut pos = self.tables_end;
        let mut placed: u64 = 0;
        while placed < total && pos < bytes.len() {
            let part = if prefix != 0 {
                let Some(raw) = bytes.get(pos..pos + 4) else {
                    break;
                };
                let n = i32::from_le_bytes(raw.try_into().unwrap());
                match usize::try_from(n).ok().filter(|&n| n < self.parts.len()) {
                    Some(n) => n,
                    None => break,
                }
            } else {
                0
            };
            let p = &self.parts[part];
            let Some(len) = p.kind.chunk_len(bytes, pos, prefix) else {
                break;
            };
            let index = p
                .shape
                .index_of(p.kind, &bytes[pos + prefix..pos + len])
                .filter(|&i| i < p.chunk_count);
            let Some(index) = index else { break };
            let slot = &mut slots[part][index as usize];
            if slot.is_none() {
                placed += 1;
            }
            *slot = Some(pos..pos + len);
            pos += len;
        }
        let mut parts = Vec::with_capacity(slots.len());
        for (p, table) in self.parts.iter().zip(slots) {
            let found = table.iter().filter(|s| s.is_some()).count();
            if found as u64 != p.chunk_count {
                return Err(LayoutViolation::ChunksMissing {
                    part: p.index,
                    found: found as u32,
                    expected: p.chunk_count.min(u32::MAX as u64) as u32,
                }
                .into());
            }
            parts.push(table.into_iter().flatten().collect());
        }
        Ok(ChunkMap {
            parts,
            rebuilt: true,
            zero_part: 0,
        })
    }

    /// Version flags a writer gives a file with these parts (§2.1 of
    /// the note): multi-part files never carry the single-part-tiled
    /// flag; the long-name flag follows the names.
    fn written_flags(multipart: bool, kinds: &[PartKind], long_names: bool) -> u32 {
        let any_deep = kinds.iter().any(|k| k.is_deep());
        let mut raw = 2u32;
        if multipart {
            raw |= FLAG_MULTIPART;
        } else if kinds[0] == PartKind::Tiled {
            raw |= FLAG_SINGLE_TILE;
        }
        if any_deep {
            raw |= FLAG_DEEP;
        }
        if long_names {
            raw |= FLAG_LONG_NAMES;
        }
        raw
    }
}

/// Read the layout of `bytes` and follow its offset tables.
///
/// Lenient (`strict == false`): fails on what a conforming reader
/// refuses. Strict: additionally fails on the first tolerated
/// irregularity (see [`LayoutViolation::is_tolerated`]) — the files a
/// conforming *writer* never produces.
pub fn check_layout(bytes: &[u8], strict: bool) -> Result<FileLayout> {
    let mut layout = FileLayout::read(bytes)?;
    let map = layout.chunks(bytes)?;
    if map.rebuilt {
        layout
            .irregularities
            .push(LayoutViolation::OffsetTableZero {
                part: map.zero_part,
            });
    }
    if strict && let Some(v) = layout.irregularities.first() {
        return Err(v.clone().into());
    }
    Ok(layout)
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

/// One part ready to be written: its attribute table as raw attribute
/// records and its chunks (without part-number prefix) in table order.
struct PartImage<'a> {
    kind: PartKind,
    attributes: Vec<Cow<'a, [u8]>>,
    long_names: bool,
    chunks: Vec<&'a [u8]>,
}

fn attribute_record(a: &Attribute) -> Vec<u8> {
    let (type_name, payload) = encode_attribute_value(&a.value);
    let mut out = Vec::with_capacity(a.name.len() + type_name.len() + 6 + payload.len());
    out.extend_from_slice(a.name.as_bytes());
    out.push(0);
    out.extend_from_slice(type_name.as_bytes());
    out.push(0);
    out.extend_from_slice(&(payload.len() as i32).to_le_bytes());
    out.extend_from_slice(&payload);
    out
}

fn has_long_name(attributes: &[Attribute]) -> bool {
    attributes.iter().any(|a| {
        a.name.len() > SHORT_NAME_MAX
            || matches!(&a.value, AttributeValue::Channels(chs)
                if chs.iter().any(|c| c.name.len() > SHORT_NAME_MAX))
    })
}

/// Write a file from parts: magic, version, headers (plus the empty
/// header of a multi-part file), offset tables, chunks in table order.
fn assemble(multipart: bool, parts: &[PartImage<'_>]) -> Vec<u8> {
    let kinds: Vec<PartKind> = parts.iter().map(|p| p.kind).collect();
    let long = parts.iter().any(|p| p.long_names);
    let raw = FileLayout::written_flags(multipart, &kinds, long);
    let prefix = if multipart { 4 } else { 0 };
    let chunk_bytes: usize = parts
        .iter()
        .flat_map(|p| p.chunks.iter())
        .map(|c| c.len() + prefix)
        .sum();
    let mut out = Vec::with_capacity(1024 + chunk_bytes);
    out.extend_from_slice(&EXR_MAGIC.to_le_bytes());
    out.extend_from_slice(&raw.to_le_bytes());
    for p in parts {
        for a in &p.attributes {
            out.extend_from_slice(a);
        }
        out.push(0);
    }
    if multipart {
        out.push(0);
    }
    let total: usize = parts.iter().map(|p| p.chunks.len()).sum();
    let mut at = out.len() + total * 8;
    for p in parts {
        for c in &p.chunks {
            out.extend_from_slice(&(at as u64).to_le_bytes());
            at += prefix + c.len();
        }
    }
    for (i, p) in parts.iter().enumerate() {
        for c in &p.chunks {
            if multipart {
                out.extend_from_slice(&(i as u32).to_le_bytes());
            }
            out.extend_from_slice(c);
        }
    }
    out
}

/// Put `record` (an attribute named `name`) into a raw attribute list:
/// in place of an attribute of the same name, else at the position
/// that keeps a byte-wise sorted list sorted, else at the end.
fn upsert<'a>(
    names: &mut Vec<String>,
    records: &mut Vec<Cow<'a, [u8]>>,
    name: &str,
    record: Vec<u8>,
) {
    if let Some(i) = names.iter().position(|n| n == name) {
        records[i] = Cow::Owned(record);
        return;
    }
    let sorted = names.windows(2).all(|w| w[0].as_bytes() <= w[1].as_bytes());
    let at = if sorted {
        names.partition_point(|n| n.as_bytes() < name.as_bytes())
    } else {
        names.len()
    };
    names.insert(at, name.to_string());
    records.insert(at, Cow::Owned(record));
}

// ---------------------------------------------------------------------------
// Reader front pass
// ---------------------------------------------------------------------------

/// Validate a multi-part file and hand the sequential part readers a
/// file they can walk: `chunkCount` present and exact in every part,
/// and the chunks stored back to back, in offset-table order, right
/// after the tables.
///
/// A regular file is returned as is. An irregular one that a
/// conforming reader accepts (missing / wrong `chunkCount`, chunks
/// stored in another order or after a gap, zeroed tables) is rewritten
/// in that regular form without touching a payload byte.
pub(crate) fn regular_multipart(bytes: &[u8]) -> Result<Cow<'_, [u8]>> {
    let layout = FileLayout::read(bytes)?;
    if !layout.is_multipart() {
        return Ok(Cow::Borrowed(bytes));
    }
    let map = layout.chunks(bytes)?;
    let counts_exact = layout
        .parts
        .iter()
        .all(|p| p.declared_chunk_count.map(|d| d as i64) == Some(p.chunk_count as i64));
    let mut expected = layout.tables_end;
    let mut back_to_back = true;
    'walk: for ranges in &map.parts {
        for r in ranges {
            if r.start != expected {
                back_to_back = false;
                break 'walk;
            }
            expected = r.end;
        }
    }
    if counts_exact && back_to_back {
        return Ok(Cow::Borrowed(bytes));
    }
    let mut parts = Vec::with_capacity(layout.parts.len());
    for (p, ranges) in layout.parts.iter().zip(&map.parts) {
        let mut names: Vec<String> = p.attributes.iter().map(|a| a.name.clone()).collect();
        let mut records: Vec<Cow<'_, [u8]>> = p
            .spans
            .iter()
            .map(|s| Cow::Borrowed(&bytes[s.clone()]))
            .collect();
        let count = Attribute {
            name: "chunkCount".to_string(),
            value: AttributeValue::Int(p.chunk_count as i32),
        };
        upsert(
            &mut names,
            &mut records,
            "chunkCount",
            attribute_record(&count),
        );
        parts.push(PartImage {
            kind: p.kind,
            attributes: records,
            long_names: layout.version.long_names || has_long_name(&p.attributes),
            chunks: ranges.iter().map(|r| &bytes[r.start + 4..r.end]).collect(),
        });
    }
    let mut out = assemble(true, &parts);
    // Keep the deep flag as the file had it: the parts decide.
    let raw = u32::from_le_bytes(out[4..8].try_into().unwrap())
        | (layout.version.raw & (FLAG_DEEP | FLAG_LONG_NAMES));
    out[4..8].copy_from_slice(&raw.to_le_bytes());
    Ok(Cow::Owned(out))
}

/// Finish a file a writer of this crate assembled: set the long-name
/// flag when a name needs it, then hold the file to the strict layout
/// rules (unique part names, shared attributes, exact `chunkCount`,
/// filled tables, part numbers).
pub(crate) fn finish_written(mut out: Vec<u8>) -> Result<Vec<u8>> {
    let list = parse_header_list(&out)?;
    let long = list.headers.iter().any(|h| {
        h.iter().any(|a| {
            a.attribute.name.len() > SHORT_NAME_MAX
                || matches!(&a.attribute.value, AttributeValue::Channels(chs)
                    if chs.iter().any(|c| c.name.len() > SHORT_NAME_MAX))
        })
    });
    if long && !list.version.long_names {
        let raw = list.version.raw | FLAG_LONG_NAMES;
        out[4..8].copy_from_slice(&raw.to_le_bytes());
    }
    check_layout(&out, true)?;
    Ok(out)
}

/// Rewrite the `dataWindow` and `displayWindow` of a single-part file
/// in place (both are fixed-size `box2i` values, so nothing moves).
/// The data window must keep its size: chunk coordinates of a tiled
/// part are tile indices relative to the window's corner and stay
/// valid; scanline chunks carry absolute `y` values and are not
/// handled here.
pub(crate) fn place_windows(mut file: Vec<u8>, data: Box2i, display: Box2i) -> Result<Vec<u8>> {
    let list = parse_header_list(&file)?;
    for (name, window) in [("dataWindow", data), ("displayWindow", display)] {
        let attr = find(&list.headers[0], name).ok_or_else(|| {
            ExrError::invalid(format!("OpenEXR encoder: written header has no '{name}'"))
        })?;
        match attr.attribute.value {
            AttributeValue::Box2i(old) if old == window => {}
            AttributeValue::Box2i(old) => {
                if name == "dataWindow"
                    && (old.width() != window.width() || old.height() != window.height())
                {
                    return Err(ExrError::invalid(
                        "OpenEXR encoder: a data window can be moved, not resized",
                    ));
                }
                let at = attr.span.end - 16;
                for (i, v) in [window.x_min, window.y_min, window.x_max, window.y_max]
                    .into_iter()
                    .enumerate()
                {
                    file[at + i * 4..at + i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            _ => {
                return Err(ExrError::invalid(format!(
                    "OpenEXR encoder: '{name}' is not a box2i"
                )));
            }
        }
    }
    Ok(file)
}

// ---------------------------------------------------------------------------
// Repack
// ---------------------------------------------------------------------------

/// How [`split_parts_with`] writes each part.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct SplitOptions {
    /// Drop the attributes only a multi-part file needs: `name`, `type`
    /// and `chunkCount` from a flat part (the result is what a writer
    /// produces for a single-part file), `name` from a deep part (which
    /// keeps `type` and `chunkCount`). `false` (default) keeps every
    /// header unchanged, so [`combine_parts`] restores the file.
    pub strip_part_attributes: bool,
}

impl SplitOptions {
    /// Defaults: headers unchanged.
    pub fn new() -> Self {
        Self::default()
    }
    /// Set [`SplitOptions::strip_part_attributes`].
    pub fn with_strip_part_attributes(mut self, strip: bool) -> Self {
        self.strip_part_attributes = strip;
        self
    }
}

/// How [`combine_parts_with`] reconciles its inputs.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct CombineOptions {
    /// Give every part the union of the inputs' display windows instead
    /// of refusing inputs that disagree (`displayWindow` is shared by
    /// all parts of a file). Default `false`.
    pub unify_display_window: bool,
    /// Rename a part whose name is already taken (`<name>.<n>`) instead
    /// of refusing it. Default `false`.
    pub rename_duplicates: bool,
    /// Part names by input position, replacing the inputs' own `name`
    /// attributes. Inputs beyond the list keep theirs (or get
    /// `part<i>`). Default empty.
    pub names: Vec<String>,
}

impl CombineOptions {
    /// Defaults: refuse disagreeing display windows and duplicate names.
    pub fn new() -> Self {
        Self::default()
    }
    /// Set [`CombineOptions::unify_display_window`].
    pub fn with_unify_display_window(mut self, unify: bool) -> Self {
        self.unify_display_window = unify;
        self
    }
    /// Set [`CombineOptions::names`].
    pub fn with_names<S: Into<String>>(mut self, names: impl IntoIterator<Item = S>) -> Self {
        self.names = names.into_iter().map(Into::into).collect();
        self
    }
    /// Set [`CombineOptions::rename_duplicates`].
    pub fn with_rename_duplicates(mut self, rename: bool) -> Self {
        self.rename_duplicates = rename;
        self
    }
}

/// Every part of `bytes` as a single-part file, headers unchanged
/// ([`split_parts_with`] with default options). A single-part input
/// comes back as its one part.
pub fn split_parts(bytes: &[u8]) -> Result<Vec<Vec<u8>>> {
    split_parts_with(bytes, &SplitOptions::default())
}

/// Every part of `bytes` as a single-part file (§6.2 of the multi-part
/// note): the part's header, its offset table with rewritten entries,
/// and its chunks without their part-number prefix, in table order.
/// Payloads are copied untouched; flat, tiled and deep parts alike.
///
/// The version field of each output is 2 plus the single-part-tiled
/// flag for a `tiledimage` part, the deep flag for a deep part and the
/// long-name flag when that header has a name above 31 bytes.
pub fn split_parts_with(bytes: &[u8], opts: &SplitOptions) -> Result<Vec<Vec<u8>>> {
    let layout = FileLayout::read(bytes)?;
    let map = layout.chunks(bytes)?;
    let prefix = layout.prefix();
    let mut out = Vec::with_capacity(layout.parts.len());
    for (p, ranges) in layout.parts.iter().zip(&map.parts) {
        let keep = |name: &str| {
            if !opts.strip_part_attributes {
                return true;
            }
            match name {
                "name" => false,
                "type" | "chunkCount" => p.kind.is_deep(),
                _ => true,
            }
        };
        let kept: Vec<usize> = (0..p.attributes.len())
            .filter(|&i| keep(&p.attributes[i].name))
            .collect();
        let long_names = kept.iter().any(|&i| {
            let a = &p.attributes[i];
            a.name.len() > SHORT_NAME_MAX
                || matches!(&a.value, AttributeValue::Channels(chs)
                    if chs.iter().any(|c| c.name.len() > SHORT_NAME_MAX))
        });
        let part = PartImage {
            kind: p.kind,
            attributes: kept
                .iter()
                .map(|&i| Cow::Borrowed(&bytes[p.spans[i].clone()]))
                .collect(),
            long_names,
            chunks: ranges
                .iter()
                .map(|r| &bytes[r.start + prefix..r.end])
                .collect(),
        };
        out.push(assemble(false, &[part]));
    }
    Ok(out)
}

/// Combine single-part files into one multi-part file
/// ([`combine_parts_with`] with default options: inputs whose shared
/// attributes disagree, or whose names collide, are refused).
pub fn combine_parts<B: AsRef<[u8]>>(files: &[B]) -> Result<Vec<u8>> {
    combine_parts_with(files, &CombineOptions::default())
}

/// Combine single-part files — flat scanline, flat tiled, deep
/// scanline, deep tiled, in any mix — into one multi-part file (§6.1
/// of the multi-part note). One input gives a one-part multi-part
/// file.
///
/// Each header is copied (minus a stray `tiles` attribute on a
/// scanline input); `name` (`part<i>` when the input has none)
/// and `type` are added when absent and `chunkCount` is added or
/// corrected, each placed in byte-wise name order when the header is
/// sorted that way. Chunks are copied behind their part number and the
/// offset tables rewritten. The version field is 2 with the multi-part
/// flag, the deep flag when a part is deep and the long-name flag when
/// any name exceeds 31 bytes.
///
/// The attributes all parts must share are checked first:
/// `displayWindow` (unified instead under
/// [`CombineOptions::unify_display_window`]), `pixelAspectRatio`
/// (absent = 1.0) and `chromaticities` (absent everywhere, or present
/// in part 0 and identical wherever else it appears) —
/// [`LayoutViolation::SharedAttributeMismatch`] otherwise. Part names
/// must be unique ([`LayoutViolation::DuplicatePartName`], or renamed
/// under [`CombineOptions::rename_duplicates`]).
pub fn combine_parts_with<B: AsRef<[u8]>>(files: &[B], opts: &CombineOptions) -> Result<Vec<u8>> {
    if files.is_empty() {
        return Err(LayoutViolation::NoParts.into());
    }
    struct Input<'a> {
        bytes: &'a [u8],
        part: PartLayout,
        chunks: Vec<Range<usize>>,
    }
    let mut inputs: Vec<Input<'_>> = Vec::with_capacity(files.len());
    for (i, f) in files.iter().enumerate() {
        let bytes = f.as_ref();
        let layout = FileLayout::read(bytes)?;
        if layout.is_multipart() {
            return Err(ExrError::invalid(format!(
                "OpenEXR combine: input {i} is already a multi-part file"
            )));
        }
        let mut map = layout.chunks(bytes)?;
        let mut part = layout.parts.into_iter().next().expect("one header");
        part.index = i as u32;
        inputs.push(Input {
            bytes,
            part,
            chunks: map.parts.swap_remove(0),
        });
    }

    // Names: keep, default, and disambiguate or refuse.
    let mut names: Vec<String> = Vec::with_capacity(inputs.len());
    for (i, input) in inputs.iter().enumerate() {
        let mut name = opts.names.get(i).cloned().unwrap_or_else(|| {
            input
                .part
                .name
                .clone()
                .filter(|n| !n.is_empty())
                .unwrap_or_else(|| format!("part{i}"))
        });
        if let Some(first) = names.iter().position(|n| *n == name) {
            if !opts.rename_duplicates {
                return Err(LayoutViolation::DuplicatePartName {
                    part: i as u32,
                    first: first as u32,
                    name,
                }
                .into());
            }
            let base = name.clone();
            let mut n = i;
            loop {
                name = format!("{base}.{n}");
                if !names.contains(&name) {
                    break;
                }
                n += 1;
            }
        }
        names.push(name);
    }

    let shared_window = if opts.unify_display_window {
        inputs
            .iter()
            .filter_map(|input| {
                input.part.attributes.iter().find_map(|a| match &a.value {
                    AttributeValue::Box2i(b) if a.name == "displayWindow" => Some(*b),
                    _ => None,
                })
            })
            .reduce(|a, b| Box2i {
                x_min: a.x_min.min(b.x_min),
                y_min: a.y_min.min(b.y_min),
                x_max: a.x_max.max(b.x_max),
                y_max: a.y_max.max(b.y_max),
            })
    } else {
        None
    };

    let mut parts = Vec::with_capacity(inputs.len());
    for (input, name) in inputs.iter().zip(&names) {
        let p = &input.part;
        // A `tiles` attribute on a scanline file is inert in a
        // single-part file but makes a scanline part unreadable: it
        // does not travel.
        let keep = |a: &&Attribute| p.kind.is_tiled() || a.name != "tiles";
        let mut attr_names: Vec<String> = p
            .attributes
            .iter()
            .filter(keep)
            .map(|a| a.name.clone())
            .collect();
        let mut records: Vec<Cow<'_, [u8]>> = p
            .attributes
            .iter()
            .zip(&p.spans)
            .filter(|(a, _)| keep(a))
            .map(|(_, s)| Cow::Borrowed(&input.bytes[s.clone()]))
            .collect();
        let mut long_names = has_long_name(&p.attributes);
        let mut set = |name: &str, value: AttributeValue| {
            let a = Attribute {
                name: name.to_string(),
                value,
            };
            upsert(&mut attr_names, &mut records, name, attribute_record(&a));
        };
        if p.name.as_ref() != Some(name) {
            set("name", AttributeValue::String(name.clone()));
        }
        // In a single-part flat file `type` is not authoritative; in a
        // multi-part file it selects the chunk layout.
        if !p.kind.is_deep() {
            set(
                "type",
                AttributeValue::String(p.kind.type_name().to_string()),
            );
        }
        if p.declared_chunk_count.map(|d| d as i64) != Some(p.chunk_count as i64) {
            set("chunkCount", AttributeValue::Int(p.chunk_count as i32));
        }
        if let Some(w) = shared_window {
            let current = p.attributes.iter().find(|a| a.name == "displayWindow");
            if current.map(|a| &a.value) != Some(&AttributeValue::Box2i(w)) {
                set("displayWindow", AttributeValue::Box2i(w));
            }
        }
        long_names |= name.len() > SHORT_NAME_MAX;
        parts.push(PartImage {
            kind: p.kind,
            attributes: records,
            long_names,
            chunks: input
                .chunks
                .iter()
                .map(|r| &input.bytes[r.clone()])
                .collect(),
        });
    }
    let out = assemble(true, &parts);
    // Shared attributes, names, counts and tables are held to the
    // strict rules on the assembled file.
    check_layout(&out, true)?;
    Ok(out)
}
