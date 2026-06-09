//! Tiled OpenEXR file support (read-only).
//!
//! When the version-field's `single_tile` bit (0x200) is set, pixel
//! data is laid out as a grid of `tileXSize × tileYSize` tiles instead
//! of as scanline blocks. The header carries an extra `tiles` attribute
//! (`tiledesc`) describing the tile grid:
//!
//! ```text
//! tiledesc: { u32 tileXSize, u32 tileYSize, u8 mode }
//!     mode: low 4 bits  = level mode  (0=ONE_LEVEL, 1=MIPMAP, 2=RIPMAP)
//!           high 4 bits = round mode  (0=ROUND_DOWN, 1=ROUND_UP)
//! ```
//!
//! Tile chunk on disk:
//!
//! ```text
//! i32 tileX
//! i32 tileY
//! i32 levelX     (always 0 for ONE_LEVEL)
//! i32 levelY     (always 0 for ONE_LEVEL)
//! i32 payloadSize
//! u8  payload[payloadSize]
//! ```
//!
//! Round 2: ONE_LEVEL only; multi-level mip/rip maps deferred.

use crate::error::{ExrError, Result};
use crate::types::AttributeValue;

/// Decoded `tiledesc` attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileDesc {
    pub x_size: u32,
    pub y_size: u32,
    /// Low 4 bits of the mode byte.
    pub level_mode: u8,
    /// High 4 bits of the mode byte.
    pub round_mode: u8,
}

impl TileDesc {
    /// Parse the 9-byte `tiledesc` payload.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() != 9 {
            return Err(ExrError::invalid(format!(
                "tiledesc payload size {} != 9",
                data.len()
            )));
        }
        let x_size = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let y_size = u32::from_le_bytes(data[4..8].try_into().unwrap());
        let mode = data[8];
        Ok(TileDesc {
            x_size,
            y_size,
            level_mode: mode & 0x0F,
            round_mode: (mode >> 4) & 0x0F,
        })
    }

    /// Encode back to the on-disk 9-byte payload — round-trip inverse of
    /// [`TileDesc::from_bytes`].
    ///
    /// `level_mode` and `round_mode` are masked to their 4-bit nibbles
    /// before packing (high nibble = round mode, low nibble = level
    /// mode). Returns exactly 9 bytes: two LE `u32` followed by the
    /// packed mode byte.
    pub fn to_bytes(&self) -> [u8; 9] {
        let mut out = [0u8; 9];
        out[0..4].copy_from_slice(&self.x_size.to_le_bytes());
        out[4..8].copy_from_slice(&self.y_size.to_le_bytes());
        out[8] = ((self.round_mode & 0x0F) << 4) | (self.level_mode & 0x0F);
        out
    }
}

/// Pull a tile-desc out of a parsed attribute value.
///
/// Accepts both the new typed [`AttributeValue::TileDesc`] variant
/// produced by [`crate::header::parse_attribute_value`] and the legacy
/// [`AttributeValue::Other`] shape (with `type_name == "tiledesc"` and
/// a 9-byte payload) that pre-existing encoder sites still emit, so
/// every caller route continues to work unchanged.
pub fn tiledesc_from_attribute(value: &AttributeValue) -> Result<TileDesc> {
    match value {
        AttributeValue::TileDesc(td) => Ok(*td),
        AttributeValue::Other { type_name, data } if type_name == "tiledesc" => {
            TileDesc::from_bytes(data)
        }
        _ => Err(ExrError::invalid(
            "expected `tiles` attribute of type tiledesc".to_string(),
        )),
    }
}

/// Pull the raw `(x_size, y_size, mode)` triple from a parsed `tiles`
/// attribute value. Accepts both the typed [`AttributeValue::TileDesc`]
/// variant and the legacy [`AttributeValue::Other`] shape with
/// `type_name == "tiledesc"` and a 9-byte payload.
///
/// `mode` is the packed mode byte (high nibble = round mode, low nibble
/// = level mode); callers that need to dispatch on the raw byte (e.g.
/// the deep-tiled readers' MIPMAP/RIPMAP routing) prefer this raw form
/// over the split-nibble [`TileDesc`] view.
pub fn tiledesc_raw_from_attribute(value: &AttributeValue) -> Option<(u32, u32, u8)> {
    match value {
        AttributeValue::TileDesc(td) => Some((
            td.x_size,
            td.y_size,
            ((td.round_mode & 0x0F) << 4) | (td.level_mode & 0x0F),
        )),
        AttributeValue::Other { type_name, data } if type_name == "tiledesc" && data.len() == 9 => {
            let xs = u32::from_le_bytes(data[0..4].try_into().unwrap());
            let ys = u32::from_le_bytes(data[4..8].try_into().unwrap());
            Some((xs, ys, data[8]))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_one_level_64x64_round_down() {
        let mut bytes = Vec::with_capacity(9);
        bytes.extend_from_slice(&64u32.to_le_bytes());
        bytes.extend_from_slice(&64u32.to_le_bytes());
        bytes.push(0); // OneLevel + RoundDown
        let td = TileDesc::from_bytes(&bytes).unwrap();
        assert_eq!(td.x_size, 64);
        assert_eq!(td.y_size, 64);
        assert_eq!(td.level_mode, 0);
        assert_eq!(td.round_mode, 0);
    }

    #[test]
    fn parse_mipmap_round_up() {
        let mut bytes = Vec::with_capacity(9);
        bytes.extend_from_slice(&32u32.to_le_bytes());
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.push(0x11); // MipmapLevels (1) + RoundUp (1)
        let td = TileDesc::from_bytes(&bytes).unwrap();
        assert_eq!(td.x_size, 32);
        assert_eq!(td.y_size, 16);
        assert_eq!(td.level_mode, 1);
        assert_eq!(td.round_mode, 1);
    }

    #[test]
    fn rejects_bad_size() {
        assert!(TileDesc::from_bytes(&[0u8; 8]).is_err());
        assert!(TileDesc::from_bytes(&[0u8; 10]).is_err());
    }

    #[test]
    fn to_bytes_roundtrip_one_level() {
        let td = TileDesc {
            x_size: 64,
            y_size: 64,
            level_mode: 0,
            round_mode: 0,
        };
        let bytes = td.to_bytes();
        assert_eq!(bytes.len(), 9);
        assert_eq!(TileDesc::from_bytes(&bytes).unwrap(), td);
    }

    #[test]
    fn to_bytes_packs_mode_nibbles() {
        // MIPMAP (1) + ROUND_UP (1) -> high nibble 1, low nibble 1 -> 0x11.
        let td = TileDesc {
            x_size: 32,
            y_size: 16,
            level_mode: 1,
            round_mode: 1,
        };
        let bytes = td.to_bytes();
        assert_eq!(bytes[8], 0x11);
        // RIPMAP (2) + ROUND_DOWN (0) -> 0x02.
        let td2 = TileDesc {
            x_size: 8,
            y_size: 8,
            level_mode: 2,
            round_mode: 0,
        };
        assert_eq!(td2.to_bytes()[8], 0x02);
    }

    #[test]
    fn to_bytes_little_endian_sizes() {
        // x_size = 0x01020304, y_size = 0x05060708; expected LE: 04 03 02 01 08 07 06 05
        let td = TileDesc {
            x_size: 0x0102_0304,
            y_size: 0x0506_0708,
            level_mode: 0,
            round_mode: 0,
        };
        let bytes = td.to_bytes();
        assert_eq!(&bytes[0..4], &[0x04, 0x03, 0x02, 0x01]);
        assert_eq!(&bytes[4..8], &[0x08, 0x07, 0x06, 0x05]);
        assert_eq!(bytes[8], 0x00);
    }
}
