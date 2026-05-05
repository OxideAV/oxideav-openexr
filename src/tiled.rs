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
}

/// Pull a tile-desc out of a parsed attribute value.
pub fn tiledesc_from_attribute(value: &AttributeValue) -> Result<TileDesc> {
    match value {
        AttributeValue::Other { type_name, data } if type_name == "tiledesc" => {
            TileDesc::from_bytes(data)
        }
        _ => Err(ExrError::invalid(
            "expected `tiles` attribute of type tiledesc".to_string(),
        )),
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
}
