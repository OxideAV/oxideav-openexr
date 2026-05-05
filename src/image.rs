//! Standalone image container returned by `oxideav-openexr`'s framework-free
//! decode API and accepted by the standalone encode API.
//!
//! Defined here so the crate can be built with the default `registry`
//! feature off (no `oxideav-core` dep). The HDR scanline data is exposed
//! as separate `f32` planes per channel so callers don't have to know
//! whether the file used HALF or FLOAT encoding — both decode to f32.
//!
//! Channels are returned in alphabetical order (matching the on-disk
//! pixel data layout). Standard OpenEXR images use names `R`/`G`/`B`/`A`
//! but the field name is preserved verbatim from the file.

use crate::types::{Attribute, Box2i, Channel, Compression, LineOrder};

/// One decoded channel: name + pixel data, always converted to `f32`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExrPlane {
    pub name: String,
    /// Row-major pixel samples, `width * height` long.
    pub samples: Vec<f32>,
}

/// One decoded EXR image.
///
/// `data_window` is the file's `dataWindow` attribute. `display_window`
/// is the file's `displayWindow`. `width()` / `height()` are the data
/// window dimensions (which is what the pixel planes are sized for).
#[derive(Debug, Clone, PartialEq)]
pub struct ExrImage {
    pub data_window: Box2i,
    pub display_window: Box2i,
    pub line_order: LineOrder,
    pub compression: Compression,
    pub pixel_aspect_ratio: f32,
    pub screen_window_center: (f32, f32),
    pub screen_window_width: f32,
    /// One [`ExrPlane`] per channel, in alphabetical order matching the
    /// channel list.
    pub channels: Vec<Channel>,
    pub planes: Vec<ExrPlane>,
    /// All header attributes, in file order, including the typed ones.
    /// Useful for inspecting / round-tripping non-required attributes.
    pub attributes: Vec<Attribute>,
}

impl ExrImage {
    pub fn width(&self) -> u32 {
        self.data_window.width()
    }
    pub fn height(&self) -> u32 {
        self.data_window.height()
    }
}
