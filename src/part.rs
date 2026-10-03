//! One flat OpenEXR part with its channels as named `f32` planes — the
//! depth type behind [`crate::parse_exr`] and the multi-part readers.
//!
//! Defined without any `oxideav-core` dependency so the crate builds with
//! the default `registry` feature off. The HDR pixel data is exposed as
//! separate `f32` planes per channel so callers don't have to know
//! whether the file used HALF or FLOAT encoding — both decode to f32
//! (HALF → f32 is exact; UINT is converted, exact below 2^24).
//!
//! Channels are returned in alphabetical order (matching the on-disk
//! pixel data layout). Standard OpenEXR images use names `R`/`G`/`B`/`A`
//! but the field name is preserved verbatim from the file.
//!
//! The contract-shaped [`crate::ExrImage`] (one packed `RgbF32Le` /
//! `RgbaF32Le` / `GrayF32Le` plane) is the *view* of a part's colour
//! channels; [`ExrPart`] keeps every channel, whatever its name. Before
//! the image-crate API contract this type was called `ExrImage`.

use crate::types::{Attribute, Box2i, Channel, Compression, LineOrder};

/// One decoded channel: name + pixel data, always converted to `f32`.
#[derive(Debug, Clone, PartialEq)]
pub struct ExrPlane {
    pub name: String,
    /// Row-major pixel samples, `width * height` long.
    pub samples: Vec<f32>,
}

/// One decoded flat EXR part: every channel as a named `f32` plane.
///
/// `data_window` is the file's `dataWindow` attribute. `display_window`
/// is the file's `displayWindow`. `width()` / `height()` are the data
/// window dimensions (which is what the pixel planes are sized for).
#[derive(Debug, Clone, PartialEq)]
pub struct ExrPart {
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

impl ExrPart {
    pub fn width(&self) -> u32 {
        self.data_window.width()
    }
    pub fn height(&self) -> u32 {
        self.data_window.height()
    }
    /// The layers of this image's channel list (see [`crate::layers`]):
    /// the base layer first, then every `prefix.` layer, each with its
    /// channel indices, colour shape and view.
    pub fn layers(&self) -> Vec<crate::layers::ExrLayer> {
        crate::layers::enumerate_layers(&self.channels, &self.attributes)
    }
}
