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
#[non_exhaustive]
pub struct ExrPlane {
    pub name: String,
    /// Row-major pixel samples, `width * height` long.
    pub samples: Vec<f32>,
}

impl ExrPlane {
    /// Build `ExrPlane` from every field, in declaration order.
    pub fn new(name: String, samples: Vec<f32>) -> Self {
        Self { name, samples }
    }
}

/// One decoded flat EXR part: every channel as a named `f32` plane.
///
/// `data_window` is the file's `dataWindow` attribute. `display_window`
/// is the file's `displayWindow`. `width()` / `height()` are the data
/// window dimensions (which is what the pixel planes are sized for).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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
    /// Build `ExrPart` from every field, in declaration order.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        data_window: Box2i,
        display_window: Box2i,
        line_order: LineOrder,
        compression: Compression,
        pixel_aspect_ratio: f32,
        screen_window_center: (f32, f32),
        screen_window_width: f32,
        channels: Vec<Channel>,
        planes: Vec<ExrPlane>,
        attributes: Vec<Attribute>,
    ) -> Self {
        Self {
            data_window,
            display_window,
            line_order,
            compression,
            pixel_aspect_ratio,
            screen_window_center,
            screen_window_width,
            channels,
            planes,
            attributes,
        }
    }
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
    /// The `multiView` view names (first = default view); empty when
    /// the header lists none.
    pub fn views(&self) -> &[String] {
        crate::layers::multi_view(&self.attributes).unwrap_or(&[])
    }
    /// The view channel `index` belongs to ([`crate::layers::channel_view`]);
    /// `None` without views, for a channel in no view, or for an
    /// out-of-range index.
    pub fn channel_view(&self, index: usize) -> Option<&str> {
        let ch = self.channels.get(index)?;
        crate::layers::channel_view(&ch.name, &self.attributes)
    }
}
