//! Layered channel names (`diffuse.R`, `left.R` / `right.R`,
//! `beauty.specular.B`, …): a typed enumeration of the layers a channel
//! list contains and the colour shape each one carries.
//!
//! An OpenEXR channel name is free-form, but the convention used by
//! layered and multi-view files is `<layer>.<base>`: everything up to
//! the last `.` names the layer (arbitrarily deep, `a.b.c.R` is layer
//! `a.b.c`), the remainder is the base channel (`R`, `G`, `B`, `A`,
//! `Y`, `RY`, `BY`, `Z`, …). Names without a `.` belong to the **base
//! layer** (named `""` here). Multi-view files list their views in the
//! `multiView` string-vector attribute; the first view's channels are
//! stored unprefixed (the base layer) and every other view's under its
//! own prefix, so `left` / `right` stereo typically shows up as the
//! base layer plus a `right` layer.
//!
//! [`enumerate_layers`] groups a channel list by layer prefix, keeps
//! each layer's channel indices in file (alphabetical) order, classifies
//! the layer by the base names it contains ([`LayerKind`]) and tags it
//! with the view it belongs to when the header carries `multiView`.
//! The classification mirrors the registry decoder's channel mapping so
//! a caller can tell in advance which layers will decode to a frame.

use crate::types::{Attribute, AttributeValue, Channel};

/// Split a channel name at its last `.` into `(layer, base)`; a name
/// without `.` is `("", name)`.
pub fn split_channel_name(name: &str) -> (&str, &str) {
    match name.rfind('.') {
        Some(i) => (&name[..i], &name[i + 1..]),
        None => ("", name),
    }
}

/// The `multiView` attribute (view names, first = default view) when
/// present and typed as a string vector.
pub fn multi_view(attrs: &[Attribute]) -> Option<&[String]> {
    attrs.iter().find_map(|a| match (&a.name[..], &a.value) {
        ("multiView", AttributeValue::StringVector(v)) => Some(v.as_slice()),
        _ => None,
    })
}

/// Colour shape of a layer, by the base channel names it contains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// `R`, `G`, `B` and `A`.
    Rgba,
    /// `R`, `G`, `B` without `A`.
    Rgb,
    /// `Y`, `RY`, `BY` and `A` (luminance/chroma with alpha).
    LumaChromaAlpha,
    /// `Y`, `RY`, `BY` without `A`.
    LumaChroma,
    /// `Y` and `A` only.
    GrayAlpha,
    /// `Y` only.
    Gray,
    /// `Z` without any colour channel (depth-only).
    Depth,
    /// Anything else — a partial colour triple, motion vectors, ids,
    /// arbitrary AOVs.
    Other,
}

impl LayerKind {
    /// `true` when the registry decoder can map a layer of this kind to
    /// a frame.
    pub fn has_frame_mapping(self) -> bool {
        !matches!(self, Self::Depth | Self::Other)
    }
}

/// One layer of a channel list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExrLayer {
    /// Layer prefix without the trailing `.`; `""` for the base layer.
    pub name: String,
    /// Indices into the channel list, in file order.
    pub channels: Vec<usize>,
    /// Colour shape (see [`LayerKind`]).
    pub kind: LayerKind,
    /// The view this layer belongs to, when the header lists views:
    /// the base layer is the first (default) view, a prefixed layer is
    /// a view when one of its prefix components names a view (so the
    /// `right` layer and a `diffuse.right` layer both belong to
    /// `right`).
    pub view: Option<String>,
}

impl ExrLayer {
    /// Full channel name for base name `base` within this layer.
    pub fn channel_name(&self, base: &str) -> String {
        if self.name.is_empty() {
            base.to_string()
        } else {
            format!("{}.{base}", self.name)
        }
    }
}

/// Classify a set of base channel names.
fn classify<'a>(bases: impl Iterator<Item = &'a str>) -> LayerKind {
    let (mut r, mut g, mut b, mut a, mut y, mut ry, mut by, mut z) =
        (false, false, false, false, false, false, false, false);
    for base in bases {
        match base {
            "R" => r = true,
            "G" => g = true,
            "B" => b = true,
            "A" => a = true,
            "Y" => y = true,
            "RY" => ry = true,
            "BY" => by = true,
            "Z" => z = true,
            _ => {}
        }
    }
    match (r && g && b, y, ry && by, a) {
        (true, _, _, true) => LayerKind::Rgba,
        (true, _, _, false) => LayerKind::Rgb,
        (false, true, true, true) => LayerKind::LumaChromaAlpha,
        (false, true, true, false) => LayerKind::LumaChroma,
        (false, true, false, true) => LayerKind::GrayAlpha,
        (false, true, false, false) => LayerKind::Gray,
        _ if z && !r && !g && !b => LayerKind::Depth,
        _ => LayerKind::Other,
    }
}

/// Group `channels` by layer prefix (module docs). The base layer, when
/// it has channels, comes first; the other layers follow in file order
/// of their first channel (alphabetical, since the channel list is
/// sorted).
pub fn enumerate_layers(channels: &[Channel], attributes: &[Attribute]) -> Vec<ExrLayer> {
    let views = multi_view(attributes);
    let mut layers: Vec<ExrLayer> = Vec::new();
    for (idx, ch) in channels.iter().enumerate() {
        let (layer, _) = split_channel_name(&ch.name);
        match layers.iter_mut().find(|l| l.name == layer) {
            Some(l) => l.channels.push(idx),
            None => layers.push(ExrLayer {
                name: layer.to_string(),
                channels: vec![idx],
                kind: LayerKind::Other,
                view: None,
            }),
        }
    }
    if let Some(pos) = layers.iter().position(|l| l.name.is_empty()) {
        if pos != 0 {
            let base = layers.remove(pos);
            layers.insert(0, base);
        }
    }
    for layer in &mut layers {
        layer.kind = classify(
            layer
                .channels
                .iter()
                .map(|&i| split_channel_name(&channels[i].name).1),
        );
        layer.view = views.and_then(|views| {
            if layer.name.is_empty() {
                views.first().cloned()
            } else {
                layer
                    .name
                    .split('.')
                    .find(|component| views.iter().any(|v| v == component))
                    .map(str::to_string)
            }
        });
    }
    layers
}

/// Find the layer a caller asked for by name: an exact layer-prefix
/// match first; failing that, when `name` is the default view (the
/// first `multiView` entry) the base layer.
pub fn find_layer<'a>(layers: &'a [ExrLayer], name: &str) -> Option<&'a ExrLayer> {
    layers.iter().find(|l| l.name == name).or_else(|| {
        layers
            .first()
            .filter(|l| l.name.is_empty() && l.view.as_deref() == Some(name))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PixelType;

    fn ch(name: &str) -> Channel {
        Channel {
            name: name.to_string(),
            pixel_type: PixelType::Half,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        }
    }

    #[test]
    fn split_at_last_dot() {
        assert_eq!(split_channel_name("R"), ("", "R"));
        assert_eq!(split_channel_name("diffuse.R"), ("diffuse", "R"));
        assert_eq!(split_channel_name("a.b.c.RY"), ("a.b.c", "RY"));
        assert_eq!(split_channel_name("trailing."), ("trailing", ""));
    }

    #[test]
    fn layers_group_and_classify() {
        let names = [
            "A",
            "B",
            "G",
            "R",
            "Z",
            "depth.Z",
            "diffuse.B",
            "diffuse.G",
            "diffuse.R",
            "ids.id",
            "lum.BY",
            "lum.RY",
            "lum.Y",
            "mask.A",
            "mask.Y",
            "vel.x",
            "vel.y",
        ];
        let chs: Vec<Channel> = names.iter().map(|n| ch(n)).collect();
        let layers = enumerate_layers(&chs, &[]);
        let summary: Vec<(&str, LayerKind, usize)> = layers
            .iter()
            .map(|l| (l.name.as_str(), l.kind, l.channels.len()))
            .collect();
        assert_eq!(
            summary,
            [
                ("", LayerKind::Rgba, 5),
                ("depth", LayerKind::Depth, 1),
                ("diffuse", LayerKind::Rgb, 3),
                ("ids", LayerKind::Other, 1),
                ("lum", LayerKind::LumaChroma, 3),
                ("mask", LayerKind::GrayAlpha, 2),
                ("vel", LayerKind::Other, 2),
            ]
        );
        assert_eq!(layers[2].channels, vec![6, 7, 8]);
        assert_eq!(layers[2].channel_name("G"), "diffuse.G");
        assert_eq!(layers[0].channel_name("G"), "G");
        assert!(layers.iter().all(|l| l.view.is_none()));
        assert!(LayerKind::Rgb.has_frame_mapping());
        assert!(!LayerKind::Depth.has_frame_mapping());
    }

    #[test]
    fn base_layer_comes_first_even_when_prefixed_names_sort_earlier() {
        // Uppercase prefixes sort before the bare lowercase-free names?
        // No — '.'-containing names can sort before "R": "A.x" < "R".
        let chs = vec![ch("Beauty.R"), ch("R"), ch("Beauty.G"), ch("Beauty.B")];
        let layers = enumerate_layers(&chs, &[]);
        assert_eq!(layers[0].name, "");
        assert_eq!(layers[0].kind, LayerKind::Other);
        assert_eq!(layers[1].name, "Beauty");
        assert_eq!(layers[1].kind, LayerKind::Rgb);
    }

    #[test]
    fn views_are_tagged_from_multi_view() {
        let chs: Vec<Channel> = [
            "B",
            "G",
            "R",
            "right.B",
            "right.G",
            "right.R",
            "spec.right.Y",
        ]
        .iter()
        .map(|n| ch(n))
        .collect();
        let attrs = vec![Attribute {
            name: "multiView".to_string(),
            value: AttributeValue::StringVector(vec!["left".to_string(), "right".to_string()]),
        }];
        let layers = enumerate_layers(&chs, &attrs);
        assert_eq!(layers[0].view.as_deref(), Some("left"));
        assert_eq!(layers[1].name, "right");
        assert_eq!(layers[1].view.as_deref(), Some("right"));
        assert_eq!(layers[1].kind, LayerKind::Rgb);
        assert_eq!(layers[2].name, "spec.right");
        assert_eq!(layers[2].view.as_deref(), Some("right"));
        assert_eq!(layers[2].kind, LayerKind::Gray);
        // Lookup by layer name or by default-view name.
        assert_eq!(find_layer(&layers, "right").unwrap().name, "right");
        assert_eq!(find_layer(&layers, "left").unwrap().name, "");
        assert_eq!(find_layer(&layers, "").unwrap().name, "");
        assert!(find_layer(&layers, "centre").is_none());
        assert_eq!(multi_view(&attrs).unwrap().len(), 2);
        assert!(multi_view(&[]).is_none());
    }
}
