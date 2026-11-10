//! Layered and multi-view channel names (`diffuse.R`, `right.R`,
//! `spec.left.Y`, …): a typed enumeration of the layers a channel list
//! contains, the colour shape each one carries and the view it belongs
//! to.
//!
//! # Layers
//!
//! A channel name is `<prefix>.<base>`: the base is the text after the
//! **last** `.` (`R`, `G`, `B`, `A`, `Y`, `RY`, `BY`, `Z`, …) and the
//! prefix — which may itself contain dots (`a.b.R` is layer `a.b`) —
//! names the layer. Names without a `.` belong to the **base layer**
//! (named `""` here). Selecting layer `L` selects exactly the channels
//! `L.<base>`; matching is literal on the full prefix (`a` does not
//! select `a.b.R`, `diffuse.` with a trailing dot selects nothing).
//!
//! # Views
//!
//! The `multiView` string-vector attribute lists view names, the first
//! being the **default view**. A channel's view follows from its name
//! ([`channel_view`]): an un-prefixed name belongs to the default view;
//! a prefixed one belongs to the view named by the **last component of
//! its prefix** when that component is listed (`right.R` → `right`,
//! `diffuse.left.R` → `left`, `x.y.z.right.B` → `right`), and to no view
//! otherwise (`diffuse.R`, `right.spec.G`). Writers store the default
//! view's channels un-prefixed, so asking for a layer named after the
//! default view selects the **base layer** (and ignores a stray literal
//! `left.*` set); without a `multiView` attribute, or for a view that is
//! not the first entry, the name is matched literally. Views within
//! layers are addressed by the full prefix (`spec.left`).
//!
//! [`enumerate_layers`] groups a channel list by prefix, keeps each
//! layer's channel indices in file (alphabetical) order, classifies the
//! layer by the base names it contains ([`LayerKind`], the same rules
//! the colour view decodes by) and tags it with its view.

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
///
/// Luminance channels take precedence: a layer holding `Y`, `RY` or `BY`
/// decodes through the luminance path even when `R` / `G` / `B` are also
/// present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LayerKind {
    /// `R`, `G`, `B` and `A`.
    Rgba,
    /// `R`, `G`, `B` without `A`.
    Rgb,
    /// Luminance/chroma with `A`: `RY` and/or `BY` (a missing one reads
    /// as `0`) with `Y` (a missing `Y` reads as `0.5`).
    LumaChromaAlpha,
    /// Luminance/chroma without `A` (as [`Self::LumaChromaAlpha`]).
    LumaChroma,
    /// `Y` and `A` only.
    GrayAlpha,
    /// `Y` only.
    Gray,
    /// `Z` without any colour channel (depth-only).
    Depth,
    /// One or two of `R`, `G`, `B` (± `A`, no luminance channel): the
    /// missing colour components decode as `0`.
    PartialRgb,
    /// Anything else — motion vectors, ids, alpha alone, arbitrary AOVs.
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
#[non_exhaustive]
pub struct ExrLayer {
    /// Layer prefix without the trailing `.`; `""` for the base layer.
    pub name: String,
    /// Indices into the channel list, in file order.
    pub channels: Vec<usize>,
    /// Colour shape (see [`LayerKind`]).
    pub kind: LayerKind,
    /// The view this layer belongs to, when the header lists views:
    /// the base layer is the first (default) view, a prefixed layer
    /// belongs to the view its last prefix component names (so the
    /// `right` layer and a `diffuse.right` layer both belong to `right`,
    /// `right.spec` to none) — module docs, *Views*.
    pub view: Option<String>,
}

impl ExrLayer {
    /// Build `ExrLayer` from every field, in declaration order.
    pub fn new(name: String, channels: Vec<usize>, kind: LayerKind, view: Option<String>) -> Self {
        Self {
            name,
            channels,
            kind,
            view,
        }
    }
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

/// Classify a set of base channel names (the colour view's rules).
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
    if ry || by {
        return if a {
            LayerKind::LumaChromaAlpha
        } else {
            LayerKind::LumaChroma
        };
    }
    if y {
        return if a {
            LayerKind::GrayAlpha
        } else {
            LayerKind::Gray
        };
    }
    match (r && g && b, r || g || b, a) {
        (true, _, true) => LayerKind::Rgba,
        (true, _, false) => LayerKind::Rgb,
        (false, true, _) => LayerKind::PartialRgb,
        _ if z => LayerKind::Depth,
        _ => LayerKind::Other,
    }
}

/// The view `prefix` (a layer name, `""` for the base layer) belongs
/// to under the view list `views` (module docs, *Views*).
fn prefix_view<'a>(prefix: &str, views: &'a [String]) -> Option<&'a str> {
    if prefix.is_empty() {
        return views.first().map(String::as_str);
    }
    let last = prefix.rsplit('.').next().unwrap_or(prefix);
    views.iter().find(|v| *v == last).map(String::as_str)
}

/// The view channel `name` belongs to under the header's `multiView`
/// attribute (module docs, *Views*); `None` without the attribute or
/// when the name places the channel in no view.
pub fn channel_view<'a>(name: &str, attributes: &'a [Attribute]) -> Option<&'a str> {
    let views = multi_view(attributes)?;
    prefix_view(split_channel_name(name).0, views)
}

/// The default view (the first `multiView` entry), when the header
/// lists views.
pub fn default_view(attributes: &[Attribute]) -> Option<&str> {
    multi_view(attributes)?.first().map(String::as_str)
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
    if let Some(pos) = layers.iter().position(|l| l.name.is_empty())
        && pos != 0
    {
        let base = layers.remove(pos);
        layers.insert(0, base);
    }
    for layer in &mut layers {
        layer.kind = classify(
            layer
                .channels
                .iter()
                .map(|&i| split_channel_name(&channels[i].name).1),
        );
        layer.view = views
            .and_then(|views| prefix_view(&layer.name, views))
            .map(str::to_string);
    }
    layers
}

/// Find the layer a caller asked for by name (module docs): with a
/// `multiView` header, the default view's name selects the base layer
/// (never a literal `<default>.*` layer); otherwise the layer whose
/// prefix equals `name` exactly (`""` = the base layer). A file whose
/// base layer is empty has no default-view set to select, so the
/// default view's name then falls back to the literal match.
pub fn find_layer<'a>(layers: &'a [ExrLayer], name: &str) -> Option<&'a ExrLayer> {
    let base = layers.first().filter(|l| l.name.is_empty());
    if !name.is_empty()
        && let Some(base) = base
        && base.view.as_deref() == Some(name)
    {
        return Some(base);
    }
    layers.iter().find(|l| l.name == name)
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
        assert_eq!(layers[0].kind, LayerKind::PartialRgb);
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
            "right.spec.G",
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
        assert_eq!(layers[2].name, "right.spec");
        assert_eq!(layers[2].view, None);
        assert_eq!(layers[2].kind, LayerKind::PartialRgb);
        assert_eq!(layers[3].name, "spec.right");
        assert_eq!(layers[3].view.as_deref(), Some("right"));
        assert_eq!(layers[3].kind, LayerKind::Gray);
        // Lookup by layer name or by default-view name.
        assert_eq!(find_layer(&layers, "right").unwrap().name, "right");
        assert_eq!(find_layer(&layers, "left").unwrap().name, "");
        assert_eq!(find_layer(&layers, "").unwrap().name, "");
        assert!(find_layer(&layers, "centre").is_none());
        // The view is the LAST prefix component; listed anywhere else it
        // does not count.
        assert_eq!(channel_view("R", &attrs), Some("left"));
        assert_eq!(channel_view("right.R", &attrs), Some("right"));
        assert_eq!(channel_view("diffuse.left.R", &attrs), Some("left"));
        assert_eq!(channel_view("x.y.z.right.B", &attrs), Some("right"));
        assert_eq!(channel_view("right.spec.G", &attrs), None);
        assert_eq!(channel_view("diffuse.R", &attrs), None);
        assert_eq!(channel_view("R", &[]), None);
        assert_eq!(default_view(&attrs), Some("left"));
        assert_eq!(default_view(&[]), None);
        // With views, the default view's name selects the base layer
        // even when a literal `left.*` layer exists; without the
        // attribute it is matched literally.
        let chs: Vec<Channel> = ["Y", "left.Y"].iter().map(|n| ch(n)).collect();
        let with = enumerate_layers(&chs, &attrs);
        assert_eq!(find_layer(&with, "left").unwrap().name, "");
        let without = enumerate_layers(&chs, &[]);
        assert_eq!(find_layer(&without, "left").unwrap().name, "left");
        // A view that is not the first entry is literal.
        let rl = vec![Attribute {
            name: "multiView".to_string(),
            value: AttributeValue::StringVector(vec!["right".to_string(), "left".to_string()]),
        }];
        let swapped = enumerate_layers(&chs, &rl);
        assert_eq!(find_layer(&swapped, "left").unwrap().name, "left");
        // Literal prefix matching: `a` does not select `a.b.R`.
        let chs: Vec<Channel> = ["a.b.R", "a.b.G", "a.b.B"].iter().map(|n| ch(n)).collect();
        let layers = enumerate_layers(&chs, &[]);
        assert!(find_layer(&layers, "a").is_none());
        assert!(find_layer(&layers, "a.b.").is_none());
        assert_eq!(find_layer(&layers, "a.b").unwrap().kind, LayerKind::Rgb);
        assert_eq!(multi_view(&attrs).unwrap().len(), 2);
        assert!(multi_view(&[]).is_none());
    }
}
