//! The colour view: mapping a flat part's named channel set to one packed
//! `f32` plane ([`crate::ExrImage`]) and back to the channel planes the
//! depth encoders take. Framework-free; the registry adapter and the
//! standalone API share this single implementation.
//!
//! # Decode mapping
//!
//! The selected layer's channels (see [`crate::layers`]) are matched by
//! the channel-presence rules of the reference RGBA interface:
//!
//! 1. Any of `RY` / `BY` present → luminance/chroma: RGB is
//!    reconstructed by [`crate::luma_chroma::luma_chroma_to_rgb`] with
//!    the part's luminance weights (its `chromaticities` attribute, else
//!    BT.709). A missing `RY` or `BY` reads as `0`, a missing `Y` as
//!    `0.5`. `RgbF32Le`, or `RgbaF32Le` with an `A`.
//! 2. `Y` without chroma → `GrayF32Le`; `Y` + `A` → `RgbaF32Le` with `Y`
//!    copied into R, G and B. Luminance channels win over `R` / `G` /
//!    `B` in the same layer (those are then ignored).
//! 3. Otherwise `R`, `G`, `B` → `RgbF32Le`, `RgbaF32Le` with `A`; when
//!    only one or two of them exist the missing ones read as `0`.
//! 4. Anything else is [`ExrError::Unsupported`] naming the channels.
//!
//! Every directly-mapped channel must be at 1×1 sampling; only `RY` /
//! `BY` may be sub-sampled. Channels outside the view are ignored
//! (reachable through [`crate::ExrPart`], which always carries the
//! planes as stored).

use crate::error::{ExrError, Result};
use crate::image::{ColorInfo, ExrImage, Metadata, PixelFormat, Plane};
use crate::layers::{ExrLayer, enumerate_layers, find_layer};
use crate::luma_chroma::{
    ChromaPlane, luma_chroma_to_rgb, luminance_weights_of, rgb_to_luma_chroma_with,
};
use crate::options::{ColourLayout, EncodeOptions};
use crate::part::{ExrPart, ExrPlane};
use crate::types::{Attribute, Box2i, Channel};

/// One decoded flat part at full resolution, normalised across the
/// single-part / multi-part / multi-level readers.
pub(crate) struct FlatPixels {
    pub(crate) data_window: Box2i,
    pub(crate) display_window: Box2i,
    /// Alphabetical, matching `planes`.
    pub(crate) channels: Vec<Channel>,
    pub(crate) planes: Vec<ExrPlane>,
    /// The part's header attributes (chromaticities, multiView, …).
    pub(crate) attributes: Vec<Attribute>,
}

impl From<ExrPart> for FlatPixels {
    fn from(p: ExrPart) -> Self {
        Self {
            data_window: p.data_window,
            display_window: p.display_window,
            channels: p.channels,
            planes: p.planes,
            attributes: p.attributes,
        }
    }
}

/// Which channels (indices into a channel list) form the view.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ViewPlan {
    Rgba {
        r: usize,
        g: usize,
        b: usize,
        a: usize,
    },
    Rgb {
        r: usize,
        g: usize,
        b: usize,
    },
    PartialRgb {
        r: Option<usize>,
        g: Option<usize>,
        b: Option<usize>,
        a: Option<usize>,
    },
    LumaChroma {
        y: Option<usize>,
        ry: Option<usize>,
        by: Option<usize>,
        a: Option<usize>,
    },
    GrayAlpha {
        y: usize,
        a: usize,
    },
    Gray {
        y: usize,
    },
}

impl ViewPlan {
    pub(crate) fn format(&self) -> PixelFormat {
        match self {
            Self::Rgba { .. } | Self::GrayAlpha { .. } => PixelFormat::RgbaF32Le,
            Self::LumaChroma { a: Some(_), .. } | Self::PartialRgb { a: Some(_), .. } => {
                PixelFormat::RgbaF32Le
            }
            Self::Rgb { .. }
            | Self::LumaChroma { a: None, .. }
            | Self::PartialRgb { a: None, .. } => PixelFormat::RgbF32Le,
            Self::Gray { .. } => PixelFormat::GrayF32Le,
        }
    }
}

/// Select the layer named `layer` from `channels` / `attributes`.
fn select_layer(channels: &[Channel], attributes: &[Attribute], layer: &str) -> Result<ExrLayer> {
    let layers = enumerate_layers(channels, attributes);
    match find_layer(&layers, layer) {
        Some(l) => Ok(l.clone()),
        None => {
            let available: Vec<String> = layers
                .iter()
                .map(|l| {
                    let name = if l.name.is_empty() {
                        "\"\""
                    } else {
                        l.name.as_str()
                    };
                    match &l.view {
                        Some(v) => format!("{name} ({:?}, view {v})", l.kind),
                        None => format!("{name} ({:?})", l.kind),
                    }
                })
                .collect();
            let msg = format!(
                "OpenEXR decoder: no layer '{layer}'; the part has [{}]",
                available.join(", ")
            );
            // A missing base layer means the part has no default colour
            // view (Unsupported); a missing named layer is a caller
            // error (InvalidData).
            if layer.is_empty() {
                Err(ExrError::unsupported(msg))
            } else {
                Err(ExrError::invalid(msg))
            }
        }
    }
}

/// Decide the view for a channel list (header only — module docs,
/// *Decode mapping*). `channels` may be in any order; indices refer to
/// it.
pub(crate) fn plan_view(
    channels: &[Channel],
    attributes: &[Attribute],
    layer: &str,
) -> Result<ViewPlan> {
    let sel = select_layer(channels, attributes, layer)?;
    let find = |base: &str| {
        let name = sel.channel_name(base);
        channels.iter().position(|c| c.name == name)
    };
    let (r, g, b, a, y) = (find("R"), find("G"), find("B"), find("A"), find("Y"));
    let (ry, by) = (find("RY"), find("BY"));
    let plan = match (r, g, b, a, y, ry, by) {
        (.., y, ry, by) if ry.is_some() || by.is_some() => ViewPlan::LumaChroma { y, ry, by, a },
        (_, _, _, Some(a), Some(y), ..) => ViewPlan::GrayAlpha { y, a },
        (_, _, _, None, Some(y), ..) => ViewPlan::Gray { y },
        (Some(r), Some(g), Some(b), Some(a), ..) => ViewPlan::Rgba { r, g, b, a },
        (Some(r), Some(g), Some(b), None, ..) => ViewPlan::Rgb { r, g, b },
        (r, g, b, a, ..) if r.is_some() || g.is_some() || b.is_some() => {
            ViewPlan::PartialRgb { r, g, b, a }
        }
        _ => {
            let names: Vec<&str> = sel
                .channels
                .iter()
                .map(|&i| channels[i].name.as_str())
                .collect();
            return Err(ExrError::unsupported(format!(
                "OpenEXR decoder: layer '{}' channel set [{}] has no RGB(A) / Y / Y RY BY view",
                sel.name,
                names.join(", ")
            )));
        }
    };
    // Directly-mapped channels must be full resolution.
    let direct: Vec<usize> = match &plan {
        ViewPlan::Rgba { r, g, b, a } => vec![*r, *g, *b, *a],
        ViewPlan::Rgb { r, g, b } => vec![*r, *g, *b],
        ViewPlan::PartialRgb { r, g, b, a } => {
            [r, g, b, a].into_iter().flatten().copied().collect()
        }
        ViewPlan::LumaChroma { y, a, .. } => [y, a].into_iter().flatten().copied().collect(),
        ViewPlan::GrayAlpha { y, a } => vec![*y, *a],
        ViewPlan::Gray { y } => vec![*y],
    };
    for idx in direct {
        let ch = &channels[idx];
        if ch.x_sampling != 1 || ch.y_sampling != 1 {
            return Err(ExrError::unsupported(format!(
                "OpenEXR decoder: colour channel '{}' is sub-sampled ({}x{}); the view is \
                 full-resolution",
                ch.name, ch.x_sampling, ch.y_sampling
            )));
        }
    }
    if let ViewPlan::LumaChroma { ry, by, .. } = &plan {
        for idx in [ry, by].into_iter().flatten().copied() {
            let ch = &channels[idx];
            if ch.x_sampling <= 0 || ch.y_sampling <= 0 {
                return Err(ExrError::invalid(format!(
                    "OpenEXR decoder: chroma channel '{}' declares sampling {}x{}",
                    ch.name, ch.x_sampling, ch.y_sampling
                )));
            }
        }
    }
    Ok(plan)
}

/// Build the packed view of a decoded flat part (module docs).
pub(crate) fn flat_to_image(img: &FlatPixels, layer: &str) -> Result<ExrImage> {
    let plan = plan_view(&img.channels, &img.attributes, layer)?;
    let w = img.data_window.width();
    let h = img.data_window.height();
    let pixels = (w as usize) * (h as usize);

    let full_res = |idx: usize| -> Result<&[f32]> {
        let samples = &img.planes[idx].samples;
        if samples.len() != pixels {
            return Err(ExrError::invalid(format!(
                "OpenEXR decoder: channel '{}' holds {} samples for {w}x{h}",
                img.channels[idx].name,
                samples.len()
            )));
        }
        Ok(samples.as_slice())
    };
    let chroma = |idx: usize| -> ChromaPlane<'_> {
        let ch = &img.channels[idx];
        ChromaPlane {
            samples: &img.planes[idx].samples,
            x_sampling: ch.x_sampling as u32,
            y_sampling: ch.y_sampling as u32,
        }
    };

    let converted;
    let zeros;
    let sources: Vec<&[f32]> = match &plan {
        ViewPlan::Rgba { r, g, b, a } => {
            vec![full_res(*r)?, full_res(*g)?, full_res(*b)?, full_res(*a)?]
        }
        ViewPlan::Rgb { r, g, b } => vec![full_res(*r)?, full_res(*g)?, full_res(*b)?],
        ViewPlan::PartialRgb { r, g, b, a } => {
            zeros = vec![0.0f32; pixels];
            let mut v = Vec::with_capacity(4);
            for c in [r, g, b] {
                v.push(match c {
                    Some(c) => full_res(*c)?,
                    None => zeros.as_slice(),
                });
            }
            if let Some(a) = a {
                v.push(full_res(*a)?);
            }
            v
        }
        ViewPlan::LumaChroma { y, ry, by, a } => {
            let weights = luminance_weights_of(&img.attributes);
            // Missing chroma reads as 0 (a 1×1 zero plane), a missing
            // luminance as 0.5.
            zeros = vec![0.0f32; pixels];
            let half_grey;
            let y_plane = match y {
                Some(y) => full_res(*y)?,
                None => {
                    half_grey = vec![0.5f32; pixels];
                    half_grey.as_slice()
                }
            };
            let pick = |c: &Option<usize>| match c {
                Some(c) => chroma(*c),
                None => ChromaPlane::new(&zeros, 1, 1),
            };
            converted = luma_chroma_to_rgb(w, h, y_plane, pick(ry), pick(by), weights)?;
            match a {
                Some(a) => vec![&converted.r, &converted.g, &converted.b, full_res(*a)?],
                None => vec![&converted.r, &converted.g, &converted.b],
            }
        }
        ViewPlan::GrayAlpha { y, a } => {
            let y = full_res(*y)?;
            vec![y, y, y, full_res(*a)?]
        }
        ViewPlan::Gray { y } => vec![full_res(*y)?],
    };

    let format = plan.format();
    let stride = (w as usize)
        .checked_mul(format.bytes_per_pixel())
        .ok_or_else(|| ExrError::unsupported(format!("OpenEXR decoder: {w}x{h} row overflows")))?;
    let total = stride.checked_mul(h as usize).ok_or_else(|| {
        ExrError::unsupported(format!("OpenEXR decoder: {w}x{h} plane overflows"))
    })?;
    let mut data = Vec::with_capacity(total);
    for px in 0..pixels {
        for src in &sources {
            data.extend_from_slice(&src[px].to_le_bytes());
        }
    }
    let mut out = ExrImage::new(w, h, format, vec![Plane::new(stride, data)])?;
    out.color = ColorInfo::from_attributes(&img.attributes);
    out.metadata = Metadata::default();
    out.data_window = img.data_window;
    out.display_window = img.display_window;
    out.attributes = ExrImage::non_structural(&img.attributes);
    Ok(out)
}

/// The channel list and `f32` planes (alphabetical, the order the depth
/// encoders require) that write `image` under `opts` (`colour`,
/// `chroma_sampling`, `layer`, `pixel_type`).
pub(crate) fn image_to_channels(
    image: &ExrImage,
    opts: &EncodeOptions,
) -> Result<(Vec<Channel>, Vec<Vec<f32>>)> {
    let width = image.width;
    let height = image.height;
    let comps = image.component_planes();
    let mk = |name: &str, sampling: u32| Channel {
        name: if opts.layer.is_empty() {
            name.to_string()
        } else {
            format!("{}.{name}", opts.layer)
        },
        pixel_type: opts.pixel_type,
        p_linear: false,
        x_sampling: sampling as i32,
        y_sampling: sampling as i32,
    };
    Ok(match (opts.colour, image.format) {
        (ColourLayout::LumaChroma, PixelFormat::RgbaF32Le | PixelFormat::RgbF32Le) => {
            let s = opts.chroma_sampling;
            if !width.is_multiple_of(s) || !height.is_multiple_of(s) {
                return Err(ExrError::invalid(format!(
                    "OpenEXR encoder: {width}x{height} image is not a multiple of \
                     chroma_sampling={s} (conforming readers require sub-sampled extents \
                     divisible by the sampling factor; use chroma_sampling=1)"
                )));
            }
            // Luminance weights follow the image's chromaticities (the
            // attribute is written back, so readers reconstruct with
            // the same weights); BT.709 when it has none.
            let weights = luminance_weights_of(&image.attributes);
            let yc = rgb_to_luma_chroma_with(
                width,
                height,
                [&comps[0], &comps[1], &comps[2]],
                weights,
                (s, s),
                opts.luma_chroma_rounding,
            )?;
            let mut chs = Vec::with_capacity(4);
            let mut pls = Vec::with_capacity(4);
            if let Some(a) = comps.get(3) {
                chs.push(mk("A", 1));
                pls.push(a.clone());
            }
            chs.extend([mk("BY", s), mk("RY", s), mk("Y", 1)]);
            pls.extend([yc.by, yc.ry, yc.y]);
            (chs, pls)
        }
        (_, PixelFormat::RgbaF32Le) => {
            let mut it = comps.into_iter();
            let (r, g, b, a) = (
                it.next().unwrap(),
                it.next().unwrap(),
                it.next().unwrap(),
                it.next().unwrap(),
            );
            (
                vec![mk("A", 1), mk("B", 1), mk("G", 1), mk("R", 1)],
                vec![a, b, g, r],
            )
        }
        (_, PixelFormat::RgbF32Le) => {
            let mut it = comps.into_iter();
            let (r, g, b) = (it.next().unwrap(), it.next().unwrap(), it.next().unwrap());
            (vec![mk("B", 1), mk("G", 1), mk("R", 1)], vec![b, g, r])
        }
        (_, PixelFormat::GrayF32Le) => (vec![mk("Y", 1)], comps),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PixelType;

    fn ch(name: &str, s: i32) -> Channel {
        Channel {
            name: name.to_string(),
            pixel_type: PixelType::Half,
            p_linear: false,
            x_sampling: s,
            y_sampling: s,
        }
    }

    #[test]
    fn plans_follow_the_documented_order() {
        let p = |names: &[&str]| {
            let chs: Vec<Channel> = names.iter().map(|n| ch(n, 1)).collect();
            plan_view(&chs, &[], "").map(|p| p.format())
        };
        assert_eq!(p(&["A", "B", "G", "R"]).unwrap(), PixelFormat::RgbaF32Le);
        assert_eq!(p(&["B", "G", "R", "Z"]).unwrap(), PixelFormat::RgbF32Le);
        assert_eq!(p(&["Y"]).unwrap(), PixelFormat::GrayF32Le);
        assert_eq!(p(&["A", "Y"]).unwrap(), PixelFormat::RgbaF32Le);
        assert_eq!(p(&["BY", "RY", "Y"]).unwrap(), PixelFormat::RgbF32Le);
        assert_eq!(p(&["A", "BY", "RY", "Y"]).unwrap(), PixelFormat::RgbaF32Le);
        assert!(matches!(p(&["Z"]), Err(ExrError::Unsupported(_))));
        assert!(matches!(p(&["A"]), Err(ExrError::Unsupported(_))));
        // Reference channel-presence rules: a partial RGB triple fills
        // the missing components with 0, Y + RY reads BY = 0, chroma
        // without Y reads Y = 0.5, and luminance wins over R/G/B.
        assert_eq!(p(&["G", "R"]).unwrap(), PixelFormat::RgbF32Le);
        assert_eq!(p(&["A", "R"]).unwrap(), PixelFormat::RgbaF32Le);
        assert_eq!(p(&["RY", "Y"]).unwrap(), PixelFormat::RgbF32Le);
        assert_eq!(p(&["BY", "RY"]).unwrap(), PixelFormat::RgbF32Le);
        assert_eq!(p(&["B", "G", "R", "Y"]).unwrap(), PixelFormat::GrayF32Le);
        assert_eq!(
            p(&["A", "B", "BY", "G", "R", "RY", "Y"]).unwrap(),
            PixelFormat::RgbaF32Le
        );
        // Sub-sampled colour channels are refused; chroma may be.
        let chs = vec![ch("B", 2), ch("G", 1), ch("R", 1)];
        assert!(matches!(
            plan_view(&chs, &[], ""),
            Err(ExrError::Unsupported(_))
        ));
        let chs = vec![ch("BY", 2), ch("RY", 2), ch("Y", 1)];
        assert_eq!(
            plan_view(&chs, &[], "").unwrap(),
            ViewPlan::LumaChroma {
                y: Some(2),
                ry: Some(1),
                by: Some(0),
                a: None
            }
        );
        // Layer selection.
        let chs: Vec<Channel> = ["diffuse.B", "diffuse.G", "diffuse.R", "Z"]
            .iter()
            .map(|n| ch(n, 1))
            .collect();
        assert_eq!(
            plan_view(&chs, &[], "diffuse").unwrap(),
            ViewPlan::Rgb { r: 2, g: 1, b: 0 }
        );
        assert!(matches!(
            plan_view(&chs, &[], ""),
            Err(ExrError::Unsupported(_))
        ));
        assert!(matches!(
            plan_view(&chs, &[], "nope"),
            Err(ExrError::InvalidData(_))
        ));
    }
}
