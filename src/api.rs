//! The root vocabulary of the image-crate API contract
//! (`IMAGE_CRATE_API`): `probe` / `info` / `decode*` / `encode*`.
//!
//! Every function here is framework-free (builds with
//! `default-features = false`) and is the single implementation the
//! registry `Decoder` / `Encoder` adapters call.

use std::io::{Read, Write};

use crate::decoder::{extract_required, parse_exr};
use crate::encoder::{encode_exr_scanline, merge_extra_attributes};
use crate::error::{ExrError, Result};
use crate::header::{parse_multipart_headers, ParsedHeader, VersionField};
use crate::image::{string_attribute, ExrImage, Frame, ImageInfo, RgbImage, RgbaImage};
use crate::mipmap_encoder::{
    build_box_filter_pyramid, build_box_filter_ripmap, encode_exr_tiled_mipmap_with_attributes,
    encode_exr_tiled_ripmap_with_attributes,
};
use crate::multipart_encoder::{encode_exr_multipart_with_attributes, MultipartScanlinePart};
use crate::multipart_mixed_encoder::{parse_exr_multipart_mixed, MultipartMixedImage};
use crate::options::{DecodeOptions, EncodeOptions, LevelMode};
use crate::tile_encoder::encode_exr_tiled_with_attributes;
use crate::types::{Attribute, AttributeValue, Box2i, Channel, EXR_MAGIC};
use crate::view::{flat_to_image, image_to_channels, plan_view, FlatPixels};

/// `true` when `bytes` starts with the OpenEXR magic `76 2F 31 01`.
/// Total, allocation-free, `false` on short input.
pub fn probe(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && bytes[..4] == EXR_MAGIC.to_le_bytes()
}

// ---------------------------------------------------------------------------
// Header plumbing
// ---------------------------------------------------------------------------

/// The version field and every part header of `bytes` (one for a
/// single-part file). Reads the header region only.
fn headers(bytes: &[u8]) -> Result<(VersionField, Vec<ParsedHeader>)> {
    if bytes.len() < 8 {
        return Err(ExrError::invalid(
            "OpenEXR: file shorter than the magic + version field",
        ));
    }
    let version = VersionField::from_u32(u32::from_le_bytes(bytes[4..8].try_into().unwrap()));
    let headers = if version.multipart {
        parse_multipart_headers(bytes)?
    } else {
        vec![crate::deep::parse_header_allow_deep(bytes)?]
    };
    if headers.is_empty() {
        return Err(ExrError::invalid("OpenEXR: file declares no parts"));
    }
    Ok((version, headers))
}

/// Resolve the part `opts` selects: by `part_name` when non-empty
/// (listing the file's names on a miss), else by `part` index.
fn select_part(headers: &[ParsedHeader], opts: &DecodeOptions) -> Result<usize> {
    if !opts.part_name.is_empty() {
        let names: Vec<Option<String>> = headers
            .iter()
            .map(|h| string_attribute(&h.attributes, "name"))
            .collect();
        if let Some(idx) = names
            .iter()
            .position(|n| n.as_deref() == Some(opts.part_name.as_str()))
        {
            return Ok(idx);
        }
        let listed: Vec<String> = names
            .iter()
            .enumerate()
            .map(|(i, n)| match n {
                Some(n) => format!("{i}: {n:?}"),
                None => format!("{i}: (unnamed)"),
            })
            .collect();
        return Err(ExrError::invalid(format!(
            "OpenEXR decoder: no part named {:?}; the file has [{}]",
            opts.part_name,
            listed.join(", ")
        )));
    }
    let idx = opts.part as usize;
    if idx >= headers.len() {
        return Err(ExrError::invalid(if headers.len() == 1 {
            format!(
                "OpenEXR decoder: part {} requested from a single-part file",
                opts.part
            )
        } else {
            format!(
                "OpenEXR decoder: part {} requested but the file has {} part(s)",
                opts.part,
                headers.len()
            )
        }));
    }
    Ok(idx)
}

/// `(tiled, deep)` of one part, from the version field (single-part
/// files) or its `type` attribute (multi-part files).
fn part_shape(version: VersionField, header: &ParsedHeader) -> (bool, bool) {
    let has_tiles = header.attributes.iter().any(|a| a.name == "tiles");
    match string_attribute(&header.attributes, "type").as_deref() {
        Some("tiledimage") => (true, false),
        Some("deeptile") => (true, true),
        Some("deepscanline") => (false, true),
        Some("scanlineimage") => (false, false),
        _ if version.multipart => (has_tiles, false),
        _ => (version.single_tile || has_tiles, version.non_image),
    }
}

/// Geometry of one part as the limits see it: `(width, height, bytes of
/// `f32` planes the decoder allocates)`.
fn part_geometry(channels: &[Channel], data_window: Box2i) -> Result<(u32, u32, u64)> {
    let width = data_window.width();
    let height = data_window.height();
    if width == 0 || height == 0 {
        return Err(ExrError::invalid(format!(
            "OpenEXR: dataWindow width={width} height={height} — must both be > 0"
        )));
    }
    // Upper bound: every channel at full resolution, 4 bytes a sample.
    let bytes = u64::from(width)
        .saturating_mul(u64::from(height))
        .saturating_mul(channels.len().max(1) as u64)
        .saturating_mul(4);
    Ok((width, height, bytes))
}

/// `strict` rule: a sub-sampled channel's data window must be aligned
/// to and divisible by its sampling factors.
fn check_sampling_alignment(channels: &[Channel], data_window: Box2i) -> Result<()> {
    for ch in channels {
        let (xs, ys) = (ch.x_sampling, ch.y_sampling);
        if xs <= 0 || ys <= 0 {
            return Err(ExrError::invalid(format!(
                "OpenEXR: channel '{}' has non-positive sampling factor x={xs} y={ys}",
                ch.name
            )));
        }
        let (w, h) = (data_window.width() as i64, data_window.height() as i64);
        if xs > 1 && (data_window.x_min.rem_euclid(xs) != 0 || w % xs as i64 != 0)
            || ys > 1 && (data_window.y_min.rem_euclid(ys) != 0 || h % ys as i64 != 0)
        {
            return Err(ExrError::invalid(format!(
                "OpenEXR (strict): channel '{}' sampling {xs}x{ys} does not divide the data \
                 window {}..={} x {}..={}",
                ch.name, data_window.x_min, data_window.x_max, data_window.y_min, data_window.y_max
            )));
        }
    }
    Ok(())
}

/// Validate every part's geometry against `opts` (all parts of a
/// multi-part file are decoded together, so the byte budget covers
/// their sum) and the strict rules; returns the selected part's plan
/// errors early so nothing is allocated for an unviewable part.
fn check_limits(
    headers: &[ParsedHeader],
    opts: &DecodeOptions,
    selected: Option<usize>,
) -> Result<()> {
    let mut total: u64 = 0;
    for (i, h) in headers.iter().enumerate() {
        let req = extract_required(&h.attributes)?;
        let (w, hgt, bytes) = part_geometry(&req.channels, req.data_window)?;
        if selected.is_none() || selected == Some(i) {
            opts.check(w, hgt, bytes)?;
            if opts.strict {
                check_sampling_alignment(&req.channels, req.data_window)?;
            }
        }
        total = total.saturating_add(bytes);
    }
    if let Some(m) = opts.max_bytes {
        if total > m {
            return Err(ExrError::limit(format!(
                "OpenEXR: decoded planes of all {} parts ({total} bytes) exceed max_bytes {m}",
                headers.len()
            )));
        }
    }
    Ok(())
}

/// Decode part `idx` of `bytes` as flat pixels (level `(0, 0)` of a
/// multi-level part). Deep parts are [`ExrError::Unsupported`].
fn decode_flat(bytes: &[u8], version: VersionField, idx: usize) -> Result<FlatPixels> {
    if !version.multipart {
        if version.non_image {
            return Err(ExrError::unsupported(
                "OpenEXR decoder: deep image (variable samples per pixel) has no colour view; \
                 use the parse_exr_deep_* API",
            ));
        }
        return Ok(parse_exr(bytes)?.into());
    }
    let mut parts = parse_exr_multipart_mixed(bytes)?;
    if idx >= parts.len() {
        return Err(ExrError::invalid(format!(
            "OpenEXR decoder: part {idx} requested but the file has {} part(s)",
            parts.len()
        )));
    }
    mixed_to_flat(parts.swap_remove(idx), idx)
}

/// Normalise one decoded multi-part part to flat pixels.
fn mixed_to_flat(part: MultipartMixedImage, idx: usize) -> Result<FlatPixels> {
    match part {
        MultipartMixedImage::Scanline(img) | MultipartMixedImage::Tiled(img) => Ok(img.into()),
        MultipartMixedImage::TiledMipmap(p) | MultipartMixedImage::TiledRipmap(p) => {
            let level = p
                .levels
                .into_iter()
                .find(|l| l.level_x == 0 && l.level_y == 0)
                .ok_or_else(|| {
                    ExrError::invalid(format!(
                        "OpenEXR decoder: multi-level part {idx} has no level (0, 0)"
                    ))
                })?;
            Ok(FlatPixels {
                data_window: p.data_window,
                display_window: p.display_window,
                channels: p.channels,
                planes: level.planes,
                attributes: p.attributes,
            })
        }
        MultipartMixedImage::DeepScanline(_)
        | MultipartMixedImage::DeepTiled(_)
        | MultipartMixedImage::DeepTiledMipmap(_)
        | MultipartMixedImage::DeepTiledRipmap(_) => Err(ExrError::unsupported(format!(
            "OpenEXR decoder: part {idx} is a deep part (variable samples per pixel) with no \
             colour view; use the parse_exr_deep_* API"
        ))),
    }
}

// ---------------------------------------------------------------------------
// info / decode
// ---------------------------------------------------------------------------

/// Header only: the first part's data-window geometry, its colour-view
/// [`crate::PixelFormat`], the part count (`frames`), alpha, colour
/// derived from `chromaticities`, the channel list, compression and the
/// tiled / deep / multi-part flags. Reads the header region and
/// nothing else; the chunk data may be missing or truncated.
///
/// Errors: [`ExrError::InvalidData`] for a bad magic, a malformed
/// attribute table or a missing required attribute;
/// [`ExrError::Unsupported`] when the first part's channel set has no
/// RGB(A) / `Y` / `Y RY BY` view (inspect it with
/// [`crate::parse_header`] instead).
pub fn info(bytes: &[u8]) -> Result<ImageInfo> {
    let (version, headers) = headers(bytes)?;
    let hdr = &headers[0];
    let req = extract_required(&hdr.attributes)?;
    let plan = plan_view(&req.channels, &hdr.attributes, "")?;
    let (tiled, deep) = part_shape(version, hdr);
    Ok(ImageInfo::new(
        req.data_window,
        req.display_window,
        plan.format(),
        headers.len() as u32,
        req.channels,
        req.compression,
        tiled,
        deep,
        version.multipart,
        &hdr.attributes,
    ))
}

/// Decode the first part's colour view with [`DecodeOptions::default`]
/// (65 535 × 65 535, 1 GiB of planes, lenient, part 0, base layer).
pub fn decode(bytes: &[u8]) -> Result<ExrImage> {
    decode_with(bytes, &DecodeOptions::default())
}

/// [`decode`] with explicit limits, strictness and part / layer
/// selection. Limits are checked against the header before any plane is
/// allocated ([`ExrError::LimitExceeded`]); a deep part or a channel set
/// without a colour view is [`ExrError::Unsupported`].
pub fn decode_with(bytes: &[u8], opts: &DecodeOptions) -> Result<ExrImage> {
    let (version, headers) = headers(bytes)?;
    let idx = select_part(&headers, opts)?;
    let hdr = &headers[idx];
    let (_, deep) = part_shape(version, hdr);
    if deep {
        return Err(ExrError::unsupported(format!(
            "OpenEXR decoder: part {idx} is a deep part (variable samples per pixel) with no \
             colour view; use the parse_exr_deep_* API"
        )));
    }
    let req = extract_required(&hdr.attributes)?;
    // Plan before decoding so an unviewable channel set costs nothing.
    plan_view(&req.channels, &hdr.attributes, &opts.layer)?;
    check_limits(&headers, opts, Some(idx))?;
    let flat = decode_flat(bytes, version, idx)?;
    flat_to_image(&flat, &opts.layer)
}

/// Decode straight to tightly packed 8-bit RGB ([`ExrImage::to_rgb8`]:
/// clamp `[0, 1]` × 255, no exposure / tone curve), default options.
pub fn decode_rgb8(bytes: &[u8]) -> Result<RgbImage> {
    let img = decode(bytes)?;
    Ok(RgbImage::new(img.width, img.height, img.to_rgb8()))
}

/// Decode straight to tightly packed 8-bit RGBA (alpha from the `A`
/// channel, else `255`), default options.
pub fn decode_rgba8(bytes: &[u8]) -> Result<RgbaImage> {
    let img = decode(bytes)?;
    Ok(RgbaImage::new(img.width, img.height, img.to_rgba8()))
}

/// Every part's colour view as a [`Frame`] (`index`, `name`, `part_type`
/// from the part header; `delay` always `None`) with
/// [`DecodeOptions::default`]. A single-part file yields one frame.
///
/// Parts without a colour view (deep parts, depth / AOV-only channel
/// sets) are **skipped** — their indices are missing from the result —
/// unless `strict` is set, which makes them [`ExrError::Unsupported`].
/// A file in which no part has a view is [`ExrError::Unsupported`].
pub fn decode_all(bytes: &[u8]) -> Result<Vec<Frame>> {
    decode_all_with(bytes, &DecodeOptions::default())
}

/// [`decode_all`] with explicit options (`part` / `part_name` are
/// ignored; `layer` applies to every part).
pub fn decode_all_with(bytes: &[u8], opts: &DecodeOptions) -> Result<Vec<Frame>> {
    let (version, headers) = headers(bytes)?;
    check_limits(&headers, opts, None)?;
    let frame = |img: ExrImage, idx: usize, attrs: &[Attribute]| {
        Frame::new(img, idx as u32)
            .with_name(string_attribute(attrs, "name"))
            .with_part_type(string_attribute(attrs, "type"))
    };
    if !version.multipart {
        let hdr = &headers[0];
        let req = extract_required(&hdr.attributes)?;
        plan_view(&req.channels, &hdr.attributes, &opts.layer)?;
        let flat = decode_flat(bytes, version, 0)?;
        let img = flat_to_image(&flat, &opts.layer)?;
        return Ok(vec![frame(img, 0, &hdr.attributes)]);
    }
    let parts = parse_exr_multipart_mixed(bytes)?;
    let mut out = Vec::with_capacity(parts.len());
    let mut skipped: Vec<String> = Vec::new();
    for (idx, part) in parts.into_iter().enumerate() {
        let viewed = mixed_to_flat(part, idx).and_then(|flat| {
            let img = flat_to_image(&flat, &opts.layer)?;
            Ok((img, flat.attributes))
        });
        match viewed {
            Ok((img, attrs)) => out.push(frame(img, idx, &attrs)),
            Err(ExrError::Unsupported(msg)) if !opts.strict => skipped.push(msg),
            Err(e) => return Err(e),
        }
    }
    if out.is_empty() {
        return Err(ExrError::unsupported(format!(
            "OpenEXR decoder: no part has a colour view: {}",
            skipped.join("; ")
        )));
    }
    Ok(out)
}

/// Read `r` to end and [`decode`] it. Read failures surface as
/// [`ExrError::Io`].
pub fn decode_from<R: Read>(mut r: R) -> Result<ExrImage> {
    let mut buf = Vec::new();
    r.read_to_end(&mut buf)?;
    decode(&buf)
}

// ---------------------------------------------------------------------------
// encode
// ---------------------------------------------------------------------------

/// The structural attributes of a scanline part, in the canonical order,
/// followed by the image's extras ([`merge_extra_attributes`]).
fn scanline_attributes(
    image: &ExrImage,
    channels: &[Channel],
    opts: &EncodeOptions,
    data_window: Box2i,
    display_window: Box2i,
) -> Vec<Attribute> {
    let mut attrs = vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(channels.to_vec()),
        },
        Attribute {
            name: "compression".to_string(),
            value: AttributeValue::Compression(opts.compression),
        },
        Attribute {
            name: "dataWindow".to_string(),
            value: AttributeValue::Box2i(data_window),
        },
        Attribute {
            name: "displayWindow".to_string(),
            value: AttributeValue::Box2i(display_window),
        },
        Attribute {
            name: "lineOrder".to_string(),
            value: AttributeValue::LineOrder(opts.line_order),
        },
    ];
    merge_extra_attributes(&mut attrs, &image.attributes);
    // The spec's three required viewing attributes, when the image did
    // not carry them.
    let missing: Vec<Attribute> = ExrImage::default_attributes()
        .into_iter()
        .filter(|d| !attrs.iter().any(|a| a.name == d.name))
        .collect();
    merge_extra_attributes(&mut attrs, &missing);
    attrs
}

/// The data window `encode` writes for `image` under `opts`
/// (override or the image's), validated against the image extents.
fn effective_windows(image: &ExrImage, opts: &EncodeOptions) -> Result<(Box2i, Box2i)> {
    let data_window = opts.data_window.unwrap_or(image.data_window);
    if data_window.width() != image.width || data_window.height() != image.height {
        return Err(ExrError::invalid(format!(
            "OpenEXR encoder: dataWindow {}x{} does not match the {}x{} image",
            data_window.width(),
            data_window.height(),
            image.width,
            image.height
        )));
    }
    Ok((
        data_window,
        opts.display_window.unwrap_or(image.display_window),
    ))
}

/// Encode `image` as a single-part OpenEXR file: its one packed float
/// plane becomes `R G B (A)` / `Y` channels (or `Y RY BY (A)` under
/// [`crate::ColourLayout::LumaChroma`]) of the option's pixel type and
/// compression, in a scanline file (default) or a tiled one
/// (`tile_size > 0`, with `levels`). The image's non-structural header
/// attributes (`chromaticities`, `pixelAspectRatio`, `owner`, …) and its
/// data / display windows are written back, so `decode(encode(img)) ==
/// img` for `Float` + a lossless compression.
///
/// Every native layout encodes as given (a plane with row padding is
/// repacked); nothing is converted silently. [`ExrError::Unsupported`]
/// for UINT output, and for tiled output with a data window off the
/// origin; [`ExrError::InvalidData`] for inconsistent options (see
/// [`EncodeOptions`]). ICC / Exif / XMP metadata cannot be carried and is
/// ignored.
pub fn encode(image: &ExrImage, opts: &EncodeOptions) -> Result<Vec<u8>> {
    opts.validate()?;
    let (data_window, display_window) = effective_windows(image, opts)?;
    let (channels, planes) = image_to_channels(image, opts)?;
    let plane_refs: Vec<&[f32]> = planes.iter().map(|p| p.as_slice()).collect();
    let (width, height) = (image.width, image.height);
    let tile = opts.tile_size;
    if tile == 0 {
        let attrs = scanline_attributes(image, &channels, opts, data_window, display_window);
        return encode_exr_scanline(
            width,
            height,
            &channels,
            &plane_refs,
            opts.compression,
            attrs,
        );
    }
    let origin = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (width - 1) as i32,
        y_max: (height - 1) as i32,
    };
    if data_window != origin || display_window != origin {
        return Err(ExrError::unsupported(
            "OpenEXR encoder: tiled output needs the data and display windows at the origin \
             ([0, width) x [0, height)); write a scanline file (tile_size = 0) to keep the \
             window position",
        ));
    }
    if channels
        .iter()
        .any(|c| c.x_sampling != 1 || c.y_sampling != 1)
    {
        return Err(ExrError::invalid(
            "OpenEXR encoder: tiled files need full-resolution channels; use chroma_sampling=1 \
             with colour=luma_chroma",
        ));
    }
    let extra = &image.attributes;
    match opts.levels {
        LevelMode::One => encode_exr_tiled_with_attributes(
            width,
            height,
            &channels,
            &plane_refs,
            opts.compression,
            tile,
            tile,
            opts.line_order,
            extra,
        ),
        LevelMode::Mipmap => {
            let pyramid = build_box_filter_pyramid(width, height, &planes);
            encode_exr_tiled_mipmap_with_attributes(
                &channels,
                &pyramid,
                opts.compression,
                tile,
                tile,
                opts.line_order,
                extra,
            )
        }
        LevelMode::Ripmap => {
            let pyramid = build_box_filter_ripmap(width, height, &planes);
            encode_exr_tiled_ripmap_with_attributes(
                &channels,
                &pyramid,
                opts.compression,
                tile,
                tile,
                opts.line_order,
                extra,
            )
        }
    }
}

/// Encode tightly packed 8-bit RGB (`3 × width × height` bytes) as an
/// `RgbF32Le` OpenEXR file: each byte `b` becomes the linear float
/// `b / 255` — or `(b / 255) ^ g` with [`EncodeOptions::input_gamma`] =
/// `Some(g)` — and the image is written like any float image (`B G R`
/// channels, BT.709 default primaries). A short buffer is
/// [`ExrError::InvalidData`].
pub fn encode_rgb8(width: u32, height: u32, rgb: &[u8], opts: &EncodeOptions) -> Result<Vec<u8>> {
    let img = ExrImage::from_8bit(
        width,
        height,
        rgb,
        crate::image::PixelFormat::RgbF32Le,
        opts.input_gamma,
    )?;
    encode(&img, opts)
}

/// Encode tightly packed 8-bit RGBA (`4 × width × height` bytes) as an
/// `RgbaF32Le` OpenEXR file (`A B G R` channels); alpha is kept, as
/// `a / 255` (never gamma-adjusted).
pub fn encode_rgba8(width: u32, height: u32, rgba: &[u8], opts: &EncodeOptions) -> Result<Vec<u8>> {
    let img = ExrImage::from_8bit(
        width,
        height,
        rgba,
        crate::image::PixelFormat::RgbaF32Le,
        opts.input_gamma,
    )?;
    encode(&img, opts)
}

/// [`encode`] into a writer. Write failures surface as [`ExrError::Io`].
pub fn encode_to<W: Write>(image: &ExrImage, opts: &EncodeOptions, mut w: W) -> Result<()> {
    let bytes = encode(image, opts)?;
    w.write_all(&bytes)?;
    Ok(())
}

/// The mirror of [`decode_all`]: write `frames` as one multi-part
/// scanline file, one part per frame in order. Each part is named by
/// its frame's `name` (`part<index>` when `None`; names must be unique),
/// carries the frame image's non-structural attributes, and is written
/// with the option's pixel type, compression and channel layout
/// (`tile_size`, `levels`, `line_order` and the window overrides are
/// not available for multi-part output — [`ExrError::Unsupported`] when
/// set). Part data windows sit at the origin; the shared display window
/// is the largest part extent. An empty slice is
/// [`ExrError::InvalidData`].
pub fn encode_all(frames: &[Frame], opts: &EncodeOptions) -> Result<Vec<u8>> {
    opts.validate()?;
    if frames.is_empty() {
        return Err(ExrError::invalid(
            "OpenEXR encoder: encode_all needs at least one frame",
        ));
    }
    if opts.tile_size != 0 || opts.line_order != crate::types::LineOrder::IncreasingY {
        return Err(ExrError::unsupported(
            "OpenEXR encoder: multi-part output is scanline INCREASING_Y only (tile_size = 0)",
        ));
    }
    if opts.data_window.is_some() || opts.display_window.is_some() {
        return Err(ExrError::unsupported(
            "OpenEXR encoder: multi-part output places every part at the origin (no window \
             overrides)",
        ));
    }
    let mut planes_per_part = Vec::with_capacity(frames.len());
    let mut channels_per_part = Vec::with_capacity(frames.len());
    let mut extras = Vec::with_capacity(frames.len());
    for f in frames {
        let (channels, planes) = image_to_channels(&f.image, opts)?;
        channels_per_part.push(channels);
        planes_per_part.push(planes);
        extras.push(f.image.attributes.clone());
    }
    let parts: Vec<MultipartScanlinePart> = frames
        .iter()
        .enumerate()
        .map(|(i, f)| MultipartScanlinePart {
            name: f.name.clone().unwrap_or_else(|| format!("part{}", f.index)),
            width: f.image.width,
            height: f.image.height,
            channels: channels_per_part[i].clone(),
            planes: planes_per_part[i].iter().map(|p| p.as_slice()).collect(),
            compression: opts.compression,
        })
        .collect();
    encode_exr_multipart_with_attributes(&parts, &extras)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::parse_header;
    use crate::image::{ColorInfo, PixelFormat, Plane};
    use crate::options::ColourLayout;
    use crate::types::{Compression, PixelType};

    fn ramp(w: u32, h: u32, format: PixelFormat) -> ExrImage {
        let comps = format.components();
        let samples: Vec<f32> = (0..(w * h) as usize * comps)
            .map(|i| (i as f32 * 0.37 + (i % comps) as f32) * 0.01 - 0.5)
            .collect();
        ExrImage::from_f32(w, h, format, &samples).unwrap()
    }

    #[test]
    fn probe_is_total_and_magic_only() {
        assert!(!probe(b""));
        assert!(!probe(b"\x76\x2f\x31"));
        assert!(probe(b"\x76\x2f\x31\x01"));
        assert!(!probe(b"\x89PNG\r\n\x1a\n"));
        let bytes = encode(
            &ramp(4, 2, PixelFormat::RgbF32Le),
            &EncodeOptions::default(),
        )
        .unwrap();
        assert!(probe(&bytes));
    }

    #[test]
    fn info_reads_the_header_only() {
        let img = ramp(7, 3, PixelFormat::RgbaF32Le);
        let bytes = encode(
            &img,
            &EncodeOptions::default().with_compression(Compression::Piz),
        )
        .unwrap();
        let hdr_end = parse_header(&bytes).unwrap().end_offset;
        let i = info(&bytes[..hdr_end]).unwrap();
        assert_eq!((i.width, i.height), (7, 3));
        assert_eq!(i.format, PixelFormat::RgbaF32Le);
        assert_eq!(i.frames, 1);
        assert!(i.has_alpha);
        assert!(!i.has_icc && !i.has_exif && !i.has_xmp);
        assert_eq!(i.color, ColorInfo::exr_default());
        assert_eq!(i.compression, Compression::Piz);
        assert_eq!(i.channels.len(), 4);
        assert!(!i.tiled && !i.deep && !i.multipart);
        assert!(i.part_name.is_none());
        // Tiled files report the flag.
        let bytes = encode(&img, &EncodeOptions::default().with_tile_size(4)).unwrap();
        assert!(info(&bytes).unwrap().tiled);
        // Short / foreign input is InvalidData.
        assert!(matches!(info(b"PNG"), Err(ExrError::InvalidData(_))));
        assert!(matches!(
            info(b"\x76\x2f\x31\x01\x02\x00\x00\x00"),
            Err(ExrError::InvalidData(_))
        ));
    }

    #[test]
    fn lossless_round_trip_is_exact() {
        for format in [
            PixelFormat::GrayF32Le,
            PixelFormat::RgbF32Le,
            PixelFormat::RgbaF32Le,
        ] {
            let img = ramp(9, 5, format);
            for compression in [
                Compression::None,
                Compression::Rle,
                Compression::Zips,
                Compression::Zip,
                Compression::Piz,
            ] {
                let opts = EncodeOptions::default().with_compression(compression);
                let back = decode(&encode(&img, &opts).unwrap()).unwrap();
                assert_eq!(back, img, "{format:?} {compression:?}");
                // Tiled too (ONE_LEVEL / MIPMAP / RIPMAP all decode level 0).
                for levels in [LevelMode::One, LevelMode::Mipmap, LevelMode::Ripmap] {
                    let opts = opts.clone().with_tile_size(4).with_levels(levels);
                    let back = decode(&encode(&img, &opts).unwrap()).unwrap();
                    assert_eq!(back, img, "{format:?} {compression:?} tiled {levels:?}");
                }
            }
            // A decoded image re-encodes byte-identically.
            let bytes = encode(&img, &EncodeOptions::default()).unwrap();
            let back = decode(&bytes).unwrap();
            assert_eq!(encode(&back, &EncodeOptions::default()).unwrap(), bytes);
        }
    }

    #[test]
    fn half_output_rounds_to_binary16() {
        let img = ramp(6, 2, PixelFormat::RgbF32Le);
        let opts = EncodeOptions::default().with_pixel_type(PixelType::Half);
        let back = decode(&encode(&img, &opts).unwrap()).unwrap();
        for (a, b) in back.pixels().iter().zip(img.pixels()) {
            let want = crate::half::half_to_f32(crate::half::f32_to_half(b));
            assert_eq!(*a, want);
        }
        // Lossy schemes decode to the right shape.
        for c in [
            Compression::Pxr24,
            Compression::B44,
            Compression::B44a,
            Compression::Dwaa,
            Compression::Dwab,
        ] {
            let back = decode(&encode(&img, &opts.clone().with_compression(c)).unwrap()).unwrap();
            assert_eq!(back.format, PixelFormat::RgbF32Le);
            assert_eq!((back.width, back.height), (6, 2));
        }
    }

    #[test]
    fn windows_and_attributes_round_trip() {
        let img = ramp(5, 4, PixelFormat::RgbF32Le)
            .with_data_window(Box2i {
                x_min: -2,
                y_min: 10,
                x_max: 2,
                y_max: 13,
            })
            .unwrap()
            .with_display_window(Box2i {
                x_min: 0,
                y_min: 0,
                x_max: 99,
                y_max: 49,
            })
            .with_chromaticities(crate::image::ColorInfo::chromaticities_for(9).unwrap());
        let mut img = img;
        img.attributes.push(Attribute {
            name: "owner".to_string(),
            value: AttributeValue::String("oxideav test".to_string()),
        });
        let bytes = encode(&img, &EncodeOptions::default()).unwrap();
        let back = decode(&bytes).unwrap();
        assert_eq!(back, img);
        assert_eq!(back.color.primaries, 9);
        assert_eq!(
            back.string_attribute("owner").as_deref(),
            Some("oxideav test")
        );
        let i = info(&bytes).unwrap();
        assert_eq!(i.data_window, img.data_window);
        assert_eq!(i.display_window, img.display_window);
        assert_eq!(i.color.primaries, 9);
        // Option overrides win over the image's windows.
        let moved = decode(
            &encode(
                &img,
                &EncodeOptions::default().with_data_window(Box2i {
                    x_min: 0,
                    y_min: 0,
                    x_max: 4,
                    y_max: 3,
                }),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(moved.data_window.y_min, 0);
        assert!(matches!(
            encode(
                &img,
                &EncodeOptions::default().with_data_window(Box2i {
                    x_min: 0,
                    y_min: 0,
                    x_max: 1,
                    y_max: 1,
                })
            ),
            Err(ExrError::InvalidData(_))
        ));
        // Tiled output refuses an off-origin window but keeps extras.
        assert!(matches!(
            encode(&img, &EncodeOptions::default().with_tile_size(4)),
            Err(ExrError::Unsupported(_))
        ));
        let origin = ramp(5, 4, PixelFormat::RgbF32Le)
            .with_chromaticities(crate::image::ColorInfo::chromaticities_for(12).unwrap());
        for levels in [LevelMode::One, LevelMode::Mipmap, LevelMode::Ripmap] {
            let back = decode(
                &encode(
                    &origin,
                    &EncodeOptions::default()
                        .with_tile_size(4)
                        .with_levels(levels),
                )
                .unwrap(),
            )
            .unwrap();
            assert_eq!(back, origin, "{levels:?}");
            assert_eq!(back.color.primaries, 12);
        }
    }

    #[test]
    fn luma_chroma_layout_round_trips_at_full_chroma() {
        let img = ramp(8, 4, PixelFormat::RgbaF32Le);
        let opts = EncodeOptions::default()
            .with_colour(ColourLayout::LumaChroma)
            .with_chroma_sampling(1)
            .with_compression(Compression::None);
        let bytes = encode(&img, &opts).unwrap();
        let i = info(&bytes).unwrap();
        let names: Vec<&str> = i.channels.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["A", "BY", "RY", "Y"]);
        assert_eq!(i.format, PixelFormat::RgbaF32Le);
        let back = decode(&bytes).unwrap();
        assert_eq!(back.format, PixelFormat::RgbaF32Le);
        for (a, b) in back.pixels().iter().zip(img.pixels()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        // 2x2 chroma needs even extents.
        assert!(matches!(
            encode(
                &ramp(7, 4, PixelFormat::RgbF32Le),
                &EncodeOptions::default().with_colour(ColourLayout::LumaChroma)
            ),
            Err(ExrError::InvalidData(_))
        ));
        // Gray writes Y under either layout.
        let g = ramp(4, 4, PixelFormat::GrayF32Le);
        let bytes = encode(
            &g,
            &EncodeOptions::default().with_colour(ColourLayout::LumaChroma),
        )
        .unwrap();
        assert_eq!(info(&bytes).unwrap().format, PixelFormat::GrayF32Le);
        assert_eq!(decode(&bytes).unwrap(), g);
    }

    #[test]
    fn layer_prefix_and_selection() {
        let img = ramp(4, 2, PixelFormat::RgbF32Le);
        let bytes = encode(&img, &EncodeOptions::default().with_layer("diffuse")).unwrap();
        let names: Vec<String> = info(&bytes)
            .map(|i| i.channels.iter().map(|c| c.name.clone()).collect())
            .unwrap_or_default();
        assert!(names.is_empty(), "base layer has no view: {names:?}");
        assert!(matches!(info(&bytes), Err(ExrError::Unsupported(_))));
        assert!(matches!(decode(&bytes), Err(ExrError::Unsupported(_))));
        let back = decode_with(&bytes, &DecodeOptions::default().with_layer("diffuse")).unwrap();
        assert_eq!(back, img);
        assert!(matches!(
            decode_with(&bytes, &DecodeOptions::default().with_layer("spec")),
            Err(ExrError::InvalidData(_))
        ));
    }

    #[test]
    fn rgb8_and_rgba8_raw_paths() {
        let flat = [0u8, 255, 128, 0, 255, 128];
        let bytes = encode_rgb8(2, 1, &flat, &EncodeOptions::default()).unwrap();
        let img = decode(&bytes).unwrap();
        assert_eq!(img.format, PixelFormat::RgbF32Le);
        assert_eq!(img.pixel(0, 0), vec![0.0, 1.0, 128.0 / 255.0]);
        let out = decode_rgb8(&bytes).unwrap();
        assert_eq!((out.width, out.height), (2, 1));
        assert_eq!(out.as_bytes(), &flat);
        let rgba = decode_rgba8(&bytes).unwrap();
        assert_eq!(rgba.into_raw(), vec![0, 255, 128, 255, 0, 255, 128, 255]);
        // RGBA keeps alpha.
        let rgba_in = [10u8, 20, 30, 7, 40, 50, 60, 0];
        let bytes = encode_rgba8(2, 1, &rgba_in, &EncodeOptions::default()).unwrap();
        assert_eq!(info(&bytes).unwrap().format, PixelFormat::RgbaF32Le);
        assert_eq!(decode_rgba8(&bytes).unwrap().as_bytes(), &rgba_in);
        assert_eq!(
            decode_rgb8(&bytes).unwrap().as_bytes(),
            &[10, 20, 30, 40, 50, 60]
        );
        assert!(matches!(
            encode_rgb8(2, 2, &flat, &EncodeOptions::default()),
            Err(ExrError::InvalidData(_))
        ));
        // input_gamma linearises colour, not alpha.
        let bytes = encode_rgba8(
            1,
            1,
            &[128, 128, 128, 128],
            &EncodeOptions::default().with_input_gamma(2.2),
        )
        .unwrap();
        let px = decode(&bytes).unwrap().pixel(0, 0);
        assert!((px[0] - (128.0f32 / 255.0).powf(2.2)).abs() < 1e-6);
        assert!((px[3] - 128.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn limits_fire_before_decoding() {
        let img = ramp(16, 8, PixelFormat::RgbaF32Le);
        let bytes = encode(&img, &EncodeOptions::default()).unwrap();
        let small = |o: DecodeOptions| decode_with(&bytes, &o);
        assert!(matches!(
            small(DecodeOptions::default().with_max_width(15u32)),
            Err(ExrError::LimitExceeded(_))
        ));
        assert!(matches!(
            small(DecodeOptions::default().with_max_height(7u32)),
            Err(ExrError::LimitExceeded(_))
        ));
        assert!(matches!(
            small(DecodeOptions::default().with_max_pixels(127u64)),
            Err(ExrError::LimitExceeded(_))
        ));
        assert!(matches!(
            small(DecodeOptions::default().with_max_bytes(16 * 8 * 16 - 1)),
            Err(ExrError::LimitExceeded(_))
        ));
        assert!(small(DecodeOptions::default().with_max_bytes(16 * 8 * 16)).is_ok());
        // A hostile header with a gigantic window fails on the limit, not
        // on allocation.
        let hdr_end = parse_header(&bytes).unwrap().end_offset;
        let mut hostile = bytes[..hdr_end].to_vec();
        let pos = hostile
            .windows(10)
            .position(|w| w == b"dataWindow")
            .unwrap();
        // name NUL type("box2i") NUL size(4) then x_min y_min x_max y_max.
        let val = pos + "dataWindow".len() + 1 + "box2i".len() + 1 + 4;
        hostile[val + 8..val + 12].copy_from_slice(&(60_000i32).to_le_bytes());
        hostile[val + 12..val + 16].copy_from_slice(&(60_000i32).to_le_bytes());
        assert!(matches!(decode(&hostile), Err(ExrError::LimitExceeded(_))));
        let i = info(&hostile).unwrap();
        assert_eq!((i.width, i.height), (60_001, 60_001));
    }

    #[test]
    fn decode_from_and_encode_to_stream() {
        let img = ramp(3, 3, PixelFormat::GrayF32Le);
        let mut buf = Vec::new();
        encode_to(&img, &EncodeOptions::default(), &mut buf).unwrap();
        let back = decode_from(std::io::Cursor::new(&buf)).unwrap();
        assert_eq!(back, img);
        struct Failing;
        impl Read for Failing {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("nope"))
            }
        }
        assert!(matches!(decode_from(Failing), Err(ExrError::Io(_))));
    }

    #[test]
    fn multipart_decode_all_and_encode_all() {
        let a = ramp(4, 2, PixelFormat::RgbF32Le);
        let b = ramp(2, 3, PixelFormat::RgbaF32Le)
            .with_chromaticities(crate::image::ColorInfo::chromaticities_for(9).unwrap());
        let c = ramp(3, 3, PixelFormat::GrayF32Le);
        let frames = vec![
            Frame::new(a.clone(), 0).with_name("left".to_string()),
            Frame::new(b.clone(), 1).with_name("right".to_string()),
            Frame::new(c.clone(), 2),
        ];
        let bytes = encode_all(&frames, &EncodeOptions::default()).unwrap();
        let i = info(&bytes).unwrap();
        assert!(i.multipart);
        assert_eq!(i.frames, 3);
        assert_eq!(i.part_name.as_deref(), Some("left"));
        assert_eq!(i.part_type.as_deref(), Some("scanlineimage"));
        // decode = part 0; part / part_name select.
        assert_eq!(decode(&bytes).unwrap().width, 4);
        let right = decode_with(&bytes, &DecodeOptions::default().with_part(1)).unwrap();
        assert_eq!(right.format, PixelFormat::RgbaF32Le);
        assert_eq!(right.color.primaries, 9);
        assert_eq!(
            decode_with(&bytes, &DecodeOptions::default().with_part_name("right")).unwrap(),
            right
        );
        assert!(matches!(
            decode_with(&bytes, &DecodeOptions::default().with_part(3)),
            Err(ExrError::InvalidData(_))
        ));
        assert!(matches!(
            decode_with(&bytes, &DecodeOptions::default().with_part_name("centre")),
            Err(ExrError::InvalidData(_))
        ));
        let all = decode_all(&bytes).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].name.as_deref(), Some("left"));
        assert_eq!(all[2].name.as_deref(), Some("part2"));
        assert_eq!(all[1].index, 1);
        assert!(all.iter().all(|f| f.delay.is_none()));
        // Display window is shared (largest extent), data windows at origin.
        assert_eq!(all[0].image.display_window, all[2].image.display_window);
        assert_eq!(all[0].image.display_window.width(), 4);
        assert_eq!(all[0].image.display_window.height(), 3);
        for (f, want) in all.iter().zip([&a, &b, &c]) {
            assert_eq!(f.image.planes, want.planes);
            assert_eq!(f.image.attributes, want.attributes);
        }
        // A single-part file yields one frame.
        let one = decode_all(&encode(&a, &EncodeOptions::default()).unwrap()).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].image, a);
        assert!(one[0].name.is_none());
        // encode_all refuses what multi-part output cannot express.
        assert!(matches!(
            encode_all(&[], &EncodeOptions::default()),
            Err(ExrError::InvalidData(_))
        ));
        assert!(matches!(
            encode_all(&frames, &EncodeOptions::default().with_tile_size(8)),
            Err(ExrError::Unsupported(_))
        ));
    }

    #[test]
    fn decode_all_skips_unviewable_parts_unless_strict() {
        // Hand-build a two-part file whose second part is depth-only.
        let a = ramp(4, 2, PixelFormat::RgbF32Le);
        let z: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let (chs, planes) = image_to_channels(&a, &EncodeOptions::default()).unwrap();
        let parts = vec![
            MultipartScanlinePart {
                name: "beauty".to_string(),
                width: 4,
                height: 2,
                channels: chs,
                planes: planes.iter().map(|p| p.as_slice()).collect(),
                compression: Compression::None,
            },
            MultipartScanlinePart {
                name: "depth".to_string(),
                width: 4,
                height: 2,
                channels: vec![Channel {
                    name: "Z".to_string(),
                    pixel_type: PixelType::Float,
                    p_linear: false,
                    x_sampling: 1,
                    y_sampling: 1,
                }],
                planes: vec![&z],
                compression: Compression::None,
            },
        ];
        let bytes = crate::multipart_encoder::encode_exr_multipart(&parts).unwrap();
        let all = decode_all(&bytes).unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].index, 0);
        assert!(matches!(
            decode_all_with(&bytes, &DecodeOptions::default().with_strict(true)),
            Err(ExrError::Unsupported(_))
        ));
        assert!(matches!(
            decode_with(&bytes, &DecodeOptions::default().with_part(1)),
            Err(ExrError::Unsupported(_))
        ));
        assert!(matches!(
            decode_with(&bytes, &DecodeOptions::default().with_part_name("depth")),
            Err(ExrError::Unsupported(_))
        ));
    }

    #[test]
    fn deep_files_are_unsupported_for_the_view() {
        let counts = [1u32, 2];
        let samples = [1.0f32, 2.0, 3.0];
        let bytes = crate::deep::encode_exr_deep_scanline(&crate::deep::DeepScanlineInput {
            width: 2,
            height: 1,
            channels: vec![Channel {
                name: "Y".to_string(),
                pixel_type: PixelType::Float,
                p_linear: false,
                x_sampling: 1,
                y_sampling: 1,
            }],
            samples_per_pixel: &counts,
            channel_samples: vec![&samples],
            compression: Compression::None,
        })
        .unwrap();
        assert!(info(&bytes).unwrap().deep);
        assert!(matches!(decode(&bytes), Err(ExrError::Unsupported(_))));
        assert!(matches!(decode_all(&bytes), Err(ExrError::Unsupported(_))));
    }

    #[test]
    fn strict_rejects_misaligned_subsampling() {
        // 2x2 chroma over an odd data window: lenient reads it, strict
        // refuses it.
        let img = ramp(6, 4, PixelFormat::RgbF32Le);
        let bytes = encode(
            &img,
            &EncodeOptions::default().with_colour(ColourLayout::LumaChroma),
        )
        .unwrap();
        assert!(decode(&bytes).is_ok());
        assert!(decode_with(&bytes, &DecodeOptions::default().with_strict(true)).is_ok());
        let odd = img
            .with_data_window(Box2i {
                x_min: 1,
                y_min: 0,
                x_max: 6,
                y_max: 3,
            })
            .unwrap();
        let bytes = encode(
            &odd,
            &EncodeOptions::default().with_colour(ColourLayout::LumaChroma),
        )
        .unwrap();
        assert!(decode(&bytes).is_ok());
        assert!(matches!(
            decode_with(&bytes, &DecodeOptions::default().with_strict(true)),
            Err(ExrError::InvalidData(_))
        ));
    }

    #[test]
    fn padded_plane_is_repacked() {
        let tight = ramp(3, 2, PixelFormat::GrayF32Le);
        let mut padded_data = Vec::new();
        for row in tight.as_bytes().unwrap().chunks_exact(12) {
            padded_data.extend_from_slice(row);
            padded_data.extend_from_slice(&[0xAA; 4]);
        }
        let padded = ExrImage::new(
            3,
            2,
            PixelFormat::GrayF32Le,
            vec![Plane::new(16, padded_data)],
        )
        .unwrap();
        assert_eq!(
            encode(&padded, &EncodeOptions::default()).unwrap(),
            encode(&tight, &EncodeOptions::default()).unwrap()
        );
    }
}
