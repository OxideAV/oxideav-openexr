//! Multi-part EXR encoder (scanline parts only — round 40).
//!
//! A multi-part file (version-field bit 0x1000) carries one or more
//! independent parts in a single .exr stream. Each part has its own
//! header (with the standard required attributes plus `name`, `type`,
//! and `chunkCount`), its own offset table, and its own chunks. Chunks
//! are interleaved across parts and tagged with a `part_number` prefix.
//!
//! Binary layout:
//!
//! ```text
//! magic(4) | version(4 with multipart bit set)
//! | header_0 ... NUL | header_1 ... NUL | NUL          (extra NUL = end-of-headers)
//! | offset_table_0(chunkCount_0×u64) | offset_table_1(...) | ...
//! | chunks: each starts with i32 part_number,
//!           then i32 Y | i32 size | payload[size] for scanlineimage parts.
//! ```
//!
//! Round-40 surface: scanline-image parts only (`type = "scanlineimage"`).
//! Tiled parts (`type = "tiledimage"`) and deep parts are deferred to a
//! followup. The decoder's `parse_exr_multipart` already round-trips
//! files we emit here.

use crate::decoder::{apply_zip_interleave, apply_zip_predictor, subsampled_dim};
use crate::error::{ExrError, Result};
use crate::header::{encode_attribute_value, VersionField};
use crate::types::{
    Attribute, AttributeValue, Box2i, Channel, Compression, LineOrder, PixelType, EXR_MAGIC,
};

/// One scanline part for [`encode_exr_multipart`]. The caller supplies
/// pixel planes in alphabetical-channel-name order plus a few
/// per-part metadata bits; the standard required attributes are
/// derived automatically.
pub struct MultipartScanlinePart<'a> {
    /// The part name (must be unique across all parts in the file).
    pub name: String,
    /// Width / height of the data window (display window matches it).
    pub width: u32,
    pub height: u32,
    pub channels: Vec<Channel>,
    /// One `width * height` `f32` slice per channel, in the same
    /// alphabetical order as `channels`.
    pub planes: Vec<&'a [f32]>,
    pub compression: Compression,
}

/// Encode a multi-part EXR file from one or more
/// [`MultipartScanlinePart`] descriptions. Parts must each carry a
/// unique `name`.
///
/// Round-trips bit-exactly through
/// [`crate::parse_exr_multipart`].
pub fn encode_exr_multipart(parts: &[MultipartScanlinePart]) -> Result<Vec<u8>> {
    if parts.is_empty() {
        return Err(ExrError::invalid(
            "encode_exr_multipart: at least one part required".to_string(),
        ));
    }
    // Check unique names + per-part validation.
    for (i, p) in parts.iter().enumerate() {
        if p.name.is_empty() {
            return Err(ExrError::invalid(format!("part {i}: empty name")));
        }
        for (j, other) in parts.iter().enumerate() {
            if j != i && other.name == p.name {
                return Err(ExrError::invalid(format!(
                    "duplicate part name '{}' (parts {i} and {j})",
                    p.name
                )));
            }
        }
        if p.channels.len() != p.planes.len() {
            return Err(ExrError::invalid(format!(
                "part '{}': channels.len()={} != planes.len()={}",
                p.name,
                p.channels.len(),
                p.planes.len()
            )));
        }
        for win in p.channels.windows(2) {
            if win[0].name >= win[1].name {
                return Err(ExrError::invalid(format!(
                    "part '{}': channels not alphabetical: '{}' >= '{}'",
                    p.name, win[0].name, win[1].name
                )));
            }
        }
        for (ch, plane) in p.channels.iter().zip(p.planes.iter()) {
            if ch.x_sampling <= 0 || ch.y_sampling <= 0 {
                return Err(ExrError::invalid(format!(
                    "part '{}': channel '{}' x_sampling={} y_sampling={} (must be positive)",
                    p.name, ch.name, ch.x_sampling, ch.y_sampling
                )));
            }
            let pw = subsampled_dim(p.width, ch.x_sampling as u32) as usize;
            let ph = subsampled_dim(p.height, ch.y_sampling as u32) as usize;
            let need = pw * ph;
            if plane.len() != need {
                return Err(ExrError::invalid(format!(
                    "part '{}': channel '{}' plane length {} != subsampled width*height = {pw}*{ph} = {need}",
                    p.name,
                    ch.name,
                    plane.len()
                )));
            }
        }
        if !matches!(
            p.compression,
            Compression::None | Compression::Zip | Compression::Zips | Compression::Rle
        ) {
            return Err(ExrError::unsupported(format!(
                "part '{}': compression {:?} (multipart encoder supports NONE/ZIP/ZIPS/RLE)",
                p.name, p.compression
            )));
        }
    }

    // ---- Build per-part headers (without the trailing NUL terminator) ----
    let mut header_byte_blocks: Vec<Vec<u8>> = Vec::with_capacity(parts.len());
    let mut chunk_counts: Vec<u32> = Vec::with_capacity(parts.len());

    for p in parts {
        let block_h = p.compression.scanlines_per_block();
        let cc = p.height.div_ceil(block_h);
        chunk_counts.push(cc);

        let attrs = build_scanline_part_attrs(p, cc);
        header_byte_blocks.push(encode_part_header_attributes(&attrs));
    }

    // ---- Stitch the file together: magic + version + headers + final NUL ----
    let version = VersionField::from_u32(2 | 0x1000); // multipart bit
    let mut out = Vec::with_capacity(1024);
    out.extend_from_slice(&EXR_MAGIC.to_le_bytes());
    out.extend_from_slice(&version.to_u32().to_le_bytes());
    for hb in &header_byte_blocks {
        out.extend_from_slice(hb);
        out.push(0); // per-part header terminator
    }
    out.push(0); // double-NUL = end-of-all-headers

    // ---- Build per-part block payloads ----
    let mut part_block_payloads: Vec<Vec<Vec<u8>>> = Vec::with_capacity(parts.len());
    for p in parts {
        let block_h = p.compression.scanlines_per_block();
        let cc = chunk_counts[part_block_payloads.len()] as usize;

        let mut blocks: Vec<Vec<u8>> = Vec::with_capacity(cc);
        for block_idx in 0..cc {
            let row0 = block_idx as u32 * block_h;
            let lines_in_block = (p.height - row0).min(block_h) as usize;
            let mut raw: Vec<u8> = Vec::new();
            for line in 0..lines_in_block {
                let y = row0 as usize + line;
                for (ch_idx, ch) in p.channels.iter().enumerate() {
                    let ys = ch.y_sampling as u32;
                    if (y as u32) % ys != 0 {
                        continue;
                    }
                    let xs = ch.x_sampling as u32;
                    let pw = subsampled_dim(p.width, xs) as usize;
                    let plane_y = y / ys as usize;
                    let plane = p.planes[ch_idx];
                    for x in 0..pw {
                        let v = plane[plane_y * pw + x];
                        match ch.pixel_type {
                            PixelType::Half => {
                                raw.extend_from_slice(&crate::half::f32_to_half(v).to_le_bytes())
                            }
                            PixelType::Float => raw.extend_from_slice(&v.to_le_bytes()),
                            PixelType::Uint => {
                                let u = if v.is_nan() || v < 0.0 {
                                    0u32
                                } else if v >= (u32::MAX as f32) {
                                    u32::MAX
                                } else {
                                    (v + 0.5) as u32
                                };
                                raw.extend_from_slice(&u.to_le_bytes());
                            }
                        }
                    }
                }
            }
            let payload = compress_block(raw, p.compression)?;
            blocks.push(payload);
        }
        part_block_payloads.push(blocks);
    }

    // ---- Compute offsets and emit offset tables + chunks ----
    let header_bytes_so_far = out.len();
    let total_chunks: usize = chunk_counts.iter().map(|&c| c as usize).sum();
    let offset_table_bytes = total_chunks * 8;
    // Chunks region starts after all offset tables.
    let chunks_start = header_bytes_so_far + offset_table_bytes;

    // Compute the absolute offset of every chunk in flat order
    // (part_0_chunks then part_1_chunks ...). Each chunk's record on
    // disk is `i32 part | i32 Y | i32 size | payload[size]`, total
    // 12 + size bytes.
    let mut chunk_records: Vec<(u32, u32, u32, Vec<u8>)> = Vec::with_capacity(total_chunks);
    {
        let mut running = chunks_start;
        for (part_idx, part) in parts.iter().enumerate() {
            let block_h = part.compression.scanlines_per_block();
            for (block_idx, payload) in part_block_payloads[part_idx].iter().enumerate() {
                let y = block_idx as u32 * block_h;
                chunk_records.push((part_idx as u32, y, running as u32, payload.clone()));
                running += 12 + payload.len();
            }
        }
        // Sanity: keep all offsets fitting in u64 (always true on disk).
        let _ = running;
    }

    // Per-part offset tables, populated in part-block order so each
    // part's table lists its own block offsets in increasing-Y order.
    let mut tables: Vec<Vec<u64>> = vec![Vec::new(); parts.len()];
    for (pi, _y, off, _) in &chunk_records {
        tables[*pi as usize].push(*off as u64);
    }
    for table in &tables {
        for &o in table {
            out.extend_from_slice(&o.to_le_bytes());
        }
    }

    // Now emit each chunk record in the same order as chunk_records.
    for (part_idx, y, _off, payload) in chunk_records {
        out.extend_from_slice(&(part_idx as i32).to_le_bytes());
        out.extend_from_slice(&(y as i32).to_le_bytes());
        out.extend_from_slice(&(payload.len() as i32).to_le_bytes());
        out.extend_from_slice(&payload);
    }

    Ok(out)
}

/// Convenience entry point: encode a single-part RGBA float multipart
/// file. (Useful as a smoke test that our multipart writer interoperates
/// with the multipart reader at trivial part counts.)
pub fn encode_exr_multipart_rgba_float_with(
    parts: &[(String, u32, u32, &[f32], Compression)],
) -> Result<Vec<u8>> {
    // Reshape each (name, w, h, samples, comp) into a
    // MultipartScanlinePart with planes A, B, G, R.
    // Since we need the planes to outlive the call, keep the per-part
    // f32 vectors in a holder.
    let mut holder: Vec<RgbaPlanes> = Vec::with_capacity(parts.len());
    for (name, w, h, samples, _comp) in parts {
        let pixels = (*w as usize) * (*h as usize);
        if samples.len() != pixels * 4 {
            return Err(ExrError::invalid(format!(
                "part '{name}': samples len {} != {pixels}*4",
                samples.len()
            )));
        }
        let mut a = Vec::with_capacity(pixels);
        let mut b = Vec::with_capacity(pixels);
        let mut g = Vec::with_capacity(pixels);
        let mut r = Vec::with_capacity(pixels);
        for px in 0..pixels {
            r.push(samples[px * 4]);
            g.push(samples[px * 4 + 1]);
            b.push(samples[px * 4 + 2]);
            a.push(samples[px * 4 + 3]);
        }
        holder.push(RgbaPlanes { a, b, g, r });
    }

    let chs = vec![
        Channel {
            name: "A".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        },
        Channel {
            name: "B".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        },
        Channel {
            name: "G".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        },
        Channel {
            name: "R".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        },
    ];

    let descs: Vec<MultipartScanlinePart> = parts
        .iter()
        .zip(holder.iter())
        .map(|((name, w, h, _, comp), pl)| MultipartScanlinePart {
            name: name.clone(),
            width: *w,
            height: *h,
            channels: chs.clone(),
            planes: vec![
                pl.a.as_slice(),
                pl.b.as_slice(),
                pl.g.as_slice(),
                pl.r.as_slice(),
            ],
            compression: *comp,
        })
        .collect();
    encode_exr_multipart(&descs)
}

/// Per-part RGBA float plane container used by
/// [`encode_exr_multipart_rgba_float_with`] to keep the planes alive
/// for the duration of the call.
struct RgbaPlanes {
    a: Vec<f32>,
    b: Vec<f32>,
    g: Vec<f32>,
    r: Vec<f32>,
}

/// Build the per-part attribute set for a scanline part (standard
/// required attrs + name + type + chunkCount).
fn build_scanline_part_attrs(part: &MultipartScanlinePart, chunk_count: u32) -> Vec<Attribute> {
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (part.width - 1) as i32,
        y_max: (part.height - 1) as i32,
    };
    vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(part.channels.clone()),
        },
        Attribute {
            name: "chunkCount".to_string(),
            value: AttributeValue::Other {
                type_name: "int".to_string(),
                data: (chunk_count as i32).to_le_bytes().to_vec(),
            },
        },
        Attribute {
            name: "compression".to_string(),
            value: AttributeValue::Compression(part.compression),
        },
        Attribute {
            name: "dataWindow".to_string(),
            value: AttributeValue::Box2i(win),
        },
        Attribute {
            name: "displayWindow".to_string(),
            value: AttributeValue::Box2i(win),
        },
        Attribute {
            name: "lineOrder".to_string(),
            value: AttributeValue::LineOrder(LineOrder::IncreasingY),
        },
        Attribute {
            name: "name".to_string(),
            value: AttributeValue::Other {
                type_name: "string".to_string(),
                data: part.name.as_bytes().to_vec(),
            },
        },
        Attribute {
            name: "pixelAspectRatio".to_string(),
            value: AttributeValue::Float(1.0),
        },
        Attribute {
            name: "screenWindowCenter".to_string(),
            value: AttributeValue::V2f(0.0, 0.0),
        },
        Attribute {
            name: "screenWindowWidth".to_string(),
            value: AttributeValue::Float(1.0),
        },
        Attribute {
            name: "type".to_string(),
            value: AttributeValue::Other {
                type_name: "string".to_string(),
                data: b"scanlineimage".to_vec(),
            },
        },
    ]
}

/// Encode the attribute table for one part of a multipart file
/// (without the trailing per-part NUL terminator — caller appends).
fn encode_part_header_attributes(attrs: &[Attribute]) -> Vec<u8> {
    let mut out = Vec::with_capacity(256);
    for a in attrs {
        out.extend_from_slice(a.name.as_bytes());
        out.push(0);
        let (type_name, payload) = encode_attribute_value(&a.value);
        out.extend_from_slice(type_name.as_bytes());
        out.push(0);
        out.extend_from_slice(&(payload.len() as i32).to_le_bytes());
        out.extend_from_slice(&payload);
    }
    out
}

/// Compress one block of raw bytes per the requested compression.
fn compress_block(raw: Vec<u8>, compression: Compression) -> Result<Vec<u8>> {
    Ok(match compression {
        Compression::None => raw,
        Compression::Zip | Compression::Zips => {
            let mut interleaved = vec![0u8; raw.len()];
            apply_zip_interleave(&raw, &mut interleaved);
            apply_zip_predictor(&mut interleaved);
            let compressed = crate::encoder::zlib_deflate_pub(&interleaved)?;
            if compressed.len() >= raw.len() {
                raw
            } else {
                compressed
            }
        }
        Compression::Rle => {
            let mut interleaved = vec![0u8; raw.len()];
            apply_zip_interleave(&raw, &mut interleaved);
            apply_zip_predictor(&mut interleaved);
            let compressed = crate::rle::rle_compress(&interleaved);
            if compressed.len() >= raw.len() {
                raw
            } else {
                compressed
            }
        }
        _ => unreachable!("filtered above"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_exr_multipart;

    fn make_image(w: u32, h: u32, salt: f32) -> Vec<f32> {
        let mut s = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                s.push((x as f32 / w as f32) + salt);
                s.push(y as f32 / h as f32);
                s.push(((x ^ y) as f32) * 0.01);
                s.push(1.0);
            }
        }
        s
    }

    #[test]
    fn multipart_two_parts_zip_self_roundtrip() {
        let w = 16;
        let h = 16;
        let s_a = make_image(w, h, 0.0);
        let s_b = make_image(w, h, 0.5);
        let bytes = encode_exr_multipart_rgba_float_with(&[
            ("partA".to_string(), w, h, s_a.as_slice(), Compression::Zip),
            ("partB".to_string(), w, h, s_b.as_slice(), Compression::Zip),
        ])
        .unwrap();
        let parts = parse_exr_multipart(&bytes).unwrap();
        assert_eq!(parts.len(), 2);
        // Verify pixel data round-trips (alphabetical: A=0, B=1, G=2, R=3).
        for (img, source) in parts.iter().zip([&s_a, &s_b]) {
            assert_eq!(img.width(), w);
            assert_eq!(img.height(), h);
            let a = &img.planes[0].samples;
            let bp = &img.planes[1].samples;
            let g = &img.planes[2].samples;
            let r = &img.planes[3].samples;
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let off = y * w as usize + x;
                    assert_eq!(r[off], source[off * 4]);
                    assert_eq!(g[off], source[off * 4 + 1]);
                    assert_eq!(bp[off], source[off * 4 + 2]);
                    assert_eq!(a[off], source[off * 4 + 3]);
                }
            }
        }
    }

    #[test]
    fn multipart_three_parts_mixed_compression() {
        let w = 8;
        let h = 8;
        let s_a = make_image(w, h, 0.0);
        let s_b = make_image(w, h, 0.25);
        let s_c = make_image(w, h, 0.75);
        let bytes = encode_exr_multipart_rgba_float_with(&[
            ("alpha".to_string(), w, h, s_a.as_slice(), Compression::None),
            ("beta".to_string(), w, h, s_b.as_slice(), Compression::Zips),
            ("gamma".to_string(), w, h, s_c.as_slice(), Compression::Rle),
        ])
        .unwrap();
        let parts = parse_exr_multipart(&bytes).unwrap();
        assert_eq!(parts.len(), 3);
        let sources = [&s_a, &s_b, &s_c];
        for (img, source) in parts.iter().zip(sources.iter()) {
            let a = &img.planes[0].samples;
            let bp = &img.planes[1].samples;
            let g = &img.planes[2].samples;
            let r = &img.planes[3].samples;
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let off = y * w as usize + x;
                    assert_eq!(r[off], source[off * 4]);
                    assert_eq!(g[off], source[off * 4 + 1]);
                    assert_eq!(bp[off], source[off * 4 + 2]);
                    assert_eq!(a[off], source[off * 4 + 3]);
                }
            }
        }
    }

    #[test]
    fn multipart_rejects_duplicate_names() {
        let w = 4;
        let h = 4;
        let s = make_image(w, h, 0.0);
        let r = encode_exr_multipart_rgba_float_with(&[
            ("dup".to_string(), w, h, s.as_slice(), Compression::None),
            ("dup".to_string(), w, h, s.as_slice(), Compression::None),
        ]);
        assert!(r.is_err());
    }
}
