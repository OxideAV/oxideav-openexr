//! Self-roundtrip integration tests for the round-2 additions: ZIPS,
//! RLE, sub-sampled channels, UINT pixel type. Each test encodes
//! synthetic data, re-parses it, and checks every sample matches
//! bit-exactly (FLOAT) or within an LSB (HALF).

use oxideav_openexr::{
    encode_exr_scanline, encode_exr_scanline_rgba_float_with, header::encode_header,
    header::VersionField, parse_exr, Attribute, AttributeValue, Box2i, Channel, Compression,
    LineOrder, PixelType,
};

fn make_sample_image(w: u32, h: u32) -> Vec<f32> {
    let mut s = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let r = (x as f32 / w as f32) * 1.5;
            let g = (y as f32 / h as f32) * 1.5;
            let b = ((x ^ y) as f32 / 255.0) * 8.0;
            let a = if (x + y) % 2 == 0 { 0.25 } else { 1.75 };
            s.extend_from_slice(&[r, g, b, a]);
        }
    }
    s
}

fn assert_planes_match_rgba(img: &oxideav_openexr::ExrImage, source_rgba: &[f32]) {
    let w = img.width() as usize;
    let h = img.height() as usize;
    let a = &img.planes[0].samples;
    let b = &img.planes[1].samples;
    let g = &img.planes[2].samples;
    let r = &img.planes[3].samples;
    for y in 0..h {
        for x in 0..w {
            let off = y * w + x;
            assert_eq!(r[off], source_rgba[off * 4], "R mismatch at ({x},{y})");
            assert_eq!(g[off], source_rgba[off * 4 + 1], "G mismatch at ({x},{y})");
            assert_eq!(b[off], source_rgba[off * 4 + 2], "B mismatch at ({x},{y})");
            assert_eq!(a[off], source_rgba[off * 4 + 3], "A mismatch at ({x},{y})");
        }
    }
}

#[test]
fn roundtrip_zips_per_scanline() {
    let w = 16;
    let h = 8;
    let samples = make_sample_image(w, h);
    let bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::Zips).unwrap();
    let img = parse_exr(&bytes).unwrap();
    assert_eq!(img.compression, Compression::Zips);
    assert_planes_match_rgba(&img, &samples);
}

#[test]
fn roundtrip_rle() {
    let w = 32;
    let h = 4;
    let samples = make_sample_image(w, h);
    let bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::Rle).unwrap();
    let img = parse_exr(&bytes).unwrap();
    assert_eq!(img.compression, Compression::Rle);
    assert_planes_match_rgba(&img, &samples);
}

#[test]
fn roundtrip_rle_constant_image() {
    // Constant-data image — ideal for RLE; the compressed payload
    // should be much smaller than the raw bytes, and the decoder
    // should still recover them exactly.
    let w = 64;
    let h = 16;
    let samples = vec![0.5_f32; (w * h * 4) as usize];
    let bytes = encode_exr_scanline_rgba_float_with(w, h, &samples, Compression::Rle).unwrap();
    let img = parse_exr(&bytes).unwrap();
    assert_planes_match_rgba(&img, &samples);
}

fn make_uint_image(w: u32, h: u32) -> Vec<u32> {
    (0..(w * h)).map(|i| i.wrapping_mul(0x12345)).collect()
}

#[test]
fn roundtrip_uint_pixel_type() {
    let w = 8;
    let h = 8;
    let uints = make_uint_image(w, h);
    let plane: Vec<f32> = uints.iter().map(|&u| u as f32).collect();

    // Build a single-channel UINT image manually.
    let chs = vec![Channel {
        name: "Z".to_string(),
        pixel_type: PixelType::Uint,
        p_linear: false,
        x_sampling: 1,
        y_sampling: 1,
    }];
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (w - 1) as i32,
        y_max: (h - 1) as i32,
    };
    let attrs = vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(chs.clone()),
        },
        Attribute {
            name: "compression".to_string(),
            value: AttributeValue::Compression(Compression::Zip),
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
    ];
    let bytes = encode_exr_scanline(w, h, &chs, &[&plane], Compression::Zip, attrs).unwrap();
    let img = parse_exr(&bytes).unwrap();
    assert_eq!(img.channels[0].pixel_type, PixelType::Uint);
    let z = &img.planes[0].samples;
    for (i, &expected) in uints.iter().enumerate() {
        // Bit-exact for values that fit in the f32 mantissa (24 bits).
        // Our test values stay well below 2^24.
        if expected < (1 << 24) {
            assert_eq!(
                z[i] as u32, expected,
                "UINT round-trip mismatch at index {i}: got {} expected {expected}",
                z[i] as u32
            );
        }
    }
    let _ = encode_header; // satisfy unused-import warning if we drop the helper
    let _ = VersionField::from_u32(2);
}

#[test]
fn roundtrip_subsampled_chroma_yby_ry() {
    // Synthesize a manually-built EXR with one full-rate Y channel and
    // two ½-x ½-y sub-sampled BY/RY channels, then re-parse and check
    // every channel's plane size matches the sub-sampled dimensions.
    let w: u32 = 8;
    let h: u32 = 8;
    let y_plane: Vec<f32> = (0..(w * h) as usize).map(|i| (i as f32) * 0.01).collect();
    // Sub-sampled planes: 4×4 each.
    let sw = (w / 2) as usize;
    let sh = (h / 2) as usize;
    let by_plane: Vec<f32> = (0..(sw * sh)).map(|i| 0.1 + (i as f32) * 0.02).collect();
    let ry_plane: Vec<f32> = (0..(sw * sh)).map(|i| 0.2 + (i as f32) * 0.03).collect();

    let chs = vec![
        Channel {
            name: "BY".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 2,
            y_sampling: 2,
        },
        Channel {
            name: "RY".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 2,
            y_sampling: 2,
        },
        Channel {
            name: "Y".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        },
    ];
    let win = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (w - 1) as i32,
        y_max: (h - 1) as i32,
    };
    let attrs = vec![
        Attribute {
            name: "channels".to_string(),
            value: AttributeValue::Channels(chs.clone()),
        },
        Attribute {
            name: "compression".to_string(),
            value: AttributeValue::Compression(Compression::None),
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
    ];

    // Build the file manually since our encoder rejects sub-sampled
    // input. The on-disk layout for this case:
    //   block 0: rows 0..1 -> contains Y row 0, BY row 0 (since y%2==0), RY row 0, Y row 1
    //   block 1: rows 2..3 -> Y row 2, BY row 1, RY row 1, Y row 3
    //   ... 4 blocks total with NO_COMPRESSION
    //
    // Per-block layout: for each image row in the block, for each
    // channel (alphabetical: BY, RY, Y), if (row % y_sampling == 0)
    // emit the channel's samples for that row. Channel sub-sampled
    // width is `width.div_ceil(x_sampling)`.

    let header_bytes = encode_header(VersionField::from_u32(2), &attrs);
    let block_h = 1u32; // NONE compression -> one scanline per block
    let num_blocks = h.div_ceil(block_h) as usize;

    // Build per-block payloads.
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(num_blocks);
    for blk in 0..num_blocks {
        let row = blk as u32 * block_h;
        let mut payload = Vec::new();
        // Channels in alphabetical order: BY, RY, Y
        for ch in &chs {
            let ys = ch.y_sampling as u32;
            if row % ys != 0 {
                continue;
            }
            let plane: &[f32] = match ch.name.as_str() {
                "BY" => &by_plane,
                "RY" => &ry_plane,
                "Y" => &y_plane,
                _ => unreachable!(),
            };
            let xs = ch.x_sampling as u32;
            let pw = w.div_ceil(xs) as usize;
            let row_sub = (row / ys) as usize;
            for x in 0..pw {
                let v = plane[row_sub * pw + x];
                payload.extend_from_slice(&v.to_le_bytes());
            }
        }
        payloads.push(payload);
    }

    // Compute offsets and assemble.
    let offset_table_size = num_blocks * 8;
    let mut block_offsets = Vec::with_capacity(num_blocks);
    let mut running = header_bytes.len() + offset_table_size;
    for p in &payloads {
        block_offsets.push(running as u64);
        running += 8 + p.len(); // Y(i32) + size(i32)
    }
    let mut bytes = Vec::with_capacity(running);
    bytes.extend_from_slice(&header_bytes);
    for &off in &block_offsets {
        bytes.extend_from_slice(&off.to_le_bytes());
    }
    for (blk, p) in payloads.iter().enumerate() {
        let y = blk as u32 * block_h;
        bytes.extend_from_slice(&(y as i32).to_le_bytes());
        bytes.extend_from_slice(&(p.len() as i32).to_le_bytes());
        bytes.extend_from_slice(p);
    }

    // Decode and verify.
    let img = parse_exr(&bytes).unwrap();
    assert_eq!(img.channels.len(), 3);
    // alphabetical: BY=0, RY=1, Y=2
    assert_eq!(img.channels[0].name, "BY");
    assert_eq!(img.channels[1].name, "RY");
    assert_eq!(img.channels[2].name, "Y");
    assert_eq!(img.planes[0].samples.len(), sw * sh);
    assert_eq!(img.planes[1].samples.len(), sw * sh);
    assert_eq!(img.planes[2].samples.len(), (w * h) as usize);
    assert_eq!(img.planes[0].samples, by_plane);
    assert_eq!(img.planes[1].samples, ry_plane);
    assert_eq!(img.planes[2].samples, y_plane);
}
