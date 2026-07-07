//! Regression guard for the multi-part `displayWindow` invariant.
//!
//! Every part of a multi-part EXR file must carry an **identical**
//! `displayWindow` — it is a file-global concept. The reference reader
//! refuses to open a file whose parts disagree (it fails at open time
//! with a generic "unable to open" error). The `dataWindow`, by
//! contrast, is legitimately per-part.
//!
//! Before this guard, the scanline / tiled / mipmap / ripmap multi-part
//! writers each set every part's `displayWindow` equal to that part's
//! own `dataWindow`. For equal-sized parts the windows coincide and the
//! bug was invisible; with unequal-sized parts the per-part
//! displayWindows diverged and the whole file became unreadable by any
//! conforming reader even though our own parser round-tripped it.
//!
//! These checks are binary-independent (they parse our own output and
//! compare the emitted displayWindow attributes), so they run on every
//! CI host regardless of whether an OpenEXR install is present.

use oxideav_openexr::{
    encode_exr_multipart, encode_exr_multipart_tiled, encode_exr_multipart_tiled_mipmap,
    encode_exr_multipart_tiled_ripmap, mipmap_level_count_round_down, parse_exr_multipart,
    parse_exr_multipart_tiled, parse_exr_multipart_tiled_multilevel,
    ripmap_level_counts_round_down, Attribute, AttributeValue, Box2i, Channel, Compression,
    MipmapLevel, MultipartMipmapTiledPart, MultipartRipmapTiledPart, MultipartScanlinePart,
    MultipartTiledPart, RipmapPyramid,
};

fn display_window(attrs: &[Attribute]) -> Box2i {
    match &attrs
        .iter()
        .find(|a| a.name == "displayWindow")
        .expect("displayWindow attribute missing")
        .value
    {
        AttributeValue::Box2i(b) => *b,
        other => panic!("displayWindow decoded as wrong variant: {other:?}"),
    }
}

fn data_window(attrs: &[Attribute]) -> Box2i {
    match &attrs
        .iter()
        .find(|a| a.name == "dataWindow")
        .expect("dataWindow attribute missing")
        .value
    {
        AttributeValue::Box2i(b) => *b,
        other => panic!("dataWindow decoded as wrong variant: {other:?}"),
    }
}

fn gray_channel() -> Vec<Channel> {
    vec![Channel {
        name: "Y".to_string(),
        pixel_type: oxideav_openexr::PixelType::Half,
        p_linear: false,
        x_sampling: 1,
        y_sampling: 1,
    }]
}

/// Assert every part shares one displayWindow equal to the bounding box
/// of the largest part, and that data windows stay per-part.
fn assert_shared_display(images: &[oxideav_openexr::ExrImage], expect: Box2i) {
    let mut saw_smaller_data_window = false;
    for img in images {
        assert_eq!(
            display_window(&img.attributes),
            expect,
            "a part's displayWindow diverges from the file-global window"
        );
        if data_window(&img.attributes) != expect {
            saw_smaller_data_window = true;
        }
    }
    assert!(
        saw_smaller_data_window,
        "test is not exercising unequal-sized parts (data windows all equal)"
    );
}

#[test]
fn scanline_multipart_shares_display_window() {
    let (w0, h0, w1, h1) = (12u32, 10u32, 8u32, 6u32);
    let g0: Vec<f32> = (0..(w0 * h0) as usize).map(|i| i as f32 * 0.02).collect();
    let g1: Vec<f32> = (0..(w1 * h1) as usize).map(|i| i as f32 * 0.03).collect();
    let parts = vec![
        MultipartScanlinePart {
            name: "big".to_string(),
            width: w0,
            height: h0,
            channels: gray_channel(),
            planes: vec![g0.as_slice()],
            compression: Compression::Zip,
        },
        MultipartScanlinePart {
            name: "small".to_string(),
            width: w1,
            height: h1,
            channels: gray_channel(),
            planes: vec![g1.as_slice()],
            compression: Compression::Rle,
        },
    ];
    let bytes = encode_exr_multipart(&parts).unwrap();
    let images = parse_exr_multipart(&bytes).unwrap();
    let expect = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (w0 - 1) as i32,
        y_max: (h0 - 1) as i32,
    };
    assert_shared_display(&images, expect);
}

#[test]
fn tiled_multipart_shares_display_window() {
    let (w0, h0, w1, h1) = (24u32, 20u32, 16u32, 12u32);
    let g0: Vec<f32> = vec![0.5; (w0 * h0) as usize];
    let g1: Vec<f32> = vec![0.7; (w1 * h1) as usize];
    let parts = vec![
        MultipartTiledPart {
            name: "big".to_string(),
            width: w0,
            height: h0,
            tile_x: 8,
            tile_y: 8,
            channels: gray_channel(),
            planes: vec![g0.as_slice()],
            compression: Compression::Zip,
        },
        MultipartTiledPart {
            name: "small".to_string(),
            width: w1,
            height: h1,
            tile_x: 8,
            tile_y: 8,
            channels: gray_channel(),
            planes: vec![g1.as_slice()],
            compression: Compression::Zip,
        },
    ];
    let bytes = encode_exr_multipart_tiled(&parts).unwrap();
    let images = parse_exr_multipart_tiled(&bytes).unwrap();
    let expect = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (w0 - 1) as i32,
        y_max: (h0 - 1) as i32,
    };
    assert_shared_display(&images, expect);
}

fn mipmap_pyramid(w: u32, h: u32, fill: f32) -> Vec<MipmapLevel> {
    let n = mipmap_level_count_round_down(w, h);
    let mut levels = Vec::with_capacity(n as usize);
    let (mut lw, mut lh) = (w, h);
    for _ in 0..n {
        levels.push(MipmapLevel {
            width: lw,
            height: lh,
            planes: vec![vec![fill; (lw * lh) as usize]],
        });
        lw = (lw / 2).max(1);
        lh = (lh / 2).max(1);
    }
    levels
}

#[test]
fn mipmap_multipart_shares_display_window() {
    let (w0, h0, w1, h1) = (32u32, 32u32, 16u32, 16u32);
    let parts = vec![
        MultipartMipmapTiledPart {
            name: "big".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: gray_channel(),
            pyramid: mipmap_pyramid(w0, h0, 0.5),
            compression: Compression::Zip,
        },
        MultipartMipmapTiledPart {
            name: "small".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: gray_channel(),
            pyramid: mipmap_pyramid(w1, h1, 0.7),
            compression: Compression::Zip,
        },
    ];
    let bytes = encode_exr_multipart_tiled_mipmap(&parts).unwrap();
    let parts_out = parse_exr_multipart_tiled_multilevel(&bytes).unwrap();
    let expect = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (w0 - 1) as i32,
        y_max: (h0 - 1) as i32,
    };
    let mut saw_smaller = false;
    for p in &parts_out {
        assert_eq!(
            p.display_window, expect,
            "a mipmap part's displayWindow diverges from the file-global window"
        );
        if p.data_window != expect {
            saw_smaller = true;
        }
    }
    assert!(
        saw_smaller,
        "test not exercising unequal-sized mipmap parts"
    );
}

fn ripmap_pyramid(w: u32, h: u32, fill: f32) -> RipmapPyramid {
    let (nx, ny) = ripmap_level_counts_round_down(w, h);
    let mut grid: Vec<Vec<MipmapLevel>> = Vec::with_capacity(ny as usize);
    for ly in 0..ny {
        let mut row = Vec::with_capacity(nx as usize);
        for lx in 0..nx {
            let lw = (w >> lx).max(1);
            let lh = (h >> ly).max(1);
            row.push(MipmapLevel {
                width: lw,
                height: lh,
                planes: vec![vec![fill; (lw * lh) as usize]],
            });
        }
        grid.push(row);
    }
    RipmapPyramid { grid }
}

#[test]
fn ripmap_multipart_shares_display_window() {
    let (w0, h0, w1, h1) = (32u32, 32u32, 16u32, 16u32);
    let parts = vec![
        MultipartRipmapTiledPart {
            name: "big".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: gray_channel(),
            pyramid: ripmap_pyramid(w0, h0, 0.5),
            compression: Compression::Zip,
        },
        MultipartRipmapTiledPart {
            name: "small".to_string(),
            tile_x: 8,
            tile_y: 8,
            channels: gray_channel(),
            pyramid: ripmap_pyramid(w1, h1, 0.7),
            compression: Compression::Zip,
        },
    ];
    let bytes = encode_exr_multipart_tiled_ripmap(&parts).unwrap();
    let parts_out = parse_exr_multipart_tiled_multilevel(&bytes).unwrap();
    let expect = Box2i {
        x_min: 0,
        y_min: 0,
        x_max: (w0 - 1) as i32,
        y_max: (h0 - 1) as i32,
    };
    let mut saw_smaller = false;
    for p in &parts_out {
        assert_eq!(
            p.display_window, expect,
            "a ripmap part's displayWindow diverges from the file-global window"
        );
        if p.data_window != expect {
            saw_smaller = true;
        }
    }
    assert!(
        saw_smaller,
        "test not exercising unequal-sized ripmap parts"
    );
}
