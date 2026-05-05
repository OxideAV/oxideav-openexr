//! Round-trip a HALF-channel scanline EXR. Uses the lower-level
//! [`encode_exr_scanline`] entry point so we can configure the channel
//! pixel type to HALF.

use oxideav_openexr::{
    encode_exr_scanline, parse_exr, Attribute, AttributeValue, Box2i, Channel, Compression,
    LineOrder, PixelType,
};

#[test]
fn half_channels_roundtrip_zip() {
    let w: u32 = 8;
    let h: u32 = 8;
    let pixels = (w * h) as usize;

    // Pick a couple of values that are exactly representable in HALF
    // (so encode->decode is bit-exact).
    let mut g_plane = Vec::with_capacity(pixels);
    let mut y_plane = Vec::with_capacity(pixels);
    for i in 0..pixels {
        // 0.0, 0.5, 1.0, 2.0 — all exactly representable.
        g_plane.push(((i as u32 % 4) as f32) * 0.5);
        y_plane.push(if (i & 1) == 0 { 1.0 } else { 0.5 });
    }

    let chs = vec![
        Channel {
            name: "G".to_string(),
            pixel_type: PixelType::Half,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        },
        Channel {
            name: "Y".to_string(),
            pixel_type: PixelType::Half,
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

    let bytes =
        encode_exr_scanline(w, h, &chs, &[&g_plane, &y_plane], Compression::Zip, attrs).unwrap();
    let img = parse_exr(&bytes).unwrap();
    assert_eq!(img.width(), w);
    assert_eq!(img.height(), h);
    assert_eq!(img.compression, Compression::Zip);
    assert_eq!(img.channels.len(), 2);
    assert_eq!(img.channels[0].name, "G");
    assert_eq!(img.channels[1].name, "Y");
    assert_eq!(img.planes[0].samples, g_plane);
    assert_eq!(img.planes[1].samples, y_plane);
}
