//! The registry encoder's part-shape options (tiled ONE_LEVEL / MIPMAP
//! / RIPMAP, line orders) produce files the reference tools accept:
//! `exrheader` reports the tile description and line order, and
//! `exrmetrics --convert` fully decodes and re-encodes every one of
//! them (an opaque process) with level 0 matching our own decode for
//! the lossless cases.
//! Auto-skips when the binaries are absent.
#![cfg(feature = "registry")]

use std::process::Command;

use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, RuntimeContext, VideoFrame};
use oxideav_openexr::parse_exr;

fn tool_available(name: &str) -> bool {
    Command::new(name)
        .arg("--help")
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

fn tempdir() -> std::path::PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "oxideav-openexr-encsurface-{nanos}-{}-{seq}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn encode(w: u32, h: u32, opts: &[(&str, &str)]) -> Vec<u8> {
    let mut ctx = RuntimeContext::new();
    oxideav_openexr::register(&mut ctx);
    let mut params = CodecParameters::video(CodecId::new(oxideav_openexr::CODEC_ID_STR));
    params.width = Some(w);
    params.height = Some(h);
    params.pixel_format = Some(PixelFormat::RgbaF32Le);
    for (k, v) in opts {
        params.options.insert(*k, v.to_string());
    }
    let mut data = Vec::with_capacity((w * h) as usize * 16);
    for px in 0..(w * h) as usize {
        for c in 0..4 {
            data.extend_from_slice(&((px % 9) as f32 * 0.25 + c as f32 * 0.1).to_le_bytes());
        }
    }
    let vf = VideoFrame {
        pts: None,
        planes: vec![oxideav_core::VideoPlane {
            stride: w as usize * 16,
            data,
        }],
    };
    let mut enc = ctx.codecs.first_encoder(&params).unwrap();
    enc.send_frame(&Frame::Video(vf)).unwrap();
    enc.receive_packet().unwrap().data
}

#[test]
fn reference_tools_accept_every_registry_part_shape() {
    if !tool_available("exrheader") || !tool_available("exrmetrics") {
        eprintln!("reference tools not available, skipping");
        return;
    }
    let dir = tempdir();
    let (w, h) = (40u32, 24u32);
    type Case = (
        &'static [(&'static str, &'static str)],
        &'static str,
        &'static str,
    );
    let cases: [Case; 7] = [
        (
            &[("line_order", "decreasing_y")],
            "scanlineimage",
            "decreasing y",
        ),
        (&[("tile_size", "16")], "tiledimage", "tile size 16 by 16"),
        (
            &[
                ("tile_size", "16"),
                ("levels", "mipmap"),
                ("compression", "piz"),
            ],
            "tiledimage",
            "mip-map",
        ),
        (
            &[
                ("tile_size", "16"),
                ("levels", "ripmap"),
                ("compression", "dwaa"),
            ],
            "tiledimage",
            "rip-map",
        ),
        (
            &[
                ("tile_size", "8"),
                ("line_order", "decreasing_y"),
                ("compression", "b44a"),
            ],
            "tiledimage",
            "decreasing y",
        ),
        (
            &[
                ("tile_size", "8"),
                ("levels", "mipmap"),
                ("line_order", "random_y"),
            ],
            "tiledimage",
            "random y",
        ),
        (
            &[
                ("tile_size", "16"),
                ("levels", "ripmap"),
                ("compression", "pxr24"),
                ("pixel_type", "half"),
            ],
            "tiledimage",
            "rip-map",
        ),
    ];
    for (i, (opts, part_type, needle)) in cases.iter().enumerate() {
        let bytes = encode(w, h, opts);
        let path = dir.join(format!("case{i}.exr"));
        std::fs::write(&path, &bytes).unwrap();
        let out = Command::new("exrheader").arg(&path).output().unwrap();
        assert!(out.status.success(), "exrheader rejected case {i} {opts:?}");
        let text = String::from_utf8_lossy(&out.stdout).to_lowercase();
        assert!(text.contains(part_type), "case {i}: {text}");
        assert!(text.contains(needle), "case {i} ({opts:?}): {text}");
        // Full reference decode + re-encode (an opaque process); level 0
        // of what it wrote must match our own decode of our file for the
        // lossless cases.
        let conv = dir.join(format!("case{i}_conv.exr"));
        let out = Command::new("exrmetrics")
            .args(["--convert", "-z", "none", "-o"])
            .arg(&conv)
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "reference full decode failed for case {i} {opts:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let lossless = !opts
            .iter()
            .any(|(k, v)| *k == "compression" && matches!(*v, "dwaa" | "b44a" | "pxr24"));
        if lossless {
            let ours = parse_exr(&bytes).unwrap();
            let theirs = parse_exr(&std::fs::read(&conv).unwrap()).unwrap();
            assert_eq!(theirs.channels, ours.channels, "case {i}");
            assert_eq!(theirs.planes, ours.planes, "case {i}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
