//! Layered / multi-view files produced by the reference tools (opaque
//! processes) decode through the layer enumeration and the registry's
//! `layer` option. `exrmultiview` combines two single-view files into
//! one multi-view file: the first view's channels stay unprefixed and
//! the second view's get a `<view>.` prefix, with a `multiView`
//! attribute naming both. Auto-skips when the binaries are absent.
#![cfg(feature = "registry")]

use std::process::Command;

use oxideav_core::{
    CodecId, CodecParameters, Frame, Packet, PixelFormat, RuntimeContext, TimeBase, VideoFrame,
};
use oxideav_openexr::{parse_exr, LayerKind};

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
        "oxideav-openexr-layers-{nanos}-{}-{seq}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn ctx() -> RuntimeContext {
    let mut ctx = RuntimeContext::new();
    oxideav_openexr::register(&mut ctx);
    ctx
}

fn frame(w: u32, h: u32, comps: usize, f: impl Fn(usize, usize) -> f32) -> VideoFrame {
    let mut data = Vec::with_capacity((w * h) as usize * comps * 4);
    for px in 0..(w * h) as usize {
        for c in 0..comps {
            data.extend_from_slice(&f(px, c).to_le_bytes());
        }
    }
    VideoFrame {
        pts: None,
        planes: vec![oxideav_core::VideoPlane {
            stride: w as usize * comps * 4,
            data,
        }],
    }
}

fn encode(ctx: &RuntimeContext, vf: VideoFrame, w: u32, h: u32, opts: &[(&str, &str)]) -> Vec<u8> {
    let mut params = CodecParameters::video(CodecId::new(oxideav_openexr::CODEC_ID_STR));
    params.width = Some(w);
    params.height = Some(h);
    params.pixel_format = Some(PixelFormat::RgbF32Le);
    for (k, v) in opts {
        params.options.insert(*k, v.to_string());
    }
    let mut enc = ctx.codecs.first_encoder(&params).unwrap();
    enc.send_frame(&Frame::Video(vf)).unwrap();
    enc.receive_packet().unwrap().data
}

fn decode(ctx: &RuntimeContext, bytes: &[u8], layer: &str) -> oxideav_core::Result<VideoFrame> {
    let mut params = CodecParameters::video(CodecId::new(oxideav_openexr::CODEC_ID_STR));
    params.options.insert("layer", layer.to_string());
    let mut dec = ctx.codecs.first_decoder(&params)?;
    dec.send_packet(&Packet::new(0, TimeBase::new(1, 1), bytes.to_vec()))?;
    match dec.receive_frame()? {
        Frame::Video(v) => Ok(v),
        _ => panic!("expected video frame"),
    }
}

fn f32_at(vf: &VideoFrame, px: usize, comps: usize, c: usize) -> f32 {
    let b = px * comps * 4 + c * 4;
    f32::from_le_bytes(vf.planes[0].data[b..b + 4].try_into().unwrap())
}

#[test]
fn reference_multi_view_file_decodes_per_view() {
    if !tool_available("exrmultiview") {
        eprintln!("exrmultiview not available, skipping");
        return;
    }
    let (w, h) = (16u32, 8u32);
    let n = (w * h) as usize;
    let left = |px: usize, c: usize| 0.1 + px as f32 / n as f32 + c as f32 * 0.2;
    let right = |px: usize, c: usize| 2.0 - px as f32 / n as f32 - c as f32 * 0.3;
    let ctx = ctx();
    let dir = tempdir();
    let l = dir.join("left.exr");
    let r = dir.join("right.exr");
    let mv = dir.join("mv.exr");
    std::fs::write(&l, encode(&ctx, frame(w, h, 3, left), w, h, &[])).unwrap();
    std::fs::write(&r, encode(&ctx, frame(w, h, 3, right), w, h, &[])).unwrap();
    let status = Command::new("exrmultiview")
        .args(["left", l.to_str().unwrap(), "right", r.to_str().unwrap()])
        .arg(&mv)
        .status()
        .unwrap();
    assert!(status.success(), "exrmultiview failed");
    let bytes = std::fs::read(&mv).unwrap();

    // Enumeration: base layer = left view, `right` layer = right view.
    let img = parse_exr(&bytes).unwrap();
    let layers = img.layers();
    assert_eq!(layers.len(), 2, "{layers:?}");
    assert_eq!(layers[0].name, "");
    assert_eq!(layers[0].kind, LayerKind::Rgb);
    assert_eq!(layers[0].view.as_deref(), Some("left"));
    assert_eq!(layers[1].name, "right");
    assert_eq!(layers[1].kind, LayerKind::Rgb);
    assert_eq!(layers[1].view.as_deref(), Some("right"));

    // The reference wrote the file with its own (lossless PIZ / HALF or
    // FLOAT) choices; compare to the half-rounded source at worst.
    let tol = |v: f32| 1e-3 * (1.0 + v.abs());
    let vf = decode(&ctx, &bytes, "").unwrap();
    let vf_named = decode(&ctx, &bytes, "left").unwrap();
    assert_eq!(vf.planes[0].data, vf_named.planes[0].data);
    for px in 0..n {
        for c in 0..3 {
            let expect = left(px, c);
            let got = f32_at(&vf, px, 3, c);
            assert!(
                (got - expect).abs() <= tol(expect),
                "left px{px} c{c} {got} vs {expect}"
            );
        }
    }
    let vf = decode(&ctx, &bytes, "right").unwrap();
    for px in 0..n {
        for c in 0..3 {
            let expect = right(px, c);
            let got = f32_at(&vf, px, 3, c);
            assert!(
                (got - expect).abs() <= tol(expect),
                "right px{px} c{c} {got} vs {expect}"
            );
        }
    }
    assert!(decode(&ctx, &bytes, "centre").is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn our_layer_prefixed_file_is_read_back_by_the_reference_header_tool() {
    if !tool_available("exrheader") {
        eprintln!("exrheader not available, skipping");
        return;
    }
    let (w, h) = (4u32, 4u32);
    let ctx = ctx();
    let bytes = encode(
        &ctx,
        frame(w, h, 3, |px, c| px as f32 + c as f32),
        w,
        h,
        &[("layer", "beauty.diffuse"), ("colour", "luma_chroma")],
    );
    let dir = tempdir();
    let path = dir.join("layered.exr");
    std::fs::write(&path, &bytes).unwrap();
    let out = Command::new("exrheader").arg(&path).output().unwrap();
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout);
    for name in ["beauty.diffuse.BY", "beauty.diffuse.RY", "beauty.diffuse.Y"] {
        assert!(text.contains(name), "{name} missing from\n{text}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
