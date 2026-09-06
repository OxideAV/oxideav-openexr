//! Framework-path luminance/chroma encode validated against the
//! reference tools as opaque processes: a frame encoded with
//! `colour=luma_chroma` must be accepted by the reference (`exrheader`
//! reports the sub-sampled `RY` / `BY`, `exr2aces` converts it), and
//! the reference's ACES conversion of that file must agree with its
//! conversion of the same frame written as plain RGB — which is the
//! reference reading *our* chroma through its own reconstruction.
//! Auto-skips when the binaries are absent.
#![cfg(feature = "registry")]

use std::path::Path;
use std::process::Command;

use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, RuntimeContext, VideoFrame};
use oxideav_openexr::{luminance_weights_of, parse_exr, ExrImage};

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
        "oxideav-openexr-reg-lumachroma-{nanos}-{}-{seq}",
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

fn encode(ctx: &RuntimeContext, vf: VideoFrame, w: u32, h: u32, opts: &[(&str, &str)]) -> Vec<u8> {
    let cid = CodecId::new(oxideav_openexr::CODEC_ID_STR);
    let mut params = CodecParameters::video(cid.clone());
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

fn plane<'a>(img: &'a ExrImage, name: &str) -> &'a [f32] {
    let idx = img.channels.iter().position(|c| c.name == name).unwrap();
    &img.planes[idx].samples
}

fn exr2aces(input: &Path, output: &Path) -> bool {
    Command::new("exr2aces")
        .arg(input)
        .arg(output)
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[test]
fn registry_luma_chroma_encode_agrees_with_the_rgb_path_through_the_reference() {
    if !tool_available("exr2aces") || !tool_available("exrheader") {
        eprintln!("reference tools not available, skipping");
        return;
    }
    let (w, h) = (32u32, 16u32);
    let n = (w * h) as usize;
    let mut data = Vec::with_capacity(n * 12);
    for y in 0..h as usize {
        for x in 0..w as usize {
            let fx = x as f32 / (w - 1) as f32;
            let fy = y as f32 / (h - 1) as f32;
            for v in [0.1 + 0.8 * fx, 0.2 + 0.6 * fy, 0.9 - 0.7 * fx * fy] {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let vf = VideoFrame {
        pts: None,
        planes: vec![oxideav_core::VideoPlane {
            stride: w as usize * 12,
            data,
        }],
    };
    let ctx = ctx();
    let yc = encode(
        &ctx,
        vf.clone(),
        w,
        h,
        &[("colour", "luma_chroma"), ("compression", "zip")],
    );
    let rgb = encode(&ctx, vf, w, h, &[("compression", "zip")]);
    let dir = tempdir();
    let yc_path = dir.join("yc.exr");
    let rgb_path = dir.join("rgb.exr");
    std::fs::write(&yc_path, &yc).unwrap();
    std::fs::write(&rgb_path, &rgb).unwrap();

    let out = Command::new("exrheader").arg(&yc_path).output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        text.contains("RY, 32-bit floating-point, sampling 2 2"),
        "{text}"
    );
    assert!(
        text.contains("BY, 32-bit floating-point, sampling 2 2"),
        "{text}"
    );

    let yc_out = dir.join("yc_aces.exr");
    let rgb_out = dir.join("rgb_aces.exr");
    assert!(
        exr2aces(&yc_path, &yc_out),
        "reference rejected our luma/chroma file"
    );
    assert!(exr2aces(&rgb_path, &rgb_out));
    let ya = parse_exr(&std::fs::read(&yc_out).unwrap()).unwrap();
    let ra = parse_exr(&std::fs::read(&rgb_out).unwrap()).unwrap();
    let wa = luminance_weights_of(&ya.attributes);
    // Luminance agrees tightly (filter-independent) …
    for (i, &y) in plane(&ya, "Y").iter().enumerate() {
        let expect =
            wa[0] * plane(&ra, "R")[i] + wa[1] * plane(&ra, "G")[i] + wa[2] * plane(&ra, "B")[i];
        assert!((y - expect).abs() <= 5e-3, "Y px{i}: {y} vs {expect}");
    }
    // … and the reference's chroma of our file matches the chroma it
    // derives from the RGB file at interior sample positions.
    let mine = oxideav_openexr::rgb_to_luma_chroma(
        w,
        h,
        [plane(&ra, "R"), plane(&ra, "G"), plane(&ra, "B")],
        wa,
        (1, 1),
    )
    .unwrap();
    let pw = (w / 2) as usize;
    for jy in 1..(h / 2) as usize - 1 {
        for jx in 1..pw - 1 {
            let j = jy * pw + jx;
            let i = (2 * jy) * w as usize + 2 * jx;
            assert!(
                (plane(&ya, "RY")[j] - mine.ry[i]).abs() <= 0.03,
                "RY sample ({jx},{jy}): {} vs {}",
                plane(&ya, "RY")[j],
                mine.ry[i]
            );
            assert!(
                (plane(&ya, "BY")[j] - mine.by[i]).abs() <= 0.03,
                "BY sample ({jx},{jy}): {} vs {}",
                plane(&ya, "BY")[j],
                mine.by[i]
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
