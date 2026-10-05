//! Fuzz the `openexr` container (registry feature): the magic probe, the
//! demuxer's header walk + multi-part repack, the registry decoder on
//! every emitted packet, and the muxer's multi-part combine on the
//! demuxed packets. Every input must end in `Ok` or `Err`, never a panic,
//! a debug overflow or an attacker-sized allocation.
//!
//! Two modes, selected by the first byte:
//!
//! * raw: the remaining bytes are the file;
//! * overlay: the remaining bytes are spliced over a writer-built base
//!   (single-part scanline / tiled, or a two-part multi-part file) so the
//!   chunk walk, offset tables and the repack are reached past the
//!   header parser.
#![no_main]

use std::io::Cursor;

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Error, Packet, RuntimeContext, StreamInfo, TimeBase};
use oxideav_openexr::{Compression, EncodeOptions, ExrImage, ExrPixelFormat, Frame};

fn ctx() -> RuntimeContext {
    let mut ctx = RuntimeContext::new();
    oxideav_openexr::register(&mut ctx);
    ctx
}

fn base(kind: u8) -> Vec<u8> {
    let img = |w: u32, h: u32, f: ExrPixelFormat, salt: f32| {
        let n = match f {
            ExrPixelFormat::GrayF32Le => 1,
            ExrPixelFormat::RgbF32Le => 3,
            _ => 4,
        };
        let samples: Vec<f32> = (0..(w * h) as usize * n)
            .map(|i| i as f32 * 0.03 + salt)
            .collect();
        ExrImage::from_f32(w, h, f, &samples).unwrap()
    };
    match kind % 4 {
        0 => oxideav_openexr::encode(
            &img(9, 7, ExrPixelFormat::RgbaF32Le, 0.0),
            &EncodeOptions::default().with_compression(Compression::Zip),
        )
        .unwrap(),
        1 => oxideav_openexr::encode(
            &img(9, 7, ExrPixelFormat::RgbF32Le, 0.0),
            &EncodeOptions::default().with_tile_size(4),
        )
        .unwrap(),
        2 => oxideav_openexr::encode_all(
            &[
                Frame::new(img(6, 5, ExrPixelFormat::RgbF32Le, 0.0), 0)
                    .with_name(Some("a".to_string())),
                Frame::new(img(4, 4, ExrPixelFormat::GrayF32Le, 1.0), 1)
                    .with_name(Some("b".to_string())),
            ],
            &EncodeOptions::default().with_compression(Compression::Piz),
        )
        .unwrap(),
        _ => oxideav_openexr::encode_all(
            &[
                Frame::new(img(6, 5, ExrPixelFormat::RgbaF32Le, 0.0), 0)
                    .with_name(Some("a".to_string())),
                Frame::new(img(6, 5, ExrPixelFormat::RgbaF32Le, 2.0), 1)
                    .with_name(Some("b".to_string())),
            ],
            &EncodeOptions::default(),
        )
        .unwrap(),
    }
}

fn drive(ctx: &RuntimeContext, file: &[u8]) {
    let _ = ctx
        .containers
        .probe_input(&mut Cursor::new(file), Some("exr"));
    let Ok(mut demuxer) = ctx.containers.open_demuxer(
        "openexr",
        Box::new(Cursor::new(file.to_vec())),
        &ctx.codecs,
    ) else {
        return;
    };
    let stream = demuxer.streams()[0].clone();
    let mut dec = ctx.codecs.first_decoder(&stream.params).ok();
    let mut packets = Vec::new();
    loop {
        match demuxer.next_packet() {
            Ok(pkt) => {
                if let Some(d) = dec.as_mut() {
                    if d.send_packet(&pkt).is_ok() {
                        let _ = d.receive_frame();
                    }
                }
                packets.push(pkt);
                if packets.len() > 8 {
                    break;
                }
            }
            Err(Error::Eof) | Err(_) => break,
        }
    }
    let _ = demuxer.metadata();
    // Re-mux what came out (plus a hostile packet) into a multi-part file.
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 1),
        duration: None,
        start_time: Some(0),
        params: CodecParameters::video(CodecId::new(oxideav_openexr::CODEC_ID_STR)),
    };
    let Ok(mut muxer) = ctx.containers.open_muxer(
        "openexr",
        Box::new(Cursor::new(Vec::<u8>::new())),
        &[stream],
    ) else {
        return;
    };
    let _ = muxer.write_header();
    for p in &packets {
        let _ = muxer.write_packet(p);
    }
    let _ = muxer.write_packet(&Packet::new(0, TimeBase::new(1, 1), file.to_vec()));
    let _ = muxer.write_trailer();
}

fuzz_target!(|data: &[u8]| {
    if data.len() > 1 << 20 {
        return;
    }
    let ctx = ctx();
    let Some((&mode, rest)) = data.split_first() else {
        return;
    };
    if mode & 0x80 == 0 {
        drive(&ctx, rest);
        return;
    }
    let mut file = base(mode);
    // Overlay: `rest` = [u16 LE offset][bytes…] spliced over the base.
    if rest.len() >= 2 {
        let off = u16::from_le_bytes([rest[0], rest[1]]) as usize % file.len().max(1);
        let patch = &rest[2..];
        let end = (off + patch.len()).min(file.len());
        file[off..end].copy_from_slice(&patch[..end - off]);
        if end - off < patch.len() {
            file.extend_from_slice(&patch[end - off..]);
        }
    }
    drive(&ctx, &file);
});
