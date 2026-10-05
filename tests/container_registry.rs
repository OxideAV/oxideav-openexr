//! The `openexr` container through the `oxideav-core` registries: probe,
//! demuxer (single- and multi-part), muxer (single- and multi-part), and
//! the Layer-1-vs-registry byte-exact matrix (`IMAGE_CRATE_API`, Layer 2
//! acceptance for a container).
#![cfg(feature = "registry")]

use std::io::{Cursor, Seek, SeekFrom, Write};
use std::process::Command;
use std::sync::{Arc, Mutex};

use oxideav_core::{
    CodecId, CodecParameters, Error, Frame as CoreFrame, Packet, PixelFormat, RuntimeContext,
    StreamInfo, TimeBase, VideoFrame,
};
use oxideav_openexr::{
    encode_exr_multipart_mixed, Channel, ColourLayout, Compression, DecodeOptions, EncodeOptions,
    ExrImage, ExrPixelFormat, Frame, LevelMode, MultipartMixedPart, PixelType, CODEC_ID_STR,
};

const CONTAINER: &str = "openexr";

/// A `Send + 'static` in-memory sink the muxer can own while the test
/// keeps a handle to read the bytes back.
#[derive(Clone, Default)]
struct SharedSink(Arc<Mutex<Cursor<Vec<u8>>>>);

impl SharedSink {
    fn bytes(&self) -> Vec<u8> {
        self.0.lock().unwrap().get_ref().clone()
    }
}

impl Write for SharedSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0.lock().unwrap().flush()
    }
}

impl Seek for SharedSink {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.0.lock().unwrap().seek(pos)
    }
}

fn ctx() -> RuntimeContext {
    let mut ctx = RuntimeContext::new();
    oxideav_openexr::register(&mut ctx);
    ctx
}

fn comps(format: ExrPixelFormat) -> usize {
    match format {
        ExrPixelFormat::GrayF32Le => 1,
        ExrPixelFormat::RgbF32Le => 3,
        ExrPixelFormat::RgbaF32Le => 4,
        _ => unreachable!("the contract layouts"),
    }
}

/// Scene-referred signal: speculars above 1, negative excursions, a ramp.
fn image(w: u32, h: u32, format: ExrPixelFormat, salt: f32) -> ExrImage {
    let n = comps(format);
    let samples: Vec<f32> = (0..(w * h) as usize)
        .flat_map(|px| {
            (0..n).map(move |c| {
                let t = px as f32 / (w * h) as f32;
                let base = 0.125 + t * 1.5 + c as f32 * 0.375 + salt;
                if px % 7 == 0 {
                    base * 12.0
                } else if px % 11 == 0 {
                    -base * 0.5
                } else {
                    base
                }
            })
        })
        .collect();
    ExrImage::from_f32(w, h, format, &samples).unwrap()
}

/// What one registry pass produced: the stream table, the packets in
/// file order, and the frame each packet decoded to (same order).
type Demuxed = (Vec<StreamInfo>, Vec<Packet>, Vec<VideoFrame>);

/// Open `bytes` through the registry and pump every packet through the
/// decoder of its own stream.
fn demux_decode(ctx: &RuntimeContext, bytes: &[u8]) -> Demuxed {
    let name = ctx
        .containers
        .probe_input(&mut Cursor::new(bytes), None)
        .expect("probe");
    assert_eq!(name, CONTAINER);
    let mut demuxer = ctx
        .containers
        .open_demuxer(&name, Box::new(Cursor::new(bytes.to_vec())), &ctx.codecs)
        .expect("open_demuxer");
    assert_eq!(demuxer.format_name(), CONTAINER);
    let streams = demuxer.streams().to_vec();
    assert!(!streams.is_empty(), "at least one video stream");
    let mut decoders = Vec::new();
    for (i, stream) in streams.iter().enumerate() {
        assert_eq!(stream.index as usize, i);
        assert_eq!(stream.params.codec_id, CodecId::new(CODEC_ID_STR));
        assert_eq!(stream.time_base, TimeBase::new(1, 1));
        assert!(stream.params.pixel_format.is_some());
        decoders.push(
            ctx.codecs
                .first_decoder(&stream.params)
                .expect("first_decoder"),
        );
    }
    let mut packets = Vec::new();
    let mut frames = Vec::new();
    loop {
        match demuxer.next_packet() {
            Ok(pkt) => {
                assert!(pkt.flags.keyframe);
                let s = pkt.stream_index as usize;
                assert!(s < streams.len(), "packet on an undeclared stream");
                let dec = &mut decoders[s];
                dec.send_packet(&pkt).expect("send_packet");
                match dec.receive_frame().expect("receive_frame") {
                    CoreFrame::Video(v) => {
                        // The frame's plane is exactly what its own
                        // stream's parameters describe.
                        let p = &streams[s].params;
                        let fmt = p.pixel_format.unwrap();
                        let (w, h) = (p.width.unwrap(), p.height.unwrap());
                        let (pw, ph) = fmt.plane_dimensions(0, w, h).unwrap();
                        let row = fmt.plane_row_bytes(0, pw).unwrap();
                        assert_eq!(v.planes[0].stride, row, "stride vs stream {s}");
                        assert_eq!(
                            v.planes[0].data.len(),
                            row * ph as usize,
                            "plane size vs stream {s} ({fmt:?} {w}×{h})"
                        );
                        frames.push(v)
                    }
                    _ => panic!("expected a video frame"),
                }
                packets.push(pkt);
            }
            Err(Error::Eof) => break,
            Err(e) => panic!("next_packet: {e}"),
        }
    }
    assert!(matches!(demuxer.next_packet(), Err(Error::Eof)));
    (streams, packets, frames)
}

/// Encode `frames` through the registry encoder and the muxer.
fn registry_mux(ctx: &RuntimeContext, format: PixelFormat, images: &[ExrImage]) -> Vec<u8> {
    let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
    params.width = Some(images[0].width);
    params.height = Some(images[0].height);
    params.pixel_format = Some(format);
    let mut enc = ctx.codecs.first_encoder(&params).expect("encoder");
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 1),
        duration: None,
        start_time: Some(0),
        params: enc.output_params().clone(),
    };
    let out = SharedSink::default();
    {
        let mut muxer = ctx
            .containers
            .open_muxer(CONTAINER, Box::new(out.clone()), &[stream])
            .expect("open_muxer");
        assert_eq!(muxer.format_name(), CONTAINER);
        muxer.write_header().unwrap();
        for (i, img) in images.iter().enumerate() {
            let mut vf: VideoFrame = img.clone().into();
            vf.pts = Some(i as i64);
            enc.send_frame(&CoreFrame::Video(vf)).unwrap();
            let mut pkt = enc.receive_packet().unwrap();
            pkt.pts = Some(i as i64);
            muxer.write_packet(&pkt).unwrap();
        }
        muxer.write_trailer().unwrap();
    }
    out.bytes()
}

// ---- probe ----------------------------------------------------------------

#[test]
fn probe_names_the_container_from_magic_and_extension() {
    let ctx = ctx();
    let bytes = oxideav_openexr::encode(
        &image(3, 2, ExrPixelFormat::RgbF32Le, 0.0),
        &EncodeOptions::default(),
    )
    .unwrap();
    assert_eq!(
        ctx.containers
            .probe_input(&mut Cursor::new(&bytes), None)
            .unwrap(),
        CONTAINER
    );
    assert_eq!(
        ctx.containers
            .probe_input(&mut Cursor::new(&bytes), Some("exr"))
            .unwrap(),
        CONTAINER
    );
    // Foreign files: farbfeld, PNG, Netpbm, garbage.
    let mut farbfeld = b"farbfeld".to_vec();
    farbfeld.extend_from_slice(&1u32.to_be_bytes());
    farbfeld.extend_from_slice(&1u32.to_be_bytes());
    farbfeld.extend_from_slice(&[0u8; 8]);
    let png = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13];
    for foreign in [
        farbfeld.as_slice(),
        &png,
        b"P6\n1 1\n255\n\0\0\0",
        &[0u8; 64],
    ] {
        assert!(
            ctx.containers
                .probe_input(&mut Cursor::new(foreign), None)
                .is_err(),
            "foreign bytes must not probe as openexr"
        );
        assert!(ctx
            .containers
            .probe_input(&mut Cursor::new(foreign), Some("ff"))
            .is_err());
    }
    assert!(ctx.containers.demuxer_names().any(|n| n == CONTAINER));
    assert!(ctx.containers.muxer_names().any(|n| n == CONTAINER));
    assert_eq!(
        ctx.containers.container_for_extension("exr"),
        Some(CONTAINER)
    );
}

// ---- single-part demux: Layer 1 vs registry -------------------------------

#[test]
fn single_part_demux_matches_layer1_across_layouts_and_compressions() {
    let ctx = ctx();
    let layouts = [
        (ExrPixelFormat::RgbaF32Le, PixelFormat::RgbaF32Le),
        (ExrPixelFormat::RgbF32Le, PixelFormat::RgbF32Le),
        (ExrPixelFormat::GrayF32Le, PixelFormat::GrayF32Le),
    ];
    let codecs = [
        Compression::None,
        Compression::Rle,
        Compression::Zips,
        Compression::Zip,
        Compression::Piz,
        Compression::Pxr24,
        Compression::B44,
        Compression::B44a,
        Compression::Dwaa,
        Compression::Dwab,
    ];
    let mut pinned = 0;
    for &(fmt, core_fmt) in &layouts {
        for &c in &codecs {
            for pixel_type in [PixelType::Float, PixelType::Half] {
                for tile in [0u32, 8] {
                    let opts = EncodeOptions::default()
                        .with_compression(c)
                        .with_pixel_type(pixel_type)
                        .with_tile_size(tile);
                    let bytes = oxideav_openexr::encode(&image(37, 19, fmt, 0.0), &opts).unwrap();
                    let info = oxideav_openexr::info(&bytes).unwrap();
                    let l1 = oxideav_openexr::decode(&bytes).unwrap();
                    let (streams, packets, frames) = demux_decode(&ctx, &bytes);
                    let label = format!("{fmt:?}/{c:?}/{pixel_type:?}/tile{tile}");
                    assert_eq!(streams.len(), 1, "{label}");
                    let stream = &streams[0];
                    assert_eq!(stream.params.width, Some(info.width), "{label}");
                    assert_eq!(stream.params.height, Some(info.height), "{label}");
                    assert_eq!(stream.params.pixel_format, Some(core_fmt), "{label}");
                    assert_eq!(info.format, fmt, "{label}");
                    assert_eq!(packets.len(), 1, "{label}");
                    assert_eq!(packets[0].pts, Some(0), "{label}");
                    assert_eq!(packets[0].data, bytes, "{label}: whole file is the packet");
                    assert_eq!(frames.len(), 1, "{label}");
                    assert_eq!(frames[0].image_planes().len(), 1, "{label}");
                    assert_eq!(
                        frames[0].planes[0].data, l1.planes[0].data,
                        "{label}: registry planes == Layer 1 planes"
                    );
                    assert_eq!(frames[0].planes[0].stride, l1.planes[0].stride, "{label}");
                    pinned += 1;
                }
            }
        }
    }
    assert_eq!(pinned, 3 * 10 * 2 * 2);
}

#[test]
fn single_part_luma_chroma_and_colour_signal() {
    let ctx = ctx();
    let img = image(16, 12, ExrPixelFormat::RgbaF32Le, 0.0).with_chromaticities(
        oxideav_openexr::ColorInfo::chromaticities_for(
            oxideav_openexr::ColorInfo::PRIMARIES_BT2020,
        )
        .unwrap(),
    );
    let bytes = oxideav_openexr::encode(
        &img,
        &EncodeOptions::default()
            .with_colour(ColourLayout::LumaChroma)
            .with_chroma_sampling(2),
    )
    .unwrap();
    let info = oxideav_openexr::info(&bytes).unwrap();
    assert_eq!(info.color.primaries, 9, "chromaticities written and read");
    let l1 = oxideav_openexr::decode(&bytes).unwrap();
    let (streams, _packets, frames) = demux_decode(&ctx, &bytes);
    let stream = &streams[0];
    assert_eq!(stream.params.pixel_format, Some(PixelFormat::RgbaF32Le));
    let sig = stream.params.color_signal;
    assert_eq!(
        sig,
        oxideav_openexr::registry::to_color_signal(&info.color),
        "stream colour signal comes from the file's chromaticities"
    );
    assert_eq!(frames[0].planes[0].data, l1.planes[0].data);
    assert_eq!(frames[0].color_signal(), Some(sig));

    // A file without chromaticities carries the format default (BT.709).
    let plain = oxideav_openexr::encode(
        &image(4, 4, ExrPixelFormat::GrayF32Le, 0.0),
        &EncodeOptions::default(),
    )
    .unwrap();
    let (streams, ..) = demux_decode(&ctx, &plain);
    assert_eq!(
        streams[0].params.color_signal,
        oxideav_openexr::registry::to_color_signal(&oxideav_openexr::ColorInfo::exr_default())
    );
}

// ---- multi-part demux -----------------------------------------------------

fn frames_for_multipart() -> Vec<Frame> {
    vec![
        Frame::new(image(12, 9, ExrPixelFormat::RgbaF32Le, 0.0), 0)
            .with_name(Some("beauty".into())),
        Frame::new(image(7, 5, ExrPixelFormat::RgbF32Le, 1.0), 1).with_name(Some("diffuse".into())),
        Frame::new(image(9, 9, ExrPixelFormat::GrayF32Le, 2.0), 2).with_name(Some("depth".into())),
    ]
}

#[test]
fn multipart_demux_one_packet_per_part_matches_decode_all() {
    let ctx = ctx();
    for c in [
        Compression::None,
        Compression::Zip,
        Compression::Piz,
        Compression::B44a,
    ] {
        let bytes = oxideav_openexr::encode_all(
            &frames_for_multipart(),
            &EncodeOptions::default().with_compression(c),
        )
        .unwrap();
        let info = oxideav_openexr::info(&bytes).unwrap();
        assert_eq!(info.frames, 3);
        let l1 = oxideav_openexr::decode_all(&bytes).unwrap();
        let (streams, packets, frames) = demux_decode(&ctx, &bytes);
        // Three parts of three distinct (geometry, layout): three streams,
        // each describing its own part.
        assert_eq!(streams.len(), 3, "{c:?}");
        assert_eq!(streams[0].params.width, Some(info.width));
        assert_eq!(streams[0].params.height, Some(info.height));
        for (k, (fmt, (w, h))) in [
            (PixelFormat::RgbaF32Le, (12, 9)),
            (PixelFormat::RgbF32Le, (7, 5)),
            (PixelFormat::GrayF32Le, (9, 9)),
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                streams[k].params.pixel_format,
                Some(fmt),
                "{c:?} stream {k}"
            );
            assert_eq!(
                (streams[k].params.width, streams[k].params.height),
                (Some(w), Some(h)),
                "{c:?} stream {k}"
            );
            assert_eq!(packets[k].stream_index, k as u32);
        }
        assert_eq!(packets.len(), 3, "{c:?}");
        assert_eq!(frames.len(), 3, "{c:?}");
        for (i, (pkt, vf)) in packets.iter().zip(&frames).enumerate() {
            assert_eq!(pkt.pts, Some(i as i64), "{c:?}: pts = part index");
            assert_eq!(pkt.dts, Some(i as i64));
            assert_eq!(pkt.duration, None, "parts are not timed");
            assert_eq!(vf.pts, Some(i as i64));
            assert_eq!(l1[i].index, i as u32);
            assert_eq!(
                vf.planes[0].data, l1[i].image.planes[0].data,
                "{c:?}: part {i} planes == decode_all"
            );
            // Every packet is a valid single-part file on its own, with
            // the part's view, attributes and compression intact.
            assert!(oxideav_openexr::probe(&pkt.data));
            let single = oxideav_openexr::info(&pkt.data).unwrap();
            assert!(!single.multipart);
            assert_eq!(single.frames, 1);
            assert_eq!(single.compression, c);
            assert_eq!(single.part_name, l1[i].name);
            assert_eq!(
                oxideav_openexr::decode(&pkt.data).unwrap().planes,
                l1[i].image.planes
            );
        }
    }
}

#[test]
fn multipart_demux_metadata_carries_part_names() {
    let ctx = ctx();
    let bytes =
        oxideav_openexr::encode_all(&frames_for_multipart(), &EncodeOptions::default()).unwrap();
    let mut demuxer = ctx
        .containers
        .open_demuxer(CONTAINER, Box::new(Cursor::new(bytes)), &ctx.codecs)
        .unwrap();
    let md = demuxer.metadata().to_vec();
    assert!(md.contains(&("parts".to_string(), "3".to_string())));
    assert!(md.contains(&("part_name:0".to_string(), "beauty".to_string())));
    assert!(md.contains(&("part_name:2".to_string(), "depth".to_string())));
    assert!(demuxer.next_packet().is_ok());
}

fn rgba_float_channels() -> Vec<Channel> {
    ["A", "B", "G", "R"]
        .iter()
        .map(|n| Channel {
            name: (*n).to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        })
        .collect()
}

fn planes(w: u32, h: u32, salt: f32) -> Vec<Vec<f32>> {
    (0..4)
        .map(|c| {
            (0..(w * h) as usize)
                .map(|i| i as f32 * 0.25 + c as f32 + salt)
                .collect()
        })
        .collect()
}

#[test]
fn multipart_demux_skips_deep_and_viewless_parts_like_decode_all() {
    let ctx = ctx();
    let (w, h) = (8u32, 6u32);
    let p0 = planes(w, h, 0.0);
    let p2 = planes(w, h, 5.0);
    // Part 1: a channel set with no colour view (AOV-only).
    let aov: Vec<f32> = (0..(w * h) as usize).map(|i| i as f32).collect();
    // Part 3: a deep scanline part.
    let deep_counts = vec![1u32; (w * h) as usize];
    let deep_z: Vec<f32> = (0..(w * h) as usize).map(|i| i as f32 * 0.5).collect();
    let bytes = encode_exr_multipart_mixed(&[
        MultipartMixedPart::Scanline {
            name: "rgba".to_string(),
            width: w,
            height: h,
            channels: rgba_float_channels(),
            planes: p0.iter().map(|v| v.as_slice()).collect(),
            compression: Compression::Zips,
        },
        MultipartMixedPart::Scanline {
            name: "aov".to_string(),
            width: w,
            height: h,
            channels: vec![Channel {
                name: "N.x".to_string(),
                pixel_type: PixelType::Float,
                p_linear: false,
                x_sampling: 1,
                y_sampling: 1,
            }],
            planes: vec![&aov],
            compression: Compression::None,
        },
        MultipartMixedPart::Tiled {
            name: "tiled".to_string(),
            width: w,
            height: h,
            tile_x: 4,
            tile_y: 4,
            channels: rgba_float_channels(),
            planes: p2.iter().map(|v| v.as_slice()).collect(),
            compression: Compression::Zip,
        },
        MultipartMixedPart::DeepScanline {
            name: "deep".to_string(),
            width: w,
            height: h,
            channels: vec![Channel {
                name: "Z".to_string(),
                pixel_type: PixelType::Float,
                p_linear: false,
                x_sampling: 1,
                y_sampling: 1,
            }],
            samples_per_pixel: &deep_counts,
            channel_samples: vec![&deep_z],
            compression: Compression::Zips,
        },
    ])
    .unwrap();
    let l1 = oxideav_openexr::decode_all(&bytes).unwrap();
    assert_eq!(l1.len(), 2);
    assert_eq!((l1[0].index, l1[1].index), (0, 2));
    let (streams, packets, frames) = demux_decode(&ctx, &bytes);
    assert_eq!(
        streams.len(),
        1,
        "both viewable parts are 8×6 RgbaF32Le: one shared stream"
    );
    assert_eq!(streams[0].params.pixel_format, Some(PixelFormat::RgbaF32Le));
    assert_eq!(packets.len(), 2, "aov-only and deep parts are skipped");
    assert!(packets.iter().all(|p| p.stream_index == 0));
    assert_eq!(packets[0].pts, Some(0));
    assert_eq!(packets[1].pts, Some(2), "pts keeps the file part index");
    assert_eq!(frames[0].planes[0].data, l1[0].image.planes[0].data);
    assert_eq!(frames[1].planes[0].data, l1[1].image.planes[0].data);
    // The tiled part repacks as a single-part tiled file.
    assert!(oxideav_openexr::info(&packets[1].data).unwrap().tiled);

    // Strict Layer 1 refuses the skip; the demuxer follows the lenient default.
    assert!(matches!(
        oxideav_openexr::decode_all_with(&bytes, &DecodeOptions::default().with_strict(true)),
        Err(oxideav_openexr::ExrError::Unsupported(_))
    ));
}

#[test]
fn multipart_demux_multilevel_tiled_part_decodes_level_zero() {
    let ctx = ctx();
    // A single-part MIPMAP file combined into a multi-part file through
    // the muxer, then demuxed: the multi-level part must repack with its
    // complete tile table and decode to level (0, 0).
    let mip = oxideav_openexr::encode(
        &image(16, 16, ExrPixelFormat::RgbF32Le, 0.0),
        &EncodeOptions::default()
            .with_tile_size(4)
            .with_levels(LevelMode::Mipmap),
    )
    .unwrap();
    let flat = oxideav_openexr::encode(
        &image(16, 16, ExrPixelFormat::RgbF32Le, 3.0),
        &EncodeOptions::default(),
    )
    .unwrap();
    let sink = SharedSink::default();
    {
        let mut params = CodecParameters::video(CodecId::new(CODEC_ID_STR));
        params.width = Some(16);
        params.height = Some(16);
        params.pixel_format = Some(PixelFormat::RgbF32Le);
        let stream = StreamInfo {
            index: 0,
            time_base: TimeBase::new(1, 1),
            duration: None,
            start_time: Some(0),
            params,
        };
        let mut muxer = ctx
            .containers
            .open_muxer(CONTAINER, Box::new(sink.clone()), &[stream])
            .unwrap();
        muxer.write_header().unwrap();
        for (i, f) in [&mip, &flat].iter().enumerate() {
            let mut pkt = Packet::new(0, TimeBase::new(1, 1), (*f).clone());
            pkt.pts = Some(i as i64);
            muxer.write_packet(&pkt).unwrap();
        }
        muxer.write_trailer().unwrap();
    }
    let out = sink.bytes();
    let l1 = oxideav_openexr::decode_all(&out).unwrap();
    assert_eq!(l1.len(), 2);
    assert_eq!(
        l1[0].image.planes,
        oxideav_openexr::decode(&mip).unwrap().planes
    );
    let (_, packets, frames) = demux_decode(&ctx, &out);
    assert_eq!(packets.len(), 2);
    assert_eq!(frames[0].planes[0].data, l1[0].image.planes[0].data);
    assert_eq!(frames[1].planes[0].data, l1[1].image.planes[0].data);
    assert!(oxideav_openexr::parse_exr_tiled_multilevel(&packets[0].data).is_ok());
}

// ---- mux ------------------------------------------------------------------

#[test]
fn mux_single_packet_writes_the_encoder_file_verbatim() {
    let ctx = ctx();
    for (fmt, core) in [
        (ExrPixelFormat::RgbaF32Le, PixelFormat::RgbaF32Le),
        (ExrPixelFormat::RgbF32Le, PixelFormat::RgbF32Le),
        (ExrPixelFormat::GrayF32Le, PixelFormat::GrayF32Le),
    ] {
        let img = image(11, 7, fmt, 0.5);
        let out = registry_mux(&ctx, core, std::slice::from_ref(&img));
        let back = oxideav_openexr::decode(&out).unwrap();
        assert_eq!(
            back.planes, img.planes,
            "{fmt:?}: lossless FLOAT round trip"
        );
        assert_eq!(back.format, fmt);
        let info = oxideav_openexr::info(&out).unwrap();
        assert!(!info.multipart);
        assert_eq!(info.frames, 1);
        // demux(mux(frame)) == frame through the registry too.
        let (_, _, frames) = demux_decode(&ctx, &out);
        assert_eq!(frames[0].planes[0].data, img.planes[0].data);
    }
}

#[test]
fn mux_several_packets_writes_a_multipart_file_decode_all_reads_back() {
    let ctx = ctx();
    let images = [
        image(10, 6, ExrPixelFormat::RgbF32Le, 0.0),
        image(10, 6, ExrPixelFormat::RgbF32Le, 1.0),
        image(10, 6, ExrPixelFormat::RgbF32Le, 2.0),
    ];
    let out = registry_mux(&ctx, PixelFormat::RgbF32Le, &images);
    let info = oxideav_openexr::info(&out).unwrap();
    assert!(info.multipart);
    assert_eq!(info.frames, 3);
    let l1 = oxideav_openexr::decode_all(&out).unwrap();
    assert_eq!(l1.len(), 3);
    for (i, f) in l1.iter().enumerate() {
        assert_eq!(f.index, i as u32);
        assert_eq!(f.image.planes, images[i].planes, "part {i} lossless");
        assert_eq!(f.name.as_deref(), Some(format!("part{i}").as_str()));
        assert_eq!(f.part_type.as_deref(), Some("scanlineimage"));
    }
    // The depth reader agrees.
    let parts = oxideav_openexr::parse_exr_multipart(&out).unwrap();
    assert_eq!(parts.len(), 3);
    // Registry round trip: demux(mux(frames)) == frames.
    let (_, packets, frames) = demux_decode(&ctx, &out);
    assert_eq!(packets.len(), 3);
    for (i, vf) in frames.iter().enumerate() {
        assert_eq!(vf.pts, Some(i as i64));
        assert_eq!(vf.planes[0].data, images[i].planes[0].data);
    }
}

#[test]
fn mux_keeps_part_names_from_the_packets_and_disambiguates_duplicates() {
    let ctx = ctx();
    let frames = frames_for_multipart();
    let multi = oxideav_openexr::encode_all(&frames, &EncodeOptions::default()).unwrap();
    // Demux the multi-part file into single-part packets, then mux them
    // again: names survive, a repeated name gets a suffix.
    let (streams, packets, _) = demux_decode(&ctx, &multi);
    assert_eq!(streams.len(), 3);
    let sink = SharedSink::default();
    {
        let mut muxer = ctx
            .containers
            .open_muxer(CONTAINER, Box::new(sink.clone()), &streams)
            .unwrap();
        muxer.write_header().unwrap();
        for pkt in &packets {
            muxer.write_packet(pkt).unwrap();
        }
        muxer.write_packet(&packets[0]).unwrap(); // duplicate "beauty"
        muxer.write_trailer().unwrap();
    }
    let out = sink.bytes();
    let l1 = oxideav_openexr::decode_all(&out).unwrap();
    let names: Vec<&str> = l1.iter().map(|f| f.name.as_deref().unwrap()).collect();
    assert_eq!(names, ["beauty", "diffuse", "depth", "beauty.3"]);
    for (f, orig) in l1
        .iter()
        .zip(frames.iter().chain(std::iter::once(&frames[0])))
    {
        assert_eq!(f.image.planes, orig.image.planes);
    }
}

#[test]
fn mux_rejects_bad_streams_and_packets() {
    let ctx = ctx();
    let video = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 1),
        duration: None,
        start_time: Some(0),
        params: CodecParameters::video(CodecId::new(CODEC_ID_STR)),
    };
    let sink = || Box::new(Cursor::new(Vec::<u8>::new()));
    assert!(ctx.containers.open_muxer(CONTAINER, sink(), &[]).is_err());
    // Several video streams are fine (one per part layout) …
    let second = StreamInfo {
        index: 1,
        ..video.clone()
    };
    let mut two = ctx
        .containers
        .open_muxer(CONTAINER, sink(), &[video.clone(), second])
        .unwrap();
    // … but a packet must name a declared stream.
    let pict = oxideav_openexr::encode(
        &image(2, 2, ExrPixelFormat::GrayF32Le, 0.0),
        &EncodeOptions::default(),
    )
    .unwrap();
    assert!(two
        .write_packet(&Packet::new(2, TimeBase::new(1, 1), pict.clone()))
        .is_err());
    two.write_packet(&Packet::new(1, TimeBase::new(1, 1), pict))
        .unwrap();
    two.write_trailer().unwrap();
    let audio = StreamInfo {
        params: CodecParameters::audio(CodecId::new("pcm")),
        ..video.clone()
    };
    assert!(ctx
        .containers
        .open_muxer(CONTAINER, sink(), &[audio])
        .is_err());

    let mut muxer = ctx
        .containers
        .open_muxer(CONTAINER, sink(), std::slice::from_ref(&video))
        .unwrap();
    muxer.write_header().unwrap();
    assert!(muxer
        .write_packet(&Packet::new(0, TimeBase::new(1, 1), Vec::new()))
        .is_err());
    assert!(muxer
        .write_packet(&Packet::new(0, TimeBase::new(1, 1), b"not an exr".to_vec()))
        .is_err());
    assert!(muxer.write_trailer().is_err(), "no packet, no file");

    // A deep single-part packet cannot combine into a multi-part file.
    let flat = oxideav_openexr::encode(
        &image(4, 4, ExrPixelFormat::GrayF32Le, 0.0),
        &EncodeOptions::default(),
    )
    .unwrap();
    let counts = vec![1u32; 16];
    let z: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let deep = oxideav_openexr::encode_exr_deep_scanline(&oxideav_openexr::DeepScanlineInput {
        width: 4,
        height: 4,
        channels: vec![Channel {
            name: "Z".to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        }],
        samples_per_pixel: &counts,
        channel_samples: vec![&z],
        compression: Compression::None,
    })
    .unwrap();
    let mut muxer = ctx
        .containers
        .open_muxer(CONTAINER, sink(), std::slice::from_ref(&video))
        .unwrap();
    muxer.write_header().unwrap();
    muxer
        .write_packet(&Packet::new(0, TimeBase::new(1, 1), flat))
        .unwrap();
    muxer
        .write_packet(&Packet::new(0, TimeBase::new(1, 1), deep))
        .unwrap();
    assert!(matches!(muxer.write_trailer(), Err(Error::Unsupported(_))));
}

// ---- registration ---------------------------------------------------------

#[test]
fn register_installs_codec_and_container() {
    let ctx = ctx();
    assert!(ctx
        .codecs
        .decoder_ids()
        .any(|c| *c == CodecId::new(CODEC_ID_STR)));
    assert!(ctx.containers.demuxer_names().any(|n| n == CONTAINER));
    assert!(ctx.containers.muxer_names().any(|n| n == CONTAINER));
    assert_eq!(
        ctx.containers.container_for_extension("exr"),
        Some(CONTAINER)
    );
    let mut via_entry = RuntimeContext::new();
    oxideav_openexr::__oxideav_entry(&mut via_entry);
    assert!(via_entry.containers.demuxer_names().any(|n| n == CONTAINER));
    assert!(via_entry.containers.muxer_names().any(|n| n == CONTAINER));
}

// ---- hostile input --------------------------------------------------------

#[test]
fn hostile_inputs_never_panic() {
    let ctx = ctx();
    let single = oxideav_openexr::encode(
        &image(5, 4, ExrPixelFormat::RgbaF32Le, 0.0),
        &EncodeOptions::default().with_compression(Compression::Zip),
    )
    .unwrap();
    let multi =
        oxideav_openexr::encode_all(&frames_for_multipart(), &EncodeOptions::default()).unwrap();
    let open = |bytes: &[u8]| {
        ctx.containers
            .open_demuxer(
                CONTAINER,
                Box::new(Cursor::new(bytes.to_vec())),
                &ctx.codecs,
            )
            .map(|mut d| while d.next_packet().is_ok() {})
    };
    assert!(open(&[]).is_err());
    assert!(open(&[0x76, 0x2f, 0x31, 0x01]).is_err());
    for src in [&single, &multi] {
        for cut in (0..src.len()).step_by(7) {
            let _ = open(&src[..cut]);
        }
        // Absurd dimensions / counts: flip bytes through the header region.
        let mut mutated = src.clone();
        for i in 8..src.len().min(600) {
            for v in [0x00, 0xff, 0x7f] {
                let keep = mutated[i];
                mutated[i] = v;
                let _ = open(&mutated);
                mutated[i] = keep;
            }
        }
    }
    // Multi-part header claiming a gigantic chunkCount: refused before
    // any allocation.
    let mut huge = multi.clone();
    let needle = b"chunkCount\0int\0";
    if let Some(p) = huge.windows(needle.len()).position(|w| w == needle) {
        let at = p + needle.len() + 4;
        huge[at..at + 4].copy_from_slice(&i32::MAX.to_le_bytes());
        assert!(open(&huge).is_err());
    }
}

// ---- per-layout streams ---------------------------------------------------

/// The coordinator's pin: RgbF32Le + GrayF32Le + RgbaF32Le parts of the
/// same 2×2 geometry → three streams, each frame sized by its own
/// stream's `plane_dimensions`, planes byte-exact vs `decode_all`, and
/// the several-streams muxer inverse round-trips.
#[test]
fn multipart_parts_of_different_layouts_get_their_own_streams() {
    let ctx = ctx();
    let frames = vec![
        Frame::new(image(2, 2, ExrPixelFormat::RgbF32Le, 0.0), 0).with_name(Some("rgb".into())),
        Frame::new(image(2, 2, ExrPixelFormat::GrayF32Le, 1.0), 1).with_name(Some("depth".into())),
        Frame::new(image(2, 2, ExrPixelFormat::RgbaF32Le, 2.0), 2).with_name(Some("rgba".into())),
        // A second grey part: shares the depth stream.
        Frame::new(image(2, 2, ExrPixelFormat::GrayF32Le, 3.0), 3).with_name(Some("mask".into())),
    ];
    let bytes = oxideav_openexr::encode_all(&frames, &EncodeOptions::default()).unwrap();
    let l1 = oxideav_openexr::decode_all(&bytes).unwrap();
    let (streams, packets, out) = demux_decode(&ctx, &bytes);
    assert_eq!(streams.len(), 3, "three distinct layouts");
    let layouts: Vec<PixelFormat> = streams
        .iter()
        .map(|s| s.params.pixel_format.unwrap())
        .collect();
    assert_eq!(
        layouts,
        [
            PixelFormat::RgbF32Le,
            PixelFormat::GrayF32Le,
            PixelFormat::RgbaF32Le
        ],
        "first-appearance order"
    );
    assert_eq!(
        packets.iter().map(|p| p.stream_index).collect::<Vec<_>>(),
        [0, 1, 2, 1],
        "the second grey part rides the grey stream"
    );
    assert_eq!(
        packets.iter().map(|p| p.pts).collect::<Vec<_>>(),
        [Some(0), Some(1), Some(2), Some(3)],
        "pts stays the file part index"
    );
    for (k, vf) in out.iter().enumerate() {
        let s = &streams[packets[k].stream_index as usize].params;
        let fmt = s.pixel_format.unwrap();
        let expected = fmt.plane_row_bytes(0, 2).unwrap() * 2;
        assert_eq!(
            vf.planes[0].data.len(),
            expected,
            "part {k}: {fmt:?} plane bytes"
        );
        assert_eq!(
            vf.planes[0].data, l1[k].image.planes[0].data,
            "part {k} == decode_all"
        );
    }
    // The grey part is a 16-byte plane with stride 8 — not the 96/24 of an
    // RgbF32Le mislabel.
    assert_eq!(
        (out[1].planes[0].data.len(), out[1].planes[0].stride),
        (16, 8)
    );

    // Inverse: several streams back into one multi-part file.
    let sink = SharedSink::default();
    {
        let mut muxer = ctx
            .containers
            .open_muxer(CONTAINER, Box::new(sink.clone()), &streams)
            .unwrap();
        muxer.write_header().unwrap();
        for p in &packets {
            muxer.write_packet(p).unwrap();
        }
        muxer.write_trailer().unwrap();
    }
    let muxed = sink.bytes();
    let back = oxideav_openexr::decode_all(&muxed).unwrap();
    assert_eq!(back.len(), 4);
    for (f, orig) in back.iter().zip(&frames) {
        assert_eq!(f.image.planes, orig.image.planes);
        assert_eq!(f.image.format, orig.image.format);
        assert_eq!(f.name, orig.name);
    }
    let (streams2, packets2, out2) = demux_decode(&ctx, &muxed);
    assert_eq!(streams2.len(), 3);
    assert_eq!(
        packets2.iter().map(|p| p.stream_index).collect::<Vec<_>>(),
        [0, 1, 2, 1]
    );
    for (vf, orig) in out2.iter().zip(&frames) {
        assert_eq!(vf.planes[0].data, orig.image.planes[0].data);
    }
}

// ---- black-box reference readers ------------------------------------------

fn tool_available(bin: &str) -> bool {
    Command::new(bin)
        .arg("--help")
        .output()
        .map(|o| o.status.success() || !o.stdout.is_empty() || !o.stderr.is_empty())
        .unwrap_or(false)
}

fn tempdir() -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "oxideav-openexr-container-{nanos}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The muxer's multi-part file and the demuxer's repacked single-part
/// packets must open in an independent reader (`exrinfo`, the file walker
/// of the reference OpenEXR tool set, used as an opaque process). Parts
/// of different sizes exercise the shared-displayWindow rule. Skips with
/// a printed reason when the tool is absent.
#[test]
fn independent_reader_opens_muxed_and_repacked_files() {
    if !tool_available("exrinfo") {
        eprintln!("exrinfo not available — skipping the independent-reader check");
        return;
    }
    let ctx = ctx();
    let a = oxideav_openexr::encode(
        &image(20, 14, ExrPixelFormat::RgbaF32Le, 0.0),
        &EncodeOptions::default().with_compression(Compression::Zip),
    )
    .unwrap();
    let b = oxideav_openexr::encode(
        &image(9, 7, ExrPixelFormat::RgbF32Le, 1.0),
        &EncodeOptions::default()
            .with_tile_size(8)
            .with_compression(Compression::Piz),
    )
    .unwrap();
    let c = oxideav_openexr::encode(
        &image(16, 16, ExrPixelFormat::GrayF32Le, 2.0),
        &EncodeOptions::default()
            .with_tile_size(4)
            .with_levels(LevelMode::Mipmap),
    )
    .unwrap();
    let stream = StreamInfo {
        index: 0,
        time_base: TimeBase::new(1, 1),
        duration: None,
        start_time: Some(0),
        params: CodecParameters::video(CodecId::new(CODEC_ID_STR)),
    };
    let sink = SharedSink::default();
    {
        let mut muxer = ctx
            .containers
            .open_muxer(CONTAINER, Box::new(sink.clone()), &[stream])
            .unwrap();
        muxer.write_header().unwrap();
        for f in [&a, &b, &c] {
            muxer
                .write_packet(&Packet::new(0, TimeBase::new(1, 1), (*f).clone()))
                .unwrap();
        }
        muxer.write_trailer().unwrap();
    }
    let multi = sink.bytes();
    let dir = tempdir();
    let run = |name: &str, bytes: &[u8]| -> String {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        let out = Command::new("exrinfo").arg(&path).output().unwrap();
        assert!(
            out.status.success(),
            "exrinfo rejected {name}: {}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let report = run("muxed.exr", &multi);
    assert!(report.contains("part 1: part0"), "{report}");
    assert!(report.contains("part 2: part1"), "{report}");
    assert!(report.contains("part 3: part2"), "{report}");
    assert!(report.contains("'tiledimage'"), "{report}");
    assert!(report.contains("mipmap"), "{report}");
    // Shared displayWindow: the union, 20 × 16.
    let l1 = oxideav_openexr::decode_all(&multi).unwrap();
    assert!(l1.iter().all(|f| (
        f.image.display_window.width(),
        f.image.display_window.height()
    ) == (20, 16)));
    assert_eq!(
        l1[1].image.planes,
        oxideav_openexr::decode(&b).unwrap().planes
    );

    // Repacked packets of a writer-built multi-part file and of the muxed one.
    let writer_multi =
        oxideav_openexr::encode_all(&frames_for_multipart(), &EncodeOptions::default()).unwrap();
    for (tag, file) in [("writer", &writer_multi), ("muxed", &multi)] {
        let (_, packets, _) = demux_decode(&ctx, file);
        for (i, p) in packets.iter().enumerate() {
            let report = run(&format!("{tag}_part{i}.exr"), &p.data);
            assert!(report.contains("part 1:"), "{report}");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
