//! The worker budget changes how fast chunks are read and written,
//! never what comes out: the same bytes from every writer, the same
//! samples from every reader, and the same error on malformed input,
//! at budgets 1, 2, 3 and 8.

use oxideav_openexr::{
    ColourLayout, Compression, DecodeOptions, EncodeOptions, ExrImage, ExrPixelFormat, Frame,
    LevelMode, LineOrder, PixelType, decode_all_with, decode_with, encode, encode_all, parse_exr,
    parse_exr_tiled_multilevel, with_threads,
};

const BUDGETS: [usize; 3] = [2, 3, 8];

const SCHEMES: [Compression; 10] = [
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

fn image(w: u32, h: u32, format: ExrPixelFormat, seed: u32) -> ExrImage {
    let comps = format.components();
    let mut state = seed.wrapping_mul(2_654_435_761).max(1);
    let samples: Vec<f32> = (0..(w * h) as usize * comps)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let smooth = (i / comps % w as usize) as f32 / w as f32;
            smooth + (state >> 20) as f32 / 65_536.0
        })
        .collect();
    ExrImage::from_f32(w, h, format, &samples).unwrap()
}

/// Encode at every budget (bytes must agree), then decode at every
/// budget (images must agree). Returns the file.
fn check(img: &ExrImage, opts: &EncodeOptions, label: &str) -> Vec<u8> {
    let bytes = encode(img, opts).unwrap_or_else(|e| panic!("{label}: {e}"));
    let serial = decode_with(&bytes, &DecodeOptions::default()).unwrap();
    for threads in BUDGETS {
        let again = encode(img, &opts.clone().with_threads(threads)).unwrap();
        assert!(
            again == bytes,
            "{label}: encode differs at budget {threads}"
        );
        let decoded = decode_with(&bytes, &DecodeOptions::default().with_threads(threads)).unwrap();
        assert!(
            decoded == serial,
            "{label}: decode differs at budget {threads}"
        );
    }
    bytes
}

#[test]
fn every_scheme_and_layout_is_budget_independent() {
    // Sizes straddle the chunk heights (1, 16, 32, 256 lines) and leave
    // edge tiles; 300 rows give DWAB two chunks.
    let rgba = image(67, 300, ExrPixelFormat::RgbaF32Le, 1);
    let small = image(40, 37, ExrPixelFormat::RgbF32Le, 2);
    for compression in SCHEMES {
        for pixel_type in [PixelType::Half, PixelType::Float] {
            let base = EncodeOptions::default()
                .with_compression(compression)
                .with_pixel_type(pixel_type);
            let label = format!("{compression:?} {pixel_type:?}");
            check(&rgba, &base, &format!("{label} scanline"));
            check(
                &rgba,
                &base.clone().with_line_order(LineOrder::DecreasingY),
                &format!("{label} scanline decreasing"),
            );
            check(
                &rgba,
                &base.clone().with_tile_size(32),
                &format!("{label} tiled"),
            );
            for levels in [LevelMode::Mipmap, LevelMode::Ripmap] {
                let bytes = check(
                    &small,
                    &base.clone().with_tile_size(16).with_levels(levels),
                    &format!("{label} {levels:?}"),
                );
                // Every level, through the depth reader, under a budget.
                let serial = parse_exr_tiled_multilevel(&bytes).unwrap();
                let wide = with_threads(8, || parse_exr_tiled_multilevel(&bytes)).unwrap();
                assert_eq!(serial.levels.len(), wide.levels.len());
                for (a, b) in serial.levels.iter().zip(&wide.levels) {
                    assert!(a.planes == b.planes, "{label} {levels:?}");
                }
            }
        }
    }
}

#[test]
fn subsampled_chroma_and_gray_are_budget_independent() {
    // Sub-sampled chroma rows: band boundaries must respect the
    // vertical sampling for every line-count per chunk.
    let rgb = image(34, 66, ExrPixelFormat::RgbF32Le, 3);
    let gray = image(19, 51, ExrPixelFormat::GrayF32Le, 4);
    for compression in SCHEMES {
        let luma = EncodeOptions::default()
            .with_compression(compression)
            .with_pixel_type(PixelType::Half)
            .with_colour(ColourLayout::LumaChroma);
        check(&rgb, &luma, &format!("{compression:?} luminance/chroma"));
        check(
            &gray,
            &EncodeOptions::default().with_compression(compression),
            &format!("{compression:?} gray"),
        );
    }
}

#[test]
fn multi_part_files_are_budget_independent() {
    let frames = vec![
        Frame::new(image(33, 70, ExrPixelFormat::RgbaF32Le, 5), 0).with_name("a".to_string()),
        Frame::new(image(64, 40, ExrPixelFormat::RgbF32Le, 6), 1).with_name("b".to_string()),
        Frame::new(image(9, 130, ExrPixelFormat::GrayF32Le, 7), 2),
    ];
    for opts in [
        EncodeOptions::default().with_compression(Compression::Zip),
        EncodeOptions::default()
            .with_compression(Compression::Piz)
            .with_tile_size(16),
        EncodeOptions::default()
            .with_compression(Compression::Rle)
            .with_tile_size(8)
            .with_levels(LevelMode::Mipmap),
    ] {
        let bytes = encode_all(&frames, &opts).unwrap();
        let serial = decode_all_with(&bytes, &DecodeOptions::default()).unwrap();
        assert_eq!(serial.len(), 3);
        for (f, want) in serial.iter().zip(&frames) {
            assert!(f.image.planes == want.image.planes);
        }
        for threads in BUDGETS {
            assert!(encode_all(&frames, &opts.clone().with_threads(threads)).unwrap() == bytes);
            let read = DecodeOptions::default().with_threads(threads);
            assert!(decode_all_with(&bytes, &read).unwrap() == serial);
            assert!(decode_with(&bytes, &read.clone().with_part(1)).unwrap() == serial[1].image);
        }
    }
}

#[test]
fn errors_are_budget_independent() {
    let img = image(48, 200, ExrPixelFormat::RgbaF32Le, 8);
    for (compression, tile) in [
        (Compression::Zip, 0),
        (Compression::Piz, 0),
        (Compression::Rle, 16),
        (Compression::None, 0),
        (Compression::Zips, 32),
    ] {
        let bytes = encode(
            &img,
            &EncodeOptions::default()
                .with_compression(compression)
                .with_tile_size(tile),
        )
        .unwrap();
        let outcome = |data: &[u8], threads: usize| {
            let r = decode_with(data, &DecodeOptions::default().with_threads(threads));
            match r {
                Ok(img) => Ok(img),
                Err(e) => Err(e.to_string()),
            }
        };
        // Damage walks through the file: headers, offset tables, chunk
        // framing, payloads; plus truncations.
        let step = (bytes.len() / 97).max(1);
        for at in (8..bytes.len()).step_by(step) {
            let mut bad = bytes.clone();
            bad[at] ^= 0xA5;
            let serial = outcome(&bad, 1);
            for threads in BUDGETS {
                assert!(
                    outcome(&bad, threads) == serial,
                    "{compression:?} tile {tile}: byte {at} at budget {threads}"
                );
            }
            let cut = &bytes[..at];
            let serial = outcome(cut, 1);
            assert!(outcome(cut, 8) == serial, "{compression:?}: cut at {at}");
        }
    }
}

#[test]
fn the_multi_level_reader_is_budget_independent_on_damaged_files() {
    let img = image(70, 45, ExrPixelFormat::RgbF32Le, 11);
    for (levels, compression) in [
        (LevelMode::Mipmap, Compression::Zip),
        (LevelMode::Ripmap, Compression::Piz),
        (LevelMode::Mipmap, Compression::None),
    ] {
        let bytes = encode(
            &img,
            &EncodeOptions::default()
                .with_compression(compression)
                .with_tile_size(16)
                .with_levels(levels),
        )
        .unwrap();
        // Every level's planes, or the error text.
        let outcome = |data: &[u8], threads: usize| {
            with_threads(threads, || parse_exr_tiled_multilevel(data))
                .map(|img| img.levels.into_iter().map(|l| l.planes).collect::<Vec<_>>())
                .map_err(|e| e.to_string())
        };
        assert!(outcome(&bytes, 1).is_ok());
        let step = (bytes.len() / 131).max(1);
        for at in (8..bytes.len()).step_by(step) {
            let mut bad = bytes.clone();
            bad[at] ^= 0x5A;
            let serial = outcome(&bad, 1);
            for threads in BUDGETS {
                assert!(
                    outcome(&bad, threads) == serial,
                    "{levels:?} {compression:?}: byte {at} at budget {threads}"
                );
            }
            assert!(outcome(&bytes[..at], 8) == outcome(&bytes[..at], 1));
        }
    }
}

#[test]
fn the_depth_api_runs_under_a_scoped_budget() {
    let img = image(31, 90, ExrPixelFormat::RgbF32Le, 9);
    let bytes = encode(
        &img,
        &EncodeOptions::default().with_compression(Compression::Zips),
    )
    .unwrap();
    let serial = parse_exr(&bytes).unwrap();
    for threads in BUDGETS {
        let wide = with_threads(threads, || parse_exr(&bytes)).unwrap();
        assert!(wide == serial);
    }
    // The budget ends with the scope.
    let nested = with_threads(8, || with_threads(1, || parse_exr(&bytes))).unwrap();
    assert!(nested == serial);
}

#[cfg(feature = "registry")]
#[test]
fn the_execution_context_sets_the_budget() {
    use oxideav_core::{CodecId, CodecParameters, ExecutionContext, Frame, PixelFormat};
    let img = image(40, 100, ExrPixelFormat::RgbaF32Le, 10);
    let cid = CodecId::new(oxideav_openexr::CODEC_ID_STR);
    let run = |ctx: &ExecutionContext| {
        let mut params = CodecParameters::video(cid.clone());
        params.width = Some(40);
        params.height = Some(100);
        params.pixel_format = Some(PixelFormat::RgbaF32Le);
        params.options.insert("compression", "piz");
        let mut enc = oxideav_openexr::make_encoder(&params).expect("encoder");
        enc.set_execution_context(ctx);
        enc.send_frame(&Frame::Video(img.clone().into())).unwrap();
        let packet = enc.receive_packet().unwrap();
        let mut dec =
            oxideav_openexr::make_decoder(&CodecParameters::video(cid.clone())).expect("decoder");
        dec.set_execution_context(ctx);
        dec.send_packet(&packet).unwrap();
        let Frame::Video(frame) = dec.receive_frame().unwrap() else {
            panic!("video frame");
        };
        (packet.data, frame.planes[0].data.clone())
    };
    let serial = run(&ExecutionContext::serial());
    assert!(run(&ExecutionContext::with_threads(8)) == serial);
    assert!(run(&ExecutionContext::with_threads(3)) == serial);
}

/// The backing check (a header must not declare more sample data than
/// its chunks can hold at the scheme's best ratio) never refuses a real
/// file: constant images are the most compressible input there is.
#[test]
fn constant_images_pass_the_backing_check() {
    for value in [0.0f32, 1.0, 0.18] {
        let samples = vec![value; 640 * 300 * 4];
        let img = ExrImage::from_f32(640, 300, ExrPixelFormat::RgbaF32Le, &samples).unwrap();
        for compression in SCHEMES {
            for pixel_type in [PixelType::Half, PixelType::Float] {
                for tile in [0, 256] {
                    let opts = EncodeOptions::default()
                        .with_compression(compression)
                        .with_pixel_type(pixel_type)
                        .with_tile_size(tile)
                        .with_threads(4);
                    let label = format!("{value} {compression:?} {pixel_type:?} tile {tile}");
                    let bytes = encode(&img, &opts).unwrap_or_else(|e| panic!("{label}: {e}"));
                    oxideav_openexr::check_layout(&bytes, true)
                        .unwrap_or_else(|e| panic!("{label}: {e}"));
                    let read = DecodeOptions::default().with_threads(4);
                    let back =
                        decode_with(&bytes, &read).unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_eq!((back.width(), back.height()), (640, 300), "{label}");
                    if value == 0.0 {
                        assert!(back == img, "{label}");
                    }
                }
            }
        }
    }
}
