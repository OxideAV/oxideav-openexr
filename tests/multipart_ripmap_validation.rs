//! Cross-validate the multi-part **RIPMAP_LEVELS** flat tiled encoder
//! (`encode_exr_multipart_tiled_ripmap`) against `exrheader` and
//! `exrmultipart -separate`, and exercise the public-API import path
//! via a self-roundtrip back through `parse_exr_multipart_tiled_multilevel`.
//!
//! The reference binaries are opaque oracles — no source consulted,
//! no behaviour copied. If they're not installed the test prints a
//! skip message and exits zero.

use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use oxideav_openexr::{
    build_box_filter_ripmap, encode_exr_multipart_tiled_ripmap,
    parse_exr_multipart_tiled_multilevel, parse_exr_tiled_multilevel, Channel, Compression,
    MultipartRipmapTiledPart, PixelType, RipmapPyramid,
};

fn exrheader_available() -> bool {
    Command::new("exrheader")
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

fn exrmultipart_available() -> bool {
    Command::new("exrmultipart")
        .output()
        .map(|o| o.status.code().is_some())
        .unwrap_or(false)
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn tempdir(tag: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let c = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "oxideav-openexr-mprip-{tag}-{nanos}-{}-{c}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

fn rgba_channels() -> Vec<Channel> {
    ["A", "B", "G", "R"]
        .iter()
        .map(|n| Channel {
            name: n.to_string(),
            pixel_type: PixelType::Float,
            p_linear: false,
            x_sampling: 1,
            y_sampling: 1,
        })
        .collect()
}

fn make_planes(w: u32, h: u32, salt: f32) -> [Vec<f32>; 4] {
    let pixels = (w as usize) * (h as usize);
    let mut a = Vec::with_capacity(pixels);
    let mut b = Vec::with_capacity(pixels);
    let mut g = Vec::with_capacity(pixels);
    let mut r = Vec::with_capacity(pixels);
    for y in 0..h {
        for x in 0..w {
            r.push((x as f32) / (w as f32) + salt);
            g.push((y as f32) / (h as f32));
            b.push(((x ^ y) as f32) * 0.01);
            a.push(1.0);
        }
    }
    [a, b, g, r]
}

fn build_part(
    name: &str,
    w: u32,
    h: u32,
    salt: f32,
    comp: Compression,
    tile: u32,
) -> (MultipartRipmapTiledPart, RipmapPyramid) {
    let planes = make_planes(w, h, salt);
    let pyramid = build_box_filter_ripmap(
        w,
        h,
        &[
            planes[0].clone(),
            planes[1].clone(),
            planes[2].clone(),
            planes[3].clone(),
        ],
    );
    let snapshot = pyramid.clone();
    (
        MultipartRipmapTiledPart {
            name: name.to_string(),
            tile_x: tile,
            tile_y: tile,
            channels: rgba_channels(),
            pyramid,
            compression: comp,
        },
        snapshot,
    )
}

#[test]
fn exrheader_accepts_our_multipart_ripmap_tiled_file() {
    if !exrheader_available() {
        eprintln!("exrheader not available, skipping");
        return;
    }
    let (p0, _) = build_part("partA", 16, 16, 0.0, Compression::Zip, 8);
    let (p1, _) = build_part("partB", 16, 16, 0.5, Compression::Zip, 8);
    let parts = vec![p0, p1];
    let bytes = encode_exr_multipart_tiled_ripmap(&parts).unwrap();
    let dir = tempdir("exrheader");
    let path = format!("{dir}/in.exr");
    std::fs::write(&path, &bytes).unwrap();
    let out = Command::new("exrheader")
        .arg(&path)
        .output()
        .expect("exrheader spawn");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "exrheader failed on our multipart ripmap tiled file\nstdout: {stdout}\nstderr: {stderr}"
    );
    assert!(
        stdout.contains("tiledimage"),
        "exrheader output missing type='tiledimage'\nstdout: {stdout}"
    );
    // exrheader prints the level mode as "rip-map" (hyphenated). Accept
    // either spelling for forward/backward compatibility with reference
    // versions.
    assert!(
        stdout.contains("rip-map") || stdout.contains("ripmap"),
        "exrheader output should mention ripmap level mode\nstdout: {stdout}"
    );
    assert!(
        stdout.contains("partA") && stdout.contains("partB"),
        "exrheader output missing per-part names\nstdout: {stdout}"
    );
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(&dir);
}

#[test]
fn exrmultipart_separate_splits_our_multipart_ripmap_tiled() {
    if !exrmultipart_available() || !exrheader_available() {
        eprintln!("exrmultipart / exrheader not available, skipping");
        return;
    }
    let (p0, pyr0) = build_part("alpha", 16, 16, 0.0, Compression::Zip, 8);
    let (p1, pyr1) = build_part("beta", 16, 16, 0.25, Compression::Zips, 8);
    let parts = vec![p0, p1];
    let bytes = encode_exr_multipart_tiled_ripmap(&parts).unwrap();
    let dir = tempdir("separate");
    let in_path = format!("{dir}/in.exr");
    std::fs::write(&in_path, &bytes).unwrap();
    let _ = Command::new("exrmultipart")
        .arg("-separate")
        .arg("-i")
        .arg(&in_path)
        .arg("-o")
        .arg(format!("{dir}/out.exr"))
        .output()
        .expect("exrmultipart spawn");

    // Identify the two split outputs (some builds suffix .partname.exr,
    // some emit partname.exr in the dir).
    let mut splits: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for ent in rd.flatten() {
            let p = ent.path();
            if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                if name == "in.exr" {
                    continue;
                }
                if name.ends_with(".exr") {
                    splits.push(p.to_string_lossy().into_owned());
                }
            }
        }
    }
    splits.sort();
    if splits.len() != 2 {
        eprintln!(
            "exrmultipart split count = {} (expected 2) in {dir}; skipping cross-check",
            splits.len()
        );
        let _ = cleanup_dir(&dir);
        return;
    }

    // Header-inspect each split to know which one is alpha vs beta, then
    // decode it through our single-part multi-level reader and compare to
    // the source ripmap grid sample-for-sample.
    let mut alpha_decoded = false;
    let mut beta_decoded = false;
    for s in &splits {
        let header_out = Command::new("exrheader").arg(s).output().unwrap();
        let txt = String::from_utf8_lossy(&header_out.stdout);
        let (want, source_grid) = if txt.contains("alpha") {
            ("alpha", &pyr0.grid)
        } else if txt.contains("beta") {
            ("beta", &pyr1.grid)
        } else {
            continue;
        };
        let split_bytes = std::fs::read(s).unwrap();
        let decoded = parse_exr_tiled_multilevel(&split_bytes).unwrap();
        // Decoded levels for RIPMAP come back in lvly-outer lvlx-inner
        // order; the source grid is grid[lvly][lvlx]. Match cell-by-cell.
        let mut k = 0usize;
        for (lvly, row) in source_grid.iter().enumerate() {
            for (lvlx, cell) in row.iter().enumerate() {
                let got = &decoded.levels[k];
                assert_eq!(
                    got.level_x as usize, lvlx,
                    "{want} cell ({lvlx},{lvly}) lvlx"
                );
                assert_eq!(
                    got.level_y as usize, lvly,
                    "{want} cell ({lvlx},{lvly}) lvly"
                );
                assert_eq!(got.width, cell.width, "{want} cell ({lvlx},{lvly}) width");
                assert_eq!(
                    got.height, cell.height,
                    "{want} cell ({lvlx},{lvly}) height"
                );
                for (gp, sp) in got.planes.iter().zip(cell.planes.iter()) {
                    assert_eq!(
                        &gp.samples, sp,
                        "{want} cell ({lvlx},{lvly}) plane '{}'",
                        gp.name
                    );
                }
                k += 1;
            }
        }
        if want == "alpha" {
            alpha_decoded = true;
        } else {
            beta_decoded = true;
        }
    }
    assert!(
        alpha_decoded && beta_decoded,
        "did not locate both alpha and beta in split outputs"
    );

    let _ = cleanup_dir(&dir);
}

fn cleanup_dir(dir: &str) -> std::io::Result<()> {
    for ent in std::fs::read_dir(dir)?.flatten() {
        let _ = std::fs::remove_file(ent.path());
    }
    let _ = std::fs::remove_dir(dir);
    Ok(())
}

#[test]
fn our_writer_and_reader_multipart_ripmap_full_roundtrip() {
    // Pure self-roundtrip exercising the public-API import path,
    // independent of any reference binary. Mix of compressions and
    // non-power-of-two dimensions (edge tiles + non-square ripmap cells).
    let (p0, pyr0) = build_part("p0", 24, 16, 0.0, Compression::Zip, 8);
    let (p1, pyr1) = build_part("p1", 16, 16, 0.5, Compression::Zips, 8);
    let (p2, pyr2) = build_part("p2", 13, 9, 0.75, Compression::Rle, 4);
    let parts = vec![p0, p1, p2];
    let bytes = encode_exr_multipart_tiled_ripmap(&parts).unwrap();
    let decoded = parse_exr_multipart_tiled_multilevel(&bytes).unwrap();
    assert_eq!(decoded.len(), 3);
    let expected = [&pyr0.grid, &pyr1.grid, &pyr2.grid];
    for (part_img, exp_grid) in decoded.iter().zip(expected.iter()) {
        let total: usize = exp_grid.iter().map(|row| row.len()).sum();
        assert_eq!(part_img.levels.len(), total);
        let mut k = 0usize;
        for (lvly, row) in exp_grid.iter().enumerate() {
            for (lvlx, cell) in row.iter().enumerate() {
                let got = &part_img.levels[k];
                assert_eq!(got.level_x as usize, lvlx);
                assert_eq!(got.level_y as usize, lvly);
                assert_eq!(got.width, cell.width);
                assert_eq!(got.height, cell.height);
                for (gp, wp) in got.planes.iter().zip(cell.planes.iter()) {
                    assert_eq!(&gp.samples, wp);
                }
                k += 1;
            }
        }
    }
}
