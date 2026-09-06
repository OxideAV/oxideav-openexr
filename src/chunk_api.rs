//! Chunk-level decode entry points — one compressed scanline block in,
//! decoded planes out — for the coverage-guided fuzz harness and for
//! profiling. Not part of the supported API surface (hidden from
//! rustdoc / semver); the file-level readers are the public route.

use crate::decoder::{
    decode_pxr24_payload, decode_rle_payload, decode_zip_payload, scatter_b44_block_into_planes,
    scatter_block_into_planes, subsampled_dim, Pxr24RowSpec,
};
use crate::error::{ExrError, Result};
use crate::image::ExrPlane;
use crate::piz::ChunkShape;
use crate::types::{Channel, Compression};

/// Native (uncompressed, interleaved) byte size of a scanline chunk
/// covering image rows `block_y0 .. block_y0 + lines_in_block` for the
/// sorted channel list, honouring each channel's sampling.
pub fn native_chunk_size(
    sorted_channels: &[Channel],
    width: u32,
    block_y0: u32,
    lines_in_block: usize,
) -> usize {
    let mut total = 0usize;
    for ch in sorted_channels {
        let ys = ch.y_sampling.max(1) as u32;
        let rows = (0..lines_in_block as u32)
            .filter(|&l| (block_y0 + l) % ys == 0)
            .count();
        let pw = subsampled_dim(width, ch.x_sampling.max(1) as u32) as usize;
        total += rows * pw * ch.pixel_type.bytes_per_sample();
    }
    total
}

/// Decode one scanline chunk `payload` written with `compression` for
/// the given shape, returning one plane per channel sized for an image
/// of `block_y0 + lines_in_block` rows (rows above the chunk stay
/// zero). Every byte slice yields `Ok` or `Err`; a panic is a bug.
pub fn decode_scanline_chunk(
    compression: Compression,
    payload: &[u8],
    sorted_channels: &[Channel],
    width: u32,
    block_y0: u32,
    lines_in_block: usize,
) -> Result<Vec<ExrPlane>> {
    for ch in sorted_channels {
        if ch.x_sampling <= 0 || ch.y_sampling <= 0 {
            return Err(ExrError::invalid(format!(
                "channel '{}' has non-positive sampling",
                ch.name
            )));
        }
    }
    let height = block_y0 as usize + lines_in_block;
    let mut planes: Vec<ExrPlane> = sorted_channels
        .iter()
        .map(|ch| {
            let pw = subsampled_dim(width, ch.x_sampling as u32) as usize;
            let ph = subsampled_dim(height as u32, ch.y_sampling as u32) as usize;
            ExrPlane {
                name: ch.name.clone(),
                samples: vec![0.0; pw * ph],
            }
        })
        .collect();
    let uncompressed_size = native_chunk_size(sorted_channels, width, block_y0, lines_in_block);

    if matches!(compression, Compression::B44 | Compression::B44a) {
        scatter_b44_block_into_planes(
            payload,
            sorted_channels,
            &mut planes,
            width,
            block_y0,
            lines_in_block,
            uncompressed_size,
        )?;
        return Ok(planes);
    }
    let uncompressed: Vec<u8> = match compression {
        Compression::None => {
            if payload.len() != uncompressed_size {
                return Err(ExrError::invalid("uncompressed size mismatch".to_string()));
            }
            payload.to_vec()
        }
        Compression::Zip | Compression::Zips => decode_zip_payload(payload, uncompressed_size)?,
        Compression::Rle => decode_rle_payload(payload, uncompressed_size)?,
        Compression::Pxr24 => decode_pxr24_payload(
            payload,
            &Pxr24RowSpec {
                sorted_channels,
                width,
                block_y0,
                lines_in_block,
            },
            uncompressed_size,
        )?,
        Compression::Piz => crate::piz::decode_piz_payload(
            payload,
            &ChunkShape {
                sorted_channels,
                width,
                block_y0,
                lines_in_block,
            },
            uncompressed_size,
        )?,
        Compression::Dwaa | Compression::Dwab => crate::dwa::decode_dwa_payload(
            payload,
            &ChunkShape {
                sorted_channels,
                width,
                block_y0,
                lines_in_block,
            },
            uncompressed_size,
        )?,
        Compression::B44 | Compression::B44a => unreachable!("handled above"),
    };
    scatter_block_into_planes(
        &uncompressed,
        sorted_channels,
        &mut planes,
        width,
        height as u32,
        block_y0,
        lines_in_block,
    )?;
    Ok(planes)
}
