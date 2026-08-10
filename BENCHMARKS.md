# oxideav-openexr — benchmarks

Criterion harness: `benches/codec_benchmarks.rs`. Run with:

```sh
CARGO_TARGET_DIR=/tmp/oxideav-openexr-bench cargo bench -p oxideav-openexr
```

Fixture: 256×256, four channels (A,B,G,R), photographic-ish gradients
(compressible but not flat). Throughput is measured in **raw
interleaved pixel bytes** (`width · height · 4ch · bytes/sample`) so
the schemes are directly comparable. Numbers below are from an Apple
Silicon (aarch64) macOS machine, rustc release profile, single thread;
treat them as relative guides, not absolutes.

## Scanline decode (`parse_exr`)

| Scheme | HALF | FLOAT |
| ------ | ---- | ----- |
| NONE   | 2.61 GiB/s | 19.7 GiB/s |
| RLE    | 570 MiB/s  | 1.00 GiB/s |
| ZIPS   | 267 MiB/s  | 337 MiB/s  |
| ZIP    | 656 MiB/s  | 697 MiB/s  |
| PXR24  | 1.07 GiB/s | 2.36 GiB/s |
| PIZ    | 203 MiB/s  | 237 MiB/s  |
| B44    | 1.07 GiB/s | 30.8 GiB/s¹ |
| B44A   | 1.12 GiB/s | 29.8 GiB/s¹ |
| DWAA   | 189 MiB/s  | 357 MiB/s² |
| DWAB   | 210 MiB/s  | 398 MiB/s² |

¹ B44 stores FLOAT channels uncompressed (the scheme only packs HALF),
so the FLOAT rows measure the raw-copy path.

² DWA is a half-precision codec internally; FLOAT channels ride the
same half DCT pipeline (twice the accounted bytes per sample explains
the higher apparent FLOAT throughput). PIZ and DWA rows are round-439
first-cut implementations — entropy stages are scalar and unoptimised.

## Tiled ONE_LEVEL decode (64×64 tiles)

| Scheme | HALF | FLOAT |
| ------ | ---- | ----- |
| NONE   | 2.54 GiB/s | 15.3 GiB/s |
| RLE    | 562 MiB/s  | 1.01 GiB/s |
| ZIPS   | 610 MiB/s  | 711 MiB/s  |
| ZIP    | 626 MiB/s  | 709 MiB/s  |
| PXR24  | 1.02 GiB/s | 2.26 GiB/s |
| PIZ    | 178 MiB/s  | 218 MiB/s  |
| B44    | 1.08 GiB/s | 8.5 GiB/s¹ |
| B44A   | 1.13 GiB/s | 8.4 GiB/s¹ |
| DWAA   | 170 MiB/s  | 316 MiB/s² |
| DWAB   | 172 MiB/s  | 312 MiB/s² |

## Scanline encode (`encode_exr_scanline`)

| Scheme | HALF | FLOAT |
| ------ | ---- | ----- |
| NONE   | 1.03 GiB/s | 4.63 GiB/s |
| RLE    | 520 MiB/s  | 988 MiB/s  |
| ZIPS   | 110 MiB/s  | 151 MiB/s  |
| ZIP    | 238 MiB/s  | 309 MiB/s  |
| PXR24  | 251 MiB/s  | 521 MiB/s  |
| PIZ    | 182 MiB/s  | 236 MiB/s  |
| B44    | 428 MiB/s  | 3.84 GiB/s¹ |
| B44A   | 455 MiB/s  | 3.79 GiB/s¹ |
| DWAA   | 150 MiB/s  | 299 MiB/s² |
| DWAB   | 199 MiB/s  | 395 MiB/s² |

## Primitives

| Op | Rate |
| -- | ---- |
| `half_to_f32` (all 65 536 codes) | ~985 Melem/s |
| `f32_to_half` (all 65 536 codes) | ~996 Melem/s |

## Round-385 optimisation deltas

The harness landed alongside a profiling pass; measured improvements
over the pre-round code, same machine:

| Path | Change |
| ---- | ------ |
| scanline decode, HALF NONE  | 1.40 → 2.6 GiB/s (**+90%**) |
| scanline decode, FLOAT NONE | 5.5 → 19.7 GiB/s (**+244%**) |
| scanline decode, ZIP        | +25% |
| scanline decode, RLE        | +23% |
| scanline decode, ZIPS       | +12% |
| PXR24 decode (scanline + tiled) | +40% HALF / +47% FLOAT |
| PXR24 encode                | +6% HALF / +9% FLOAT |
| scanline encode, NONE       | +6% |

Three changes produced these: (1) the interleaved block scatter
(`scatter_block_into_planes`) hoists the pixel-type dispatch out of the
per-sample loop and walks each channel/row as one bounds-checked span
of exact-size chunks; (2) the tile scatter (`scatter_tile_into_planes`)
gets the same shape; (3) the PXR24 byte-plane prefix-sum/delta loops
are specialised per pixel type over per-plane sub-slices. ZIP/ZIPS/RLE
remain dominated by their entropy stage (zlib / RLE), B44 by the block
unpack; those are the next candidates if more throughput is needed.
