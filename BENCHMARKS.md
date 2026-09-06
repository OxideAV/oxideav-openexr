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
| NONE   | 2.50 GiB/s | 19.84 GiB/s |
| RLE    | 639 MiB/s | 1.33 GiB/s |
| ZIPS   | 377 MiB/s | 412 MiB/s |
| ZIP    | 779 MiB/s | 823 MiB/s |
| PXR24  | 1.04 GiB/s | 2.56 GiB/s |
| PIZ    | 306 MiB/s | 369 MiB/s |
| B44    | 1.06 GiB/s | 32.05 GiB/s¹ |
| B44A   | 1.11 GiB/s | 31.85 GiB/s¹ |
| DWAA   | 389 MiB/s | 685 MiB/s² |
| DWAB   | 475 MiB/s | 831 MiB/s² |

¹ B44 stores FLOAT channels uncompressed (the scheme only packs HALF),
so the FLOAT rows measure the raw-copy path.

² DWA is a half-precision codec internally; FLOAT channels ride the
same half DCT pipeline (twice the accounted bytes per sample explains
the higher apparent FLOAT throughput). PIZ and DWA rows reflect the
round-446 optimisation pass over the round-439 first-cut entropy
stages; every row reflects the round-457 pass (below).

## Tiled ONE_LEVEL decode (64×64 tiles)

| Scheme | HALF | FLOAT |
| ------ | ---- | ----- |
| NONE   | 2.57 GiB/s | 17.67 GiB/s |
| RLE    | 631 MiB/s | 1.32 GiB/s |
| ZIPS   | 741 MiB/s | 829 MiB/s |
| ZIP    | 751 MiB/s | 825 MiB/s |
| PXR24  | 1.05 GiB/s | 2.32 GiB/s |
| PIZ    | 278 MiB/s | 343 MiB/s |
| B44    | 1.09 GiB/s | 8.15 GiB/s¹ |
| B44A   | 1.14 GiB/s | 8.00 GiB/s¹ |
| DWAA   | 325 MiB/s | 578 MiB/s² |
| DWAB   | 326 MiB/s | 589 MiB/s² |

## Scanline encode (`encode_exr_scanline`)

| Scheme | HALF | FLOAT |
| ------ | ---- | ----- |
| NONE   | 1.08 GiB/s | 4.58 GiB/s |
| RLE    | 523 MiB/s | 978 MiB/s |
| ZIPS   | 136 MiB/s | 170 MiB/s |
| ZIP    | 236 MiB/s | 294 MiB/s |
| PXR24  | 252 MiB/s | 491 MiB/s |
| PIZ    | 219 MiB/s | 265 MiB/s |
| B44    | 451 MiB/s | 3.74 GiB/s¹ |
| B44A   | 471 MiB/s | 3.84 GiB/s¹ |
| DWAA   | 171 MiB/s | 332 MiB/s² |
| DWAB   | 226 MiB/s | 429 MiB/s² |

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

## Round-446 optimisation deltas

A profiling pass over the round-439 first-cut PIZ / DWA entropy
stages; measured improvements over the pre-round code, same machine:

| Path | Change |
| ---- | ------ |
| PIZ scanline decode  | 203 → 270 MiB/s HALF (**+33%**), 237 → 353 FLOAT (**+49%**) |
| PIZ tiled decode     | +26% HALF / +40% FLOAT |
| PIZ scanline encode  | +14% HALF / +17% FLOAT |
| DWAA scanline decode | +13% HALF / +9% FLOAT |
| DWAB scanline decode | +14% HALF / +11% FLOAT |
| DWAB scanline encode | +12% HALF / +8% FLOAT |

Five changes produced these: (1) the shared Huffman payload bit reader
keeps a 64-bit left-aligned accumulator refilled seven bytes at a time
from one unaligned load instead of extracting bits one at a time;
(2) the PIZ wavelet drivers monomorphize over the step function so the
14-bit / modulo-2^16 variants inline into the innermost loop; (3) DWA
AC un-RLE writes non-zero coefficients straight through the inverse
zig-zag into the raster block and DC-only blocks take a folded
single-lane IDCT with the identical IEEE operation sequence (bit-exact,
unit-pinned across -0.0 / inf / NaN DC codes); (4) the Huffman bit
writer flushes whole eight-byte accumulators; (5) the DWA encoder
copies interior 8×8 blocks row-wise and the decoder hoists the
channel-set dispatch out of the per-texel write-back. All bit-exact:
the reference cross-validation suites pass unchanged on both sides.

## Round-457 optimisation deltas

A profiling pass (sampling profiler over the chunk decoders) after the
luminance/chroma and layer work; measured improvements over the
pre-round code, same machine, byte-identical output on every path
(the reference cross-validation suites pass unchanged):

| Path | Change |
| ---- | ------ |
| ZIPS scanline decode | 272 → 377 MiB/s HALF (**+39%**), 330 → 412 FLOAT (**+25%**) |
| ZIP scanline decode  | 608 → 779 MiB/s HALF (**+28%**), 681 → 823 FLOAT (**+21%**) |
| RLE scanline decode  | +15% HALF / +37% FLOAT |
| ZIP / ZIPS / RLE tiled decode | +17% … +39% |
| PIZ decode           | 271 → 306 MiB/s HALF scanline (**+13%**), 218 → 278 tiled (**+28%**); FLOAT +11% / +17% |
| PXR24 decode         | +6% … +11% |
| DWAA decode          | 213 → 389 MiB/s HALF scanline (**+83%**), 406 → 685 FLOAT (**+69%**); tiled +75% HALF |
| DWAB decode          | 246 → 475 MiB/s HALF scanline (**+94%**), 447 → 831 FLOAT (**+86%**); tiled +73% HALF |
| DWAA / DWAB encode   | +8% / +6% HALF |
| HALF encode, every scheme | +8% (NONE) … +9% (B44) |
| ZIPS encode          | 113 → 140 MiB/s HALF (**+24%**), 151 → 174 FLOAT (**+15%**) |
| PIZ encode           | 191 → 223 MiB/s HALF (**+17%**) |
| every other encode path | +1% … +5% |

Five changes produced these: (0) `f32_to_half` is a branch-light
classify-then-round form — magnitude range tests, an add-`0x0FFF + lsb`
rounding whose mantissa carry rolls into the exponent by itself, the
same add-then-shift rounding at the subnormal shift — pinned bit-exact
to the original straight-line encoder over 16.7 million `f32` patterns
plus dense sweeps at every boundary (the DWA decoder converts every
texel through it, and the profile put ~38% of DWA decode time in the
old cascade; the inverse perceptual LUT is also hoisted out of the
texel loop); (1) the zlib inflater is one
`flate2::Decompress` state per thread, reset per chunk and driven
through `decompress_vec` with the same capped reservation and
one-past-expected ceiling as before — a ZIPS file inflates one
scanline per chunk, and rebuilding the decompressor (plus the reader
adapter's 32 KiB buffer) per chunk cost more than the inflate; (2) the
streaming zlib encoder is likewise recycled per thread (`reset`
restores the fresh-stream state; a unit test pins its output to a fresh
encoder's byte-for-byte); (3) the ZIP-family unpredict + de-interleave
is one fused pass writing each recovered byte straight to its
de-interleaved slot; (4) the Huffman decoder keeps its bit-reader state
in plain locals (the method-based reader spilled every field to the
stack once per symbol), resolves codes longer than the 14-bit fast
table by testing the accumulator against each length's canonical range
directly instead of growing the code one bit at a time (falling back to
the incremental scan only when the whole code is not yet available, so
the exhaustion errors are unchanged), builds its per-length symbol
ranges in one flat allocation, and the PIZ range-compaction LUTs skip
empty bitmap bytes.
