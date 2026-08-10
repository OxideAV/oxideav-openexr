# DWA stream-ordering facts — black-box derivation record (round 439)

The staged observer-spec (`docs/image/openexr/openexr-piz-dwa-observer-spec.md`
§3) fixes the DWAA/DWAB chunk header, sub-stream framing, per-scheme
data preparation and all constants, but leaves a handful of orderings
implicit. Each was pinned by generating controlled NONE files with this
crate's writer, converting them with a reference conversion binary
invoked as an **opaque process**, and reading the produced chunk bytes
(inflating sub-streams and decoding the AC Huffman payload with this
crate's own §2.5 implementation). No implementation source of any kind
was consulted — every fact below is a statement about produced wire
bytes.

## Facts established

1. **Block iteration is row-major.** Six 8×8 blocks with distinct
   constant values per block produce DC values in `(by, bx)` row-major
   order.

2. **DC stream is plane-major per channel-set.** For an R/G/B triple the
   DC sub-stream carries all blocks' Y values, then all C1, then all C2
   (`4443 45d7 …` patterns from per-block constants). Lossy singleton
   channels contribute one plane each.

3. **AC stream interleaves components per block within a triple.** With
   four identical image blocks in an R/G/B file the AC stream shows a
   repeating period of *three* distinct block groups (Y, C1, C2 for
   block 0, then Y, C1, C2 for block 1, …) — not four — ruling out
   plane-major AC and pinning per-block component interleave. For
   singleton sets the AC stream is simply all of that channel's blocks
   in order.

4. **Channel-sets: complete CSC triples first, then lossy singletons in
   sorted-channel order.** A file whose lossy singleton (`BY`) sorts
   *before* every member of a `sub.R/G/B` triple still emits the
   triple's planes first in both AC and DC streams.

5. **Perceptual LUT is applied per channel BEFORE the colour
   transform.** Constant RGB = (0.5, 1.0, 2.0) yields a DC of
   `8 × (0.2126·0.5^(1/2.2) + 0.7152·1 + 0.0722·(ln 2 / 2.2 + 1))`
   = half `0x47b9`, matching LUT-then-CSC and excluding CSC-then-LUT.

6. **The RLE byte-plane split runs over the channel's whole chunk
   region, not per row.** For an `A` HALF channel with a per-pixel ramp,
   the un-RLE'd bytes equal all low bytes of every sample in the chunk
   followed by all high bytes (whole-region hypothesis matched; the
   per-row hypothesis did not).

7. **Verbatim sub-stream is channel-major.** Two verbatim FLOAT
   channels (`Q`, `Z`) inflate to all of `Q`'s chunk rows followed by
   all of `Z`'s.

8. **The version-2 rule block carries only the rules matched by present
   channels, in rule-set order.** An RGBA file yields rules
   `R (0x14), G (0x24), B (0x34), A (0x08)`; a lone-`Y` file yields just
   `Y (0x04)`; a file with only unmatched (verbatim) channels yields an
   empty rule block.

9. **Reference chunks are version 2 with `acCompression = 0`** (the
   static-Huffman payload container shared with PIZ) in every observed
   conversion.

10. **IDCT even-pair dataflow.** One sample in 29 700 of a
    reference-encoded DWAB file decoded one half-ULP high with the DC
    pair computed as `a·c0 ± a·c4`. Recomputing the extracted block
    coefficients in binary32 across candidate dataflows showed the only
    variant matching the reference's decode is `a·(c0 ± c4)` — fold the
    sum/difference first, then multiply — with rows-then-columns
    separability (columns-first also diverges). The remaining odd-part
    grouping choices all agree on observed data.

11. **Raw fallback.** Chunks whose compressed form would not shrink
    are stored as the native interleaved bytes
    (`compressed_len == uncompressed_len`), for DWA exactly as for the
    other schemes.

## Validation anchors

- `tests/dwa_decode_validation.rs` — reference-encoded DWAA/DWAB files
  (smooth, noisy, CSC triples, FLOAT channels, mixed schemes,
  multi-chunk DWAB) decode **bit-exactly** to the reference's own
  decode of the same bytes.
- `tests/dwa_encode_validation.rs` — our version-2 chunks are accepted
  by the reference binary and its decode agrees with ours bit-exactly,
  including edge-mirrored odd dimensions and the `dwaCompressionLevel`
  attribute.
