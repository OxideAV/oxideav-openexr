# `lineOrder` wire-fact derivation record (round 410)

The `lineOrder` storage-order semantics implemented by this crate were
derived empirically by black-box observation of the reference OpenEXR
command-line tools (`exrheader`, `exrinfo`, `exrmetrics`, `exr2aces`),
invoked strictly as opaque processes (bytes in, status + text out). No
external implementation source was consulted. This file records the
experiment so the derivation is reproducible.

## Method

1. A base 32×40 RGBA-FLOAT gradient was written by this crate's own
   encoder, once as a scanline file (ZIP, 16 lines/block → 3 chunks,
   chunk `y` values 0/16/32) and once as a ONE_LEVEL tiled file
   (16×16 tiles → 2×3 = 6 chunks).
2. A byte-editing script (attribute-table walk per this crate's own
   header layout) produced candidate variants that flip the `lineOrder`
   attribute value byte and permute the physical chunk storage order
   and/or the offset-table entry order.
3. Each variant was fed to `exrmetrics --convert -z none` (chunk-walking
   reader) and `exrheader`; accepted outputs were decoded and compared
   pixel-exactly against the base gradient. `exr2aces` (classic-reader
   codepath) cross-checked one accepted variant.

## Variants and verdicts

Scanline (base chunks stored top-first, table entries `[y=0, y=16, y=32]`):

| Variant | lineOrder | Physical storage | Offset-table keying | Reference verdict |
|---------|-----------|------------------|---------------------|-------------------|
| `dec_A` | 1 (decreasing) | top-first (unchanged) | top-first (unchanged) | accepted, pixel-exact |
| `dec_B` | 1 (decreasing) | bottom-first | follows storage (entry 0 → y=32) | **rejected**: chunk-leader error "scanline says 32, expected 0" |
| `dec_C` | 1 (decreasing) | bottom-first | top-first (entry i → y=16·i) | accepted, pixel-exact (both reader codepaths) |
| `rand_scan` | 2 (random) | top-first | top-first | **rejected**: "Invalid line order in image header" |

Tiled ONE_LEVEL (canonical order ty-outer/tx-inner):

| Variant | lineOrder | Physical storage | Offset-table keying | Reference verdict |
|---------|-----------|------------------|---------------------|-------------------|
| `tile_D1` | 1 (decreasing) | tile rows bottom-first | canonical | accepted, pixel-exact |
| `tile_D2` | 1 (decreasing) | tile rows bottom-first | follows storage | **rejected**: "bad tile Y coordinate (2, expect 0)" |
| `tile_R1` | 2 (random) | deterministic shuffle | canonical | accepted, pixel-exact |

## Derived wire facts (implemented + pinned in `line_order_validation.rs`)

1. The chunk offset table is ALWAYS keyed in canonical top-first order
   (scanline entry `i` ↔ block starting at `y_min + i·blockHeight`;
   tiled entry `i` ↔ tile `i` of the canonical ty-outer/tx-inner walk),
   for every `lineOrder` value.
2. `lineOrder` governs only the physical storage/streaming order of the
   chunks inside the file: DECREASING_Y = bottom-first rows; RANDOM_Y =
   arbitrary order.
3. RANDOM_Y is invalid in a scanline image header (reference readers
   refuse to open the file) and valid for tiled images.

## Artifact digests (SHA-256)

The experiment inputs are writer-generated + byte-patched, so they are
not committed; digests of the exact bytes used:

```
6dbd6ef80a72e046ec43485ce6275698a518c98f03f1f7cf54de42aa6771bea0  base_inc.exr
b271d1cb45f34e2c62798eed7d819101907bdefd398a06f867447e46d27f8af5  dec_A.exr
0bbc46b311e9e2262f2e291686d6f48c39d616b55ca278f914f224890b5ee5e3  dec_B.exr
45b42cc5eb240b11886b7e49801d7c1319420bb3fd45c8ae770d6153be75b3ba  dec_C.exr
c418247f4e0bb34f1bc5ce07304cad57d89b6cf937dddb2d135a9cd4edec1e03  rand_scan.exr
f6e36cc6c06919d2c11817ff40dc8cfdf4b4a86798f48e40c4c49f7cb9831640  base_tiled.exr
660ca60208f73a060a151500a4bf2f7248406738b28c208c66f0845fe9d27cfd  tile_D1.exr
167cdf727415690f8fb6cb232291649def541c9dcbb942b0dddc4d6fe825e41b  tile_D2.exr
023a8391e833c412a285a3ab5e62e2e198130bc2e073ef66da8d7b764c66dd73  tile_R1.exr
```

Reference tool versions: the Homebrew OpenEXR tool suite present on the
build host at derivation time (`exrheader`/`exrinfo`/`exrmetrics`
accepting/rejecting behavior is what is recorded above; the tests
re-validate against whatever suite is installed and auto-skip when
absent).
