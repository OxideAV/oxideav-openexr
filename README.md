# oxideav-openexr

[![CI](https://github.com/OxideAV/oxideav-openexr/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-openexr/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-openexr.svg)](https://crates.io/crates/oxideav-openexr) [![docs.rs](https://docs.rs/oxideav-openexr/badge.svg)](https://docs.rs/oxideav-openexr) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust OpenEXR reader + writer for the
[oxideav](https://github.com/OxideAV/oxideav-workspace) workspace:
scanline, tiled (ONE_LEVEL / MIPMAP / RIPMAP), multi-part and deep
files; every compression scheme (NONE, RLE, ZIPS, ZIP, PIZ, PXR24, B44,
B44A, DWAA, DWAB) for flat parts; HALF / FLOAT / UINT channels,
sub-sampled channels, layered and multi-view channel names, `lineOrder`
storage orders.

The crate root follows the OxideAV
[image-crate API contract](../../IMAGE_CRATE_API.md): the decoder hands
back one packed little-endian `f32` plane (`RgbF32Le` / `RgbaF32Le` /
`GrayF32Le`, scene-referred linear light, never clamped or tone-mapped)
and the encoder takes the same shape. The OpenEXR depth — every channel
by name, deep data, the per-channel writers, header attributes — lives
under its own names alongside.

Clean-room from the public OpenEXR file-format documentation. No
external library source consulted.

## Standalone use

```toml
oxideav-openexr = { version = "0.0", default-features = false }
```

```rust
let bytes = std::fs::read("in.exr")?;
if oxideav_openexr::probe(&bytes) {
    let info = oxideav_openexr::info(&bytes)?;      // header only: size, layout, parts, channels, colour
    let img = oxideav_openexr::decode(&bytes)?;     // ExrImage: one packed f32 plane
    let floats: Vec<f32> = img.pixels();            // width * height * components linear samples
    let rgba8: Vec<u8> = img.to_rgba8();            // clamp [0, 1] × 255 (no tone curve)
    let (w, h) = (img.width(), img.height());

    let opts = oxideav_openexr::EncodeOptions::default()
        .with_compression(oxideav_openexr::Compression::Piz);
    std::fs::write("out.exr", oxideav_openexr::encode(&img, &opts)?)?;

    // 8-bit in: bytes are linear / 255 unless `with_input_gamma(2.2)`.
    let _ = oxideav_openexr::encode_rgba8(w, h, &rgba8, &opts)?;
}
```

Root vocabulary: `probe`, `info -> ImageInfo`, `decode -> ExrImage`,
`decode_with(&DecodeOptions)`, `decode_rgb8 -> RgbImage`,
`decode_rgba8 -> RgbaImage`, `decode_all -> Vec<Frame>` (one per part;
`decode_all_with` takes options), `decode_from<R: Read>`,
`encode(&ExrImage, &EncodeOptions)`, `encode_rgb8`, `encode_rgba8`,
`encode_to<W: Write>`, `encode_all(&[Frame], &EncodeOptions)`
(multi-part); types `ExrImage { width, height, format, planes, color,
metadata, data_window, display_window, attributes }`, `Plane`,
`ColorInfo`, `ColorRange`, `Metadata`, `RgbImage`, `RgbaImage`,
`ImageInfo`, `Frame { image, delay, index, name, part_type }`,
`PixelFormat` (= `ExrPixelFormat`: `GrayF32Le`, `RgbF32Le`,
`RgbaF32Le`), `ExrError` (= `Error`: `InvalidData`, `Unsupported`,
`LimitExceeded`, `Io`). OpenEXR has no palette, so there is no `palette`
field.

`ExrImage` is built with the fallible constructors `new` / `packed` /
`from_f32` / `from_rgb8` / `from_rgba8` (geometry validated, so
`to_rgb8` / `to_rgba8` never fail); `with_color` / `with_metadata` /
`with_attributes` / `with_chromaticities` / `with_data_window` /
`with_display_window` fill the rest in. The float view is `pixels()`
(tight copy) and `pixel(x, y)`; the byte view is `as_bytes()` /
`into_raw()`.

### The colour view and the depth API

An OpenEXR part is an arbitrarily named channel set. `decode` returns
the part's **colour view**, chosen in this order from the selected layer
(`DecodeOptions::layer`, default the unprefixed base layer):

| Channels present | `ExrImage::format` |
|---|---|
| `R G B A` | `RgbaF32Le` |
| `R G B` | `RgbF32Le` (alpha is not synthesised) |
| `Y RY BY` (+ `A`) | `RgbF32Le` / `RgbaF32Le` — RGB reconstructed from luminance/chroma with the part's `chromaticities` weights (BT.709 default), chroma interpolated up from its sampling |
| `Y` + `A` | `RgbaF32Le` with `Y` replicated into R, G, B |
| `Y` | `GrayF32Le` |

HALF channels widen to `f32` exactly, FLOAT copies bit-for-bit, UINT
converts (exact below 2^24). Any other channel set (depth-only `Z`,
AOV layers, a file whose base layer is empty) is `Error::Unsupported`
naming the channels, as are deep parts; channels outside the view are
ignored. All of it stays reachable through the depth API, which keeps
its names: `parse_exr -> ExrPart` (every channel as a named `f32`
plane — before the contract this type was called `ExrImage`),
`parse_exr_multipart_mixed`, `parse_exr_tiled_multilevel`, the
`parse_exr_deep_*` readers, `parse_header`, `enumerate_layers`, and the
per-channel writers `encode_exr_scanline`, `encode_exr_tiled`,
`encode_exr_tiled_mipmap` / `_ripmap`, `encode_exr_multipart*`,
`encode_exr_deep_*`.

The pre-contract RGBA-float convenience writers
(`encode_exr_scanline_rgba_float*`, `encode_exr_tiled_rgba_float*`,
`encode_exr_multipart_rgba_float_with`) and the registry option type
names (`ExrDecoderOptions`, `ExrEncoderOptions`) remain for one release
as deprecated wrappers; see the CHANGELOG for the mapping.

## Framework use

```toml
oxideav-openexr = "0.0"    # default `registry` feature: pulls oxideav-core
```

`oxideav_openexr::register(&mut RuntimeContext)` installs the `openexr`
codec (decoder + encoder, `openexr_sw`) and the `.exr` extension hint;
`register_codecs` / `register_containers` / `register_registries` take
the individual registries, `make_decoder` / `make_encoder` are the
factories. `oxideav_meta::register_all` calls `register` for you.

The framework `Decoder` and `Encoder` are thin adapters over
`decode_with` / `encode` (one implementation). The decoder emits the
view's native layout — `RgbaF32Le` / `RgbF32Le` / `GrayF32Le`, one
packed plane, colour signal attached — with the options `part`,
`part_name` and `layer`. The encoder accepts the same three formats
natively and `Rgb24` / `Rgba` by the raw-path rule (`b / 255`,
`input_gamma` to linearise); its options schema is `pixel_type`,
`compression`, `colour`, `chroma_sampling`, `layer`, `tile_size`,
`levels`, `line_order`, `input_gamma`. One packet is one single-part
file; multi-part and deep files have no frame mapping (use
`decode_all` / `encode_all` and the depth API). The frame bridge is
`From<ExrImage> for VideoFrame` and `ExrImage::from_video_frame(&VideoFrame,
&CodecParameters) -> Result<ExrImage, ExrError>` (also
`TryFrom<(&VideoFrame, &CodecParameters)>`).

## Supported layouts

| Decode (native) | Encode |
|---|---|
| `GrayF32Le` — `Y` channel, 4 bytes/pixel | `GrayF32Le` → `Y` (FLOAT or HALF) |
| `RgbF32Le` — `R G B` or `Y RY BY` reconstructed, 12 bytes/pixel | `RgbF32Le` → `B G R`, or `BY RY Y` under `ColourLayout::LumaChroma` |
| `RgbaF32Le` — `R G B A`, `Y RY BY A`, or `Y A` replicated, 16 bytes/pixel | `RgbaF32Le` → `A B G R`, or `A BY RY Y` under `LumaChroma` |
| — | `encode_rgb8` / `encode_rgba8` / `Rgb24` / `Rgba` frames: `b / 255` → float (or `(b / 255) ^ input_gamma`), alpha kept as `a / 255` |

Every layout encodes as given (padded planes are repacked); nothing is
converted silently. `to_rgb8` / `to_rgba8` clamp each sample to `[0, 1]`
and scale `× 255` (nearest; `NaN` → 0) — no exposure or tone curve; gray
replicates into RGB, missing alpha is `255`. `decode(encode(img)) == img`
holds (planes, colour, windows and attributes) for `PixelType::Float`
with NONE / RLE / ZIPS / ZIP / PIZ, scanline and tiled; `Half` rounds
to binary16 (nearest even) and PXR24 / B44 / B44A / DWAA / DWAB are
lossy by design.

## Options

`EncodeOptions` (`#[non_exhaustive]`, `Default`, `with_*`): `pixel_type:
PixelType` (`Float` default, `Half`; `Uint` is `Unsupported` here — use
`encode_exr_scanline`), `compression: Compression` (`Zip` default; any of
the ten schemes), `colour: ColourLayout` (`Rgb` / `LumaChroma`),
`chroma_sampling: u32` (`2`; `1` = full-resolution chroma, exact round
trip; the image extents must be multiples of it), `layer: String`
(prefix for the written channel names), `tile_size: u32` (`0` =
scanline; else `N × N` tiles), `levels: LevelMode` (`One` / `Mipmap` /
`Ripmap`, box-filtered, tiled only), `line_order: LineOrder`
(`IncreasingY` / `DecreasingY` / tiled `RandomY`), `data_window` /
`display_window: Option<Box2i>` (overrides; `None` writes the image's),
`input_gamma: Option<f32>` (8-bit paths). Tiled output needs
full-resolution channels and windows at the origin (`Unsupported`
otherwise); `encode_all` is scanline / INCREASING_Y with parts at the
origin and a shared display window.

`DecodeOptions` (`#[non_exhaustive]`, `Default`, `with_*`): `max_width`
/ `max_height` (`Some(65_535)`), `max_pixels` (`None`), `max_bytes`
(`Some(1 GiB)` — the `f32` channel planes the part decodes to, every
channel × 4 bytes; a multi-part file is decoded whole, so the sum of its
parts is checked too), `strict` (`false`), `part: u32` (`0`),
`part_name: String` (overrides `part`), `layer: String` (`""` = base
layer). `unlimited()` lifts every limit.

## Metadata and colour

OpenEXR headers carry no ICC / Exif / XMP and no gamma record, so
`Metadata { icc, exif, xmp, gamma }` is always empty. Everything the
header does carry is `ExrImage::attributes`: the part's attributes
**except** the structural set the encoder regenerates (`channels`,
`compression`, `dataWindow`, `displayWindow`, `lineOrder`, `tiles`,
`chunkCount`, `version`, `type`, `name`, `maxSamplesPerPixel`), in file
order — `pixelAspectRatio`, `screenWindowCenter`, `screenWindowWidth`,
`chromaticities`, `owner`, `comments`, `capDate`, … A fresh image carries
the three required viewing attributes at their defaults
(`ExrImage::default_attributes()`); `encode` writes them all back
verbatim (scanline, tiled and multi-part). The structural facts are on
`ImageInfo` (`data_window`, `display_window`, `channels`, `compression`,
`tiled`, `deep`, `multipart`, `part_name`, `part_type`, `frames` =
part count) and `Frame` (`index`, `name`, `part_type`).

`ColorInfo { range, primaries, transfer, matrix }` (H.273 code points):
OpenEXR stores scene-referred linear light, so every image is `Full`
range, `transfer` 8 (linear), `matrix` 0 (RGB). `primaries` is the code
point whose chromaticities match the part's `chromaticities` attribute
within 1e-3 — BT.709 / sRGB 1, BT.470 M 4, BT.470 B/G 5, BT.601-525 /
ST 240 6, generic film 8, BT.2020 9, ST 428 XYZ 10, P3 DCI 11, P3 D65
12, EBU 3213 22 — else 2 with the exact coordinates on
`ExrImage::chromaticities()`. A part without the attribute follows the
format's documented default, Rec. ITU-R BT.709 primaries with D65
white, and reports 1. `ColorInfo::chromaticities_for(code)` is the
inverse; `with_chromaticities` sets the attribute and re-derives
`color`. The registry decoder stamps this as the frame's colour signal
(the format defines its colour semantics), and `from_video_frame`
writes a recognised non-BT.709 signal back as a `chromaticities`
attribute.

## Limits

Every `DecodeOptions` limit is checked against the part header(s)
**before** any plane is allocated (`ExrError::LimitExceeded`); the
view's channel set is planned from the header too, so an unviewable
part costs nothing. `info` applies no limit and reads only the header
region (a 60 001 × 60 001 window is described, not rejected). `probe`
is total and allocation-free. `strict` rejects a sub-sampled channel
whose data window is not aligned to and divisible by its sampling
factors (the lenient path reads ceil-sized planes, as before) and makes
`decode_all` fail on a part without a colour view instead of skipping
it. Below the contract layer the chunk readers keep their own hostile
input guards — bounds-checked offset tables, size-bounded inflate, the
DWA / PIZ reservation caps — exercised by the fuzz targets below.

## Format specifics

### Capability matrix

| Capability                          | Status                                           |
| ----------------------------------- | ------------------------------------------------ |
| Magic + version field               | parse + write (format-version 2)                 |
| Attribute table                     | parse + write; eight required attributes typed, plus typed inspectors for `int` / `double` / `string` / `v2i` / `v2d` / `v3i` / `v3f` / `v3d` / `m33f` / `m44f` / `m33d` / `m44d` / `chromaticities` / `box2f` / `tiledesc` / `rational` / `timecode` (BCD time accessors) / `keycode` / `stringvector` / `envmap` / `preview` / `floatvector` / `deepImageState` |
| Channel list (`chlist`)             | parse + write — `HALF`, `FLOAT`, `UINT`          |
| `lineOrder` (INCREASING_Y / DECREASING_Y / RANDOM_Y) | parse + write — observer-derived (r410): the chunk offset table is ALWAYS keyed canonically (top-first / ty-outer-tx-inner walk) and `lineOrder` governs only physical chunk storage order; RANDOM_Y is invalid for scanline images (writer rejects; reader stays lenient since chunks self-describe coordinates). `*_with_line_order` writer variants for scanline, tiled ONE_LEVEL, MIPMAP and RIPMAP; DECREASING_Y stores rows bottom-first, RANDOM_Y (tiled only) a deterministic shuffle. Reference-validated (header echo + independent reader + convert pixel-exact); derivation record in `tests/line_order_observer_notes.md` |
| Compression: `NONE`                 | parse + write                                    |
| Compression: `ZIP`  (16 lines/blk)  | parse + write (zlib)                             |
| Compression: `ZIPS` (1 line/blk)    | parse + write (zlib)                             |
| Compression: `RLE`                  | parse + write (byte-RLE + spec preprocessing)    |
| Compression: `PXR24` (16 lines/blk) | **parse + write** (scanline + tiled + multi-part) — encode: FLOAT→24-bit reduction (round mantissa to 15 bits) + byte-plane horizontal-delta + zlib deflate with raw fallback; decode: zlib inflate + prefix-sum + 24-bit reconstruction. HALF/UINT lossless. Decode validated bit-exact against the staged observer-spec's 24-bit reduction; encode round-trips through our decoder AND is accepted + decoded identically by a reference EXR validator binary. r410: the raw fallback now stores/detects the NATIVE chunk bytes (`compressed_len == uncompressed_len`), matching conforming readers — incompressible chunks carry FLOAT at full precision |
| Compression: `PIZ` (32 lines/blk)   | **parse + write** (scanline + tiled — ONE_LEVEL / MIPMAP / RIPMAP — + multi-part + mixed) — lossless: occupancy bitmap + chunk-derived range-compaction LUT + hierarchical 2D wavelet (14-bit and modulo-2^16 variants, selection recomputed from the bitmap) + canonical static Huffman (58-bit max codes, run-length escape at `iM`); HALF/FLOAT/UINT + sub-sampled channels; shared raw fallback. Landed r439 from the staged trace `openexr-piz-dwa-observer-spec.md` §2. Validated **bit-exact both directions** against a reference EXR binary (opaque process): reference-encoded PIZ decodes identically, and our PIZ files are accepted + decoded identically |
| Compression: `DWAA` (32 lines/blk) / `DWAB` (256 lines/blk) | **parse + write** (scanline + tiled — ONE_LEVEL / MIPMAP / RIPMAP — + multi-part + mixed) — 88-byte eleven-slot chunk header, version-2 rule block (+ staged legacy rule set for v0/v1), verbatim / AC / DC / RLE sub-streams, (suffix, pixel-type) channel classification, BT.709 CSC triples, binary32 perceptual half LUTs, staged truncated-π IDCT butterfly, plane-major DC + per-block AC with half-NaN run escapes, whole-region byte-plane RLE split; encoder emits v2 + static-Huffman AC, honours `dwaCompressionLevel` (default 45). Landed r439 (trace §3 + nine staged tables; implicit stream orderings observer-derived, record in `tests/dwa_observer_notes.md`). Decode validated **bit-exact** vs the reference's own decode of reference-encoded files; our chunks accepted + decoded bit-identically by the reference |
| Compression: `B44` / `B44A` (32 lines/blk) | **parse + write** (scanline + tiled + multi-part) — per-channel planes; HALF 4×4 blocks (14-byte packed + B44A 3-byte flat), edge replication, optional pLinear exp/log quantisation (tables computed bit-exact vs staged 65 536-entry CSVs); FLOAT/UINT copied raw; shared raw fallback. Encode searches the smallest 6-bit shift, applies the non-linear `exactmax` `t[0]` correction, and emits 3-byte flat blocks for B44A. Decode validated bit-exact against the staged observer-spec's B44 reduction; encode round-trips through our decoder AND is accepted + decoded identically by a reference EXR validator binary (b44/b44a) |
| Single-part scanline                | parse + write                                    |
| Single-part tiled (`ONE_LEVEL`)     | parse + write                                    |
| Tiled `MIPMAP_LEVELS`               | parse + write — full pyramid via `parse_exr_tiled_multilevel`; NONE / ZIP / ZIPS / RLE / PXR24 / B44 / B44A (r410: the single-part writer now carries the lossy schemes too, reference-validated incl. bit-exact reference-decode vs our-decode). `parse_exr` returns level-0 only |
| Tiled `RIPMAP_LEVELS`               | parse + write — full 2-D reduction grid; NONE / ZIP / ZIPS / RLE / PXR24 / B44 / B44A (r410: single-part writer incl. lossy) |
| Multi-part EXR (scanline parts)     | parse + write                                    |
| Multi-part EXR (flat tiled parts)   | parse + write — ONE_LEVEL + MIPMAP_LEVELS + RIPMAP_LEVELS, edge-tile aware |
| Sub-sampled channels (`xSampling` / `ySampling != 1`) | parse + write — lossless AND lossy (PXR24 / B44 / B44A) scanline paths; luminance/chroma (`Y` + 2×2 `BY`/`RY`) layouts validated bit-exact against a reference EXR validator binary. Note: the reference reader requires sub-sampled data-window extents divisible by the sampling factor; our reader additionally accepts ceil-sized odd extents |
| Layered / multi-view channel names (`diffuse.R`, `left.R` / `right.R`, `a.b.c.Y` …) | **typed enumeration + framework mapping** — `layers` module groups the channel list by prefix (arbitrary depth), classifies each layer (RGBA / RGB / luma-chroma / gray / depth / other) and tags views from `multiView`; `DecodeOptions::layer` (and the registry `layer` option) selects a layer (or the default view) and `EncodeOptions::layer` writes prefixed names. Validated against reference-produced multi-view files |
| Luminance/chroma colour (`Y` + `RY` + `BY` ↔ RGB) | **decode + encode** (`luma_chroma` module + the colour view) — `RY = (R − Y) / Y`, `BY = (B − Y) / Y` with luminance weights derived from the `chromaticities` attribute (BT.709 when absent); chroma reconstructed bilinearly from any `(xSampling, ySampling)`, reduced with a centred tent filter. Validated against a reference EXR tool (opaque process): chroma ratios and luminance match to HALF precision on constant-chroma images (filter-independent), smooth gradients agree to a colour-level tolerance; our files are accepted by the reference |
| Deep scanline (`deepscanline`)      | parse + write — NONE / RLE / ZIPS; single- and multi-part |
| Deep tiled (`deeptile`)             | parse + write — ONE_LEVEL + MIPMAP_LEVELS + RIPMAP_LEVELS, edge-tile aware; single- and multi-part |
| Multi-part **mixed** flat + deep    | parse + write — one file may freely mix `scanlineimage`, `tiledimage` (ONE_LEVEL / MIPMAP / RIPMAP), `deepscanline`, and `deeptile` (ONE_LEVEL / MIPMAP / RIPMAP) in any order. Multi-level flat **and deep** tiled parts now carry their full pyramid/grid inline (`MultipartMixedPart::DeepTiledMipmap` / `DeepTiledRipmap`, surfaced as `MultipartMixedImage::DeepTiledMipmap` / `DeepTiledRipmap`). Flat `scanlineimage` and `tiledimage` parts (ONE_LEVEL, MIPMAP, RIPMAP) also carry `PXR24` / `B44` / `B44A` (alongside NONE / ZIP / ZIPS / RLE), reusing the shared block builders + decoders; **deep** parts (scanline, ONE_LEVEL / MIPMAP / RIPMAP tiled) stay NONE / ZIPS / RLE |
| `HALF` (binary16)                   | round-trips every representable pattern (65 536) |
| `UINT` pixel type                   | parse + write (f32 view, bit-exact `< 2^24`)     |


### Not covered

* Lossy `PXR24` / `B44` / `B44A` / `PIZ` / `DWA` for **deep** parts —
  deep parts stay NONE / ZIPS / RLE (the spec text forbids PIZ for deep
  data and the format validators reject deep ZIP even though the spec
  page lists it).
* A reference EXR B44A decoder zeroes pLinear channels (its plain-B44
  decoder of identical data does not); our codec follows the
  observer-spec, so pLinear validation runs on the self-consistent
  plain-B44 path.
* Frames / contract images for channel sets outside RGB(A) / `Y` /
  `Y RY BY`: depth-only and AOV-only layers have no `PixelFormat`
  mapping and are `Unsupported` at the root — `parse_exr` returns every
  channel. Deep parts likewise (variable samples per pixel); use
  `parse_exr_deep_*`.
* One layer per decode: a frame / image carries a single layout, so
  layered and multi-view files are decoded one layer at a time with
  `DecodeOptions::layer` (enumerate them with `enumerate_layers` /
  `ExrPart::layers`).
* Tiled contract output keeps the data and display windows at the
  origin; `encode_all` writes scanline INCREASING_Y parts at the origin.
* The luminance/chroma reconstruction uses bilinear chroma
  interpolation (decode) and a centred tent reduction (encode); a
  reference EXR reader applies a different filter, so colour-level
  agreement is to a tolerance (tight in the interior, loosest at the
  image edges) while the container level stays bit-exact.

## Benchmarks

A Criterion harness (`benches/codec_benchmarks.rs`) measures decode /
encode throughput across every compression scheme for HALF and FLOAT,
scanline and tiled — see [`BENCHMARKS.md`](BENCHMARKS.md) for the
current numbers (scanline HALF NONE decode 2.5 GiB/s, FLOAT NONE
19.8 GiB/s, ZIP 0.78–0.82 GiB/s, PXR24 1.0–2.6 GiB/s, PIZ 0.31–0.37
GiB/s and DWAA/DWAB 0.38–0.83 GiB/s after the round-457 optimisation
pass, on Apple Silicon).

## Fuzzing

Five coverage-guided `cargo-fuzz` targets live under `fuzz/`:

```sh
cargo +nightly fuzz run contract_api
cargo +nightly fuzz run parse_flat
cargo +nightly fuzz run parse_deep_scanline
cargo +nightly fuzz run parse_multipart_mixed
cargo +nightly fuzz run decode_chunk
```

`contract_api` attacks the contract root — `probe`, `info`, `decode` /
`decode_with` (limits, strict, part / layer selection), `decode_rgb8` /
`decode_rgba8`, `decode_all` — raw and by splicing fuzz bytes over the
chunk region of writer-built scanline / luma-chroma / tiled multi-level /
multi-part bases across every compression scheme, so the header-only
planning, the pre-allocation limit checks and the view mapping are
reached directly; a successful decode must also survive `to_rgba8` and
re-`encode`.

`parse_flat` attacks the single-part flat readers — `parse_exr`
(scanline + tiled ONE_LEVEL) and `parse_exr_tiled_multilevel`
(MIPMAP / RIPMAP) — the only route into the PXR24 byte-plane/delta,
B44/B44A 4×4-block, PIZ (bitmap / range LUT / wavelet /
canonical-Huffman) and DWAA/DWAB (rule block / sub-streams / AC run
coding / IDCT) decode arithmetic; its overlay mode splices fuzz bytes
over the offset-table + chunk region of writer-produced valid files
across shape × compression × lineOrder combinations (all ten
compression codes since r439).
`decode_chunk` attacks the compressed-chunk decoders in isolation
through the hidden `chunk_api::decode_scanline_chunk` entry point —
PIZ, DWAA/DWAB, B44/B44A, PXR24 and the ZIP/RLE pipelines — so every
fuzz byte lands in chunk arithmetic instead of the header parser; its
overlay mode splices fuzz bytes over a writer-produced valid chunk of
the chosen scheme × channel set × width.
`parse_deep_scanline` attacks the deep scanline chunk walk.
`parse_multipart_mixed` attacks the mixed multi-part reader — the
per-part chunk-shape dispatch (flat scanline / flat + deep tiled at
every level mode), the concatenated offset tables, the i32 tile/level
coordinates, and the u64 deep chunk sizes — both raw and by splicing
fuzz bytes over the chunk region of a writer-produced two-part file.

The decode contract is that every byte slice returns `Ok` or `Err`,
never panicking, integer-overflowing (debug build), indexing out of
bounds, or allocating an attacker-claimed length the input can't back.
Offset-table entries (absolute `u64` byte positions read off the wire)
are bounds-checked with overflow-safe arithmetic so a near-`usize::MAX`
entry yields an error rather than wrapping past its EOF guard — see
`tests/offset_table_overflow_hardening.rs` and
`tests/multipart_mixed_hardening.rs`.

Sustained fuzzing of the DWA decode path surfaced three defects, all
fixed and pinned by unit tests: two reservation out-of-memories where a
chunk header's declared inflated size / symbol count sized an
allocation before any bytes were produced (the shared inflate helper
and the static-Huffman decoder now cap the eager reservation and let
the buffer grow only to what the stream yields), and an out-of-bounds
panic where an over-subscribed code-length table produced a Huffman
code that did not fit its bit length (the length distribution is now
validated as a proper prefix code before it indexes the decode table).

## License

MIT — see `LICENSE`.
