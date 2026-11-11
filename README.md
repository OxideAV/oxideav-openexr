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
# Ok::<(), Box<dyn std::error::Error>>(())
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
`LimitExceeded`, `Layout(LayoutViolation)`, `Io`). OpenEXR has no palette, so there is no `palette`
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
the part's **colour view**, chosen from the selected layer
(`DecodeOptions::layer`, default the unprefixed base layer) by the
channel-presence rules of the reference RGBA interface:

| Channels present | `ExrImage::format` |
|---|---|
| `RY` and/or `BY` (+ `Y`, + `A`) | `RgbF32Le` / `RgbaF32Le` — RGB reconstructed from luminance/chroma (below); a missing `RY` / `BY` reads `0`, a missing `Y` reads `0.5` |
| `Y` + `A` | `RgbaF32Le` with `Y` copied into R, G, B |
| `Y` | `GrayF32Le` (luminance wins over `R` / `G` / `B` in the same layer) |
| `R G B A` | `RgbaF32Le` |
| `R G B` | `RgbF32Le` (alpha is not synthesised) |
| one or two of `R G B` (+ `A`) | `RgbF32Le` / `RgbaF32Le`, the missing components `0` |

Luminance/chroma decode reproduces the reference RGBA reader sample for
sample: luminance weights solved from the `chromaticities` attribute
(BT.709 / D65 default) in binary64 and rounded to binary32; chroma
co-sited with luma `(2i, 2j)` and brought up by the separable 27-tap
reconstruction filter (horizontal first, binary16 between passes, edge
replication); `R = (RY + 1)·Y`, `B = (BY + 1)·Y`, `G = (Y − R·w_r −
B·w_b) / w_g`; then the diagonal-neighbour saturation limiter (which
also clamps negative components). Every one of the nine traced
fixtures (`tests/fixtures/ryby/`) decodes **bit-exactly** — 11 776 / 11 776
RGBA samples, saturation stress fixture and multi-view layers included.

HALF channels widen to `f32` exactly, FLOAT copies bit-for-bit, UINT
converts (exact below 2^24). Any other channel set (depth-only `Z`,
AOV layers, alpha alone, a file whose base layer is empty) is `Error::Unsupported`
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
codec (decoder + encoder, `openexr_sw`) **and the `openexr` container**
(probe, demuxer, muxer, `.exr` extension); `register_codecs` /
`register_containers` / `register_registries` take the individual
registries, `make_decoder` / `make_encoder` are the factories.
`oxideav_meta::register_all` calls `register` for you, so
`oxideav_image::open(&ctx, "scene.exr")` resolves probe → demuxer →
decoder through the registry.

The framework `Decoder` and `Encoder` are thin adapters over
`decode_with` / `encode` (one implementation). The decoder emits the
view's native layout — `RgbaF32Le` / `RgbF32Le` / `GrayF32Le`, one
packed plane, colour signal attached — with the options `part`,
`part_name` and `layer`. The encoder accepts the same three formats
natively and `Rgb24` / `Rgba` by the raw-path rule (`b / 255`,
`input_gamma` to linearise); its options schema is `pixel_type`,
`compression`, `colour`, `chroma_sampling`, `luma_chroma_rounding`, `layer`, `tile_size`,
`levels`, `line_order`, `input_gamma`. One packet is one single-part
file. The frame bridge is `From<ExrImage> for VideoFrame` and
`ExrImage::from_video_frame(&VideoFrame, &CodecParameters) ->
Result<ExrImage, ExrError>` (also `TryFrom<(&VideoFrame,
&CodecParameters)>`).

### The `openexr` container

The demuxer (`container::open_demuxer`) declares **one video stream per
distinct (width, height, native layout, colour signal)** among the
viewable parts, in first-appearance order — a single-part file has one
stream; a stereo `left` / `right` pair shares one; a `RgbaF32Le` beauty
next to a `GrayF32Le` depth gives two — so every stream's parameters
(`width` / `height`, the `pixel_format` `info().format` reports for that
part, its colour signal: linear light, primaries from `chromaticities`,
BT.709 when absent) describe every packet on it. Packets carry their
stream index. The demuxer never touches pixels:

- a **single-part file** is one packet holding the whole file;
- a **multi-part file** yields one packet per part that has a colour
  view, in file order, each a valid single-part file repacked byte for
  byte from the part's header and chunks (`split_parts`: multi-part bit
  cleared, tiled bit set for `tiledimage` parts, offset table rewritten,
  chunks found through the tables and stripped of their part number;
  compressed payloads untouched). Deep parts and
  parts without a view are skipped exactly as `decode_all` skips them;
  a file with no viewable part is `Unsupported`;
- `pts` = the part's zero-based index in the file (gaps where parts were
  skipped, matching `Frame::index`) in a `1/1` time base. **OpenEXR
  parts are not timed**: packets carry no `duration`;
- `metadata()` carries `comment` / `owner` / `date` from the first
  part's `comments` / `owner` / `capDate` attributes and, for
  multi-part files, `parts` and `part_name:<index>`.

The muxer (`container::open_muxer`) accepts one or more video streams
and takes the encoder's packets: one packet in total is written
verbatim; several (on any streams, in arrival order) are combined into a
multi-part file
(`combine_parts_with`, the inverse repack — `name`, `type`, `chunkCount`
added, part names taken from the packets or `part<i>`, duplicates
suffixed `.<i>`; every part receives the shared `displayWindow`, the
union of the packets' windows, the rule `encode_all` applies, because a
multi-part file carries one display window for all its parts; packets
that disagree on `pixelAspectRatio` or `chromaticities`, the other two
shared attributes, are refused). Only flat scanline /
tiled single-part packets combine (what the encoder emits); a deep
packet is `Unsupported`. `decode_all` and `parse_exr_multipart*` read
the result, the registry round trip `demux(mux(frames)) == frames` is
pinned for every layout, and an independent reader (`exrinfo`, run as
an opaque process when installed) opens both the combined file and
every repacked packet.

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
`chroma_sampling: u32` (`2`, the layout conforming readers accept —
decimated with the luminance/chroma writer's 27-tap filter; `1` =
full-resolution chroma; the data window's origin and extents must be
multiples of it), `luma_chroma_rounding: bool` (`true`: stored `Y` keeps
7 and `RY` / `BY` 5 mantissa bits, as the reference writer stores them;
`false` keeps full binary16), `layer: String`
(prefix for the written channel names), `tile_size: u32` (`0` =
scanline; else `N × N` tiles), `levels: LevelMode` (`One` / `Mipmap` /
`Ripmap`, box-filtered, tiled only), `line_order: LineOrder`
(`IncreasingY` / `DecreasingY` / tiled `RandomY`), `data_window` /
`display_window: Option<Box2i>` (overrides; `None` writes the image's),
`input_gamma: Option<f32>` (8-bit paths), `threads: usize` (`1`;
worker budget — the bytes written do not depend on it). Tiled output needs
full-resolution channels (`chroma_sampling = 1` under `LumaChroma`) and
keeps the image's data / display windows like scanline output does.
`encode_all` writes each frame exactly as `encode` would — tiles,
levels, storage order, luminance/chroma, windows — and combines the
results into one multi-part file with a shared display window (the
union of the frames').

`DecodeOptions` (`#[non_exhaustive]`, `Default`, `with_*`): `max_width`
/ `max_height` (`Some(65_535)`), `max_pixels` (`None`), `max_bytes`
(`Some(1 GiB)` — the `f32` channel planes the part decodes to, every
channel × 4 bytes; a multi-part file is decoded whole, so the sum of its
parts is checked too), `strict` (`false`), `part: u32` (`0`),
`part_name: String` (overrides `part`), `layer: String` (`""` = base
layer), `threads: usize` (`1`; worker budget, see *Benchmarks*).
`unlimited()` lifts every limit.

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
`tiled`, `deep`, `multipart`, `part_name`, `part_type`, `layers`, `views`, `frames` =
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
it, and refuses the layout irregularities listed under *Multi-part
layout* below. Below the contract layer nothing is sized from a header the file does
not back: offset tables are sized from the part geometry and must fit
in the file (a huge `chunkCount`, data window, tile grid or part count
costs nothing), and a part declaring more sample data than its chunks
could hold at the best ratio of its compression scheme is refused
before a plane is allocated (`LayoutViolation::PixelDataNotBacked`).
The chunk readers keep their own hostile
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
| Multi-part layout rules             | **read + write, per the staged multi-part note** — version flags, header list, `name` / `type` / `chunkCount`, shared attributes, per-part offset tables, part-number prefixes; typed `LayoutViolation` errors, lenient / strict; byte-level `split_parts` / `combine_parts` (see *Multi-part layout*) |
| Multi-part EXR (scanline parts)     | parse + write                                    |
| Multi-part EXR (flat tiled parts)   | parse + write — ONE_LEVEL + MIPMAP_LEVELS + RIPMAP_LEVELS, edge-tile aware |
| Sub-sampled channels (`xSampling` / `ySampling != 1`) | parse + write — lossless AND lossy (PXR24 / B44 / B44A) scanline paths; luminance/chroma (`Y` + 2×2 `BY`/`RY`) layouts validated bit-exact against a reference EXR validator binary. Note: the reference reader requires sub-sampled data-window extents divisible by the sampling factor; our reader additionally accepts ceil-sized odd extents |
| Layered / multi-view channel names (`diffuse.R`, `right.R`, `spec.left.Y`, `a.b.c.Y` …) | **typed enumeration + framework mapping** — `layers` module groups the channel list by prefix (arbitrary depth, base = text after the last `.`), classifies each layer by the colour view's rules and tags views from `multiView`: a channel's view is the last prefix component when listed (`channel_view`; un-prefixed = the default view, `right.spec.G` = none). `DecodeOptions::layer` (and the registry `layer` option) matches the full prefix literally, except that the default view's name selects the un-prefixed set; `EncodeOptions::layer` writes prefixed names. The traced multi-view fixture decodes bit-exactly under every layer name and the per-channel view classification matches the reference's |
| Luminance/chroma colour (`Y` + `RY` + `BY` ↔ RGB) | **decode + encode, bit-exact to the reference RGBA interface** (`luma_chroma` module + the colour view) — decode: chromaticities-derived weights, 27-tap reconstruction, conversion, diagonal-neighbour saturation limiting (9/9 traced fixtures bit-exact). Encode (`ColourLayout::LumaChroma`): binary16 `Y`, per-pixel binary16 ratios, separable 27-tap decimation at even positions, 7-bit `Y` / 5-bit ratio mantissas — the reference-written fixture's stored `Y` / `RY` / `BY` planes reproduced 384 / 384. Sampling factors other than 1×1 / 2×2 fall back to bilinear / tent filters. Raw planes stay reachable as stored through `parse_exr` |
| Deep scanline (`deepscanline`)      | parse + write — NONE / RLE / ZIPS; single- and multi-part |
| Deep tiled (`deeptile`)             | parse + write — ONE_LEVEL + MIPMAP_LEVELS + RIPMAP_LEVELS, edge-tile aware; single- and multi-part. Tile sample data in the stored order (scanline by scanline within the tile, each channel contiguous per scanline) — the staged reference-written deep-tiled fixture decodes sample for sample equal to its deep-scanline twin |
| Multi-part **mixed** flat + deep    | parse + write — one file may freely mix `scanlineimage`, `tiledimage` (ONE_LEVEL / MIPMAP / RIPMAP), `deepscanline`, and `deeptile` (ONE_LEVEL / MIPMAP / RIPMAP) in any order. Multi-level flat **and deep** tiled parts now carry their full pyramid/grid inline (`MultipartMixedPart::DeepTiledMipmap` / `DeepTiledRipmap`, surfaced as `MultipartMixedImage::DeepTiledMipmap` / `DeepTiledRipmap`). Flat `scanlineimage` and `tiledimage` parts (ONE_LEVEL, MIPMAP, RIPMAP) also carry `PXR24` / `B44` / `B44A` (alongside NONE / ZIP / ZIPS / RLE), reusing the shared block builders + decoders; **deep** parts (scanline, ONE_LEVEL / MIPMAP / RIPMAP tiled) stay NONE / ZIPS / RLE |
| `HALF` (binary16)                   | round-trips every representable pattern (65 536) |
| `UINT` pixel type                   | parse + write (f32 view, bit-exact `< 2^24`)     |


### Multi-part layout

The `layout` module implements the staged multi-part observer note
end to end, without looking inside a chunk payload.

**Reading.** `FileLayout::read(bytes)` lays a file out — version flags,
one `PartLayout` per header (name, `PartKind`, the chunk count computed
from the part geometry, table offset, attributes) — and
`FileLayout::chunks` locates every chunk through the offset tables.
What a conforming reader refuses is an `ExrError::Layout(LayoutViolation)`
in every reader of this crate:

| Rule | Violation |
|---|---|
| format version 2; only flags `0x200` / `0x400` / `0x800` / `0x1000`; `0x200` never with `0x800` or `0x1000`, and only with a `tiles` attribute | `FormatVersion`, `UnknownVersionFlags`, `ConflictingVersionFlags`, `TiledFlagWithoutTiles` |
| the header list of a multi-part file ends in an empty header | `HeaderListNotTerminated` |
| every part of a multi-part file has a string `name` and a `type` (also required in a deep file) out of `scanlineimage` / `tiledimage` / `deepscanline` / `deeptile` | `MissingName`, `NameNotString`, `MissingType`, `UnknownPartType` |
| a tiled part has `tiles`, a scanline part of a multi-part file has none; deep data version 1 | `MissingTiles`, `UnexpectedTiles`, `DeepDataVersion` |
| `displayWindow`, `pixelAspectRatio` (absent = 1.0) and `chromaticities` (may be absent from a later part, not present there without being identical in part 0) agree with part 0 | `SharedAttributeMismatch { part, attribute }` |
| the offset tables (sized by the note's formulas) fit in the file; every entry points at a chunk inside the file | `OffsetTableTruncated`, `ChunkOutOfBounds`, `ChunksMissing` |
| a chunk carries the number of the part whose table led to it | `ChunkPartNumber` |

Chunks are found through the tables, never by position: any physical
order, interleaving between parts and bytes between tables and chunks
are fine. What a conforming reader tolerates decodes as well, and is
listed in `FileLayout::irregularities` — `ChunkCountMissing`,
`ChunkCountMismatch` (the count is never trusted), `OffsetTableZero`
(the tables are rebuilt from the chunks' part numbers and coordinates),
`DuplicatePartName`, `LongNameFlagClear`, `DeepFlagWithoutDeepPart`,
`DeepPartWithoutDeepFlag`. `DecodeOptions::strict` and
`check_layout(bytes, true)` turn the first of them into the error.

The eleven staged rejected files fail with the rule their edit breaks,
the eleven staged tolerated files decode to the unmodified pixels, and
every part of every staged fixture decodes to the SHA-1 pixel hash its
producer reported (`tests/multipart_fixtures.rs`).

**Writing.** Every writer's output is held to the strict rules before
it is returned: exact `chunkCount` in every part, filled tables in
geometry order, unique part names, shared attributes that agree, the
long-name flag whenever a name exceeds 31 bytes, the deep flag exactly
when a part is deep, and never the single-part-tiled flag on a
multi-part file. `encode_exr_multipart_mixed` mixes `scanlineimage`,
`tiledimage`, `deepscanline` and `deeptile` parts freely; black-box
readers read the headers (two tools) and the pixels (a third, which
re-writes every chunk) of every such file in the property test
(`tests/multipart_layout_roundtrip.rs`).

**Repacking.** `split_parts(bytes)` returns every part as a single-part
file (flat or deep, scanline or tiled; `SplitOptions::strip_part_attributes`
drops `name` / `type` / `chunkCount` to give the file a single-part
writer produces); `combine_parts(&files)` is the inverse for any mix of
single-part files (`CombineOptions`: explicit part names, renaming
duplicates, unifying the display window). No payload byte is touched,
and the staged fixtures are reproduced byte for byte both ways: the
split parts equal the staged single-part files, and the single-part
files combine into the staged multi-part files.

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
* Luminance/chroma channels are modelled on HALF data; FLOAT `Y` /
  `RY` / `BY` go through the same binary16 model (the reference RGBA
  interface's output type). A layer name that selects no channel is a
  caller error (`InvalidData`) rather than an empty frame.

## Benchmarks

### Worker budget

Scanline blocks and tiles are independent, so reading and writing fan
them out over a caller-chosen worker budget: `DecodeOptions::threads` /
`EncodeOptions::threads` on the contract API, the framework's
`ExecutionContext` on the registry decoder / encoder
(`set_execution_context`), or `with_threads(n, || ..)` around a
depth-API call (`parse_exr`, `parse_exr_tiled_multilevel`,
`encode_exr_scanline`, `encode_exr_tiled*`; the `parse_exr_multipart*`
/ deep readers and the `encode_exr_multipart*` / deep writers run
serially). The budget is `1` unless raised; **decoded samples, written bytes
and the error on malformed input are the same at every budget**
(`tests/thread_budget.rs`: every scheme x scanline / tiled / mip-map /
rip-map x HALF / FLOAT, sub-sampled chroma, multi-part files, damaged
and truncated files, budgets 1 / 2 / 3 / 8).

- encode: every block / tile payload is built by a worker, then the
  file is assembled in order;
- decode, scanline: bands of rows (aligned to the block height and the
  channels' vertical sampling) are decoded by workers and stitched;
- decode, tiles: each tile is decoded by a worker and stitched in table
  order (`parse_exr` and the every-level reader
  `parse_exr_tiled_multilevel` alike); the packed colour view is
  interleaved in bands;
- a part of a multi-part file is lifted out as a single-part file and
  takes the same path, so `decode_with(part = k)` decodes only part `k`
  (it used to decode every part and keep one).

Contract API (`decode_with` / `encode`), megapixels per second, on a
2048 x 1024 RGBA image (gradients plus fine noise), Apple Silicon,
`cargo run --release --example throughput`. *Before* is the serial code
before this pass, built from its commit and run alternately with the
current code. The machine was shared with other builds, so each figure
is the best of four runs (each run the median of five passes); serial
differences below about 15 % are noise — an interleaved six-round A/B of
PIZ, B44 and DWAB showed the same serial speed before and after, except
where noted below.

| Scheme | Layout | Type | decode before | decode ×1 | decode ×8 | encode before | encode ×1 | encode ×8 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| NONE | scanline | HALF | 179 | 267 | 784 | 107 | 132 | 304 |
| NONE | scanline | FLOAT | 318 | 574 | 1086 | 214 | 232 | 341 |
| NONE | tiled 256 | HALF | 182 | 226 | 424 | 76 | 86 | 265 |
| NONE | tiled 256 | FLOAT | 255 | 386 | 564 | 115 | 150 | 314 |
| RLE | scanline | HALF | 78 | 91 | 463 | 59 | 69 | 232 |
| RLE | scanline | FLOAT | 78 | 88 | 464 | 62 | 64 | 231 |
| RLE | tiled 256 | HALF | 76 | 88 | 314 | 51 | 56 | 215 |
| RLE | tiled 256 | FLOAT | 73 | 79 | 306 | 51 | 53 | 210 |
| ZIPS | scanline | HALF | 27 | 28 | 180 | 9 | 9 | 57 |
| ZIPS | scanline | FLOAT | 16 | 18 | 117 | 5 | 6 | 38 |
| ZIPS | tiled 256 | HALF | 30 | 34 | 172 | 6 | 6 | 41 |
| ZIPS | tiled 256 | FLOAT | 15 | 17 | 102 | 4 | 4 | 27 |
| ZIP | scanline | HALF | 30 | 33 | 198 | 7 | 7 | 44 |
| ZIP | scanline | FLOAT | 16 | 17 | 108 | 5 | 4 | 25 |
| ZIP | tiled 256 | HALF | 32 | 32 | 166 | 6 | 6 | 40 |
| ZIP | tiled 256 | FLOAT | 16 | 16 | 99 | 4 | 4 | 25 |
| PIZ | scanline | HALF | 41 | 55 | 265 | 41 | 40 | 169 |
| PIZ | scanline | FLOAT | 17 | 17 | 113 | 9 | 18 | 99 |
| PIZ | tiled 256 | HALF | 42 | 53 | 200 | 38 | 37 | 166 |
| PIZ | tiled 256 | FLOAT | 17 | 17 | 99 | 9 | 18 | 96 |
| PXR24 | scanline | HALF | 28 | 29 | 187 | 3 | 3 | 21 |
| PXR24 | scanline | FLOAT | 19 | 19 | 128 | 3 | 3 | 18 |
| PXR24 | tiled 256 | HALF | 31 | 32 | 159 | 4 | 4 | 25 |
| PXR24 | tiled 256 | FLOAT | 19 | 20 | 118 | 3 | 3 | 21 |
| B44 | scanline | HALF | 119 | 105 | 474 | 33 | 30 | 142 |
| B44 | scanline | FLOAT | 423 | 608 | 1183 | 173 | 151 | 314 |
| B44 | tiled 256 | HALF | 116 | 112 | 300 | 28 | 25 | 127 |
| B44 | tiled 256 | FLOAT | 225 | 258 | 449 | 86 | 83 | 246 |
| B44A | scanline | HALF | 123 | 104 | 523 | 34 | 31 | 144 |
| B44A | scanline | FLOAT | 415 | 523 | 1211 | 174 | 148 | 305 |
| B44A | tiled 256 | HALF | 116 | 109 | 310 | 27 | 25 | 131 |
| B44A | tiled 256 | FLOAT | 207 | 252 | 485 | 89 | 85 | 246 |
| DWAA | scanline | HALF | 44 | 47 | 244 | 21 | 23 | 114 |
| DWAA | scanline | FLOAT | 38 | 40 | 232 | 20 | 21 | 111 |
| DWAA | tiled 256 | HALF | 44 | 45 | 202 | 19 | 21 | 111 |
| DWAA | tiled 256 | FLOAT | 38 | 37 | 172 | 19 | 20 | 109 |
| DWAB | scanline | HALF | 44 | 45 | 141 | 20 | 22 | 65 |
| DWAB | scanline | FLOAT | 40 | 38 | 129 | 19 | 22 | 64 |
| DWAB | tiled 256 | HALF | 44 | 44 | 192 | 19 | 21 | 104 |
| DWAB | tiled 256 | FLOAT | 37 | 37 | 171 | 19 | 20 | 107 |

What moved besides the fan-out (serial columns): the packed colour view
is filled in place instead of grown sample by sample (NONE decode
179 → 267, FLOAT 318 → 574; B44 / B44A FLOAT, which store FLOAT
uncompressed, 423 → 608), and tile payloads are built from row slices
with the pixel-type dispatch outside the sample loop (NONE FLOAT tiled
encode 115 → 150). In PIZ, whose time is mostly entropy coding: the
Huffman code lengths are built with a two-queue merge instead of a
priority queue (FLOAT chunks code tens of thousands of distinct words:
PIZ FLOAT encode 9 → 18), and the decoder resolves several table codes
per refill and widens its lookup table to 16 bits on large chunks with
long codes (PIZ HALF decode 41 → 55). Both are pinned against the
previous behaviour: identical code lengths (so identical files) and
identical decode results at every table width. The DWA encoder's
coefficient quantiser evaluates its candidates in place instead of
converting each back from binary16 (the same choice for every input,
pinned; DWAA encode 21 → 23). The zlib family is bounded by the deflate
/ inflate of the workspace's compression crate (a profile of ZIP and
PXR24 puts about 90 % of the time inside it), which is also why
PXR24 encode sits at 3 Mpx/s: its byte planes compress poorly.

### Criterion harness

A Criterion harness (`benches/codec_benchmarks.rs`) measures the depth
readers and writers per compression scheme on a small image — see
[`BENCHMARKS.md`](BENCHMARKS.md).

## Fuzzing

Coverage-guided `cargo-fuzz` targets live under `fuzz/`:

```sh
cargo +nightly fuzz run contract_api
cargo +nightly fuzz run parse_flat
cargo +nightly fuzz run parse_deep_scanline
cargo +nightly fuzz run parse_multipart_mixed
cargo +nightly fuzz run decode_chunk
cargo +nightly fuzz run demux
cargo +nightly fuzz run layout
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

`layout` attacks the file-layout module — header list, version flags,
geometry-derived chunk counts, offset tables, part numbers, and the
split / combine repack — raw, with fuzz bytes over the tables of a
writer-built four-kind mixed file, and with short patches inside its
header region; a file the strict check accepts must split, recombine
and reach a fixed point, and every input must decode to the same frames
or the same error at worker budgets 1 and 3.

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
