# REOSSI method blocks and implementation map

REOSSI means **Resolution Enhancement by Overlapping of Shifted and Scaled Identifiers**. The method organizes displaced low-resolution acquisitions, transforms their decoded identifiers and combines the transformed layers. The scanner also needs calibration, Gray classification and triangulation; those shared foundations are adapted from Charron and the sources acknowledged in [the notices](../THIRD_PARTY_NOTICES.md).

This map identifies the maintained responsibilities. Renaming or reorganizing inherited functions does not change their origin. The initial numerical profiles preserve the recovered implementation, including unresolved placement and coordinate conventions.

## Source responsibilities

| Block | Origin and maintained responsibility |
| --- | --- |
| Conventional Gray patterns, baseline illumination estimation, bit classification and Gray-to-integer decoding | Adapted scanner foundation from Charron and the algorithmic sources retained in the notices. |
| Maximum over repeated bit classifications, before integer decoding | Already present in Charron's [capture/decode script](https://github.com/guillaume-charron/3DScanner-GrayCode/blob/6f8abf0257c727eed81cc249edb85d43010c7b3d/src/3-capture_decode.py); this is not the later REOSSI displacement fusion. |
| One-based identifier adaptation and displaced low-resolution acquisition orchestration | Rolando's historical REOSSI implementation, built on that scanner foundation. |
| Per-displacement identifier scaling, maximum over transformed maps and optional target-grid scaling | Rolando's historical REOSSI operations, retained explicitly in the reconstruction pipeline. |
| ChArUco/projector calibration, geometric triangulation and visualization foundation | Adapted from Charron, with explicit configuration, file contracts and lifecycle corrections in the maintained project. |
| OAK acquisition integration | Rolando's historical hardware extension, separated into an optional adapter. |
| Packaging, configuration validation, independent IO workflows and three digital entry levels | Engineering structure of this project; these changes do not themselves establish a new scientific algorithm or physical resolution result. |

In particular, `np.max(cache_h_codes, axis=0)` belongs to repeated-bit aggregation. Scaling the resulting integer maps for each displacement and taking their final maximum is a separate operation, even though both stages use a maximum.

Original contributions are licensed under PolyForm Noncommercial 1.0.0; inherited
portions retain their MIT terms. The [licensing map](licensing.md) identifies the
original operations, adapted implementations and mixed-file boundaries.

## 1. Plan the native, emulated and target grids

Implementation: [`ScanPlan`](../src/reossi/plan.py).

- `projector_size` describes the native projector raster.
- `emulated_size` describes the low-resolution pattern raster.
- `target_size` optionally describes an intermediate maximum grid.
- `repeats` counts actual capture sequences at each position.
- `displacement_count` includes the initial position. It is one for conventional/LD mode and the source scale factor for REOSSI.

The scale factor retains the integer ratio of maximum dimensions. Without an intermediate target, native 1920×1080 and emulated 320×180 give factor 6, positions 0 through 5 and 38 frames per sequence. Two repetitions give 456 captured frames in total. A saved recording containing only one sequence per position must use `repeats=1`; it cannot supply the absent second acquisition.

Native dimensions, pattern dimensions and physical image dimensions are separate. Changing a ratio changes the acquisition count and numerical mapping. Nonproportional dimensions and truncated ratios are reported as compatibility notes rather than treated as a validated physical generalization.

## 2. Construct and acquire displaced patterns

Implementations:

- [`generate_pattern_sequence`](../src/reossi/pipeline.py) composes the chosen pattern profile.
- [`generate_gray_codewords` and `render_gray_patterns`](../src/reossi/structured_light/patterns.py) implement Gray words and raster order.
- [`run_capture`](../src/reossi/acquisition/capture.py) projects/captures actual images.
- [`iter_recorded_sequences`](../src/reossi/acquisition/recording.py) reads stored captures without creating missing runs.

The conventional profile uses zero-based identifiers. LD/REOSSI use one-based identifiers. The default `emulated_cells` placement retains the recovered generator's whole-cell shift before rasterization. The explicit `native_pixels` placement instead enlarges the unshifted raster and translates column/row bit frames right/down by `d` projector-raster pixels at position `d`, without wrapping. Black/white references remain fixed and newly exposed prefixes encode zero. Pattern export, physical capture and triangle simulation share this choice; select it with `--pattern-placement native_pixels` or `acquisition.pattern_placement`.

Chapter 8's experiment headed “Resultados usando REOSSI para crear una nube de puntos” states `Fe=6`, `Nd=5` and `step=W_HD=1` for 1920→320. This gives the initial position plus five one-pixel shifts. Here `W_LD` and `W_HD` denote fringe widths in projector-raster pixels, not image dimensions. The earlier general expression `step=W_LD/W_HD` conflicts with that example's step; the explicit native-target example determines this opt-in profile. Equal integer enlargement is required on both axes. The intermediate-target spatial and identifier units remain unresolved, so this profile rejects an intermediate target. See the [comparison](comparison.md) and [physical validation workflow](physical-validation.md).

Physical and digital acquisition produce the same grayscale stack contract `(frames,height,width)`. The ordering is black, white, interleaved column/row patterns and their inverses. The virtual raster/mesh modules create those images and then call the same decoder. They do not insert a different answer into the reconstructed maps.

## 3. Classify bits, combine repetitions and decode

Implementations:

- [`estimate_illumination`](../src/reossi/structured_light/illumination.py).
- [`classify_gray_bits`, `aggregate_repeated_bits` and `decode_identifier_maps`](../src/reossi/structured_light/decoding.py).
- [`decode_displacement`](../src/reossi/pipeline.py), which composes those operations for one position.

Direct/global estimation and Gray-bit classification belong to the inherited scanner foundation. Repeated classifications are combined with the inherited per-bit maximum before Gray words become integer identifiers. This operation is distinct from REOSSI's later combination of displacement maps.

Both source arithmetic and assignment order matter. A stored uint8 stack can differ from a float64 capture buffer, and later classification rules can overwrite earlier assignments. The implementation retains these behaviors. Alternate illumination estimation enters through an ordinary callable returning grayscale direct/global arrays; the default core does not require an external estimator.

## 4. Transform each decoded identifier layer

Implementation: [`scale_identifiers`](../src/reossi/reconstruction/identifiers.py).

For an identifier `m`, factor `F`, displacement index `d` and numerical increment `q`, the source transform is:

```text
S_d = F * (m - 1) + q * d + 1
```

The low-level function mutates its array and retains its dtype. By default exactly `-1` is preserved; zero and other negative values follow the formula. This rule is different from prototype variants that preserve both zero and `-1`, or transform every value. Alternative sentinel policies are explicit arguments, and experiments must propagate their validity masks accordingly.

`d` is an acquisition index. The numerical increment `q` must not automatically be interpreted as a measured physical displacement. In the main scanner profile it defaults to one. The intermediate-grid path retains the recovered native/target increment and second transform after fusion; that path requires an explicit unit review before physical resolution claims.

## 5. Fuse transformed displacement maps

Implementations: [`fuse_identifier_maps`](../src/reossi/reconstruction/identifiers.py) and [`reconstruct_identifier_maps`](../src/reossi/pipeline.py).

Column and row layers are aligned by camera pixel. The source selects the maximum transformed identifier at each pixel. Historical column stacking/reduction uses axis 1 and row stacking/reduction axis 0; both compute the same kind of per-pixel operation over their respective displacement layers.

The pipeline retains every decoded layer needed by that source operation. It does not replace fusion with averaging, a vote, interpolation or a learned reconstruction. The optional intermediate-target path applies its retained second scale afterward.

## 6. Distinguish the three digital entry levels

| Entry level | Implementation | What traverses the shared core |
| --- | --- | --- |
| Direct identifiers | [`simulation/identifiers.py`](../src/reossi/simulation/identifiers.py) | Shifted maps enter transformation/fusion directly |
| Raster captures | [`simulation/raster.py`](../src/reossi/simulation/raster.py) | Known sampled Gray images traverse classification, repeated-bit aggregation, decoding and REOSSI reconstruction |
| Triangle scenes | [`simulation/scene.py`](../src/reossi/simulation/scene.py) | Virtual camera/projector images include surface intersections and visibility before traversing the same image pipeline |

A clean direct-map construction with unit sample shifts can recover its target ramp. The `native_pixels` image profile also recovers native identifiers in the ideal-plane comparison through the shared classifier, decoder and fusion. The retained `emulated_cells` profile can instead fuse to the initial LD map under noiseless nearest-raster enlargement. Simulated coordinate truth remains independent so that distinction is visible. Neither check removes the inherited one-based terminal-identifier limitation on power-of-two emulated grids or establishes physical optical performance.

The triangle renderer knows surface points, projector coordinates and visibility from geometry. It models direct response and optional sensor noise, with no global light transport. Identifier-level errors, image noise and metric 3D errors remain separate quantities.

## 7. Reconstruct geometry and inspect results

Implementations:

- [`calibration/`](../src/reossi/calibration/) estimates camera/projector parameters or loads known virtual calibration.
- [`StereoTriangulator`](../src/reossi/reconstruction/geometry.py) retains the inherited geometric calculation.
- [`reconstruct_scan`](../src/reossi/reconstruction/workflow.py) handles explicit identifier origin, coordinate frame, aligned filtering and export.
- [`visualization.py`](../src/reossi/visualization.py) displays saved results independently of reconstruction.

Triangulation is shared scanner infrastructure. A denser identifier map does not by itself establish more accurate metric geometry. Reconstruction preserves camera-pixel/color association and exports the selected frame and length unit. Calibration quality, physical frame association and photometric realism require their own evidence.

Application-specific detection, distance queries and historical plotting campaigns are outside the initial core. Experiments can import these maintained operations without copying the scanner or performing work at import time.

## Numerical continuity and configurable choices

Resolution, repetitions, thresholds, normalization, camera settings, board geometry, grid placement and solver criteria are explicit. The homography RANSAC tolerance is separately configurable in destination board-length units. Some detector/initialization constants remain deliberately source-compatible and are listed in [calibration](calibration.md).

See [architecture](architecture.md) for coordinate/sentinel conventions, [configuration](configuration.md) for concrete files, and [development notes](development-notes.md) for implementation findings. The [Chapter 8 adversarial review](chapter-8-review.md) records alignment, corrected issues and retained limits. Source continuity, successful commands, known-geometry accuracy and physical validation are evaluated separately.
