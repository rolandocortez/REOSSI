# Development notes and open validation items

These notes collect small implementation findings and limitations that matter when using or extending the project. Experimental conclusions require the corresponding inputs and controls; successful tests are not a physical-device validation.

## Numerical source conventions retained

- Repeated classified bits and transformed displacement identifiers are combined at different stages.
- Local profiles start identifiers at 1; conventional processing starts at 0. Geometric reconstruction requires an explicit interpretation.
- The recovered `emulated_cells` profile shifts whole LD cells. The opt-in `native_pixels` profile follows Chapter 8's explicit native-target example: enlarge first, then shift by one projector-raster pixel per position. Export, capture and triangle simulation use the same generator. The intermediate-target unit convention and identity of the historical producer remain unresolved; physical operation of this refactored path still needs a rig session.
- One-based power-of-two grids retain their final-identifier truncation. Maximum-dimension ratios retain integer truncation and the target-grid path retains its second scaling.
- Stored uint8 intensity arithmetic differs from float64 capture buffers. Source-compatible decoding does not silently promote recordings.
- The source classifier's later assignments can overwrite an earlier low-direct-illumination assignment.
- A single physical board definition replaces ambiguous per-stage assumptions. Historical local board lengths must be checked against the actual print.

## Calibration convergence finding

An autonomous known-geometry projector test uses an initial focal length of 3000 pixels and true focal lengths of 720/700 pixels. OpenCV's default termination stops at RMS≈0.383748 with estimated focal lengths≈2679/2444. An explicit criterion of 100 iterations and epsilon 1e-12 recovers the known intrinsics with RMS≈2.96e-6. This is a bounded synthetic case, not a statement that every default calibration fails.

Source defaults remain available when criteria are omitted. The example configuration explicitly supplies camera, projector and stereo criteria. Reprojection RMS alone does not certify correct intrinsics or adequate pose diversity.

The pixel-to-board homography has a separate `homography_reprojection_threshold`, measured in destination board-length units. Its numeric source default of 5.0 is retained and exposed explicitly. A known-correspondence test confirms that physically equivalent metre/millimetre thresholds preserve inliers, while the same numeric threshold in both units can select different inliers. This parameter is separate from solver termination.

## Recorded-input diagnostic

A bounded replay used one 128×128 camera crop from each of six available historical sequences, with an explicit one-run plan. Stored uint8 input and explicitly converted float64 input both returned positive joint identifiers at all 16,384 pixels, but their column identifiers differed at 14,397 pixels and row identifiers at 13,748 pixels. Under the diagnostic's declared 1920×1080 projector size, 12,128 uint8 pixels exceeded each axis's maximum; the float64 variant had no such exceedances.

These are observations about the available inputs and arithmetic, not correspondence ground truth. The declared projector dimensions are compatible with inspected sources but are not independently established by recording metadata. The likely capture-side exporter retains the last raw run, so one stored sequence cannot reproduce the original combination of two runs. Available color references also lack proven pairing across runs/angles. The package preserves dtype and requires an explicit repeat count rather than manufacturing absent captures.

## Resource and input corrections

- Reject NaN/Inf resource limits and invalid sensor/tolerance/pose settings before costly work.
- Check the ray-triangle budget before creating camera rays or generated scene meshes.
- Reject extra repetitions before decoding or caching them; incomplete sequences and empty frames fail clearly.
- Bound valid reconstruction correspondences before Python-list allocation. The default cap is 1 million; changing it is an explicit resource choice.
- Capture guards apply to the requested size before opening a camera and to actual frame size before allocating its buffer.
- Recorded stacks are checked against both their ZIP entry size and their declared NPY shape before loading. A small archive cannot bypass the stack limit by declaring a much larger array.

## Virtual geometry tolerances

Ray intersection tolerances scale with the scene extent so that ordinary changes of world units preserve visibility. Samples exactly on a half-pixel projector boundary can still change under floating-point rounding. A scene containing a distant, unused mesh component can also enlarge the extent and its tolerances; remove irrelevant geometry when constructing controlled scenes.

The `camera_visible_unilluminated_pixels` model counter includes out-of-viewport and behind-projector points as well as occlusions. An unobstructed plane can therefore have a nonzero count; this quantity is not a shadow-only measurement.

## Acquisition lifecycle

Adapters initialize shared frame state before starting readers, coordinate frame access, and release drivers after their reader exits. Constructor failures close opened resources. A stuck backend read raises a bounded shutdown error rather than racing a release against the reader. Actual SDK blocking behavior still needs device validation.

Scan and calibration collection close only windows they own. Completed captures and effective settings survive interruption. The original exception is preserved if cleanup also fails.

## Export and visualization

Reconstruction retains camera-pixel association through finite/box filtering and writes a valid empty PLY for empty results. Camera, projector and legacy coordinate frames are explicit. Plotting is separate from numerical filtering/export; viewing a cloud no longer invokes an implicit statistical filter or overwrites a PLY file.

The reference correspondence collector supplies float32 image points. A known planar end-to-end example has micrometre-scale numerical differences at a two-metre depth even with exact integer image correspondences; closed-loop tests use an explicit float32 geometric tolerance. This is numerical precision evidence, not physical metrology.

## Validation boundaries

Small tests cover analytical words, source-compatible arithmetic, known geometric correspondences, shadow visibility, real OpenCV solves on generated observations, replay and mocked hardware lifetime. Full-resolution data, physical timing and actual camera/projector behavior remain separately pending. The direct-only renderer does not establish global-illumination accuracy.
