# Synthetic inputs and virtual structured light

There are distinct entry levels. Raster and 3D scene simulations generate images and traverse the same decoder as real acquisition. Identifier-map demonstrations enter directly at the REOSSI transformation/fusion stage. Identifier-level results do not by themselves validate projection, classification or optical behavior.

## Fast raster diagnostic

```bash
reossi simulate --config configs/synthetic.json --output outputs/raster-clean
reossi simulate --config configs/synthetic.json --output outputs/raster-noisy \
  --noise-std 3 --seed 7
```

Each camera pixel samples a known location over a rectangular native projector field. Emulated patterns are enlarged by integer-floor nearest-neighbor lookup. Captures use `black_level + direct_level * pattern/255`, optional Gaussian intensity noise, clipping to 0..255 and float64 storage. Each repetition gets independent noise from a seeded generator.

Decoded maps and independent expected identifiers are saved with model settings. Conventional coordinates start at 0 and local LD/REOSSI identifiers start at 1. Validity uses the applicable origin; zero is not credited as a valid one-based identifier.

## A virtual scanner over real 3D surfaces

```bash
reossi simulate-scene --config configs/synthetic_3d.json \
  --scene-config configs/scene_plane.json \
  --output outputs/virtual-plane --save-captures

reossi simulate-scene --config configs/synthetic_3d.json \
  --scene-config configs/scene_curved.json \
  --output outputs/virtual-curved --save-captures
```

The renderer casts calibrated camera rays into a triangle mesh. For each camera-visible surface point it computes projector coordinates, tests whether the projector sees that point, and samples the displayed Gray pattern. Geometry is computed independently of decoded identifiers. The implementation uses the barycentric ray/triangle equations described by [Möller and Trumbore](https://my.eng.utah.edu/~cs6965/papers/MT97.pdf).

A second visibility query distinguishes projector shadows from camera background. Missed surfaces remain NaN in geometric truth and never become illuminated points. Continuous projector coordinates are rounded to the nearest native pixel before the emulated raster lookup. This is an explicitly discrete projector model.

Scene configuration specifies device image sizes, intrinsics or focal lengths, poses, a generated plane/height field and sensor parameters. Device poses satisfy `X_device = R @ X_world + T`. The camera-to-projector calibration is derived from those poses, not estimated using decoded results. World/translation units are explicit.

Use `--mesh object.ply` for an external triangle mesh. NPZ files containing `vertices` and `faces` need no additional package; OBJ/STL/PLY loading uses optional `trimesh`. Input points without faces are rejected. Surface reconstruction from a point cloud is a separate modeling decision; no automatic meshing is implied.

### Saved results

| File | Contents |
| --- | --- |
| `h_pixels.npy`, `v_pixels.npy` | Actual decoded/fused identifiers |
| `true_columns.npy`, `true_rows.npy` | Independently known native pixel identifiers, including origin convention |
| `true_projector_pixels.npy` | Continuous geometric projector coordinates |
| `true_points_camera.npy` | Camera-frame surface geometry |
| `camera_visible.npy`, `illuminated.npy` | Separate camera and projector visibility masks |
| `calibration.npz` | Known camera/projector intrinsics and relative pose |
| `scene_mesh.npz` | Exact triangular scene used by the run |
| `white.png` | Captured virtual white-reference image |
| `simulation.json`, `scan.json` | Effective model, metrics and plan |
| `captures/` when requested | Replay-compatible run/displacement NPZ files and white reference |

Replay can therefore verify the full capture contract:

```bash
reossi replay --config configs/synthetic_3d.json \
  --input outputs/virtual-curved/captures --output outputs/virtual-replayed
reossi reconstruct --input outputs/virtual-replayed \
  --calibration outputs/virtual-curved/calibration.npz \
  --identifier-origin one --output outputs/virtual-cloud
```

### Metrics and limits

Coverage is measured over truly illuminated pixels. Identifier errors are measured on jointly valid illuminated pixels; decoded pixels outside the true illuminated area are reported separately. Metric 3D error requires comparing reconstructed camera pixels with their corresponding true surface points; a dense-looking cloud is not an accuracy result.

The model counter `camera_visible_unilluminated_pixels` includes camera-visible surface points outside the projector viewport, behind the projector, or occluded from it. It does not count projector shadows alone.

The initial model has uniform direct response, scalar black level and optional additive sensor noise. It has no lens distortion, projector blur, texture/BRDF, transparency, interreflection or subsurface scattering. Its direct-only images are not a ground-truth model of global light separation.

The default `emulated_cells` placement preserves the recovered REOSSI generator's whole-cell shifts and their coordinate errors. The explicit `native_pixels` option enlarges an unshifted pattern to the native projector raster and shifts its column/row bit images by one native pixel per position. Black/white references remain fixed; shifted prefixes encode zero without wrapping. This option represents the Chapter 8 experiment with native target resolution, for which `step=W_HD=1` is stated explicitly. It requires equal integer enlargement factors in both axes and does not support an intermediate target grid. It changes neither the default placement nor physical capture behavior.

For example, add `--pattern-placement native_pixels` to `simulate-scene`, or run the [three-mode comparison](comparison.md). The model records the selected placement in `simulation.json`. Conventional ideal-plane checks and identifier reconstruction remain separate from optical realism or physical scanner accuracy.

Resources are bounded before expensive allocation: one image-stack limit, a camera/shadow ray-triangle work budget, and a separate reconstruction correspondence limit. The direct triangle implementation is O(camera pixels × triangles); large scenes require an accelerated renderer. Check process memory separately from the per-stack limit.

Intersection distance tolerances use the scene extent and are tested across changes of world units and rigid poses. Exact half-pixel projector boundaries remain sensitive to floating-point rounding. Distant mesh components that are irrelevant to the image can inflate extent-based tolerances; keep controlled scenes geometrically bounded.
