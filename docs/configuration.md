# Configuration

Sizes use `[width, height]`; image arrays use `[row, column]`. A scan configuration is either a scan-plan object or a combined object with numerical fields under `plan` and hardware fields under `acquisition`. Calibration has its own top-level fields, described below. The files in `configs/` are templates to adapt to the actual setup.

## Scan plan and resolution

```json
{
  "plan": {
    "mode": "reossi",
    "projector_size": [1920, 1080],
    "emulated_size": [320, 180],
    "target_size": null,
    "repeats": 2
  },
  "acquisition": {
    "camera": "opencv",
    "camera_backend": "auto"
  }
}
```

| Plan field | Default | Meaning and accepted values |
| --- | --- | --- |
| `mode` | `"reossi"` | `conventional` uses native patterns; `ld` processes the initial low-resolution position; `reossi` processes all configured source displacements. |
| `projector_size` | `[1920,1080]` | Native projector raster, independent of camera image size. |
| `emulated_size` | `[320,180]` | Low-resolution pattern grid for LD/REOSSI; conventional mode instead uses `projector_size`. |
| `target_size` | `null` | LD/REOSSI-only intermediate grid used by the retained coordinate transforms; it does not resize the camera or change the native projector raster. |
| `repeats` | `2` | Positive integer: actual sequences captured at each displacement. Missing repetitions are errors. |
| `contrast_margin` | `1` | Finite, nonnegative Gray-classification intensity margin. |
| `minimum_direct` | `10` | Finite, nonnegative direct-illumination threshold in the classifier. |
| `illumination_epsilon` | `null` | Finite, nonnegative normalization override. Null selects `0` for conventional mode and `1e-5` for LD/REOSSI. |

Resolution components are integers at least 2. The selected pattern grid must be landscape/square and require 3..16 bits for the inherited renderer/selectors. A one-based power-of-two grid retains its final-identifier limitation.

For `pattern_size`, use the native size in conventional mode and the emulated size otherwise. The sequence length is `4*ceil(log2(max(pattern_size)))+2`. The scale factor retains `int(max(target_size or projector_size)/max(pattern_size))`; REOSSI acquires that many positions including position 0, while LD/conventional acquire one. Without an intermediate target, the native step is 1. With one, it retains `int(max(projector_size)/max(target_size))` and the historical second scaling. Both integer factors must be at least one. See [method and displacement units](method.md) before interpreting that increment physically.

For example, native 1920×1080 with emulated 320×180 gives factor 6 and 38 frames per sequence; emulated 640×360 gives factor 3 and 42 frames. `reossi plan --config ...` reports effective factors, frame counts and retained aspect-ratio, integer-truncation and encoding limitations without allocating patterns. The profiles `conventional.json`, `ld_320.json`, `reossi_320.json` and `target_960.json` cover these different choices.

Input dtype and numeric scalar types are part of source-compatible decoding. JSON `1` remains a built-in integer and `1.0` a float; changing `contrast_margin` between them can change NumPy arithmetic on a preserved uint8 recording. Live capture uses float64, while replay preserves the stored dtype. Keep both dtype and threshold types fixed when comparing equivalent runs; see [preserved numerical behavior](architecture.md#preserved-reossi-behavior).

## Camera selection and scan acquisition

CLI camera overrides are shared by `capture`, `collect-camera-calibration` and `collect-projector-calibration`. JSON keys belong to `acquisition`.

| Acquisition field | CLI override | Default and scope |
| --- | --- | --- |
| `camera` | `--camera` | `opencv` or `oak`; default `opencv`. |
| `camera_index` | `--camera-index` | Integer source index, default 0. For OAK this is retained interface compatibility; use `device_id` for SDK selection. |
| `camera_backend` | `--camera-backend` | OpenCV `dshow` by default; `auto` and `v4l2` are alternatives. It does not select an OAK device. The example files explicitly use `auto`. |
| `camera_size` | `--camera-size WIDTHxHEIGHT` | Requested positive integer dimensions: default 1920×1080 for OpenCV and 3840×2160 for OAK. Calibration collection uses the top-level calibration `camera_size`; an override must match it. |
| `fps` | `--fps` | Finite positive frame-rate request, default 30. |
| `device_id` | `--device-id` | OAK-only nonempty MXID/IP/USB-name string; null/omitted retains SDK discovery. |
| `settle_seconds` | `--settle-seconds` (scan) | Scan only: finite, nonnegative pattern dwell; default .3 s for conventional mode and 1 s for LD/REOSSI. The first pattern of the first sequence waits three times that value. |
| `flip` | `--flip` / `--no-flip` (scan) | Scan default is false for conventional mode, true for LD/REOSSI; calibration collection defaults false. True rotates frames 180° before processing/saving. |
| `window_position` | `--window-x` and `--window-y` together (scan) | Projector-window desktop coordinates. Scan default: unchanged placement for conventional mode, `[2020,100]` for LD/REOSSI. Projector-calibration collection defaults to unchanged placement. Adapt this to the actual monitor arrangement. |
| `fullscreen` | `--fullscreen` / `--no-fullscreen` (scan) | Projector windows default true. These flags override scan acquisition; calibration collection reads the JSON setting. |
| `frame_timeout_seconds` | `--frame-timeout-seconds` (scan) | Finite positive timeout, default 30 s. For scans it is the allowance after dwell; for calibration it limits the interval without a camera frame. Collection uses JSON. |

Keep dimensions, orientation, focus and lens settings consistent between calibration and scans. The camera size is a request, not an intrinsic-matrix resize operation. Scan `acquisition.json` records requested and observed dimensions, settings, completed sequences and session status. Projection-window placement does not change the numerical projector raster.

### OAK startup controls

Replace the `acquisition` object with an explicit OAK setup, for example:

```json
{
  "camera": "oak",
  "device_id": "YOUR_OAK_MXID_OR_IP",
  "camera_size": [1920, 1080],
  "fps": 30,
  "flip": false,
  "frame_timeout_seconds": 30,
  "window_position": [2020, 100],
  "fullscreen": true,
  "oak_options": {
    "focus": 135,
    "exposure_us": 32500,
    "iso": 250,
    "sharpness": 4,
    "close_timeout_seconds": 2.0
  }
}
```

`device_id` and nonempty `oak_options` require `camera="oak"`. The adapter uses the optional DepthAI v2 dependency; neither this example nor the configuration checks operate or validate a physical device.

| `oak_options` field | Startup default | Package validation / override |
| --- | --- | --- |
| `focus` | `135` | Integer 0..255; `--focus`. |
| `exposure_us` | `32500` | Positive integer microseconds; `--exposure-us`. |
| `iso` | `250` | Positive integer; `--iso`. |
| `sharpness` | `4` | Integer 0..4; `--sharpness`. |
| `close_timeout_seconds` | `2.0` | Finite positive reader-shutdown timeout; JSON only. |
| `keyboard_controls` | See below | Optional object with `focus`, `exposure_us`, `iso`, `sharpness`; JSON only, separate retained adjustment state. |

These are software checks; supported sensor modes and control limits still depend on the device. The retained OAK pipeline selects the 4K sensor mode for requested width 3840, 1080p for width 1920, and 1080p with ISP scale 2/3 for other widths. It does not implement arbitrary resizing from the requested height. Check the actual frame dimensions and calibrate for those dimensions.

The historical keyboard state defaults to focus 135, exposure 20000 µs, ISO 800 and sharpness 0, independently of the startup sensor values above. Changing `keyboard_controls` changes that state, not the controls sent at startup. The current scan and calibration workflows do **not** forward keyboard events to `OakCamera.cam_params_control`; their `q`/`c` actions do not enable interactive focus/exposure adjustment.

## Physical calibration configuration

Use a separate file such as [configs/calibration.json](../configs/calibration.json). Its fields are top-level, rather than nested under `plan`. Camera/projector image dimensions and the physical board dimensions have different meanings.

| Calibration field | Default / requirement | Meaning |
| --- | --- | --- |
| `camera_size` | Required for collection/camera solving | Actual full-resolution camera image dimensions; calibration images are not resized before solving. |
| `projector_size` | Required for projector/grid commands | Native projector image dimensions; use the same grid when collecting and solving. |
| `board.squares` | `[5,7]` | Two integer ChArUco square counts, each at least 2. |
| `board.square_length`, `board.marker_length` | `.04`, `.02` | Measured physical lengths satisfying `0 < marker_length < square_length`, in `length_unit`. |
| `board.dictionary` | `"DICT_6X6_250"` | Named OpenCV `DICT_` constant. Numeric `board.dictionary_id` is an alternative; supply only one. |
| `board_image_size` | `[1000,1400]` | Printable raster dimensions for `board`; pixel count does not establish physical print scale. |
| `circle_grid.size` | `[4,11]` | Counts in the asymmetric projected circle grid. |
| `circle_grid.radius` | `15` | Positive integer circle radius in projector pixels. |
| `circle_grid.origin` | `[800,350]` | Integer projector-pixel placement. Grid extent and white padding of three radii must fit inside `projector_size`. |
| `image_pattern` | `"*.jpg"` | Saved-image glob used by the solvers; collection supports matching `.jpg` or `.png` files. |
| `minimum_views` | `3` if omitted; template uses `10` | Positive minimum accepted views. Count alone does not establish pose diversity or accuracy. |
| `length_unit` | Camera default `"m"`; projector inherits the camera unit if omitted | Unit label for the supplied board lengths and resulting translation. Values are not converted automatically. |
| `homography_reprojection_threshold` | `5.0` | Finite positive RANSAC tolerance in destination **board-length units**, not camera pixels. Convert it with the board coordinates when changing units. |
| `stereo_image_size` | `[2560,1440]` | Retained OpenCV stereo-solver argument, saved separately; it does not resize camera observations or specify the projector raster. |
| `camera_criteria`, `projector_criteria`, `stereo_criteria` | Omitted/null retains the respective solver's source behavior | Explicit termination objects, shown below. |

The template records these explicit solver settings; they are examples, not newly inferred defaults:

```json
{
  "camera_criteria": {"max_iterations": 500, "epsilon": 1e-10},
  "projector_criteria": {"max_iterations": 100, "epsilon": 1e-12},
  "stereo_criteria": {"max_iterations": 100, "epsilon": 1e-10}
}
```

Each supplied criterion requires a positive integer `max_iterations` and finite positive `epsilon`; both OpenCV termination conditions are enabled. Keep these choices fixed when comparing calibrations. Camera/projector focal guesses of 1000/3000 pixels and detector/refinement constants remain source-compatible, rather than JSON knobs; see the [retained-constants table](calibration.md#configurable-settings-and-retained-constants). `calibrate-projector --projector-calibration FILE` takes an existing REOSSI stereo-calibration NPZ, reuses its projector intrinsics and solves stereo only. Its projector RMS belongs to the earlier fit; see the [reuse example](calibration.md#calibrate-the-projector-and-stereo-geometry).

Calibration collection also reads these `acquisition` fields, in addition to camera selection, controls, flip and timeout:

| Collection field | Default | Scope |
| --- | --- | --- |
| `preview_size` | `[960,540]` | Camera preview only; saved frames retain their original dimensions. Scan capture has no camera-preview-size setting. |
| `extension` | Suffix of `image_pattern`, normally `.jpg` | `.jpg` or `.png`; must match the solver's image-pattern suffix. |
| `consecutive_valid_frames` | `6` | Projector collection only: nonnegative detection count for automatic saving. `0` selects manual saving with `c`; this is not a motion threshold. |
| `window_position`, `fullscreen` | No position override; fullscreen true | Projected grid window, independent of the camera preview. |

`--max-images` limits each collection command to a positive number of images, default 20. Camera collection saves a detected view with `c`; both collection commands exit with `q`. See [calibration](calibration.md) for physical board/grid placement, observation review, units and solver limitations.

## Digital scenes

`configs/synthetic.json` is a small raster plan. `synthetic_3d.json` supplies the scan plan for `scene_plane.json` or `scene_curved.json`; changing the camera size does not implicitly change the projector or emulated grid.

Scene JSON separates these fields:

| Object | Fields and meaning |
| --- | --- |
| `camera`, `projector` | `image_size`, a pinhole `matrix` or generated `focal_length`, world-space `center`, and either world-to-device `rotation` or a look-at `target`. Poses use `X_device = rotation @ (X_world - center)`. |
| `surface` | Positive `width`, `height`, `depth`; integer `columns`/`rows` at least 2; finite `tilt` and `curvature`. Generated geometry follows `z=depth+tilt*x+curvature*x²`. Lengths share world units; curvature has inverse-length units. |
| `sensor` | Finite nonnegative `black_level`, `direct_level`, `noise_std`, defaulting to 2/180/0 in 0..255 intensity units. Noise is added before clipping; these are not erasure probabilities. |
| Root | `seed` for reproducibility and `length_unit` for world/calibration coordinates. |

`--mesh` replaces the generated surface, `--seed`/`--noise-std` override scene values, and `--save-captures` preserves replay-compatible images. `simulate-scene --pattern-placement` selects `emulated_cells` (the source-compatible default) or `native_pixels` (one native pixel per position, with no intermediate target grid). This scene-rendering option is saved in the model metadata and does not change scan-plan JSON or physical acquisition. The raster command exposes camera size, intensity noise and seed as CLI flags. See [simulation](simulation.md) for image formation, supported mesh formats and model limits.

## Replay, reconstruction and resource controls

| Command option | Default / interpretation |
| --- | --- |
| `replay --layout` | `runs`: `run_R_displacement_D.npz`. `single` reads `displacement_D.npz` and requires `repeats=1`; missing runs are never manufactured. An available `white.png` is copied. |
| `--max-sequence-mib` | 1024 MiB for capture/replay/pattern export; 64 MiB for raster/scene simulation. This CLI limit bounds one stack, not total RAM or all decoder temporaries. |
| `simulate-scene --max-triangle-tests` | 20000000 camera/shadow ray-triangle tests; large meshes require an accelerated renderer. |
| `reconstruct --identifier-origin` | `legacy` passes source coordinates directly; `zero` selects zero-based input; `one` subtracts one from valid one-based IDs. Interpretation is recorded separately from the plan's origin. |
| `reconstruct --coordinate-frame` | `camera`; alternatives `projector` and `legacy`, with units from calibration. |
| `reconstruct --white-image` | Overrides the scan's `white.png` color reference explicitly. |
| `reconstruct --box-threshold` | Omitted means no box filter; a finite positive value retains points strictly inside that cube in the selected frame and calibration units. |
| `reconstruct --max-points` | 1000000 valid candidate correspondences before Python-list collection; positive integer CLI override. It is not a bound on total process memory. |

A 38-frame 3840×2160 float64 capture stack alone needs about 2405 MiB, exceeding the default 1024 MiB limit. Changing the OAK request to 4K therefore also requires an explicit sequence budget supported by available RAM; decoding and queues need additional memory. Resource flags belong to the command line, not the `acquisition` JSON object.

CLI output paths must be new. Keep completed recordings and their effective configuration when retrying an interrupted scan. Reconstruction requires maps and a color reference matching the calibrated camera dimensions; it does not infer an intrinsic resize from a new camera-size request.
