# Physical capture validation

This workflow checks the maintained scanner on a real camera/projector rig. The original Charron workflow and the thesis prototype already operated physically; the pending validation concerns this refactored implementation and its explicit pattern-placement options. Current automated tests use analytical images and mocked acquisition. They do not operate the HY300 or OAK-1.

## Prepare the rig and calibration

Keep the camera and projector rigidly fixed. Use the projector's actual input raster and check display scaling, fullscreen placement and any keystone correction before interpreting a software shift as a projected pixel. The HY300 input raster and the physical panel resolution need not be identical. The configured `projector_size` describes the raster used for calibration and projection.

Use a measured ChArUco board and varied panel orientations, with printed markers and projected circles on the same plane. Follow the [camera and projector calibration workflow](calibration.md). Reuse an existing REOSSI calibration NPZ only when image dimensions, orientation, focus and rig geometry match. The original separate NPY calibration files are not direct CLI inputs.

Adapt `configs/scan_real.json`: select the actual camera, requested dimensions, OAK device identifier, exposure/focus, frame orientation, and projector desktop position. A request does not guarantee the observed sensor dimensions. Keep settings fixed between calibration and capture, and avoid clipped reference frames.

## Inspect the Chapter 8 placement before capture

These commands open no camera or projection window:

```bash
reossi plan --config configs/scan_real.json --pattern-placement native_pixels

reossi patterns --config configs/scan_real.json \
  --pattern-placement native_pixels --output outputs/native-patterns
```

For a 1920×1080 raster with a 320×180 emulated grid, check factor 6, six positions (0 through 5), 38 frames per sequence and an effective pattern raster of 1920×1080. Position `d` translates column/row bit frames right/down by `d` raster pixels after enlargement; black/white references stay fixed and the new prefix encodes identifier zero. Two repetitions require 456 captured frames. This covers the thesis's native-target case, `step=W_HD=1`; an intermediate target is rejected by this profile.

The default `emulated_cells` profile retains the recovered source's whole-cell displacement. Choose placement explicitly for a comparison. `--pattern-placement` overrides `acquisition.pattern_placement`; omitting both retains that default. Exported `plan.json` retains the selected placement and can be reused as the configuration.

## Record, replay and reconstruct

The following command operates hardware when invoked. Replace the device identifier and adjust the configuration to the rig first:

```bash
reossi capture --config configs/scan_real.json --camera oak \
  --camera-size 1920x1080 --device-id YOUR_OAK_MXID \
  --pattern-placement native_pixels --output recordings/native-object

reossi replay --config recordings/native-object/scan.json \
  --input recordings/native-object --layout runs \
  --output outputs/native-replay

reossi reconstruct --input outputs/native-replay \
  --calibration outputs/stereo.npz --identifier-origin one \
  --output outputs/native-cloud
```

Capture saves float64 grayscale stacks as `run_R_displacement_D.npz`, one BGR `white.png`, the numerical plan, identifier maps and `acquisition.json`. The latter records placement, raster dimensions, requested/observed camera dimensions, completed sequences and session status. It is not a complete color recording. Keep this metadata with the recording. The reusable `scan.json` also retains placement and raster dimensions; replay carries that placement forward as metadata while decoding the saved arrays without reprojecting patterns.

Verify that the session is complete, every requested run and displacement exists, and replay matches the captured identifier maps. Inspect actual pattern/frame association and dwell on the rig. Use `--identifier-origin one` for LD/REOSSI reconstruction and `zero` for conventional mode; camera dimensions and calibration units must match the saved data.

The sequence limit bounds one array, not process RAM. At 1920×1080, a 38-frame float64 camera stack alone is about 601 MiB and its uint8 native pattern stack about 75 MiB. At 3840×2160 the camera stack alone is about 2405 MiB, above the default 1024 MiB limit. Account for decoding temporaries and queues before selecting a larger budget.

## Assess a comparison

Capture conventional, initial LD and REOSSI scans of the same static scene with the same rig/calibration and documented optical settings. Retain acquisition counts, reference intensities, valid coverage, and point-cloud views with consistent limits. Use known geometry or an independent reference to assess metric error. A denser cloud or a more complete image does not alone establish geometric accuracy.

The [digital comparison](comparison.md) checks shared software and coordinate recovery under an ideal direct-light model. It does not model HY300 blur or establish physical superiority over conventional structured light. Record physical results separately from those software checks.
