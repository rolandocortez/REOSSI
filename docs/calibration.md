# Camera, projector and stereo calibration

Calibration connects the physical scanner to the same reconstruction workflow used by virtual captures. The camera and projector need intrinsics and distortion coefficients; stereo calibration adds the transformation between their coordinate systems. Camera collection and projector collection are separate operations.

Camera intrinsics, projector intrinsics and stereo extrinsics remain available as distinct stages. Charron's original [camera](https://github.com/guillaume-charron/3DScanner-GrayCode/blob/6f8abf0257c727eed81cc249edb85d43010c7b3d/src/1-calibrate_camera.py) and [projector/stereo](https://github.com/guillaume-charron/3DScanner-GrayCode/blob/6f8abf0257c727eed81cc249edb85d43010c7b3d/src/2-calibrate_projcam.py) scripts combine capture (`c`) and solving (`k`) in one window. REOSSI separates collection and solving into CLI commands so saved observations can be reviewed and reused.

## Configure and measure the board

Edit `configs/calibration.json` before collecting images. The example describes the upstream 5-by-7 ChArUco board, square length 0.04, marker length 0.02, dictionary `DICT_6X6_250`, with lengths expressed in metres. It is a template rather than a measurement of an existing printed board.

Generate its printable image:

```bash
reossi board --config configs/calibration.json --output outputs/board.png
```

Print it on a flat, rigid surface and measure the actual square and marker lengths. Printer scaling must agree with the configuration. Image pixel dimensions do not establish the physical print scale. The older local camera scripts used inconsistent board lengths between detection and solving; existing calibration files cannot be validated from their directory names. Use one measured board definition for each calibration workflow.

Set `camera_size` to the actual saved frame dimensions. Keep camera orientation, focus, lens settings and image flipping consistent between calibration and scans. Frames are never resized before solving. Preview resizing affects only the window.

## Calibrate the camera

```bash
reossi collect-camera-calibration --config configs/calibration.json \
  --output data/camera-observations --max-images 20

reossi calibrate-camera --config configs/calibration.json \
  --input data/camera-observations --output outputs/camera.npz
```

During collection, `c` saves an unannotated frame when the board is detected and `q` exits. Cover different image regions, distances and board tilts. Multiple views of an unchanged pose do not provide the same calibration information as varied views. The minimum-view setting is an input guard, not a guarantee of sufficient geometric diversity.

Existing images can be supplied directly to `calibrate-camera`; collection is optional. Set `image_pattern` to match their extension. Unreadable or incorrectly sized images fail clearly. Images without usable board detections are recorded as rejected.

`camera.npz` contains the camera matrix, distortion, image size, physical board, length unit, overall RMS and per-view RMS reprojection errors. Inspect the accepted/rejected view list and per-view errors; a low average error alone does not establish physical accuracy.

## Calibrate the projector and stereo geometry

Keep the camera and projector fixed relative to one another while moving a rigid, flat calibration panel between views. Vary its position, distance and orientation to obtain different perspectives. The printed ChArUco and projected circle grid must lie on the same physical plane: the homography inferred from the printed markers is used to locate the projected circles on that plane. Reserve a blank area beside the print, as in the left-print/right-projection arrangement shown in the [upstream calibration instructions](https://github.com/guillaume-charron/3DScanner-GrayCode/blob/6f8abf0257c727eed81cc249edb85d43010c7b3d/README.md#application-usage).

The circle pattern stays fixed in projector pixel coordinates while the panel moves. Its intersection with the panel changes with perspective; it is not a printed texture moving with the ChArUco. Use the same projector grid coordinates when collecting and solving. The example configuration retains the source 4-by-11 grid, radius 15 pixels and minimum of 10 usable views. Distinct perspectives matter in addition to view count.

```bash
reossi circle-grid --config configs/calibration.json --output outputs/circle-grid.png

reossi collect-projector-calibration --config configs/calibration.json \
  --camera-calibration outputs/camera.npz \
  --output data/projector-observations --max-images 20

reossi calibrate-projector --config configs/calibration.json \
  --camera-calibration outputs/camera.npz \
  --input data/projector-observations --output outputs/stereo.npz
```

Collection projects the configured grid, detects the board and projected circles, and saves their original camera frames. `c` requests a detected view; `q` exits. The optional `consecutive_valid_frames` setting saves after that many successive detections; six preserves the original acquisition condition. It is detection continuity, not a measured motion criterion. Set it to 0 for manual acquisition, move the board between captures, and review redundant images before solving.

The grid origin, radius and white padding must fit inside `projector_size`; invalid placement is rejected. Use the same grid when collecting and solving. The saved grid image retains the source's four-channel raster; the acquisition window displays that raster directly.

The workflow first solves projector intrinsics, then fixes the camera/projector intrinsics to solve stereo rotation and translation. Supply `--projector-calibration` to reuse an existing projector calibration and solve only the stereo relation; the Python API supports the same choice. This preserves the source script's `calibrate_proj=False` path. Reused intrinsics must match the device resolution and optical settings. The resulting NPZ records both image sizes, board, units, solver settings, accepted/rejected images, projector RMS and stereo RMS.

The reuse input is a REOSSI stereo-calibration NPZ produced by `calibrate-projector`, for example:

```bash
reossi calibrate-projector --config configs/calibration.json \
  --camera-calibration outputs/camera.npz \
  --projector-calibration outputs/stereo.npz \
  --input data/new-projector-observations --output outputs/new-stereo.npz
```

The original separate `proj_mtx.npy` and `proj_dist.npy` files are not direct CLI inputs. Reuse reads the projector intrinsics and their recorded size from the NPZ; it estimates a new stereo transformation rather than reusing the old rotation/translation. The reported `projector_rms` is retained from the earlier intrinsic fit in this mode. Only `stereo_rms` describes the newly solved relation; the current diagnostics have no new projector-fit RMS.

Camera intrinsics can be reused with a different explicitly configured projector-calibration board. If the projector board is omitted, the Python API inherits the camera file's board definition. An explicit length unit applies to the current projector board and resulting stereo translation; board dimensions are not converted automatically, and an old translation is not reused. Tests cover an explicit different board in both metres and millimetres.

`stereo_image_size` is explicit. The default 2560-by-1440 value preserves the source solver argument and can differ from the camera image size. Choose and record the appropriate argument for a new setup rather than assuming the template describes the local camera. It is stored separately from camera and projector dimensions.

## Homography threshold and physical units

`homography_reprojection_threshold` controls RANSAC when mapping detected camera corners to the board plane. The destination points are board coordinates, so the threshold uses the board's configured length unit. It is not a camera-pixel threshold. The setting is applied during observation collection and projector calibration, validated before opening collection hardware, and saved with its coordinate space and unit in stereo diagnostics.

The API and example JSON retain the source numeric default `5.0`. This preserves the source behavior; it does not establish a suitable threshold for a measured board. If a physical threshold is expressed in different units, convert it together with the board coordinates: a threshold of 0.0005 m corresponds to 0.5 mm. An autonomous controlled test obtains matching inliers and predictions for those equivalent settings. Keeping the numeric value 5.0 in both m and mm instead changes the physical tolerance and selects different inliers in that test. These test values are not optimized defaults for a real rig.

Review this parameter against measured board dimensions and detection errors before collecting a physical dataset. The code does not infer a new tolerance or change the homography direction or RANSAC method.

## Configurable settings and retained constants

| Setting | Current source-compatible choice | Configuration boundary |
| --- | --- | --- |
| Initial camera focal lengths | 1000 px in both axes when no complete initial calibration is supplied | Retained solver initialization; a low-level supplied camera matrix/distortion can replace it. There is no JSON initial-focal field. |
| Initial projector focal lengths | 3000 px in both axes when no matrix is supplied | Retained solver initialization. Existing projector intrinsics can be reused explicitly instead of solving them again. |
| Corner refinement search | `cornerSubPix(winSize=(3,3))`, meaning an effective 7×7 search window | Retained detector constant, not a configurable full 3×3 window. |
| Corner refinement termination | 100 iterations and epsilon 1e−5 | Retained detector settings, separate from calibration solver criteria. |
| Homography inlier tolerance | `homography_reprojection_threshold=5.0` in board/destination coordinates | Configurable constructor, workflow and JSON value; saved with effective coordinate space and length unit. |
| Calibration solver termination | Camera/projector/stereo criteria retain source behavior when omitted | `camera_criteria`, `projector_criteria` and `stereo_criteria` explicitly configure iteration count/epsilon and are saved. |
| Stereo solver image size | 2560×1440 source default | `stereo_image_size` is configurable and recorded independently of actual camera/projector image sizes. |

## Solver termination is part of the experiment

The Python API leaves source defaults unchanged when termination criteria are unspecified. Explicit camera, projector and stereo criteria are recorded in diagnostics. CLI configuration can specify iteration limits and epsilon without editing solver code.

The autonomous known-geometry projector test demonstrates why this matters: the inherited 3000-pixel focal guess with OpenCV's default termination stops at RMS about 0.384 pixels and substantially incorrect focal lengths for that particular small set of views. With the same observations and an explicit 100-iteration/EPS 1e-12 criterion, it recovers the 720/700-pixel focal lengths with RMS about 0.000003 pixels. The default case and explicit recovery are both tested. These synthetic observations do not establish performance on a physical board.

Do not silently change solver criteria while comparing runs. Board coverage, pose diversity, image quality, initialization and termination all affect calibration.

## Reconstruct a saved scan

```bash
reossi reconstruct --input recordings/object --calibration outputs/stereo.npz \
  --output outputs/object-cloud
```

The input contains `scan.json`, `h_pixels.npy`, `v_pixels.npy` and a three-channel `white.png` used for colors. `--white-image` selects another reference image explicitly. Map and image dimensions must match calibrated camera dimensions; the workflow does not infer camera-intrinsic resizing. Projector intrinsics are scaled from their recorded calibration raster to the scan's native projector raster using the reference geometry.

Coordinate-frame choices are explicit:

- `--coordinate-frame camera`: camera origin and camera axes, the default.
- `--coordinate-frame projector`: projector origin and projector axes, including translation.
- `--coordinate-frame legacy`: camera origin with axes rotated into projector orientation, matching the raw historical triangulation output.

Stereo extrinsics follow `projector_point = rotation @ camera_point + translation`. Point units follow the physical board's configured length unit. `--box-threshold` retains points strictly inside an axis-aligned cube in the selected output frame.

`--identifier-origin legacy` passes decoded coordinates directly, preserving the source behavior. `zero` explicitly selects zero-based coordinates; `one` subtracts one from positive one-based identifiers before triangulation. The plan's origin and selected interpretation are both saved; no conversion is silently inferred. This distinction matters for the local one-based REOSSI generator.

Outputs include ASCII `cloud.ply`, points shaped `(3,N)`, RGB colors `(N,3)` in 0..1, aligned camera/projector pixel coordinates, a retained camera-pixel mask, and `reconstruction.json`. Nonfinite points and optional box filtering remove the corresponding entries from every aligned array. An empty result produces a valid zero-vertex PLY and empty aligned arrays.

The reconstruction workflow defaults to a limit of 1,000,000 valid candidate points before allocating the source collector's Python lists. `--max-points` sets an explicit alternative; the Python API accepts `None` to disable the guard. Invalid identifier pixels do not count toward this limit. It bounds candidate count rather than total RAM, and large image maps still require memory to load and validate.

## Validation status

Tests exercise real OpenCV solvers on small analytically generated observations, real board-image detection, mocked observation windows, known 3D geometry, frame transforms and export alignment. They operate no camera or projector. Physical board acquisition, synchronization and scanner accuracy require validation on the intended rig.

Separate offline image diagnostics used 800-by-600 geometric rasters, a reduced 4-by-5 circle grid and twelve prescribed poses with modest Rodrigues-vector components up to 0.20 radians. A camera-coverage follow-up used components up to 0.26 radians while reusing the original projector images. These reduced geometric diagnostics exercised actual detections and solvers but failed their predetermined known-parameter accuracy gates. They did not faithfully reproduce Charron's physical capture protocol, full grid, image formation or perspective diversity. Their failures do not establish that the physical calibration method fails.
