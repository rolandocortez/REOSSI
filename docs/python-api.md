# Python API and extension boundaries

The CLI composes ordinary Python functions. Importing them does not open a camera, start an experiment or create a window. Use the same processing core for stored recordings, physical acquisition and virtual captures.

## Replay ordered captures

```python
from reossi import ScanPlan
from reossi.acquisition.recording import (
    iter_recorded_sequences,
    save_identifier_maps,
)
from reossi.pipeline import reconstruct_identifier_maps

plan = ScanPlan(
    projector_size=(1920, 1080),
    emulated_size=(320, 180),
    repeats=1,
)
captures = iter_recorded_sequences(
    "recordings/object",
    plan,
    layout="single",
    max_sequence_mib=512,
)
maps = reconstruct_identifier_maps(captures, plan)
save_identifier_maps("outputs/replayed", maps, plan)
```

The outer iterable yields positions in displacement order, including the initial position. Each inner iterable yields exactly `plan.repeats` actual capture stacks. A stack has shape `(frames,height,width)`, with `frames == plan.sequence_length`. Missing or extra positions/runs are errors. The iterator loads one stack at a time; this does not bound all decoder intermediates or accumulated maps.

`IdentifierMaps.columns` and `.rows` are indexed by camera pixel `[row,column]`. Repeated classifications are combined as bits before integer decoding. Scaling and displacement fusion occur afterward. Input dtype is retained; an explicit conversion changes the numerical experiment.

The low-level save helper writes its requested destination; use a new path when retaining experiments. CLI commands additionally reject existing output paths.

## Generate a small virtual scan

```python
from reossi import ScanPlan
from reossi.simulation.geometry import make_device, make_surface
from reossi.simulation.scene import save_scene_result, simulate_scene_scan

virtual_plan = ScanPlan(
    mode="conventional",
    projector_size=(31, 23),
    emulated_size=(31, 23),
    repeats=1,
)
mesh = make_surface(width=4, height=3, depth=2)
camera = make_device((16, 12))
projector = make_device(virtual_plan.projector_size, center=(0.15, 0, 0))
result = simulate_scene_scan(mesh, camera, projector, virtual_plan)
save_scene_result("outputs/virtual-api", result, virtual_plan)
```

World coordinates, mesh dimensions and device translations share a length unit. Known calibration is derived from the device poses. `result.maps` contains actual decoded identifiers; geometric truth and visibility masks are independent fields. The generated capture model is direct-only and does not establish photometric realism. REOSSI plans retain the recovered cell-displacement convention described in [architecture](architecture.md).

## Replace only the illumination estimator

`decode_displacement` and `reconstruct_identifier_maps` accept an optional callable with this contract:

```text
illumination_estimator(images) -> (direct, global_component)
images:          (frames,height,width)
each component:  (height,width)
```

For example, explicitly supplying the configured baseline keeps the default computation:

```python
from functools import partial

from reossi.structured_light.illumination import estimate_illumination

captures = iter_recorded_sequences(
    "recordings/object", plan, layout="single", max_sequence_mib=512
)
estimator = partial(
    estimate_illumination,
    normalization_epsilon=plan.normalization_epsilon,
)
maps = reconstruct_identifier_maps(
    captures,
    plan,
    illumination_estimator=estimator,
)
```

Create a fresh capture iterator before another processing pass. An estimator owns its parameters and any preparation of multiple acquisitions. RGB/BGR components require an explicit caller-selected conversion to the grayscale contract; their channel axis is not accepted as a camera dimension. The core imports no alternate implementation and discovers no plugins automatically.

## Enter directly with identifier maps

`reossi.reconstruction.identifiers.scale_identifiers` applies the recovered transform **in place** and returns the same array. Pass a copy when retaining the original. Its default `invalid_values=(-1,)` preserves exactly that sentinel; `()` transforms all values and `(-1,0)` preserves both. An experiment selecting a different policy must propagate validity accordingly, rather than assuming the invalid marker stays numerically unchanged.

`fuse_identifier_maps(column_layers, row_layers)` takes aligned two-dimensional layers and returns the source maximum fusion. The direct `simulate_identifier_map` demonstration in `reossi.simulation.identifiers` exposes base, shifted, transformed and fused layers without rendering images. This entry point supports method-level checks; it does not replace image decoding or metric 3D validation.

## Calibration and reconstruction

Use `calibrate_camera_images` and `calibrate_projector_images` from `reossi.calibration.workflow` for image-based calibration. Their typed result objects save self-describing NPZ files and diagnostics. Observation collection lives separately in `reossi.calibration.observations` and operates an explicitly supplied camera. See [calibration](calibration.md) for board units, solver criteria and homography thresholds.

`reossi.reconstruction.workflow.reconstruct_scan` consumes saved identifier maps and calibration, retains pixel/color associations, and exports points in an explicit coordinate frame. Low-level triangulation preserves the source mathematics and its mutation of a supplied projector matrix; the workflow copies calibration arrays before constructing it.
