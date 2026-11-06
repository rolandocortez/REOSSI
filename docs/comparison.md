# Compare conventional, LD and REOSSI processing

This example follows three scans through the same image decoder and geometric
reconstruction. The virtual camera, projector, plane, sensor response and length
unit are shared. Calibration is known from the virtual device poses.

```bash
python examples/compare_modes.py --output outputs/mode-comparison --plot
```

Plotting requires the `plots` extra. Omit `--plot` to save numerical outputs and
PLY clouds with only the base dependencies. Each invocation requires a new output
directory.

For a small analogue retaining the thesis's factor of six and six positions,
use a 54×36 projector raster and 9×6 emulated patterns:

```bash
python examples/compare_modes.py --output outputs/mode-comparison-factor6 \
  --projector-size 54x36 --emulated-size 9x6 --depth 2.25 --plot
```

These dimensions keep the example small; they do not reproduce the physical
1920×1080 experiment. The chosen plane depth gives an integer three-pixel stereo
disparity with this virtual rig, separating decoding checks from correspondence
rounding error.

## Pattern placement

The example defaults to `native_pixels`: generate the unshifted low-resolution
patterns, enlarge them by nearest-neighbor sampling, then move column patterns
right and row patterns down by one native pixel per position. Black/white
references remain fixed. The zero-coded prefix does not wrap to the other edge.

Chapter 8 explicitly specifies `F_e=1920/320=6`, five additional acquisitions and
`step=W_HD=1` for its physical experiment. This corresponds to positions 0 through
5 in native projector pixels. A contradictory general expression for `step` does
not override those explicit experiment parameters. Intermediate target grids
still require separate spatial and identifier-unit definitions.

The source-compatible profile remains available:

```bash
python examples/compare_modes.py --output outputs/source-placement-comparison \
  --pattern-placement emulated_cells --plot
```

The main scanner and standalone scene command retain `emulated_cells` as their
default. This walkthrough adds an explicit simulation choice; it does not change
the physical capture profile.

## Reading the results

Each mode saves its decoded maps, independent geometric truth, known calibration
and reconstructed cloud. `comparison.json` records the selected placement,
acquisition cost, identifier metrics and camera-associated 3D errors. Conventional
identifiers start at zero; LD/REOSSI identifiers start at one. Reconstruction uses
that origin explicitly. Comparisons use common native projector coordinates.

Frame counts differ: low-resolution modes use fewer bits per sequence, while
REOSSI acquires additional displaced sequences. These counts are reported rather
than treated as a fixed acquisition budget. Three-dimensional error is evaluated
against the truth belonging to each retained camera pixel, with illuminated and
finite masks applied explicitly.

The initial scene has no sensor noise, projector blur or indirect illumination.
An ideal conventional scan can therefore be exact. Native-pixel REOSSI can recover
the fine coordinates lost by a single LD scan; this does not show superiority over
an ideal conventional projector. The preserved whole-cell placement can yield
identical LD and REOSSI maps. Both outcomes are retained rather than hidden.

The HY300's physical optical limitations and the thesis point-cloud improvement
need their own observations. This example demonstrates the shared software
pipeline and displacement model, not a reproduction of that physical experiment.
