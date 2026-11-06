# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Rolando Cortez Garcia
"""Compare three scanner workflows on one small, noiseless virtual plane.

The native-pixel placement follows the chapter 8 experiment with no intermediate
target grid. The preserved emulated-cell placement is an explicit alternative.
This direct-only scene does not model the optical limitations of a real projector.
"""

import argparse
import json
from dataclasses import asdict, replace
from math import isfinite
from pathlib import Path

import numpy as np

from reossi import ScanPlan
from reossi.cli.common import image_size, require_new_output
from reossi.pipeline import pattern_raster_size
from reossi.reconstruction.workflow import ReconstructionResult, reconstruct_scan
from reossi.simulation.geometry import make_device, make_surface
from reossi.simulation.scene import (
    SceneScanResult,
    save_scene_result,
    simulate_scene_scan,
)
from reossi.simulation.sensor import SensorModel
from reossi.visualization import require_plotting

MODES = ("conventional", "ld", "reossi")


def _geometry_metrics(scan: SceneScanResult, cloud: ReconstructionResult) -> dict:
    """Measure camera-aligned 3D errors on finite, truly illuminated samples."""
    pixels = cloud.camera_pixels.astype(np.intp)
    truth = scan.geometry.points_camera[pixels[:, 1], pixels[:, 0]]
    evaluated = scan.geometry.illuminated[pixels[:, 1], pixels[:, 0]] & np.isfinite(
        truth
    ).all(axis=1)
    errors = np.linalg.norm(cloud.points.T[evaluated] - truth[evaluated], axis=1)
    illuminated_count = int(scan.geometry.illuminated.sum())
    return {
        "coordinate_frame": "camera",
        "length_unit": scan.calibration.length_unit,
        "retained_points": int(cloud.points.shape[1]),
        "evaluated_illuminated_points": int(evaluated.sum()),
        "retained_outside_illuminated_points": int((~evaluated).sum()),
        "illuminated_surface_coverage": (
            float(evaluated.sum() / illuminated_count) if illuminated_count else None
        ),
        "mean_euclidean_error": float(errors.mean()) if errors.size else None,
        "root_mean_square_euclidean_error": (
            float(np.sqrt(np.mean(errors**2))) if errors.size else None
        ),
        "maximum_euclidean_error": float(errors.max()) if errors.size else None,
        "nonfinite_points_removed": cloud.metadata["nonfinite_points_removed"],
        "denominator": "finite ground truth at truly illuminated camera pixels",
    }


def _range_metrics(scan: SceneScanResult, plan: ScanPlan) -> dict:
    """Expose valid-looking IDs beyond the raster before triangulation clips them."""
    origin = plan.identifier_origin
    valid = (scan.maps.columns >= origin) & (scan.maps.rows >= origin)
    outside = valid & (
        (scan.maps.columns >= plan.projector_size[0] + origin)
        | (scan.maps.rows >= plan.projector_size[1] + origin)
    )
    return {
        "out_of_range_valid_pixels": int(outside.sum()),
        "out_of_range_illuminated_pixels": int(
            (outside & scan.geometry.illuminated).sum()
        ),
    }


def _plot_comparison(
    results: dict[str, tuple[SceneScanResult, ReconstructionResult]],
    plans: dict[str, ScanPlan],
    path: Path,
    pattern_placement: str,
) -> None:
    """Save shared-scale identifiers, identifier errors and unfiltered 3D clouds."""
    plt = require_plotting()
    figures = []
    all_points = []
    maximum_error = 0.0
    for mode in MODES:
        scan, cloud = results[mode]
        valid = (
            (scan.maps.columns >= plans[mode].identifier_origin)
            & (scan.maps.rows >= plans[mode].identifier_origin)
            & scan.geometry.illuminated
        )
        columns = scan.maps.columns.astype(float) - plans[mode].identifier_origin
        errors = np.abs(scan.maps.columns - scan.true_columns).astype(float)
        maximum_error = max(
            maximum_error, float(errors[valid].max()) if valid.any() else 0
        )
        figures.append(
            (np.ma.array(columns, mask=~valid), np.ma.array(errors, mask=~valid))
        )
        if cloud.points.shape[1]:
            all_points.append(cloud.points)
    finite_truth = results[MODES[0]][0].geometry.points_camera.reshape(-1, 3)
    finite_truth = finite_truth[np.isfinite(finite_truth).all(axis=1)]
    if finite_truth.size:
        all_points.append(finite_truth.T)
    combined = np.concatenate(all_points, axis=1) if all_points else np.zeros((3, 1))
    lower, upper = combined.min(axis=1), combined.max(axis=1)
    margin = np.maximum(upper - lower, 1e-3) * 0.03
    figure = plt.figure(figsize=(15, 11), layout="constrained")
    color_map = plt.get_cmap("viridis").copy()
    color_map.set_bad("#dedede")
    error_map = plt.get_cmap("magma").copy()
    error_map.set_bad("#dedede")
    try:
        for column, mode in enumerate(MODES):
            scan, cloud = results[mode]
            plan = plans[mode]
            ids_axes = figure.add_subplot(3, 3, column + 1)
            ids = ids_axes.imshow(
                figures[column][0],
                cmap=color_map,
                vmin=0,
                vmax=plan.projector_size[0] - 1,
            )
            frames = plan.sequence_length * plan.repeats * plan.displacement_count
            ids_axes.set_title(f"{mode.upper()} · {frames} projected frames")
            ids_axes.set_xlabel("Camera column (px)")
            ids_axes.set_ylabel("Camera row (px)")
            figure.colorbar(ids, ax=ids_axes, label="Projector column (zero-based px)")
            error_axes = figure.add_subplot(3, 3, column + 4)
            error = error_axes.imshow(
                figures[column][1], cmap=error_map, vmin=0, vmax=max(maximum_error, 1.0)
            )
            error_axes.set_title("Absolute column-identifier error")
            error_axes.set_xlabel("Camera column (px)")
            error_axes.set_ylabel("Camera row (px)")
            figure.colorbar(error, ax=error_axes, label="Native projector pixels")
            cloud_axes = figure.add_subplot(3, 3, column + 7, projection="3d")
            cloud_axes.scatter(*cloud.points, color="#444444", s=3)
            cloud_axes.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
            cloud_axes.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
            cloud_axes.set_zlim(lower[2] - margin[2], upper[2] + margin[2])
            cloud_axes.set_box_aspect(np.maximum(upper - lower, 0.05))
            cloud_axes.view_init(elev=24, azim=-65)
            unit = scan.calibration.length_unit
            cloud_axes.set(
                xlabel=f"X ({unit})", ylabel=f"Y ({unit})", zlabel=f"Z ({unit})"
            )
            cloud_axes.set_title(f"Camera-frame cloud · {cloud.points.shape[1]} points")
        figure.suptitle(
            f"Ideal-scene workflow comparison · {pattern_placement}\n"
            "Direct illumination; no projector blur or global light transport"
        )
        figure.savefig(path, dpi=160)
    finally:
        plt.close(figure)


def run_comparison(
    output: str | Path,
    *,
    projector_size: tuple[int, int] = (48, 32),
    camera_size: tuple[int, int] | None = None,
    emulated_size: tuple[int, int] = (12, 8),
    repeats: int = 1,
    pattern_placement: str = "native_pixels",
    baseline: float = 0.125,
    depth: float = 2.0,
    max_sequence_mib: float = 8,
    max_triangle_tests: int = 1_000_000,
    plot: bool = False,
) -> dict:
    """Run all modes on the same plane and save their actual maps and clouds.

    Each mode retains its own source-compatible classifier and identifier origin.
    The capture counts differ. This is a pipeline walkthrough, not a fixed-cost
    experiment or a reproduction of the thesis's physical improvement.
    """
    output = require_new_output(output)
    if not all(isfinite(value) and value > 0 for value in (baseline, depth)):
        raise ValueError("baseline and depth must be finite and positive")
    base = ScanPlan(
        projector_size=projector_size, emulated_size=emulated_size, repeats=repeats
    )
    camera = make_device(camera_size or base.projector_size)
    projector = make_device(base.projector_size, center=(baseline, 0, 0))
    mesh = make_surface(width=4, height=3, depth=depth)
    sensor = SensorModel(noise_std=0)
    plans = {mode: replace(base, mode=mode) for mode in MODES}
    for plan in plans.values():
        pattern_raster_size(plan, pattern_placement=pattern_placement)
    if plot:
        require_plotting()
    results = {}
    comparison = {
        "kind": "ideal_scene_workflow_comparison",
        "pattern_placement": pattern_placement,
        "length_unit": "m",
        "sensor": asdict(sensor),
        "seed": 0,
        "baseline": baseline,
        "plane_depth": depth,
        "camera_size": list(camera.image_size),
        "projector_size": list(projector.image_size),
        "emulated_size": list(base.emulated_size),
        "max_sequence_mib": max_sequence_mib,
        "max_triangle_tests": max_triangle_tests,
        "limitations": [
            "Ideal direct illumination does not model the HY300 optical limitations.",
            "Profiles use different capture counts and source-compatible classifier settings.",
            "Intermediate target grids are outside this example.",
        ],
        "modes": {},
    }
    output.mkdir(parents=True, exist_ok=False)
    for mode, plan in plans.items():
        scan = simulate_scene_scan(
            mesh,
            camera,
            projector,
            plan,
            sensor=sensor,
            seed=0,
            max_sequence_mib=max_sequence_mib,
            max_triangle_tests=max_triangle_tests,
            pattern_placement=pattern_placement,
        )
        scan_folder = output / mode / "scan"
        save_scene_result(scan_folder, scan, plan)
        cloud = reconstruct_scan(
            scan_folder,
            scan.calibration,
            output / mode / "cloud",
            identifier_origin=plan.identifier_origin,
            coordinate_frame="camera",
            max_points=camera.image_size[0] * camera.image_size[1],
        )
        comparison["modes"][mode] = {
            "plan": plan.to_dict(),
            "model": scan.model,
            "projected_frames": plan.sequence_length
            * plan.repeats
            * plan.displacement_count,
            "identifier_metrics": scan.metrics | _range_metrics(scan, plan),
            "geometry_metrics": _geometry_metrics(scan, cloud),
        }
        results[mode] = scan, cloud
    if plot:
        _plot_comparison(results, plans, output / "comparison.png", pattern_placement)
    (output / "comparison.json").write_text(
        json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return comparison


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projector-size", type=image_size, default=(48, 32))
    parser.add_argument("--camera-size", type=image_size)
    parser.add_argument("--emulated-size", type=image_size, default=(12, 8))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--pattern-placement",
        choices=("emulated_cells", "native_pixels"),
        default="native_pixels",
    )
    parser.add_argument(
        "--baseline", type=float, default=0.125, help="Projector baseline in metres"
    )
    parser.add_argument(
        "--depth", type=float, default=2, help="Known plane depth in metres"
    )
    parser.add_argument("--max-sequence-mib", type=float, default=8)
    parser.add_argument("--max-triangle-tests", type=int, default=1_000_000)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    result = run_comparison(**vars(args))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
