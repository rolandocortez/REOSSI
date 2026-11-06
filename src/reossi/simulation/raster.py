"""Small digital captures for exercising the actual Gray-code scan pipeline.

This forward model samples a nearest-neighbor projector raster with a camera
covering the same rectangular field. A scalar black level, direct intensity
and optional Gaussian sensor noise produce float64 frames, as in live capture.
It does not model scattering, defocus, occlusion, or projector/camera calibration.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from pathlib import Path

import numpy as np

from ..acquisition.recording import save_identifier_maps
from ..pipeline import (
    IdentifierMaps,
    generate_pattern_sequence,
    reconstruct_identifier_maps,
)
from ..plan import ScanPlan
from .sensor import SensorModel


@dataclass
class SyntheticResult:
    """Decoded maps, expected native identifiers, and error metrics.

    Expected identifiers include the configured zero/one-based origin.
    """

    maps: IdentifierMaps
    true_columns: np.ndarray
    true_rows: np.ndarray
    metrics: dict
    model: dict


def sample_pattern_sequence(
    patterns: np.ndarray,
    projector_columns: np.ndarray,
    projector_rows: np.ndarray,
    projector_size: tuple[int, int],
    *,
    black_level: float = 2.0,
    direct_level: float = 180.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample ``(N, h, w)`` patterns at known zero-based native coordinates.

    Coordinates have camera shape ``(H, W)``. Native projector pixels map to
    pattern cells by integer floor, equivalent to explicit nearest-neighbor
    enlargement. Values are clipped to 0..255 after adding sensor noise.
    No REOSSI displacement correction is inserted into this forward model.
    """
    SensorModel(black_level, direct_level, noise_std)
    if projector_columns.shape != projector_rows.shape or projector_columns.ndim != 2:
        raise ValueError("Projector coordinate maps must have matching (H,W) shapes")
    width, height = projector_size
    if (
        not np.isfinite(projector_columns).all()
        or not np.isfinite(projector_rows).all()
        or np.any(projector_columns < 0)
        or np.any(projector_columns >= width)
        or np.any(projector_rows < 0)
        or np.any(projector_rows >= height)
    ):
        raise ValueError(
            "Projector coordinates must be finite and inside the native raster"
        )
    columns = np.floor(projector_columns * patterns.shape[2] / width).astype(int)
    rows = np.floor(projector_rows * patterns.shape[1] / height).astype(int)
    images = black_level + direct_level * patterns[:, rows, columns].astype(float) / 255
    if noise_std:
        rng = rng if rng is not None else np.random.default_rng()
        images += rng.normal(0.0, noise_std, images.shape)
    return np.clip(images, 0.0, 255.0)


def _map_metrics(
    maps: IdentifierMaps,
    columns: np.ndarray,
    rows: np.ndarray,
    *,
    identifier_origin: int = 0,
) -> dict:
    valid = (maps.columns >= identifier_origin) & (maps.rows >= identifier_origin)
    metrics = {"pixels": int(valid.size), "valid_fraction": float(valid.mean())}
    for name, actual, expected in (
        ("columns", maps.columns, columns),
        ("rows", maps.rows, rows),
    ):
        difference = np.abs(actual[valid] - expected[valid])
        metrics[f"{name}_mean_absolute_error_native_pixels"] = (
            float(difference.mean()) if difference.size else None
        )
        metrics[f"{name}_exact_fraction_of_valid"] = (
            float((difference == 0).mean()) if difference.size else None
        )
    return metrics


def simulate_scan(
    plan: ScanPlan,
    *,
    camera_size: tuple[int, int] = (48, 32),
    black_level: float = 2.0,
    direct_level: float = 180.0,
    noise_std: float = 0.0,
    seed: int = 0,
    max_sequence_mib: float = 64,
) -> SyntheticResult:
    """Run generated float64 captures through the unchanged decoding pipeline.

    Each repeat gets independent noise from one seeded generator. The guard
    bounds one sequence and the generated pattern stack, not total peak RAM.
    Error metrics compare native projector coordinates with decoded identifiers;
    a nonzero error is retained, rather than adjusting the simulator to force
    an expected REOSSI result. Use small resolutions for this diagnostic.
    """
    if len(camera_size) != 2 or any(
        not isinstance(v, Integral) or isinstance(v, bool) or v < 1 for v in camera_size
    ):
        raise ValueError("Camera dimensions must be positive integers")
    camera_width, camera_height = camera_size
    SensorModel(black_level, direct_level, noise_std)
    sequence_bytes = plan.sequence_length * (
        camera_width * camera_height * 8 + np.prod(plan.pattern_size)
    )
    if (
        not isfinite(max_sequence_mib)
        or max_sequence_mib <= 0
        or sequence_bytes > max_sequence_mib * 1024**2
    ):
        raise ValueError(
            "Synthetic sequence exceeds max_sequence_mib; use a smaller plan"
        )
    width, height = plan.projector_size
    column_line = np.floor((np.arange(camera_width) + 0.5) * width / camera_width)
    row_line = np.floor((np.arange(camera_height) + 0.5) * height / camera_height)
    columns, rows = np.meshgrid(column_line.astype(int), row_line.astype(int))
    rng = np.random.default_rng(seed)

    def repeats(patterns: np.ndarray) -> Iterator[np.ndarray]:
        for _ in range(plan.repeats):
            yield sample_pattern_sequence(
                patterns,
                columns,
                rows,
                plan.projector_size,
                black_level=black_level,
                direct_level=direct_level,
                noise_std=noise_std,
                rng=rng,
            )

    def positions() -> Iterator[Iterator[np.ndarray]]:
        for displacement in range(plan.displacement_count):
            yield repeats(generate_pattern_sequence(plan, displacement))

    maps = reconstruct_identifier_maps(positions(), plan)
    expected_columns = columns + plan.identifier_origin
    expected_rows = rows + plan.identifier_origin
    model = {
        "name": "nearest_raster_scalar_sensor",
        "camera_size": list(camera_size),
        "black_level": black_level,
        "direct_level": direct_level,
        "noise_std": noise_std,
        "seed": seed,
    }
    return SyntheticResult(
        maps,
        expected_columns,
        expected_rows,
        _map_metrics(
            maps,
            expected_columns,
            expected_rows,
            identifier_origin=plan.identifier_origin,
        ),
        model,
    )


def save_synthetic_result(
    folder: str | Path, result: SyntheticResult, plan: ScanPlan
) -> None:
    """Save raw decoded/reference maps and the explicit forward-model settings."""
    folder = Path(folder)
    save_identifier_maps(folder, result.maps, plan)
    np.save(folder / "true_columns.npy", result.true_columns)
    np.save(folder / "true_rows.npy", result.true_rows)
    (folder / "simulation.json").write_text(
        json.dumps({"model": result.model, "metrics": result.metrics}, indent=2) + "\n",
        encoding="utf-8",
    )
