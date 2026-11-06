"""Project actual Gray patterns onto a triangle mesh and decode virtual captures.

This initial renderer models calibrated pinhole geometry, camera/projector
visibility, uniform direct response, a scalar black level and optional sensor
noise. Its known surface geometry is independent of the decoded identifiers.
"""

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path

import cv2
import numpy as np

from ..acquisition.recording import save_identifier_maps
from ..calibration.workflow import StereoCalibration
from ..pipeline import (
    IdentifierMaps,
    generate_pattern_sequence,
    reconstruct_identifier_maps,
)
from ..plan import ScanPlan
from .geometry import PinholeDevice, TriangleMesh, relative_calibration
from .raytrace import SceneCorrespondences, scene_correspondences
from .sensor import SensorModel


def render_sequence(
    patterns: np.ndarray,
    geometry: SceneCorrespondences,
    projector_size: tuple[int, int],
    *,
    sensor: SensorModel = SensorModel(),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return grayscale ``(N,H,W)`` images for one displayed pattern sequence.

    Continuous projector coordinates sample the nearest native projector
    pixel, then map into the low-resolution raster by integer floor. This
    retains the legacy placement of patterns; no subcell shift is invented.
    Unilluminated/background pixels receive only the black level and noise.
    """
    pixels = np.where(
        np.isfinite(geometry.projector_pixels), geometry.projector_pixels, 0
    )
    native_columns = np.floor(pixels[..., 0] + 0.5).astype(int)
    native_rows = np.floor(pixels[..., 1] + 0.5).astype(int)
    width, height = projector_size
    columns = np.clip(
        native_columns * patterns.shape[2] // width, 0, patterns.shape[2] - 1
    )
    rows = np.clip(native_rows * patterns.shape[1] // height, 0, patterns.shape[1] - 1)
    images = np.full(
        (len(patterns), *geometry.illuminated.shape),
        sensor.black_level,
        dtype=np.float64,
    )
    images += (
        sensor.direct_level
        * patterns[:, rows, columns].astype(float)
        / 255
        * geometry.illuminated
    )
    if sensor.noise_std:
        rng = rng if rng is not None else np.random.default_rng()
        images += rng.normal(0, sensor.noise_std, images.shape)
    return np.clip(images, 0, 255)


def evaluate_scene_maps(
    maps: IdentifierMaps,
    columns: np.ndarray,
    rows: np.ndarray,
    illuminated: np.ndarray,
    *,
    identifier_origin: int = 0,
) -> dict:
    """Evaluate identifiers against independently known projector pixels.

    Coverage uses truly illuminated pixels as its denominator. Coordinate
    errors use pixels valid in both axes and truly illuminated. Decoding
    outside that mask is counted separately, not credited as coverage.
    """
    predicted_valid = (maps.columns >= identifier_origin) & (
        maps.rows >= identifier_origin
    )
    valid = predicted_valid & illuminated
    count = int(illuminated.sum())
    metrics = {
        "camera_pixels": int(illuminated.size),
        "illuminated_pixels": count,
        "valid_pixels": int(valid.sum()),
        "valid_fraction": float(valid.sum() / count) if count else None,
        "decoded_outside_illuminated_pixels": int(
            (predicted_valid & ~illuminated).sum()
        ),
    }
    for name, actual, expected in (
        ("columns", maps.columns, columns),
        ("rows", maps.rows, rows),
    ):
        error = np.abs(actual[valid] - expected[valid])
        metrics[f"{name}_mean_absolute_error_native_pixels"] = (
            float(error.mean()) if error.size else None
        )
        metrics[f"{name}_exact_fraction_of_valid"] = (
            float((error == 0).mean()) if error.size else None
        )
    metrics["joint_exact_fraction_of_illuminated"] = (
        float(((maps.columns == columns) & (maps.rows == rows) & valid).sum() / count)
        if count
        else None
    )
    return metrics


@dataclass
class SceneScanResult:
    """Virtual scan output with independent geometry and known calibration."""

    maps: IdentifierMaps
    true_columns: np.ndarray
    true_rows: np.ndarray
    geometry: SceneCorrespondences
    calibration: StereoCalibration
    white_image: np.ndarray
    metrics: dict
    model: dict
    mesh: TriangleMesh


def simulate_scene_scan(
    mesh: TriangleMesh,
    camera: PinholeDevice,
    projector: PinholeDevice,
    plan: ScanPlan,
    *,
    sensor: SensorModel = SensorModel(),
    seed: int = 0,
    max_sequence_mib: float = 64,
    max_triangle_tests: int = 20_000_000,
    recording_folder: str | Path | None = None,
    length_unit: str = "m",
) -> SceneScanResult:
    """Render ordered virtual captures and run the shared REOSSI pipeline.

    At most one capture sequence is generated at a time. Optional recordings
    use the same run/displacement NPZ layout as live acquisition. The sequence
    guard includes capture and pattern arrays, not all decoder intermediates.
    """
    if projector.image_size != plan.projector_size:
        raise ValueError("Virtual projector dimensions must match the scan plan")
    width, height = camera.image_size
    sequence_bytes = plan.sequence_length * (
        width * height * 8 + np.prod(plan.pattern_size)
    )
    if (
        not isfinite(max_sequence_mib)
        or max_sequence_mib <= 0
        or sequence_bytes > max_sequence_mib * 1024**2
    ):
        raise ValueError("Virtual capture exceeds max_sequence_mib; use a smaller scan")
    geometry = scene_correspondences(
        mesh, camera, projector, max_triangle_tests=max_triangle_tests
    )
    rotation, translation = relative_calibration(camera, projector)
    calibration = StereoCalibration(
        camera_matrix=camera.matrix,
        camera_distortion=np.zeros(5),
        camera_size=camera.image_size,
        projector_matrix=projector.matrix,
        projector_distortion=np.zeros(5),
        projector_size=projector.image_size,
        rotation=rotation,
        translation=translation,
        length_unit=length_unit,
        stereo_image_size=camera.image_size,
        diagnostics={
            "source": "known_virtual_geometry",
            "estimated_from_images": False,
        },
    )
    recording_path = Path(recording_folder) if recording_folder is not None else None
    if recording_path is not None:
        recording_path.mkdir(parents=True, exist_ok=False)
        (recording_path / "plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2) + "\n"
        )
    rng = np.random.default_rng(seed)
    white_image = None

    def positions() -> Iterator[Iterator[np.ndarray]]:
        for displacement in range(plan.displacement_count):
            patterns = generate_pattern_sequence(plan, displacement)

            def runs() -> Iterator[np.ndarray]:
                nonlocal white_image
                for run in range(plan.repeats):
                    images = render_sequence(
                        patterns, geometry, plan.projector_size, sensor=sensor, rng=rng
                    )
                    if white_image is None:
                        white_image = np.repeat(images[1, ..., None], 3, axis=2).astype(
                            np.uint8
                        )
                        if recording_path is not None and not cv2.imwrite(
                            str(recording_path / "white.png"), white_image
                        ):
                            raise OSError(
                                "Could not save virtual recording's white reference"
                            )
                    if recording_path is not None:
                        np.savez_compressed(
                            recording_path
                            / f"run_{run}_displacement_{displacement}.npz",
                            images=images,
                        )
                    yield images
                    del images

            yield runs()

    maps = reconstruct_identifier_maps(positions(), plan)
    finite_pixels = np.where(
        np.isfinite(geometry.projector_pixels), geometry.projector_pixels, 0
    )
    expected = np.floor(finite_pixels + 0.5).astype(np.int64) + plan.identifier_origin
    expected[~geometry.illuminated] = -1
    metrics = evaluate_scene_maps(
        maps,
        expected[..., 0],
        expected[..., 1],
        geometry.illuminated,
        identifier_origin=plan.identifier_origin,
    )
    model = {
        "name": "direct_pinhole_triangle_mesh",
        "sensor": asdict(sensor),
        "seed": seed,
        "length_unit": length_unit,
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),
        "camera_visible_pixels": int(geometry.camera_visible.sum()),
        "camera_visible_unilluminated_pixels": int(
            (geometry.camera_visible & ~geometry.illuminated).sum()
        ),
        "max_triangle_tests": max_triangle_tests,
        "limitations": [
            "uniform_direct_response",
            "no_lens_distortion",
            "no_global_light_transport",
        ],
    }
    for name, device in (("camera", camera), ("projector", projector)):
        model[name] = {
            "image_size": list(device.image_size),
            "matrix": device.matrix.tolist(),
            "rotation": device.rotation.tolist(),
            "translation": device.translation.tolist(),
        }
    return SceneScanResult(
        maps,
        expected[..., 0],
        expected[..., 1],
        geometry,
        calibration,
        white_image,
        metrics,
        model,
        mesh,
    )


def save_scene_result(
    folder: str | Path, result: SceneScanResult, plan: ScanPlan
) -> None:
    """Save decoded maps, geometry truth, known calibration and sensor settings."""
    folder = Path(folder)
    save_identifier_maps(folder, result.maps, plan)
    for name, array in {
        "true_columns": result.true_columns,
        "true_rows": result.true_rows,
        "true_points_camera": result.geometry.points_camera,
        "true_projector_pixels": result.geometry.projector_pixels,
        "illuminated": result.geometry.illuminated,
        "camera_visible": result.geometry.camera_visible,
    }.items():
        np.save(folder / f"{name}.npy", array)
    result.calibration.save(folder / "calibration.npz")
    np.savez_compressed(
        folder / "scene_mesh.npz",
        vertices=result.mesh.vertices,
        faces=result.mesh.faces,
    )
    if not cv2.imwrite(str(folder / "white.png"), result.white_image):
        raise OSError("Could not save virtual white-reference image")
    (folder / "simulation.json").write_text(
        json.dumps({"model": result.model, "metrics": result.metrics}, indent=2) + "\n"
    )
