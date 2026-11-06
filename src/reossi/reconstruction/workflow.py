"""Reconstruct recorded identifier maps while preserving pixel-to-point links."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from ..acquisition.recording import read_plan
from ..calibration.workflow import StereoCalibration, load_stereo_calibration
from .geometry import StereoTriangulator


@dataclass
class ReconstructionResult:
    """Aligned point/color/pixel arrays and a mask of the retained camera pixels.

    Points have shape ``(3, N)``; RGB colors ``(N, 3)`` in 0..1; camera and
    projector coordinates ``(N, 2)``; mask ``(camera_height, camera_width)``.
    """

    points: np.ndarray
    colors: np.ndarray
    camera_pixels: np.ndarray
    projector_pixels: np.ndarray
    valid_mask: np.ndarray
    metadata: dict


def write_ply(path: str | Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Write finite ``(3,N)`` points and aligned RGB colors as ASCII PLY.

    No Open3D dependency is required. Existing files are protected, and an
    empty cloud is represented by a valid zero-vertex PLY file.
    """
    points = np.asarray(points)
    colors = np.asarray(colors)
    if points.ndim != 2 or points.shape[0] != 3 or colors.shape != (points.shape[1], 3):
        raise ValueError("PLY expects points (3,N) and colors (N,3)")
    if not np.isfinite(points).all() or not np.isfinite(colors).all():
        raise ValueError("PLY points and colors must be finite")
    if np.any((colors < 0) | (colors > 1)):
        raise ValueError("PLY RGB colors must lie in 0..1")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {points.shape[1]}\n"
            "property double x\nproperty double y\nproperty double z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        color_bytes = np.rint(colors * 255).astype(np.uint8)
        values = np.column_stack((points.T, color_bytes))
        np.savetxt(stream, values, fmt=["%.17g"] * 3 + ["%d"] * 3)


def reconstruct_scan(
    scan_folder: str | Path,
    calibration_file: StereoCalibration | str | Path,
    output_folder: str | Path,
    *,
    white_image: str | Path | None = None,
    coordinate_frame: Literal["camera", "projector", "legacy"] = "camera",
    box_threshold: float | None = None,
    reject_negative: bool = True,
    identifier_origin: int | None = None,
    max_points: int | None = 1_000_000,
) -> ReconstructionResult:
    """Load decoded maps, triangulate and export aligned numeric/PLY outputs.

    The default preserves the source's projector-coordinate use: identifiers
    are passed directly without subtracting a one-based origin. Set
    ``identifier_origin=1`` explicitly to convert one-based identifiers to
    zero-based projector pixels; values below that origin then become invalid.
    The selected interpretation is recorded and never inferred silently.

    ``camera`` returns camera-centered camera axes. ``projector`` returns
    projector-centered projector axes, including translation. ``legacy``
    retains the source's camera-centered, projector-oriented output. Filtering
    occurs in the selected frame, with units taken from calibration.
    ``max_points`` bounds valid candidate correspondences before the source
    collector creates Python lists. None explicitly disables that guard; the
    limit is not a bound on total process memory.
    """
    if coordinate_frame not in {"camera", "projector", "legacy"}:
        raise ValueError("coordinate_frame must be camera, projector, or legacy")
    if identifier_origin not in {None, 0, 1}:
        raise ValueError("identifier_origin must be None (source behavior), 0, or 1")
    if max_points is not None:
        if (
            not isinstance(max_points, (int, np.integer))
            or isinstance(max_points, bool)
            or max_points <= 0
        ):
            raise ValueError("max_points must be a positive integer or None")
        max_points = int(max_points)
    if box_threshold is not None and (
        not np.isfinite(box_threshold) or box_threshold <= 0
    ):
        raise ValueError("box_threshold must be finite and positive")
    output_folder = Path(output_folder)
    if output_folder.exists():
        raise FileExistsError(f"Reconstruction output already exists: {output_folder}")
    scan_folder = Path(scan_folder)
    plan = read_plan(scan_folder / "scan.json")
    if isinstance(calibration_file, (str, Path)):
        calibration = load_stereo_calibration(calibration_file)
    else:
        calibration = calibration_file
    columns = np.load(scan_folder / "h_pixels.npy", allow_pickle=False)
    rows = np.load(scan_folder / "v_pixels.npy", allow_pickle=False)
    expected_shape = (calibration.camera_size[1], calibration.camera_size[0])
    if columns.shape != expected_shape or rows.shape != expected_shape:
        raise ValueError("Identifier maps must match the calibrated camera image size")
    color_path = (
        Path(white_image) if white_image is not None else scan_folder / "white.png"
    )
    bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(
            f"Could not read the scan's white-reference image: {color_path}"
        )
    if bgr.shape[:2] != expected_shape:
        raise ValueError("White-reference image dimensions differ from identifier maps")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finite_identifiers = np.isfinite(columns) & np.isfinite(rows)
    column_coordinates = np.asarray(columns, dtype=np.float64).copy()
    row_coordinates = np.asarray(rows, dtype=np.float64).copy()
    if identifier_origin == 1:
        finite_identifiers &= (columns >= 1) & (rows >= 1)
        column_coordinates -= 1
        row_coordinates -= 1
    column_coordinates[~finite_identifiers] = -1
    row_coordinates[~finite_identifiers] = -1
    if reject_negative:
        candidates = (column_coordinates > -1) & (row_coordinates > -1)
    else:
        candidates = (column_coordinates != -1) & (row_coordinates != -1)
    candidate_count = int(np.count_nonzero(candidates))
    if max_points is not None and candidate_count > max_points:
        raise ValueError(
            f"Scan has {candidate_count} valid candidate points, exceeding "
            f"max_points={max_points}; select a larger explicit limit if resources allow"
        )
    triangulator = StereoTriangulator(
        camera_size=calibration.camera_size,
        camera_matrix=calibration.camera_matrix.copy(),
        camera_distortion=calibration.camera_distortion.copy(),
        projector_size=plan.projector_size,
        projector_calibration_size=calibration.projector_size,
        projector_matrix=calibration.projector_matrix.copy(),
        projector_distortion=calibration.projector_distortion.copy(),
        rotation=calibration.rotation.copy(),
        translation=calibration.translation.copy(),
    )
    camera_pixels, projector_pixels, colors = triangulator.collect_correspondences(
        column_coordinates, row_coordinates, rgb, reject_negative=reject_negative
    )
    if camera_pixels.size:
        with np.errstate(invalid="ignore", divide="ignore"):
            points = triangulator.triangulate(camera_pixels, projector_pixels)
        if coordinate_frame == "camera":
            points = triangulator.to_camera_frame(points)
        elif coordinate_frame == "projector":
            points = points + calibration.translation
    else:
        points = np.empty((3, 0), dtype=np.float64)
        camera_pixels = np.empty((0, 2), dtype=np.float32)
        projector_pixels = np.empty((0, 2), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.float64)
    finite_points = np.isfinite(points).all(axis=0)
    retained = finite_points.copy()
    if box_threshold is not None:
        retained &= (np.abs(points) < box_threshold).all(axis=0)
    initial_count = points.shape[1]
    points = points[:, retained]
    camera_pixels = camera_pixels[retained]
    projector_pixels = projector_pixels[retained]
    colors = colors[retained]
    valid_mask = np.zeros(expected_shape, dtype=bool)
    if camera_pixels.size:
        indices = camera_pixels.astype(np.intp)
        valid_mask[indices[:, 1], indices[:, 0]] = True
    metadata = {
        "format_version": 1,
        "coordinate_frame": coordinate_frame,
        "length_unit": calibration.length_unit,
        "identifier_origin": "legacy"
        if identifier_origin is None
        else identifier_origin,
        "plan_identifier_origin": plan.identifier_origin,
        "projector_size": list(plan.projector_size),
        "projector_calibration_size": list(calibration.projector_size),
        "camera_size": list(calibration.camera_size),
        "reject_negative": reject_negative,
        "box_threshold": box_threshold,
        "max_points": max_points,
        "candidate_correspondences": initial_count,
        "nonfinite_points_removed": int((~finite_points).sum()),
        "retained_points": int(points.shape[1]),
        "projector_coordinates_clipped_to_upper_edge": True,
    }
    result = ReconstructionResult(
        points, colors, camera_pixels, projector_pixels, valid_mask, metadata
    )
    output_folder.mkdir(parents=True, exist_ok=False)
    for name, array in (
        ("pts_3d", result.points),
        ("colors", result.colors),
        ("camera_pixels", result.camera_pixels),
        ("projector_pixels", result.projector_pixels),
        ("valid_mask", result.valid_mask),
    ):
        np.save(output_folder / f"{name}.npy", array)
    write_ply(output_folder / "cloud.ply", result.points, result.colors)
    (output_folder / "reconstruction.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return result
