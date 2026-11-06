"""Calibration workflows and self-describing camera/projector parameter files.

Board dimensions and translation use ``length_unit`` without automatic unit
conversion. Camera/projector pixel sizes are always ``(width, height)``. Stereo
extrinsics map camera coordinates to projector coordinates: ``Xp = R @ Xc + T``.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .solvers import CameraCalibrator, ChArUcoBoardConfig, ProjectorCameraCalibrator


def _image_size(value, name="image_size") -> tuple[int, int]:
    if (
        not isinstance(value, (tuple, list, np.ndarray))
        or len(value) != 2
        or any(
            not isinstance(item, (int, np.integer))
            or isinstance(item, bool)
            or item < 1
            for item in value
        )
    ):
        raise ValueError(f"{name} must contain two positive integer dimensions")
    return int(value[0]), int(value[1])


def _matrix(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite array of shape {shape}")
    return array.copy()


def _intrinsics(value, name):
    array = _matrix(value, (3, 3), name)
    if array[0, 0] <= 0 or array[1, 1] <= 0 or not np.allclose(array[2], [0, 0, 1]):
        raise ValueError(f"{name} needs positive focal lengths and final row [0, 0, 1]")
    return array


def _distortion(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.size not in {4, 5, 8, 12, 14} or not np.isfinite(array).all():
        raise ValueError(f"{name} must have 4, 5, 8, 12, or 14 finite coefficients")
    return array.reshape(-1, 1).copy()


def _check_unit_and_rms(length_unit, *rms_values):
    if not isinstance(length_unit, str) or not length_unit.strip():
        raise ValueError("length_unit must explicitly name the physical board unit")
    if any(
        value is not None and (not np.isfinite(value) or value < 0)
        for value in rms_values
    ):
        raise ValueError("Calibration RMS values must be finite and nonnegative")


def _solver_criteria(value, name):
    if value is None:
        return None
    if (
        len(value) != 3
        or not isinstance(value[0], int)
        or value[0] not in {0, 1, 2, 3}
        or not isinstance(value[1], int)
        or value[1] < 1
        or not np.isfinite(value[2])
        or value[2] < 0
    ):
        raise ValueError(
            f"{name} must contain flags 0..3, a positive iteration count, and nonnegative epsilon"
        )
    return tuple(value)


def _board_values(board):
    return None if board is None else asdict(board)


def _board_from_values(values):
    if values is None:
        return None
    return ChArUcoBoardConfig(
        squares=tuple(values["squares"]),
        square_length=values["square_length"],
        marker_length=values["marker_length"],
        dictionary_id=values["dictionary_id"],
    )


def _save_archive(path, arrays, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects a previous calibration from accidental replacement.
    with path.open("xb") as stream:
        np.savez_compressed(stream, metadata=json.dumps(metadata), **arrays)


def _load_archive(path, kind):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("format_version") != 1 or metadata.get("kind") != kind:
            raise ValueError(f"Expected a version-1 {kind} calibration file")
        arrays = {
            name: archive[name].copy() for name in archive.files if name != "metadata"
        }
    return arrays, metadata


@dataclass
class CameraCalibration:
    """Validated camera parameters and diagnostics, independent of old files."""

    matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    board: ChArUcoBoardConfig | None = field(default_factory=ChArUcoBoardConfig)
    length_unit: str = "m"
    rms: float | None = None
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.matrix = _intrinsics(self.matrix, "camera matrix")
        self.distortion = _distortion(self.distortion, "camera distortion")
        self.image_size = _image_size(self.image_size)
        _check_unit_and_rms(self.length_unit, self.rms)

    def save(self, path: str | Path) -> None:
        """Write a new NPZ containing matrices, dimensions, board and diagnostics."""
        _save_archive(
            path,
            {"matrix": self.matrix, "distortion": self.distortion},
            {
                "format_version": 1,
                "kind": "camera",
                "image_size": self.image_size,
                "board": _board_values(self.board),
                "length_unit": self.length_unit,
                "rms": self.rms,
                "diagnostics": self.diagnostics,
            },
        )


@dataclass
class StereoCalibration:
    """Camera/projector parameters with camera-to-projector extrinsics.

    ``stereo_image_size`` records the solver argument independently of camera
    and projector dimensions. Its source default is deliberately retained.
    """

    camera_matrix: np.ndarray
    camera_distortion: np.ndarray
    camera_size: tuple[int, int]
    projector_matrix: np.ndarray
    projector_distortion: np.ndarray
    projector_size: tuple[int, int]
    rotation: np.ndarray
    translation: np.ndarray
    length_unit: str = "m"
    board: ChArUcoBoardConfig | None = None
    camera_rms: float | None = None
    projector_rms: float | None = None
    stereo_rms: float | None = None
    stereo_image_size: tuple[int, int] = (2560, 1440)
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self):
        self.camera_matrix = _intrinsics(self.camera_matrix, "camera matrix")
        self.camera_distortion = _distortion(
            self.camera_distortion, "camera distortion"
        )
        self.projector_matrix = _intrinsics(self.projector_matrix, "projector matrix")
        self.projector_distortion = _distortion(
            self.projector_distortion, "projector distortion"
        )
        self.camera_size = _image_size(self.camera_size, "camera_size")
        self.projector_size = _image_size(self.projector_size, "projector_size")
        self.stereo_image_size = _image_size(
            self.stereo_image_size, "stereo_image_size"
        )
        self.rotation = _matrix(self.rotation, (3, 3), "rotation")
        if not np.allclose(
            self.rotation.T @ self.rotation, np.eye(3), atol=1e-6
        ) or not np.isclose(np.linalg.det(self.rotation), 1.0, atol=1e-6):
            raise ValueError("rotation must be an orthonormal proper rotation matrix")
        translation = np.asarray(self.translation, dtype=np.float64)
        if (
            translation.size != 3
            or not np.isfinite(translation).all()
            or np.linalg.norm(translation) == 0
        ):
            raise ValueError("translation must contain a finite, nonzero three-vector")
        self.translation = translation.reshape(3, 1).copy()
        _check_unit_and_rms(
            self.length_unit, self.camera_rms, self.projector_rms, self.stereo_rms
        )

    def save(self, path: str | Path) -> None:
        """Write validated parameters to a new self-describing NPZ file."""
        arrays = {
            name: getattr(self, name)
            for name in (
                "camera_matrix",
                "camera_distortion",
                "projector_matrix",
                "projector_distortion",
                "rotation",
                "translation",
            )
        }
        _save_archive(
            path,
            arrays,
            {
                "format_version": 1,
                "kind": "stereo",
                "camera_size": self.camera_size,
                "projector_size": self.projector_size,
                "stereo_image_size": self.stereo_image_size,
                "length_unit": self.length_unit,
                "board": _board_values(self.board),
                "camera_rms": self.camera_rms,
                "projector_rms": self.projector_rms,
                "stereo_rms": self.stereo_rms,
                "diagnostics": self.diagnostics,
                "extrinsics": "projector_point = rotation @ camera_point + translation",
            },
        )


def load_camera_calibration(path: str | Path) -> CameraCalibration:
    """Read and validate the camera calibration format produced here."""
    arrays, metadata = _load_archive(path, "camera")
    return CameraCalibration(
        arrays["matrix"],
        arrays["distortion"],
        tuple(metadata["image_size"]),
        board=_board_from_values(metadata["board"]),
        length_unit=metadata["length_unit"],
        rms=metadata["rms"],
        diagnostics=metadata["diagnostics"],
    )


def load_stereo_calibration(path: str | Path) -> StereoCalibration:
    """Read validated camera/projector parameters and their coordinate convention."""
    arrays, metadata = _load_archive(path, "stereo")
    return StereoCalibration(
        arrays["camera_matrix"],
        arrays["camera_distortion"],
        tuple(metadata["camera_size"]),
        arrays["projector_matrix"],
        arrays["projector_distortion"],
        tuple(metadata["projector_size"]),
        arrays["rotation"],
        arrays["translation"],
        length_unit=metadata["length_unit"],
        board=_board_from_values(metadata["board"]),
        camera_rms=metadata["camera_rms"],
        projector_rms=metadata["projector_rms"],
        stereo_rms=metadata["stereo_rms"],
        stereo_image_size=tuple(metadata["stereo_image_size"]),
        diagnostics=metadata["diagnostics"],
    )


def calibrate_camera_images(
    input_folder: str | Path,
    output_file: str | Path,
    *,
    image_size: tuple[int, int],
    board: ChArUcoBoardConfig = ChArUcoBoardConfig(),
    image_pattern: str = "*.jpg",
    length_unit: str = "m",
    minimum_views: int = 3,
    criteria: tuple[int, int, float] | None = None,
    flags: int = cv2.CALIB_USE_INTRINSIC_GUESS,
) -> CameraCalibration:
    """Detect a configured board, calibrate and save matrices plus RMS diagnostics.

    No image is resized and no physical board preset is inferred. ``criteria``
    defaults to the original solver expression; supplying a different criterion
    is an explicit numerical choice, recorded in diagnostics.
    """
    image_size = _image_size(image_size)
    criteria = _solver_criteria(criteria, "criteria")
    _check_unit_and_rms(length_unit)
    if (
        not isinstance(minimum_views, int)
        or isinstance(minimum_views, bool)
        or minimum_views < 1
    ):
        raise ValueError("minimum_views must be positive for a calibration workflow")
    if Path(output_file).exists():
        raise FileExistsError(f"Calibration output already exists: {output_file}")
    solver = CameraCalibrator(*image_size, board=board, image_pattern=image_pattern)
    matrix, distortion = solver.calibrate(
        str(input_folder),
        flags,
        minimum_views=minimum_views,
        criteria=criteria,
        verbose=False,
    )
    calibration = CameraCalibration(
        matrix,
        distortion,
        image_size,
        board=board,
        length_unit=length_unit,
        rms=solver.diagnostics["rms"],
        diagnostics=solver.diagnostics,
    )
    calibration.save(output_file)
    return calibration


def _circle_calibrator(
    projector_size,
    grid_size,
    circle_radius,
    grid_origin,
    *,
    render=True,
    homography_reprojection_threshold=5.0,
    **kwargs,
):
    projector_size = _image_size(projector_size, "projector_size")
    grid_size = _image_size(grid_size, "grid_size")
    if not isinstance(circle_radius, int) or circle_radius < 1:
        raise ValueError("circle_radius must be a positive integer")
    if len(grid_origin) != 2 or any(
        not isinstance(value, int) for value in grid_origin
    ):
        raise ValueError("grid_origin must contain two integer pixel coordinates")
    solver = ProjectorCameraCalibrator(
        proj_width=projector_size[0],
        proj_height=projector_size[1],
        circle_grid_size=grid_size,
        circle_r=circle_radius,
        homography_reprojection_threshold=homography_reprojection_threshold,
        **kwargs,
    )
    points = solver.circle_2d_pts + np.asarray(grid_origin)
    padding = 3 * circle_radius
    if np.any(points.min(axis=0) - padding < 0) or np.any(
        points.max(axis=0) + padding > np.asarray(projector_size)
    ):
        raise ValueError(
            "Projected grid and its white padding must fit within projector_size"
        )
    if render:
        solver.get_circle_grid_image(*grid_origin)
    else:
        solver.circle_2d = points
    return solver


def save_board_image(
    output_file: str | Path,
    *,
    board: ChArUcoBoardConfig = ChArUcoBoardConfig(),
    image_size: tuple[int, int] = (1000, 1400),
    margin: int = 20,
) -> Path:
    """Generate the configured printable ChArUco board without opening a device.

    Printed square dimensions must be measured against ``board.square_length``;
    pixel dimensions alone do not establish physical print scale.
    """
    image_size = _image_size(image_size)
    if margin < 0 or 2 * margin >= min(image_size):
        raise ValueError("margin must fit inside the board image")
    dictionary = cv2.aruco.getPredefinedDictionary(board.dictionary_id)
    target = cv2.aruco.CharucoBoard(
        board.squares, board.square_length, board.marker_length, dictionary
    )
    image = target.generateImage(image_size, marginSize=margin)
    return _save_image(output_file, image)


def _save_image(output_file, image):
    path = Path(output_file)
    if path.exists():
        raise FileExistsError(f"Image output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")
    return path


def save_circle_grid(
    output_file: str | Path,
    *,
    projector_size: tuple[int, int] = (1920, 1080),
    grid_size: tuple[int, int] = (4, 11),
    circle_radius: int = 15,
    grid_origin: tuple[int, int] = (800, 350),
) -> Path:
    """Save the reference asymmetric projected grid at explicit pixel coordinates."""
    solver = _circle_calibrator(
        projector_size, grid_size, circle_radius, grid_origin, render=False
    )
    image = solver.get_circle_grid_image(*grid_origin)
    return _save_image(output_file, image)


def calibrate_projector_images(
    input_folder: str | Path,
    output_file: str | Path,
    camera_calibration: CameraCalibration | str | Path,
    *,
    projector_size: tuple[int, int],
    grid_size: tuple[int, int] = (4, 11),
    circle_radius: int = 15,
    grid_origin: tuple[int, int] = (800, 350),
    board: ChArUcoBoardConfig | None = None,
    stereo_image_size: tuple[int, int] = (2560, 1440),
    image_pattern: str = "*.jpg",
    minimum_views: int = 3,
    length_unit: str | None = None,
    projector_calibration: StereoCalibration | str | Path | None = None,
    projector_criteria: tuple[int, int, float] | None = None,
    stereo_criteria: tuple[int, int, float] | None = None,
    homography_reprojection_threshold: float = 5.0,
) -> StereoCalibration:
    """Calibrate projector intrinsics and stereo extrinsics from paired views.

    Passing ``projector_calibration`` fixes its existing intrinsics and solves
    only the stereo relation. An explicit projector board can differ from the
    board used for camera intrinsics. When omitted, its definition and length
    unit are inherited from camera calibration. No board lengths are converted.
    ``stereo_image_size`` retains the reference default unless explicitly set;
    its effective value is saved, even when it differs from camera dimensions.
    The homography threshold is measured in the destination board coordinates,
    using ``length_unit``; its source default5.0 is not a pixel threshold here.
    """
    if isinstance(camera_calibration, (str, Path)):
        camera_calibration = load_camera_calibration(camera_calibration)
    if isinstance(projector_calibration, (str, Path)):
        projector_calibration = load_stereo_calibration(projector_calibration)
    projector_criteria = _solver_criteria(projector_criteria, "projector_criteria")
    stereo_criteria = _solver_criteria(stereo_criteria, "stereo_criteria")
    if (
        not isinstance(minimum_views, int)
        or isinstance(minimum_views, bool)
        or minimum_views < 1
    ):
        raise ValueError("minimum_views must be positive for a calibration workflow")
    if Path(output_file).exists():
        raise FileExistsError(f"Calibration output already exists: {output_file}")
    board = board if board is not None else camera_calibration.board
    if board is None:
        raise ValueError(
            "A physical ChArUco board must be specified for projector calibration"
        )
    length_unit = camera_calibration.length_unit if length_unit is None else length_unit
    _check_unit_and_rms(length_unit)
    if (
        projector_calibration is not None
        and tuple(projector_size) != projector_calibration.projector_size
    ):
        raise ValueError("Existing projector calibration has a different image size")
    solver = _circle_calibrator(
        projector_size,
        grid_size,
        circle_radius,
        grid_origin,
        render=False,
        cam_width=camera_calibration.image_size[0],
        cam_height=camera_calibration.image_size[1],
        cam_mtx=camera_calibration.matrix.copy(),
        cam_dist=camera_calibration.distortion.copy(),
        proj_mtx=None
        if projector_calibration is None
        else projector_calibration.projector_matrix.copy(),
        proj_dist=None
        if projector_calibration is None
        else projector_calibration.projector_distortion.copy(),
        board=board,
        stereo_image_size=_image_size(stereo_image_size),
        image_pattern=image_pattern,
        homography_reprojection_threshold=homography_reprojection_threshold,
    )
    values = solver.calibrate(
        str(input_folder),
        calibrate_proj=projector_calibration is None,
        minimum_views=minimum_views,
        verbose=False,
        projector_criteria=projector_criteria,
        stereo_criteria=stereo_criteria,
    )
    projector_matrix, projector_distortion, rotation, translation = values[:4]
    diagnostics = dict(solver.diagnostics)
    diagnostics.update(
        grid_size=list(grid_size),
        circle_radius=circle_radius,
        grid_origin=list(grid_origin),
        homography_reprojection_threshold_unit=length_unit,
    )
    calibration = StereoCalibration(
        camera_calibration.matrix,
        camera_calibration.distortion,
        camera_calibration.image_size,
        projector_matrix,
        projector_distortion,
        projector_size,
        rotation,
        translation,
        length_unit=length_unit,
        board=board,
        camera_rms=camera_calibration.rms,
        projector_rms=diagnostics["projector_rms"]
        if projector_calibration is None
        else projector_calibration.projector_rms,
        stereo_rms=diagnostics["stereo_rms"],
        stereo_image_size=stereo_image_size,
        diagnostics=diagnostics,
    )
    calibration.save(output_file)
    return calibration
