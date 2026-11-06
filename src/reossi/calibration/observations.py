"""Interactive calibration-image acquisition using an already opened camera.

The caller owns the camera lifetime. These functions create only their own
preview/projector windows and save unannotated frames when the board is valid.
"""

import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from .solvers import CameraCalibrator, ChArUcoBoardConfig
from .workflow import (
    CameraCalibration,
    _circle_calibrator,
    _image_size,
    load_camera_calibration,
)


def _capture_observations(
    camera,
    output_folder,
    detector: Callable[[np.ndarray], tuple[bool, np.ndarray]],
    *,
    flip,
    max_images,
    frame_timeout_seconds,
    preview_size,
    extension,
    projector_image=None,
    window_position=None,
    fullscreen=True,
    consecutive_valid_frames=0,
) -> list[Path]:
    if not isinstance(max_images, int) or max_images < 1:
        raise ValueError("max_images must be a positive integer")
    if (
        not np.isfinite(frame_timeout_seconds)
        or frame_timeout_seconds <= 0
        or not isinstance(consecutive_valid_frames, int)
        or consecutive_valid_frames < 0
    ):
        raise ValueError(
            "Frame timeout must be positive and consecutive_valid_frames nonnegative"
        )
    preview_size = _image_size(preview_size, "preview_size")
    if window_position is not None and (
        not isinstance(window_position, (tuple, list))
        or len(window_position) != 2
        or any(not isinstance(value, int) for value in window_position)
    ):
        raise ValueError("window_position must contain two integer coordinates")
    if extension not in {".jpg", ".png"}:
        raise ValueError("Calibration image extension must be .jpg or .png")
    folder = Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)
    preview_window = "REOSSI calibration camera"
    projector_window = "REOSSI calibration projector"
    windows = []
    saved = []
    consecutive = 0
    latest_original = None
    latest_valid = False
    next_index = 0
    last_frame_time = time.monotonic()
    try:
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
        windows.append(preview_window)
        if projector_image is not None:
            cv2.namedWindow(projector_window, cv2.WINDOW_NORMAL)
            windows.append(projector_window)
            if window_position is not None:
                cv2.moveWindow(projector_window, *window_position)
            if fullscreen:
                cv2.setWindowProperty(
                    projector_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )
            cv2.imshow(projector_window, projector_image)
        while len(saved) < max_images:
            frame = camera.get_frame()
            auto_save = False
            if frame is not None:
                last_frame_time = time.monotonic()
                if flip:
                    frame = cv2.flip(frame, -1)
                latest_original = frame.copy()
                latest_valid, annotated = detector(frame.copy())
                cv2.imshow(preview_window, cv2.resize(annotated, preview_size))
                consecutive = consecutive + 1 if latest_valid else 0
                auto_save = (
                    consecutive_valid_frames > 0
                    and consecutive >= consecutive_valid_frames
                )
            elif time.monotonic() - last_frame_time > frame_timeout_seconds:
                raise TimeoutError(
                    "No camera frame arrived before the calibration timeout"
                )
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if (
                latest_valid
                and latest_original is not None
                and (auto_save or key == ord("c"))
            ):
                path = folder / f"calibrate_{next_index:04d}{extension}"
                while path.exists():
                    next_index += 1
                    path = folder / f"calibrate_{next_index:04d}{extension}"
                if not cv2.imwrite(str(path), latest_original):
                    raise OSError(f"Could not save calibration image: {path}")
                saved.append(path)
                next_index += 1
                consecutive = 0
    finally:
        for name in reversed(windows):
            try:
                cv2.destroyWindow(name)
            except cv2.error:
                # A user may already have closed the window through its title bar.
                pass
    return saved


def capture_camera_calibration_images(
    camera,
    output_folder: str | Path,
    *,
    board: ChArUcoBoardConfig = ChArUcoBoardConfig(),
    flip: bool = False,
    max_images: int = 20,
    frame_timeout_seconds: float = 30,
    preview_size: tuple[int, int] = (960, 540),
    extension: str = ".jpg",
) -> list[Path]:
    """Preview a physical board; c saves a detected view and q exits.

    The supplied camera is not stopped here. Only unannotated frames are saved;
    the board used for acquisition must also be used when solving calibration.
    """
    solver = CameraCalibrator(board=board)

    def detect(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        valid, annotated, _, _ = solver.detect_markers(frame, gray, draw=True)
        return bool(valid), annotated

    return _capture_observations(
        camera,
        output_folder,
        detect,
        flip=flip,
        max_images=max_images,
        frame_timeout_seconds=frame_timeout_seconds,
        preview_size=preview_size,
        extension=extension,
    )


def capture_projector_calibration_images(
    camera,
    output_folder: str | Path,
    camera_calibration: CameraCalibration | str | Path,
    *,
    projector_size: tuple[int, int],
    grid_size: tuple[int, int] = (4, 11),
    circle_radius: int = 15,
    grid_origin: tuple[int, int] = (800, 350),
    board: ChArUcoBoardConfig | None = None,
    homography_reprojection_threshold: float = 5.0,
    window_position: tuple[int, int] | None = None,
    fullscreen: bool = True,
    consecutive_valid_frames: int = 6,
    flip: bool = False,
    max_images: int = 20,
    frame_timeout_seconds: float = 30,
    preview_size: tuple[int, int] = (960, 540),
    extension: str = ".jpg",
) -> list[Path]:
    """Project the configured grid and capture jointly detected board/circle views.

    A view is saved after ``consecutive_valid_frames`` successive detections (six follows
    the original >5 rule), or with c. This means detection continuity, not a
    measured motion threshold. Set consecutive_valid_frames=0 for manual acquisition only.
    q exits. The caller owns and must stop the supplied camera.
    The homography threshold uses the configured board's length units, not
    camera pixels. Use the same effective setting when collecting and solving.
    """
    if isinstance(camera_calibration, (str, Path)):
        camera_calibration = load_camera_calibration(camera_calibration)
    board = board if board is not None else camera_calibration.board
    if board is None:
        raise ValueError("A physical ChArUco board is required")
    solver = _circle_calibrator(
        projector_size,
        grid_size,
        circle_radius,
        grid_origin,
        render=False,
        cam_width=camera_calibration.image_size[0],
        cam_height=camera_calibration.image_size[1],
        cam_mtx=camera_calibration.matrix,
        cam_dist=camera_calibration.distortion,
        board=board,
        homography_reprojection_threshold=homography_reprojection_threshold,
    )
    projector_image = solver.get_circle_grid_image(*grid_origin)

    def detect(frame):
        if frame.shape[1::-1] != camera_calibration.image_size:
            raise ValueError("Camera frame size differs from calibration image size")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        valid, annotated, homography = solver.detect_markers(frame, gray, draw=True)
        if valid and homography is not None:
            valid, annotated, _, _ = solver.detect_circle_grid(
                annotated, gray, homography, draw=True
            )
        else:
            valid = False
        return bool(valid), annotated

    return _capture_observations(
        camera,
        output_folder,
        detect,
        flip=flip,
        max_images=max_images,
        frame_timeout_seconds=frame_timeout_seconds,
        preview_size=preview_size,
        extension=extension,
        projector_image=projector_image,
        window_position=window_position,
        fullscreen=fullscreen,
        consecutive_valid_frames=consecutive_valid_frames,
    )
