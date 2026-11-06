"""Connected camera/projector calibration commands and observation capture."""

import logging
from math import isfinite
from pathlib import Path

import cv2

from ..acquisition.devices import open_camera
from ..calibration import ChArUcoBoardConfig
from ..calibration.observations import (
    capture_camera_calibration_images,
    capture_projector_calibration_images,
)
from ..calibration.workflow import (
    _circle_calibrator,
    _image_size,
    calibrate_camera_images,
    calibrate_projector_images,
    load_camera_calibration,
    save_board_image,
    save_circle_grid,
)
from .common import print_json, read_json, require_new_output
from .scan import add_camera_arguments, camera_options

logger = logging.getLogger(__name__)


def register_calibration_commands(commands) -> None:
    for name, description, handler in (
        ("board", "Generate the configured physical calibration board", board_image),
        ("circle-grid", "Generate the projected circle grid", circle_grid),
        (
            "calibrate-camera",
            "Calibrate a camera from saved board images",
            camera_calibration,
        ),
        (
            "calibrate-projector",
            "Calibrate projector and stereo geometry from saved observations",
            projector_calibration,
        ),
        (
            "collect-camera-calibration",
            "Capture physical board observations",
            collect_camera,
        ),
        (
            "collect-projector-calibration",
            "Capture projected-grid observations",
            collect_projector,
        ),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.set_defaults(handler=handler)
        if name in {"calibrate-camera", "calibrate-projector"}:
            command.add_argument("--input", type=Path, required=True)
        if name in {"calibrate-projector", "collect-projector-calibration"}:
            command.add_argument("--camera-calibration", type=Path, required=True)
        if name == "calibrate-projector":
            command.add_argument(
                "--projector-calibration",
                type=Path,
                help="Reuse existing projector intrinsics; solve stereo only",
            )
        if name.startswith("collect-"):
            add_camera_arguments(command)
            command.add_argument("--max-images", type=int, default=20)


def board_settings(settings: dict) -> ChArUcoBoardConfig:
    values = dict(settings.get("board", {}))
    if "dictionary" in values:
        name = values.pop("dictionary")
        if (
            not isinstance(name, str)
            or not name.startswith("DICT_")
            or not isinstance(getattr(cv2.aruco, name, None), int)
        ):
            raise ValueError("board.dictionary must name an OpenCV DICT_ constant")
        if "dictionary_id" in values:
            raise ValueError("Select either dictionary or dictionary_id")
        values["dictionary_id"] = getattr(cv2.aruco, name)
    if "squares" in values:
        values["squares"] = tuple(values["squares"])
    return ChArUcoBoardConfig(**values)


def grid_settings(settings: dict) -> dict:
    values = settings.get("circle_grid", {})
    return {
        "projector_size": tuple(settings["projector_size"]),
        "grid_size": tuple(values.get("size", (4, 11))),
        "circle_radius": values.get("radius", 15),
        "grid_origin": tuple(values.get("origin", (800, 350))),
    }


def solver_criteria(settings: dict, name: str):
    """Translate explicit iteration/epsilon settings into OpenCV termination."""
    values = settings.get(name)
    if values is None:
        return None
    if not isinstance(values, dict) or set(values) - {"max_iterations", "epsilon"}:
        raise ValueError(f"{name} requires max_iterations and epsilon")
    count, epsilon = values.get("max_iterations"), values.get("epsilon")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 1
        or not isinstance(epsilon, (int, float))
        or isinstance(epsilon, bool)
        or not isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError(f"{name} needs positive finite termination settings")
    return (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, count, epsilon)


def board_image(args) -> None:
    settings = read_json(args.config)
    save_board_image(
        args.output,
        board=board_settings(settings),
        image_size=tuple(settings.get("board_image_size", (1000, 1400))),
    )
    print_json({"board_image": str(args.output)})


def circle_grid(args) -> None:
    save_circle_grid(args.output, **grid_settings(read_json(args.config)))
    print_json({"circle_grid": str(args.output)})


def camera_calibration(args) -> None:
    settings = read_json(args.config)
    calibration = calibrate_camera_images(
        args.input,
        args.output,
        image_size=tuple(settings["camera_size"]),
        board=board_settings(settings),
        image_pattern=settings.get("image_pattern", "*.jpg"),
        length_unit=settings.get("length_unit", "m"),
        minimum_views=settings.get("minimum_views", 3),
        criteria=solver_criteria(settings, "camera_criteria"),
    )
    print_json(
        {
            "calibration": str(args.output),
            "rms": calibration.rms,
            "length_unit": calibration.length_unit,
            "diagnostics": calibration.diagnostics,
        }
    )


def projector_calibration(args) -> None:
    settings = read_json(args.config)
    calibration = calibrate_projector_images(
        args.input,
        args.output,
        load_camera_calibration(args.camera_calibration),
        **grid_settings(settings),
        board=board_settings(settings),
        stereo_image_size=tuple(settings.get("stereo_image_size", (2560, 1440))),
        image_pattern=settings.get("image_pattern", "*.jpg"),
        minimum_views=settings.get("minimum_views", 3),
        length_unit=settings.get("length_unit"),
        projector_criteria=solver_criteria(settings, "projector_criteria"),
        stereo_criteria=solver_criteria(settings, "stereo_criteria"),
        projector_calibration=args.projector_calibration,
        homography_reprojection_threshold=settings.get(
            "homography_reprojection_threshold", 5.0
        ),
    )
    print_json(
        {
            "calibration": str(args.output),
            "projector_rms": calibration.projector_rms,
            "stereo_rms": calibration.stereo_rms,
            "diagnostics": calibration.diagnostics,
        }
    )


def _capture(args, *, projector: bool) -> None:
    settings = read_json(args.config)
    acquisition = settings.get("acquisition", {})
    options = camera_options(
        args, acquisition | {"camera_size": settings["camera_size"]}
    )
    output = require_new_output(args.output)
    board = board_settings(settings)
    if args.max_images < 1:
        raise ValueError("max_images must be positive")
    timeout = acquisition.get("frame_timeout_seconds", 30)
    if not isfinite(timeout) or timeout <= 0:
        raise ValueError("frame_timeout_seconds must be positive")
    calibration = (
        load_camera_calibration(args.camera_calibration) if projector else None
    )
    grid = grid_settings(settings) if projector else None
    requested_size = _image_size(options["camera_size"])
    if requested_size != tuple(settings["camera_size"]):
        raise ValueError(
            "Calibration camera size must match camera_size in the configuration"
        )
    if calibration is not None and requested_size != calibration.image_size:
        raise ValueError("Requested camera size differs from camera calibration")
    if projector:
        _circle_calibrator(
            **grid,
            board=board,
            render=False,
            homography_reprojection_threshold=settings.get(
                "homography_reprojection_threshold", 5.0
            ),
        )
    preview = _image_size(acquisition.get("preview_size", (960, 540)), "preview_size")
    extension = acquisition.get(
        "extension", Path(settings.get("image_pattern", "*.jpg")).suffix
    )
    if extension not in {".png", ".jpg"}:
        raise ValueError("Collection image_pattern/extension must use .png or .jpg")
    if extension != Path(settings.get("image_pattern", "*.jpg")).suffix:
        raise ValueError("Collection extension must match calibration image_pattern")
    consecutive = acquisition.get("consecutive_valid_frames", 6)
    if (
        not isinstance(consecutive, int)
        or isinstance(consecutive, bool)
        or consecutive < 0
    ):
        raise ValueError("consecutive_valid_frames must be nonnegative")
    camera = None
    primary_error = None
    try:
        camera = open_camera(**options)
        common = {
            "board": board,
            "flip": acquisition.get("flip", False),
            "max_images": args.max_images,
            "frame_timeout_seconds": timeout,
            "preview_size": preview,
            "extension": extension,
        }
        if projector:
            paths = capture_projector_calibration_images(
                camera,
                output,
                calibration,
                **grid,
                consecutive_valid_frames=consecutive,
                homography_reprojection_threshold=settings.get(
                    "homography_reprojection_threshold", 5.0
                ),
                window_position=acquisition.get("window_position"),
                fullscreen=acquisition.get("fullscreen", True),
                **common,
            )
        else:
            paths = capture_camera_calibration_images(camera, output, **common)
        print_json({"observations": len(paths), "output": str(output)})
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if camera is not None:
            try:
                camera.stop_cam()
            except Exception:
                # Preserve the original collection failure or interruption.
                if primary_error is None:
                    raise
                logger.error(
                    "Camera shutdown also failed during collection", exc_info=True
                )


def collect_camera(args) -> None:
    _capture(args, projector=False)


def collect_projector(args) -> None:
    _capture(args, projector=True)
