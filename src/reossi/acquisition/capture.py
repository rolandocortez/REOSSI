"""One-view acquisition using the historical display/capture sequence.

Importing this module does not open a camera or create a display window.
Numerical processing remains in :mod:`reossi.pipeline`.
"""

import json
import math
import time
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from pathlib import Path

import cv2
import numpy as np

from ..pipeline import generate_pattern_sequence, reconstruct_identifier_maps
from ..plan import ScanPlan
from .devices import open_camera, validate_camera_options
from .recording import save_identifier_maps


def check_sequence_size(
    sequence_length: int,
    image_size: tuple[int, int],
    *,
    itemsize: int,
    max_sequence_mib: float,
) -> float:
    """Check a single sequence allocation and return its estimated MiB.

    This bound covers the image stack only, not peak decoding memory, camera
    queues, generated patterns, or accumulated identifier maps.
    """
    if not math.isfinite(max_sequence_mib) or max_sequence_mib <= 0:
        raise ValueError("max_sequence_mib must be finite and positive")
    if (
        not isinstance(sequence_length, int)
        or isinstance(sequence_length, bool)
        or sequence_length < 1
    ):
        raise ValueError("sequence_length must be a positive integer")
    if len(image_size) != 2 or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in image_size
    ):
        raise ValueError("image_size must contain two positive integers")
    if not isinstance(itemsize, int) or isinstance(itemsize, bool) or itemsize < 1:
        raise ValueError("itemsize must be a positive integer")
    width, height = image_size
    required_mib = sequence_length * width * height * itemsize / 1024**2
    if required_mib > max_sequence_mib:
        raise ValueError(
            f"One sequence needs {required_mib:.1f} MiB, exceeding the "
            f"{max_sequence_mib:g} MiB sequence limit"
        )
    return required_mib


def capture_sequence(
    camera,
    patterns: np.ndarray,
    *,
    settle_seconds: float,
    flip: bool,
    first_sequence: bool = False,
    max_sequence_mib: float = 1024,
    frame_timeout_seconds: float = 30,
    white_path: Path | None = None,
) -> np.ndarray:
    """Capture one ``float64 (N, H, W)`` grayscale sequence.

    The first exposure of the entire scan waits three times the usual dwell,
    matching the legacy scripts. A frame is polled after displaying each
    pattern; ``flip=True`` rotates it by 180 degrees before grayscale conversion.
    Pressing q aborts without deleting any completed recordings.
    """
    if (
        not math.isfinite(settle_seconds)
        or settle_seconds < 0
        or not math.isfinite(frame_timeout_seconds)
        or frame_timeout_seconds <= 0
    ):
        raise ValueError(
            "Dwell must be finite/nonnegative and frame timeout finite/positive"
        )
    if not math.isfinite(max_sequence_mib) or max_sequence_mib <= 0:
        raise ValueError("max_sequence_mib must be finite and positive")
    if patterns.ndim != 3 or min(patterns.shape) < 1:
        raise ValueError("Expected a nonempty grayscale pattern stack (N,H,W)")
    images = None
    for pattern_id, pattern in enumerate(patterns):
        cv2.imshow("Gray Code Pattern", pattern)
        started = time.monotonic()
        # Process the new pattern even when a camera frame is already ready.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt("Capture stopped with q")
        dwell = settle_seconds * (3 if first_sequence and pattern_id == 0 else 1)
        while True:
            frame = camera.get_frame()
            error = getattr(camera, "last_error", None)
            if error is not None:
                raise RuntimeError("Camera frame acquisition failed") from error
            elapsed = time.monotonic() - started
            if frame is not None and elapsed > dwell:
                if flip:
                    frame = cv2.flip(frame, -1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if images is None:
                    height, width = gray.shape
                    check_sequence_size(
                        len(patterns),
                        (width, height),
                        itemsize=8,
                        max_sequence_mib=max_sequence_mib,
                    )
                    # Retain the float64 allocation used by live legacy capture.
                    images = np.zeros((len(patterns), height, width))
                if gray.shape != images.shape[1:]:
                    raise ValueError("Camera dimensions changed during the sequence")
                images[pattern_id] = gray
                if white_path is not None and pattern_id == 1:
                    if not cv2.imwrite(str(white_path), frame):
                        raise OSError(
                            f"Could not write white-reference image: {white_path}"
                        )
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt("Capture stopped with q")
            if elapsed > dwell + frame_timeout_seconds:
                raise TimeoutError("Camera did not provide a frame before the timeout")
    if images is None:
        raise ValueError("Pattern sequence must not be empty")
    return images


def run_capture(
    plan: ScanPlan,
    output: str | Path,
    *,
    camera_kind: str = "opencv",
    camera_index: int = 0,
    camera_backend: int = cv2.CAP_DSHOW,
    device_id: str | None = None,
    oak_options: dict | None = None,
    camera_size: tuple[int, int] | None = None,
    fps: float = 30,
    settle_seconds: float | None = None,
    flip: bool | None = None,
    window_position: tuple[int, int] | None = None,
    fullscreen: bool = True,
    max_sequence_mib: float = 1024,
    frame_timeout_seconds: float = 30,
):
    """Record every run of one view, decode it, and save H/V identifier maps.

    ``output`` must be a new directory. Camera construction is explicit and
    occurs only after argument and requested-allocation checks. Actual frame
    dimensions are checked again before allocating the capture stack. Plan and
    requested acquisition settings are written before opening hardware, and
    completed sequence records survive a failed or interrupted session.
    """
    if camera_size is None:
        camera_size = (3840, 2160) if camera_kind == "oak" else (1920, 1080)
    validate_camera_options(
        camera_kind=camera_kind,
        camera_index=camera_index,
        camera_size=camera_size,
        fps=fps,
        camera_backend=camera_backend,
        device_id=device_id,
        oak_options=oak_options,
    )
    if settle_seconds is None:
        settle_seconds = 0.3 if plan.mode == "conventional" else 1.0
    if flip is None:
        flip = plan.mode != "conventional"
    if window_position is None and plan.mode != "conventional":
        window_position = (2020, 100)
    if (
        not math.isfinite(settle_seconds)
        or settle_seconds < 0
        or not math.isfinite(frame_timeout_seconds)
        or frame_timeout_seconds <= 0
    ):
        raise ValueError(
            "Dwell must be finite/nonnegative and frame timeout finite/positive"
        )
    check_sequence_size(
        plan.sequence_length,
        camera_size,
        itemsize=8,
        max_sequence_mib=max_sequence_mib,
    )
    check_sequence_size(
        plan.sequence_length,
        plan.pattern_size,
        itemsize=1,
        max_sequence_mib=max_sequence_mib,
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "scan.json").write_text(
        json.dumps({"format_version": 1, "plan": plan.to_dict()}, indent=2) + "\n",
        encoding="utf-8",
    )
    options = dict(oak_options or {})
    if is_dataclass(options.get("keyboard_controls")):
        options["keyboard_controls"] = asdict(options["keyboard_controls"])
    acquisition = {
        "status": "starting",
        "camera_kind": camera_kind,
        "camera_index": camera_index,
        "camera_backend": camera_backend,
        "camera_size_requested": list(camera_size),
        "fps": fps,
        "device_id": device_id,
        "oak_options": options,
        "settle_seconds": settle_seconds,
        "flip": flip,
        "window_position": window_position,
        "fullscreen": fullscreen,
        "max_sequence_mib": max_sequence_mib,
        "frame_timeout_seconds": frame_timeout_seconds,
        "completed_sequences": [],
    }

    def write_metadata():
        temporary = output / "acquisition.json.tmp"
        temporary.write_text(
            json.dumps(acquisition, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        temporary.replace(output / "acquisition.json")

    write_metadata()
    camera = None
    window_opened = False
    primary_error = None
    try:
        camera = open_camera(
            camera_kind=camera_kind,
            camera_index=camera_index,
            camera_size=camera_size,
            fps=fps,
            camera_backend=camera_backend,
            device_id=device_id,
            oak_options=oak_options,
        )
        cv2.namedWindow("Gray Code Pattern", cv2.WINDOW_NORMAL)
        window_opened = True
        if window_position is not None:
            cv2.moveWindow("Gray Code Pattern", *window_position)
        if fullscreen:
            cv2.setWindowProperty(
                "Gray Code Pattern", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        acquisition["status"] = "capturing"
        write_metadata()

        def recorded_positions():
            for displacement in range(plan.displacement_count):
                patterns = generate_pattern_sequence(plan, displacement)

                def recorded_runs():
                    for run in range(plan.repeats):
                        images = capture_sequence(
                            camera,
                            patterns,
                            settle_seconds=settle_seconds,
                            flip=flip,
                            first_sequence=displacement == 0 and run == 0,
                            max_sequence_mib=max_sequence_mib,
                            frame_timeout_seconds=frame_timeout_seconds,
                            white_path=output / "white.png"
                            if displacement == 0 and run == 0
                            else None,
                        )
                        np.savez_compressed(
                            output / f"run_{run}_displacement_{displacement}.npz",
                            images=images,
                        )
                        acquisition["camera_size_observed"] = [
                            images.shape[2],
                            images.shape[1],
                        ]
                        acquisition["completed_sequences"].append(
                            {
                                "run": run,
                                "displacement": displacement,
                                "file": f"run_{run}_displacement_{displacement}.npz",
                                "frames": len(images),
                                "dtype": str(images.dtype),
                            }
                        )
                        write_metadata()
                        yield images
                        del images

                yield recorded_runs()

        maps = reconstruct_identifier_maps(recorded_positions(), plan)
        save_identifier_maps(output, maps, plan)
        acquisition["status"] = "complete"
        return maps
    except BaseException as error:
        primary_error = error
        acquisition["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        acquisition["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        close_error = None
        if camera is not None:
            try:
                camera.stop_cam()
            except Exception as error:
                close_error = error
                acquisition["close_error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
        if window_opened:
            with suppress(cv2.error):
                cv2.destroyWindow("Gray Code Pattern")
        if close_error is not None and primary_error is None:
            acquisition["status"] = "failed"
            write_metadata()
            raise close_error
        if primary_error is None:
            write_metadata()
        else:
            with suppress(OSError):
                write_metadata()
