"""Explicit construction shared by scanning and calibration acquisition."""

import math
from dataclasses import asdict, is_dataclass

import cv2


def validate_camera_options(
    *,
    camera_kind="opencv",
    camera_index=0,
    camera_size=(1920, 1080),
    fps=30,
    camera_backend=cv2.CAP_DSHOW,
    device_id=None,
    oak_options=None,
):
    """Validate shared construction options without importing a device SDK."""
    if camera_kind not in {"opencv", "oak"}:
        raise ValueError("camera_kind must be opencv or oak")
    if len(camera_size) != 2 or any(
        not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in camera_size
    ):
        raise ValueError("camera_size must contain two positive integers")
    if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if device_id is not None and (
        not isinstance(device_id, str) or not device_id.strip()
    ):
        raise ValueError("device_id must be a nonempty MXID/IP/USB name or None")
    if oak_options is not None and not isinstance(oak_options, dict):
        raise ValueError("oak_options must be a dictionary")
    if camera_kind != "oak" and (device_id is not None or oak_options):
        raise ValueError("device_id and oak_options require camera_kind=oak")
    options = oak_options or {}
    allowed = {
        "focus",
        "exposure_us",
        "iso",
        "sharpness",
        "keyboard_controls",
        "close_timeout_seconds",
    }
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown OAK options: {sorted(unknown)}")
    for name in ("focus", "exposure_us", "iso", "sharpness"):
        if name in options and (
            not isinstance(options[name], int) or isinstance(options[name], bool)
        ):
            raise ValueError(f"OAK {name} must be an integer")
    for name in ("exposure_us", "iso"):
        if name in options and options[name] <= 0:
            raise ValueError(f"OAK {name} must be positive")
    for name, maximum in (("focus", 255), ("sharpness", 4)):
        if name in options and not 0 <= options[name] <= maximum:
            raise ValueError(f"OAK {name} must be between zero and {maximum}")
    timeout = options.get("close_timeout_seconds", 2)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("OAK close_timeout_seconds must be finite and positive")
    keyboard = options.get("keyboard_controls")
    if keyboard is not None:
        keyboard = asdict(keyboard) if is_dataclass(keyboard) else keyboard
        if not isinstance(keyboard, dict) or set(keyboard) - {
            "focus",
            "exposure_us",
            "iso",
            "sharpness",
        }:
            raise ValueError(
                "keyboard_controls must contain focus/exposure_us/iso/sharpness settings"
            )
        validate_camera_options(
            camera_kind="oak", camera_size=camera_size, fps=fps, oak_options=keyboard
        )


def open_camera(
    *,
    camera_kind="opencv",
    camera_index=0,
    camera_size=(1920, 1080),
    fps=30,
    camera_backend=cv2.CAP_DSHOW,
    device_id=None,
    oak_options=None,
):
    """Open the explicit adapter after shared validation; this operates hardware.

    An OAK ``device_id`` selects a DepthAI v2 DeviceInfo; None retains default
    discovery. Numeric OAK ``camera_index`` remains interface compatibility.
    Sensor options and the separate historical keyboard state are optional.
    """
    validate_camera_options(
        camera_kind=camera_kind,
        camera_index=camera_index,
        camera_size=camera_size,
        fps=fps,
        camera_backend=camera_backend,
        device_id=device_id,
        oak_options=oak_options,
    )
    if camera_kind == "opencv":
        from .opencv import OpenCVCamera

        return OpenCVCamera(
            camera_index,
            width=camera_size[0],
            height=camera_size[1],
            fps=fps,
            backend=camera_backend,
        )
    from .oak import OakCamera, OakKeyboardControls

    options = dict(oak_options or {})
    if isinstance(options.get("keyboard_controls"), dict):
        options["keyboard_controls"] = OakKeyboardControls(
            **options["keyboard_controls"]
        )
    return OakCamera(
        camera_index,
        width=camera_size[0],
        height=camera_size[1],
        fps=fps,
        device_id=device_id,
        **options,
    )
