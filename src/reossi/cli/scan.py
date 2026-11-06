"""Thin commands for plans, pattern export, capture and replay."""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from ..acquisition.capture import check_sequence_size, run_capture
from ..acquisition.recording import (
    iter_recorded_sequences,
    read_plan,
    save_identifier_maps,
)
from ..pipeline import (
    PatternPlacement,
    generate_pattern_sequence,
    pattern_raster_size,
    reconstruct_identifier_maps,
)
from .common import image_size, print_json, read_json, require_new_output


def register_scan_commands(commands) -> None:
    for name, description, handler in (
        ("plan", "Inspect parameters without generating images", inspect_plan),
        ("patterns", "Export Gray sequences without hardware", export_patterns),
        ("replay", "Decode saved NPZ capture sequences", replay),
        ("capture", "Capture one view with a physical camera/projector", capture),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--config", type=Path, required=True)
        command.set_defaults(handler=handler)
        if name != "replay":
            command.add_argument(
                "--pattern-placement",
                choices=("emulated_cells", "native_pixels"),
                default=None,
                help="Override acquisition.pattern_placement; source cells by default",
            )
        if name != "plan":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument(
                "--max-sequence-mib",
                type=float,
                default=1024,
                help="Bound one stack, not total decoder memory",
            )
        if name == "replay":
            command.add_argument("--input", type=Path, required=True)
            command.add_argument("--layout", choices=("runs", "single"), default="runs")
        if name == "capture":
            add_camera_arguments(command)
            command.add_argument("--settle-seconds", type=float)
            command.add_argument(
                "--flip", action=argparse.BooleanOptionalAction, default=None
            )
            command.add_argument("--window-x", type=int)
            command.add_argument("--window-y", type=int)
            command.add_argument(
                "--fullscreen", action=argparse.BooleanOptionalAction, default=None
            )
            command.add_argument("--frame-timeout-seconds", type=float)


def add_camera_arguments(command) -> None:
    """Add overrides shared by scan and calibration acquisition commands."""
    command.add_argument("--camera", choices=("opencv", "oak"))
    command.add_argument("--camera-index", type=int)
    command.add_argument("--camera-size", type=image_size)
    command.add_argument("--camera-backend", choices=("dshow", "auto", "v4l2"))
    command.add_argument("--fps", type=float)
    command.add_argument(
        "--device-id", help="OAK MXID/IP/USB name; optional SDK selection"
    )
    for name in ("focus", "exposure-us", "iso", "sharpness"):
        command.add_argument(f"--{name}", type=int, help="OAK initial sensor control")


def camera_options(args, settings: dict) -> dict:
    """Resolve camera overrides without importing or opening a camera."""

    if not isinstance(settings, dict):
        raise ValueError("acquisition must be a JSON object")

    def option(attribute, key, default):
        value = getattr(args, attribute, None)
        return settings.get(key, default) if value is None else value

    kind = option("camera", "camera", "opencv")
    backend = option("camera_backend", "camera_backend", "dshow")
    if backend not in {"dshow", "auto", "v4l2"}:
        raise ValueError("camera_backend must be dshow, auto, or v4l2")
    default_size = (3840, 2160) if kind == "oak" else (1920, 1080)
    oak = dict(settings.get("oak_options") or {})
    for name in ("focus", "exposure_us", "iso", "sharpness"):
        value = getattr(args, name, None)
        if value is not None:
            oak[name] = value
    return {
        "camera_kind": kind,
        "camera_index": option("camera_index", "camera_index", 0),
        "camera_size": tuple(option("camera_size", "camera_size", default_size)),
        "fps": option("fps", "fps", 30),
        "device_id": option("device_id", "device_id", None),
        "oak_options": oak or None,
        "camera_backend": {
            "dshow": cv2.CAP_DSHOW,
            "auto": cv2.CAP_ANY,
            "v4l2": cv2.CAP_V4L2,
        }[backend],
    }


def inspect_plan(args) -> None:
    plan = read_plan(args.config)
    settings = read_json(args.config).get("acquisition", {})
    placement = placement_option(args, settings)
    raster_size = pattern_raster_size(plan, placement)
    notes = {"compatibility_notes": plan.compatibility_notes}
    if placement == "native_pixels":
        # Encoding limits still affect this profile; only cell placement differs.
        cell_note = "Displacements roll whole LD identifier cells before rendering"
        notes = {
            "compatibility_notes": [
                note for note in plan.compatibility_notes if note != cell_note
            ],
            "source_compatibility_notes": [
                note for note in plan.compatibility_notes if note == cell_note
            ],
        }
    print_json(
        plan.to_dict()
        | {
            "pattern_size": plan.pattern_size,
            "pattern_placement": placement,
            "pattern_raster_size": raster_size,
            "scale_factor": plan.scale_factor,
            "native_step": plan.native_step,
            "displacement_count": plan.displacement_count,
            "sequence_length": plan.sequence_length,
            "sequence_count": plan.displacement_count * plan.repeats,
            "frame_count": plan.displacement_count
            * plan.repeats
            * plan.sequence_length,
            "normalization_epsilon": plan.normalization_epsilon,
        }
        | notes
    )


def export_patterns(args) -> None:
    plan = read_plan(args.config)
    settings = read_json(args.config).get("acquisition", {})
    placement = placement_option(args, settings)
    raster_size = pattern_raster_size(plan, placement)
    check_sequence_size(
        plan.sequence_length,
        raster_size,
        itemsize=1,
        max_sequence_mib=args.max_sequence_mib,
    )
    output = require_new_output(args.output)
    output.mkdir(parents=True)
    for displacement in range(plan.displacement_count):
        np.savez_compressed(
            output / f"patterns_displacement_{displacement}.npz",
            images=generate_pattern_sequence(
                plan, displacement, pattern_placement=placement
            ),
        )
    (output / "plan.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "plan": plan.to_dict(),
                "acquisition": settings | {"pattern_placement": placement},
                "pattern_raster_size": list(raster_size),
            },
            indent=2,
        )
        + "\n"
    )


def placement_option(args, settings: dict) -> PatternPlacement:
    """Resolve the explicit CLI override before the acquisition configuration."""
    if not isinstance(settings, dict):
        raise ValueError("acquisition must be a JSON object")
    value = getattr(args, "pattern_placement", None)
    return (
        settings.get("pattern_placement", "emulated_cells") if value is None else value
    )


def replay(args) -> None:
    plan = read_plan(args.config)
    settings = read_json(args.config).get("acquisition", {})
    if not isinstance(settings, dict):
        raise ValueError("acquisition must be a JSON object")
    placement = settings.get("pattern_placement")
    if placement is not None:
        pattern_raster_size(plan, placement)
    output = require_new_output(args.output)
    sequences = iter_recorded_sequences(
        args.input, plan, layout=args.layout, max_sequence_mib=args.max_sequence_mib
    )
    save_identifier_maps(
        output,
        reconstruct_identifier_maps(sequences, plan),
        plan,
        pattern_placement=placement,
    )
    white = args.input / "white.png"
    if white.is_file():
        shutil.copy2(white, output / "white.png")


def capture(args) -> None:
    plan = read_plan(args.config)
    settings = read_json(args.config).get("acquisition", {})
    options = camera_options(args, settings)
    options["pattern_placement"] = placement_option(args, settings)
    for attribute, default in (
        ("settle_seconds", None),
        ("flip", None),
        ("fullscreen", True),
        ("frame_timeout_seconds", 30),
    ):
        value = getattr(args, attribute)
        options[attribute] = (
            settings.get(attribute, default) if value is None else value
        )
    if (args.window_x is None) != (args.window_y is None):
        raise ValueError("Specify both --window-x and --window-y")
    options["window_position"] = (
        (args.window_x, args.window_y)
        if args.window_x is not None
        else settings.get("window_position")
    )
    run_capture(
        plan,
        require_new_output(args.output),
        max_sequence_mib=args.max_sequence_mib,
        **options,
    )
