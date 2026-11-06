"""Digital raster and triangle-scene scans, without physical devices."""

from math import isfinite
from pathlib import Path

import numpy as np

from ..acquisition.capture import check_sequence_size
from ..acquisition.recording import read_plan
from ..plan import ScanPlan
from .common import image_size, print_json, read_json, require_new_output


def register_simulation_commands(commands) -> None:
    identifiers = commands.add_parser(
        "simulate-identifiers",
        help="Run direct identifier-map shifts, scaling and fusion",
    )
    identifiers.add_argument("--config", type=Path)
    identifiers.add_argument("--output", type=Path, required=True)
    identifiers.set_defaults(handler=simulate_identifiers)
    raster = commands.add_parser(
        "simulate", help="Run a small rectangular digital scan"
    )
    raster.add_argument("--config", type=Path)
    raster.add_argument("--output", type=Path, required=True)
    raster.add_argument("--camera-size", type=image_size, default=(48, 32))
    raster.add_argument("--noise-std", type=float, default=0)
    raster.add_argument("--seed", type=int, default=0)
    raster.add_argument("--max-sequence-mib", type=float, default=64)
    raster.set_defaults(handler=simulate_raster)
    scene = commands.add_parser(
        "simulate-scene", help="Project Gray patterns onto an actual triangle mesh"
    )
    scene.add_argument("--config", type=Path, required=True, help="Scan plan JSON")
    scene.add_argument(
        "--scene-config",
        type=Path,
        required=True,
        help="Virtual devices, surface and sensor JSON",
    )
    scene.add_argument(
        "--mesh",
        type=Path,
        help="Replace the configured generated surface with a mesh file",
    )
    scene.add_argument("--output", type=Path, required=True)
    scene.add_argument("--save-captures", action="store_true")
    scene.add_argument("--seed", type=int)
    scene.add_argument("--noise-std", type=float)
    scene.add_argument(
        "--pattern-placement",
        choices=("emulated_cells", "native_pixels"),
        default="emulated_cells",
        help="Retain source cell shifts, or shift the enlarged raster by native pixels",
    )
    scene.add_argument("--max-sequence-mib", type=float, default=64)
    scene.add_argument("--max-triangle-tests", type=int, default=20_000_000)
    scene.set_defaults(handler=simulate_scene)


def simulate_identifiers(args) -> None:
    import json

    from ..simulation.identifiers import simulate_identifier_map

    output = require_new_output(args.output)
    settings = read_json(args.config) if args.config is not None else {}
    result = simulate_identifier_map(**settings)
    output.mkdir(parents=True)
    np.save(output / "base.npy", result.base)
    np.save(output / "layers.npy", np.stack(result.layers))
    np.save(output / "scaled_layers.npy", np.stack(result.scaled_layers))
    np.save(output / "fused.npy", result.fused)
    (output / "identifier-simulation.json").write_text(
        json.dumps(result.config, indent=2) + "\n"
    )
    print_json(
        {
            "shape": list(result.fused.shape),
            "positions": len(result.layers),
            "output": str(output),
        }
    )


def simulate_raster(args) -> None:
    from ..simulation.raster import save_synthetic_result, simulate_scan

    plan = (
        read_plan(args.config)
        if args.config is not None
        else ScanPlan(projector_size=(48, 32), emulated_size=(12, 8), repeats=2)
    )
    output = require_new_output(args.output)
    check_sequence_size(
        plan.sequence_length,
        args.camera_size,
        itemsize=8,
        max_sequence_mib=args.max_sequence_mib,
    )
    result = simulate_scan(
        plan,
        camera_size=args.camera_size,
        noise_std=args.noise_std,
        seed=args.seed,
        max_sequence_mib=args.max_sequence_mib,
    )
    save_synthetic_result(output, result, plan)
    print_json(result.metrics)


def _device(settings: dict, default_size):
    from ..simulation.geometry import PinholeDevice, look_at_rotation, make_device

    size = tuple(settings.get("image_size", default_size))
    center = np.array(settings.get("center", (0, 0, 0)), dtype=float)
    if "matrix" in settings:
        matrix = np.asarray(settings["matrix"], dtype=float)
    else:
        matrix = make_device(size, focal_length=settings.get("focal_length")).matrix
    if "rotation" in settings and "target" in settings:
        raise ValueError(
            "Use either a rotation or a look-at target for a virtual device"
        )
    rotation = (
        look_at_rotation(center, np.array(settings["target"], dtype=float))
        if "target" in settings
        else np.array(settings.get("rotation", np.eye(3)), dtype=float)
    )
    return PinholeDevice(size, matrix, rotation, -rotation @ center)


def simulate_scene(args) -> None:
    from ..simulation.geometry import load_mesh, make_surface
    from ..simulation.scene import SensorModel, save_scene_result, simulate_scene_scan

    plan = read_plan(args.config)
    settings = read_json(args.scene_config)
    output = require_new_output(args.output)
    camera = _device(settings.get("camera", {}), (60, 40))
    projector = _device(settings.get("projector", {}), plan.projector_size)
    if (
        not isfinite(args.max_sequence_mib)
        or args.max_sequence_mib <= 0
        or args.max_triangle_tests < 1
    ):
        raise ValueError("Simulation resource limits must be finite and positive")
    check_sequence_size(
        plan.sequence_length,
        camera.image_size,
        itemsize=8,
        max_sequence_mib=args.max_sequence_mib,
    )
    surface = settings.get("surface", {})
    if args.mesh is None:
        columns, rows = surface.get("columns", 2), surface.get("rows", 2)
        if any(
            not isinstance(v, int) or isinstance(v, bool) or v < 2
            for v in (columns, rows)
        ):
            raise ValueError("Surface rows/columns must be integers >= 2")
        if (
            2
            * camera.image_size[0]
            * camera.image_size[1]
            * 2
            * (columns - 1)
            * (rows - 1)
            > args.max_triangle_tests
        ):
            raise ValueError(
                "Generated scene exceeds ray/triangle budget before mesh allocation"
            )
    mesh = (
        load_mesh(
            args.mesh,
            max_faces=args.max_triangle_tests
            // (2 * camera.image_size[0] * camera.image_size[1]),
        )
        if args.mesh is not None
        else make_surface(**surface)
    )
    sensor_values = dict(settings.get("sensor", {}))
    if args.noise_std is not None:
        sensor_values["noise_std"] = args.noise_std
    result = simulate_scene_scan(
        mesh,
        camera,
        projector,
        plan,
        sensor=SensorModel(**sensor_values),
        seed=args.seed if args.seed is not None else settings.get("seed", 0),
        pattern_placement=args.pattern_placement,
        max_sequence_mib=args.max_sequence_mib,
        max_triangle_tests=args.max_triangle_tests,
        recording_folder=output / "captures" if args.save_captures else None,
        length_unit=settings.get("length_unit", "m"),
    )
    save_scene_result(output, result, plan)
    print_json(result.metrics)
