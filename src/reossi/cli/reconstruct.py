"""Point-cloud reconstruction from saved calibration and identifier maps."""

from pathlib import Path

from ..reconstruction.workflow import reconstruct_scan
from .common import print_json


def register_reconstruction_commands(commands) -> None:
    command = commands.add_parser(
        "reconstruct", help="Triangulate a saved scan and export PLY"
    )
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--calibration", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--white-image", type=Path)
    command.add_argument(
        "--coordinate-frame",
        choices=("camera", "projector", "legacy"),
        default="camera",
    )
    command.add_argument(
        "--identifier-origin",
        choices=("legacy", "zero", "one"),
        default="legacy",
        help="Explicit projector pixel convention; legacy retains source behavior",
    )
    command.add_argument("--box-threshold", type=float)
    command.add_argument(
        "--max-points",
        type=int,
        default=1_000_000,
        help="Bound valid correspondence count before Python-list allocation",
    )
    command.set_defaults(handler=reconstruct)


def reconstruct(args) -> None:
    result = reconstruct_scan(
        args.input,
        args.calibration,
        args.output,
        white_image=args.white_image,
        coordinate_frame=args.coordinate_frame,
        identifier_origin={"legacy": None, "zero": 0, "one": 1}[args.identifier_origin],
        box_threshold=args.box_threshold,
        max_points=args.max_points,
    )
    print_json(result.metadata)
