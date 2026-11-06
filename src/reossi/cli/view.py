"""Optional views of saved identifiers and clouds, without changing data."""

import argparse
import json
from pathlib import Path

import numpy as np

from ..acquisition.recording import read_plan
from .common import print_json


def register_view_commands(commands) -> None:
    for name, description, handler in (
        ("view", "View or save a figure of an exported cloud", view_cloud),
        (
            "view-identifiers",
            "View or save decoded projector-coordinate colors",
            view_identifiers,
        ),
    ):
        command = commands.add_parser(name, help=description)
        command.add_argument("--input", type=Path, required=True)
        command.add_argument(
            "--output",
            type=Path,
            help="Optional figure file; numerical data are unchanged",
        )
        command.add_argument(
            "--show", action=argparse.BooleanOptionalAction, default=True
        )
        command.set_defaults(handler=handler)
        if name == "view":
            command.add_argument(
                "--max-points",
                type=int,
                default=50_000,
                help="Bound displayed points using a deterministic evenly spaced subset",
            )


def view_cloud(args) -> None:
    from ..visualization import plot_point_cloud, require_plotting

    plt = require_plotting()

    if args.max_points < 1:
        raise ValueError("max_points must be positive")
    points = np.load(args.input / "pts_3d.npy", allow_pickle=False)
    colors = np.load(args.input / "colors.npy", allow_pickle=False)
    metadata = json.loads((args.input / "reconstruction.json").read_text())
    original = points.shape[1]
    if original > args.max_points:
        indices = np.linspace(0, original - 1, args.max_points, dtype=int)
        points, colors = points[:, indices], colors[indices]
    figure, _ = plot_point_cloud(
        points,
        colors,
        args.output,
        length_unit=metadata["length_unit"],
        coordinate_frame=metadata["coordinate_frame"],
        show=args.show,
    )
    if not args.show:
        plt.close(figure)
    print_json(
        {
            "stored_points": original,
            "displayed_points": points.shape[1],
            "figure": str(args.output) if args.output is not None else None,
        }
    )


def view_identifiers(args) -> None:
    from ..visualization import (
        create_color_map,
        plot_decoded_identifiers,
        require_plotting,
    )

    plt = require_plotting()

    plan = read_plan(args.input / "scan.json")
    columns = np.load(args.input / "h_pixels.npy", allow_pickle=False)
    rows = np.load(args.input / "v_pixels.npy", allow_pickle=False)
    figure, _ = plot_decoded_identifiers(
        columns,
        rows,
        create_color_map(*plan.projector_size),
        args.output,
        show=args.show,
    )
    if not args.show:
        plt.close(figure)
    print_json(
        {
            "camera_shape": list(columns.shape),
            "figure": str(args.output) if args.output is not None else None,
        }
    )
