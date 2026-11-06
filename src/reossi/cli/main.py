"""Compose thin command groups without opening devices at import time."""

import argparse

import cv2

from .calibrate import register_calibration_commands
from .reconstruct import register_reconstruction_commands
from .scan import register_scan_commands
from .simulate import register_simulation_commands
from .view import register_view_commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Structured-light scanning and REOSSI research"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    register_scan_commands(commands)
    register_calibration_commands(commands)
    register_reconstruction_commands(commands)
    register_simulation_commands(commands)
    register_view_commands(commands)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one explicit workflow stage, reporting input errors concisely."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ValueError, OSError, KeyError, ImportError, cv2.error) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        parser.exit(130, "Operation interrupted; completed recordings were retained.\n")
    return 0
