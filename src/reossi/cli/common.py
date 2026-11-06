"""Small shared CLI parsing helpers; numerical validation belongs to the API."""

import argparse
import json
from pathlib import Path


def image_size(value: str) -> tuple[int, int]:
    """Parse WIDTHxHEIGHT with a useful argparse error."""
    try:
        width, height = (int(part) for part in value.lower().split("x"))
        if min(width, height) < 1:
            raise ValueError
        return width, height
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Use positive WIDTHxHEIGHT, e.g. 1920x1080"
        ) from error


def read_json(path: str | Path) -> dict:
    """Read a JSON object; lists/scalars are invalid configuration documents."""
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object")
    return values


def require_new_output(path: str | Path) -> Path:
    """Protect existing research outputs before running work."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    return path


def print_json(values: dict) -> None:
    print(json.dumps(values, indent=2, allow_nan=False))
