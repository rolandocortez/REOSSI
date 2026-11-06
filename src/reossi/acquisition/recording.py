"""Read explicit legacy NPZ layouts and save maps without changing their dtype."""

import json
import math
from collections.abc import Iterator
from math import prod
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from ..pipeline import IdentifierMaps
from ..plan import ScanPlan


def read_plan(path: str | Path) -> ScanPlan:
    """Read a configuration file or the ``plan`` object in scan metadata."""
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object")
    return ScanPlan.from_dict(values.get("plan", values))


def load_image_sequence(path: Path, *, max_sequence_mib: float = 1024) -> np.ndarray:
    """Load the ``images`` entry after checking its uncompressed entry size.

    The limit applies to the sequence alone, not peak decoder memory. No dtype
    conversion is performed. Object arrays are not accepted.
    """
    if not math.isfinite(max_sequence_mib) or max_sequence_mib <= 0:
        raise ValueError("max_sequence_mib must be finite and positive")
    with ZipFile(path) as archive:
        entry = archive.getinfo("images.npy")
        if entry.file_size > max_sequence_mib * 1024**2:
            raise ValueError(
                f"Sequence exceeds the {max_sequence_mib:g} MiB limit: {path}"
            )
        with archive.open(entry) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else:
                raise ValueError("Recording requires NPY format version 1 or 2")
            header_bytes = stream.tell()
        if (
            len(shape) != 3
            or min(shape[1:]) < 1
            or not 14 <= shape[0] <= 66
            or (shape[0] - 2) % 4
            or dtype.kind not in "uif"
        ):
            raise ValueError(
                "Recording must contain a numeric (4*bits+2,H,W) Gray sequence"
            )
        declared_bytes = prod(shape) * dtype.itemsize
        if (
            declared_bytes > max_sequence_mib * 1024**2
            or declared_bytes > entry.file_size - header_bytes
        ):
            raise ValueError(
                "Declared recording array exceeds its size limit or available bytes"
            )
    with np.load(path, allow_pickle=False) as data:
        return data["images"]


def iter_recorded_sequences(
    folder: str | Path,
    plan: ScanPlan,
    *,
    layout: str = "runs",
    max_sequence_mib: float = 1024,
) -> Iterator[Iterator[np.ndarray]]:
    """Yield captures by displacement, then run, using an explicit file layout.

    ``runs`` reads ``run_R_displacement_D.npz``. ``single`` reads
    ``displacement_D.npz`` and requires repeats=1; missing repeated captures
    are never manufactured or reused.
    """
    folder = Path(folder)
    if layout not in {"runs", "single"}:
        raise ValueError(f"Unknown recording layout: {layout}")
    if layout == "single" and plan.repeats != 1:
        raise ValueError("The single layout requires an explicit repeats=1 plan")
    for displacement in range(plan.displacement_count):
        paths = [
            folder
            / (
                f"displacement_{displacement}.npz"
                if layout == "single"
                else f"run_{run}_displacement_{displacement}.npz"
            )
            for run in range(plan.repeats)
        ]
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing captured sequences in {folder}: {missing}"
            )
        yield (
            load_image_sequence(path, max_sequence_mib=max_sequence_mib)
            for path in paths
        )


def save_identifier_maps(
    folder: str | Path, maps: IdentifierMaps, plan: ScanPlan
) -> None:
    """Save the historical H/V filenames plus explicit processing parameters."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "h_pixels.npy", maps.columns)
    np.save(folder / "v_pixels.npy", maps.rows)
    metadata = {"format_version": 1, "plan": plan.to_dict()}
    (folder / "scan.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
