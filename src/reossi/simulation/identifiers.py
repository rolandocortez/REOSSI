"""Direct identifier-map demonstrations, independent of image formation.

The curved stripe geometry and shift/scale/max ordering reproduce Rolando's
small cylindrical demonstration. These are synthetic identifiers rather than
rendered images, estimated photometry, or a metric 3D reconstruction.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np

from ..reconstruction.identifiers import scale_identifiers


def _integer(value, name, minimum=0) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _curvature(value) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(value)
        or value < 0
        or value > np.iinfo(np.intp).max
    ):
        raise ValueError(
            "curvature must be finite, nonnegative, and fit pixel coordinates"
        )
    return float(value)


def _map_shape(array, max_pixels):
    max_pixels = _integer(max_pixels, "max_pixels", 1)
    if not isinstance(array, np.ndarray) or array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(
            "Identifier input must be a nonempty two-dimensional NumPy array"
        )
    if array.size > max_pixels:
        raise ValueError(f"Identifier map exceeds max_pixels={max_pixels}")
    if (
        not (
            np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)
        )
        or not np.isfinite(array).all()
    ):
        raise ValueError("Identifier input must contain finite real numeric values")
    return array.shape


def generate_stripe_map(
    stripe_count: int = 4,
    stripe_width: int = 6,
    height: int = 10,
    curvature: float = 0,
    *,
    max_pixels: int = 1_000_000,
) -> np.ndarray:
    """Generate one-based vertical identifier stripes and an optional row warp.

    Output shape is ``(height, stripe_count * stripe_width)`` with signed native
    integer dtype. Curvature is the maximum nominal horizontal pixel shift of
    the source's circular row profile; uncovered leading cells are zero. This
    construction illustrates identifier geometry, not a physical cylinder.
    """
    stripe_count = _integer(stripe_count, "stripe_count", 1)
    stripe_width = _integer(stripe_width, "stripe_width", 1)
    height = _integer(height, "height", 1)
    max_pixels = _integer(max_pixels, "max_pixels", 1)
    curvature = _curvature(curvature)
    width = stripe_count * stripe_width
    if height * width > max_pixels:
        raise ValueError(f"Generated identifier map exceeds max_pixels={max_pixels}")
    if curvature and height < 2:
        raise ValueError("A nonzero curved profile needs height >= 2")
    base_row = np.repeat(np.arange(1, stripe_count + 1, dtype=int), stripe_width)
    result = np.broadcast_to(base_row, (height, width)).copy()
    if curvature:
        radius = height // 2
        for row in range(height):
            displacement = int(
                curvature * np.sqrt(max(radius**2 - (row - radius) ** 2, 0)) / radius
            )
            result[row] = np.roll(result[row], displacement)
            result[row, :displacement] = 0
    return result


def shift_identifier_map(
    array: np.ndarray,
    displacement: int,
    shift_pixels: int = 1,
    axis: int = 1,
    fill_value: int | float = 0,
    *,
    max_pixels: int = 1_000_000,
) -> np.ndarray:
    """Copy and shift a 2D map, replacing wraparound by ``fill_value``.

    The shift is ``displacement * shift_pixels`` along rows (axis0) or columns
    (axis1). Nonnegative shifts only are supported. Dtype and assignment casting
    are retained; choose a signed or floating dtype when using a -1 sentinel.
    """
    _map_shape(array, max_pixels)
    displacement = _integer(displacement, "displacement")
    shift_pixels = _integer(shift_pixels, "shift_pixels")
    axis = _integer(axis, "axis")
    if axis not in {0, 1}:
        raise ValueError("axis must be 0 or 1")
    if (
        not isinstance(fill_value, Real)
        or isinstance(fill_value, bool)
        or not isfinite(fill_value)
    ):
        raise ValueError("fill_value must be a finite real number")
    amount = displacement * shift_pixels
    shifted = np.roll(array, amount, axis=axis)
    leading = [slice(None), slice(None)]
    leading[axis] = slice(0, amount)
    shifted[tuple(leading)] = fill_value
    return shifted


@dataclass
class IdentifierSimulation:
    """Base, independent shifted/scaled layers and their maximum-value fusion."""

    base: np.ndarray
    layers: tuple[np.ndarray, ...]
    scaled_layers: tuple[np.ndarray, ...]
    fused: np.ndarray
    config: dict


def simulate_identifier_map(
    base: np.ndarray | None = None,
    *,
    stripe_count: int = 4,
    stripe_width: int = 6,
    height: int = 10,
    curvature: float = 5,
    scale_factor: int = 3,
    positions: int = 3,
    shift_pixels: int = 2,
    step: int = 1,
    axis: int = 1,
    fill_value: int | float = 0,
    max_pixels: int = 1_000_000,
    max_layer_pixels: int = 8_000_000,
) -> IdentifierSimulation:
    """Shift, transform copies, and fuse a generated or supplied identifier map.

    ``positions`` includes the original position0. The defaults reproduce the
    small historical curved example. Supplied arrays are never modified; each
    shifted layer and transformed layer has independent storage. Exactly -1
    survives the shared transformation; zero and other negatives follow its
    existing formula without a new validity policy.

    ``max_pixels`` bounds one map. ``max_layer_pixels`` bounds a conservative
    ``(3 * positions + 3) * map.size`` working-cell estimate before allocation,
    including retained layers, the fusion stack and temporary/output maps.
    These are cell-count guards, not a guarantee of total process memory.
    """
    scale_factor = _integer(scale_factor, "scale_factor", 1)
    positions = _integer(positions, "positions", 1)
    shift_pixels = _integer(shift_pixels, "shift_pixels")
    step = _integer(step, "step")
    integer_limit = np.iinfo(np.int64).max
    if scale_factor > integer_limit or step * (positions - 1) > integer_limit:
        raise ValueError(
            "scale_factor and displacement offsets must fit signed 64-bit integers"
        )
    axis = _integer(axis, "axis")
    if axis not in {0, 1}:
        raise ValueError("axis must be 0 or 1")
    max_pixels = _integer(max_pixels, "max_pixels", 1)
    max_layer_pixels = _integer(max_layer_pixels, "max_layer_pixels", 1)
    if (
        not isinstance(fill_value, Real)
        or isinstance(fill_value, bool)
        or not isfinite(fill_value)
    ):
        raise ValueError("fill_value must be a finite real number")
    if base is None:
        stripe_count = _integer(stripe_count, "stripe_count", 1)
        stripe_width = _integer(stripe_width, "stripe_width", 1)
        height = _integer(height, "height", 1)
        curvature = _curvature(curvature)
        shape = (height, stripe_count * stripe_width)
        if curvature and height < 2:
            raise ValueError("A nonzero curved profile needs height >= 2")
        input_kind = "generated_stripes"
    else:
        shape = _map_shape(base, max_pixels)
        input_kind = "provided_identifiers"
    pixels = int(shape[0]) * int(shape[1])
    if pixels > max_pixels:
        raise ValueError(f"Identifier map exceeds max_pixels={max_pixels}")
    working_cells = (3 * positions + 3) * pixels
    if working_cells > max_layer_pixels:
        raise ValueError(
            f"Identifier layers exceed max_layer_pixels={max_layer_pixels}"
        )
    if base is None:
        base = generate_stripe_map(
            stripe_count, stripe_width, height, curvature, max_pixels=max_pixels
        )
    else:
        base = base.copy()
    layers = tuple(
        shift_identifier_map(
            base, displacement, shift_pixels, axis, fill_value, max_pixels=max_pixels
        )
        for displacement in range(positions)
    )
    scaled_layers = tuple(
        scale_identifiers(layer.copy(), scale_factor, displacement, step)
        for displacement, layer in enumerate(layers)
    )
    fused = np.max(np.stack(scaled_layers, axis=0), axis=0)
    config = {
        "input": input_kind,
        "shape": list(shape),
        "scale_factor": scale_factor,
        "positions": positions,
        "shift_pixels": shift_pixels,
        "step": step,
        "axis": axis,
        "fill_value": fill_value.item()
        if isinstance(fill_value, np.generic)
        else fill_value,
        "max_pixels": max_pixels,
        "max_layer_pixels": max_layer_pixels,
        "estimated_working_cells": working_cells,
    }
    if input_kind == "generated_stripes":
        config.update(
            stripe_count=stripe_count,
            stripe_width=stripe_width,
            height=height,
            curvature=curvature,
        )
    return IdentifierSimulation(base, layers, scaled_layers, fused, config)
