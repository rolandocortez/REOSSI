"""Legacy REOSSI identifier transformation and layer fusion."""

# REOSSI operations derived from Rolando Cortez's legacy implementation.
# Shared structured-light provenance is documented in THIRD_PARTY_NOTICES.md.

from collections.abc import Sequence
from math import isfinite
from numbers import Real

import numpy as np


def scale_identifiers(
    array: np.ndarray,
    scale_factor: int | float,
    displacement: int,
    step: int | float = 1,
    *,
    invalid_values: tuple[int | float, ...] = (-1,),
) -> np.ndarray:
    """Transform identifiers in place and return the same array.

    The legacy formula is ``scale_factor * (value - 1) + step * displacement
    + 1``. By default exactly ``-1`` is preserved, as in the scanner. An empty
    ``invalid_values`` tuple reproduces the identifier-row experiment's
    transform of every value; ``(-1,0)`` selects the separate zero-preserving
    convention. These are explicit numerical choices, not cleanup changes.
    Assignment retains the input dtype and its casting behavior.
    """
    if not isinstance(invalid_values, tuple) or any(
        not isinstance(v, Real) or isinstance(v, bool) or not isfinite(v)
        for v in invalid_values
    ):
        raise ValueError("invalid_values must be a tuple of finite numeric sentinels")
    valid = np.ones(array.shape, dtype=bool)
    for value in invalid_values:
        valid &= array != value
    array[valid] = scale_factor * (array[valid] - 1) + step * displacement + 1
    return array


def fuse_identifier_maps(
    column_layers: Sequence[np.ndarray], row_layers: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse scaled ``(H, W)`` layers by the legacy maximum at each pixel.

    Row layers use stack/reduction axis zero and column layers axis one,
    preserving the original operations and output dtypes.
    """
    stacked_rows = np.stack(row_layers, axis=0)
    row_identifiers = np.max(stacked_rows, axis=0)
    stacked_columns = np.stack(column_layers, axis=1)
    column_identifiers = np.max(stacked_columns, axis=1)
    return column_identifiers, row_identifiers
