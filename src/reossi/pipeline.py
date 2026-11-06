"""Compose shared Gray decoding with the historical REOSSI operations."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .plan import ScanPlan
from .reconstruction.identifiers import fuse_identifier_maps, scale_identifiers
from .structured_light.decoding import (
    aggregate_repeated_bits,
    decode_identifier_maps,
    decode_sequence,
)
from .structured_light.illumination import IlluminationEstimator
from .structured_light.patterns import generate_gray_codewords, render_gray_patterns


@dataclass
class IdentifierMaps:
    """Projector column/row identifiers indexed by camera pixel ``[y, x]``."""

    columns: np.ndarray
    rows: np.ndarray


def generate_pattern_sequence(plan: ScanPlan, displacement: int = 0) -> np.ndarray:
    """Generate the same LD/native-sized sequence as the chosen source path.

    This intentionally retains the historical shift of identifier cells before
    rasterization. It does not introduce native-resolution subcell shifting.
    """
    if not 0 <= displacement < plan.displacement_count:
        raise ValueError("Displacement is outside this scan plan")
    width, height = plan.pattern_size
    words = generate_gray_codewords(
        width,
        height,
        identifier_origin=plan.identifier_origin,
        displacement=displacement,
    )
    return render_gray_patterns(words, width, height)


def decode_displacement(
    sequences: Iterable[np.ndarray],
    plan: ScanPlan,
    displacement: int,
    *,
    illumination_estimator: IlluminationEstimator | None = None,
) -> IdentifierMaps:
    """Decode repeated ``(N, H, W)`` captures and transform one position.

    Input dtypes are preserved. In particular, uint8 replay is not silently
    converted to the float64 buffer used by the original live capture.
    """
    if not 0 <= displacement < plan.displacement_count:
        raise ValueError("Displacement is outside this scan plan")
    column_runs = []
    row_runs = []
    for run, images in enumerate(sequences):
        if run >= plan.repeats:
            raise ValueError(
                f"Expected {plan.repeats} repeated sequences; found an extra run"
            )
        if (
            images.ndim != 3
            or len(images) != plan.sequence_length
            or min(images.shape[1:]) < 1
        ):
            raise ValueError(
                f"Expected ({plan.sequence_length}, H, W) images, got {images.shape}"
            )
        columns, rows = decode_sequence(
            images,
            normalization_epsilon=plan.normalization_epsilon,
            contrast_margin=plan.contrast_margin,
            minimum_direct=plan.minimum_direct,
            illumination_estimator=illumination_estimator,
        )
        column_runs.append(columns)
        row_runs.append(rows)
        del images
    if len(column_runs) != plan.repeats:
        raise ValueError(
            f"Expected {plan.repeats} repeated sequences, got {len(column_runs)}. "
            "Select repeats=1 explicitly for a recording that retained only one run."
        )
    column_bits, row_bits = aggregate_repeated_bits(column_runs, row_runs)
    columns, rows = decode_identifier_maps(column_bits, row_bits)
    if plan.mode != "conventional":
        columns = scale_identifiers(
            columns, plan.scale_factor, displacement, plan.native_step
        )
        rows = scale_identifiers(
            rows, plan.scale_factor, displacement, plan.native_step
        )
    return IdentifierMaps(columns, rows)


def reconstruct_identifier_maps(
    displacement_sequences: Iterable[Iterable[np.ndarray]],
    plan: ScanPlan,
    *,
    illumination_estimator: IlluminationEstimator | None = None,
) -> IdentifierMaps:
    """Process positions in order, then perform the source's per-pixel maximum.

    The historical stack/reduction operations remain unchanged. No streaming
    maximum, dtype promotion, classifier replacement, or identifier correction
    is introduced in this equivalence-focused extraction.
    """
    column_layers = []
    row_layers = []
    for displacement, sequences in enumerate(displacement_sequences):
        if displacement >= plan.displacement_count:
            raise ValueError("Recording contains more displacements than the plan")
        maps = decode_displacement(
            sequences, plan, displacement, illumination_estimator=illumination_estimator
        )
        column_layers.append(maps.columns)
        row_layers.append(maps.rows)
    if len(column_layers) != plan.displacement_count:
        raise ValueError(
            f"Expected {plan.displacement_count} positions, got {len(column_layers)}"
        )
    columns, rows = fuse_identifier_maps(column_layers, row_layers)
    if plan.target_size is not None:
        columns = scale_identifiers(columns, plan.native_step, 0, 1)
        rows = scale_identifiers(rows, plan.native_step, 0, 1)
    return IdentifierMaps(columns, rows)
