# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Rolando Cortez Garcia
"""Compose shared Gray decoding with the historical REOSSI operations."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .plan import ScanPlan
from .reconstruction.identifiers import fuse_identifier_maps, scale_identifiers
from .structured_light.decoding import (
    aggregate_repeated_bits,
    decode_identifier_maps,
    decode_sequence,
)
from .structured_light.illumination import IlluminationEstimator
from .structured_light.patterns import (
    generate_gray_codewords,
    render_gray_patterns,
    shift_native_patterns,
)

PatternPlacement = Literal["emulated_cells", "native_pixels"]


@dataclass
class IdentifierMaps:
    """Projector column/row identifiers indexed by camera pixel ``[y, x]``."""

    columns: np.ndarray
    rows: np.ndarray


def pattern_raster_size(
    plan: ScanPlan, pattern_placement: PatternPlacement = "emulated_cells"
) -> tuple[int, int]:
    """Resolve the effective raster and validate placement before allocation.

    Native-pixel placement currently covers the chapter-8 case whose target
    raster is the projector raster. Equal integer enlargement on both axes
    keeps its physical step distinct from a displacement of entire LD cells.
    """
    if not isinstance(pattern_placement, str) or pattern_placement not in {
        "emulated_cells",
        "native_pixels",
    }:
        raise ValueError("pattern_placement must be emulated_cells or native_pixels")
    if pattern_placement == "emulated_cells":
        return plan.pattern_size
    if plan.target_size is not None:
        raise ValueError(
            "Native-pixel placement does not support an intermediate target"
        )
    native_width, native_height = plan.projector_size
    width, height = plan.pattern_size
    if (
        native_width % width
        or native_height % height
        or native_width // width != native_height // height
    ):
        raise ValueError("Native-pixel placement requires equal integer scale factors")
    return plan.projector_size


def generate_pattern_sequence(
    plan: ScanPlan,
    displacement: int = 0,
    *,
    pattern_placement: PatternPlacement = "emulated_cells",
) -> np.ndarray:
    """Generate the selected placement while retaining the source default.

    ``emulated_cells`` preserves the historical cell roll before rasterizing.
    ``native_pixels`` enlarges an unshifted LD raster, then translates its bit
    frames right/down by the native-pixel displacement. References are fixed.
    """
    pattern_raster_size(plan, pattern_placement)
    if not 0 <= displacement < plan.displacement_count:
        raise ValueError("Displacement is outside this scan plan")
    width, height = plan.pattern_size
    words = generate_gray_codewords(
        width,
        height,
        identifier_origin=plan.identifier_origin,
        displacement=displacement if pattern_placement == "emulated_cells" else 0,
    )
    patterns = render_gray_patterns(words, width, height)
    if pattern_placement == "native_pixels" and plan.mode != "conventional":
        return shift_native_patterns(patterns, plan.projector_size, displacement)
    return patterns


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
