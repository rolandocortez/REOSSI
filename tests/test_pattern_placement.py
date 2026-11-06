# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Rolando Cortez Garcia
"""Explicit native-pixel placement, with retained source-cell defaults."""

import json
from dataclasses import replace
from math import prod

import numpy as np
import pytest

from reossi import pipeline
from reossi.acquisition.recording import iter_recorded_sequences
from reossi.pipeline import generate_pattern_sequence, reconstruct_identifier_maps
from reossi.plan import ScanPlan
from reossi.simulation import scene
from reossi.simulation.geometry import make_device, make_surface
from reossi.simulation.scene import save_scene_result, simulate_scene_scan
from reossi.structured_light.patterns import (
    generate_gray_codewords,
    render_gray_patterns,
)


def placement_plan():
    return ScanPlan(projector_size=(28, 16), emulated_size=(7, 4), repeats=1)


@pytest.mark.parametrize("mode", ["conventional", "ld", "reossi"])
def test_default_placement_retains_source_cell_raster(mode):
    plan = replace(placement_plan(), mode=mode)
    width, height = plan.pattern_size
    for displacement in range(plan.displacement_count):
        source = render_gray_patterns(
            generate_gray_codewords(
                width,
                height,
                identifier_origin=plan.identifier_origin,
                displacement=displacement,
            ),
            width,
            height,
        )
        np.testing.assert_array_equal(
            generate_pattern_sequence(plan, displacement), source
        )
        np.testing.assert_array_equal(
            generate_pattern_sequence(
                plan, displacement, pattern_placement="emulated_cells"
            ),
            source,
        )


def test_native_zero_displacement_is_exact_nearest_neighbor_enlargement():
    plan = placement_plan()
    source = generate_pattern_sequence(plan)
    native = generate_pattern_sequence(plan, pattern_placement="native_pixels")
    np.testing.assert_array_equal(
        native, np.repeat(np.repeat(source, 4, axis=1), 4, axis=2)
    )
    assert native.dtype == np.uint8
    assert native.shape == (plan.sequence_length, 16, 28)


def test_native_bit_edges_move_one_pixel_not_one_four_pixel_cell():
    plan = placement_plan()
    baseline = generate_pattern_sequence(plan, pattern_placement="native_pixels")
    shifted = generate_pattern_sequence(plan, 1, pattern_placement="native_pixels")
    whole_cell = np.repeat(
        np.repeat(generate_pattern_sequence(plan, 1), 4, axis=1), 4, axis=2
    )
    # The last column bit changes at x=8 in the unshifted one-based raster.
    assert baseline[6, 0, 8] == 0
    assert shifted[6, 0, 8] == 255
    assert shifted[6, 0, 9] == 0
    assert whole_cell[6, 0, 11] == 255
    assert whole_cell[6, 0, 12] == 0
    # The row-bit transition follows the same native step in the other axis.
    assert baseline[3, 8, 0] == 0
    assert shifted[3, 8, 0] == 255
    assert shifted[3, 9, 0] == 0


@pytest.mark.parametrize("displacement", [1, 2, 3])
def test_native_references_complements_and_gray_zero_prefix(displacement):
    plan = placement_plan()
    original = generate_pattern_sequence(plan, pattern_placement="native_pixels")
    shifted = generate_pattern_sequence(
        plan, displacement, pattern_placement="native_pixels"
    )
    inverse_start = 2 + (plan.sequence_length - 2) // 2
    np.testing.assert_array_equal(shifted[:2], original[:2])
    np.testing.assert_array_equal(
        shifted[inverse_start:], 255 - shifted[2:inverse_start]
    )
    for frame_index in range(2, inverse_start):
        if frame_index % 2 == 0:
            np.testing.assert_array_equal(
                shifted[frame_index, :, displacement:],
                original[frame_index, :, :-displacement],
            )
            assert not shifted[frame_index, :, :displacement].any()
        else:
            np.testing.assert_array_equal(
                shifted[frame_index, displacement:, :],
                original[frame_index, :-displacement, :],
            )
            assert not shifted[frame_index, :displacement, :].any()


def test_conventional_raster_is_identical_under_both_placement_options():
    plan = replace(placement_plan(), mode="conventional")
    np.testing.assert_array_equal(
        generate_pattern_sequence(plan),
        generate_pattern_sequence(plan, pattern_placement="native_pixels"),
    )


@pytest.mark.parametrize(
    "plan,placement,error",
    [
        (placement_plan(), "unknown", "pattern_placement"),
        (
            replace(placement_plan(), target_size=(14, 8)),
            "native_pixels",
            "intermediate target",
        ),
        (
            replace(placement_plan(), projector_size=(29, 16)),
            "native_pixels",
            "equal integer",
        ),
        (
            replace(placement_plan(), projector_size=(28, 12)),
            "native_pixels",
            "equal integer",
        ),
    ],
)
def test_unsupported_placement_fails_before_patterns_and_rays(
    monkeypatch, plan, placement, error
):
    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported placement reached pattern allocation or ray tracing")

    monkeypatch.setattr(pipeline, "generate_gray_codewords", unexpected)
    monkeypatch.setattr(scene, "scene_correspondences", unexpected)
    with pytest.raises(ValueError, match=error):
        generate_pattern_sequence(plan, pattern_placement=placement)
    with pytest.raises(ValueError, match=error):
        simulate_scene_scan(
            make_surface(),
            make_device((8, 6)),
            make_device(plan.projector_size),
            plan,
            pattern_placement=placement,
            max_sequence_mib=8,
        )


@pytest.mark.parametrize("scale", [20, 10**12])
def test_native_raster_budget_is_checked_before_rays_or_patterns(monkeypatch, scale):
    plan = ScanPlan(
        projector_size=(20 * scale, 12 * scale),
        emulated_size=(20, 12),
        repeats=1,
    )
    camera_size = (8, 6)
    maximum_mib = 0.1
    retained_raster_bytes = plan.sequence_length * (
        prod(camera_size) * 8 + prod(plan.pattern_size)
    )
    native_raster_bytes = plan.sequence_length * (
        prod(camera_size) * 8 + prod(plan.projector_size)
    )
    assert retained_raster_bytes < maximum_mib * 1024**2 < native_raster_bytes

    def unexpected(*args, **kwargs):
        pytest.fail("The native raster budget was checked after expensive work")

    monkeypatch.setattr(scene, "scene_correspondences", unexpected)
    monkeypatch.setattr(scene, "generate_pattern_sequence", unexpected)
    with pytest.raises(ValueError, match="max_sequence_mib"):
        simulate_scene_scan(
            make_surface(),
            make_device(camera_size),
            make_device(plan.projector_size),
            plan,
            pattern_placement="native_pixels",
            max_sequence_mib=maximum_mib,
        )


def test_native_reossi_scene_recovers_known_identifiers_and_replays(tmp_path):
    plan = ScanPlan(projector_size=(48, 32), emulated_size=(12, 8), repeats=1)
    result = simulate_scene_scan(
        make_surface(width=4, height=3, depth=2),
        make_device((48, 32)),
        make_device((48, 32), center=(0.125, 0, 0)),
        plan,
        pattern_placement="native_pixels",
        max_sequence_mib=8,
        max_triangle_tests=100_000,
        recording_folder=tmp_path / "captures",
    )
    illuminated = result.geometry.illuminated
    assert illuminated.any()
    np.testing.assert_array_equal(
        result.maps.columns[illuminated], result.true_columns[illuminated]
    )
    np.testing.assert_array_equal(
        result.maps.rows[illuminated], result.true_rows[illuminated]
    )
    assert result.metrics["joint_exact_fraction_of_illuminated"] == 1
    assert result.model["pattern_placement"] == "native_pixels"
    replayed = reconstruct_identifier_maps(
        iter_recorded_sequences(tmp_path / "captures", plan), plan
    )
    np.testing.assert_array_equal(replayed.columns, result.maps.columns)
    np.testing.assert_array_equal(replayed.rows, result.maps.rows)
    save_scene_result(tmp_path / "scan", result, plan)
    saved = json.loads((tmp_path / "scan/simulation.json").read_text())
    assert saved["model"]["pattern_placement"] == "native_pixels"
