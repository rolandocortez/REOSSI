"""Explicit scan profiles and recording contracts, using tiny local arrays."""

import json
from io import BytesIO
from zipfile import ZipFile

import numpy as np
import pytest

from reossi.acquisition.recording import (
    iter_recorded_sequences,
    load_image_sequence,
    read_plan,
)
from reossi.plan import ScanPlan


def test_native_emulated_and_target_sizes_have_distinct_historical_roles():
    basic = ScanPlan(projector_size=(1920, 1080), emulated_size=(320, 180))
    assert (basic.scale_factor, basic.native_step, basic.displacement_count) == (
        6,
        1,
        6,
    )
    assert basic.sequence_length == 38
    general = ScanPlan(
        projector_size=(1920, 1080), target_size=(960, 540), emulated_size=(480, 270)
    )
    assert (general.scale_factor, general.native_step, general.displacement_count) == (
        2,
        2,
        2,
    )
    assert general.sequence_length == 38
    larger = ScanPlan(projector_size=(1920, 1080), emulated_size=(640, 360))
    assert larger.sequence_length == 42


def test_nondivisible_and_power_of_two_choices_are_visible():
    plan = ScanPlan(projector_size=(25, 15), emulated_size=(8, 4))
    notes = " ".join(plan.compatibility_notes)
    assert "truncated" in notes
    assert "aspect ratios" in notes
    assert "final one-based identifier" in notes
    assert plan.scale_factor == 3  # Original truncation is retained.


@pytest.mark.parametrize(
    "values",
    [
        {"emulated_size": (4, 8)},
        {"emulated_size": (4, 2)},
        {"emulated_size": (8.5, 4)},
        {"repeats": 0},
        {"repeats": 1.5},
        {"contrast_margin": -1},
        {"illumination_epsilon": float("nan")},
    ],
)
def test_unsupported_configuration_fails_before_array_allocation(values):
    with pytest.raises(ValueError):
        ScanPlan(**values)


def test_single_layout_requires_explicit_one_run(tmp_path):
    with pytest.raises(ValueError, match="repeats=1"):
        list(iter_recorded_sequences(tmp_path, ScanPlan(), layout="single"))


def test_missing_run_is_not_reused_or_manufactured(tmp_path):
    plan = ScanPlan(mode="conventional", projector_size=(7, 4), repeats=2)
    np.savez(tmp_path / "run_0_displacement_0.npz", images=np.zeros((14, 2, 3)))
    with pytest.raises(FileNotFoundError, match="run_1_displacement_0"):
        list(iter_recorded_sequences(tmp_path, plan))


def test_npz_load_preserves_dtype_and_checks_uncompressed_size(tmp_path):
    path = tmp_path / "sequence.npz"
    images = np.zeros((14, 2, 3), dtype=np.uint8)
    np.savez_compressed(path, images=images)
    restored = load_image_sequence(path)
    assert restored.dtype == np.uint8
    np.testing.assert_array_equal(restored, images)
    with pytest.raises(ValueError, match="exceeds"):
        load_image_sequence(path, max_sequence_mib=0.00001)


def test_metadata_round_trip_keeps_processing_parameters(tmp_path):
    plan = ScanPlan(illumination_epsilon=0.003, contrast_margin=2, minimum_direct=20)
    path = tmp_path / "scan.json"
    path.write_text(json.dumps({"format_version": 1, "plan": plan.to_dict()}))
    assert read_plan(path) == plan
    assert read_plan(path).normalization_epsilon == 0.003


def test_recording_header_is_checked_before_numpy_allocation(tmp_path, monkeypatch):
    path = tmp_path / "declared.npz"
    stream = BytesIO()
    np.lib.format.write_array_header_1_0(
        stream, {"descr": "<f8", "fortran_order": False, "shape": (14, 10**9, 10**9)}
    )
    with ZipFile(path, "w") as archive:
        archive.writestr("images.npy", stream.getvalue())
    monkeypatch.setattr(
        np,
        "load",
        lambda *a, **kw: pytest.fail("Header guard should precede array loading"),
    )
    with pytest.raises(ValueError, match="Declared recording"):
        load_image_sequence(path)


def test_recording_rejects_incomplete_gray_sequence_before_loading(tmp_path):
    path = tmp_path / "incomplete.npz"
    np.savez_compressed(path, images=np.zeros((15, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="Gray sequence"):
        load_image_sequence(path)
