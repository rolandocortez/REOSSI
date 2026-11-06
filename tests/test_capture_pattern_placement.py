# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Rolando Cortez Garcia
"""Projector placement controls, using mocked acquisition and tiny recordings."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from reossi.acquisition import capture
from reossi.acquisition.recording import read_plan
from reossi.cli import scan
from reossi.cli.main import main
from reossi.plan import ScanPlan


def small_plan():
    return ScanPlan(projector_size=(28, 16), emulated_size=(7, 4), repeats=1)


def write_config(path, *, placement=None, plan=None):
    acquisition = {
        "camera_backend": "auto",
        "camera_size": [28, 16],
        "settle_seconds": 0,
        "flip": False,
        "fullscreen": False,
    }
    if placement is not None:
        acquisition["pattern_placement"] = placement
    path.write_text(
        json.dumps(
            {"plan": (plan or small_plan()).to_dict(), "acquisition": acquisition}
        )
    )
    return path


def direct_captures(patterns):
    """Sample the displayed raster on a tiny 28-by-16 virtual camera."""
    rows = np.arange(16) * patterns.shape[1] // 16
    columns = np.arange(28) * patterns.shape[2] // 28
    return 2 + 180 * patterns[:, rows[:, None], columns[None, :]].astype(float) / 255


@pytest.fixture
def mocked_capture(monkeypatch):
    stop = MagicMock()
    camera = SimpleNamespace(stop_cam=stop)
    opener = MagicMock(return_value=camera)
    acquisitions = []
    monkeypatch.setattr(capture, "open_camera", opener)
    for name in ("namedWindow", "moveWindow", "setWindowProperty", "destroyWindow"):
        monkeypatch.setattr(capture.cv2, name, MagicMock())

    def acquire(selected_camera, patterns, **kwargs):
        assert selected_camera is camera
        acquisitions.append((patterns.copy(), kwargs))
        images = direct_captures(patterns)
        if kwargs["white_path"] is not None:
            white = np.repeat(images[1, ..., None], 3, axis=2).astype(np.uint8)
            assert cv2.imwrite(str(kwargs["white_path"]), white)
        return images

    monkeypatch.setattr(capture, "capture_sequence", acquire)
    return SimpleNamespace(opener=opener, stop=stop, acquisitions=acquisitions)


def test_capture_native_raster_moves_one_pixel_and_records_effective_profile(
    tmp_path, mocked_capture
):
    plan = small_plan()
    output = tmp_path / "native"
    maps = capture.run_capture(
        plan,
        output,
        camera_size=(28, 16),
        settle_seconds=0,
        flip=False,
        fullscreen=False,
        pattern_placement="native_pixels",
        max_sequence_mib=1,
    )
    np.testing.assert_array_equal(
        maps.columns, np.broadcast_to(np.arange(1, 29), (16, 28))
    )
    np.testing.assert_array_equal(
        maps.rows, np.broadcast_to(np.arange(1, 17)[:, None], (16, 28))
    )
    metadata = json.loads((output / "acquisition.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["pattern_placement"] == "native_pixels"
    assert metadata["pattern_raster_size"] == [28, 16]
    assert len(metadata["completed_sequences"]) == plan.displacement_count
    assert metadata["camera_size_observed"] == [28, 16]
    acquisitions = mocked_capture.acquisitions
    assert len(acquisitions) == 4
    first, second = acquisitions[0][0], acquisitions[1][0]
    assert first.shape == second.shape == (14, 16, 28)
    assert first[6, 0, 8] == 0
    assert second[6, 0, 8] == 255 and second[6, 0, 9] == 0
    np.testing.assert_array_equal(first[:2], second[:2])
    assert [settings["first_sequence"] for _, settings in acquisitions] == [
        True,
        False,
        False,
        False,
    ]
    assert all(settings["settle_seconds"] == 0 for _, settings in acquisitions)
    assert all(settings["flip"] is False for _, settings in acquisitions)
    mocked_capture.opener.assert_called_once()
    mocked_capture.stop.assert_called_once()


def test_capture_default_keeps_the_original_cell_rasters_and_maps(
    tmp_path, mocked_capture
):
    plan = small_plan()
    settings = {
        "camera_size": (28, 16),
        "settle_seconds": 0,
        "flip": False,
        "fullscreen": False,
        "max_sequence_mib": 1,
    }
    default = capture.run_capture(plan, tmp_path / "default", **settings)
    explicit = capture.run_capture(
        plan, tmp_path / "explicit", pattern_placement="emulated_cells", **settings
    )
    expected_columns = np.repeat(4 * np.arange(7) + 1, 4)
    expected_rows = np.repeat(4 * np.arange(4) + 1, 4)
    np.testing.assert_array_equal(
        default.columns, np.broadcast_to(expected_columns, (16, 28))
    )
    np.testing.assert_array_equal(
        default.rows, np.broadcast_to(expected_rows[:, None], (16, 28))
    )
    np.testing.assert_array_equal(default.columns, explicit.columns)
    np.testing.assert_array_equal(default.rows, explicit.rows)
    for first, second in zip(
        mocked_capture.acquisitions[:4], mocked_capture.acquisitions[4:]
    ):
        assert first[0].shape == (14, 4, 7)
        np.testing.assert_array_equal(first[0], second[0])
    metadata = json.loads((tmp_path / "default/acquisition.json").read_text())
    assert metadata["pattern_placement"] == "emulated_cells"
    assert metadata["pattern_raster_size"] == [7, 4]


def test_native_recording_retains_placement_through_replay_and_pattern_export(
    tmp_path, mocked_capture
):
    recording = tmp_path / "recorded"
    captured = capture.run_capture(
        small_plan(),
        recording,
        camera_size=(28, 16),
        settle_seconds=0,
        flip=False,
        fullscreen=False,
        pattern_placement="native_pixels",
        max_sequence_mib=1,
    )
    metadata = json.loads((recording / "scan.json").read_text())
    assert metadata["acquisition"]["pattern_placement"] == "native_pixels"
    assert metadata["pattern_raster_size"] == [28, 16]
    replayed = tmp_path / "replayed"
    assert (
        main(
            [
                "replay",
                "--config",
                str(recording / "scan.json"),
                "--input",
                str(recording),
                "--output",
                str(replayed),
            ]
        )
        == 0
    )
    np.testing.assert_array_equal(np.load(replayed / "h_pixels.npy"), captured.columns)
    np.testing.assert_array_equal(np.load(replayed / "v_pixels.npy"), captured.rows)
    exported = tmp_path / "regenerated"
    assert (
        main(
            [
                "patterns",
                "--config",
                str(replayed / "scan.json"),
                "--output",
                str(exported),
            ]
        )
        == 0
    )
    for displacement, (patterns, _) in enumerate(mocked_capture.acquisitions):
        with np.load(exported / f"patterns_displacement_{displacement}.npz") as data:
            np.testing.assert_array_equal(data["images"], patterns)


@pytest.mark.parametrize(
    "plan,placement,error",
    [
        (small_plan(), "unknown", "pattern_placement"),
        (
            replace(small_plan(), target_size=(14, 8)),
            "native_pixels",
            "intermediate target",
        ),
        (
            replace(small_plan(), projector_size=(28, 12)),
            "native_pixels",
            "equal integer",
        ),
    ],
)
def test_capture_rejects_unsupported_placement_before_camera_or_output(
    tmp_path, monkeypatch, plan, placement, error
):
    opener = MagicMock()
    generator = MagicMock()
    monkeypatch.setattr(capture, "open_camera", opener)
    monkeypatch.setattr(capture, "generate_pattern_sequence", generator)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match=error):
        capture.run_capture(
            plan,
            output,
            camera_size=(3, 2),
            pattern_placement=placement,
            max_sequence_mib=1,
        )
    opener.assert_not_called()
    generator.assert_not_called()
    assert not output.exists()


def test_capture_budget_counts_native_patterns_before_camera_or_output(
    tmp_path, monkeypatch
):
    opener = MagicMock()
    generator = MagicMock()
    monkeypatch.setattr(capture, "open_camera", opener)
    monkeypatch.setattr(capture, "generate_pattern_sequence", generator)
    output = tmp_path / "too-large"
    # Camera stack: 672 bytes; source-cell raster: 392; native raster: 6272.
    with pytest.raises(ValueError, match="sequence limit"):
        capture.run_capture(
            small_plan(),
            output,
            camera_size=(3, 2),
            pattern_placement="native_pixels",
            max_sequence_mib=0.001,
        )
    opener.assert_not_called()
    generator.assert_not_called()
    assert not output.exists()


@pytest.mark.parametrize(
    "configured,override,expected",
    [
        (None, None, "emulated_cells"),
        ("native_pixels", None, "native_pixels"),
        ("native_pixels", "emulated_cells", "emulated_cells"),
        ("emulated_cells", "native_pixels", "native_pixels"),
    ],
)
def test_capture_cli_placement_flag_overrides_acquisition_json(
    tmp_path, monkeypatch, configured, override, expected
):
    config = write_config(tmp_path / "config.json", placement=configured)
    execute = MagicMock()
    monkeypatch.setattr(scan, "run_capture", execute)
    arguments = [
        "capture",
        "--config",
        str(config),
        "--output",
        str(tmp_path / "capture"),
    ]
    if override is not None:
        arguments += ["--pattern-placement", override]
    assert main(arguments) == 0
    execute.assert_called_once()
    assert execute.call_args.kwargs["pattern_placement"] == expected


def test_plan_reports_native_raster_without_generating_patterns(
    tmp_path, capsys, monkeypatch
):
    config = write_config(tmp_path / "config.json", placement="native_pixels")

    def forbidden(*args, **kwargs):
        pytest.fail("Plan inspection generated a pattern stack")

    monkeypatch.setattr(scan, "generate_pattern_sequence", forbidden)
    assert main(["plan", "--config", str(config)]) == 0
    values = json.loads(capsys.readouterr().out)
    assert values["pattern_placement"] == "native_pixels"
    assert values["pattern_size"] == [7, 4]
    assert values["pattern_raster_size"] == [28, 16]
    assert "source_compatibility_notes" in values
    assert values["compatibility_notes"] == []
    assert (
        main(["plan", "--config", str(config), "--pattern-placement", "emulated_cells"])
        == 0
    )
    override = json.loads(capsys.readouterr().out)
    assert override["pattern_placement"] == "emulated_cells"
    assert override["pattern_raster_size"] == [7, 4]
    assert "compatibility_notes" in override


def test_native_plan_keeps_active_power_of_two_encoding_warning(tmp_path, capsys):
    config = write_config(
        tmp_path / "power-two.json",
        placement="native_pixels",
        plan=ScanPlan(projector_size=(16, 8), emulated_size=(8, 4), repeats=1),
    )
    assert main(["plan", "--config", str(config)]) == 0
    values = json.loads(capsys.readouterr().out)
    assert values["compatibility_notes"] == [
        "The final one-based identifier exceeds the historical Gray bit count"
    ]
    assert values["source_compatibility_notes"] == [
        "Displacements roll whole LD identifier cells before rendering"
    ]


def test_exported_native_patterns_replay_with_their_saved_plan(tmp_path, capsys):
    config = write_config(tmp_path / "config.json", placement="native_pixels")
    exported = tmp_path / "patterns"
    assert main(["patterns", "--config", str(config), "--output", str(exported)]) == 0
    metadata = json.loads((exported / "plan.json").read_text())
    assert metadata["acquisition"]["pattern_placement"] == "native_pixels"
    assert metadata["pattern_raster_size"] == [28, 16]
    assert read_plan(exported / "plan.json") == small_plan()
    recording = tmp_path / "recording"
    recording.mkdir()
    for displacement in range(4):
        with np.load(exported / f"patterns_displacement_{displacement}.npz") as data:
            patterns = data["images"]
            assert patterns.shape == (14, 16, 28)
            np.savez_compressed(
                recording / f"run_0_displacement_{displacement}.npz",
                images=direct_captures(patterns),
            )
    decoded = tmp_path / "decoded"
    assert (
        main(
            [
                "replay",
                "--config",
                str(exported / "plan.json"),
                "--input",
                str(recording),
                "--output",
                str(decoded),
            ]
        )
        == 0
    )
    np.testing.assert_array_equal(
        np.load(decoded / "h_pixels.npy"), np.broadcast_to(np.arange(1, 29), (16, 28))
    )
    np.testing.assert_array_equal(
        np.load(decoded / "v_pixels.npy"),
        np.broadcast_to(np.arange(1, 17)[:, None], (16, 28)),
    )
    # The exported configuration retains placement without repeating its flag.
    second = tmp_path / "export-again"
    assert (
        main(
            [
                "patterns",
                "--config",
                str(exported / "plan.json"),
                "--output",
                str(second),
            ]
        )
        == 0
    )
    for displacement in range(4):
        filename = f"patterns_displacement_{displacement}.npz"
        with (
            np.load(exported / filename) as original,
            np.load(second / filename) as replay,
        ):
            np.testing.assert_array_equal(original["images"], replay["images"])


@pytest.mark.parametrize("command", ["patterns", "capture"])
def test_cli_native_budget_rejects_before_creating_output(
    tmp_path, monkeypatch, command
):
    config = write_config(tmp_path / "config.json", placement="native_pixels")
    opener = MagicMock()
    monkeypatch.setattr(capture, "open_camera", opener)
    output = tmp_path / "too-large"
    arguments = [
        command,
        "--config",
        str(config),
        "--output",
        str(output),
        "--max-sequence-mib",
        "0.001",
    ]
    if command == "capture":
        arguments += ["--camera-size", "3x2"]
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    opener.assert_not_called()
    assert not output.exists()
