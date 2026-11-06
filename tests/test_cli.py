"""Exercise entry points on tiny arrays without camera or display access."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from reossi.acquisition import capture
from reossi.cli.main import main


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "mode": "conventional",
                "projector_size": [8, 5],
                "emulated_size": [8, 5],
                "repeats": 1,
            }
        )
    )
    return path


def test_plan_reports_counts_without_generating_patterns(config, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Plan inspection must not generate image stacks")

    monkeypatch.setattr("reossi.cli.scan.generate_pattern_sequence", forbidden)
    assert main(["plan", "--config", str(config)]) == 0
    values = json.loads(capsys.readouterr().out)
    assert values["sequence_length"] == 14
    assert values["sequence_count"] == 1
    assert values["frame_count"] == 14


def test_pattern_export_and_replay(config, tmp_path):
    patterns = tmp_path / "patterns"
    assert main(["patterns", "--config", str(config), "--output", str(patterns)]) == 0
    recording = tmp_path / "recording"
    recording.mkdir()
    with np.load(patterns / "patterns_displacement_0.npz") as data:
        # The source live-capture path uses float64, unlike uint8 JPEG replay.
        np.savez(
            recording / "displacement_0.npz", images=data["images"].astype(np.float64)
        )
    output = tmp_path / "decoded"
    assert (
        main(
            [
                "replay",
                "--config",
                str(config),
                "--input",
                str(recording),
                "--output",
                str(output),
                "--layout",
                "single",
            ]
        )
        == 0
    )
    assert np.load(output / "h_pixels.npy").shape == (5, 8)
    assert np.load(output / "v_pixels.npy").shape == (5, 8)
    assert json.loads((output / "scan.json").read_text())["plan"]["repeats"] == 1


def test_pattern_memory_limit_prevents_output_creation(config, tmp_path):
    output = tmp_path / "too-large"
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "patterns",
                "--config",
                str(config),
                "--output",
                str(output),
                "--max-sequence-mib",
                "0.0001",
            ]
        )
    assert not output.exists()


def test_capture_sequence_keeps_float64_and_flips_before_gray(monkeypatch):
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    camera = SimpleNamespace(get_frame=lambda: frame.copy())
    displays = []
    monkeypatch.setattr(
        capture.cv2, "imshow", lambda name, image: displays.append(name)
    )
    monkeypatch.setattr(capture.cv2, "waitKey", lambda value: -1)
    result = capture.capture_sequence(
        camera,
        np.zeros((2, 2, 3), dtype=np.uint8),
        settle_seconds=0,
        flip=True,
    )
    expected = capture.cv2.cvtColor(
        capture.cv2.flip(frame, -1), capture.cv2.COLOR_BGR2GRAY
    )
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, np.stack([expected, expected]))
    assert displays == ["Gray Code Pattern", "Gray Code Pattern"]


def test_requested_capture_allocation_is_checked_before_opening_camera(tmp_path):
    from reossi.plan import ScanPlan

    with pytest.raises(ValueError, match="sequence limit"):
        capture.run_capture(ScanPlan(), tmp_path / "new", max_sequence_mib=0.001)
    assert not (tmp_path / "new").exists()


def test_capture_failure_closes_camera_and_display(tmp_path, monkeypatch):
    from reossi.acquisition import opencv as cameras
    from reossi.plan import ScanPlan

    events = []
    camera = SimpleNamespace(stop_cam=lambda: events.append("stop"))
    monkeypatch.setattr(cameras, "OpenCVCamera", lambda *args, **kwargs: camera)
    monkeypatch.setattr(capture.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(capture.cv2, "setWindowProperty", lambda *args: None)
    monkeypatch.setattr(
        capture.cv2, "destroyWindow", lambda name: events.append("close")
    )

    def failed_capture(*args, **kwargs):
        raise RuntimeError("Synthetic capture failure")

    monkeypatch.setattr(capture, "capture_sequence", failed_capture)
    plan = ScanPlan(mode="conventional", projector_size=(7, 4), repeats=1)
    with pytest.raises(RuntimeError, match="Synthetic capture failure"):
        capture.run_capture(plan, tmp_path / "partial", camera_size=(3, 2))
    assert events == ["stop", "close"]
    assert (
        json.loads((tmp_path / "partial" / "acquisition.json").read_text())["status"]
        == "failed"
    )
    assert (tmp_path / "partial" / "scan.json").is_file()
