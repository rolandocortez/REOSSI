"""Camera lifecycle checks use mocked drivers and at most 8-by-8 frames."""

import importlib
import json
import sys
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from reossi.acquisition import capture, devices, opencv
from reossi.plan import ScanPlan


class PassiveThread:
    def __init__(self, *, target, daemon):
        self.target = target
        self.daemon = daemon
        self.alive = False

    def start(self):
        pass

    def is_alive(self):
        return self.alive

    def join(self, *, timeout):
        self.alive = False


@pytest.fixture
def fake_opencv(monkeypatch):
    driver = MagicMock()
    driver.isOpened.return_value = True
    monkeypatch.setattr(opencv.cv2, "VideoCapture", MagicMock(return_value=driver))
    monkeypatch.setattr(opencv, "Thread", PassiveThread)
    monkeypatch.setattr(opencv.cv2, "destroyAllWindows", MagicMock())
    monkeypatch.setattr(opencv.cv2, "destroyWindow", MagicMock())
    return driver


def test_opencv_initializes_frame_state_before_thread_start(fake_opencv, monkeypatch):
    frame = np.full((2, 3, 3), 7, dtype=np.uint8)

    class PublishingThread(PassiveThread):
        def start(self):
            camera = self.target.__self__
            assert camera.frame is None
            assert camera.isNewFrame is False
            assert camera.mtx is None and camera.dist is None
            camera.frame = frame
            camera.isNewFrame = True

    monkeypatch.setattr(opencv, "Thread", PublishingThread)
    camera = opencv.OpenCVCamera(width=3, height=2)
    assert camera.get_frame() is frame
    assert camera.get_frame() is None
    camera.stop_cam()
    fake_opencv.release.assert_called_once_with()


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_opencv_rejects_bad_fps_before_open(fake_opencv, fps):
    with pytest.raises(ValueError, match="fps"):
        opencv.OpenCVCamera(fps=fps)
    opencv.cv2.VideoCapture.assert_not_called()


def test_opencv_constructor_failure_releases_without_starting_thread(fake_opencv):
    fake_opencv.set.side_effect = RuntimeError("Driver configuration failed")
    with pytest.raises(RuntimeError, match="configuration"):
        opencv.OpenCVCamera()
    fake_opencv.release.assert_called_once_with()
    opencv.cv2.destroyAllWindows.assert_not_called()


def test_opencv_failed_read_clears_a_stale_frame(fake_opencv):
    camera = opencv.OpenCVCamera()
    camera.frame = np.ones((2, 3, 3), dtype=np.uint8)
    camera.isNewFrame = True

    def failed_read():
        camera.thread_active = False
        return False, None

    fake_opencv.read.side_effect = failed_read
    camera.update()
    assert camera.get_frame() is None
    assert camera.status is False
    camera.stop_cam()
    fake_opencv.release.assert_called_once_with()


def test_opencv_waits_for_reader_before_release_and_close_is_idempotent(fake_opencv):
    camera = opencv.OpenCVCamera()
    events = []
    camera.thread.alive = True

    def join(*, timeout):
        events.append("join")
        assert timeout == 2
        camera.thread.alive = False

    camera.thread.join = join
    fake_opencv.release.side_effect = lambda: events.append("release")
    camera.stop_cam()
    camera.stop_cam()
    assert events == ["join", "release"]
    opencv.cv2.destroyAllWindows.assert_not_called()
    opencv.cv2.destroyWindow.assert_not_called()


def test_opencv_stuck_reader_is_not_released_concurrently(fake_opencv):
    camera = opencv.OpenCVCamera(close_timeout_seconds=0.001)
    camera.thread.alive = True
    camera.thread.join = lambda **kwargs: None
    with pytest.raises(TimeoutError, match="not released concurrently"):
        camera.stop_cam()
    fake_opencv.release.assert_not_called()
    camera.thread.alive = False
    camera.stop_cam()


def test_opencv_real_reader_exit_precedes_release_with_a_mock_driver(
    fake_opencv, monkeypatch
):
    started, finish = Event(), Event()
    read_active = Event()

    def read():
        read_active.set()
        started.set()
        assert finish.wait(timeout=1)
        read_active.clear()
        return False, None

    def release():
        assert not read_active.is_set()

    fake_opencv.read.side_effect = read
    fake_opencv.release.side_effect = release
    monkeypatch.setattr(opencv, "Thread", Thread)
    camera = opencv.OpenCVCamera(close_timeout_seconds=1)
    assert started.wait(timeout=1)
    closer = Thread(target=camera.stop_cam)
    closer.start()
    try:
        assert camera._stop_event.wait(timeout=1)
        fake_opencv.release.assert_not_called()
    finally:
        finish.set()
        closer.join(timeout=1)
    assert not closer.is_alive()
    fake_opencv.release.assert_called_once_with()


def test_opencv_closes_only_the_preview_it_opened(fake_opencv, monkeypatch):
    monkeypatch.setattr(opencv.cv2, "imshow", MagicMock())
    camera = opencv.OpenCVCamera()
    camera.frame = np.ones((2, 3, 3), dtype=np.uint8)
    camera.show_frame()
    camera.stop_cam()
    opencv.cv2.destroyWindow.assert_called_once_with("frame")
    opencv.cv2.destroyAllWindows.assert_not_called()


@pytest.fixture
def fake_oak(monkeypatch):
    sdk = SimpleNamespace(
        Device=MagicMock(),
        DeviceInfo=MagicMock(),
        Pipeline=MagicMock(),
        node=SimpleNamespace(ColorCamera="color", XLinkOut="out", XLinkIn="in"),
        ColorCameraProperties=SimpleNamespace(
            SensorResolution=SimpleNamespace(THE_4_K="4k", THE_1080_P="1080p")
        ),
        CameraControl=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "depthai", sdk)
    monkeypatch.delitem(sys.modules, "reossi.acquisition.oak", raising=False)
    module = importlib.import_module("reossi.acquisition.oak")
    monkeypatch.setattr(module, "Thread", PassiveThread)
    yield module, sdk
    sys.modules.pop("reossi.acquisition.oak", None)


def test_oak_initializes_queues_and_keyboard_state_before_thread_start(
    fake_oak, monkeypatch
):
    module, sdk = fake_oak
    frame = np.ones((2, 3, 3), dtype=np.uint8)

    class PublishingThread(PassiveThread):
        def start(self):
            camera = self.target.__self__
            assert camera.frame is None and camera.isNewFrame is False
            assert camera.ispQueue is not None and camera.controlQueue is not None
            assert (
                camera.lensPos,
                camera.expTime,
                camera.sensIso,
                camera.sharpness,
            ) == (135, 20000, 800, 0)
            camera.frame, camera.isNewFrame = frame, True

    monkeypatch.setattr(module, "Thread", PublishingThread)
    camera = module.OakCamera(device_id="MY-MOCK-OAK")
    sdk.DeviceInfo.assert_called_once_with("MY-MOCK-OAK")
    sdk.Device.assert_called_once_with(sdk.DeviceInfo.return_value)
    assert camera.get_frame() is frame
    camera.stop_cam()
    camera.stop_cam()
    sdk.Device.return_value.close.assert_called_once_with()


def test_oak_pipeline_failure_closes_opened_device(fake_oak):
    module, sdk = fake_oak
    sdk.Device.return_value.startPipeline.side_effect = RuntimeError(
        "Pipeline startup failed"
    )
    with pytest.raises(RuntimeError, match="startup"):
        module.OakCamera()
    sdk.Device.return_value.close.assert_called_once_with()


@pytest.mark.parametrize("fps", [0, float("nan"), float("inf")])
def test_oak_bad_fps_is_rejected_before_device_open(fake_oak, fps):
    module, sdk = fake_oak
    with pytest.raises(ValueError, match="fps"):
        module.OakCamera(fps=fps)
    sdk.Device.assert_not_called()


def test_oak_failed_frame_does_not_publish_old_image(fake_oak):
    module, sdk = fake_oak
    camera = module.OakCamera()
    camera.frame = np.ones((2, 3, 3), dtype=np.uint8)
    camera.isNewFrame = True

    def missing_frame():
        camera.thread_active = False
        return None

    packet = SimpleNamespace(getCvFrame=missing_frame)
    camera.ispQueue.tryGet.return_value = packet
    camera.update()
    assert camera.get_frame() is None
    camera.stop_cam()
    sdk.Device.return_value.close.assert_called_once_with()


def test_shared_factory_forwards_explicit_oak_controls_and_device(
    fake_oak, monkeypatch
):
    module, _ = fake_oak
    factory = MagicMock()
    monkeypatch.setattr(module, "OakCamera", factory)
    devices.open_camera(
        camera_kind="oak",
        camera_size=(3840, 2160),
        device_id="mock-id",
        oak_options={"focus": 120, "keyboard_controls": {"iso": 400}},
    )
    options = factory.call_args.kwargs
    assert options["device_id"] == "mock-id"
    assert options["focus"] == 120
    assert options["keyboard_controls"].iso == 400
    assert options["keyboard_controls"].exposure_us == 20000


def test_capture_pumps_each_pattern_before_reading_a_ready_frame_at_zero_dwell(
    monkeypatch,
):
    events = []
    frame = np.full((2, 3, 3), 50, dtype=np.uint8)

    def read_ready_frame():
        events.append("read")
        return frame.copy()

    def process_events(delay):
        events.append("pump")
        return -1

    monkeypatch.setattr(capture.cv2, "imshow", lambda *args: events.append("show"))
    monkeypatch.setattr(capture.cv2, "waitKey", process_events)
    images = capture.capture_sequence(
        SimpleNamespace(get_frame=read_ready_frame),
        np.zeros((2, 2, 3), dtype=np.uint8),
        settle_seconds=0,
        flip=False,
    )
    assert events == ["show", "pump", "read", "show", "pump", "read"]
    assert images.dtype == np.float64
    np.testing.assert_array_equal(images, np.full((2, 2, 3), 50.0))


def test_capture_processes_quit_before_reading_a_ready_frame(monkeypatch):
    events = []
    camera = SimpleNamespace(get_frame=MagicMock())
    monkeypatch.setattr(capture.cv2, "imshow", lambda *args: events.append("show"))

    def quit_event(delay):
        events.append("pump")
        return ord("q")

    monkeypatch.setattr(capture.cv2, "waitKey", quit_event)
    with pytest.raises(KeyboardInterrupt, match="Capture stopped with q"):
        capture.capture_sequence(
            camera,
            np.zeros((2, 2, 3), dtype=np.uint8),
            settle_seconds=0,
            flip=False,
        )
    assert events == ["show", "pump"]
    camera.get_frame.assert_not_called()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0])
def test_capture_memory_limit_is_finite_before_open(tmp_path, monkeypatch, value):
    opener = MagicMock()
    monkeypatch.setattr(capture, "open_camera", opener)
    with pytest.raises(ValueError, match="finite and positive"):
        capture.run_capture(ScanPlan(), tmp_path / "invalid", max_sequence_mib=value)
    opener.assert_not_called()
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize(
    "settings",
    [
        {"settle_seconds": float("nan")},
        {"frame_timeout_seconds": float("inf")},
        {"fps": float("inf")},
    ],
)
def test_capture_times_are_validated_before_open(tmp_path, monkeypatch, settings):
    opener = MagicMock()
    monkeypatch.setattr(capture, "open_camera", opener)
    with pytest.raises(ValueError):
        capture.run_capture(ScanPlan(), tmp_path / "invalid", **settings)
    opener.assert_not_called()
    assert not (tmp_path / "invalid").exists()


def test_interrupted_capture_retains_completed_sequences_and_configuration(
    tmp_path, monkeypatch
):
    events = []
    camera = SimpleNamespace(stop_cam=lambda: events.append("stop"))
    monkeypatch.setattr(capture, "open_camera", MagicMock(return_value=camera))
    monkeypatch.setattr(capture.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(capture.cv2, "setWindowProperty", lambda *args: None)
    monkeypatch.setattr(capture.cv2, "destroyWindow", lambda name: events.append(name))
    calls = 0

    def acquire(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("Mock interruption")
        return np.full((14, 2, 3), 50.0)

    monkeypatch.setattr(capture, "capture_sequence", acquire)
    plan = ScanPlan(mode="conventional", projector_size=(7, 4), repeats=2)
    output = tmp_path / "partial"
    with pytest.raises(KeyboardInterrupt):
        capture.run_capture(plan, output, camera_size=(3, 2), fullscreen=False)
    metadata = json.loads((output / "acquisition.json").read_text())
    assert metadata["status"] == "interrupted"
    assert metadata["completed_sequences"] == [
        {
            "run": 0,
            "displacement": 0,
            "file": "run_0_displacement_0.npz",
            "frames": 14,
            "dtype": "float64",
        }
    ]
    assert metadata["camera_size_observed"] == [3, 2]
    assert (output / "run_0_displacement_0.npz").is_file()
    assert json.loads((output / "scan.json").read_text())["plan"]["repeats"] == 2
    assert events == ["stop", "Gray Code Pattern"]


def test_capture_preserves_primary_error_when_camera_close_also_fails(
    tmp_path, monkeypatch
):
    def stop():
        raise TimeoutError("Mock reader is blocked")

    monkeypatch.setattr(
        capture, "open_camera", lambda **kwargs: SimpleNamespace(stop_cam=stop)
    )
    monkeypatch.setattr(capture.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(capture.cv2, "destroyWindow", lambda name: None)

    def failed(*args, **kwargs):
        raise RuntimeError("Primary acquisition failure")

    monkeypatch.setattr(capture, "capture_sequence", failed)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="Primary acquisition failure"):
        capture.run_capture(
            ScanPlan(mode="conventional", projector_size=(7, 4), repeats=1),
            output,
            camera_size=(3, 2),
            fullscreen=False,
        )
    metadata = json.loads((output / "acquisition.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["error"]["type"] == "RuntimeError"
    assert metadata["close_error"]["type"] == "TimeoutError"
