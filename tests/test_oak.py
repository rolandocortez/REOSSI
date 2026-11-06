"""Check sensor configuration against mocks; DepthAI and hardware are unused."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mocked_oak(monkeypatch):
    fake_depthai = SimpleNamespace(
        Device=MagicMock(),
        Pipeline=MagicMock(),
        node=SimpleNamespace(ColorCamera="color", XLinkOut="out", XLinkIn="in"),
        ColorCameraProperties=SimpleNamespace(
            SensorResolution=SimpleNamespace(THE_4_K="4k", THE_1080_P="1080p")
        ),
        CameraControl=MagicMock(),
    )
    monkeypatch.setitem(sys.modules, "depthai", fake_depthai)
    # Force this optional module to bind to the fake dependency for this test.
    monkeypatch.delitem(sys.modules, "reossi.acquisition.oak", raising=False)
    module = importlib.import_module("reossi.acquisition.oak")
    monkeypatch.setattr(module, "Thread", MagicMock())
    yield module, fake_depthai
    sys.modules.pop("reossi.acquisition.oak", None)


def test_default_applied_controls_and_keyboard_state_are_distinct(mocked_oak):
    module, _ = mocked_oak
    camera = module.OakCamera()
    control = camera.camRgb.initialControl
    control.setManualFocus.assert_called_once_with(135)
    control.setManualExposure.assert_called_once_with(32500, 250)
    control.setSharpness.assert_called_once_with(4)
    assert (camera.lensPos, camera.expTime, camera.sensIso, camera.sharpness) == (
        135,
        20000,
        800,
        0,
    )
    module.Thread.return_value.start.assert_called_once_with()


def test_sensor_settings_and_keyboard_state_can_be_set_independently(mocked_oak):
    module, _ = mocked_oak
    keyboard = module.OakKeyboardControls(
        focus=100, exposure_us=10000, iso=400, sharpness=2
    )
    camera = module.OakCamera(
        focus=120, exposure_us=15000, iso=600, sharpness=3, keyboard_controls=keyboard
    )
    camera.camRgb.initialControl.setManualFocus.assert_called_once_with(120)
    camera.camRgb.initialControl.setManualExposure.assert_called_once_with(15000, 600)
    camera.camRgb.initialControl.setSharpness.assert_called_once_with(3)
    assert (camera.lensPos, camera.expTime, camera.sensIso, camera.sharpness) == (
        100,
        10000,
        400,
        2,
    )
    camera.cam_params_control(ord("o"))
    control = module.dai.CameraControl.return_value
    control.setManualExposure.assert_called_once_with(10500, 400)
    camera.controlQueue.send.assert_called_once_with(control)
