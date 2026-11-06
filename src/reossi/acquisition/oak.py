"""Historical OAK adapter from Rolando Cortez's local REOSSI scanner.

DepthAI is an optional dependency of this module only. No device is opened
until ``OakCamera`` is constructed. Sensor and keyboard defaults retain their
historical distinction; initialization and shutdown have explicit ownership.
"""

import math
from contextlib import suppress
from dataclasses import dataclass
from threading import Event, Lock, Thread, current_thread

import cv2
import depthai as dai


def _clamp(num, v0, v1):
    return max(v0, min(num, v1))


@dataclass(frozen=True)
class OakKeyboardControls:
    """Initial values used by the legacy keyboard adjustments.

    These are independent of the values applied to the sensor at startup.
    Defaults preserve the historical difference between those two states.
    """

    focus: int = 135
    exposure_us: int = 20000
    iso: int = 800
    sharpness: int = 0


class OakCamera:
    """Retain the historical local OAK capture and keyboard controls.

    ``device_id`` explicitly selects an MXID/IP/USB name through DepthAI v2's
    DeviceInfo constructor; None retains the original default-device search.
    The numeric ``src`` remains accepted for interface compatibility. No frame
    thread starts before queues and frame/control state have been initialized.
    """

    def __init__(
        self,
        src=0,
        width=3840,
        height=2160,
        fps=30,
        *,
        focus: int = 135,
        exposure_us: int = 32500,
        iso: int = 250,
        sharpness: int = 4,
        keyboard_controls: OakKeyboardControls = OakKeyboardControls(),
        device_id: str | None = None,
        close_timeout_seconds: float = 2.0,
    ):
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in (width, height)
        ):
            raise ValueError("Camera width and height must be positive integers")
        if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        if not math.isfinite(close_timeout_seconds) or close_timeout_seconds <= 0:
            raise ValueError("close_timeout_seconds must be finite and positive")
        if device_id is not None and (
            not isinstance(device_id, str) or not device_id.strip()
        ):
            raise ValueError("device_id must be a nonempty MXID/IP/USB name or None")
        self.width = width
        self.height = height
        self.fps = fps
        self.close_timeout_seconds = close_timeout_seconds
        self.device_id = device_id
        self.SECONDS_PER_IMAGE = 1 / fps
        self.WAIT_MS = int(self.SECONDS_PER_IMAGE * 1000)
        self.mtx = None
        self.dist = None
        self.status = False
        self.frame = None
        self.isNewFrame = False
        self.last_error = None
        self.thread_active = False
        self.thread = None
        self.device = None
        self._lock = Lock()
        self._release_lock = Lock()
        self._stop_event = Event()
        self._released = False
        self._preview_opened = False
        self.lensPos = keyboard_controls.focus
        self.expTime = keyboard_controls.exposure_us
        self.sensIso = keyboard_controls.iso
        self.awb_lock = False
        self.sharpness = keyboard_controls.sharpness
        try:
            self._configure_pipeline(width, fps, focus, exposure_us, iso, sharpness)
            self.device = (
                dai.Device()
                if device_id is None
                else dai.Device(dai.DeviceInfo(device_id))
            )
            self.device.startPipeline(self.pipeline)
            self.ispQueue = self.device.getOutputQueue(
                name="isp", maxSize=4, blocking=False
            )
            self.controlQueue = self.device.getInputQueue("control")
            self.thread_active = True
            self.thread = Thread(target=self.update, daemon=True)
            self.thread.start()
        except BaseException:
            self.thread_active = False
            self._stop_event.set()
            self._close_device()
            raise

    def _configure_pipeline(self, width, fps, focus, exposure_us, iso, sharpness):
        """Build the historical ISP/control pipeline before opening a device."""

        # Configure the camera pipeline
        self.pipeline = dai.Pipeline()
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        if width == 3840:
            self.camRgb.setResolution(
                dai.ColorCameraProperties.SensorResolution.THE_4_K
            )
        elif width == 1920:
            self.camRgb.setResolution(
                dai.ColorCameraProperties.SensorResolution.THE_1080_P
            )
        else:
            self.camRgb.setResolution(
                dai.ColorCameraProperties.SensorResolution.THE_1080_P
            )
            self.camRgb.setIspScale(2, 3)
        self.camRgb.setFps(fps)
        self.camRgb.initialControl.setManualFocus(focus)
        self.camRgb.initialControl.setManualExposure(exposure_us, iso)
        self.camRgb.initialControl.setSharpness(sharpness)

        # Output queues
        self.ispOut = self.pipeline.create(dai.node.XLinkOut)
        self.ispOut.setStreamName("isp")
        self.camRgb.isp.link(self.ispOut.input)

        # **Create a control input queue**
        self.controlIn = self.pipeline.create(dai.node.XLinkIn)
        self.controlIn.setStreamName("control")
        self.controlIn.out.link(
            self.camRgb.inputControl
        )  # Link the control input to the camera control

    def _close_device(self):
        with self._release_lock:
            if self.device is not None and not self._released:
                self.device.close()
                self._released = True

    def update(self):
        """Poll the nonblocking ISP queue and publish complete frames."""
        try:
            while self.thread_active and not self._stop_event.is_set():
                packet = self.ispQueue.tryGet()
                if packet is not None:
                    frame = packet.getCvFrame()
                    with self._lock:
                        self.status = frame is not None
                        self.frame = frame
                        self.isNewFrame = self.status
                self._stop_event.wait(self.SECONDS_PER_IMAGE)
        except Exception as error:
            with self._lock:
                self.status = False
                self.frame = None
                self.isNewFrame = False
                self.last_error = error
        finally:
            self.thread_active = False
            self._close_device()

    def get_frame(self):
        """
        Returns the current frame
        """
        with self._lock:
            if self.isNewFrame and not self._stop_event.is_set():
                self.isNewFrame = False
                return self.frame
            return None

    def show_frame(self):
        """
        Shows the current frame
        """
        with self._lock:
            frame = self.frame
        if frame is not None:
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
            self._preview_opened = True

    def remove_dist(self, frame):
        """
        Removes distortion from a frame

        Parameters:
        ----------
        frame : np.array
            The frame to remove distortion from.

        Returns:
        -------
        dst : np.array
            The frame without distortion.
        """
        if self.mtx is None or self.dist is None:
            return frame
        else:
            h, w = frame.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                self.mtx, self.dist, (w, h), 1, (w, h)
            )
            dst = cv2.undistort(frame, self.mtx, self.dist, None, newcameramtx)
            # crop the image
            x, y, w, h = roi
            dst = dst[y : y + h, x : x + w]
            return dst

    def cam_params_control(self, key):
        """
        Control camera parameters with keyboard input.
        """
        if key in [ord(","), ord(".")]:
            if key == ord(","):
                self.lensPos -= 3
            if key == ord("."):
                self.lensPos += 3
            self.lensPos = _clamp(self.lensPos, 0, 255)
            print("Manual focus, lens position: ", self.lensPos)
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(self.lensPos)
            self.controlQueue.send(ctrl)
        elif key in [ord("i"), ord("o"), ord("h"), ord("j")]:
            if key == ord("i"):
                self.expTime -= 500
            if key == ord("o"):
                self.expTime += 500
            if key == ord("h"):
                self.sensIso -= 50
            if key == ord("j"):
                self.sensIso += 50
            self.expTime = _clamp(self.expTime, 1, 33000)
            self.sensIso = _clamp(self.sensIso, 100, 1600)
            print("Manual exposure, time: ", self.expTime, "ISO: ", self.sensIso)
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(self.expTime, self.sensIso)
            self.controlQueue.send(ctrl)
        elif key == ord("9"):
            self.awb_lock = not self.awb_lock
            print("Auto white balance lock:", self.awb_lock)
            ctrl = dai.CameraControl()
            ctrl.setAutoWhiteBalanceLock(self.awb_lock)
            self.controlQueue.send(ctrl)
        elif key == ord("0"):
            self.sharpness = (
                _clamp(self.sharpness + 1, 0, 4)
                if key == ord("=")
                else _clamp(self.sharpness - 1, 0, 4)
            )
            print("Sharpness:", self.sharpness)
            ctrl = dai.CameraControl()
            ctrl.setSharpness(self.sharpness)
            self.controlQueue.send(ctrl)

    def stop_cam(self):
        """Stop queue polling before closing the device and owned preview."""
        self.thread_active = False
        self._stop_event.set()
        if self.thread is current_thread():
            return
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=self.close_timeout_seconds)
            if self.thread.is_alive():
                raise TimeoutError(
                    "OAK reader is still active; its device was not closed concurrently"
                )
        self._close_device()
        with self._lock:
            self.isNewFrame = False
            self.frame = None
        if self._preview_opened:
            with suppress(cv2.error):
                cv2.destroyWindow("frame")
            self._preview_opened = False
