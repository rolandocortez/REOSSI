"""Threaded OpenCV acquisition adapted from Charron's reference scanner."""

import math
from contextlib import suppress
from threading import Event, Lock, Thread, current_thread

import cv2


class OpenCVCamera:
    """Poll the requested camera and release it after its reader stops.

    The DirectShow/MJPG defaults and image contents follow the source adapter.
    Frame publication and resource ownership are synchronized. If a backend
    blocks in read beyond the close timeout, shutdown raises instead of
    releasing a handle concurrently with that read; the reader releases it
    when it eventually exits. Hardware behavior still requires a real rig.
    """

    def __init__(
        self,
        src=0,
        width=1920,
        height=1080,
        fps=30,
        *,
        backend=cv2.CAP_DSHOW,
        close_timeout_seconds=2.0,
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
        self.fps = fps
        self.SECONDS_PER_IMAGE = 1 / fps
        self.WAIT_MS = int(self.SECONDS_PER_IMAGE * 1000)
        self.close_timeout_seconds = close_timeout_seconds
        self.mtx = None
        self.dist = None
        self.status = False
        self.frame = None
        self.isNewFrame = False
        self.last_error = None
        self.thread_active = False
        self.thread = None
        self.capture = None
        self._lock = Lock()
        self._release_lock = Lock()
        self._stop_event = Event()
        self._released = False
        self._preview_opened = False
        try:
            self.capture = cv2.VideoCapture(src, backend)
            if not self.capture.isOpened():
                raise OSError(f"Could not open OpenCV camera source {src!r}")
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.capture.set(cv2.CAP_PROP_FPS, fps)
            self.capture.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G")
            )
            self.thread_active = True
            self.thread = Thread(target=self.update, daemon=True)
            self.thread.start()
        except BaseException:
            self.thread_active = False
            self._stop_event.set()
            self._release_capture()
            raise

    def _release_capture(self):
        with self._release_lock:
            if self.capture is not None and not self._released:
                self.capture.release()
                self._released = True

    def update(self):
        """Publish successful reads; a failed read never republishes a stale frame."""
        try:
            while self.thread_active and not self._stop_event.is_set():
                status, frame = self.capture.read()
                with self._lock:
                    self.status = bool(status and frame is not None)
                    self.frame = frame if self.status else None
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
            self._release_capture()

    def get_frame(self):
        """Return the next available frame, or None when its flag is clear."""
        with self._lock:
            if self.isNewFrame and not self._stop_event.is_set():
                self.isNewFrame = False
                return self.frame
            return None

    def show_frame(self):
        """Display the most recently retrieved frame."""
        with self._lock:
            frame = self.frame
        if frame is not None:
            cv2.imshow("frame", frame)
            self._preview_opened = True

    def remove_dist(self, frame):
        """Undistort and crop using ``mtx`` and ``dist``, when supplied."""
        if self.mtx is None or self.dist is None:
            return frame
        height, width = frame.shape[:2]
        camera_matrix, region = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (width, height), 1, (width, height)
        )
        image = cv2.undistort(frame, self.mtx, self.dist, None, camera_matrix)
        column, row, width, height = region
        return image[row : row + height, column : column + width]

    def stop_cam(self):
        """Stop polling, then release once; close only this adapter's preview."""
        self.thread_active = False
        self._stop_event.set()
        if self.thread is current_thread():
            return
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=self.close_timeout_seconds)
            if self.thread.is_alive():
                raise TimeoutError(
                    "Camera reader is still active; its handle was not released concurrently"
                )
        self._release_capture()
        with self._lock:
            self.isNewFrame = False
            self.frame = None
        if self._preview_opened:
            with suppress(cv2.error):
                cv2.destroyWindow("frame")
            self._preview_opened = False
