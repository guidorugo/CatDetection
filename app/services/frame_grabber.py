import threading
import time

import cv2
import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


class FrameGrabber:
    """Threaded per-camera frame capture. Keeps only the latest frame."""

    def __init__(self, camera_id: int, source_url: str, expected_fps: int = 30):
        self.camera_id = camera_id
        self.source_url = source_url
        self.expected_fps = expected_fps

        self._frame: np.ndarray | None = None
        self._frame_time: float = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None

        # Health metrics
        self.actual_fps: float = 0.0
        self.is_connected: bool = False
        self.reconnect_attempts: int = 0
        self.last_frame_at: float = 0.0

    def start(self):
        """Start the frame grabbing thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("FrameGrabber started for camera %d: %s", self.camera_id, self.source_url)

    def stop(self):
        """Stop the frame grabbing thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._cap:
            self._cap.release()
            self._cap = None
        self.is_connected = False
        logger.info("FrameGrabber stopped for camera %d", self.camera_id)

    def get_frame(self) -> tuple[np.ndarray | None, float]:
        """Get the latest frame and its timestamp."""
        with self._lock:
            return self._frame, self._frame_time

    def _capture_loop(self):
        """Main capture loop running in a separate thread."""
        fps_counter = 0
        fps_start = time.monotonic()
        backoff = 1.0

        while self._running:
            # Connect if needed
            if self._cap is None or not self._cap.isOpened():
                self._connect(backoff)
                if not self.is_connected:
                    backoff = min(backoff * 2, 30.0)
                    time.sleep(backoff)
                    continue
                backoff = 1.0

            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Frame read failed for camera %d", self.camera_id)
                self.is_connected = False
                self._cap.release()
                self._cap = None
                continue

            now = time.monotonic()
            with self._lock:
                self._frame = frame
                self._frame_time = now

            self.last_frame_at = now
            fps_counter += 1

            # Update FPS every second
            elapsed = now - fps_start
            if elapsed >= 1.0:
                self.actual_fps = fps_counter / elapsed
                fps_counter = 0
                fps_start = now

    def _connect(self, backoff: float):
        """Attempt to connect to the camera source."""
        self.reconnect_attempts += 1
        logger.info(
            "Connecting to camera %d (attempt %d): %s",
            self.camera_id,
            self.reconnect_attempts,
            self.source_url,
        )
        try:
            cap = cv2.VideoCapture(self.source_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                self._cap = cap
                self.is_connected = True
                self.reconnect_attempts = 0
                logger.info("Connected to camera %d", self.camera_id)
            else:
                cap.release()
                logger.warning("Failed to open camera %d", self.camera_id)
        except Exception as e:
            logger.error("Error connecting to camera %d: %s", self.camera_id, e)
