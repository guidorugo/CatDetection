import threading
import time

import numpy as np
import pytest

from app.services.frame_grabber import FrameGrabber


def test_frame_grabber_init():
    fg = FrameGrabber(camera_id=1, source_url="rtsp://fake", expected_fps=30)
    assert fg.camera_id == 1
    assert fg.source_url == "rtsp://fake"
    assert fg.expected_fps == 30
    assert fg.is_connected is False
    assert fg.actual_fps == 0.0


def test_get_frame_returns_none_before_start():
    fg = FrameGrabber(camera_id=1, source_url="rtsp://fake")
    frame, ts = fg.get_frame()
    assert frame is None
    assert ts == 0.0


def test_frame_grabber_stop_without_start():
    fg = FrameGrabber(camera_id=1, source_url="rtsp://fake")
    fg.stop()  # Should not raise
    assert fg.is_connected is False
