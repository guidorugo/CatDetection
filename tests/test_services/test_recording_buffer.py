import numpy as np
import pytest

from app.services.recording_service import RecordingBuffer


def test_buffer_add_and_get():
    buf = RecordingBuffer(max_seconds=1, fps=5)  # 5 frame capacity
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(3):
        buf.add_frame(frame, float(i))

    frames = buf.get_frames()
    assert len(frames) == 3
    # Each entry is (jpeg_bytes, timestamp)
    assert isinstance(frames[0][0], bytes)
    assert frames[0][1] == 0.0


def test_buffer_rolling_eviction():
    buf = RecordingBuffer(max_seconds=1, fps=2)  # 2 frame capacity
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    buf.add_frame(frame, 1.0)
    buf.add_frame(frame, 2.0)
    buf.add_frame(frame, 3.0)  # Should evict first frame

    frames = buf.get_frames()
    assert len(frames) == 2
    assert frames[0][1] == 2.0
    assert frames[1][1] == 3.0


def test_buffer_clear():
    buf = RecordingBuffer(max_seconds=1, fps=10)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    buf.add_frame(frame, 1.0)
    buf.clear()
    assert len(buf.get_frames()) == 0


def test_buffer_jpeg_compression():
    buf = RecordingBuffer(max_seconds=1, fps=5)
    # Create a larger frame to verify JPEG compression actually works
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    buf.add_frame(frame, 1.0)

    frames = buf.get_frames()
    jpeg_bytes = frames[0][0]
    # JPEG starts with FF D8
    assert jpeg_bytes[:2] == b'\xff\xd8'
    # Compressed should be much smaller than raw (720*1280*3 = 2,764,800)
    assert len(jpeg_bytes) < 720 * 1280
