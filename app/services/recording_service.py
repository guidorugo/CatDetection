import asyncio
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings
from app.core.database import async_session
from app.core.logging import get_logger
from app.models.recording import Recording

logger = get_logger(__name__)


class RecordingBuffer:
    """Rolling JPEG-compressed frame buffer for pre-roll recording."""

    def __init__(self, max_seconds: int = 5, fps: int = 30):
        self.max_frames = max_seconds * fps
        self._buffer: deque[tuple[bytes, float]] = deque(maxlen=self.max_frames)
        self._lock = threading.Lock()

    def add_frame(self, frame: np.ndarray, timestamp: float):
        """Add a frame to the rolling buffer (JPEG compressed)."""
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        with self._lock:
            self._buffer.append((jpeg.tobytes(), timestamp))

    def get_frames(self) -> list[tuple[bytes, float]]:
        """Get all buffered frames as (jpeg_bytes, timestamp) pairs."""
        with self._lock:
            return list(self._buffer)

    def clear(self):
        with self._lock:
            self._buffer.clear()


class RecordingService:
    """Manages event-triggered video recording using FFmpeg subprocess."""

    def __init__(self):
        self._active_recordings: dict[int, asyncio.Task] = {}  # camera_id -> task
        self._buffers: dict[int, RecordingBuffer] = {}  # camera_id -> buffer
        self._lock = asyncio.Lock()

    def get_or_create_buffer(self, camera_id: int) -> RecordingBuffer:
        if camera_id not in self._buffers:
            self._buffers[camera_id] = RecordingBuffer(
                max_seconds=settings.RECORDING_PRE_ROLL_SECONDS,
                fps=settings.RECORDING_FPS,
            )
        return self._buffers[camera_id]

    async def trigger_recording(
        self,
        camera_id: int,
        get_frame_callback,
    ) -> int | None:
        """Start a recording for a camera. Returns recording DB id or None."""
        async with self._lock:
            # Check concurrent limit
            active = len(self._active_recordings)
            if active >= settings.RECORDING_MAX_CONCURRENT:
                logger.warning("Max concurrent recordings reached (%d)", active)
                return None

            # Don't start if already recording this camera
            if camera_id in self._active_recordings:
                return None

        # Create DB record
        now = datetime.now(timezone.utc)
        filename = f"cam{camera_id}_{now.strftime('%Y%m%d_%H%M%S')}.mp4"
        file_path = Path(settings.RECORDINGS_DIR) / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        async with async_session() as db:
            recording = Recording(
                camera_id=camera_id,
                file_path=str(file_path),
                start_time=now,
                resolution=settings.RECORDING_RESOLUTION,
                fps=settings.RECORDING_FPS,
                codec="h264",
                status="recording",
            )
            db.add(recording)
            await db.commit()
            await db.refresh(recording)
            recording_id = recording.id

        # Start recording task
        task = asyncio.create_task(
            self._record(camera_id, recording_id, str(file_path), get_frame_callback)
        )
        async with self._lock:
            self._active_recordings[camera_id] = task

        logger.info("Started recording for camera %d -> %s", camera_id, file_path)
        return recording_id

    async def _record(
        self,
        camera_id: int,
        recording_id: int,
        file_path: str,
        get_frame_callback,
    ):
        """Record video using FFmpeg subprocess with piped raw frames."""
        w, h = map(int, settings.RECORDING_RESOLUTION.split("x"))
        fps = settings.RECORDING_FPS

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", str(settings.RECORDING_CRF),
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            file_path,
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

            # Write pre-roll frames
            buffer = self._buffers.get(camera_id)
            if buffer:
                for jpeg_bytes, _ in buffer.get_frames():
                    frame = cv2.imdecode(
                        np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is not None:
                        frame = cv2.resize(frame, (w, h))
                        proc.stdin.write(frame.tobytes())
                        await proc.stdin.drain()

            # Record post-roll
            start = time.monotonic()
            max_duration = settings.RECORDING_MAX_DURATION
            post_roll = settings.RECORDING_POST_ROLL_SECONDS
            frame_interval = 1.0 / fps

            while time.monotonic() - start < post_roll:
                frame_start = time.monotonic()
                frame, _ = await asyncio.to_thread(get_frame_callback)
                if frame is not None:
                    frame = cv2.resize(frame, (w, h))
                    proc.stdin.write(frame.tobytes())
                    await proc.stdin.drain()

                elapsed = time.monotonic() - frame_start
                if elapsed < frame_interval:
                    await asyncio.sleep(frame_interval - elapsed)

            proc.stdin.close()
            await proc.wait()

            # Update DB record
            end_time = datetime.now(timezone.utc)
            file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
            duration = (end_time - datetime.now(timezone.utc)).total_seconds()

            async with async_session() as db:
                from sqlalchemy import select

                result = await db.execute(
                    select(Recording).where(Recording.id == recording_id)
                )
                rec = result.scalar_one_or_none()
                if rec:
                    rec.status = "completed"
                    rec.end_time = end_time
                    rec.file_size = file_size
                    rec.duration = time.monotonic() - start
                    await db.commit()

            logger.info("Recording completed for camera %d: %s", camera_id, file_path)

        except Exception as e:
            logger.error("Recording error for camera %d: %s", camera_id, e)
            async with async_session() as db:
                from sqlalchemy import select

                result = await db.execute(
                    select(Recording).where(Recording.id == recording_id)
                )
                rec = result.scalar_one_or_none()
                if rec:
                    rec.status = "failed"
                    await db.commit()
        finally:
            async with self._lock:
                self._active_recordings.pop(camera_id, None)

    async def stop_all(self):
        """Cancel all active recordings."""
        async with self._lock:
            for task in self._active_recordings.values():
                task.cancel()
            self._active_recordings.clear()
