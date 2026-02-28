import asyncio
import time

from app.core.logging import get_logger
from app.services.notification_service import NotificationService
from app.services.stream_manager import StreamManager

logger = get_logger(__name__)


class CameraHealthMonitor:
    """Monitors camera health and triggers alerts."""

    # Thresholds
    FPS_WARNING_RATIO = 0.5  # Alert if actual FPS < 50% of expected
    NO_FRAME_CRITICAL_SECONDS = 5.0
    MAX_RECONNECT_ATTEMPTS = 5

    def __init__(self, stream_manager: StreamManager, notification_service: NotificationService):
        self._stream_manager = stream_manager
        self._notifications = notification_service
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_status: dict[int, str] = {}

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("CameraHealthMonitor started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("CameraHealthMonitor stopped")

    async def _monitor_loop(self):
        while self._running:
            try:
                await self._check_all_cameras()
            except Exception as e:
                logger.error("Health check error: %s", e)
            await asyncio.sleep(2.0)

    async def _check_all_cameras(self):
        now = time.monotonic()
        for camera_id, grabber in self._stream_manager.get_all_grabbers().items():
            old_status = self._last_status.get(camera_id, "unknown")

            if not grabber.is_connected:
                if grabber.reconnect_attempts > self.MAX_RECONNECT_ATTEMPTS:
                    status = "offline"
                else:
                    status = "disconnected"
            elif now - grabber.last_frame_at > self.NO_FRAME_CRITICAL_SECONDS:
                status = "disconnected"
            elif (
                grabber.expected_fps > 0
                and grabber.actual_fps < grabber.expected_fps * self.FPS_WARNING_RATIO
            ):
                status = "degraded"
            else:
                status = "connected"

            if status != old_status:
                self._last_status[camera_id] = status
                logger.info("Camera %d status: %s -> %s", camera_id, old_status, status)
                await self._notifications.broadcast_status(
                    {
                        "type": "camera_health",
                        "camera_id": camera_id,
                        "status": status,
                        "actual_fps": round(grabber.actual_fps, 1),
                        "reconnect_attempts": grabber.reconnect_attempts,
                    }
                )

    def get_health(self, camera_id: int) -> dict:
        grabber = self._stream_manager.get_grabber(camera_id)
        if not grabber:
            return {"camera_id": camera_id, "status": "not_found"}
        return {
            "camera_id": camera_id,
            "status": self._last_status.get(camera_id, "unknown"),
            "actual_fps": round(grabber.actual_fps, 1),
            "is_connected": grabber.is_connected,
            "reconnect_attempts": grabber.reconnect_attempts,
        }
