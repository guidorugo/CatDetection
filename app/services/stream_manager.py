import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session
from app.core.logging import get_logger
from app.models.camera import Camera
from app.services.frame_grabber import FrameGrabber

logger = get_logger(__name__)


class StreamManager:
    """Manages camera connections lifecycle."""

    def __init__(self):
        self._grabbers: dict[int, FrameGrabber] = {}

    async def start(self):
        """Load enabled cameras from DB and start grabbers."""
        async with async_session() as db:
            result = await db.execute(select(Camera).where(Camera.is_enabled == True))
            cameras = result.scalars().all()

        for camera in cameras:
            self.add_camera(camera.id, camera.source_url, camera.expected_fps)

        logger.info("StreamManager started with %d cameras", len(self._grabbers))

    async def stop(self):
        """Stop all frame grabbers."""
        for grabber in self._grabbers.values():
            grabber.stop()
        self._grabbers.clear()
        logger.info("StreamManager stopped")

    def add_camera(self, camera_id: int, source_url: str, expected_fps: int = 30):
        """Add and start a camera."""
        if camera_id in self._grabbers:
            logger.warning("Camera %d already active", camera_id)
            return
        grabber = FrameGrabber(camera_id, source_url, expected_fps)
        grabber.start()
        self._grabbers[camera_id] = grabber

    def remove_camera(self, camera_id: int):
        """Stop and remove a camera."""
        grabber = self._grabbers.pop(camera_id, None)
        if grabber:
            grabber.stop()

    def get_grabber(self, camera_id: int) -> FrameGrabber | None:
        return self._grabbers.get(camera_id)

    def get_all_grabbers(self) -> dict[int, FrameGrabber]:
        return dict(self._grabbers)

    @property
    def active_cameras(self) -> list[int]:
        return list(self._grabbers.keys())
