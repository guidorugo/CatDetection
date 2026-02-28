from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.core.config import settings
from app.core.database import async_session
from app.core.logging import get_logger
from app.models.event import DetectionEvent

logger = get_logger(__name__)


class EventService:
    """Handles detection event persistence and thumbnail generation."""

    async def create_event(
        self,
        camera_id: int,
        cat_id: int | None,
        detection_confidence: float,
        identification_confidence: float | None,
        bbox: tuple[int, int, int, int],
        frame: np.ndarray,
        recording_id: int | None = None,
    ) -> int:
        """Create a detection event with thumbnail. Returns event id."""
        # Save thumbnail
        thumbnail_path = self._save_thumbnail(camera_id, bbox, frame)

        async with async_session() as db:
            event = DetectionEvent(
                camera_id=camera_id,
                cat_id=cat_id,
                timestamp=datetime.now(timezone.utc),
                detection_confidence=detection_confidence,
                identification_confidence=identification_confidence,
                bbox_x=bbox[0],
                bbox_y=bbox[1],
                bbox_w=bbox[2],
                bbox_h=bbox[3],
                thumbnail_path=thumbnail_path,
                recording_id=recording_id,
            )
            db.add(event)
            await db.commit()
            await db.refresh(event)
            return event.id

    def _save_thumbnail(
        self,
        camera_id: int,
        bbox: tuple[int, int, int, int],
        frame: np.ndarray,
    ) -> str:
        """Save a cropped thumbnail of the detection."""
        thumb_dir = Path(settings.THUMBNAILS_DIR) / "events"
        thumb_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"cam{camera_id}_{timestamp}.jpg"
        path = thumb_dir / filename

        x, y, w, h = bbox
        # Add some padding
        pad = 20
        y1 = max(0, y - pad)
        x1 = max(0, x - pad)
        y2 = min(frame.shape[0], y + h + pad)
        x2 = min(frame.shape[1], x + w + pad)
        crop = frame[y1:y2, x1:x2]

        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return str(path)
