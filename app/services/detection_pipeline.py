import asyncio
import time

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
from app.ml.detector import CatDetector
from app.ml.embeddings import EmbeddingStore
from app.ml.identifier import CatIdentifier
from app.services.event_service import EventService
from app.services.notification_service import NotificationService
from app.services.recording_service import RecordingService
from app.services.stream_manager import StreamManager

logger = get_logger(__name__)


class DetectionPipeline:
    """Main detection loop: Frame -> Detect -> Identify -> Record -> Notify."""

    def __init__(
        self,
        stream_manager: StreamManager,
        detector: CatDetector,
        identifier: CatIdentifier | None,
        embedding_store: EmbeddingStore,
        recording_service: RecordingService,
        notification_service: NotificationService,
        event_service: EventService,
    ):
        self._stream_manager = stream_manager
        self._detector = detector
        self._identifier = identifier
        self._embedding_store = embedding_store
        self._recording_service = recording_service
        self._notifications = notification_service
        self._event_service = event_service
        self._running = False
        self._task: asyncio.Task | None = None
        self._paused = False

    async def start(self):
        """Start the detection pipeline."""
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("DetectionPipeline started")

    async def stop(self):
        """Stop the detection pipeline."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("DetectionPipeline stopped")

    def pause(self):
        """Pause detection (e.g., during training)."""
        self._paused = True
        logger.info("DetectionPipeline paused")

    def resume(self):
        """Resume detection."""
        self._paused = False
        logger.info("DetectionPipeline resumed")

    async def _run(self):
        """Main detection loop."""
        frame_interval = 1.0 / settings.DETECTION_FPS

        while self._running:
            if self._paused:
                await asyncio.sleep(1.0)
                continue

            loop_start = time.monotonic()

            for camera_id, grabber in self._stream_manager.get_all_grabbers().items():
                frame, frame_time = grabber.get_frame()
                if frame is None:
                    continue

                # Feed frame to recording buffer
                buffer = self._recording_service.get_or_create_buffer(camera_id)
                buffer.add_frame(frame, frame_time)

                try:
                    await self._process_frame(camera_id, frame)
                except Exception as e:
                    logger.error("Error processing frame from camera %d: %s", camera_id, e)

            # Maintain target FPS
            elapsed = time.monotonic() - loop_start
            if elapsed < frame_interval:
                await asyncio.sleep(frame_interval - elapsed)

    async def _process_frame(self, camera_id: int, frame: np.ndarray):
        """Process a single frame: detect, identify, record, notify."""
        # Detect cats
        detections = await self._detector.detect(frame)
        if not detections:
            return

        for det in detections:
            bbox = det["bbox"]
            confidence = det["confidence"]
            x, y, w, h = bbox

            # Crop for identification
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            # Identify cat
            cat_id = None
            cat_name = None
            id_confidence = None

            if self._identifier and self._identifier.model is not None:
                embedding = await self._identifier.get_embedding(crop)
                cat_id, cat_name, similarity = self._embedding_store.find_match(embedding)
                if cat_id is not None:
                    id_confidence = similarity

            # Trigger recording
            grabber = self._stream_manager.get_grabber(camera_id)
            recording_id = None
            if grabber:
                recording_id = await self._recording_service.trigger_recording(
                    camera_id, grabber.get_frame
                )

            # Save event
            event_id = await self._event_service.create_event(
                camera_id=camera_id,
                cat_id=cat_id,
                detection_confidence=confidence,
                identification_confidence=id_confidence,
                bbox=bbox,
                frame=frame,
                recording_id=recording_id,
            )

            # Annotate frame for live view
            label = cat_name or "unknown cat"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.0%}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            # Broadcast event
            await self._notifications.broadcast_event(
                {
                    "type": "detection",
                    "event_id": event_id,
                    "camera_id": camera_id,
                    "cat_id": cat_id,
                    "cat_name": cat_name,
                    "confidence": round(confidence, 3),
                    "identification_confidence": round(id_confidence, 3) if id_confidence else None,
                    "bbox": list(bbox),
                }
            )

        # Send annotated frame to live viewers
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        await self._notifications.broadcast_frame(camera_id, jpeg.tobytes())
