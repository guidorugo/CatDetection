import asyncio
from pathlib import Path

import numpy as np

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CatDetector:
    """YOLO-based cat detector. Filters for COCO class 15 (cat)."""

    CAT_CLASS_ID = 15

    def __init__(self):
        self.model = None
        self._model_path = settings.YOLO_MODEL

    async def load(self):
        """Load YOLO model (downloads if needed)."""
        logger.info("Loading YOLO model: %s", self._model_path)
        self.model = await asyncio.to_thread(self._load_model)
        logger.info("YOLO model loaded successfully")

    def _load_model(self):
        from ultralytics import YOLO

        model = YOLO(self._model_path)
        # Try TensorRT export if available
        tensorrt_path = Path(self._model_path).with_suffix(".engine")
        if tensorrt_path.exists():
            logger.info("Loading TensorRT engine: %s", tensorrt_path)
            model = YOLO(str(tensorrt_path))
        return model

    async def detect(self, frame: np.ndarray) -> list[dict]:
        """Detect cats in a frame.

        Returns list of dicts with keys: bbox (x,y,w,h), confidence, class_id
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = await asyncio.to_thread(
            self.model.predict,
            frame,
            conf=settings.DETECTION_CONFIDENCE,
            classes=[self.CAT_CLASS_ID],
            verbose=False,
        )

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append(
                    {
                        "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                        "confidence": float(box.conf[0]),
                        "class_id": int(box.cls[0]),
                    }
                )
        return detections

    async def export_tensorrt(self, fp16: bool = True):
        """Export model to TensorRT format."""
        if self.model is None:
            await self.load()
        logger.info("Exporting YOLO to TensorRT (FP16=%s)...", fp16)
        await asyncio.to_thread(self.model.export, format="engine", half=fp16)
        logger.info("TensorRT export complete")
