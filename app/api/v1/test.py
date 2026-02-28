import asyncio

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.api.deps import get_current_user
from app.core.logging import get_logger
from app.models.user import User

logger = get_logger(__name__)

router = APIRouter()


@router.post("/detect")
async def detect_cats_in_image(
    request: Request,
    file: UploadFile = File(...),
    _: User = Depends(get_current_user),
):
    """Upload a photo, detect and identify cats, return results with bounding boxes."""
    detector = getattr(request.app.state, "detector", None)
    identifier = getattr(request.app.state, "identifier", None)
    embedding_store = getattr(request.app.state, "embedding_store", None)

    if not detector or not embedding_store:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Detection pipeline not initialized",
        )

    # Read and decode image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = await asyncio.to_thread(cv2.imdecode, nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Detect cats
    detections_raw = await detector.detect(frame)

    # Identify each detected cat
    detections = []
    for det in detections_raw:
        x, y, w, h = det["bbox"]
        crop = frame[y : y + h, x : x + w]

        cat_id = None
        cat_name = None
        similarity = 0.0

        if identifier and identifier.model is not None and crop.size > 0:
            embedding = await identifier.get_embedding(crop)
            cat_id, cat_name, similarity = embedding_store.find_match(embedding)

        detections.append(
            {
                "bbox": [x, y, w, h],
                "confidence": det["confidence"],
                "cat_id": cat_id,
                "cat_name": cat_name,
                "similarity": round(similarity, 4),
            }
        )

    return {"detections": detections}
