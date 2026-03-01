import asyncio
import time
from typing import Annotated

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.api.v1.cats import _find_cat_dir
from app.core.logging import get_logger
from app.models.cat import Cat, CatEmbedding
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


@router.post("/feedback")
async def submit_detection_feedback(
    request: Request,
    file: UploadFile = File(...),
    bbox_x: Annotated[int, Form()] = 0,
    bbox_y: Annotated[int, Form()] = 0,
    bbox_w: Annotated[int, Form()] = 0,
    bbox_h: Annotated[int, Form()] = 0,
    cat_id: Annotated[int, Form()] = 0,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Submit feedback on a detection: assign the correct cat identity.

    Crops the image at the given bbox, generates an embedding, and stores it
    as a reference embedding for the specified cat.
    """
    identifier = getattr(request.app.state, "identifier", None)
    embedding_store = getattr(request.app.state, "embedding_store", None)

    if not identifier or not identifier.model:
        raise HTTPException(status_code=503, detail="Identifier model not loaded")

    # Verify cat exists
    result = await db.execute(select(Cat).where(Cat.id == cat_id))
    cat = result.scalar_one_or_none()
    if not cat:
        raise HTTPException(status_code=404, detail="Cat not found")

    # Decode image and crop
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = await asyncio.to_thread(cv2.imdecode, nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    crop = frame[bbox_y : bbox_y + bbox_h, bbox_x : bbox_x + bbox_w]
    if crop.size == 0:
        raise HTTPException(status_code=400, detail="Invalid bounding box")

    # Save crop to data/{cat_name}/ as a reference image
    cat_dir = _find_cat_dir(cat.name)
    cat_dir.mkdir(parents=True, exist_ok=True)
    filename = f"feedback_{int(time.time() * 1000)}.jpg"
    crop_path = cat_dir / filename
    await asyncio.to_thread(cv2.imwrite, str(crop_path), crop)

    # Generate and store embedding
    embedding = await identifier.get_embedding(crop)

    from app.ml.model_registry import ModelRegistry
    registry = ModelRegistry()
    model_version = registry.get_active_version() or "unknown"

    db.add(CatEmbedding(
        cat_id=cat.id,
        embedding=embedding.tobytes(),
        model_version=model_version,
        source_image_path=str(crop_path),
    ))
    await db.commit()

    # Rebuild in-memory embeddings for this cat from DB
    if embedding_store:
        emb_result = await db.execute(
            select(CatEmbedding).where(CatEmbedding.cat_id == cat.id)
        )
        all_embeddings = [
            np.frombuffer(e.embedding, dtype=np.float32)
            for e in emb_result.scalars().all()
        ]
        embedding_store.remove_cat(cat.id)
        if all_embeddings:
            embedding_store.add_cat(cat.id, cat.name, all_embeddings)

    logger.info("Feedback: stored crop '%s' and embedding for cat '%s' (id=%d)", filename, cat.name, cat.id)

    return {"status": "ok", "cat_id": cat.id, "cat_name": cat.name}
