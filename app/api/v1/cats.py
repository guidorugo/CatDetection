import asyncio
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.core.logging import get_logger
from app.models.cat import Cat, CatEmbedding
from app.models.event import DetectionEvent
from app.models.user import User
from app.schemas.cat import CatCreate, CatResponse, CatUpdate
from app.schemas.event import EventResponse

logger = get_logger(__name__)

router = APIRouter()

MAX_IMAGES_PER_CAT = 10


@router.get("/", response_model=list[CatResponse])
async def list_cats(
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(Cat).order_by(Cat.name))
    return result.scalars().all()


@router.post("/", response_model=CatResponse, status_code=status.HTTP_201_CREATED)
async def create_cat(
    data: CatCreate,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(Cat).where(Cat.name == data.name))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Cat name already exists")
    cat = Cat(**data.model_dump())
    db.add(cat)
    await db.commit()
    await db.refresh(cat)
    return cat


@router.get("/{cat_id}", response_model=CatResponse)
async def get_cat(
    cat_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(Cat).where(Cat.id == cat_id))
    cat = result.scalar_one_or_none()
    if not cat:
        raise HTTPException(status_code=404, detail="Cat not found")
    return cat


@router.put("/{cat_id}", response_model=CatResponse)
async def update_cat(
    cat_id: int,
    data: CatUpdate,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(Cat).where(Cat.id == cat_id))
    cat = result.scalar_one_or_none()
    if not cat:
        raise HTTPException(status_code=404, detail="Cat not found")
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(cat, field, value)
    await db.commit()
    await db.refresh(cat)
    return cat


@router.delete("/{cat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_cat(
    cat_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(Cat).where(Cat.id == cat_id))
    cat = result.scalar_one_or_none()
    if not cat:
        raise HTTPException(status_code=404, detail="Cat not found")
    await db.delete(cat)
    await db.commit()


@router.post("/{cat_id}/profile-image", response_model=CatResponse)
async def upload_profile_image(
    cat_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(Cat).where(Cat.id == cat_id))
    cat = result.scalar_one_or_none()
    if not cat:
        raise HTTPException(status_code=404, detail="Cat not found")

    upload_dir = Path(settings.THUMBNAILS_DIR) / "cats"
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix if file.filename else ".jpg"
    file_path = upload_dir / f"{cat.name}{ext}"
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    cat.profile_image_path = str(file_path)
    await db.commit()
    await db.refresh(cat)
    return cat


@router.get("/{cat_id}/events", response_model=list[EventResponse])
async def get_cat_events(
    cat_id: int,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(
        select(DetectionEvent)
        .where(DetectionEvent.cat_id == cat_id)
        .order_by(DetectionEvent.timestamp.desc())
        .limit(limit)
        .offset(offset)
    )
    return result.scalars().all()


async def generate_embeddings(detector, identifier, embedding_store, db: AsyncSession):
    """Generate reference embeddings for all cats from training data images.

    For each cat in the DB, find matching images in data/{cat_name}/,
    run YOLO detection to crop, generate embeddings via the identifier,
    and store them in the DB + embedding store.

    Returns a summary dict.
    """
    model_registry_mod = __import__("app.ml.model_registry", fromlist=["ModelRegistry"])
    registry = model_registry_mod.ModelRegistry()
    model_version = registry.get_active_version() or "unknown"

    result = await db.execute(select(Cat).order_by(Cat.name))
    cats = result.scalars().all()

    summary = {}
    for cat in cats:
        # Find images in data/{cat_name}/
        cat_dir = Path(settings.DATA_DIR) / cat.name
        if not cat_dir.is_dir():
            logger.warning("No data directory for cat '%s' at %s", cat.name, cat_dir)
            summary[cat.name] = 0
            continue

        image_files = sorted(
            [f for f in cat_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        )
        if not image_files:
            logger.warning("No images found for cat '%s'", cat.name)
            summary[cat.name] = 0
            continue

        # Sample up to MAX_IMAGES_PER_CAT
        if len(image_files) > MAX_IMAGES_PER_CAT:
            image_files = random.sample(image_files, MAX_IMAGES_PER_CAT)

        # Clear old embeddings for this cat
        await db.execute(
            delete(CatEmbedding).where(CatEmbedding.cat_id == cat.id)
        )

        embeddings = []
        for img_path in image_files:
            frame = await asyncio.to_thread(cv2.imread, str(img_path))
            if frame is None:
                continue

            # Detect cat in image to get a crop
            detections = await detector.detect(frame)
            if not detections:
                # No detection — use full image as fallback
                crop = frame
            else:
                # Use the largest detection (most likely the main subject)
                best = max(detections, key=lambda d: d["bbox"][2] * d["bbox"][3])
                x, y, w, h = best["bbox"]
                crop = frame[y : y + h, x : x + w]

            if crop.size == 0:
                continue

            embedding = await identifier.get_embedding(crop)
            embeddings.append(embedding)

            # Store in DB
            db.add(CatEmbedding(
                cat_id=cat.id,
                embedding=embedding.tobytes(),
                model_version=model_version,
                source_image_path=str(img_path),
            ))

        await db.commit()

        # Update in-memory store
        if embeddings:
            embedding_store.remove_cat(cat.id)
            embedding_store.add_cat(cat.id, cat.name, embeddings)

        summary[cat.name] = len(embeddings)
        logger.info("Generated %d embeddings for cat '%s'", len(embeddings), cat.name)

    return summary


@router.post("/generate-embeddings")
async def generate_embeddings_endpoint(
    request: Request,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Generate reference embeddings for all cats from training data images."""
    detector = getattr(request.app.state, "detector", None)
    identifier = getattr(request.app.state, "identifier", None)
    embedding_store = getattr(request.app.state, "embedding_store", None)

    if not detector or not identifier or not identifier.model:
        raise HTTPException(
            status_code=503,
            detail="Detection pipeline or identifier model not loaded",
        )

    summary = await generate_embeddings(detector, identifier, embedding_store, db)

    total = sum(summary.values())
    logger.info("Embedding generation complete: %d total embeddings for %d cats", total, len(summary))

    return {
        "status": "ok",
        "total_embeddings": total,
        "cats": summary,
    }
