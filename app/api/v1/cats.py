import asyncio
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
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


@router.get("/{cat_id}/profile-image")
async def get_profile_image(
    cat_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(Cat).where(Cat.id == cat_id))
    cat = result.scalar_one_or_none()
    if not cat or not cat.profile_image_path:
        raise HTTPException(status_code=404, detail="Profile image not found")
    path = Path(cat.profile_image_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Profile image file missing")
    return FileResponse(str(path))


def _find_cat_dir(cat_name: str) -> Path:
    """Find the data directory for a cat (case-insensitive)."""
    cat_dir = Path(settings.DATA_DIR) / cat_name
    if cat_dir.is_dir():
        return cat_dir
    # Case-insensitive fallback
    data_root = Path(settings.DATA_DIR)
    if data_root.is_dir():
        for d in data_root.iterdir():
            if d.is_dir() and d.name.lower() == cat_name.lower():
                return d
    return Path(settings.DATA_DIR) / cat_name.lower()


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@router.get("/{cat_id}/images")
async def list_cat_images(
    cat_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """List all reference images for a cat."""
    result = await db.execute(select(Cat).where(Cat.id == cat_id))
    cat = result.scalar_one_or_none()
    if not cat:
        raise HTTPException(status_code=404, detail="Cat not found")

    cat_dir = _find_cat_dir(cat.name)
    if not cat_dir.is_dir():
        return {"images": []}

    images = sorted(
        f.name for f in cat_dir.iterdir()
        if f.is_file() and f.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    )
    return {"images": images}


@router.post("/{cat_id}/images")
async def upload_cat_images(
    cat_id: int,
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Upload one or more reference images for a cat."""
    result = await db.execute(select(Cat).where(Cat.id == cat_id))
    cat = result.scalar_one_or_none()
    if not cat:
        raise HTTPException(status_code=404, detail="Cat not found")

    cat_dir = _find_cat_dir(cat.name)
    cat_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for file in files:
        ext = Path(file.filename).suffix.lower() if file.filename else ".jpg"
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            continue
        # Use original filename, avoid overwriting by appending suffix if needed
        filename = Path(file.filename).name if file.filename else f"upload{ext}"
        dest = cat_dir / filename
        counter = 1
        while dest.exists():
            stem = Path(filename).stem
            dest = cat_dir / f"{stem}_{counter}{ext}"
            counter += 1

        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(dest.name)

    logger.info("Uploaded %d images for cat '%s'", len(saved), cat.name)
    return {"uploaded": saved}


@router.get("/{cat_id}/images/{filename}")
async def get_cat_image(
    cat_id: int,
    filename: str,
    db: AsyncSession = Depends(get_db),
):
    """Serve a reference image for a cat."""
    result = await db.execute(select(Cat).where(Cat.id == cat_id))
    cat = result.scalar_one_or_none()
    if not cat:
        raise HTTPException(status_code=404, detail="Cat not found")

    # Prevent path traversal
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    cat_dir = _find_cat_dir(cat.name)
    file_path = cat_dir / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(file_path))


@router.delete("/{cat_id}/images/{filename}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_cat_image(
    cat_id: int,
    filename: str,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Delete a reference image for a cat."""
    result = await db.execute(select(Cat).where(Cat.id == cat_id))
    cat = result.scalar_one_or_none()
    if not cat:
        raise HTTPException(status_code=404, detail="Cat not found")

    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    cat_dir = _find_cat_dir(cat.name)
    file_path = cat_dir / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    file_path.unlink()
    logger.info("Deleted image '%s' for cat '%s'", filename, cat.name)


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
        cat_dir = _find_cat_dir(cat.name)
        if not cat_dir.is_dir():
            logger.warning("No data directory for cat '%s'", cat.name)
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
