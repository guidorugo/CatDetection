import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.models.cat import Cat
from app.models.event import DetectionEvent
from app.models.user import User
from app.schemas.cat import CatCreate, CatResponse, CatUpdate
from app.schemas.event import EventResponse

router = APIRouter()


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
