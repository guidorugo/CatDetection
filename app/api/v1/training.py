import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.models.training_job import TrainingJob
from app.models.user import User
from app.schemas.training import TrainingJobResponse, TrainingStart

router = APIRouter()


@router.post("/start", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
async def start_training(
    data: TrainingStart,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    # Check if training is already running
    result = await db.execute(
        select(TrainingJob).where(TrainingJob.status.in_(["pending", "running"]))
    )
    if result.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="A training job is already running")

    job = TrainingJob(
        model_type=data.model_type,
        epochs_total=data.epochs,
        config=json.dumps(data.model_dump()),
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Training will be started by the training service (Phase 4)
    return job


@router.get("/jobs", response_model=list[TrainingJobResponse])
async def list_training_jobs(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(
        select(TrainingJob).order_by(TrainingJob.created_at.desc()).limit(limit)
    )
    return result.scalars().all()


@router.get("/jobs/{job_id}", response_model=TrainingJobResponse)
async def get_training_job(
    job_id: int,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    return job
