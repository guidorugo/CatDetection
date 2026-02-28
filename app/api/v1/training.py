import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.core.logging import get_logger
from app.models.training_job import TrainingJob
from app.models.user import User
from app.schemas.training import TrainingJobResponse, TrainingStart

logger = get_logger(__name__)

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


@router.post("/reload-model")
async def reload_model(
    request: Request,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Reload the cat identification model from disk.

    Used after training on a remote server and syncing the model file back.
    Pauses the detection pipeline, loads the new model, rebuilds embeddings,
    and resumes detection with zero downtime.
    """
    model_registry = getattr(request.app.state, "model_registry", None)
    pipeline = getattr(request.app.state, "detection_pipeline", None)
    embedding_store = getattr(request.app.state, "embedding_store", None)

    if not model_registry or not pipeline or not embedding_store:
        raise HTTPException(status_code=503, detail="Detection pipeline not initialized")

    model_path = model_registry.get_active_model_path()
    if not model_path or not Path(model_path).exists():
        raise HTTPException(status_code=404, detail="No active model found on disk")

    version = model_registry.get_active_version()
    logger.info("Reloading model version=%s from %s", version, model_path)

    try:
        # Load the new model into a fresh identifier
        from app.ml.identifier import CatIdentifier

        new_identifier = CatIdentifier()
        await new_identifier.load(model_path)

        # Pause pipeline, swap identifier, rebuild embeddings
        pipeline.pause()
        try:
            pipeline._identifier = new_identifier

            # Rebuild embedding store from DB
            from app.models.cat import Cat, CatEmbedding

            embedding_store.clear()
            result = await db.execute(select(Cat))
            cats = result.scalars().all()
            for cat in cats:
                result = await db.execute(
                    select(CatEmbedding).where(CatEmbedding.cat_id == cat.id)
                )
                cat_embeddings = result.scalars().all()
                if cat_embeddings:
                    embeddings = [
                        np.frombuffer(e.embedding, dtype=np.float32)
                        for e in cat_embeddings
                    ]
                    embedding_store.add_cat(cat.id, cat.name, embeddings)

            logger.info(
                "Model reloaded: version=%s, %d cats, %d embeddings",
                version, embedding_store.cat_count, embedding_store.total_embeddings,
            )
        finally:
            pipeline.resume()

    except Exception as e:
        logger.error("Failed to reload model: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")

    return {
        "status": "ok",
        "model_path": model_path,
        "version": version,
        "cats_loaded": embedding_store.cat_count,
        "embeddings_loaded": embedding_store.total_embeddings,
    }
