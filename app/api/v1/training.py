import asyncio
import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.core.database import async_session
from app.core.logging import get_logger
from app.models.training_job import TrainingJob
from app.models.user import User
from app.schemas.training import TrainingJobResponse, TrainingStart

logger = get_logger(__name__)

router = APIRouter()


async def _run_training_job(job_id: int, data: TrainingStart, app) -> None:
    """Run training in background: pause pipeline → train → register → reload → resume."""
    pipeline = getattr(app.state, "detection_pipeline", None)

    # Update job to running
    async with async_session() as db:
        result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        job = result.scalar_one()
        job.status = "running"
        await db.commit()

    try:
        # Pause detection to free GPU memory
        if pipeline:
            pipeline.pause()
            logger.info("Detection pipeline paused for training")

        # Prepare data (YOLO crop + train/val/test split) if not already done
        from app.core.config import settings

        data_dir = str(settings.DATA_DIR)
        processed_dir = str(Path(data_dir) / "processed")
        train_dir = Path(processed_dir) / "train"

        if not train_dir.exists():
            logger.info("Preparing training data (YOLO cropping)...")
            import subprocess
            import sys

            result = await asyncio.to_thread(
                subprocess.run,
                [
                    sys.executable, "scripts/prepare_data.py",
                    "--data-dir", data_dir,
                    "--output-dir", processed_dir,
                ],
                capture_output=True, text=True, timeout=1800,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Data preparation failed:\n{result.stderr[-500:]}")
            logger.info("Data preparation complete")

        from app.ml.training.trainer import CatReIDTrainer

        trainer = CatReIDTrainer(
            data_dir=processed_dir,
            epochs=data.epochs,
            learning_rate=data.learning_rate,
            freeze_epochs=data.freeze_epochs,
        )

        # Progress callback updates DB (called from trainer thread)
        def on_progress(epoch, total, loss, rank1, mAP):
            async def _update():
                async with async_session() as db:
                    result = await db.execute(
                        select(TrainingJob).where(TrainingJob.id == job_id)
                    )
                    job = result.scalar_one()
                    job.epochs_completed = epoch
                    job.best_metric = rank1
                    history = json.loads(job.loss_history) if job.loss_history else []
                    history.append({"epoch": epoch, "loss": float(loss), "rank1": float(rank1), "mAP": float(mAP)})
                    job.loss_history = json.dumps(history)
                    await db.commit()

            # Schedule the async update from the sync callback
            loop = asyncio.get_event_loop()
            asyncio.run_coroutine_threadsafe(_update(), loop).result(timeout=10)

        results = await asyncio.to_thread(trainer.train, progress_callback=on_progress)

        # Register model
        from app.ml.model_registry import ModelRegistry

        registry = ModelRegistry()
        registry.register_model(
            version=results["version"],
            path=results["model_path"],
            metrics={"rank1": results["best_rank1"]},
        )
        registry.activate_model(results["version"])
        logger.info("Registered model version %s", results["version"])

        # Update job as completed
        async with async_session() as db:
            result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
            job = result.scalar_one()
            job.status = "completed"
            job.model_version = results["version"]
            job.model_path = results["model_path"]
            job.best_metric = results["best_rank1"]
            job.loss_history = json.dumps(results["loss_history"])
            await db.commit()

        # Reload model into identifier (same logic as /reload-model)
        model_registry = getattr(app.state, "model_registry", None)
        embedding_store = getattr(app.state, "embedding_store", None)
        if model_registry and embedding_store and pipeline:
            from app.ml.identifier import CatIdentifier
            from app.models.cat import Cat, CatEmbedding

            new_identifier = CatIdentifier()
            await new_identifier.load(results["model_path"])
            pipeline._identifier = new_identifier

            embedding_store.clear()
            async with async_session() as db:
                cat_result = await db.execute(select(Cat))
                cats = cat_result.scalars().all()
                for cat in cats:
                    emb_result = await db.execute(
                        select(CatEmbedding).where(CatEmbedding.cat_id == cat.id)
                    )
                    cat_embeddings = emb_result.scalars().all()
                    if cat_embeddings:
                        embeddings = [
                            np.frombuffer(e.embedding, dtype=np.float32)
                            for e in cat_embeddings
                        ]
                        embedding_store.add_cat(cat.id, cat.name, embeddings)

            logger.info("Model reloaded after training: %d cats, %d embeddings",
                        embedding_store.cat_count, embedding_store.total_embeddings)

    except Exception as e:
        logger.error("Training job %d failed: %s", job_id, e)
        async with async_session() as db:
            result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
            job = result.scalar_one()
            job.status = "failed"
            job.error_message = str(e)
            await db.commit()
    finally:
        if pipeline:
            pipeline.resume()
            logger.info("Detection pipeline resumed")


@router.post("/start", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED)
async def start_training(
    request: Request,
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

    # Launch training in background
    asyncio.create_task(_run_training_job(job.id, data, request.app))
    logger.info("Training job %d started (epochs=%d)", job.id, data.epochs)

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
