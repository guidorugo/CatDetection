import asyncio
import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.core.database import async_session
from app.core.logging import get_logger
from app.models.training_job import TrainingJob
from app.models.user import User
from app.schemas.training import TrainingJobResponse, TrainingStart

logger = get_logger(__name__)

router = APIRouter()


async def _reload_model_after_training(app, model_path: str):
    """Reload identifier model and rebuild embedding store."""
    pipeline = getattr(app.state, "detection_pipeline", None)
    model_registry = getattr(app.state, "model_registry", None)
    embedding_store = getattr(app.state, "embedding_store", None)
    if model_registry and embedding_store and pipeline:
        from app.ml.identifier import CatIdentifier
        from app.models.cat import Cat, CatEmbedding

        new_identifier = CatIdentifier()
        await new_identifier.load(model_path)
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


async def _run_remote_training_job(job_id: int, data: TrainingStart, app) -> None:
    """Run training on the remote GPU server: rsync → train → download model → reload."""
    import re
    import subprocess
    import sys

    import httpx

    pipeline = getattr(app.state, "detection_pipeline", None)

    server_ssh = data.server_ssh or settings.TRAINING_SERVER_SSH
    server_port = data.server_port or settings.TRAINING_SERVER_PORT
    api_key = data.api_key or settings.TRAINING_API_KEY
    server_dir = data.server_dir or settings.TRAINING_SERVER_DIR
    base_url = f"http://{server_ssh.split('@')[-1]}:{server_port}"

    # Update job to running
    async with async_session() as db:
        result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
        job = result.scalar_one()
        job.status = "running"
        await db.commit()

    try:
        # Step 1: rsync data/ to remote server
        logger.info("Rsyncing data to %s:%s/data/", server_ssh, server_dir)
        rsync_result = await asyncio.to_thread(
            subprocess.run,
            [
                "rsync", "-az", "--delete",
                "-e", "ssh -o StrictHostKeyChecking=accept-new",
                str(settings.DATA_DIR) + "/",
                f"{server_ssh}:{server_dir}/data/",
            ],
            capture_output=True, text=True, timeout=600,
        )
        if rsync_result.returncode != 0:
            raise RuntimeError(f"rsync failed:\n{rsync_result.stderr[-500:]}")
        logger.info("Data rsync complete")

        # Step 2: Trigger training on the remote server
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{base_url}/prepare-and-train",
                params={"epochs": data.epochs},
                headers={"X-API-Key": api_key},
            )
            if resp.status_code == 409:
                raise RuntimeError("Remote server already has a training job running")
            resp.raise_for_status()
        logger.info("Remote training triggered (epochs=%d)", data.epochs)

        # Step 3: Poll status every 10s
        while True:
            await asyncio.sleep(10)
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{base_url}/status",
                    headers={"X-API-Key": api_key},
                )
                resp.raise_for_status()
                remote_status = resp.json()

            rs = remote_status["status"]
            progress_str = remote_status.get("progress")

            # Parse progress like "12/50 epochs"
            epochs_done = 0
            if progress_str:
                m = re.match(r"(\d+)/(\d+)", progress_str)
                if m:
                    epochs_done = int(m.group(1))

            # Update local DB with progress
            async with async_session() as db:
                result = await db.execute(
                    select(TrainingJob).where(TrainingJob.id == job_id)
                )
                job = result.scalar_one()
                job.epochs_completed = epochs_done
                await db.commit()

            if rs == "complete":
                logger.info("Remote training complete")
                break
            elif rs == "error":
                raise RuntimeError(f"Remote training failed: {remote_status.get('error')}")

        # Step 4: Download model and registry from server
        save_dir = Path(settings.MODELS_DIR) / "identification"
        save_dir.mkdir(parents=True, exist_ok=True)

        async with httpx.AsyncClient(timeout=120) as client:
            # Download model file
            resp = await client.get(
                f"{base_url}/model/latest",
                headers={"X-API-Key": api_key},
            )
            resp.raise_for_status()
            # Get filename from content-disposition or use default
            filename = "cat_reid_remote.pth"
            cd = resp.headers.get("content-disposition", "")
            if "filename=" in cd:
                filename = cd.split("filename=")[-1].strip('" ')
            model_path = str(save_dir / filename)
            Path(model_path).write_bytes(resp.content)
            logger.info("Downloaded model to %s", model_path)

            # Download registry and merge
            resp = await client.get(
                f"{base_url}/model/registry",
                headers={"X-API-Key": api_key},
            )
            resp.raise_for_status()
            remote_registry = resp.json()

        # Fix paths in registry to point to local model
        from app.ml.model_registry import ModelRegistry

        registry = ModelRegistry()
        version = remote_status.get("model_version")
        if version and version in remote_registry.get("models", {}):
            registry.register_model(
                version=version,
                path=model_path,
                metrics=remote_registry["models"][version].get("metrics", {}),
            )
            registry.activate_model(version)
            logger.info("Registered remote model version %s", version)

        # Update job as completed
        async with async_session() as db:
            result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
            job = result.scalar_one()
            job.status = "completed"
            job.model_version = version
            job.model_path = model_path
            job.epochs_completed = data.epochs
            await db.commit()

        # Reload model
        if pipeline:
            pipeline.pause()
            try:
                await _reload_model_after_training(app, model_path)
            finally:
                pipeline.resume()

    except Exception as e:
        logger.error("Remote training job %d failed: %s", job_id, e)
        async with async_session() as db:
            result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
            job = result.scalar_one()
            job.status = "failed"
            job.error_message = str(e)
            await db.commit()


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

        # Prepare data (YOLO crop + train/val/test split)
        data_dir = str(settings.DATA_DIR)
        processed_dir = str(Path(data_dir) / "processed")
        train_dir = Path(processed_dir) / "train"

        if data.prepare_data or not train_dir.exists():
            logger.info("Preparing training data (YOLO cropping)...")
            import shutil
            import subprocess
            import sys

            # Clear old processed data for a clean re-crop
            if Path(processed_dir).exists():
                shutil.rmtree(processed_dir)

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
        else:
            logger.info("Skipping data preparation (using existing processed data)")

        from app.ml.training.trainer import CatReIDTrainer

        trainer = CatReIDTrainer(
            data_dir=processed_dir,
            epochs=data.epochs,
            learning_rate=data.learning_rate,
            freeze_epochs=data.freeze_epochs,
        )

        # Store trainer reference for cancellation
        app.state._current_trainer = trainer

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

        # Clear trainer reference
        app.state._current_trainer = None

        # Check if training was cancelled
        if trainer._cancel:
            async with async_session() as db:
                result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
                job = result.scalar_one()
                job.status = "cancelled"
                await db.commit()
            logger.info("Training job %d cancelled", job_id)
            return

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

        # Reload model into identifier
        await _reload_model_after_training(app, results["model_path"])

    except Exception as e:
        logger.error("Training job %d failed: %s", job_id, e)
        app.state._current_trainer = None
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
    if data.training_location == "remote":
        server_ssh = data.server_ssh or settings.TRAINING_SERVER_SSH
        api_key = data.api_key or settings.TRAINING_API_KEY
        if not server_ssh or not api_key:
            raise HTTPException(
                status_code=400,
                detail="Remote training not configured (server SSH and API key required)",
            )
        asyncio.create_task(_run_remote_training_job(job.id, data, request.app))
        logger.info("Remote training job %d started (epochs=%d)", job.id, data.epochs)
    else:
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


@router.post("/jobs/{job_id}/cancel")
async def cancel_training_job(
    job_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Cancel a running or pending training job."""
    result = await db.execute(select(TrainingJob).where(TrainingJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")
    if job.status not in ("pending", "running"):
        raise HTTPException(status_code=400, detail=f"Cannot cancel job with status '{job.status}'")

    # For local training, signal the trainer to stop
    trainer = getattr(request.app.state, "_current_trainer", None)
    if trainer:
        trainer.cancel()

    # Mark as cancelled (for pending jobs or as a fallback)
    job.status = "cancelled"
    await db.commit()
    await db.refresh(job)

    logger.info("Training job %d cancelled", job_id)
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
