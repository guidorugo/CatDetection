from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.core.logging import get_logger
from app.ml.model_registry import ModelRegistry
from app.models.user import User

logger = get_logger(__name__)

router = APIRouter()


@router.get("")
async def list_models(
    _: User = Depends(get_current_user),
):
    """List all registered model versions with metrics and active status."""
    registry = ModelRegistry()
    return registry.list_models()


@router.post("/{version}/activate")
async def activate_model(
    version: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    _: User = Depends(get_current_user),
):
    """Activate a model version and hot-reload it into the pipeline."""
    registry = ModelRegistry()
    registry_data = registry.list_models()

    if version not in registry_data.get("models", {}):
        raise HTTPException(status_code=404, detail=f"Model version '{version}' not found")

    if registry_data.get("active") == version:
        return {"status": "ok", "message": f"Version '{version}' is already active"}

    # Activate in registry
    model_path = registry.activate_model(version)

    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model file not found: {model_path}")

    # Hot-reload into pipeline
    pipeline = getattr(request.app.state, "detection_pipeline", None)
    embedding_store = getattr(request.app.state, "embedding_store", None)

    if pipeline and embedding_store:
        from app.ml.identifier import CatIdentifier
        from app.models.cat import Cat, CatEmbedding

        new_identifier = CatIdentifier()
        await new_identifier.load(model_path)

        pipeline.pause()
        try:
            pipeline._identifier = new_identifier

            # Rebuild embedding store
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
                "Model activated: version=%s, %d cats, %d embeddings",
                version, embedding_store.cat_count, embedding_store.total_embeddings,
            )
        finally:
            pipeline.resume()

    return {
        "status": "ok",
        "version": version,
        "model_path": model_path,
    }
