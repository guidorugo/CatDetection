from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router, page_router
from app.core.config import settings
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting %s", settings.APP_NAME)

    # Ensure directories exist
    for d in [settings.MODELS_DIR, settings.RECORDINGS_DIR, settings.THUMBNAILS_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Lazy-import ML and service modules (they depend on torch/cv2/ultralytics)
    try:
        import numpy as np
        from sqlalchemy import select

        from app.core.database import async_session
        from app.ml.detector import CatDetector
        from app.ml.embeddings import EmbeddingStore
        from app.ml.identifier import CatIdentifier
        from app.ml.model_registry import ModelRegistry
        from app.services.camera_health import CameraHealthMonitor
        from app.services.detection_pipeline import DetectionPipeline
        from app.services.event_service import EventService
        from app.services.notification_service import NotificationService
        from app.services.recording_service import RecordingService
        from app.services.stream_manager import StreamManager

        # Initialize services
        stream_manager = StreamManager()
        notification_service = NotificationService()
        recording_service = RecordingService()
        event_service = EventService()
        embedding_store = EmbeddingStore()
        model_registry = ModelRegistry()
        detector = CatDetector()
        identifier = CatIdentifier()

        # Load ML models
        try:
            await detector.load()
        except Exception as e:
            logger.warning("Failed to load YOLO model (will retry on first use): %s", e)

        model_path = model_registry.get_active_model_path()
        if model_path and Path(model_path).exists():
            try:
                await identifier.load(model_path)
                # Load embeddings from DB
                from app.models.cat import Cat, CatEmbedding

                async with async_session() as db:
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
            except Exception as e:
                logger.warning("Failed to load identifier model: %s", e)

        # Start services
        await stream_manager.start()

        health_monitor = CameraHealthMonitor(stream_manager, notification_service)
        await health_monitor.start()

        detection_pipeline = DetectionPipeline(
            stream_manager=stream_manager,
            detector=detector,
            identifier=identifier if identifier.model else None,
            embedding_store=embedding_store,
            recording_service=recording_service,
            notification_service=notification_service,
            event_service=event_service,
        )
        await detection_pipeline.start()

        # Store services on app state for access from routes
        app.state.stream_manager = stream_manager
        app.state.notification_service = notification_service
        app.state.detection_pipeline = detection_pipeline
        app.state.health_monitor = health_monitor
        app.state.embedding_store = embedding_store
        app.state.model_registry = model_registry
        app.state.detector = detector
        app.state.identifier = identifier

        # Resume any orphaned remote training jobs
        from app.api.v1.training import resume_orphaned_jobs

        await resume_orphaned_jobs(app)

        yield

        # Shutdown services
        logger.info("Shutting down %s", settings.APP_NAME)
        await detection_pipeline.stop()
        await health_monitor.stop()
        await recording_service.stop_all()
        await stream_manager.stop()

    except ImportError as e:
        logger.warning("ML/service dependencies not available: %s", e)
        logger.info("Running in API-only mode (no detection pipeline)")
        yield
        logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

# Static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# API routes
app.include_router(api_router, prefix="/api/v1")

# Page routes
app.include_router(page_router)
