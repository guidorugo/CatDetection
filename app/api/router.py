from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.api.v1 import auth, cameras, cats, events, recordings, training, ws
from app.core.config import settings

api_router = APIRouter()


@api_router.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}


api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(cameras.router, prefix="/cameras", tags=["cameras"])
api_router.include_router(cats.router, prefix="/cats", tags=["cats"])
api_router.include_router(events.router, prefix="/events", tags=["events"])
api_router.include_router(recordings.router, prefix="/recordings", tags=["recordings"])
api_router.include_router(training.router, prefix="/training", tags=["training"])
api_router.include_router(ws.router, prefix="/ws", tags=["websocket"])

templates_dir = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

page_router = APIRouter()


@page_router.get("/")
async def index():
    return RedirectResponse(url="/dashboard")


@page_router.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@page_router.get("/dashboard")
async def dashboard_page(request: Request):
    return templates.TemplateResponse(
        "dashboard.html", {"request": request, "active_page": "dashboard"}
    )


@page_router.get("/cameras")
async def cameras_page(request: Request):
    return templates.TemplateResponse(
        "cameras.html", {"request": request, "active_page": "cameras"}
    )


@page_router.get("/cats")
async def cats_page(request: Request):
    return templates.TemplateResponse("cats.html", {"request": request, "active_page": "cats"})


@page_router.get("/events")
async def events_page(request: Request):
    return templates.TemplateResponse(
        "events.html", {"request": request, "active_page": "events"}
    )


@page_router.get("/recordings")
async def recordings_page(request: Request):
    return templates.TemplateResponse(
        "recordings.html", {"request": request, "active_page": "recordings"}
    )


@page_router.get("/training")
async def training_page(request: Request):
    remote_configured = bool(settings.TRAINING_SERVER_SSH and settings.TRAINING_API_KEY)
    return templates.TemplateResponse(
        "training.html", {
            "request": request,
            "active_page": "training",
            "remote_configured": remote_configured,
        }
    )
