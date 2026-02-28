from datetime import datetime
from pydantic import BaseModel


class CameraCreate(BaseModel):
    name: str
    source_url: str
    source_type: str = "rtsp"
    location: str | None = None
    expected_fps: int = 30
    resolution: str | None = None


class CameraUpdate(BaseModel):
    name: str | None = None
    source_url: str | None = None
    source_type: str | None = None
    location: str | None = None
    expected_fps: int | None = None
    resolution: str | None = None
    is_enabled: bool | None = None


class CameraResponse(BaseModel):
    id: int
    name: str
    source_url: str
    source_type: str
    location: str | None
    expected_fps: int
    resolution: str | None
    is_enabled: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CameraHealth(BaseModel):
    camera_id: int
    status: str  # connected, degraded, disconnected, offline
    actual_fps: float | None = None
    frame_latency_ms: float | None = None
    reconnect_attempts: int = 0
    last_frame_at: datetime | None = None
