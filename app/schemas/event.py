from datetime import datetime
from pydantic import BaseModel


class EventResponse(BaseModel):
    id: int
    camera_id: int
    cat_id: int | None
    cat_name: str | None = None
    camera_name: str | None = None
    timestamp: datetime
    detection_confidence: float
    identification_confidence: float | None
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    thumbnail_path: str | None
    recording_id: int | None

    model_config = {"from_attributes": True}


class EventFilter(BaseModel):
    camera_id: int | None = None
    cat_id: int | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None


class EventStats(BaseModel):
    total_events: int
    events_today: int
    cats_detected: dict[str, int]  # cat_name -> count
