from datetime import datetime
from pydantic import BaseModel


class RecordingResponse(BaseModel):
    id: int
    camera_id: int
    file_path: str
    start_time: datetime
    end_time: datetime | None
    duration: float | None
    file_size: int | None
    resolution: str | None
    fps: int | None
    codec: str
    status: str

    model_config = {"from_attributes": True}
