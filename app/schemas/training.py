from pydantic import BaseModel
from datetime import datetime


class TrainingStart(BaseModel):
    model_type: str = "cat_reid"
    epochs: int = 50
    learning_rate: float = 0.001
    batch_size: int = 24
    freeze_epochs: int = 10
    prepare_data: bool = True
    training_location: str = "local"  # "local" or "remote"


class TrainingJobResponse(BaseModel):
    id: int
    status: str
    model_type: str
    model_version: str | None
    epochs_total: int
    epochs_completed: int
    best_metric: float | None
    model_path: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
