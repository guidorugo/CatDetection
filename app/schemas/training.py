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
    # Optional per-job remote server overrides (fall back to .env settings)
    server_ssh: str | None = None       # e.g. "user@10.0.0.82"
    server_port: int | None = None      # e.g. 8001
    server_dir: str | None = None       # e.g. "~/cat-detection-project"
    api_key: str | None = None


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
