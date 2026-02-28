from datetime import datetime
from pydantic import BaseModel


class CatCreate(BaseModel):
    name: str
    description: str | None = None


class CatUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class CatResponse(BaseModel):
    id: int
    name: str
    description: str | None
    profile_image_path: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
