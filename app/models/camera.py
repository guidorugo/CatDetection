from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    source_url: Mapped[str] = mapped_column(String(500))
    source_type: Mapped[str] = mapped_column(String(20), default="rtsp")  # rtsp, mjpeg, http
    location: Mapped[str | None] = mapped_column(String(200), nullable=True)
    expected_fps: Mapped[int] = mapped_column(Integer, default=30)
    resolution: Mapped[str | None] = mapped_column(String(20), nullable=True)  # e.g. "1920x1080"
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
