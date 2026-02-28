from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending, running, completed, failed
    model_type: Mapped[str] = mapped_column(String(50))  # e.g. "cat_reid"
    model_version: Mapped[str | None] = mapped_column(String(50), nullable=True)
    config: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON config
    epochs_total: Mapped[int] = mapped_column(Integer, default=50)
    epochs_completed: Mapped[int] = mapped_column(Integer, default=0)
    best_metric: Mapped[float | None] = mapped_column(Float, nullable=True)
    loss_history: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
