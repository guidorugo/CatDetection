from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class HyperparamSearch(Base):
    __tablename__ = "hyperparam_searches"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending, running, completed, failed, cancelled
    param_grid: Mapped[str] = mapped_column(Text)  # JSON: parameter lists
    training_location: Mapped[str] = mapped_column(String(20), default="local")
    base_config: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON: remote settings
    total_trials: Mapped[int] = mapped_column(Integer, default=0)
    completed_trials: Mapped[int] = mapped_column(Integer, default=0)
    failed_trials: Mapped[int] = mapped_column(Integer, default=0)
    best_trial_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    best_metric: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    trials: Mapped[list["TrainingJob"]] = relationship(  # noqa: F821
        back_populates="search", order_by="TrainingJob.trial_number"
    )
