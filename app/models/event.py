from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    camera_id: Mapped[int] = mapped_column(Integer, ForeignKey("cameras.id"))
    cat_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("cats.id"), nullable=True
    )  # null = unknown cat
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    detection_confidence: Mapped[float] = mapped_column(Float)
    identification_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_x: Mapped[int] = mapped_column(Integer)
    bbox_y: Mapped[int] = mapped_column(Integer)
    bbox_w: Mapped[int] = mapped_column(Integer)
    bbox_h: Mapped[int] = mapped_column(Integer)
    thumbnail_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    recording_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("recordings.id"), nullable=True
    )

    camera = relationship("Camera", foreign_keys=[camera_id])
    cat = relationship("Cat", foreign_keys=[cat_id])
    recording = relationship("Recording", foreign_keys=[recording_id])
