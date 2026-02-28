from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Cat(Base):
    __tablename__ = "cats"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)
    profile_image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    embeddings: Mapped[list["CatEmbedding"]] = relationship(
        back_populates="cat", cascade="all, delete-orphan"
    )


class CatEmbedding(Base):
    __tablename__ = "cat_embeddings"

    id: Mapped[int] = mapped_column(primary_key=True)
    cat_id: Mapped[int] = mapped_column(Integer, ForeignKey("cats.id", ondelete="CASCADE"))
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    model_version: Mapped[str] = mapped_column(String(50))
    source_image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    cat: Mapped["Cat"] = relationship(back_populates="embeddings")
