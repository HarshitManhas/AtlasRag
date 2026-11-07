"""Collection ORM model for organizing documents."""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from atlas_rag.database import Base


class Collection(Base):
    """A named group of documents with shared access control."""

    __tablename__ = "collections"
    __table_args__ = (Index("ix_collections_owner_id", "owner_id"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    owner: Mapped["User"] = relationship(back_populates="collections")  # noqa: F821
    documents: Mapped[list["Document"]] = relationship(  # noqa: F821
        back_populates="collection"
    )

    def __repr__(self) -> str:
        return f"<Collection {self.name!r}>"
