"""Document ORM model for tracking uploaded files and their processing status."""

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from atlas_rag.database import Base


class DocumentStatus(str, enum.Enum):
    """Processing status of a document."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(Base):
    """Represents an uploaded document and its processing state."""

    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_owner_id", "owner_id"),
        Index("ix_documents_collection_id", "collection_id"),
        Index("ix_documents_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)

    owner_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    collection_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("collections.id", ondelete="SET NULL"),
        nullable=True,
    )

    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus), default=DocumentStatus.PENDING, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    table_count: Mapped[int] = mapped_column(Integer, default=0)

    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    owner: Mapped["User"] = relationship(back_populates="documents")  # noqa: F821
    collection: Mapped["Collection | None"] = relationship(  # noqa: F821
        back_populates="documents"
    )
    chunks: Mapped[list["Chunk"]] = relationship(  # noqa: F821
        back_populates="document", cascade="all, delete-orphan"
    )
    tables: Mapped[list["ExtractedTable"]] = relationship(  # noqa: F821
        back_populates="document", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Document {self.title!r} status={self.status.value}>"
