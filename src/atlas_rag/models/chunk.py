"""Chunk ORM model — represents an indexed text segment from a document."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from atlas_rag.database import Base


class Chunk(Base):
    """A text segment extracted from a document, indexed for retrieval."""

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_embedding_id", "embedding_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_header: Mapped[str | None] = mapped_column(String(512), nullable=True)
    embedding_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="chunks")  # noqa: F821

    def __repr__(self) -> str:
        return f"<Chunk doc={self.document_id} idx={self.chunk_index}>"
