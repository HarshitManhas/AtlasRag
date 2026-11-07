"""ExtractedTable ORM model — a structured table parsed from a document page."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from atlas_rag.database import Base


class ExtractedTable(Base):
    """A table detected and extracted from a document, stored as structured data
    for SQL-based querying and cell-level citation."""

    __tablename__ = "extracted_tables"
    __table_args__ = (Index("ix_extracted_tables_document_id", "document_id"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    table_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)

    # Structured table data
    headers: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    data: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)

    row_count: Mapped[int] = mapped_column(Integer, default=0)
    col_count: Mapped[int] = mapped_column(Integer, default=0)

    # Column type inference results (e.g., {"Revenue": "numeric", "Product": "text"})
    column_types: Mapped[dict[str, str] | None] = mapped_column(JSON, nullable=True)

    # Raw markdown/text representation for embedding
    text_representation: Mapped[str | None] = mapped_column(Text, nullable=True)

    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="tables")  # noqa: F821

    def __repr__(self) -> str:
        return (
            f"<ExtractedTable doc={self.document_id} idx={self.table_index} rows={self.row_count}>"
        )
