"""QueryLog ORM model — tracks every RAG query for analytics and auditing."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from atlas_rag.database import Base


class QueryLog(Base):
    """Logs each RAG query including retrieval results, generated answer,
    citations, and performance metrics."""

    __tablename__ = "query_logs"
    __table_args__ = (
        Index("ix_query_logs_user_id", "user_id"),
        Index("ix_query_logs_created_at", "created_at"),
        Index("ix_query_logs_trace_id", "trace_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_results: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    generated_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    groundedness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count_out: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<QueryLog {self.id} query={self.query_text[:50]!r}>"
