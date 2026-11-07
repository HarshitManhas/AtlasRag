"""Pydantic request/response schemas for API endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field

# ─── Auth Schemas ────────────────────────────────────────────────────────────


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=1, max_length=255)


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: UUID
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Document Schemas ────────────────────────────────────────────────────────


class DocumentResponse(BaseModel):
    id: UUID
    title: str
    filename: str
    file_size: int
    mime_type: str
    status: str
    page_count: int
    chunk_count: int
    table_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


# ─── Query Schemas ───────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    document_ids: list[UUID] | None = Field(None, description="Filter to specific documents")
    collection_id: UUID | None = None
    top_k: int = Field(5, ge=1, le=20)
    stream: bool = False
    include_hallucination_check: bool = True


class SourceReference(BaseModel):
    source_type: str  # "chunk" or "table_cell"
    document_id: str | None = None
    page_number: int | None = None
    section_header: str | None = None
    content_preview: str | None = None
    table_id: str | None = None
    row_index: int | None = None
    column: str | None = None
    cell_value: Any | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[SourceReference]
    references: str
    groundedness_score: float | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Collection Schemas ──────────────────────────────────────────────────────


class CollectionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    is_public: bool = False


class CollectionResponse(BaseModel):
    id: UUID
    name: str
    description: str | None
    is_public: bool
    document_count: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}
