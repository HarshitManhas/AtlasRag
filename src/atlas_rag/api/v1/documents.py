"""Document management API endpoints."""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from atlas_rag.api.middleware.auth import get_current_user
from atlas_rag.database import get_db
from atlas_rag.models.collection import Collection
from atlas_rag.models.document import Document, DocumentStatus
from atlas_rag.models.user import User
from atlas_rag.schemas.api import DocumentListResponse, DocumentResponse
from atlas_rag.services.storage import remove_document, store_document
from atlas_rag.workers.ingestion_tasks import ingest_document_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/html",
    "text/markdown",
    "text/plain",
}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = None,
    collection_id: uuid.UUID | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Upload a document for ingestion and indexing."""
    # Validate MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Supported: {ALLOWED_MIME_TYPES}",
        )

    if collection_id is not None:
        collection = await db.scalar(
            select(Collection).where(
                Collection.id == collection_id,
                Collection.owner_id == user.id,
            )
        )
        if collection is None:
            raise HTTPException(status_code=404, detail="Collection not found")

    content = await file.read(MAX_FILE_SIZE + 1)
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024 * 1024)}MB",
        )

    doc_id = uuid.uuid4()
    file_path = f"documents/{user.id}/{doc_id}"

    doc = Document(
        id=doc_id,
        title=title or file.filename or "Untitled",
        filename=file.filename or "unknown",
        file_path=file_path,
        file_size=len(content),
        mime_type=file.content_type or "application/octet-stream",
        owner_id=user.id,
        collection_id=collection_id,
        status=DocumentStatus.PENDING,
    )
    db.add(doc)
    await db.flush()

    try:
        await asyncio.to_thread(store_document, file_path, content, doc.mime_type)
    except Exception as exc:
        logger.exception("Document storage failed")
        raise HTTPException(status_code=503, detail="Document storage unavailable") from exc

    try:
        # The worker must be able to read the row before it receives the task.
        await db.commit()
    except Exception:
        await db.rollback()
        await asyncio.to_thread(remove_document, file_path)
        raise

    response = DocumentResponse.model_validate(doc)
    try:
        await asyncio.to_thread(ingest_document_task.delay, str(doc_id))
    except Exception as exc:
        doc.status = DocumentStatus.FAILED
        doc.error_message = "Unable to enqueue ingestion"
        await db.commit()
        logger.exception("Unable to enqueue document %s", doc_id)
        raise HTTPException(
            status_code=503,
            detail={"message": "Ingestion queue unavailable", "document_id": str(doc_id)},
        ) from exc

    return response


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    skip: int = 0,
    limit: int = 20,
    status_filter: str | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    """List documents owned by or accessible to the current user."""
    stmt = select(Document).where(Document.owner_id == user.id)

    if status_filter:
        stmt = stmt.where(Document.status == status_filter)

    # Count total
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar() or 0

    # Fetch page
    stmt = stmt.order_by(Document.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    docs = result.scalars().all()

    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in docs],
        total=total,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """Get a single document by ID."""
    stmt = select(Document).where(
        Document.id == document_id,
        Document.owner_id == user.id,
    )
    result = await db.execute(stmt)
    doc = result.scalar_one_or_none()

    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    return DocumentResponse.model_validate(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a document and all associated chunks, tables, and indexes."""
    stmt = select(Document).where(
        Document.id == document_id,
        Document.owner_id == user.id,
    )
    result = await db.execute(stmt.with_for_update())
    doc = result.scalar_one_or_none()

    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    from atlas_rag.core.ingestion.pipeline import IngestionPipeline

    try:
        await IngestionPipeline().remove_indexes(str(doc.id))
        await asyncio.to_thread(remove_document, doc.file_path)
    except Exception as exc:
        logger.exception("Document cleanup failed for %s", doc.id)
        raise HTTPException(
            status_code=503, detail="Document cleanup unavailable; retry deletion"
        ) from exc
    await db.delete(doc)
