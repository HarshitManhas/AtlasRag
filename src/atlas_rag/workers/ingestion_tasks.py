"""Celery tasks for asynchronous document ingestion."""

from __future__ import annotations

import asyncio
import logging
import uuid

from atlas_rag.workers.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="atlas_rag.ingest_document", max_retries=3)
def ingest_document_task(self, document_id: str) -> dict:
    """Celery task that runs the full ingestion pipeline for a document.

    This task is dispatched when a document is uploaded via the API.
    It runs the async ingestion pipeline in a sync context.
    """
    logger.info("Starting ingestion task for document %s", document_id)

    try:
        result = asyncio.run(_run_ingestion(document_id))
        return result
    except Exception as exc:
        logger.exception("Ingestion task failed for document %s", document_id)
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1)) from exc


async def _run_ingestion(document_id: str) -> dict:
    """Async wrapper for the ingestion pipeline."""
    from sqlalchemy import delete, select, update

    from atlas_rag.core.ingestion.pipeline import IngestionPipeline
    from atlas_rag.database import async_session_factory, close_db
    from atlas_rag.models.chunk import Chunk
    from atlas_rag.models.document import Document, DocumentStatus
    from atlas_rag.models.table import ExtractedTable
    from atlas_rag.services.storage import load_document

    pipeline = IngestionPipeline()

    doc_id = uuid.UUID(document_id)
    try:
        async with async_session_factory() as db:
            try:
                # Serialize duplicate deliveries and deletion for this document.
                doc = await db.scalar(
                    select(Document).where(Document.id == doc_id).with_for_update()
                )
                if doc is None:
                    return {"document_id": document_id, "status": "deleted"}
                if doc.status == DocumentStatus.READY:
                    return {"document_id": document_id, "status": "ready"}

                file_content = await asyncio.to_thread(load_document, doc.file_path)
                await db.execute(delete(Chunk).where(Chunk.document_id == doc_id))
                await db.execute(delete(ExtractedTable).where(ExtractedTable.document_id == doc_id))
                result = await pipeline.ingest_document(doc, file_content, db)
                await db.commit()
                return result
            except Exception as exc:
                await db.rollback()
                await db.execute(
                    update(Document)
                    .where(Document.id == doc_id)
                    .values(
                        status=DocumentStatus.FAILED,
                        error_message=str(exc),
                    )
                )
                await db.commit()
                raise
    finally:
        # Each Celery invocation uses a new event loop.
        await close_db()
