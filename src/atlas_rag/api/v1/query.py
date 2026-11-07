"""Query API endpoint — the primary RAG interface."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atlas_rag.api.middleware.auth import get_current_user
from atlas_rag.api.middleware.prompt_guard import guard_query
from atlas_rag.core.generation.pipeline import GenerationPipeline
from atlas_rag.core.hallucination import HallucinationDetector
from atlas_rag.core.retrieval.pipeline import RetrievalPipeline
from atlas_rag.core.table_analysis.pipeline import TableAnalysisPipeline
from atlas_rag.database import get_db
from atlas_rag.models.collection import Collection
from atlas_rag.models.document import Document, DocumentStatus
from atlas_rag.models.query_log import QueryLog
from atlas_rag.models.user import User
from atlas_rag.schemas.api import QueryRequest, QueryResponse, SourceReference

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])

_retrieval: RetrievalPipeline | None = None
_generation: GenerationPipeline | None = None
_table_pipeline: TableAnalysisPipeline | None = None
_hallucination: HallucinationDetector | None = None


def _get_retrieval() -> RetrievalPipeline:
    global _retrieval
    if _retrieval is None:
        _retrieval = RetrievalPipeline()
    return _retrieval


def _get_generation() -> GenerationPipeline:
    global _generation
    if _generation is None:
        _generation = GenerationPipeline()
    return _generation


def _get_table_pipeline() -> TableAnalysisPipeline:
    global _table_pipeline
    if _table_pipeline is None:
        _table_pipeline = TableAnalysisPipeline()
    return _table_pipeline


@router.post("", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """Execute a RAG query against indexed documents.

    Pipeline: Query → Prompt Guard → Hybrid Retrieval → Table Analysis →
    Generation → Citation → Hallucination Check → Response
    """
    start = time.perf_counter()

    # Step 1: Guard against prompt injection
    clean_query = guard_query(request.query)

    if request.stream:
        raise HTTPException(status_code=422, detail="Streaming is not supported by this endpoint")

    stmt = select(Document).where(Document.owner_id == user.id)
    if request.collection_id is not None:
        collection = await db.scalar(
            select(Collection).where(
                Collection.id == request.collection_id,
                Collection.owner_id == user.id,
            )
        )
        if collection is None:
            raise HTTPException(status_code=404, detail="Collection not found")
        stmt = stmt.where(Document.collection_id == request.collection_id)
    if request.document_ids is not None:
        stmt = stmt.where(Document.id.in_(request.document_ids))
    documents = (await db.scalars(stmt)).all()
    if request.document_ids is not None and {doc.id for doc in documents} != set(
        request.document_ids
    ):
        raise HTTPException(status_code=404, detail="Document not found")
    document_ids = [str(doc.id) for doc in documents if doc.status == DocumentStatus.READY]
    if not document_ids:
        return QueryResponse(
            answer="No indexed documents available in this scope.",
            citations=[],
            references="",
            metadata={"chunks_retrieved": 0},
        )

    # Step 2: Retrieval
    retrieval = _get_retrieval()
    try:
        context = await retrieval.retrieve(
            query=clean_query,
            document_ids=document_ids,
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.exception("Retrieval unavailable")
        raise HTTPException(status_code=503, detail="Search unavailable") from exc

    # Step 3: Table analysis (if applicable)
    if document_ids:
        table_pipeline = _get_table_pipeline()
        table_result = await table_pipeline.analyze_query(
            query=clean_query,
            document_ids=document_ids,
            db=db,
        )
        if table_result.confidence > 0.3:
            context.table_analysis = table_result

    # Step 4: Generation
    generation = _get_generation()
    try:
        gen_result = await generation.generate(context)
    except Exception as exc:
        logger.exception("Answer generation unavailable")
        raise HTTPException(status_code=503, detail="Answer generation unavailable") from exc

    # Step 5: Hallucination check
    warnings: list[str] = []
    groundedness_score = None

    if request.include_hallucination_check and context.results:
        global _hallucination
        if _hallucination is None:
            _hallucination = HallucinationDetector(mode="keyword")
        report = await _hallucination.check(gen_result.answer, context.results)
        groundedness_score = report.groundedness_score
        warnings = report.warnings

    # Step 6: Build response
    citations: list[SourceReference] = []
    for cite in gen_result.citations:
        if cite.source_type == "chunk":
            citations.append(
                SourceReference(
                    source_type="chunk",
                    document_id=cite.document_id,
                    page_number=cite.page_number,
                    section_header=cite.section_header,
                    content_preview=cite.chunk_content,
                )
            )
        elif cite.source_type == "table_cell" and cite.cell_reference:
            ref = cite.cell_reference
            citations.append(
                SourceReference(
                    source_type="table_cell",
                    page_number=ref.page,
                    table_id=ref.table_id,
                    row_index=ref.row_index,
                    column=ref.column,
                    cell_value=ref.value,
                )
            )

    elapsed = (time.perf_counter() - start) * 1000

    # Log query
    query_log = QueryLog(
        id=uuid.uuid4(),
        user_id=user.id,
        query_text=clean_query,
        generated_answer=gen_result.answer,
        citations=[c.model_dump() for c in citations],
        groundedness_score=groundedness_score,
        latency_ms=int(elapsed),
        token_count_in=gen_result.llm_response.tokens_in if gen_result.llm_response else None,
        token_count_out=gen_result.llm_response.tokens_out if gen_result.llm_response else None,
    )
    db.add(query_log)

    return QueryResponse(
        answer=gen_result.answer,
        citations=citations,
        references=gen_result.references,
        groundedness_score=groundedness_score,
        warnings=warnings,
        metadata={
            "latency_ms": round(elapsed, 2),
            "retrieval_latency_ms": context.latency_ms,
            "chunks_retrieved": context.final_count,
            **(gen_result.metadata or {}),
        },
    )
