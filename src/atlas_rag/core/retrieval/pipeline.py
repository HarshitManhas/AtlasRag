"""Retrieval Pipeline — orchestrates hybrid search, fusion, and reranking.

This is the single entry point for all retrieval operations. It:
  1. Runs dense (vector) and sparse (BM25) search in parallel
  2. Fuses results using Reciprocal Rank Fusion
  3. Reranks fused results with a cross-encoder
  4. Checks for table-related queries and routes to Table Analysis
  5. Returns the final context package for the generation stage
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from atlas_rag.config import Settings, get_settings
from atlas_rag.core.retrieval.dense import DenseRetriever, RetrievalResult
from atlas_rag.core.retrieval.fusion import ReciprocalRankFusion
from atlas_rag.core.retrieval.reranker import CrossEncoderReranker
from atlas_rag.core.retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """Complete retrieval output passed to the generation stage."""

    query: str
    results: list[RetrievalResult]
    table_analysis: Any | None = None  # TableAnalysisResult if table query
    dense_count: int = 0
    sparse_count: int = 0
    fused_count: int = 0
    final_count: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RetrievalPipeline:
    """Orchestrates hybrid retrieval with fusion and reranking.

    Query flow:
        query → [Dense Search] ──┐
                                 ├─→ [RRF Fusion] → [Cross-Encoder Rerank] → results
        query → [BM25 Search]  ──┘

    Optionally routes table queries to the Table Analysis pipeline.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dense = DenseRetriever(self.settings)
        self.sparse = SparseRetriever(self.settings)
        self.fusion = ReciprocalRankFusion(settings=self.settings)
        self.reranker = CrossEncoderReranker(self.settings)

    async def retrieve(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> RetrievalContext:
        """Execute the full hybrid retrieval pipeline.

        Args:
            query: User's natural language query.
            document_ids: Optional filter to specific documents.
            top_k: Final number of results to return.

        Returns:
            RetrievalContext with ranked results and metadata.
        """
        if document_ids == []:
            return RetrievalContext(query=query, results=[])
        start = time.perf_counter()
        retrieval_top_k = self.settings.retrieval_top_k
        final_top_k = top_k or self.settings.rerank_top_k

        # Step 1: Run dense and sparse search in parallel
        dense_task = asyncio.create_task(
            self.dense.search(query, top_k=retrieval_top_k, document_ids=document_ids)
        )
        sparse_task = asyncio.create_task(
            self.sparse.search(query, top_k=retrieval_top_k, document_ids=document_ids)
        )

        dense_results, sparse_results = await asyncio.gather(
            dense_task, sparse_task, return_exceptions=True
        )

        # Handle partial failures gracefully
        if isinstance(dense_results, Exception) and isinstance(sparse_results, Exception):
            raise RuntimeError("Both search backends failed") from dense_results
        if isinstance(dense_results, Exception):
            logger.error("Dense retrieval failed: %s", dense_results)
            dense_results = []
        if isinstance(sparse_results, Exception):
            logger.error("Sparse retrieval failed: %s", sparse_results)
            sparse_results = []

        # Step 2: Fuse with RRF
        dense_weight = self.settings.dense_weight
        sparse_weight = 1.0 - dense_weight

        fused = self.fusion.fuse(
            dense_results,
            sparse_results,
            top_k=retrieval_top_k,
            weights=[dense_weight, sparse_weight],
        )

        # Step 3: Rerank with cross-encoder
        reranked = self.reranker.rerank(
            query=query,
            candidates=fused,
            top_k=final_top_k,
        )

        elapsed = (time.perf_counter() - start) * 1000

        context = RetrievalContext(
            query=query,
            results=reranked,
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
            fused_count=len(fused),
            final_count=len(reranked),
            latency_ms=round(elapsed, 2),
        )

        logger.info(
            "Retrieval for %r: dense=%d sparse=%d fused=%d reranked=%d (%.1fms)",
            query[:60],
            context.dense_count,
            context.sparse_count,
            context.fused_count,
            context.final_count,
            context.latency_ms,
        )
        return context
