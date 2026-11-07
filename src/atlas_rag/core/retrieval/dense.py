"""Dense Retrieval — semantic vector search via Qdrant."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from atlas_rag.config import Settings, get_settings
from atlas_rag.core.ingestion.embedder import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval hit from any search backend."""

    content: str
    score: float
    document_id: str
    chunk_index: int
    page_number: int | None = None
    section_header: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source: str = "dense"  # "dense", "sparse", "fused", "reranked"


class DenseRetriever:
    """Performs semantic vector search using Qdrant."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedder = EmbeddingGenerator(self.settings)

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Search for chunks semantically similar to the query.

        Args:
            query: Natural language query.
            top_k: Number of results to return.
            document_ids: Optional filter to specific documents.

        Returns:
            List of RetrievalResult ordered by similarity score (descending).
        """
        if document_ids == []:
            return []
        top_k = top_k or self.settings.retrieval_top_k

        # Embed the query
        query_vector = self.embedder.embed_text(query)

        # Build Qdrant filter
        from qdrant_client import QdrantClient
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        client = QdrantClient(url=self.settings.qdrant_url)

        query_filter = None
        if document_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchAny(any=document_ids),
                    )
                ]
            )

        try:
            hits = client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            ).points
        finally:
            client.close()

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append(
                RetrievalResult(
                    content=payload.get("content", ""),
                    score=hit.score,
                    document_id=payload.get("document_id", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    page_number=payload.get("page_number"),
                    section_header=payload.get("section_header"),
                    metadata=payload.get("metadata", {}),
                    source="dense",
                )
            )

        logger.debug("Dense search for %r returned %d results", query[:50], len(results))
        return results
