"""Sparse Retrieval — BM25 keyword search via Elasticsearch."""

from __future__ import annotations

import logging
from typing import Any

from atlas_rag.config import Settings, get_settings
from atlas_rag.core.retrieval.dense import RetrievalResult

logger = logging.getLogger(__name__)


class SparseRetriever:
    """Performs BM25 keyword search using Elasticsearch."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievalResult]:
        """Search for chunks matching the query using BM25.

        Args:
            query: Natural language query (used as keyword search).
            top_k: Number of results to return.
            document_ids: Optional filter to specific documents.

        Returns:
            List of RetrievalResult ordered by BM25 score (descending).
        """
        if document_ids == []:
            return []
        from elasticsearch import AsyncElasticsearch

        top_k = top_k or self.settings.retrieval_top_k

        es = AsyncElasticsearch(self.settings.elasticsearch_url)

        try:
            # Build query
            must_clauses: list[dict[str, Any]] = [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^3", "section_header"],
                        "type": "best_fields",
                    }
                }
            ]

            filter_clauses: list[dict[str, Any]] = []
            if document_ids:
                filter_clauses.append({"terms": {"document_id": document_ids}})

            body: dict[str, Any] = {
                "query": {
                    "bool": {
                        "must": must_clauses,
                        "filter": filter_clauses,
                    }
                },
                "size": top_k,
            }

            response = await es.search(
                index=self.settings.elasticsearch_index,
                body=body,
            )

            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                results.append(
                    RetrievalResult(
                        content=source.get("content", ""),
                        score=hit["_score"],
                        document_id=source.get("document_id", ""),
                        chunk_index=source.get("chunk_index", 0),
                        page_number=source.get("page_number"),
                        section_header=source.get("section_header"),
                        source="sparse",
                    )
                )

            logger.debug("BM25 search for %r returned %d results", query[:50], len(results))
            return results

        finally:
            await es.close()
