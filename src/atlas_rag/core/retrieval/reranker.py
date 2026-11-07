"""Cross-Encoder Reranker — rescores retrieval candidates using pairwise relevance.

Cross-encoders process (query, document) pairs jointly, producing much more
accurate relevance scores than bi-encoder similarity. The trade-off is speed:
cross-encoders are O(N) per query, so we only rerank the top-K candidates
from the initial retrieval stage.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from atlas_rag.config import Settings, get_settings
from atlas_rag.core.retrieval.dense import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks retrieval candidates using a cross-encoder model.

    Loads a cross-encoder model (default: ms-marco-MiniLM-L-6-v2) and scores
    each (query, candidate) pair. Results are re-sorted by cross-encoder score.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any = None

    @property
    def model(self) -> Any:
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        """Load the cross-encoder model."""
        from sentence_transformers import CrossEncoder

        model_name = self.settings.reranker_model
        model = CrossEncoder(model_name, max_length=512)
        logger.info("Loaded cross-encoder reranker: %s", model_name)
        return model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rerank candidates using cross-encoder scoring.

        Args:
            query: The user's query.
            candidates: Retrieval results to rerank.
            top_k: Number of results to return after reranking.

        Returns:
            Reranked list of RetrievalResult, sorted by cross-encoder score.
        """
        top_k = top_k or self.settings.rerank_top_k

        if not candidates:
            return []

        if len(candidates) <= 1:
            return candidates[:top_k]

        try:
            # Build (query, document) pairs
            pairs = [(query, c.content) for c in candidates]

            # Score all pairs
            scores = self.model.predict(pairs, show_progress_bar=False)

            # Combine scores with original results
            scored = list(zip(candidates, scores, strict=True))
            scored.sort(key=lambda x: x[1], reverse=True)

            # Build reranked results
            reranked = []
            for result, score in scored[:top_k]:
                reranked.append(
                    replace(
                        result,
                        score=float(score),
                        source="reranked",
                        metadata={**result.metadata, "original_score": result.score},
                    )
                )

            logger.debug(
                "Reranked %d candidates → top %d (scores: %.3f to %.3f)",
                len(candidates),
                len(reranked),
                reranked[0].score if reranked else 0,
                reranked[-1].score if reranked else 0,
            )
            return reranked

        except Exception:
            logger.exception("Reranking failed, returning original ranking")
            return candidates[:top_k]
