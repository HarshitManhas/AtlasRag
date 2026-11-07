"""Reciprocal Rank Fusion (RRF) — merges results from multiple retrieval systems.

RRF combines ranked lists from dense (vector) and sparse (BM25) search into a
single fused ranking. It is robust to score scale differences between systems
because it operates on rank positions rather than raw scores.

Formula: RRF_score(d) = Σ  1 / (k + rank_i(d))
         over all systems i

Reference: Cormack, Clarke & Buettcher (2009) — "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods"
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import replace

from atlas_rag.config import Settings, get_settings
from atlas_rag.core.retrieval.dense import RetrievalResult

logger = logging.getLogger(__name__)


class ReciprocalRankFusion:
    """Fuses ranked lists from multiple retrieval systems using RRF.

    The k parameter controls how much influence top-ranked documents have.
    Higher k → more uniform weighting; lower k → more emphasis on top ranks.
    Default k=60 as recommended in the original paper.
    """

    def __init__(self, k: int = 60, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.k = k or self.settings.rrf_k

    def fuse(
        self,
        *result_lists: list[RetrievalResult],
        top_k: int | None = None,
        weights: list[float] | None = None,
    ) -> list[RetrievalResult]:
        """Fuse multiple ranked result lists into a single ranking.

        Args:
            *result_lists: Variable number of ranked result lists.
            top_k: Number of results to return after fusion.
            weights: Optional per-list weights (e.g., [0.5, 0.5] for equal).
                     Defaults to uniform weighting.

        Returns:
            Fused and re-ranked list of RetrievalResult.
        """
        top_k = top_k or self.settings.rerank_top_k
        n_lists = len(result_lists)

        if n_lists == 0:
            return []
        if n_lists == 1:
            return result_lists[0][:top_k]

        if weights is None:
            weights = [1.0] * n_lists
        elif len(weights) != n_lists:
            raise ValueError(
                f"weights length ({len(weights)}) must match number of result lists ({n_lists})"
            )

        # Build RRF scores
        # Key: (document_id, chunk_index) → unique identifier for a chunk
        rrf_scores: dict[tuple[str, int], float] = defaultdict(float)
        best_result: dict[tuple[str, int], RetrievalResult] = {}

        for list_idx, results in enumerate(result_lists):
            weight = weights[list_idx]
            for rank, result in enumerate(results):
                key = (result.document_id, result.chunk_index)
                rrf_scores[key] += weight * (1.0 / (self.k + rank + 1))

                # Keep the result with the most metadata
                if key not in best_result or len(result.content) > len(best_result[key].content):
                    best_result[key] = result

        # Sort by RRF score descending
        sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

        # Build fused results
        fused: list[RetrievalResult] = []
        for key in sorted_keys[:top_k]:
            result = best_result[key]
            fused.append(
                replace(
                    result,
                    score=rrf_scores[key],
                    source="fused",
                )
            )

        logger.debug(
            "RRF fused %d lists (%s) → %d results",
            n_lists,
            "+".join(str(len(r)) for r in result_lists),
            len(fused),
        )
        return fused
