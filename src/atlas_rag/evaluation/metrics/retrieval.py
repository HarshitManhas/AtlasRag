"""Retrieval quality metrics: Precision@K, Recall@K, MRR, NDCG@K."""

from __future__ import annotations

import math
from collections.abc import Sequence


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Compute Precision@K: fraction of top-K retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@K: fraction of all relevant items captured in top-K."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: set[str]) -> float:
    """Compute Reciprocal Rank (RR): 1 / rank of the first relevant item."""
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    all_retrieved: Sequence[Sequence[str]], all_relevant: Sequence[set[str]]
) -> float:
    """Compute Mean Reciprocal Rank (MRR) across multiple queries."""
    if not all_retrieved:
        return 0.0
    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant, strict=True)
    ]
    return sum(scores) / len(scores)


def ndcg_at_k(retrieved_ids: Sequence[str], relevance_scores: dict[str, float], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at K (NDCG@K)."""
    if k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    # Calculate DCG
    dcg = 0.0
    for i, item_id in enumerate(top_k):
        rel = relevance_scores.get(item_id, 0.0)
        dcg += rel / math.log2(i + 2)

    # Calculate Ideal DCG (IDCG)
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_scores))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
