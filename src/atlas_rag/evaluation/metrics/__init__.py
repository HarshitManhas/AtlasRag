"""Metrics package."""

from atlas_rag.evaluation.metrics.citation import citation_precision, citation_recall
from atlas_rag.evaluation.metrics.groundedness import (
    compute_groundedness_ratio,
    compute_hallucination_rate,
)
from atlas_rag.evaluation.metrics.retrieval import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from atlas_rag.evaluation.metrics.table import cell_citation_accuracy, numerical_match

__all__ = [
    "cell_citation_accuracy",
    "citation_precision",
    "citation_recall",
    "compute_groundedness_ratio",
    "compute_hallucination_rate",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "numerical_match",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
