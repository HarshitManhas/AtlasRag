"""Citation quality metrics: Citation Precision, Citation Recall, and Cell Alignment."""

from __future__ import annotations

from collections.abc import Sequence

from atlas_rag.core.generation.citation import Citation


def citation_precision(citations: Sequence[Citation], verified_citations: set[int]) -> float:
    """Fraction of emitted citations that correctly verify the associated statement."""
    if not citations:
        return 1.0
    valid_count = sum(1 for c in citations if c.citation_id in verified_citations)
    return valid_count / len(citations)


def citation_recall(total_verifiable_claims: int, claims_with_valid_citations: int) -> float:
    """Fraction of verifiable claims that have at least one valid citation."""
    if total_verifiable_claims <= 0:
        return 1.0
    return min(1.0, claims_with_valid_citations / total_verifiable_claims)
