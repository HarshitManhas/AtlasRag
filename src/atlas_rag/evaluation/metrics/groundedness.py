"""Groundedness and Hallucination metrics."""

from __future__ import annotations

from collections.abc import Sequence

from atlas_rag.core.hallucination.detector import Claim, ClaimVerdict


def compute_groundedness_ratio(claims: Sequence[Claim]) -> float:
    """Fraction of claims that are supported or partially supported by evidence."""
    if not claims:
        return 1.0
    supported = sum(
        1.0
        if c.verdict == ClaimVerdict.SUPPORTED
        else (0.5 if c.verdict == ClaimVerdict.PARTIALLY_SUPPORTED else 0.0)
        for c in claims
    )
    return supported / len(claims)


def compute_hallucination_rate(claims: Sequence[Claim]) -> float:
    """Fraction of claims identified as unsupported (hallucinated)."""
    if not claims:
        return 0.0
    unsupported = sum(1 for c in claims if c.verdict == ClaimVerdict.NOT_SUPPORTED)
    return unsupported / len(claims)
