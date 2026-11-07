"""Hallucination Detection — validates generated claims against retrieved evidence.

Decomposes LLM-generated answers into atomic claims, then checks each claim
against the retrieved evidence using NLI-based entailment scoring.

Outputs:
  - Per-claim verdict: SUPPORTED / PARTIALLY_SUPPORTED / NOT_SUPPORTED
  - Overall groundedness score (0.0 - 1.0)
  - List of unsupported claims for user warning
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from atlas_rag.core.retrieval.dense import RetrievalResult

logger = logging.getLogger(__name__)


class ClaimVerdict(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_SUPPORTED = "not_supported"


@dataclass
class Claim:
    """An atomic claim extracted from a generated answer."""

    text: str
    citation_ids: list[int] = field(default_factory=list)
    verdict: ClaimVerdict = ClaimVerdict.NOT_SUPPORTED
    confidence: float = 0.0
    supporting_evidence: list[str] = field(default_factory=list)


@dataclass
class HallucinationReport:
    """Complete hallucination analysis for a generated answer."""

    claims: list[Claim]
    groundedness_score: float  # 0.0 - 1.0
    supported_count: int = 0
    partial_count: int = 0
    unsupported_count: int = 0
    warnings: list[str] = field(default_factory=list)


class HallucinationDetector:
    """Detects hallucinations by validating claims against evidence.

    Two-stage approach:
      1. Decompose the answer into atomic claims
      2. Check each claim against retrieved evidence using NLI or keyword overlap

    Can use either:
      - A local NLI model (DeBERTa) for entailment scoring
      - LLM-as-judge for more nuanced verification
      - Fast keyword overlap as a lightweight fallback
    """

    def __init__(self, mode: str = "keyword") -> None:
        """
        Args:
            mode: Detection mode — "nli" (DeBERTa), "llm" (LLM-as-judge),
                  or "keyword" (fast heuristic fallback).
        """
        self.mode = mode
        self._nli_model: Any = None

    async def check(
        self,
        generated_text: str,
        evidence: list[RetrievalResult],
    ) -> HallucinationReport:
        """Check a generated answer for hallucinations.

        Args:
            generated_text: The LLM-generated answer.
            evidence: Retrieved context that was provided to the LLM.

        Returns:
            HallucinationReport with per-claim verdicts and overall score.
        """
        # Step 1: Decompose into atomic claims
        claims = self._extract_claims(generated_text)

        if not claims:
            return HallucinationReport(
                claims=[],
                groundedness_score=1.0,
            )

        # Step 2: Check each claim
        evidence_texts = [e.content for e in evidence]

        for claim in claims:
            if self.mode == "nli":
                claim.verdict, claim.confidence = await self._check_nli(claim.text, evidence_texts)
            elif self.mode == "llm":
                claim.verdict, claim.confidence = await self._check_llm(claim.text, evidence_texts)
            else:
                claim.verdict, claim.confidence = self._check_keyword(claim.text, evidence_texts)

            # Record supporting evidence
            if claim.verdict != ClaimVerdict.NOT_SUPPORTED:
                claim.supporting_evidence = self._find_supporting_evidence(
                    claim.text, evidence_texts
                )

        # Compute overall score
        supported = sum(1 for c in claims if c.verdict == ClaimVerdict.SUPPORTED)
        partial = sum(1 for c in claims if c.verdict == ClaimVerdict.PARTIALLY_SUPPORTED)
        unsupported = sum(1 for c in claims if c.verdict == ClaimVerdict.NOT_SUPPORTED)
        total = len(claims)

        groundedness = (supported + 0.5 * partial) / total if total > 0 else 1.0

        warnings = []
        for claim in claims:
            if claim.verdict == ClaimVerdict.NOT_SUPPORTED:
                warnings.append(
                    f'⚠️ Unsupported claim: "{claim.text[:100]}..."'
                    if len(claim.text) > 100
                    else f'⚠️ Unsupported claim: "{claim.text}"'
                )

        return HallucinationReport(
            claims=claims,
            groundedness_score=round(groundedness, 3),
            supported_count=supported,
            partial_count=partial,
            unsupported_count=unsupported,
            warnings=warnings,
        )

    # ── Claim Extraction ─────────────────────────────────────────────────────

    def _extract_claims(self, text: str) -> list[Claim]:
        """Decompose text into atomic claims.

        An atomic claim is a single factual assertion that can be independently
        verified. We split on sentence boundaries and filter out:
          - Questions
          - Meta-statements ("Based on the documents...")
          - Very short fragments
        """
        # Remove citation markers for cleaner splitting
        clean = re.sub(r"\[\d+\]", "", text)

        # Extract citation associations before cleaning
        citation_map = self._map_citations_to_sentences(text)

        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", clean)

        claims = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()

            # Filter out non-claims
            if not sentence or len(sentence) < 20:
                continue
            if sentence.endswith("?"):
                continue
            if any(
                phrase in sentence.lower()
                for phrase in [
                    "based on the",
                    "according to the context",
                    "i don't have enough",
                    "the documents",
                    "the provided context",
                ]
            ):
                continue

            claims.append(
                Claim(
                    text=sentence,
                    citation_ids=citation_map.get(i, []),
                )
            )

        return claims

    @staticmethod
    def _map_citations_to_sentences(text: str) -> dict[int, list[int]]:
        """Map citation markers [N] to the sentence index they appear in."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result: dict[int, list[int]] = {}
        for i, sentence in enumerate(sentences):
            citations = re.findall(r"\[(\d+)\]", sentence)
            if citations:
                result[i] = [int(c) for c in citations]
        return result

    # ── Keyword-Based Checking (Fast Fallback) ───────────────────────────────

    def _check_keyword(self, claim: str, evidence_texts: list[str]) -> tuple[ClaimVerdict, float]:
        """Check a claim using keyword overlap with evidence.

        Computes the fraction of significant words in the claim that appear
        in any evidence passage.
        """
        # Extract significant words (3+ chars, not stopwords)
        stopwords = {
            "the",
            "and",
            "was",
            "were",
            "are",
            "has",
            "have",
            "had",
            "been",
            "this",
            "that",
            "with",
            "from",
            "for",
            "not",
            "but",
            "its",
            "can",
            "will",
            "than",
            "also",
            "which",
            "their",
            "would",
            "could",
            "into",
            "about",
            "other",
            "more",
            "some",
            "what",
            "when",
            "each",
        }

        claim_words = {
            w.lower()
            for w in re.findall(r"\b\w+\b", claim)
            if len(w) >= 3 and w.lower() not in stopwords
        }

        if not claim_words:
            return ClaimVerdict.SUPPORTED, 0.5

        # Check against all evidence
        evidence_combined = " ".join(evidence_texts).lower()
        evidence_words = set(re.findall(r"\b\w+\b", evidence_combined))

        matched = claim_words & evidence_words
        overlap = len(matched) / len(claim_words)

        # Also check for numbers — numbers in claims must appear in evidence
        claim_numbers = set(re.findall(r"\b\d+\.?\d*\b", claim))
        evidence_numbers = set(re.findall(r"\b\d+\.?\d*\b", evidence_combined))
        if claim_numbers:
            num_matched = claim_numbers & evidence_numbers
            num_ratio = len(num_matched) / len(claim_numbers)
            # Numbers are strong evidence signals
            overlap = 0.5 * overlap + 0.5 * num_ratio

        if overlap >= 0.6:
            return ClaimVerdict.SUPPORTED, overlap
        elif overlap >= 0.3:
            return ClaimVerdict.PARTIALLY_SUPPORTED, overlap
        else:
            return ClaimVerdict.NOT_SUPPORTED, overlap

    # ── NLI-Based Checking ───────────────────────────────────────────────────

    async def _check_nli(self, claim: str, evidence_texts: list[str]) -> tuple[ClaimVerdict, float]:
        """Check a claim using a Natural Language Inference model."""
        try:
            if self._nli_model is None:
                from sentence_transformers import CrossEncoder

                self._nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base", max_length=512)

            best_score = 0.0
            for evidence in evidence_texts:
                # NLI: [contradiction, neutral, entailment]
                scores = self._nli_model.predict([(evidence, claim)])
                entailment_score = float(scores[0][2]) if len(scores[0]) > 2 else float(scores[0])
                best_score = max(best_score, entailment_score)

            if best_score >= 0.7:
                return ClaimVerdict.SUPPORTED, best_score
            elif best_score >= 0.4:
                return ClaimVerdict.PARTIALLY_SUPPORTED, best_score
            else:
                return ClaimVerdict.NOT_SUPPORTED, best_score

        except Exception as e:
            logger.warning("NLI check failed, falling back to keyword: %s", e)
            return self._check_keyword(claim, evidence_texts)

    # ── LLM-as-Judge Checking ────────────────────────────────────────────────

    async def _check_llm(self, claim: str, evidence_texts: list[str]) -> tuple[ClaimVerdict, float]:
        """Check a claim using an LLM as a judge."""
        import litellm

        evidence_block = "\n\n".join(
            f"[Evidence {i + 1}]: {e}" for i, e in enumerate(evidence_texts[:5])
        )

        try:
            response = await litellm.acompletion(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a fact-checker. Given a claim and evidence passages, "
                            "determine if the claim is SUPPORTED, PARTIALLY_SUPPORTED, or "
                            "NOT_SUPPORTED by the evidence. Respond with exactly one word."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Evidence:\n{evidence_block}\n\nClaim: {claim}",
                    },
                ],
                temperature=0.0,
                max_tokens=20,
            )

            verdict_text = response.choices[0].message.content.strip().upper()

            if (
                "SUPPORTED" in verdict_text
                and "NOT" not in verdict_text
                and "PARTIAL" not in verdict_text
            ):
                return ClaimVerdict.SUPPORTED, 0.9
            elif "PARTIAL" in verdict_text:
                return ClaimVerdict.PARTIALLY_SUPPORTED, 0.5
            else:
                return ClaimVerdict.NOT_SUPPORTED, 0.1

        except Exception as e:
            logger.warning("LLM judge failed, falling back to keyword: %s", e)
            return self._check_keyword(claim, evidence_texts)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _find_supporting_evidence(
        claim: str, evidence_texts: list[str], max_evidence: int = 3
    ) -> list[str]:
        """Find the evidence passages that best support a claim."""
        claim_words = set(re.findall(r"\b\w{3,}\b", claim.lower()))

        scored = []
        for evidence in evidence_texts:
            ev_words = set(re.findall(r"\b\w{3,}\b", evidence.lower()))
            overlap = len(claim_words & ev_words) / max(len(claim_words), 1)
            scored.append((evidence[:200], overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, s in scored[:max_evidence] if s > 0.1]
