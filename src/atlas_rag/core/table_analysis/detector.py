"""Table Query Detector — classifies whether a user query requires table analysis.

Uses a combination of keyword heuristics and LLM-based classification to determine
if a query is asking about tabular data (lookups, aggregations, comparisons, rankings).
"""

from __future__ import annotations

import logging
import re

from atlas_rag.schemas.table import StructuredTable, TableQueryType

logger = logging.getLogger(__name__)


# ─── Keyword-Based Heuristics ────────────────────────────────────────────────

# Patterns strongly suggesting a table/analytical query
_AGGREGATION_PATTERNS = [
    r"\btotal\b",
    r"\bsum\b",
    r"\baverage\b",
    r"\bmean\b",
    r"\bmedian\b",
    r"\bcount\b",
    r"\bhow many\b",
    r"\bminimum\b",
    r"\bmaximum\b",
    r"\bmin\b",
    r"\bmax\b",
    r"\baggregate\b",
]

_COMPARISON_PATTERNS = [
    r"\bcompare\b",
    r"\bcomparison\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bdifference\b",
    r"\bmore than\b",
    r"\bless than\b",
    r"\bgreater\b",
    r"\bhigher\b",
    r"\blower\b",
    r"\bexceed\b",
]

_RANKING_PATTERNS = [
    r"\btop\s+\d+",
    r"\bbottom\s+\d+",
    r"\bhighest\b",
    r"\blowest\b",
    r"\blargest\b",
    r"\bsmallest\b",
    r"\brank\b",
    r"\branking\b",
    r"\bbest\b",
    r"\bworst\b",
    r"\bmost\b",
    r"\bleast\b",
]

_TREND_PATTERNS = [
    r"\btrend\b",
    r"\bgrowth\b",
    r"\bdecline\b",
    r"\bincrease\b",
    r"\bdecrease\b",
    r"\byear[- ]over[- ]year\b",
    r"\byoy\b",
    r"\bquarter[- ]over[- ]quarter\b",
    r"\bqoq\b",
    r"\bchange\b",
]

_LOOKUP_PATTERNS = [
    r"\bwhat is the\b.*\bfor\b",
    r"\bvalue of\b",
    r"\bhow much\b",
    r"\bwhat was\b",
    r"\bfind the\b",
    r"\blook up\b",
    r"\bshow me\b",
    r"\blist\b.*\bwhere\b",
]

_FILTER_PATTERNS = [
    r"\bwhere\b",
    r"\bfilter\b",
    r"\bonly\b.*\bwith\b",
    r"\bgreater than\b",
    r"\bless than\b",
    r"\bbetween\b",
    r"\bexcluding\b",
    r"\bexcept\b",
]


def _match_patterns(text: str, patterns: list[str]) -> int:
    """Count how many patterns match in the text."""
    text_lower = text.lower()
    return sum(1 for p in patterns if re.search(p, text_lower))


class TableQueryDetector:
    """Determines whether a user query should be routed to the table analysis
    pipeline rather than (or in addition to) the standard text retrieval pipeline.

    Uses a two-stage approach:
      1. Fast keyword heuristics (always runs)
      2. Optional LLM-based classification (for borderline cases)
    """

    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.confidence_threshold = confidence_threshold

    def classify_query(
        self,
        query: str,
        available_tables: list[StructuredTable] | None = None,
    ) -> tuple[TableQueryType, float]:
        """Classify a query and return its type with confidence score.

        Args:
            query: The user's natural language query.
            available_tables: Tables from relevant documents (used for
                column-name matching).

        Returns:
            Tuple of (query_type, confidence) where confidence is 0.0-1.0.
        """
        # Stage 1: Keyword heuristics
        query_type, confidence = self._keyword_classify(query)

        # Boost confidence if query mentions column names from available tables
        if available_tables and query_type != TableQueryType.NONE:
            col_boost = self._column_name_boost(query, available_tables)
            confidence = min(1.0, confidence + col_boost)

        # If still borderline, check for numeric references
        if confidence < self.confidence_threshold and self._has_numeric_context(query):
            confidence = min(1.0, confidence + 0.15)

        logger.debug(
            "Query classification: type=%s confidence=%.2f query=%r",
            query_type.value,
            confidence,
            query[:100],
        )

        return query_type, confidence

    def _keyword_classify(self, query: str) -> tuple[TableQueryType, float]:
        """Fast keyword-based classification."""
        scores: dict[TableQueryType, float] = {
            TableQueryType.AGGREGATION: 0.0,
            TableQueryType.COMPARISON: 0.0,
            TableQueryType.RANKING: 0.0,
            TableQueryType.TREND: 0.0,
            TableQueryType.LOOKUP: 0.0,
            TableQueryType.FILTER: 0.0,
        }

        pattern_map: dict[TableQueryType, list[str]] = {
            TableQueryType.AGGREGATION: _AGGREGATION_PATTERNS,
            TableQueryType.COMPARISON: _COMPARISON_PATTERNS,
            TableQueryType.RANKING: _RANKING_PATTERNS,
            TableQueryType.TREND: _TREND_PATTERNS,
            TableQueryType.LOOKUP: _LOOKUP_PATTERNS,
            TableQueryType.FILTER: _FILTER_PATTERNS,
        }

        for qtype, patterns in pattern_map.items():
            match_count = _match_patterns(query, patterns)
            if match_count > 0:
                # More matches = higher confidence, capped at 0.85
                scores[qtype] = min(0.85, 0.3 + match_count * 0.2)

        # Find the highest-scoring type
        best_type = max(scores, key=lambda k: scores[k])
        best_score = scores[best_type]

        if best_score < 0.2:
            return TableQueryType.NONE, 0.0

        return best_type, best_score

    def _column_name_boost(self, query: str, tables: list[StructuredTable]) -> float:
        """Boost confidence if the query mentions column names from available tables."""
        query_lower = query.lower()
        all_columns = set()
        for table in tables:
            for header in table.headers:
                all_columns.add(header.lower())

        matched = sum(1 for col in all_columns if col in query_lower)
        if matched == 0:
            return 0.0
        # Each matched column adds a boost, up to 0.3
        return min(0.3, matched * 0.1)

    @staticmethod
    def _has_numeric_context(query: str) -> bool:
        """Check if the query contains numbers or numeric references."""
        return bool(re.search(r"\d+\.?\d*", query))

    def match_tables(
        self,
        query: str,
        tables: list[StructuredTable],
    ) -> list[StructuredTable]:
        """Find which tables are most relevant to the query.

        Matches based on:
          1. Column name overlap with query terms
          2. Table title relevance
          3. Data value matching
        """
        if not tables:
            return []

        scored: list[tuple[StructuredTable, float]] = []
        query_terms = set(query.lower().split())

        for table in tables:
            score = 0.0

            # Column name matching
            col_names = {h.lower() for h in table.headers}
            col_overlap = len(query_terms & col_names)
            score += col_overlap * 0.3

            # Check for partial column name matches
            for term in query_terms:
                for col in col_names:
                    if len(term) > 3 and (term in col or col in term):
                        score += 0.15

            # Title matching
            if table.title:
                title_terms = set(table.title.lower().split())
                title_overlap = len(query_terms & title_terms)
                score += title_overlap * 0.25

            # Check if query mentions specific values in the table
            for row in table.rows[:10]:  # Check first 10 rows
                for val in row.values():
                    val_str = str(val).lower()
                    if val_str in query.lower() and len(val_str) > 2:
                        score += 0.2
                        break

            if score > 0:
                scored.append((table, score))

        # Sort by score descending, return tables above threshold
        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, s in scored if s > 0.1]
