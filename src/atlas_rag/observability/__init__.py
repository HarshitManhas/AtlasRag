"""Observability — structured logging, OpenTelemetry tracing, Prometheus metrics."""

from __future__ import annotations

import logging

import structlog
from prometheus_client import Counter, Histogram, Info

# ─── Structured Logging ─────────────────────────────────────────────────────


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured logging with structlog."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure standard logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


# ─── Prometheus Metrics ──────────────────────────────────────────────────────

APP_INFO = Info("atlas_rag", "AtlasRAG application info")

QUERIES_TOTAL = Counter(
    "atlas_rag_queries_total",
    "Total number of RAG queries",
    ["status"],
)

QUERY_LATENCY = Histogram(
    "atlas_rag_query_latency_seconds",
    "End-to-end query latency",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

RETRIEVAL_LATENCY = Histogram(
    "atlas_rag_retrieval_latency_seconds",
    "Retrieval pipeline latency",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

GENERATION_LATENCY = Histogram(
    "atlas_rag_generation_latency_seconds",
    "LLM generation latency",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

TOKENS_USED = Counter(
    "atlas_rag_tokens_total",
    "Total tokens consumed",
    ["direction"],  # "input" or "output"
)

DOCUMENTS_INGESTED = Counter(
    "atlas_rag_documents_ingested_total",
    "Total documents ingested",
    ["status"],
)

HALLUCINATION_SCORE = Histogram(
    "atlas_rag_groundedness_score",
    "Distribution of groundedness scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
