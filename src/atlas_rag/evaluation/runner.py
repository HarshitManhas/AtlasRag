"""Automated Evaluation Runner for AtlasRAG.

Executes comprehensive evaluation of:
  - Hybrid Retrieval Quality (Precision@K, Recall@K, MRR, NDCG)
  - Advanced Table Reasoning (SQL accuracy, numerical correctness, cell-level citation accuracy)
  - Citation Precision and Recall
  - Hallucination Detection & Groundedness Scoring
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atlas_rag.core.generation.citation import CitationExtractor
from atlas_rag.core.hallucination.detector import HallucinationDetector
from atlas_rag.core.retrieval.dense import RetrievalResult
from atlas_rag.core.retrieval.fusion import ReciprocalRankFusion
from atlas_rag.core.table_analysis.citation import TableCitationEngine
from atlas_rag.core.table_analysis.detector import TableQueryDetector
from atlas_rag.core.table_analysis.executor import SQLExecutor
from atlas_rag.core.table_analysis.sql_generator import SQLGenerator
from atlas_rag.evaluation.metrics.citation import citation_precision, citation_recall
from atlas_rag.evaluation.metrics.retrieval import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from atlas_rag.evaluation.metrics.table import cell_citation_accuracy, numerical_match
from atlas_rag.schemas.table import (
    ColumnTypeDef,
    StructuredTable,
    TableSource,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Offline component scorecard; does not measure live RAG quality."""

    # Table reasoning
    table_query_accuracy: float = 0.0
    table_sql_execution_success: float = 0.0
    table_numerical_accuracy: float = 0.0
    table_cell_citation_f1: float = 0.0

    # Retrieval
    retrieval_mrr: float = 0.0
    retrieval_ndcg_at_5: float = 0.0
    retrieval_precision_at_3: float = 0.0
    retrieval_recall_at_5: float = 0.0

    # Citations
    citation_precision_score: float = 0.0
    citation_recall_score: float = 0.0

    # Groundedness & Hallucination
    average_groundedness: float = 0.0
    hallucination_detection_accuracy: float = 0.0

    # Summary
    overall_score: float = 0.0
    tests_run: int = 0
    passed_tests: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)


class EvaluationRunner:
    """Executes the benchmark evaluation suite and produces a performance scorecard."""

    def __init__(self, dataset_path: str | Path | None = None) -> None:
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "datasets" / "golden_set.json"
        self.dataset_path = Path(dataset_path)

        # Core components
        self.table_detector = TableQueryDetector()
        self.sql_generator = SQLGenerator()
        self.sql_executor = SQLExecutor()
        self.table_citation = TableCitationEngine()
        self.fusion = ReciprocalRankFusion(k=60)
        self.citation_extractor = CitationExtractor()
        self.hallucination_detector = HallucinationDetector(mode="keyword")

    def _get_benchmark_tables(self) -> list[StructuredTable]:
        """Load benchmark evaluation tables."""
        return [
            StructuredTable(
                table_id="tbl_001",
                title="Product Revenue Summary 2024",
                headers=["Product", "Revenue", "Units Sold", "Region"],
                column_types=[
                    ColumnTypeDef(name="Product", inferred_type="text"),
                    ColumnTypeDef(name="Revenue", inferred_type="numeric"),
                    ColumnTypeDef(name="Units Sold", inferred_type="numeric"),
                    ColumnTypeDef(name="Region", inferred_type="text"),
                ],
                rows=[
                    {
                        "Product": "Widget A",
                        "Revenue": 150000,
                        "Units Sold": 1200,
                        "Region": "North",
                    },
                    {
                        "Product": "Widget B",
                        "Revenue": 250000,
                        "Units Sold": 2100,
                        "Region": "South",
                    },
                    {"Product": "Gadget C", "Revenue": 180000, "Units Sold": 900, "Region": "East"},
                    {
                        "Product": "Gadget D",
                        "Revenue": 320000,
                        "Units Sold": 3400,
                        "Region": "West",
                    },
                    {"Product": "Widget E", "Revenue": 95000, "Units Sold": 650, "Region": "North"},
                ],
                row_count=5,
                col_count=4,
                source=TableSource(
                    page=12, file="annual_report_2024.pdf", document_id="doc_benchmark_001"
                ),
            )
        ]

    async def run(self) -> EvaluationReport:
        """Run the full evaluation benchmark."""
        report = EvaluationReport()
        benchmark_tables = self._get_benchmark_tables()

        cases = json.loads(self.dataset_path.read_text(encoding="utf-8"))
        table_cases = [case for case in cases if case["type"].startswith("table_")]
        if not table_cases:
            raise ValueError("Evaluation dataset must contain table cases")
        classifications, executions, answers, cell_scores = [], [], [], []
        for case in table_cases:
            query_type, _ = self.table_detector.classify_query(case["query"], benchmark_tables)
            classifications.append(query_type.value == case["type"].removeprefix("table_"))
            sql = self.sql_generator._fallback_generate(case["query"], benchmark_tables, query_type)
            result = self.sql_executor.execute(sql, benchmark_tables)
            executions.append(result.error is None)
            correct = False
            if not result.error and result.rows:
                row = result.rows[0]
                correct = any(
                    numerical_match(value, case["expected_answer_number"]) for value in row.values()
                )
                if case["type"] == "table_lookup":
                    correct = correct and len(result.rows) == 1
                if "expected_product" in case:
                    correct = correct and row.get("product") == case["expected_product"]
            answers.append(correct)
            expected_cells = [tuple(cell) for cell in case["expected_cells"]]
            _, _, cell_f1 = cell_citation_accuracy(result.cell_citations, expected_cells)
            cell_scores.append(cell_f1)

        report.table_query_accuracy = sum(classifications) / len(table_cases)
        report.table_sql_execution_success = sum(executions) / len(table_cases)
        report.table_numerical_accuracy = sum(answers) / len(table_cases)
        report.table_cell_citation_f1 = sum(cell_scores) / len(table_cases)

        # ── 2. Evaluate Hybrid Retrieval (RRF Fusion) ───────────────────
        dense_hits = [
            RetrievalResult(
                content="AtlasRAG architecture combines dense and BM25 search.",
                score=0.88,
                document_id="doc_1",
                chunk_index=0,
            ),
            RetrievalResult(
                content="Elasticsearch BM25 handles exact keyword matches.",
                score=0.82,
                document_id="doc_1",
                chunk_index=1,
            ),
            RetrievalResult(
                content="Qdrant manages vector embeddings.",
                score=0.79,
                document_id="doc_1",
                chunk_index=2,
            ),
        ]
        sparse_hits = [
            RetrievalResult(
                content="Elasticsearch BM25 handles exact keyword matches.",
                score=14.2,
                document_id="doc_1",
                chunk_index=1,
            ),
            RetrievalResult(
                content="AtlasRAG architecture combines dense and BM25 search.",
                score=11.5,
                document_id="doc_1",
                chunk_index=0,
            ),
            RetrievalResult(
                content="Irrelevant background log text.",
                score=5.1,
                document_id="doc_2",
                chunk_index=9,
            ),
        ]

        fused = self.fusion.fuse(dense_hits, sparse_hits, top_k=5)
        retrieved_ids = [f"{r.document_id}_{r.chunk_index}" for r in fused]
        relevant_ids = {"doc_1_0", "doc_1_1"}

        p_at_3 = precision_at_k(retrieved_ids, relevant_ids, k=3)
        r_at_5 = recall_at_k(retrieved_ids, relevant_ids, k=5)
        mrr = mean_reciprocal_rank([retrieved_ids], [relevant_ids])
        ndcg = ndcg_at_k(retrieved_ids, {"doc_1_0": 3.0, "doc_1_1": 2.0}, k=5)

        report.retrieval_precision_at_3 = p_at_3
        report.retrieval_recall_at_5 = r_at_5
        report.retrieval_mrr = mrr
        report.retrieval_ndcg_at_5 = ndcg

        # ── 3. Evaluate Citation Engine ─────────────────────────────────
        gen_text = (
            "AtlasRAG fuses dense and sparse retrieval [1], while Elasticsearch handles BM25 [2]."
        )
        _, citations = self.citation_extractor.extract_citations(gen_text, fused)
        report.citation_precision_score = citation_precision(citations, {1, 2})
        report.citation_recall_score = citation_recall(
            total_verifiable_claims=2,
            claims_with_valid_citations=len({c.citation_id for c in citations} & {1, 2}),
        )

        # ── 4. Evaluate Hallucination Detection & Groundedness ───────────
        grounded_answer = "AtlasRAG architecture combines dense and BM25 search."
        grounded_report = await self.hallucination_detector.check(grounded_answer, fused)

        hallucinated_answer = "The platform was invented in the year 1820 on Mars."
        hallucinated_report = await self.hallucination_detector.check(hallucinated_answer, fused)

        # Check if grounded answer got high score, and hallucinated answer got flagged
        is_grounded_correct = grounded_report.groundedness_score >= 0.8
        is_hallucination_caught = (hallucinated_report.groundedness_score < 0.5) or (
            len(hallucinated_report.warnings) > 0
        )

        report.average_groundedness = grounded_report.groundedness_score
        report.hallucination_detection_accuracy = (
            (1.0 if is_grounded_correct else 0.0) + (1.0 if is_hallucination_caught else 0.0)
        ) / 2.0

        metric_names = [
            "table_query_accuracy",
            "table_sql_execution_success",
            "table_numerical_accuracy",
            "table_cell_citation_f1",
            "retrieval_mrr",
            "retrieval_ndcg_at_5",
            "retrieval_precision_at_3",
            "retrieval_recall_at_5",
            "citation_precision_score",
            "citation_recall_score",
            "average_groundedness",
            "hallucination_detection_accuracy",
        ]
        report.details = [
            {"metric": name, "score": getattr(report, name), "passed": getattr(report, name) == 1.0}
            for name in metric_names
        ]
        report.overall_score = sum(item["score"] for item in report.details) / len(report.details)
        report.tests_run = len(report.details)
        report.passed_tests = sum(item["passed"] for item in report.details)
        return report
