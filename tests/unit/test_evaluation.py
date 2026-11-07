"""Unit tests for the evaluation metrics and benchmark runner."""

import pytest

from atlas_rag.core.generation.citation import Citation
from atlas_rag.core.hallucination.detector import Claim, ClaimVerdict
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
from atlas_rag.evaluation.runner import EvaluationRunner
from atlas_rag.schemas.table import CellReference


def test_retrieval_metrics():
    retrieved = ["doc_1", "doc_2", "doc_3", "doc_4"]
    relevant = {"doc_1", "doc_3"}

    assert precision_at_k(retrieved, relevant, k=2) == 0.5  # doc_1 matches
    assert precision_at_k(retrieved, relevant, k=4) == 0.5  # doc_1, doc_3 match
    assert recall_at_k(retrieved, relevant, k=1) == 0.5  # 1 / 2
    assert recall_at_k(retrieved, relevant, k=3) == 1.0  # 2 / 2
    assert reciprocal_rank(retrieved, relevant) == 1.0  # first rank is relevant

    mrr = mean_reciprocal_rank([["doc_a", "doc_b"], ["doc_x", "doc_y"]], [{"doc_b"}, {"doc_x"}])
    assert mrr == (0.5 + 1.0) / 2


def test_ndcg_at_k():
    retrieved = ["doc_1", "doc_2", "doc_3"]
    scores = {"doc_1": 3.0, "doc_2": 2.0, "doc_3": 1.0}
    # Ideal order is retrieved order -> NDCG = 1.0
    assert ndcg_at_k(retrieved, scores, k=3) == 1.0


def test_groundedness_metrics():
    claims = [
        Claim(text="Claim A", verdict=ClaimVerdict.SUPPORTED),
        Claim(text="Claim B", verdict=ClaimVerdict.PARTIALLY_SUPPORTED),
        Claim(text="Claim C", verdict=ClaimVerdict.NOT_SUPPORTED),
    ]
    # 1.0 + 0.5 + 0.0 = 1.5 / 3 = 0.5
    assert compute_groundedness_ratio(claims) == 0.5
    assert compute_hallucination_rate(claims) == 1 / 3


def test_citation_metrics():
    citations = [
        Citation(citation_id=1, text_span="", source_type="chunk"),
        Citation(citation_id=2, text_span="", source_type="chunk"),
    ]
    # Verified citations: {1}
    assert citation_precision(citations, {1}) == 0.5
    assert citation_recall(total_verifiable_claims=4, claims_with_valid_citations=2) == 0.5


def test_table_evaluation_metrics():
    assert numerical_match("150000", 150000) is True
    assert numerical_match("$150,000", 150000) is True
    assert numerical_match(33.333, 33.3331) is True

    predicted_cells = [
        CellReference(table_id="tbl_1", row_index=0, column="Revenue", value=100),
        CellReference(table_id="tbl_1", row_index=1, column="Revenue", value=200),
    ]
    expected_cells = [("tbl_1", 0, "Revenue"), ("tbl_1", 1, "Revenue")]
    p, r, f1 = cell_citation_accuracy(predicted_cells, expected_cells)
    assert p == 1.0
    assert r == 1.0
    assert f1 == 1.0


@pytest.mark.asyncio
async def test_full_evaluation_runner():
    runner = EvaluationRunner()
    report = await runner.run()
    # The heuristic lookup returns extra rows; the offline benchmark must expose it.
    assert report.table_numerical_accuracy == pytest.approx(2 / 3)
    assert report.retrieval_mrr == 1.0
    assert report.citation_precision_score == 1.0
    assert report.tests_run == len(report.details)
    assert report.passed_tests < report.tests_run
    assert report.overall_score == pytest.approx(
        sum(item["score"] for item in report.details) / report.tests_run
    )
