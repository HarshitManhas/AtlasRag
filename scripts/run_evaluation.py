#!/usr/bin/env python3
"""AtlasRAG Evaluation Benchmark CLI.

Runs the automated evaluation framework across:
  - Hybrid Retrieval (MRR, NDCG@K, Precision@K, Recall@K)
  - Table Intelligence (SQL Generation, Numerical Correctness, Cell-Level Citations)
  - Citation Accuracy & Source Attribution
  - Hallucination Detection & Groundedness Scoring
"""

import asyncio
import os
import sys
import time

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath("src"))

from atlas_rag.evaluation.runner import EvaluationRunner


async def main() -> None:
    print("=" * 70)
    print(" AtlasRAG — Offline Component Evaluation")
    print("=" * 70)
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Synthetic fixtures only: no live retrieval, reranking, PDF extraction, or LLM calls.\n")

    runner = EvaluationRunner()
    start_time = time.perf_counter()
    report = await runner.run()
    elapsed = time.perf_counter() - start_time

    print("-" * 70)
    print(" 1. ADVANCED TABLE UNDERSTANDING & REASONING")
    print("-" * 70)
    print(f"  • Query Type Classification:   {report.table_query_accuracy * 100:.1f}%")
    print(f"  • Safe SQL Execution Rate:     {report.table_sql_execution_success * 100:.1f}%")
    print(f"  • Numerical Answer Accuracy:   {report.table_numerical_accuracy * 100:.1f}%")
    print(f"  • Cell-Level Citation F1:      {report.table_cell_citation_f1 * 100:.1f}%")

    print("\n" + "-" * 70)
    print(" 2. RRF FUSION OF SYNTHETIC SEARCH RESULTS")
    print("-" * 70)
    print(f"  • Mean Reciprocal Rank (MRR):  {report.retrieval_mrr:.4f}")
    print(f"  • NDCG @ 5:                    {report.retrieval_ndcg_at_5:.4f}")
    print(f"  • Precision @ 3:               {report.retrieval_precision_at_3 * 100:.1f}%")
    print(f"  • Recall @ 5:                  {report.retrieval_recall_at_5 * 100:.1f}%")

    print("\n" + "-" * 70)
    print(" 3. CITATION ACCURACY & EVIDENCE TRACEABILITY")
    print("-" * 70)
    print(f"  • Citation Precision:          {report.citation_precision_score * 100:.1f}%")
    print(f"  • Citation Recall:             {report.citation_recall_score * 100:.1f}%")

    print("\n" + "-" * 70)
    print(" 4. HALLUCINATION DETECTION & GROUNDEDNESS")
    print("-" * 70)
    print(f"  • Average Groundedness Score:  {report.average_groundedness * 100:.1f}%")
    print(f"  • Detection Accuracy:          {report.hallucination_detection_accuracy * 100:.1f}%")

    print("\n" + "=" * 70)
    print(f" OFFLINE COMPONENT SCORE: {report.overall_score * 100:.1f} / 100.0")
    print(f" Perfect metric checks: {report.passed_tests}/{report.tests_run}")
    print(f" Benchmark Execution Time:         {elapsed:.2f} seconds")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
