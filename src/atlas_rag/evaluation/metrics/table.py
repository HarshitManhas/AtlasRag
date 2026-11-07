"""Table evaluation metrics: SQL accuracy, numerical correctness, cell citation accuracy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from atlas_rag.schemas.table import CellReference


def _clean_val(val: Any) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(",", "")
    for char in ["$", "€", "£", "¥", "₹", "%"]:
        s = s.replace(char, "")
    return float(s.strip())


def numerical_match(
    predicted: Any, ground_truth: Any, rel_tol: float = 1e-3, abs_tol: float = 1e-3
) -> bool:
    """Compare two numerical values with tolerance."""
    try:
        p_val = _clean_val(predicted)
        g_val = _clean_val(ground_truth)
        return math.isclose(p_val, g_val, rel_tol=rel_tol, abs_tol=abs_tol)
    except (ValueError, TypeError):
        return str(predicted).strip().lower() == str(ground_truth).strip().lower()


def cell_citation_accuracy(
    predicted_cells: Sequence[CellReference],
    expected_cells: Sequence[tuple[str, int, str]],  # (table_id, row_idx, column)
) -> tuple[float, float, float]:
    """Calculate Precision, Recall, and F1 for table cell citations."""
    expected_set = set(expected_cells)
    if not expected_set and not predicted_cells:
        return 1.0, 1.0, 1.0

    pred_set = {(c.table_id, c.row_index, c.column) for c in predicted_cells}

    true_positives = len(pred_set & expected_set)
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(expected_set) if expected_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1
