"""Table Citation Engine — provides cell-level citations for table-derived answers.

Maps generated answer fragments back to specific cells in extracted tables,
producing fine-grained citations like:

    "Revenue for Product A was $150 [tbl_001, row 0, col 'Revenue', page 12]"

This is the key differentiator that makes AtlasRAG's table understanding
trustworthy and verifiable.
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import Any

from atlas_rag.schemas.table import (
    CellReference,
    SQLResult,
    StructuredTable,
)

logger = logging.getLogger(__name__)


class TableCitationEngine:
    """Generates and manages cell-level citations for table-derived information.

    Works in three modes:
      1. SQL result citations — trace SQL output cells back to source tables
      2. Value-based citations — find specific values mentioned in generated text
      3. Range citations — cite a range of cells (e.g., for aggregations)
    """

    def __init__(self) -> None:
        pass

    def cite_sql_result(
        self,
        sql_result: SQLResult,
        tables: list[StructuredTable],
    ) -> list[CellReference]:
        """Generate citations for all values in a SQL result.

        This is the primary citation path — when the Table Analysis Service
        executes SQL and returns results, this method maps each result cell
        back to its source in the original table.

        Args:
            sql_result: The executed SQL result.
            tables: The source tables the SQL was run against.

        Returns:
            List of CellReference objects for each traceable value.
        """
        citations: list[CellReference] = []

        # If the SQL result already has citations from the executor, use those
        if sql_result.cell_citations:
            return sql_result.cell_citations

        # Otherwise, build citations by tracing values
        table_map = {t.table_id: t for t in tables}

        for result_row in sql_result.rows:
            for col, value in result_row.items():
                if col.startswith("_") or value is None:
                    continue

                ref = self._find_cell_in_tables(value, col, tables, table_map)
                if ref:
                    citations.append(ref)

        return citations

    def cite_aggregation(
        self,
        aggregation_value: Any,
        column: str,
        operation: str,
        table: StructuredTable,
    ) -> list[CellReference]:
        """Generate citations for an aggregation result.

        For operations like SUM, AVG, COUNT — cites ALL cells that contributed
        to the aggregation result. This provides full traceability.

        Args:
            aggregation_value: The computed aggregate value.
            column: The column the aggregation was performed on.
            operation: The aggregation operation (sum, avg, min, max, count).
            table: The source table.

        Returns:
            List of CellReferences for all cells that were aggregated.
        """
        citations: list[CellReference] = []

        if column not in table.headers:
            # Try fuzzy match
            from atlas_rag.core.table_analysis.sql_generator import _sanitize_sql_name

            for header in table.headers:
                if _sanitize_sql_name(header) == column:
                    column = header
                    break
            else:
                return citations

        for row_idx, row in enumerate(table.rows):
            cell_value = row.get(column)
            if cell_value is not None and str(cell_value).strip() != "":
                citations.append(
                    CellReference(
                        table_id=table.table_id,
                        row_index=row_idx,
                        column=column,
                        value=cell_value,
                        page=table.source.page,
                        file=table.source.file,
                    )
                )

        return citations

    def cite_from_text(
        self,
        generated_text: str,
        tables: list[StructuredTable],
    ) -> list[CellReference]:
        """Extract cell citations from generated text by finding table values
        mentioned in the text.

        Scans the generated answer for numbers and named entities that match
        cell values in the available tables.

        Args:
            generated_text: The LLM-generated answer text.
            tables: Available tables to match against.

        Returns:
            List of CellReferences for values found in the text.
        """
        citations: list[CellReference] = []
        seen: set[tuple[str, int, str]] = set()  # Dedup by (table_id, row, col)

        # Extract numbers from the text
        numbers_in_text = self._extract_numbers(generated_text)
        text_lower = generated_text.lower()

        for table in tables:
            for row_idx, row in enumerate(table.rows):
                for header in table.headers:
                    value = row.get(header)
                    if value is None:
                        continue

                    key = (table.table_id, row_idx, header)
                    if key in seen:
                        continue

                    # Check if this cell's value appears in the generated text
                    matched = False

                    # Numeric match
                    if isinstance(value, (int, float)):
                        if value in numbers_in_text:
                            matched = True
                    elif isinstance(value, str) and len(value) > 2 and value.lower() in text_lower:
                        matched = True

                    if matched:
                        seen.add(key)
                        citations.append(
                            CellReference(
                                table_id=table.table_id,
                                row_index=row_idx,
                                column=header,
                                value=value,
                                page=table.source.page,
                                file=table.source.file,
                            )
                        )

        return citations

    def format_citation(self, ref: CellReference) -> str:
        """Format a cell reference as a human-readable inline citation.

        Example: [Table tbl_001, Row 1, "Revenue" = 150, Page 12]
        """
        parts = [f"Table {ref.table_id}"]
        parts.append(f"Row {ref.row_index + 1}")  # 1-indexed for humans
        parts.append(f'"{ref.column}" = {ref.value}')
        if ref.page:
            parts.append(f"Page {ref.page}")
        if ref.file:
            parts.append(f"File: {ref.file}")
        return f"[{', '.join(parts)}]"

    def format_citations_block(self, citations: list[CellReference]) -> str:
        """Format a list of citations as a reference block for the answer.

        Groups citations by table and presents them in a structured format.
        """
        if not citations:
            return ""

        # Group by table
        by_table: dict[str, list[CellReference]] = {}
        for ref in citations:
            by_table.setdefault(ref.table_id, []).append(ref)

        blocks: list[str] = []
        blocks.append("\n📊 Table Citations:")

        for table_id, refs in by_table.items():
            source = refs[0]
            blocks.append(
                f"\n  Source: {source.file or 'unknown'}"
                f"{f', Page {source.page}' if source.page else ''}"
                f" | Table: {table_id}"
            )

            # Group by row for compactness
            by_row: dict[int, list[CellReference]] = {}
            for ref in refs:
                by_row.setdefault(ref.row_index, []).append(ref)

            for row_idx in sorted(by_row.keys()):
                row_refs = by_row[row_idx]
                cells = [f"{r.column}={r.value}" for r in row_refs]
                blocks.append(f"    Row {row_idx + 1}: {', '.join(cells)}")

        return "\n".join(blocks)

    def _find_cell_in_tables(
        self,
        value: Any,
        column_hint: str,
        tables: list[StructuredTable],
        table_map: dict[str, StructuredTable],
    ) -> CellReference | None:
        """Find a specific value in the source tables."""
        from atlas_rag.core.table_analysis.sql_generator import _sanitize_sql_name

        for table in tables:
            for header in table.headers:
                sanitized = _sanitize_sql_name(header)
                if sanitized != column_hint and header.lower() != column_hint.lower():
                    continue

                for row_idx, row in enumerate(table.rows):
                    source_val = row.get(header)
                    if self._values_match(source_val, value):
                        return CellReference(
                            table_id=table.table_id,
                            row_index=row_idx,
                            column=header,
                            value=source_val,
                            page=table.source.page,
                            file=table.source.file,
                        )
        return None

    @staticmethod
    def _extract_numbers(text: str) -> set[float]:
        """Extract all numbers from text, handling commas and currency symbols."""
        numbers: set[float] = set()
        # Match integers and decimals, with optional commas and currency symbols
        pattern = r"[\$€£¥₹]?\s*([\d,]+\.?\d*)"
        for match in re.finditer(pattern, text):
            num_str = match.group(1).replace(",", "")
            with contextlib.suppress(ValueError):
                numbers.add(float(num_str))
        return numbers

    @staticmethod
    def _values_match(source: Any, result: Any) -> bool:
        """Check if two values match, with numeric tolerance."""
        if source is None and result is None:
            return True
        if source is None or result is None:
            return False
        try:
            return abs(float(source) - float(result)) < 0.001
        except (ValueError, TypeError):
            return str(source).strip().lower() == str(result).strip().lower()
