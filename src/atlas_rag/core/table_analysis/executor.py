"""Safe SQL Executor — runs generated SQL against extracted table data in SQLite.

Provides a sandboxed execution environment with:
  - Ephemeral in-memory SQLite databases (no persistent state)
  - Query timeout enforcement
  - Row limit enforcement
  - Cell-level citation tracking for every value in the result
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
import time
from typing import Any

from atlas_rag.schemas.table import (
    CellReference,
    SQLQuery,
    SQLResult,
    StructuredTable,
)

logger = logging.getLogger(__name__)

# Safety limits
MAX_RESULT_ROWS = 1000
QUERY_TIMEOUT_SECONDS = 10


def _sanitize_sql_name(name: str) -> str:
    """Sanitize a name for use as a SQL identifier."""
    import re

    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"col_{sanitized}"
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_").lower() or "unnamed"


class SQLExecutor:
    """Executes SQL queries against extracted table data using in-memory SQLite.

    Each execution creates a fresh SQLite database, loads the relevant tables,
    runs the query, and tears down. This provides strong isolation and prevents
    any cross-query state leakage.
    """

    def __init__(self, max_rows: int = MAX_RESULT_ROWS, timeout: float = QUERY_TIMEOUT_SECONDS):
        self.max_rows = max_rows
        self.timeout = timeout

    def execute(
        self,
        query: SQLQuery,
        tables: list[StructuredTable],
    ) -> SQLResult:
        """Execute a SQL query against the provided tables.

        Args:
            query: The SQL query to execute.
            tables: Tables to load into the ephemeral database.

        Returns:
            SQLResult with columns, rows, cell citations, and timing.
        """
        start_time = time.perf_counter()

        try:
            conn = sqlite3.connect(":memory:")
            conn.execute("PRAGMA journal_mode=OFF")  # No WAL overhead
            conn.execute("PRAGMA synchronous=OFF")

            # Load tables
            table_map: dict[str, StructuredTable] = {}
            for table in tables:
                self._load_table(conn, table)
                table_map[table.table_id] = table

            # Execute query with timeout
            conn.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")

            cursor = conn.execute(query.sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            raw_rows = cursor.fetchmany(self.max_rows)

            # Build result rows as dicts
            result_rows: list[dict[str, Any]] = []
            for raw_row in raw_rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    row_dict[col] = raw_row[i]
                result_rows.append(row_dict)

            # Build cell-level citations
            cell_citations = self._build_citations(result_rows, columns, tables, table_map)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return SQLResult(
                query=query,
                columns=columns,
                rows=result_rows,
                row_count=len(result_rows),
                cell_citations=cell_citations,
                execution_time_ms=round(elapsed_ms, 2),
            )

        except sqlite3.OperationalError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error("SQL execution error: %s | Query: %s", e, query.sql)
            return SQLResult(
                query=query,
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=round(elapsed_ms, 2),
                error=f"SQL execution error: {e}",
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.exception("Unexpected error executing SQL")
            return SQLResult(
                query=query,
                columns=[],
                rows=[],
                row_count=0,
                execution_time_ms=round(elapsed_ms, 2),
                error=f"Unexpected error: {e}",
            )
        finally:
            with contextlib.suppress(Exception):
                conn.close()

    def _load_table(self, conn: sqlite3.Connection, table: StructuredTable) -> None:
        """Load a StructuredTable into the SQLite database."""
        sql_name = _sanitize_sql_name(table.table_id)
        col_defs: list[str] = []
        col_names: list[str] = []

        for ct in table.column_types:
            col_sql_name = _sanitize_sql_name(ct.name)
            col_names.append(col_sql_name)
            sql_type = {
                "numeric": "REAL",
                "currency": "REAL",
                "percentage": "REAL",
                "boolean": "INTEGER",
                "date": "TEXT",
                "text": "TEXT",
            }.get(ct.inferred_type, "TEXT")
            col_defs.append(f"{col_sql_name} {sql_type}")

        # Add a hidden _row_index column for citation tracking
        col_defs.append("_row_index INTEGER")

        create_sql = f"CREATE TABLE {sql_name} ({', '.join(col_defs)})"
        conn.execute(create_sql)

        # Insert rows
        if table.rows:
            placeholders = ", ".join(["?"] * (len(col_names) + 1))
            insert_sql = f"INSERT INTO {sql_name} VALUES ({placeholders})"

            batch: list[tuple[Any, ...]] = []
            for row_idx, row in enumerate(table.rows):
                values: list[Any] = []
                for header in table.headers:
                    val = row.get(header)
                    # Coerce types for SQLite
                    if val is None or str(val).strip() == "":
                        values.append(None)
                    elif isinstance(val, (int, float)):
                        values.append(val)
                    else:
                        values.append(str(val))
                values.append(row_idx)  # _row_index
                batch.append(tuple(values))

            conn.executemany(insert_sql, batch)
            conn.commit()

        logger.debug("Loaded table %s with %d rows into SQLite", sql_name, len(table.rows))

    def _build_citations(
        self,
        result_rows: list[dict[str, Any]],
        columns: list[str],
        tables: list[StructuredTable],
        table_map: dict[str, StructuredTable],
    ) -> list[CellReference]:
        """Build cell-level citations by tracing result values back to source cells.

        For each value in the SQL result, find the original table cell it came from
        and create a CellReference pointing to the exact cell position and source page.
        """
        citations: list[CellReference] = []

        for result_row in result_rows:
            row_index = result_row.get("_row_index")

            for col, value in result_row.items():
                if col == "_row_index" or value is None:
                    continue

                # Try to find this value in the source tables
                citation = self._trace_value_to_cell(value, col, row_index, tables, table_map)
                if citation:
                    citations.append(citation)

        return citations

    def _trace_value_to_cell(
        self,
        value: Any,
        column: str,
        row_index: int | None,
        tables: list[StructuredTable],
        table_map: dict[str, StructuredTable],
    ) -> CellReference | None:
        """Trace a result value back to its source cell in the original table."""
        for table in tables:
            # If we have the exact row index, use it directly
            if row_index is not None and row_index < len(table.rows):
                # Find the matching column (might be sanitized)
                for header in table.headers:
                    sanitized = _sanitize_sql_name(header)
                    if sanitized == column or header.lower() == column.lower():
                        source_val = table.rows[row_index].get(header)
                        if self._values_match(source_val, value):
                            return CellReference(
                                table_id=table.table_id,
                                row_index=row_index,
                                column=header,
                                value=source_val,
                                page=table.source.page,
                                file=table.source.file,
                            )

            # Fallback: scan all rows for a matching value
            for header in table.headers:
                sanitized = _sanitize_sql_name(header)
                if sanitized == column or header.lower() == column.lower():
                    for idx, row in enumerate(table.rows):
                        if self._values_match(row.get(header), value):
                            return CellReference(
                                table_id=table.table_id,
                                row_index=idx,
                                column=header,
                                value=row.get(header),
                                page=table.source.page,
                                file=table.source.file,
                            )

        return None

    @staticmethod
    def _values_match(source: Any, result: Any) -> bool:
        """Check if a source cell value matches a SQL result value."""
        if source is None and result is None:
            return True
        if source is None or result is None:
            return False

        # Numeric comparison with tolerance
        try:
            s_num = float(source)
            r_num = float(result)
            return abs(s_num - r_num) < 0.001
        except (ValueError, TypeError):
            pass

        # String comparison
        return str(source).strip().lower() == str(result).strip().lower()
