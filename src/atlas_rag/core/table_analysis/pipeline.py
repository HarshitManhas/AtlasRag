"""Table Analysis Pipeline — orchestrates the full table understanding workflow.

This is the top-level entry point for all table-related operations in AtlasRAG.
It coordinates:
  1. Table extraction from documents (extractor.py)
  2. Query classification and table matching (detector.py)
  3. SQL generation from natural language (sql_generator.py)
  4. Safe SQL execution (executor.py)
  5. Cell-level citation generation (citation.py)

Usage:
    pipeline = TableAnalysisPipeline(settings)

    # During document ingestion:
    tables = await pipeline.extract_and_store(pdf_bytes, filename, doc_id, db_session)

    # During query answering:
    result = await pipeline.analyze_query("What was the total revenue?", doc_id, db_session)
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from atlas_rag.config import Settings, get_settings
from atlas_rag.core.table_analysis.citation import TableCitationEngine
from atlas_rag.core.table_analysis.detector import TableQueryDetector
from atlas_rag.core.table_analysis.executor import SQLExecutor
from atlas_rag.core.table_analysis.extractor import TableExtractor
from atlas_rag.core.table_analysis.sql_generator import SQLGenerator
from atlas_rag.models.table import ExtractedTable
from atlas_rag.schemas.table import (
    StructuredTable,
    TableAnalysisResult,
    TableQueryType,
    TableSource,
)

logger = logging.getLogger(__name__)


class TableAnalysisPipeline:
    """End-to-end pipeline for extracting, querying, and citing table data.

    This pipeline is designed to be called from two contexts:

    1. **Ingestion time** — when a document is uploaded, `extract_and_store()`
       parses tables from the PDF and saves them to the database.

    2. **Query time** — when a user asks a question, `analyze_query()` determines
       if it's a table query, finds relevant tables, generates SQL, executes it,
       and returns results with cell-level citations.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.extractor = TableExtractor()
        self.detector = TableQueryDetector()
        self.sql_generator = SQLGenerator(llm_model=self.settings.default_llm_model)
        self.executor = SQLExecutor()
        self.citation_engine = TableCitationEngine()

    # ── Ingestion-Time API ───────────────────────────────────────────────────

    async def extract_and_store(
        self,
        pdf_bytes: bytes,
        filename: str,
        document_id: str,
        db: AsyncSession,
    ) -> list[StructuredTable]:
        """Extract tables from a PDF and persist them to the database.

        Called during document ingestion. Returns the extracted StructuredTable
        objects for optional further processing (e.g., embedding the text
        representations).

        Args:
            pdf_bytes: Raw PDF file content.
            filename: Original filename for provenance.
            document_id: AtlasRAG document UUID.
            db: Async database session.

        Returns:
            List of extracted StructuredTable objects.
        """
        logger.info("Extracting tables from document %s (%s)", document_id, filename)
        self.extractor.reset_counter()

        # Extract tables from PDF bytes
        tables = self.extractor.extract_tables_from_bytes(
            pdf_bytes=pdf_bytes,
            filename=filename,
            document_id=document_id,
        )

        if not tables:
            logger.info("No tables found in %s", filename)
            return []

        # Persist each table to the database
        for table in tables:
            db_table = ExtractedTable(
                id=uuid.uuid4(),
                document_id=uuid.UUID(document_id),
                table_index=table.source.table_index,
                page_number=table.source.page,
                title=table.title,
                headers=table.headers,
                data=table.rows,
                row_count=table.row_count,
                col_count=table.col_count,
                column_types={ct.name: ct.inferred_type for ct in table.column_types},
                text_representation=table.text_representation,
                metadata_={
                    "table_id": table.table_id,
                    "hierarchical_headers": (
                        [h.model_dump() for h in table.hierarchical_headers]
                        if table.hierarchical_headers
                        else None
                    ),
                    "merged_cells": (
                        [m.model_dump() for m in table.merged_cells] if table.merged_cells else None
                    ),
                },
            )
            db.add(db_table)

        await db.flush()
        logger.info("Stored %d tables from %s (document %s)", len(tables), filename, document_id)

        return tables

    # ── Query-Time API ───────────────────────────────────────────────────────

    async def analyze_query(
        self,
        query: str,
        document_ids: list[str],
        db: AsyncSession,
    ) -> TableAnalysisResult:
        """Analyze a user query against tables from specified documents.

        This is the main query-time entry point. It:
          1. Classifies the query type (aggregation, comparison, lookup, etc.)
          2. Loads relevant tables from the database
          3. Matches the query to the most relevant tables
          4. Generates and executes SQL
          5. Builds cell-level citations
          6. Returns a structured result

        Args:
            query: User's natural language question.
            document_ids: Document UUIDs to search for tables.
            db: Async database session.

        Returns:
            TableAnalysisResult with answer, SQL result, and citations.
        """
        # Step 1: Load tables from database
        tables = await self._load_tables(document_ids, db)
        if not tables:
            return TableAnalysisResult(
                query_type=TableQueryType.NONE,
                matched_tables=[],
                natural_language_answer="No tables found in the specified documents.",
                confidence=0.0,
            )

        # Step 2: Classify the query
        query_type, confidence = self.detector.classify_query(query, tables)

        if query_type == TableQueryType.NONE or confidence < 0.3:
            return TableAnalysisResult(
                query_type=TableQueryType.NONE,
                matched_tables=[],
                natural_language_answer="",
                confidence=confidence,
            )

        # Step 3: Match query to relevant tables
        matched_tables = self.detector.match_tables(query, tables)
        if not matched_tables:
            # Fall back to all tables if matching found nothing
            matched_tables = tables[:3]  # Limit to first 3 tables

        matched_table_ids = [t.table_id for t in matched_tables]

        # Step 4: Generate SQL
        sql_query = await self.sql_generator.generate_sql(
            question=query,
            tables=matched_tables,
            query_type=query_type,
        )

        # Step 5: Execute SQL
        sql_result = self.executor.execute(sql_query, matched_tables)

        # Step 6: Build citations
        if sql_result.error:
            cell_citations = []
            nl_answer = f"Unable to execute table query: {sql_result.error}"
        else:
            cell_citations = self.citation_engine.cite_sql_result(sql_result, matched_tables)

            # For aggregations, also cite the contributing cells
            if query_type == TableQueryType.AGGREGATION and sql_result.rows:
                for table in matched_tables:
                    for ct in table.column_types:
                        if ct.inferred_type in ("numeric", "currency", "percentage"):
                            agg_citations = self.citation_engine.cite_aggregation(
                                aggregation_value=None,
                                column=ct.name,
                                operation="aggregate",
                                table=table,
                            )
                            cell_citations.extend(agg_citations)

            nl_answer = self._build_natural_language_answer(
                query, query_type, sql_result, matched_tables
            )

        return TableAnalysisResult(
            query_type=query_type,
            matched_tables=matched_table_ids,
            sql_result=sql_result,
            natural_language_answer=nl_answer,
            cell_citations=cell_citations,
            confidence=confidence,
        )

    # ── Internal Helpers ─────────────────────────────────────────────────────

    async def _load_tables(
        self,
        document_ids: list[str],
        db: AsyncSession,
    ) -> list[StructuredTable]:
        """Load ExtractedTable rows from the DB and convert to StructuredTable objects."""
        doc_uuids = [uuid.UUID(did) for did in document_ids]

        stmt = select(ExtractedTable).where(ExtractedTable.document_id.in_(doc_uuids))
        result = await db.execute(stmt)
        db_tables = result.scalars().all()

        tables: list[StructuredTable] = []
        for db_table in db_tables:
            metadata = db_table.metadata_ or {}
            table_id = metadata.get("table_id", f"tbl_{db_table.table_index:03d}")

            source = TableSource(
                page=db_table.page_number,
                file="",  # Will be filled from document info
                document_id=str(db_table.document_id),
                table_index=db_table.table_index,
            )

            # Reconstruct column types from stored column_types dict
            from atlas_rag.schemas.table import ColumnTypeDef

            column_types = []
            stored_types = db_table.column_types or {}
            for header in db_table.headers:
                column_types.append(
                    ColumnTypeDef(
                        name=header,
                        inferred_type=stored_types.get(header, "text"),
                    )
                )

            tables.append(
                StructuredTable(
                    table_id=table_id,
                    title=db_table.title,
                    headers=db_table.headers,
                    column_types=column_types,
                    rows=db_table.data,
                    row_count=db_table.row_count,
                    col_count=db_table.col_count,
                    source=source,
                    text_representation=db_table.text_representation,
                )
            )

        return tables

    def _build_natural_language_answer(
        self,
        query: str,
        query_type: TableQueryType,
        sql_result: Any,
        tables: list[StructuredTable],
    ) -> str:
        """Build a human-readable answer from SQL results.

        This is a template-based approach for common query types.
        For complex queries, the Generation Service will use the LLM to
        produce a more sophisticated answer.
        """
        if sql_result.error:
            return f"Error analyzing table data: {sql_result.error}"

        if not sql_result.rows:
            return "No matching data found in the tables."

        if query_type == TableQueryType.AGGREGATION:
            # Format aggregation results
            row = sql_result.rows[0]
            parts = []
            for col, val in row.items():
                if col.startswith("_"):
                    continue
                if isinstance(val, float):
                    parts.append(f"{col}: {val:,.2f}")
                else:
                    parts.append(f"{col}: {val}")
            return "Based on the table data: " + ", ".join(parts)

        elif query_type == TableQueryType.RANKING:
            lines = ["Based on the table data, here are the results:\n"]
            for i, row in enumerate(sql_result.rows[:10], 1):
                vals = [f"{col}={val}" for col, val in row.items() if not col.startswith("_")]
                lines.append(f"  {i}. {', '.join(vals)}")
            return "\n".join(lines)

        elif query_type == TableQueryType.LOOKUP:
            if len(sql_result.rows) == 1:
                row = sql_result.rows[0]
                parts = [f"{col}: {val}" for col, val in row.items() if not col.startswith("_")]
                return "Found: " + ", ".join(parts)
            else:
                return f"Found {len(sql_result.rows)} matching rows in the table data."

        else:
            return f"Found {len(sql_result.rows)} results from table analysis."
