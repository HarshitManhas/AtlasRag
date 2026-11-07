"""SQL Generator — converts natural-language table queries to SQL.

Uses LLM-based text-to-SQL generation with schema-aware prompting.
The generated SQL targets SQLite, which is used as an ephemeral query engine
for extracted table data.
"""

from __future__ import annotations

import logging

from atlas_rag.core.generation.llm import LLMClient
from atlas_rag.schemas.table import SQLQuery, StructuredTable, TableQueryType

logger = logging.getLogger(__name__)


# ─── Schema Description Builder ──────────────────────────────────────────────


def build_schema_description(tables: list[StructuredTable]) -> str:
    """Build a SQL schema description from StructuredTable objects.

    Produces CREATE TABLE statements that mirror the table structure,
    which are included in the LLM prompt for accurate SQL generation.
    """
    schema_parts: list[str] = []

    for table in tables:
        # Sanitize table name for SQL
        sql_table_name = _sanitize_sql_name(table.table_id)

        col_defs: list[str] = []
        for ct in table.column_types:
            col_name = _sanitize_sql_name(ct.name)
            sql_type = _type_to_sql(ct.inferred_type)
            nullable = " NULL" if ct.nullable else " NOT NULL"
            col_defs.append(f"  {col_name} {sql_type}{nullable}")

        create_stmt = f"CREATE TABLE {sql_table_name} (\n"
        create_stmt += ",\n".join(col_defs)
        create_stmt += "\n);"

        # Add context as a comment
        context_parts = []
        if table.title:
            context_parts.append(f"Title: {table.title}")
        context_parts.append(f"Rows: {table.row_count}")
        context_parts.append(f"Source: page {table.source.page} of {table.source.file}")

        comment = f"-- {'; '.join(context_parts)}"
        schema_parts.append(f"{comment}\n{create_stmt}")

        # Add sample data as comments for context
        if table.rows:
            sample_rows = table.rows[: min(3, len(table.rows))]
            schema_parts.append(f"-- Sample data for {sql_table_name}:")
            for row in sample_rows:
                values = [f"{k}={v}" for k, v in row.items()]
                schema_parts.append(f"--   {', '.join(values)}")

    return "\n\n".join(schema_parts)


def _sanitize_sql_name(name: str) -> str:
    """Sanitize a name for use as a SQL identifier."""
    import re

    # Replace spaces and special chars with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Ensure it doesn't start with a digit
    if sanitized and sanitized[0].isdigit():
        sanitized = f"col_{sanitized}"
    # Remove consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_").lower() or "unnamed"


def _type_to_sql(inferred_type: str) -> str:
    """Map an inferred column type to a SQLite type."""
    type_map = {
        "numeric": "REAL",
        "currency": "REAL",
        "percentage": "REAL",
        "integer": "INTEGER",
        "boolean": "INTEGER",
        "date": "TEXT",
        "text": "TEXT",
    }
    return type_map.get(inferred_type, "TEXT")


# ─── Prompt Templates ───────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a SQL expert. Given a natural language question and a database schema,
generate a correct SQLite query to answer the question.

Rules:
1. Use only the tables and columns defined in the schema.
2. Use SQLite-compatible syntax only.
3. For aggregations, use appropriate GROUP BY clauses.
4. Use aliases for readability.
5. Handle NULL values properly with COALESCE or IS NOT NULL.
6. Limit results to 50 rows maximum unless the query is an aggregation.
7. NEVER use DROP, DELETE, INSERT, UPDATE, ALTER, or CREATE statements.
8. Return ONLY the SQL query, nothing else."""

_USER_PROMPT_TEMPLATE = """Database Schema:
{schema}

Question: {question}

Generate a SQLite query to answer this question. Return ONLY the SQL query."""


# ─── SQL Generator ───────────────────────────────────────────────────────────


class SQLGenerator:
    """Generates SQL queries from natural-language questions using LLM.

    Uses the table schemas to construct informed prompts, then validates
    the generated SQL before returning it.
    """

    def __init__(self, llm_model: str = "gpt-4o") -> None:
        self.llm_model = llm_model

    async def generate_sql(
        self,
        question: str,
        tables: list[StructuredTable],
        query_type: TableQueryType = TableQueryType.LOOKUP,
    ) -> SQLQuery:
        """Generate a SQL query from a natural language question.

        Args:
            question: User's natural language question.
            tables: Available tables with their schemas.
            query_type: Classified query type for context.

        Returns:
            A SQLQuery object with the generated SQL and metadata.
        """
        schema = build_schema_description(tables)
        target_table_ids = [t.table_id for t in tables]

        prompt = _USER_PROMPT_TEMPLATE.format(
            schema=schema,
            question=question,
        )

        try:
            response = await LLMClient().complete(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=500,
            )

            raw_sql = response.content.strip()
            sql = self._clean_sql(raw_sql)

            # Validate the SQL is safe
            self._validate_sql(sql)

            return SQLQuery(
                sql=sql,
                target_table_ids=target_table_ids,
                explanation=f"SQL generated for: {question}",
                is_aggregation=query_type == TableQueryType.AGGREGATION,
            )

        except Exception as e:
            logger.error("SQL generation failed: %s", e)
            # Fallback: try to generate a simple query heuristically
            return self._fallback_generate(question, tables, query_type)

    def _clean_sql(self, raw_sql: str) -> str:
        """Clean LLM-generated SQL by removing markdown fences and extra whitespace."""
        sql = raw_sql.strip()

        # Remove markdown code fences
        if sql.startswith("```"):
            lines = sql.split("\n")
            # Remove first and last lines if they're fences
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            sql = "\n".join(lines).strip()

        # Remove leading "sql" keyword if present
        if sql.lower().startswith("sql\n"):
            sql = sql[4:].strip()

        return sql

    def _validate_sql(self, sql: str) -> None:
        """Validate that generated SQL is safe (read-only, no destructive ops)."""
        sql_upper = sql.upper().strip()
        forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE"]

        for keyword in forbidden:
            # Check for keyword at word boundary
            import re

            if re.search(rf"\b{keyword}\b", sql_upper):
                raise ValueError(f"Generated SQL contains forbidden keyword: {keyword}")

    def _fallback_generate(
        self,
        question: str,
        tables: list[StructuredTable],
        query_type: TableQueryType,
    ) -> SQLQuery:
        """Heuristic-based SQL generation as fallback when LLM is unavailable.

        Handles simple cases: SELECT * lookups, basic aggregations, and filters.
        """
        if not tables:
            return SQLQuery(
                sql="SELECT 'No tables available' AS error",
                target_table_ids=[],
                explanation="No tables available for query",
            )

        table = tables[0]
        sql_name = _sanitize_sql_name(table.table_id)
        [_sanitize_sql_name(h) for h in table.headers]

        question_lower = question.lower()

        if query_type == TableQueryType.AGGREGATION:
            # Find numeric columns for aggregation
            numeric_cols = [
                _sanitize_sql_name(ct.name)
                for ct in table.column_types
                if ct.inferred_type in ("numeric", "currency", "percentage")
            ]
            if numeric_cols:
                agg_col = numeric_cols[0]
                if "total" in question_lower or "sum" in question_lower:
                    sql = f"SELECT SUM({agg_col}) AS total_{agg_col} FROM {sql_name}"
                elif "average" in question_lower or "mean" in question_lower:
                    sql = f"SELECT AVG({agg_col}) AS avg_{agg_col} FROM {sql_name}"
                elif "max" in question_lower or "highest" in question_lower:
                    sql = f"SELECT MAX({agg_col}) AS max_{agg_col} FROM {sql_name}"
                elif "min" in question_lower or "lowest" in question_lower:
                    sql = f"SELECT MIN({agg_col}) AS min_{agg_col} FROM {sql_name}"
                else:
                    sql = f"SELECT SUM({agg_col}) AS total, AVG({agg_col}) AS average, COUNT(*) AS count FROM {sql_name}"
            else:
                sql = f"SELECT COUNT(*) AS count FROM {sql_name}"

        elif query_type == TableQueryType.RANKING:
            numeric_cols = [
                _sanitize_sql_name(ct.name)
                for ct in table.column_types
                if ct.inferred_type in ("numeric", "currency", "percentage")
            ]
            if numeric_cols:
                order_col = numeric_cols[0]
                direction = (
                    "DESC"
                    if any(
                        w in question_lower for w in ["top", "highest", "largest", "best", "most"]
                    )
                    else "ASC"
                )
                limit = 5  # Default top/bottom 5
                sql = f"SELECT * FROM {sql_name} ORDER BY {order_col} {direction} LIMIT {limit}"
            else:
                sql = f"SELECT * FROM {sql_name} LIMIT 10"

        else:
            # Default: select all with limit
            sql = f"SELECT * FROM {sql_name} LIMIT 50"

        return SQLQuery(
            sql=sql,
            target_table_ids=[table.table_id],
            explanation=f"Fallback SQL for query type: {query_type.value}",
            is_aggregation=query_type == TableQueryType.AGGREGATION,
        )
