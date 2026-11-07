"""Pydantic schemas for table extraction, analysis, and citation."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field

# ─── Table Extraction Schemas ────────────────────────────────────────────────


class CellReference(BaseModel):
    """Pinpoints a single cell in an extracted table for citation purposes."""

    table_id: str = Field(..., description="Unique table identifier (e.g., tbl_001)")
    row_index: int = Field(..., ge=0, description="Zero-based row index")
    column: str = Field(..., description="Column header name")
    value: Any = Field(..., description="The cell value")
    page: int | None = Field(None, description="Source document page number")
    file: str | None = Field(None, description="Source document filename")


class ColumnTypeDef(BaseModel):
    """Inferred type information for a table column."""

    name: str
    inferred_type: str = Field(
        ..., description="One of: numeric, text, date, boolean, currency, percentage"
    )
    nullable: bool = False
    sample_values: list[Any] = Field(default_factory=list, max_length=5)


class MergedCellRegion(BaseModel):
    """Describes a merged cell region in the original table layout."""

    start_row: int
    end_row: int
    start_col: int
    end_col: int
    value: str


class TableHeader(BaseModel):
    """Represents a potentially hierarchical table header."""

    level: int = Field(0, description="Header nesting level (0 = top-level)")
    name: str
    parent: str | None = None
    span: int = Field(1, description="Number of columns this header spans")


class StructuredTable(BaseModel):
    """A fully parsed, normalized table extracted from a document.

    This is the canonical representation that AtlasRAG uses for all table
    operations: SQL-based analysis, cell-level citations, and embedding.

    Example:
        {
            "table_id": "tbl_001",
            "headers": ["Product", "Revenue"],
            "rows": [
                {"Product": "A", "Revenue": 150},
                {"Product": "B", "Revenue": 250}
            ],
            "source": {"page": 12, "file": "annual_report.pdf"}
        }
    """

    table_id: str = Field(..., description="Unique table ID (tbl_NNN format)")
    title: str | None = Field(None, description="Table caption or inferred title")

    # Column structure
    headers: list[str] = Field(..., description="Flat list of column names")
    hierarchical_headers: list[TableHeader] | None = Field(
        None, description="Multi-level header hierarchy if detected"
    )
    column_types: list[ColumnTypeDef] = Field(
        default_factory=list, description="Inferred type for each column"
    )

    # Data
    rows: list[dict[str, Any]] = Field(
        ..., description="Row data as list of column-name → value dicts"
    )
    row_count: int = Field(0, ge=0)
    col_count: int = Field(0, ge=0)

    # Provenance
    source: TableSource
    merged_cells: list[MergedCellRegion] = Field(
        default_factory=list, description="Original merged cell regions before normalization"
    )

    # Text representation for embedding / retrieval
    text_representation: str | None = Field(
        None, description="Markdown or natural-language rendering of the table"
    )
    summary: str | None = Field(
        None, description="LLM-generated or heuristic summary of table content"
    )

    def get_cell(self, row_index: int, column: str) -> CellReference:
        """Return a CellReference for a specific cell position."""
        if row_index >= len(self.rows):
            raise IndexError(
                f"Row index {row_index} out of range (table has {len(self.rows)} rows)"
            )
        if column not in self.headers:
            raise KeyError(f"Column {column!r} not in table headers: {self.headers}")
        return CellReference(
            table_id=self.table_id,
            row_index=row_index,
            column=column,
            value=self.rows[row_index].get(column),
            page=self.source.page,
            file=self.source.file,
        )

    def to_markdown(self) -> str:
        """Render the table as a Markdown table string."""
        if not self.headers or not self.rows:
            return ""
        lines = []
        lines.append("| " + " | ".join(self.headers) + " |")
        lines.append("| " + " | ".join("---" for _ in self.headers) + " |")
        for row in self.rows:
            cells = [str(row.get(h, "")) for h in self.headers]
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)


class TableSource(BaseModel):
    """Provenance information for a table."""

    page: int | None = Field(None, description="Page number in the source document")
    file: str = Field(..., description="Source filename")
    document_id: str | None = Field(None, description="AtlasRAG document UUID")
    table_index: int = Field(0, description="Table order on the page (0-based)")


# ── Update forward reference ─────────────────────────────────────────────────
StructuredTable.model_rebuild()


# ─── SQL Analysis Schemas ────────────────────────────────────────────────────


class SQLQuery(BaseModel):
    """A generated SQL query to be executed against table data."""

    sql: str = Field(..., description="The SQL query string")
    target_table_ids: list[str] = Field(..., description="Table IDs this query references")
    explanation: str = Field("", description="Natural-language explanation of the query")
    is_aggregation: bool = Field(False, description="Whether this performs aggregation")


class SQLResult(BaseModel):
    """Result of executing a SQL query against extracted table data."""

    query: SQLQuery | None = None
    columns: list[str] = Field(default_factory=list, description="Result column names")
    rows: list[dict[str, Any]] = Field(default_factory=list, description="Result rows")
    row_count: int = 0
    cell_citations: list[CellReference] = Field(
        default_factory=list,
        description="Cell-level citations for values used in the result",
    )
    execution_time_ms: float = 0.0
    error: str | None = None


# ─── Table Analysis Response ─────────────────────────────────────────────────


class TableQueryType(str, enum.Enum):
    """Classification of how a query relates to table data."""

    LOOKUP = "lookup"  # Find specific values
    AGGREGATION = "aggregation"  # Sum, avg, count, etc.
    COMPARISON = "comparison"  # Compare rows/columns
    RANKING = "ranking"  # Top-N, bottom-N
    TREND = "trend"  # Time-series, growth
    FILTER = "filter"  # Filter by conditions
    NONE = "none"  # Not a table query


class TableAnalysisResult(BaseModel):
    """Complete result of analyzing a query against table data."""

    query_type: TableQueryType
    matched_tables: list[str] = Field(..., description="Table IDs relevant to the query")
    sql_result: SQLResult | None = None
    natural_language_answer: str = Field(
        "", description="Human-readable answer derived from table data"
    )
    cell_citations: list[CellReference] = Field(
        default_factory=list,
        description="All cells referenced in producing the answer",
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)
