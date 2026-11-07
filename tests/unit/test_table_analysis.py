"""Tests for the Table Analysis subsystem.

Covers:
  - Table extraction and normalization
  - Column type inference
  - Merged cell detection and resolution
  - Query classification
  - SQL generation (fallback/heuristic)
  - SQL execution with cell citations
  - Citation engine
"""

from __future__ import annotations

import pandas as pd
import pytest

from atlas_rag.core.table_analysis.citation import TableCitationEngine
from atlas_rag.core.table_analysis.detector import TableQueryDetector
from atlas_rag.core.table_analysis.executor import SQLExecutor
from atlas_rag.core.table_analysis.extractor import (
    _clean_numeric_value,
    _detect_merged_cells,
    _infer_column_type,
    _resolve_merged_cells,
    _table_to_text,
)
from atlas_rag.core.table_analysis.sql_generator import (
    SQLGenerator,
    _sanitize_sql_name,
    build_schema_description,
)
from atlas_rag.schemas.table import (
    CellReference,
    ColumnTypeDef,
    SQLQuery,
    StructuredTable,
    TableQueryType,
    TableSource,
)

# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_table() -> StructuredTable:
    """A simple product revenue table for testing."""
    return StructuredTable(
        table_id="tbl_001",
        title="Product Revenue Summary",
        headers=["Product", "Revenue", "Units Sold", "Region"],
        column_types=[
            ColumnTypeDef(name="Product", inferred_type="text"),
            ColumnTypeDef(name="Revenue", inferred_type="numeric"),
            ColumnTypeDef(name="Units Sold", inferred_type="numeric"),
            ColumnTypeDef(name="Region", inferred_type="text"),
        ],
        rows=[
            {"Product": "Widget A", "Revenue": 150000, "Units Sold": 1200, "Region": "North"},
            {"Product": "Widget B", "Revenue": 250000, "Units Sold": 2100, "Region": "South"},
            {"Product": "Gadget C", "Revenue": 180000, "Units Sold": 900, "Region": "East"},
            {"Product": "Gadget D", "Revenue": 320000, "Units Sold": 3400, "Region": "West"},
            {"Product": "Widget E", "Revenue": 95000, "Units Sold": 650, "Region": "North"},
        ],
        row_count=5,
        col_count=4,
        source=TableSource(page=12, file="annual_report.pdf", document_id="doc-001"),
    )


@pytest.fixture
def financial_table() -> StructuredTable:
    """A financial table with currency and percentage columns."""
    return StructuredTable(
        table_id="tbl_002",
        title="Q4 Financial Results",
        headers=["Quarter", "Revenue", "Costs", "Margin"],
        column_types=[
            ColumnTypeDef(name="Quarter", inferred_type="text"),
            ColumnTypeDef(name="Revenue", inferred_type="currency"),
            ColumnTypeDef(name="Costs", inferred_type="currency"),
            ColumnTypeDef(name="Margin", inferred_type="percentage"),
        ],
        rows=[
            {"Quarter": "Q1", "Revenue": 1200000, "Costs": 800000, "Margin": 33.3},
            {"Quarter": "Q2", "Revenue": 1500000, "Costs": 950000, "Margin": 36.7},
            {"Quarter": "Q3", "Revenue": 1350000, "Costs": 870000, "Margin": 35.6},
            {"Quarter": "Q4", "Revenue": 1800000, "Costs": 1100000, "Margin": 38.9},
        ],
        row_count=4,
        col_count=4,
        source=TableSource(page=5, file="quarterly_report.pdf", document_id="doc-002"),
    )


# ─── Column Type Inference Tests ─────────────────────────────────────────────


class TestColumnTypeInference:
    def test_numeric_column(self):
        assert _infer_column_type([100, 200, 300]) == "numeric"
        assert _infer_column_type(["100", "200.5", "300"]) == "numeric"

    def test_text_column(self):
        assert _infer_column_type(["apple", "banana", "cherry"]) == "text"

    def test_currency_column(self):
        assert _infer_column_type(["$100", "$200.50", "$300"]) == "currency"
        assert _infer_column_type(["€1,000", "€2,500"]) == "currency"

    def test_percentage_column(self):
        assert _infer_column_type(["45%", "30.5%", "78%"]) == "percentage"

    def test_boolean_column(self):
        assert _infer_column_type(["yes", "no", "yes", "no"]) == "boolean"
        assert _infer_column_type(["true", "false", "true"]) == "boolean"

    def test_date_column(self):
        assert _infer_column_type(["2024-01-15", "2024-02-20"]) == "date"

    def test_empty_column(self):
        assert _infer_column_type([]) == "text"
        assert _infer_column_type([None, None, ""]) == "text"

    def test_mixed_with_majority_numeric(self):
        assert _infer_column_type(["100", "200", "N/A", "400", "500"]) == "numeric"


class TestNumericCleaning:
    def test_integer(self):
        assert _clean_numeric_value("100") == 100

    def test_float(self):
        assert _clean_numeric_value("99.5") == 99.5

    def test_currency(self):
        assert _clean_numeric_value("$1,234.56") == 1234.56

    def test_euro(self):
        assert _clean_numeric_value("€500") == 500

    def test_percentage(self):
        assert _clean_numeric_value("45%") == 45.0

    def test_comma_thousands(self):
        assert _clean_numeric_value("1,000,000") == 1000000

    def test_non_numeric(self):
        assert _clean_numeric_value("hello") == "hello"

    def test_none(self):
        assert _clean_numeric_value("") is None

    def test_already_numeric(self):
        assert _clean_numeric_value(42) == 42
        assert _clean_numeric_value(3.14) == 3.14


# ─── Merged Cell Detection Tests ────────────────────────────────────────────


class TestMergedCells:
    def test_detect_merged_cells(self):
        df = pd.DataFrame(
            {
                "Category": ["Electronics", "", "", "Clothing", ""],
                "Product": ["Phone", "Laptop", "Tablet", "Shirt", "Pants"],
                "Price": [999, 1299, 499, 49, 79],
            }
        )
        merged = _detect_merged_cells(df)
        assert len(merged) >= 1
        # Should detect "Electronics" spanning rows 0-2
        electronics_merge = [m for m in merged if m.value == "Electronics"]
        assert len(electronics_merge) == 1
        assert electronics_merge[0].start_row == 0
        assert electronics_merge[0].end_row == 2

    def test_resolve_merged_cells(self):
        df = pd.DataFrame(
            {
                "Category": ["Electronics", "", "", "Clothing", ""],
                "Product": ["Phone", "Laptop", "Tablet", "Shirt", "Pants"],
            }
        )
        resolved = _resolve_merged_cells(df)
        assert resolved.iloc[1]["Category"] == "Electronics"
        assert resolved.iloc[2]["Category"] == "Electronics"
        assert resolved.iloc[4]["Category"] == "Clothing"


# ─── Query Classification Tests ─────────────────────────────────────────────


class TestQueryDetector:
    def setup_method(self):
        self.detector = TableQueryDetector()

    def test_aggregation_query(self, sample_table):
        qtype, conf = self.detector.classify_query(
            "What is the total revenue across all products?",
            [sample_table],
        )
        assert qtype == TableQueryType.AGGREGATION
        assert conf > 0.3

    def test_comparison_query(self, sample_table):
        qtype, conf = self.detector.classify_query(
            "Compare revenue between Widget A and Widget B",
            [sample_table],
        )
        assert qtype == TableQueryType.COMPARISON
        assert conf > 0.3

    def test_ranking_query(self, sample_table):
        qtype, conf = self.detector.classify_query(
            "What are the top 3 products by revenue?",
            [sample_table],
        )
        assert qtype == TableQueryType.RANKING
        assert conf > 0.3

    def test_lookup_query(self, sample_table):
        qtype, conf = self.detector.classify_query(
            "What is the revenue for Widget A?",
            [sample_table],
        )
        assert qtype == TableQueryType.LOOKUP
        assert conf > 0.3

    def test_non_table_query(self):
        qtype, _conf = self.detector.classify_query(
            "Explain the company's mission statement and values.",
        )
        assert qtype == TableQueryType.NONE

    def test_column_name_boost(self, sample_table):
        """Confidence should increase when query mentions column names."""
        _, conf_without = self.detector.classify_query(
            "What is the total?",
        )
        _, conf_with = self.detector.classify_query(
            "What is the total revenue?",
            [sample_table],
        )
        assert conf_with >= conf_without

    def test_match_tables(self, sample_table, financial_table):
        matched = self.detector.match_tables(
            "What was the revenue for Widget A?",
            [sample_table, financial_table],
        )
        # Should match the product table (has "Widget A" and "Revenue")
        assert len(matched) >= 1
        assert matched[0].table_id == "tbl_001"


# ─── SQL Name Sanitization Tests ────────────────────────────────────────────


class TestSQLNameSanitization:
    def test_spaces(self):
        assert _sanitize_sql_name("Units Sold") == "units_sold"

    def test_special_chars(self):
        assert _sanitize_sql_name("Revenue ($)") == "revenue"

    def test_leading_digit(self):
        assert _sanitize_sql_name("2024 Revenue") == "col_2024_revenue"

    def test_empty(self):
        assert _sanitize_sql_name("") == "unnamed"


# ─── Schema Description Tests ───────────────────────────────────────────────


class TestSchemaDescription:
    def test_generates_create_table(self, sample_table):
        schema = build_schema_description([sample_table])
        assert "CREATE TABLE" in schema
        assert "tbl_001" in schema
        assert "revenue" in schema.lower()
        assert "REAL" in schema  # numeric columns

    def test_includes_sample_data(self, sample_table):
        schema = build_schema_description([sample_table])
        assert "Sample data" in schema
        assert "Widget A" in schema


# ─── SQL Execution Tests ────────────────────────────────────────────────────


class TestSQLExecutor:
    def setup_method(self):
        self.executor = SQLExecutor()

    def test_simple_select(self, sample_table):
        query = SQLQuery(
            sql=f"SELECT * FROM {_sanitize_sql_name(sample_table.table_id)}",
            target_table_ids=[sample_table.table_id],
        )
        result = self.executor.execute(query, [sample_table])
        assert result.error is None
        assert result.row_count == 5
        assert len(result.columns) > 0

    def test_aggregation(self, sample_table):
        query = SQLQuery(
            sql=f"SELECT SUM(revenue) AS total_revenue FROM {_sanitize_sql_name(sample_table.table_id)}",
            target_table_ids=[sample_table.table_id],
            is_aggregation=True,
        )
        result = self.executor.execute(query, [sample_table])
        assert result.error is None
        assert result.row_count == 1
        total = result.rows[0]["total_revenue"]
        assert total == 150000 + 250000 + 180000 + 320000 + 95000

    def test_filter(self, sample_table):
        query = SQLQuery(
            sql=f"SELECT product, revenue FROM {_sanitize_sql_name(sample_table.table_id)} WHERE revenue > 200000",
            target_table_ids=[sample_table.table_id],
        )
        result = self.executor.execute(query, [sample_table])
        assert result.error is None
        assert result.row_count == 2  # Widget B (250000) and Gadget D (320000)

    def test_ordering(self, sample_table):
        query = SQLQuery(
            sql=f"SELECT product, revenue FROM {_sanitize_sql_name(sample_table.table_id)} ORDER BY revenue DESC LIMIT 3",
            target_table_ids=[sample_table.table_id],
        )
        result = self.executor.execute(query, [sample_table])
        assert result.error is None
        assert result.row_count == 3
        assert result.rows[0]["product"] == "Gadget D"

    def test_invalid_sql(self, sample_table):
        query = SQLQuery(
            sql="SELECT * FROM nonexistent_table",
            target_table_ids=[sample_table.table_id],
        )
        result = self.executor.execute(query, [sample_table])
        assert result.error is not None

    def test_cell_citations_generated(self, sample_table):
        query = SQLQuery(
            sql=f"SELECT product, revenue FROM {_sanitize_sql_name(sample_table.table_id)} WHERE product = 'Widget A'",
            target_table_ids=[sample_table.table_id],
        )
        result = self.executor.execute(query, [sample_table])
        assert result.error is None
        assert len(result.cell_citations) > 0
        # Should cite the "Widget A" / "Revenue" cells
        revenue_cite = [c for c in result.cell_citations if c.column == "Revenue"]
        assert len(revenue_cite) > 0
        assert revenue_cite[0].value == 150000


# ─── Citation Engine Tests ───────────────────────────────────────────────────


class TestCitationEngine:
    def setup_method(self):
        self.engine = TableCitationEngine()

    def test_cite_from_text(self, sample_table):
        text = "Widget A generated revenue of 150000 units while Widget B reached 250000."
        citations = self.engine.cite_from_text(text, [sample_table])
        # Should find citations for Widget A, Widget B, 150000, 250000
        assert len(citations) >= 2

        # Check that Widget A revenue is cited
        widget_a_rev = [c for c in citations if c.value == 150000 and c.column == "Revenue"]
        assert len(widget_a_rev) == 1
        assert widget_a_rev[0].table_id == "tbl_001"
        assert widget_a_rev[0].row_index == 0
        assert widget_a_rev[0].page == 12

    def test_cite_aggregation(self, sample_table):
        citations = self.engine.cite_aggregation(
            aggregation_value=995000,
            column="Revenue",
            operation="sum",
            table=sample_table,
        )
        # Should cite ALL revenue cells (5 rows)
        assert len(citations) == 5
        assert all(c.column == "Revenue" for c in citations)

    def test_format_citation(self):
        ref = CellReference(
            table_id="tbl_001",
            row_index=0,
            column="Revenue",
            value=150000,
            page=12,
            file="annual_report.pdf",
        )
        formatted = self.engine.format_citation(ref)
        assert "tbl_001" in formatted
        assert "Row 1" in formatted  # 1-indexed for humans
        assert "Revenue" in formatted
        assert "150000" in formatted
        assert "Page 12" in formatted

    def test_format_citations_block(self, sample_table):
        citations = [
            CellReference(
                table_id="tbl_001",
                row_index=0,
                column="Revenue",
                value=150000,
                page=12,
                file="annual_report.pdf",
            ),
            CellReference(
                table_id="tbl_001",
                row_index=1,
                column="Revenue",
                value=250000,
                page=12,
                file="annual_report.pdf",
            ),
        ]
        block = self.engine.format_citations_block(citations)
        assert "Table Citations" in block
        assert "tbl_001" in block
        assert "annual_report.pdf" in block


# ─── Text Representation Tests ───────────────────────────────────────────────


class TestTextRepresentation:
    def test_generates_text(self, sample_table):
        text = _table_to_text(sample_table)
        assert "Product Revenue Summary" in text
        assert "Columns:" in text
        assert "Revenue" in text
        assert "Widget A" in text

    def test_includes_stats(self, sample_table):
        text = _table_to_text(sample_table)
        assert "min=" in text
        assert "max=" in text
        assert "sum=" in text

    def test_markdown_output(self, sample_table):
        md = sample_table.to_markdown()
        assert "| Product |" in md
        assert "| Widget A |" in md
        assert "---" in md  # separator row


# ─── Table Cell Access Tests ────────────────────────────────────────────────


class TestStructuredTableAccess:
    def test_get_cell(self, sample_table):
        ref = sample_table.get_cell(0, "Revenue")
        assert ref.table_id == "tbl_001"
        assert ref.row_index == 0
        assert ref.column == "Revenue"
        assert ref.value == 150000
        assert ref.page == 12

    def test_get_cell_out_of_range(self, sample_table):
        with pytest.raises(IndexError):
            sample_table.get_cell(100, "Revenue")

    def test_get_cell_invalid_column(self, sample_table):
        with pytest.raises(KeyError):
            sample_table.get_cell(0, "NonexistentColumn")


# ─── SQL Generator Fallback Tests ───────────────────────────────────────────


class TestSQLGeneratorFallback:
    def setup_method(self):
        self.generator = SQLGenerator()

    def test_fallback_aggregation_sum(self, sample_table):
        result = self.generator._fallback_generate(
            "What is the total revenue?",
            [sample_table],
            TableQueryType.AGGREGATION,
        )
        assert "SUM" in result.sql.upper()
        assert result.is_aggregation

    def test_fallback_aggregation_avg(self, sample_table):
        result = self.generator._fallback_generate(
            "What is the average revenue?",
            [sample_table],
            TableQueryType.AGGREGATION,
        )
        assert "AVG" in result.sql.upper()

    def test_fallback_ranking_top(self, sample_table):
        result = self.generator._fallback_generate(
            "What are the top 5 products?",
            [sample_table],
            TableQueryType.RANKING,
        )
        assert "ORDER BY" in result.sql.upper()
        assert "DESC" in result.sql.upper()

    def test_fallback_ranking_bottom(self, sample_table):
        result = self.generator._fallback_generate(
            "What are the lowest revenue products?",
            [sample_table],
            TableQueryType.RANKING,
        )
        assert "ASC" in result.sql.upper()

    def test_fallback_lookup(self, sample_table):
        result = self.generator._fallback_generate(
            "Show the data",
            [sample_table],
            TableQueryType.LOOKUP,
        )
        assert "SELECT" in result.sql.upper()
        assert "LIMIT" in result.sql.upper()

    def test_fallback_no_tables(self):
        result = self.generator._fallback_generate(
            "any query",
            [],
            TableQueryType.LOOKUP,
        )
        assert "No tables available" in result.sql

    def test_sql_validation(self):
        """Ensure destructive SQL is rejected."""
        with pytest.raises(ValueError, match="forbidden"):
            self.generator._validate_sql("DROP TABLE users")

        with pytest.raises(ValueError, match="forbidden"):
            self.generator._validate_sql("DELETE FROM chunks")

        # Safe SQL should pass
        self.generator._validate_sql("SELECT * FROM tbl_001")

    def test_clean_sql_markdown(self):
        raw = "```sql\nSELECT * FROM tbl_001\n```"
        cleaned = self.generator._clean_sql(raw)
        assert cleaned == "SELECT * FROM tbl_001"
