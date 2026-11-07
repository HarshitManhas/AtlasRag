"""Table Extractor — detects and extracts tables from PDF documents.

Handles:
  - Table detection and boundary identification
  - Multi-row and hierarchical header resolution
  - Merged cell detection and normalization
  - Column type inference (numeric, text, date, currency, %)
  - Conversion to StructuredTable schema with full provenance
"""

from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from atlas_rag.schemas.table import (
    ColumnTypeDef,
    MergedCellRegion,
    StructuredTable,
    TableHeader,
    TableSource,
)

logger = logging.getLogger(__name__)


# ─── Column Type Inference ───────────────────────────────────────────────────

_CURRENCY_RE = re.compile(r"^[\$€£¥₹]\s*[\d,]+\.?\d*$")
_PERCENTAGE_RE = re.compile(r"^[\d.]+\s*%$")
_DATE_RE = re.compile(
    r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$"
    r"|^\w+ \d{1,2},?\s*\d{4}$"
    r"|^\d{1,2} \w+ \d{4}$"
)


def _infer_column_type(values: list[Any]) -> str:
    """Infer the semantic type of a column from its non-null values.

    Returns one of: numeric, text, date, boolean, currency, percentage.
    """
    non_null = [v for v in values if v is not None and str(v).strip() != ""]
    if not non_null:
        return "text"

    sample = [str(v).strip() for v in non_null[:50]]

    # Check percentage first (before numeric, since 45% is also numeric-ish)
    pct_matches = sum(1 for s in sample if _PERCENTAGE_RE.match(s))
    if pct_matches / len(sample) > 0.6:
        return "percentage"

    # Check currency
    cur_matches = sum(1 for s in sample if _CURRENCY_RE.match(s))
    if cur_matches / len(sample) > 0.6:
        return "currency"

    # Check boolean
    bool_vals = {"true", "false", "yes", "no", "y", "n", "1", "0"}
    bool_matches = sum(1 for s in sample if s.lower() in bool_vals)
    if bool_matches / len(sample) > 0.8:
        return "boolean"

    # Check date
    date_matches = sum(1 for s in sample if _DATE_RE.match(s))
    if date_matches / len(sample) > 0.6:
        return "date"

    # Check numeric (int or float)
    numeric_count = 0
    for s in sample:
        cleaned = s.replace(",", "").replace(" ", "")
        try:
            float(cleaned)
            numeric_count += 1
        except ValueError:
            pass
    if numeric_count / len(sample) > 0.6:
        return "numeric"

    return "text"


def _build_column_types(headers: list[str], rows: list[dict[str, Any]]) -> list[ColumnTypeDef]:
    """Build ColumnTypeDef for each column in the table."""
    result = []
    for header in headers:
        values = [row.get(header) for row in rows]
        non_null = [v for v in values if v is not None and str(v).strip() != ""]
        inferred = _infer_column_type(values)
        result.append(
            ColumnTypeDef(
                name=header,
                inferred_type=inferred,
                nullable=len(non_null) < len(values),
                sample_values=[str(v) for v in non_null[:5]],
            )
        )
    return result


# ─── Merged Cell Handling ────────────────────────────────────────────────────


def _detect_merged_cells(df: pd.DataFrame) -> list[MergedCellRegion]:
    """Detect cells that span multiple rows by looking for repeated values
    in a column followed by empty cells (common pattern in PDFs).

    This is a heuristic — real merged cell detection requires bounding-box
    analysis from the PDF parser.
    """
    merged: list[MergedCellRegion] = []
    for col_idx, _col in enumerate(df.columns):
        prev_val = None
        start_row = 0
        for row_idx in range(len(df)):
            cell = df.iloc[row_idx, col_idx]
            cell_str = str(cell).strip() if pd.notna(cell) else ""

            if cell_str == "" and prev_val is not None:
                # Potentially part of a merged region — continue
                continue
            elif cell_str == prev_val and prev_val != "":
                # Same value repeated — also likely merged
                continue
            else:
                # End of a potential merged region
                if prev_val is not None and row_idx - start_row > 1:
                    merged.append(
                        MergedCellRegion(
                            start_row=start_row,
                            end_row=row_idx - 1,
                            start_col=col_idx,
                            end_col=col_idx,
                            value=prev_val,
                        )
                    )
                prev_val = cell_str
                start_row = row_idx

        # Handle merged region at end of column
        if prev_val is not None and len(df) - start_row > 1:
            merged.append(
                MergedCellRegion(
                    start_row=start_row,
                    end_row=len(df) - 1,
                    start_col=col_idx,
                    end_col=col_idx,
                    value=prev_val,
                )
            )
    return merged


def _resolve_merged_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Fill merged cell regions by forward-filling empty cells.

    In PDFs, merged cells often appear as a value in the first row of the
    region, with subsequent rows left blank. Forward-fill restores the
    logical structure.
    """
    df = df.copy()
    for col in df.columns:
        # Forward fill only when the cell is truly empty (not just zero)
        mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if mask.any():
            df[col] = df[col].replace("", pd.NA)
            df[col] = df[col].ffill()
    return df


# ─── Hierarchical Header Detection ──────────────────────────────────────────


def _detect_hierarchical_headers(
    raw_headers: list[list[str]],
) -> tuple[list[str], list[TableHeader]]:
    """Resolve multi-row headers into flat column names and a hierarchy tree.

    Many financial/scientific tables have 2-3 rows of headers:
        Row 0: "Revenue" (spanning 3 cols)  | "Costs" (spanning 2 cols)
        Row 1: "Q1" | "Q2" | "Q3"          | "Fixed" | "Variable"

    This produces flat headers: ["Revenue - Q1", "Revenue - Q2", ...]
    and a hierarchy: [TableHeader(level=0, name="Revenue", span=3), ...]
    """
    if len(raw_headers) <= 1:
        flat = raw_headers[0] if raw_headers else []
        return flat, []

    hierarchy: list[TableHeader] = []
    # Track parent spans for each level
    for level, row in enumerate(raw_headers):
        current_parent = None
        span_count = 0
        for cell in row:
            cell_str = str(cell).strip() if pd.notna(cell) else ""
            if cell_str:
                if current_parent and span_count > 0:
                    hierarchy.append(
                        TableHeader(
                            level=level,
                            name=current_parent,
                            span=span_count,
                        )
                    )
                current_parent = cell_str
                span_count = 1
            else:
                span_count += 1
        if current_parent:
            hierarchy.append(TableHeader(level=level, name=current_parent, span=span_count))

    # Build flat headers by joining parent → child with " - "
    flat_headers: list[str] = []
    last_row = raw_headers[-1]
    parent_map: dict[int, str] = {}

    # Map each column index to its parent from earlier levels
    for level in range(len(raw_headers) - 1):
        for col_cursor, cell in enumerate(raw_headers[level]):
            cell_str = str(cell).strip() if pd.notna(cell) else ""
            if cell_str:
                parent_map[col_cursor] = cell_str

    for col_idx, cell in enumerate(last_row):
        cell_str = str(cell).strip() if pd.notna(cell) else f"col_{col_idx}"
        parent = parent_map.get(col_idx)
        if parent and parent != cell_str:
            flat_headers.append(f"{parent} - {cell_str}")
        else:
            flat_headers.append(cell_str)

    return flat_headers, hierarchy


# ─── Numeric Cleaning ───────────────────────────────────────────────────────


def _clean_numeric_value(value: Any) -> Any:
    """Try to parse a cell value as a number, stripping currency symbols and
    commas. Returns the original value if parsing fails."""
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return value

    cleaned = value.strip()
    if not cleaned:
        return None

    # Strip currency symbols and percentage signs
    cleaned = re.sub(r"^[\$€£¥₹]\s*", "", cleaned)
    cleaned = cleaned.replace(",", "")
    pct = cleaned.endswith("%")
    if pct:
        cleaned = cleaned[:-1].strip()

    try:
        num = float(cleaned)
        if pct:
            return num  # Keep as raw number; type metadata says it's a percentage
        return int(num) if num == int(num) else num
    except ValueError:
        return value


def _clean_row_values(row: dict[str, Any], column_types: list[ColumnTypeDef]) -> dict[str, Any]:
    """Clean row values according to inferred column types."""
    type_map = {ct.name: ct.inferred_type for ct in column_types}
    cleaned = {}
    for col, val in row.items():
        col_type = type_map.get(col, "text")
        if col_type in ("numeric", "currency", "percentage"):
            cleaned[col] = _clean_numeric_value(val)
        else:
            cleaned[col] = val
    return cleaned


# ─── Text Representation ────────────────────────────────────────────────────


def _table_to_text(table: StructuredTable) -> str:
    """Generate a natural-language text representation of a table for embedding.

    This is more retrieval-friendly than raw markdown because it includes
    contextual descriptions of the data.
    """
    parts: list[str] = []

    if table.title:
        parts.append(f"Table: {table.title}")

    if table.source.page:
        parts.append(f"(Page {table.source.page} of {table.source.file})")

    parts.append(f"Columns: {', '.join(table.headers)}")
    parts.append(f"Rows: {table.row_count}")

    # Include column type info
    for ct in table.column_types:
        if ct.inferred_type != "text":
            parts.append(f"  - {ct.name}: {ct.inferred_type}")

    # Include a few rows as natural language
    max_sample = min(5, len(table.rows))
    if max_sample > 0:
        parts.append("Sample data:")
        for i in range(max_sample):
            row = table.rows[i]
            row_parts = [f"{h}: {row.get(h, 'N/A')}" for h in table.headers]
            parts.append(f"  Row {i + 1}: {', '.join(row_parts)}")

    # Add summary stats for numeric columns
    for ct in table.column_types:
        if ct.inferred_type in ("numeric", "currency", "percentage"):
            values = [
                table.rows[i].get(ct.name)
                for i in range(len(table.rows))
                if isinstance(table.rows[i].get(ct.name), (int, float))
            ]
            if values:
                parts.append(
                    f"  {ct.name}: min={min(values)}, max={max(values)}, "
                    f"sum={sum(values):.2f}, avg={sum(values) / len(values):.2f}"
                )

    return "\n".join(parts)


# ─── Main Extraction API ────────────────────────────────────────────────────


class TableExtractor:
    """Extracts structured tables from PDF documents.

    Uses pdfplumber for table detection and extraction, with fallback to
    camelot for complex tables. Handles merged cells, hierarchical headers,
    and produces fully normalized StructuredTable objects.
    """

    def __init__(self) -> None:
        self._table_counter = 0

    def _next_table_id(self) -> str:
        self._table_counter += 1
        return f"tbl_{self._table_counter:03d}"

    def extract_tables_from_pdf(
        self,
        pdf_path: str | Path,
        filename: str,
        document_id: str | None = None,
    ) -> list[StructuredTable]:
        """Extract all tables from a PDF file.

        Args:
            pdf_path: Path to the PDF file on disk.
            filename: Original filename for provenance tracking.
            document_id: AtlasRAG document UUID.

        Returns:
            List of StructuredTable objects, one per detected table.
        """
        import pdfplumber

        tables: list[StructuredTable] = []
        pdf_path = Path(pdf_path)

        logger.info("Extracting tables from %s", filename)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_tables = page.extract_tables()
                    if not page_tables:
                        continue

                    for tbl_idx, raw_table in enumerate(page_tables):
                        if not raw_table or len(raw_table) < 2:
                            # Need at least a header row and one data row
                            continue

                        try:
                            structured = self._process_raw_table(
                                raw_table=raw_table,
                                page_number=page_num,
                                table_index=tbl_idx,
                                filename=filename,
                                document_id=document_id,
                            )
                            tables.append(structured)
                            logger.info(
                                "Extracted table %s: %d rows x %d cols from page %d",
                                structured.table_id,
                                structured.row_count,
                                structured.col_count,
                                page_num,
                            )
                        except Exception:
                            logger.exception(
                                "Failed to process table %d on page %d of %s",
                                tbl_idx,
                                page_num,
                                filename,
                            )

        except Exception:
            logger.exception("Failed to open PDF %s for table extraction", pdf_path)

        # Try camelot as fallback if pdfplumber found nothing
        if not tables:
            tables = self._extract_with_camelot(pdf_path, filename, document_id)

        logger.info("Extracted %d tables from %s", len(tables), filename)
        return tables

    def extract_tables_from_bytes(
        self,
        pdf_bytes: bytes,
        filename: str,
        document_id: str | None = None,
    ) -> list[StructuredTable]:
        """Extract tables from in-memory PDF bytes."""
        import pdfplumber

        tables: list[StructuredTable] = []

        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_tables = page.extract_tables()
                    if not page_tables:
                        continue

                    for tbl_idx, raw_table in enumerate(page_tables):
                        if not raw_table or len(raw_table) < 2:
                            continue

                        try:
                            structured = self._process_raw_table(
                                raw_table=raw_table,
                                page_number=page_num,
                                table_index=tbl_idx,
                                filename=filename,
                                document_id=document_id,
                            )
                            tables.append(structured)
                        except Exception:
                            logger.exception(
                                "Failed to process table %d on page %d",
                                tbl_idx,
                                page_num,
                            )
        except Exception:
            logger.exception("Failed to extract tables from bytes for %s", filename)

        return tables

    def _process_raw_table(
        self,
        raw_table: list[list[str | None]],
        page_number: int,
        table_index: int,
        filename: str,
        document_id: str | None,
    ) -> StructuredTable:
        """Convert a raw 2D table (from pdfplumber) into a StructuredTable.

        Steps:
          1. Detect multi-row headers
          2. Build flat header names (joining hierarchical levels)
          3. Build row dicts
          4. Detect and resolve merged cells
          5. Infer column types
          6. Clean numeric values
          7. Generate text representation
        """
        # Step 1-2: Header detection
        # Heuristic: if the first N rows contain mostly text and no numeric
        # values, treat them as header rows.
        header_row_count = self._detect_header_rows(raw_table)
        raw_headers = raw_table[:header_row_count]
        data_rows = raw_table[header_row_count:]

        headers, hierarchical = _detect_hierarchical_headers(raw_headers)

        # Ensure unique header names
        headers = self._deduplicate_headers(headers)

        # Step 3: Build row dicts
        rows: list[dict[str, Any]] = []
        for raw_row in data_rows:
            row_dict: dict[str, Any] = {}
            for col_idx, header in enumerate(headers):
                if col_idx < len(raw_row):
                    value = raw_row[col_idx]
                    row_dict[header] = value if value is not None else ""
                else:
                    row_dict[header] = ""
            rows.append(row_dict)

        # Step 4: Detect merged cells
        df = pd.DataFrame(rows)
        merged_cells = _detect_merged_cells(df)
        if merged_cells:
            df = _resolve_merged_cells(df)
            rows = df.to_dict("records")

        # Step 5: Infer column types
        column_types = _build_column_types(headers, rows)

        # Step 6: Clean numeric values
        rows = [_clean_row_values(row, column_types) for row in rows]

        # Build StructuredTable
        source = TableSource(
            page=page_number,
            file=filename,
            document_id=document_id,
            table_index=table_index,
        )

        table = StructuredTable(
            table_id=self._next_table_id(),
            title=self._infer_table_title(raw_table, page_number),
            headers=headers,
            hierarchical_headers=hierarchical if hierarchical else None,
            column_types=column_types,
            rows=rows,
            row_count=len(rows),
            col_count=len(headers),
            source=source,
            merged_cells=merged_cells,
            text_representation=None,  # Set below
            summary=None,
        )

        # Step 7: Generate text representation
        table.text_representation = _table_to_text(table)

        return table

    def _detect_header_rows(self, raw_table: list[list[str | None]]) -> int:
        """Heuristic to detect how many leading rows are headers vs data.

        Rules:
          - At least 1 header row
          - If row has significantly fewer numeric values than later rows, it's a header
          - Max 3 header rows
        """
        if len(raw_table) < 2:
            return 1

        max_header_rows = min(3, len(raw_table) - 1)

        def numeric_ratio(row: list[str | None]) -> float:
            if not row:
                return 0.0
            numeric = 0
            for cell in row:
                if cell is None:
                    continue
                cleaned = str(cell).strip().replace(",", "").replace("$", "")
                try:
                    float(cleaned)
                    numeric += 1
                except ValueError:
                    pass
            return numeric / len(row)

        # Data rows typically have more numeric values
        # Look at the last few rows to establish a "data" baseline
        data_sample = raw_table[-min(5, len(raw_table)) :]
        avg_data_numeric = sum(numeric_ratio(r) for r in data_sample) / len(data_sample)

        header_count = 1  # Always at least 1
        for i in range(1, max_header_rows):
            row_numeric = numeric_ratio(raw_table[i])
            if row_numeric < avg_data_numeric * 0.3:
                header_count = i + 1
            else:
                break

        return header_count

    @staticmethod
    def _deduplicate_headers(headers: list[str]) -> list[str]:
        """Ensure all header names are unique by appending suffixes."""
        seen: dict[str, int] = {}
        unique: list[str] = []
        for h in headers:
            h = h.strip() if h else "unnamed"
            if not h:
                h = "unnamed"
            if h in seen:
                seen[h] += 1
                unique.append(f"{h}_{seen[h]}")
            else:
                seen[h] = 0
                unique.append(h)
        return unique

    @staticmethod
    def _infer_table_title(raw_table: list[list[str | None]], page_number: int) -> str | None:
        """Try to infer a table title from the first row if it looks like a
        single merged header cell (common in financial documents)."""
        if not raw_table:
            return None

        first_row = raw_table[0]
        non_empty = [c for c in first_row if c and str(c).strip()]

        # If only one non-empty cell in the first row and it's long-ish,
        # it's likely a title spanning the entire table width
        if len(non_empty) == 1 and len(str(non_empty[0])) > 10:
            return str(non_empty[0]).strip()

        return None

    def _extract_with_camelot(
        self,
        pdf_path: Path,
        filename: str,
        document_id: str | None,
    ) -> list[StructuredTable]:
        """Fallback extraction using Camelot for complex table layouts."""
        tables: list[StructuredTable] = []
        try:
            import camelot

            camelot_tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="lattice")
            if not camelot_tables or len(camelot_tables) == 0:
                # Try stream flavor as second fallback
                camelot_tables = camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")

            for tbl in camelot_tables:
                if tbl.df.empty or len(tbl.df) < 2:
                    continue

                df = tbl.df
                # First row as headers
                headers = [str(c).strip() for c in df.iloc[0].tolist()]
                headers = self._deduplicate_headers(headers)

                rows = []
                for _, row in df.iloc[1:].iterrows():
                    row_dict = {}
                    for col_idx, header in enumerate(headers):
                        row_dict[header] = row.iloc[col_idx] if col_idx < len(row) else ""
                    rows.append(row_dict)

                column_types = _build_column_types(headers, rows)
                rows = [_clean_row_values(r, column_types) for r in rows]

                page_num = tbl.page if hasattr(tbl, "page") else 1
                source = TableSource(
                    page=page_num,
                    file=filename,
                    document_id=document_id,
                    table_index=len(tables),
                )

                structured = StructuredTable(
                    table_id=self._next_table_id(),
                    title=None,
                    headers=headers,
                    column_types=column_types,
                    rows=rows,
                    row_count=len(rows),
                    col_count=len(headers),
                    source=source,
                )
                structured.text_representation = _table_to_text(structured)
                tables.append(structured)

        except ImportError:
            logger.warning("Camelot not available for fallback table extraction")
        except Exception:
            logger.exception("Camelot fallback extraction failed for %s", filename)

        return tables

    def reset_counter(self) -> None:
        """Reset the table ID counter (useful between documents)."""
        self._table_counter = 0
