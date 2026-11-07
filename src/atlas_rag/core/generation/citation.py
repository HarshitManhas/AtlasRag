"""Citation Extractor — maps inline citations in generated text to source references.

Parses [N] style citations from LLM output and maps them back to the
retrieval results and table cells that provided the evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from atlas_rag.core.retrieval.dense import RetrievalResult
from atlas_rag.schemas.table import CellReference


@dataclass
class Citation:
    """A resolved citation linking a generated claim to its source."""

    citation_id: int
    text_span: str  # The text around the citation marker
    source_type: str  # "chunk" or "table_cell"
    document_id: str | None = None
    page_number: int | None = None
    section_header: str | None = None
    chunk_content: str | None = None
    cell_reference: CellReference | None = None


class CitationExtractor:
    """Extracts and resolves inline citations from generated text."""

    _CITATION_PATTERN = re.compile(r"\[(\d+)\]")

    def extract_citations(
        self,
        generated_text: str,
        retrieval_results: list[RetrievalResult],
        table_citations: list[CellReference] | None = None,
    ) -> tuple[str, list[Citation]]:
        """Extract citations from generated text and resolve them to sources.

        Args:
            generated_text: The LLM-generated answer with [N] markers.
            retrieval_results: The retrieval results provided as context.
            table_citations: Optional cell-level citations from table analysis.

        Returns:
            Tuple of (cleaned_text, list_of_resolved_citations).
        """
        citations: list[Citation] = []
        seen_ids: set[int] = set()

        for match in self._CITATION_PATTERN.finditer(generated_text):
            citation_id = int(match.group(1))
            if citation_id in seen_ids:
                continue
            seen_ids.add(citation_id)

            # Get surrounding text for context
            start = max(0, match.start() - 100)
            end = min(len(generated_text), match.end() + 50)
            text_span = generated_text[start:end].strip()

            # Resolve citation to source
            if 1 <= citation_id <= len(retrieval_results):
                result = retrieval_results[citation_id - 1]
                citations.append(
                    Citation(
                        citation_id=citation_id,
                        text_span=text_span,
                        source_type="chunk",
                        document_id=result.document_id,
                        page_number=result.page_number,
                        section_header=result.section_header,
                        chunk_content=result.content[:200],
                    )
                )

        # Also include table cell citations
        if table_citations:
            for cell_ref in table_citations:
                citations.append(
                    Citation(
                        citation_id=len(citations) + 1,
                        text_span="",
                        source_type="table_cell",
                        page_number=cell_ref.page,
                        cell_reference=cell_ref,
                    )
                )

        return generated_text, citations

    def format_references(self, citations: list[Citation]) -> str:
        """Format citations as a reference list appended to the answer."""
        if not citations:
            return ""

        lines = ["\n\n---\n**References:**"]

        for cite in citations:
            if cite.source_type == "chunk":
                parts = [f"[{cite.citation_id}]"]
                if cite.section_header:
                    parts.append(f'Section: "{cite.section_header}"')
                if cite.page_number:
                    parts.append(f"Page {cite.page_number}")
                if cite.document_id:
                    parts.append(f"Document: {cite.document_id}")
                lines.append("  " + ", ".join(parts))

            elif cite.source_type == "table_cell" and cite.cell_reference:
                ref = cite.cell_reference
                lines.append(
                    f"  [Table {ref.table_id}] "
                    f"Row {ref.row_index + 1}, "
                    f'Column "{ref.column}" = {ref.value}'
                    f"{f', Page {ref.page}' if ref.page else ''}"
                )

        return "\n".join(lines)
