"""Document Parser — extracts text, metadata, and structure from PDF/DOCX/HTML files.

Supports:
  - PDF: Full text extraction with page boundaries and section headers via PyMuPDF
  - DOCX: Paragraph-level extraction with heading detection via python-docx
  - HTML/Markdown: Tag-aware extraction via BeautifulSoup
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    """A single page of extracted content."""

    page_number: int
    text: str
    section_headers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Complete parsed output from a document."""

    filename: str
    mime_type: str
    pages: list[ParsedPage]
    total_pages: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


class DocumentParser:
    """Unified document parser supporting PDF, DOCX, and HTML formats."""

    _MIME_HANDLERS: ClassVar[dict[str, str]] = {
        "application/pdf": "_parse_pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "_parse_docx",
        "text/html": "_parse_html",
        "text/markdown": "_parse_markdown",
        "text/plain": "_parse_plaintext",
    }

    def parse(self, content: bytes, filename: str, mime_type: str) -> ParsedDocument:
        """Parse a document from raw bytes.

        Args:
            content: Raw file bytes.
            filename: Original filename.
            mime_type: MIME type of the file.

        Returns:
            ParsedDocument with extracted pages and metadata.

        Raises:
            ValueError: If the MIME type is not supported.
        """
        handler_name = self._MIME_HANDLERS.get(mime_type)
        if not handler_name:
            raise ValueError(
                f"Unsupported MIME type: {mime_type}. Supported: {list(self._MIME_HANDLERS.keys())}"
            )

        handler = getattr(self, handler_name)
        logger.info("Parsing %s (%s)", filename, mime_type)
        return handler(content, filename)

    # ── PDF Parsing ──────────────────────────────────────────────────────────

    def _parse_pdf(self, content: bytes, filename: str) -> ParsedDocument:
        """Extract text from PDF using PyMuPDF (fitz) with page boundaries."""
        import fitz  # PyMuPDF

        pages: list[ParsedPage] = []
        doc_metadata: dict[str, Any] = {}

        doc = fitz.open(stream=content, filetype="pdf")
        try:
            # Extract document-level metadata
            meta = doc.metadata
            if meta:
                doc_metadata = {k: v for k, v in meta.items() if v}

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")

                # Detect section headers by font size analysis
                headers = self._extract_pdf_headers(page)

                pages.append(
                    ParsedPage(
                        page_number=page_num + 1,
                        text=text.strip(),
                        section_headers=headers,
                        metadata={"width": page.rect.width, "height": page.rect.height},
                    )
                )
        finally:
            doc.close()

        return ParsedDocument(
            filename=filename,
            mime_type="application/pdf",
            pages=pages,
            total_pages=len(pages),
            metadata=doc_metadata,
        )

    @staticmethod
    def _extract_pdf_headers(page: Any) -> list[str]:
        """Detect section headers by analyzing font sizes in the page.

        Text blocks with font size significantly larger than the body text
        are classified as headers.
        """
        blocks = page.get_text("dict", flags=11)["blocks"]
        font_sizes: list[float] = []
        text_spans: list[tuple[float, str]] = []

        for block in blocks:
            if block.get("type") != 0:  # text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 0)
                    text = span.get("text", "").strip()
                    if text and size > 0:
                        font_sizes.append(size)
                        text_spans.append((size, text))

        if not font_sizes:
            return []

        # Determine body text size (most common font size)
        from collections import Counter

        size_counts = Counter(round(s, 1) for s in font_sizes)
        body_size = size_counts.most_common(1)[0][0] if size_counts else 12.0

        # Headers are text blocks with font size > 1.2x body size
        headers = []
        for size, text in text_spans:
            if size > body_size * 1.2 and len(text) > 3:
                headers.append(text)

        return headers

    # ── DOCX Parsing ─────────────────────────────────────────────────────────

    def _parse_docx(self, content: bytes, filename: str) -> ParsedDocument:
        """Extract text from DOCX with heading/paragraph structure."""
        from docx import Document as DocxDocument

        doc = DocxDocument(BytesIO(content))
        current_text: list[str] = []
        current_headers: list[str] = []
        pages: list[ParsedPage] = []
        page_num = 1

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            # Detect headings
            if para.style and para.style.name.startswith("Heading"):
                current_headers.append(text)

            current_text.append(text)

            # DOCX doesn't have real page breaks in paragraphs — we simulate
            # pages every ~3000 characters for consistency with the page model
            if len("\n".join(current_text)) > 3000:
                pages.append(
                    ParsedPage(
                        page_number=page_num,
                        text="\n".join(current_text),
                        section_headers=current_headers.copy(),
                    )
                )
                current_text = []
                current_headers = []
                page_num += 1

        # Flush remaining content
        if current_text:
            pages.append(
                ParsedPage(
                    page_number=page_num,
                    text="\n".join(current_text),
                    section_headers=current_headers,
                )
            )

        metadata: dict[str, Any] = {}
        core = doc.core_properties
        if core.title:
            metadata["title"] = core.title
        if core.author:
            metadata["author"] = core.author

        return ParsedDocument(
            filename=filename,
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            pages=pages,
            total_pages=len(pages),
            metadata=metadata,
        )

    # ── HTML Parsing ─────────────────────────────────────────────────────────

    def _parse_html(self, content: bytes, filename: str) -> ParsedDocument:
        """Extract text from HTML, preserving heading structure."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(content, "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        headers = [tag.get_text(strip=True) for tag in soup.find_all(re.compile(r"^h[1-6]$"))]

        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        pages = [ParsedPage(page_number=1, text=text, section_headers=headers)]

        return ParsedDocument(
            filename=filename,
            mime_type="text/html",
            pages=pages,
            total_pages=1,
            metadata={"title": title} if title else {},
        )

    # ── Markdown Parsing ─────────────────────────────────────────────────────

    def _parse_markdown(self, content: bytes, filename: str) -> ParsedDocument:
        """Extract text from Markdown, detecting headers."""
        text = content.decode("utf-8", errors="replace")
        headers = re.findall(r"^#{1,6}\s+(.+)$", text, re.MULTILINE)

        pages = [ParsedPage(page_number=1, text=text, section_headers=headers)]

        return ParsedDocument(
            filename=filename,
            mime_type="text/markdown",
            pages=pages,
            total_pages=1,
        )

    # ── Plaintext ────────────────────────────────────────────────────────────

    def _parse_plaintext(self, content: bytes, filename: str) -> ParsedDocument:
        """Parse plain text files."""
        text = content.decode("utf-8", errors="replace")
        pages = [ParsedPage(page_number=1, text=text)]
        return ParsedDocument(
            filename=filename,
            mime_type="text/plain",
            pages=pages,
            total_pages=1,
        )
