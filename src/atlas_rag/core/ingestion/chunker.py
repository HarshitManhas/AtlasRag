"""Text Chunker — splits parsed documents into retrieval-optimized chunks.

Implements multiple chunking strategies:
  - Recursive character splitting (default)
  - Sentence-window chunking
  - Semantic chunking (by section headers)

Each chunk carries full provenance: source document, page number, section header,
and position index for citation tracking.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from atlas_rag.core.ingestion.parser import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A text segment ready for embedding and indexing."""

    content: str
    chunk_index: int
    page_number: int | None = None
    section_header: str | None = None
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Rough token count estimate (words x 1.3)
        if self.token_count == 0:
            self.token_count = int(len(self.content.split()) * 1.3)


class TextChunker:
    """Splits documents into overlapping chunks for retrieval.

    Supports three strategies selected via `strategy` parameter:
      - "recursive" (default): Splits by paragraph → sentence → character boundaries
      - "sentence_window": Each chunk is a window of N sentences with overlap
      - "semantic": Splits on section boundaries (headers), then sub-splits large sections
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        strategy: str = "recursive",
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        """Split a parsed document into chunks.

        Args:
            doc: A ParsedDocument from the parser.

        Returns:
            List of Chunk objects with provenance metadata.
        """
        if self.strategy == "sentence_window":
            chunks = self._sentence_window_chunk(doc)
        elif self.strategy == "semantic":
            chunks = self._semantic_chunk(doc)
        else:
            chunks = self._recursive_chunk(doc)

        logger.info(
            "Chunked %s into %d chunks (strategy=%s, size=%d, overlap=%d)",
            doc.filename,
            len(chunks),
            self.strategy,
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    # ── Recursive Character Splitting ────────────────────────────────────────

    def _recursive_chunk(self, doc: ParsedDocument) -> list[Chunk]:
        """Split text recursively using paragraph → sentence → word boundaries."""
        chunks: list[Chunk] = []
        chunk_index = 0

        for page in doc.pages:
            if not page.text.strip():
                continue

            current_header = page.section_headers[-1] if page.section_headers else None
            page_chunks = self._split_text_recursive(page.text)

            for text in page_chunks:
                chunks.append(
                    Chunk(
                        content=text,
                        chunk_index=chunk_index,
                        page_number=page.page_number,
                        section_header=current_header,
                        metadata={"source": doc.filename},
                    )
                )
                chunk_index += 1

        return chunks

    def _split_text_recursive(self, text: str) -> list[str]:
        """Recursively split text, trying paragraph breaks first, then sentences."""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        # Try splitting by decreasing separator specificity
        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]

        for sep in separators:
            parts = text.split(sep)
            if len(parts) > 1:
                return self._merge_splits(parts, sep)

        # Last resort: hard split at chunk_size
        return self._hard_split(text)

    def _merge_splits(self, parts: list[str], separator: str) -> list[str]:
        """Merge small splits into chunk_size-d pieces with overlap."""
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for part in parts:
            part_len = len(part)
            if current_len + part_len + len(separator) > self.chunk_size and current:
                chunk_text = separator.join(current).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # Keep overlap from the end of the current chunk
                separator.join(current).strip()
                overlap_parts: list[str] = []
                overlap_len = 0
                for p in reversed(current):
                    if overlap_len + len(p) > self.chunk_overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_len += len(p) + len(separator)

                current = [*overlap_parts, part]
                current_len = sum(len(p) for p in current) + len(separator) * (len(current) - 1)
            else:
                current.append(part)
                current_len += part_len + len(separator)

        if current:
            chunk_text = separator.join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """Hard-split text at chunk_size boundaries (last resort)."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.chunk_overlap
        return chunks

    # ── Sentence Window Chunking ─────────────────────────────────────────────

    def _sentence_window_chunk(self, doc: ParsedDocument) -> list[Chunk]:
        """Chunk by sliding window over sentences.

        Each chunk is a window of sentences, with overlap of N sentences
        between consecutive chunks. Good for preserving sentence coherence.
        """
        chunks: list[Chunk] = []
        chunk_index = 0
        window_size = max(3, self.chunk_size // 100)  # ~3-5 sentences per chunk
        overlap_sentences = max(1, window_size // 3)

        for page in doc.pages:
            if not page.text.strip():
                continue

            sentences = self._split_into_sentences(page.text)
            current_header = page.section_headers[-1] if page.section_headers else None

            i = 0
            while i < len(sentences):
                window = sentences[i : i + window_size]
                text = " ".join(window).strip()

                if text:
                    chunks.append(
                        Chunk(
                            content=text,
                            chunk_index=chunk_index,
                            page_number=page.page_number,
                            section_header=current_header,
                            metadata={"source": doc.filename, "strategy": "sentence_window"},
                        )
                    )
                    chunk_index += 1

                i += window_size - overlap_sentences

        return chunks

    # ── Semantic Chunking ────────────────────────────────────────────────────

    def _semantic_chunk(self, doc: ParsedDocument) -> list[Chunk]:
        """Chunk by section boundaries (headers), then sub-split large sections.

        Preserves document structure by keeping sections together when possible.
        """
        chunks: list[Chunk] = []
        chunk_index = 0

        for page in doc.pages:
            if not page.text.strip():
                continue

            sections = self._split_by_headers(page.text)

            for header, section_text in sections:
                if len(section_text) <= self.chunk_size:
                    if section_text.strip():
                        chunks.append(
                            Chunk(
                                content=section_text.strip(),
                                chunk_index=chunk_index,
                                page_number=page.page_number,
                                section_header=header,
                                metadata={"source": doc.filename, "strategy": "semantic"},
                            )
                        )
                        chunk_index += 1
                else:
                    # Sub-split large sections
                    sub_chunks = self._split_text_recursive(section_text)
                    for text in sub_chunks:
                        chunks.append(
                            Chunk(
                                content=text,
                                chunk_index=chunk_index,
                                page_number=page.page_number,
                                section_header=header,
                                metadata={"source": doc.filename, "strategy": "semantic"},
                            )
                        )
                        chunk_index += 1

        return chunks

    # ── Utilities ────────────────────────────────────────────────────────────

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Split text into sentences using regex-based heuristics."""
        # Handle common abbreviations to avoid false splits
        text = re.sub(r"(Mr|Mrs|Dr|Prof|Inc|Ltd|Jr|Sr|vs)\.", r"\1<DOT>", text)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]

    @staticmethod
    def _split_by_headers(text: str) -> list[tuple[str | None, str]]:
        """Split text at Markdown-style headers or detected heading patterns."""
        # Match lines that look like headers (Markdown # or ALL CAPS short lines)
        header_pattern = re.compile(r"^(#{1,6}\s+.+|[A-Z][A-Z\s]{5,60})$", re.MULTILINE)

        sections: list[tuple[str | None, str]] = []
        last_end = 0
        current_header: str | None = None

        for match in header_pattern.finditer(text):
            # Save the section before this header
            section_text = text[last_end : match.start()]
            if section_text.strip():
                sections.append((current_header, section_text))

            current_header = match.group().strip().lstrip("#").strip()
            last_end = match.end()

        # Remaining text after the last header
        remaining = text[last_end:]
        if remaining.strip():
            sections.append((current_header, remaining))

        if not sections:
            sections.append((None, text))

        return sections
