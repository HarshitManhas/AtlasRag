"""Generation Pipeline — orchestrates the full RAG generation workflow.

Ties together retrieval context, prompt construction, LLM generation,
citation extraction, and hallucination detection into a single pipeline.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from atlas_rag.config import Settings, get_settings
from atlas_rag.core.generation.citation import Citation, CitationExtractor
from atlas_rag.core.generation.llm import LLMClient, LLMResponse
from atlas_rag.core.generation.prompts import build_rag_prompt
from atlas_rag.core.retrieval.pipeline import RetrievalContext

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Complete output from the generation pipeline."""

    answer: str
    citations: list[Citation]
    references: str
    llm_response: LLMResponse | None = None
    groundedness_score: float | None = None
    table_analysis: Any | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class GenerationPipeline:
    """Orchestrates RAG answer generation with citations.

    Pipeline flow:
      1. Build prompt from retrieval context
      2. Call LLM for generation
      3. Extract and resolve citations
      4. (Optional) Run hallucination detection
      5. Return annotated answer
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = LLMClient(self.settings)
        self.citation_extractor = CitationExtractor()

    async def generate(
        self,
        retrieval_context: RetrievalContext,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> GenerationResult:
        """Generate an answer from retrieval context.

        Args:
            retrieval_context: Output from the retrieval pipeline.
            conversation_history: Optional conversation history.

        Returns:
            GenerationResult with answer, citations, and metadata.
        """
        start = time.perf_counter()

        # Build prompt
        messages = build_rag_prompt(
            query=retrieval_context.query,
            results=retrieval_context.results,
            table_analysis=retrieval_context.table_analysis,
            conversation_history=conversation_history,
        )

        # Call LLM
        llm_response = await self.llm.complete(messages)

        # Extract and resolve citations
        answer_text, citations = self.citation_extractor.extract_citations(
            generated_text=llm_response.content,
            retrieval_results=retrieval_context.results,
            table_citations=(
                retrieval_context.table_analysis.cell_citations
                if retrieval_context.table_analysis
                else None
            ),
        )

        references = self.citation_extractor.format_references(citations)

        elapsed = (time.perf_counter() - start) * 1000

        return GenerationResult(
            answer=answer_text,
            citations=citations,
            references=references,
            llm_response=llm_response,
            table_analysis=retrieval_context.table_analysis,
            latency_ms=round(elapsed, 2),
            metadata={
                "retrieval_latency_ms": retrieval_context.latency_ms,
                "generation_latency_ms": llm_response.latency_ms,
                "tokens_in": llm_response.tokens_in,
                "tokens_out": llm_response.tokens_out,
                "model": llm_response.model,
                "context_chunks": len(retrieval_context.results),
            },
        )

    async def generate_stream(
        self,
        retrieval_context: RetrievalContext,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream an answer from retrieval context.

        Yields text chunks as they are generated. Citations are resolved
        post-stream by the caller.
        """
        messages = build_rag_prompt(
            query=retrieval_context.query,
            results=retrieval_context.results,
            table_analysis=retrieval_context.table_analysis,
            conversation_history=conversation_history,
        )

        async for chunk in self.llm.stream(messages):
            yield chunk
