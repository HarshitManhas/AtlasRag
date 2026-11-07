"""Generation package."""

from atlas_rag.core.generation.citation import Citation, CitationExtractor
from atlas_rag.core.generation.llm import LLMClient, LLMResponse
from atlas_rag.core.generation.pipeline import GenerationPipeline, GenerationResult
from atlas_rag.core.generation.prompts import RAG_SYSTEM_PROMPT, build_rag_prompt

__all__ = [
    "RAG_SYSTEM_PROMPT",
    "Citation",
    "CitationExtractor",
    "GenerationPipeline",
    "GenerationResult",
    "LLMClient",
    "LLMResponse",
    "build_rag_prompt",
]
