"""Prompt Templates — constructs LLM prompts for RAG generation.

Manages system instructions, context injection, and citation formatting.
All prompts instruct the LLM to produce inline citations referencing
source chunks and table cells.
"""

from __future__ import annotations

from typing import Any

from atlas_rag.core.retrieval.dense import RetrievalResult

# ─── System Prompts ──────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are AtlasRAG, an enterprise document intelligence assistant.
You answer questions based ONLY on the provided context from retrieved documents.

Rules:
1. Answer using ONLY the information in the provided context. Never use prior knowledge.
2. For every claim you make, add an inline citation in the format [N] where N is the source number.
3. If the context doesn't contain enough information, say "I don't have enough information in the provided documents to answer this question."
4. When citing table data, mention the table name and specific values.
5. Be precise and concise. Prefer quantitative answers when data is available.
6. If multiple sources agree, cite all of them.
7. Never fabricate data, numbers, or citations."""

TABLE_ANALYSIS_PROMPT = """You are AtlasRAG's table analysis assistant.
You answer analytical questions using data from extracted tables.

You have access to:
- SQL query results from the document's tables
- Cell-level references for traceability

Rules:
1. Present numbers precisely as they appear in the data.
2. Cite specific table cells using the format [Table X, Row Y, Column Z].
3. For aggregations, explain the calculation and cite all contributing cells.
4. If the data is insufficient, say so clearly.
5. Format numbers with appropriate precision and units."""


# ─── Context Building ───────────────────────────────────────────────────────


def build_rag_prompt(
    query: str,
    results: list[RetrievalResult],
    table_analysis: Any | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build the complete message list for RAG generation.

    Args:
        query: User's question.
        results: Retrieved and reranked context chunks.
        table_analysis: Optional TableAnalysisResult for table queries.
        conversation_history: Optional prior conversation turns.

    Returns:
        List of chat messages ready for the LLM.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
    ]

    # Add conversation history (last 5 turns max)
    if conversation_history:
        for turn in conversation_history[-10:]:  # last 5 Q&A pairs
            messages.append(turn)

    # Build context block
    context_parts: list[str] = []

    # Text context from retrieval
    for i, result in enumerate(results, 1):
        source_info = []
        if result.page_number:
            source_info.append(f"Page {result.page_number}")
        if result.section_header:
            source_info.append(f"Section: {result.section_header}")
        source_info.append(f"Document: {result.document_id}")

        header = f"[Source {i}] ({', '.join(source_info)})"
        context_parts.append(f"{header}\n{result.content}")

    # Table analysis context
    if table_analysis and hasattr(table_analysis, "natural_language_answer"):
        if table_analysis.natural_language_answer:
            context_parts.append(
                f"\n[Table Analysis Result]\n{table_analysis.natural_language_answer}"
            )

        # Include SQL result details
        if table_analysis.sql_result and table_analysis.sql_result.rows:
            rows_str = "\n".join(str(row) for row in table_analysis.sql_result.rows[:20])
            context_parts.append(f"\n[Raw Table Data]\n{rows_str}")

        # Include cell citations
        if table_analysis.cell_citations:
            citations_str = "\n".join(
                f"  - {c.table_id}, Row {c.row_index + 1}, "
                f"Column '{c.column}' = {c.value} (Page {c.page})"
                for c in table_analysis.cell_citations[:20]
            )
            context_parts.append(f"\n[Table Cell References]\n{citations_str}")

    context = "\n\n---\n\n".join(context_parts)

    # Build user message with context
    user_message = f"""Context from retrieved documents:

{context}

---

Question: {query}

Please answer based on the context above with inline citations [N]."""

    messages.append({"role": "user", "content": user_message})

    return messages


def build_table_prompt(
    query: str,
    table_analysis: Any,
) -> list[dict[str, str]]:
    """Build a prompt specifically for table-focused answers."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": TABLE_ANALYSIS_PROMPT},
    ]

    context_parts = []

    if table_analysis.sql_result:
        context_parts.append(
            f"SQL Query: {table_analysis.sql_result.query.sql}\n"
            f"Explanation: {table_analysis.sql_result.query.explanation}"
        )
        if table_analysis.sql_result.rows:
            for row in table_analysis.sql_result.rows[:30]:
                context_parts.append(str(row))

    if table_analysis.cell_citations:
        context_parts.append("\nCell-level sources:")
        for c in table_analysis.cell_citations[:30]:
            context_parts.append(
                f"  [{c.table_id}] Row {c.row_index + 1}, "
                f"'{c.column}' = {c.value} (Page {c.page}, {c.file})"
            )

    context = "\n".join(context_parts)

    messages.append(
        {
            "role": "user",
            "content": f"Table data:\n{context}\n\nQuestion: {query}\n\n"
            f"Answer with precise cell-level citations.",
        }
    )

    return messages
