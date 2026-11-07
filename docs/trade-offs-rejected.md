# Trade-Offs Rejected in AtlasRAG

Engineering high-reliability enterprise RAG is defined more by what you **reject** than what you include. Below are the key architectural decisions, alternatives evaluated, reasons for rejection, and the battle-tested designs selected for AtlasRAG.

---

### 1. Rejected: Naive Chunked Text RAG for Tables
* **Alternative Considered**: Converting extracted tables into raw Markdown/CSV text blocks, chunking them, and inserting them into standard vector databases for semantic search.
* **Why Rejected**:
  1. **Arithmetic & Aggregation Failure**: LLMs are autoregressive token predictors, not calculators. When asked questions like *"What was the total operating revenue across all four quarters?"*, pure text RAG hallucinates sums, skips rows, or fabricates numbers.
  2. **Boundary Severing**: Standard chunkers (character or token based) split tables across chunk boundaries, separating headers from numerical rows and destroying structural hierarchy.
  3. **Lack of Coordinate Attribution**: Chunk-level citations only point to a generic block of text, leaving compliance and audit teams unable to locate the exact cell.
* **Adopted Design**: **Deterministic Table Reasoning Engine**
  - High-fidelity table extraction with `pdfplumber` and `camelot`.
  - Ingestion into typed, sandboxed SQLite tables with typed schemas (integer, float, text, currency).
  - Intent classification routing numerical/aggregation queries to a Text-to-SQL synthesis pipeline.
  - Cell-level coordinate citation tracking mapping answers back to `(table_id, row_idx, col_name, page_num)`.

---

### 2. Rejected: Dense-Only Vector Retrieval
* **Alternative Considered**: Pure vector similarity search using modern dense embedding models (e.g., `text-embedding-3-small`, `all-MiniLM-L6-v2`) via cosine distance.
* **Why Rejected**:
  1. **Keyword & Identifier Blindness**: Dense embeddings compress semantics into continuous vectors, smoothing away exact serial numbers, product codes (`Part #X702-B`), legal clauses (`Section 14.2(b)`), and financial acronyms.
  2. **Semantic Dilution**: When short, specific queries are matched against long chunk embeddings, dense search often surfaces conceptually adjacent but factually incorrect documents.
* **Adopted Design**: **Reciprocal Rank Fusion (RRF) Hybrid Search**
  - **Dense Retrieval**: Qdrant vector index for deep semantic similarity and paraphrased intents.
  - **Sparse Retrieval**: Elasticsearch 8 BM25 inverted index for token-exact lexical matching.
  - **Fusion**: Score-normalization-free Reciprocal Rank Fusion:
    $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + r_m(d)}$$
  - **Precision Reranking**: Cross-encoder (`ms-marco-MiniLM-L-6-v2`) scoring the top 20 fused candidates to guarantee maximal precision at rank 1.

---

### 3. Rejected: Arbitrary Python Execution (`eval()` / Pandas Agents)
* **Alternative Considered**: Utilizing LangChain `create_pandas_dataframe_agent` or executing dynamic Python code in a REPL to manipulate extracted data frames.
* **Why Rejected**:
  1. **Catastrophic Security Risk**: Executing generated Python code opens the door to Remote Code Execution (RCE) via prompt injection. Adversarial prompts can escape sandboxes, inspect environment variables, read local files, or spawn reverse shells.
  2. **Non-Deterministic Resource Consumption**: Unconstrained Python loops can exhaust server memory, cause runaway CPU spikes, and starve concurrent users.
* **Adopted Design**: **AST-Validated, In-Memory Read-Only SQLite Sandboxing**
  - Generated SQL queries pass through an abstract syntax tree (AST) validator.
  - Strict whitelist: only read-only `SELECT` statements are permitted; statements containing `DROP`, `DELETE`, `INSERT`, `UPDATE`, `ALTER`, `ATTACH`, or `PRAGMA` are instantly rejected.
  - Execution runs inside an ephemeral, in-memory SQLite instance with a hard 500ms timeout and maximum 1,000-row return limit.

---

### 4. Rejected: Heavyweight Multi-Agent Frameworks (LangChain / CrewAI / AutoGen)
* **Alternative Considered**: Orchestrating the system using dynamic multi-agent loops where agents autonomously debate, plan, and delegate tasks to sub-agents.
* **Why Rejected**:
  1. **Latency Explosion**: Autonomous multi-turn agent debates regularly incur 15 to 45 seconds of latency per query—unacceptable for enterprise search SLAs (< 2 seconds).
  2. **Token Bleed & Cost**: Recursive prompting generates thousands of redundant tokens per interaction, ballooning API costs 10x to 20x.
  3. **Debugging Opacity**: Complex agent traces create non-deterministic failure modes where diagnosing the root cause of an incorrect answer requires inspecting dozens of opaque conversational turns.
* **Adopted Design**: **Deterministic, Typed FastAPI Service Pipelines**
  - Linear, predictable async pipeline: `Input Guard → Hybrid Retrieval / Table SQL → Context Synthesis → LLM Completion → Citation Extractor → Output Guard`.
  - Predictable p95 latency under 1.5 seconds.
  - Direct LiteLLM client without middleware abstraction overhead.

---

### 5. Rejected: Blind LLM Trust (Unchecked Generation)
* **Alternative Considered**: Streaming raw LLM completions directly to users and assuming model accuracy.
* **Why Rejected**:
  1. **Silent Hallucinations**: Even GPT-4o fabricates facts, numbers, and references when context is dense or contradictory.
  2. **Compliance & Audit Failure**: Enterprise customers in finance, legal, and healthcare require deterministic evidence attribution for every assertion.
* **Adopted Design**: **Bidirectional Guardrails**
  - **Input Guardrail**: Pattern-based and jailbreak detection blocking injection attempts before retrieval or LLM execution occurs.
  - **Output Guardrail**: Atomic claim decomposition with Natural Language Inference (NLI) entailment verification against retrieved context chunks and table cells. Answers below the groundedness threshold trigger explicit warnings or fallback disclosures.
