# Working Demo & Interactive UI Guide

AtlasRAG includes two ways to explore the system:
1. **Interactive Web UI Dashboard** at `http://127.0.0.1:8000/` (or `/ui`).
2. **One-Command Headless Demo Script** via `python scripts/demo_walkthrough.py`.

---

## 1. Quickstart: Launching the Web UI

Start the AtlasRAG local server:

```powershell
python run.py
```

Then open your browser to:
👉 **[http://localhost:8000/](http://localhost:8000/)**

### Interactive Web UI Capabilities:
* **Authentication Modal**: Instant JWT register/login modal storing tokens securely in browser state.
* **Document Ingestion Hub**: Drag-and-drop document upload (PDF, DOCX, TXT) with real-time status polling (`pending` → `processing` → `ready`).
* **Query Playground**:
  * **Natural Language Semantic Query**: Retrieves text chunks with paragraph citations.
  * **Numerical Table Query**: Synthesizes SQL, runs in a sandboxed SQLite instance, and displays cell coordinates.
  * **Prompt Injection Defense Test**: Click the test button to observe the Input Guardrail intercepting jailbreak attacks with an immediate HTTP 400 rejection badge.
* **Evidence Citation Cards**: Deep-link badges showing exact document IDs, page numbers, and table cell coordinates (`[table_id, row_idx, col_name]`).
* **Guardrail Verdict**: Real-time Groundedness Score (%) badge showing per-claim NLI support.

---

## 2. Headless Demo Script

For automated testing, CI pipelines, or headless server environments, run:

```bash
python scripts/demo_walkthrough.py
```

This script executes a complete walkthrough in under 1 second:
1. **Authentication**: Native Bcrypt hash verification + JWT token issuance (`Editor` and `Admin`).
2. **Input Guardrail**: Evaluates benign queries vs adversarial jailbreak injections (`ignore previous instructions...`).
3. **Table Reasoning Engine**: Synthesizes SQL from natural language, enforces AST safety validation, runs within an isolated SQLite sandbox, and computes cell citations.
4. **Citation Resolution**: Extracts bracketed citations `[1]`, `[2]` and builds a reference block.
5. **Output Guardrail**: Decomposes answer into atomic claims, computes groundedness score, and outputs verification verdict.

---

## 3. REST API Quick Demo via cURL

### A. Register User & Obtain JWT
```bash
curl -X POST "http://localhost:8000/api/v1/users/register" \
  -H "Content-Type: application/json" \
  -d '{"email": "demo@atlasrag.enterprise", "password": "SecurePassword123!", "full_name": "Demo User"}'
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1Ni...",
  "token_type": "bearer",
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "email": "demo@atlasrag.enterprise",
    "role": "editor"
  }
}
```

### B. Query Documents
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <TOKEN>" \
  -d '{"query": "What is the total revenue of all products in 2024?"}'
```

Response:
```json
{
  "query": "What is the total revenue of all products in 2024?",
  "answer": "The total revenue across all products in 2024 is $995,000 [1].",
  "groundedness_score": 1.0,
  "sources": [
    {
      "source_type": "table_cell",
      "document_id": "doc_benchmark_001",
      "page_number": 12,
      "table_id": "tbl_001",
      "cell_reference": "tbl_001:row=0..4,col=Revenue"
    }
  ]
}
```
