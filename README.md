# AtlasRAG

**Document question answering with hybrid retrieval and source citations**

AtlasRAG stores uploads in S3/MinIO, processes them with Celery, and answers questions using indexed text and tables.

## Key Features

- **Hybrid Retrieval**: Dense vector search (Qdrant) + BM25 keyword search (Elasticsearch) fused via Reciprocal Rank Fusion
- **Cross-Encoder Reranking**: Pairwise relevance scoring for improved retrieval accuracy
- **Advanced Table Understanding**: Extracts complex tables from PDFs, generates SQL queries, provides cell-level citations
- **Evidence-Based Citations**: Page-level, section-level, and table-cell-level source references
- **Groundedness Checks**: Keyword-based evidence scoring; a heuristic, not a guarantee of factual accuracy
- **Access Control**: JWT authentication and owner-scoped documents, queries, collections, and deletion
- **Prompt Injection Protection**: Pattern-based detection and input sanitization
- **Health Checks**: `/health` checks the API process; `/health/ready` checks the database schema, Redis, Qdrant, Elasticsearch, and object storage

Role definitions, audit models, and observability counters exist, but role policies, audit-event recording, Prometheus export, and OpenTelemetry wiring are not implemented. The query endpoint returns JSON; streaming requests are rejected explicitly. Readiness does not verify worker availability or LLM credentials.

## Quick Start

```bash
# 1. From the project directory, create and activate an environment
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

# 2. Copy environment config
cp .env.example .env
# PowerShell: Copy-Item .env.example .env
# Set your API key and a random JWT_SECRET_KEY.
# Keep the example PostgreSQL DATABASE_URL for the Docker setup.

# 3. Start infrastructure
docker compose up -d --build

# 4. Run database migrations
python -m alembic upgrade head

# 5. Start the API server
python -m uvicorn atlas_rag.main:app --reload --host 127.0.0.1 --port 8000

# 6. In a second terminal with the same environment, start ingestion
python -m celery -A atlas_rag.workers.celery_app worker --loglevel=info
# Windows development: append --pool=solo
```

The API will be available at `http://localhost:8000/api/docs` (Swagger UI).

Register, log in, authorize with the returned bearer token, upload a document, and poll its status until `ready` before querying. Only ready documents owned by the authenticated user are searchable. The object-storage bucket is created on first upload. A queue failure returns 503 with the document ID; remove that failed upload and upload again once the queue is healthy.

The first embedding/reranking request downloads the configured Hugging Face models. API keys in `.env` are passed to the LLM client.

Initial setup builds MinIO from a pinned upstream source commit because the old published image is unavailable. The [upstream community project](https://github.com/minio/minio) is archived; this Compose stack is for local development. Use a maintained S3 service for production.

Migrations target an empty database. Existing databases created by `init_db()` must have their schema checked against the initial migration before using `alembic stamp head`.

## Architecture

```
Query → API Gateway → Hybrid Retrieval → RRF Fusion → Cross-Encoder Rerank
                                                            ↓
    Response ← Hallucination Check ← Citation Engine ← LLM Generation
                                                            ↑
                                                    Table Analysis (SQL)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI + Uvicorn |
| ORM | SQLAlchemy 2.0 (async) |
| Database | PostgreSQL 16 |
| Vector Store | Qdrant |
| Keyword Search | Elasticsearch 8 |
| Cache / Broker | Redis 7 |
| LLM | LiteLLM (OpenAI, Anthropic, Google) |
| Embeddings | sentence-transformers |
| Reranker | cross-encoder/ms-marco-MiniLM |
| Task Queue | Celery |
| Logging | Python logging; observability definitions are not yet wired |

## Development

```bash
make lint       # Run linting
make format     # Auto-format code
make test       # Run tests
make test-cov   # Run tests with coverage
make worker     # Start Celery worker
```

Without Make (including Windows), use `python -m pytest tests/`, `python -m ruff check src/ tests/`, and `python -m ruff format --check src/ tests/`.

Tests use a temporary SQLite database independent of `.env`. The workflow regression uses real parsing, chunking, local Qdrant, and citation generation, with external services and model outputs stubbed. Other regressions cover ownership, empty scopes, collection filters, queue/storage failures, retries, and readiness.

`python scripts/run_evaluation.py` runs an **offline component benchmark**, not a platform accuracy score. It measures heuristic SQL on the table cases in `golden_set.json`, RRF fusion on synthetic search results, citation extraction, and keyword groundedness. It does not measure live retrieval, cross-encoder quality, PDF extraction accuracy, or LLM answers. Metrics are computed from results; component failures lower the score. The SQL fallback handles only simple queries and can return extra rows for lookups.

## License

MIT
