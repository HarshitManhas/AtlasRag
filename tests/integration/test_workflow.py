"""Exercise upload, worker, query authorization, and failure handling offline."""

import asyncio
from unittest.mock import AsyncMock, Mock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from starlette.testclient import TestClient

from atlas_rag.api.v1 import documents, query
from atlas_rag.config import get_settings
from atlas_rag.core.generation.llm import LLMClient, LLMResponse
from atlas_rag.core.ingestion.embedder import EmbeddingGenerator
from atlas_rag.core.ingestion.pipeline import IngestionPipeline
from atlas_rag.core.retrieval.pipeline import RetrievalContext
from atlas_rag.core.retrieval.reranker import CrossEncoderReranker
from atlas_rag.core.retrieval.sparse import SparseRetriever
from atlas_rag.database import async_session_factory
from atlas_rag.main import app
from atlas_rag.models.chunk import Chunk
from atlas_rag.models.collection import Collection
from atlas_rag.models.document import Document, DocumentStatus
from atlas_rag.services import storage
from atlas_rag.workers.ingestion_tasks import _run_ingestion


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def actors(client):
    actors = []
    for name in ("owner", "outsider"):
        data = {
            "email": f"{uuid4().hex}@example.com",
            "password": "Password2026!",
            "full_name": name,
        }
        response = client.post("/api/v1/users/register", json=data)
        assert response.status_code == 201
        token = client.post("/api/v1/users/login", json=data).json()["access_token"]
        actors.append((UUID(response.json()["id"]), {"Authorization": f"Bearer {token}"}))
    return actors


async def seed_document(owner_id, status=DocumentStatus.READY, collection_id=None):
    async with async_session_factory() as db:
        doc = Document(
            id=uuid4(),
            owner_id=owner_id,
            title="Private",
            filename="private.txt",
            file_path=f"test/{uuid4()}",
            file_size=20,
            mime_type="text/plain",
            status=status,
            collection_id=collection_id,
        )
        db.add(doc)
        await db.commit()
        return str(doc.id)


async def document_status(doc_id):
    async with async_session_factory() as db:
        doc = await db.get(Document, UUID(doc_id))
        return doc.status if doc else None


def test_upload_worker_query_round_trip(client, actors, monkeypatch, tmp_path):
    """Real parser, chunker, SQLite, local Qdrant, and citations; external services stubbed."""
    from qdrant_client import QdrantClient

    _, headers = actors[0]
    objects = {}
    queued = []

    def enqueue(doc_id):
        assert asyncio.run(document_status(doc_id)) == DocumentStatus.PENDING
        queued.append(doc_id)

    monkeypatch.setattr(
        documents, "store_document", lambda key, data, mime: objects.update({key: data})
    )
    monkeypatch.setattr(storage, "load_document", objects.__getitem__)
    monkeypatch.setattr(documents.ingest_document_task, "delay", enqueue)
    monkeypatch.setattr(get_settings(), "embedding_dimension", 3)
    monkeypatch.setattr(
        EmbeddingGenerator, "embed_batch", lambda self, texts: [[1.0, 0.0, 0.0] for _ in texts]
    )
    monkeypatch.setattr(EmbeddingGenerator, "embed_text", lambda self, text: [1.0, 0.0, 0.0])
    monkeypatch.setattr(
        "qdrant_client.QdrantClient", lambda **kwargs: QdrantClient(path=str(tmp_path / "qdrant"))
    )
    keyword_index = AsyncMock()
    monkeypatch.setattr(IngestionPipeline, "_index_keywords", keyword_index)
    monkeypatch.setattr(SparseRetriever, "search", AsyncMock(return_value=[]))
    monkeypatch.setattr(CrossEncoderReranker, "rerank", lambda self, **kwargs: kwargs["candidates"])
    monkeypatch.setattr(query, "_retrieval", None)
    monkeypatch.setattr(
        LLMClient,
        "complete",
        AsyncMock(
            return_value=LLMResponse(
                content="Private fixture revenue is 731. [1]",
                model="test",
            )
        ),
    )

    response = client.post(
        "/api/v1/documents",
        headers=headers,
        files={
            "file": ("private.txt", b"Private fixture revenue is 731.", "text/plain"),
        },
    )
    assert response.status_code == 201, response.text
    doc_id = response.json()["id"]
    assert queued == [doc_id]
    assert list(objects.values()) == [b"Private fixture revenue is 731."]
    stats = asyncio.run(_run_ingestion(doc_id))
    assert stats["status"] == "ready"
    assert stats["chunks"] > 0
    # A repeated delivery must not duplicate database rows or indexed chunks.
    asyncio.run(_run_ingestion(doc_id))
    keyword_index.assert_awaited_once()

    async def count_chunks():
        async with async_session_factory() as db:
            return await db.scalar(
                select(func.count()).select_from(Chunk).where(Chunk.document_id == UUID(doc_id))
            )

    assert asyncio.run(count_chunks()) == stats["chunks"]
    answer = client.post(
        "/api/v1/query",
        headers=headers,
        json={
            "query": "What is the fixture revenue?",
            "document_ids": [doc_id],
        },
    )
    assert answer.status_code == 200, answer.text
    assert answer.json()["citations"][0]["document_id"] == doc_id
    assert "731" in answer.json()["answer"]


def test_queries_enforce_owner_and_empty_scope(client, actors, monkeypatch):
    owner_id, owner_headers = actors[0]
    outsider_id, outsider_headers = actors[1]
    private_id = asyncio.run(seed_document(owner_id))
    own_id = asyncio.run(seed_document(outsider_id))
    pending_id = asyncio.run(seed_document(outsider_id, DocumentStatus.PENDING))
    search = AsyncMock(return_value=RetrievalContext(query="fixture", results=[]))
    monkeypatch.setattr(query, "_get_retrieval", lambda: Mock(retrieve=search))
    monkeypatch.setattr(
        LLMClient,
        "complete",
        AsyncMock(return_value=LLMResponse(content="No evidence.", model="test")),
    )
    for ids in ([private_id], [own_id, private_id]):
        response = client.post(
            "/api/v1/query",
            headers=outsider_headers,
            json={"query": "fixture", "document_ids": ids},
        )
        assert response.status_code == 404
    search.assert_not_awaited()
    for ids in ([], [pending_id]):
        response = client.post(
            "/api/v1/query",
            headers=outsider_headers,
            json={"query": "fixture", "document_ids": ids},
        )
        assert response.status_code == 200
        assert response.json()["metadata"]["chunks_retrieved"] == 0
    search.assert_not_awaited()
    response = client.post("/api/v1/query", headers=outsider_headers, json={"query": "fixture"})
    assert response.status_code == 200
    assert search.call_args.kwargs["document_ids"] == [own_id]
    assert (
        client.get(f"/api/v1/documents/{private_id}", headers=outsider_headers).status_code == 404
    )
    assert client.get("/api/v1/documents/not-a-uuid", headers=owner_headers).status_code == 422
    assert (
        client.post(
            "/api/v1/query",
            headers=owner_headers,
            json={"query": "fixture", "document_ids": ["bad"]},
        ).status_code
        == 422
    )


def test_collection_scope_is_owned(client, actors, monkeypatch):
    owner_id, headers = actors[0]

    async def setup():
        async with async_session_factory() as db:
            collection = Collection(id=uuid4(), name="Private", owner_id=owner_id)
            db.add(collection)
            await db.commit()
            return collection.id

    collection_id = asyncio.run(setup())
    doc_id = asyncio.run(seed_document(owner_id, collection_id=collection_id))
    search = AsyncMock(return_value=RetrievalContext(query="fixture", results=[]))
    monkeypatch.setattr(query, "_get_retrieval", lambda: Mock(retrieve=search))
    monkeypatch.setattr(
        LLMClient,
        "complete",
        AsyncMock(return_value=LLMResponse(content="No evidence.", model="test")),
    )
    payload = {"query": "fixture", "collection_id": str(collection_id)}
    assert client.post("/api/v1/query", headers=actors[1][1], json=payload).status_code == 404
    assert client.post("/api/v1/query", headers=headers, json=payload).status_code == 200
    assert search.call_args.kwargs["document_ids"] == [doc_id]
    assert (
        client.post(
            f"/api/v1/documents?collection_id={collection_id}",
            headers=actors[1][1],
            files={"file": ("test.txt", b"text", "text/plain")},
        ).status_code
        == 404
    )


def test_upload_failures_are_visible(client, actors, monkeypatch):
    headers = actors[0][1]
    store = Mock(side_effect=OSError("storage down"))
    queue = Mock()
    monkeypatch.setattr(documents, "store_document", store)
    monkeypatch.setattr(documents.ingest_document_task, "delay", queue)
    files = {"file": ("test.txt", b"fixture", "text/plain")}
    response = client.post("/api/v1/documents", headers=headers, files=files)
    assert response.status_code == 503
    queue.assert_not_called()
    store.side_effect = None
    queue.side_effect = OSError("queue down")
    response = client.post("/api/v1/documents", headers=headers, files=files)
    assert response.status_code == 503
    doc_id = response.json()["detail"]["document_id"]
    assert asyncio.run(document_status(doc_id)) == DocumentStatus.FAILED
    monkeypatch.setattr(documents, "MAX_FILE_SIZE", 2)
    store.reset_mock()
    assert client.post("/api/v1/documents", headers=headers, files=files).status_code == 413
    store.assert_not_called()


def test_worker_failure_persists_and_can_retry(actors, monkeypatch):
    doc_id = asyncio.run(seed_document(actors[0][0], DocumentStatus.PENDING))
    monkeypatch.setattr(storage, "load_document", Mock(side_effect=OSError("storage down")))
    with pytest.raises(OSError):
        asyncio.run(_run_ingestion(doc_id))
    assert asyncio.run(document_status(doc_id)) == DocumentStatus.FAILED
    monkeypatch.setattr(storage, "load_document", lambda key: b"fixture")

    async def ingest(self, doc, content, db):
        doc.status = DocumentStatus.READY
        doc.error_message = None
        return {"status": "ready"}

    monkeypatch.setattr(IngestionPipeline, "ingest_document", ingest)
    assert asyncio.run(_run_ingestion(doc_id))["status"] == "ready"
    assert asyncio.run(document_status(doc_id)) == DocumentStatus.READY
    assert asyncio.run(_run_ingestion(str(uuid4())))["status"] == "deleted"


def test_delete_cleans_external_data_and_preserves_row_on_failure(client, actors, monkeypatch):
    doc_id = asyncio.run(seed_document(actors[0][0]))
    indexes = AsyncMock(side_effect=OSError("search down"))
    remove = Mock()
    monkeypatch.setattr(IngestionPipeline, "remove_indexes", indexes)
    monkeypatch.setattr(documents, "remove_document", remove)
    url = f"/api/v1/documents/{doc_id}"
    assert client.delete(url, headers=actors[0][1]).status_code == 503
    assert asyncio.run(document_status(doc_id)) is not None
    remove.assert_not_called()
    indexes.side_effect = None
    assert client.delete(url, headers=actors[0][1]).status_code == 204
    assert asyncio.run(document_status(doc_id)) is None
    remove.assert_called_once()
