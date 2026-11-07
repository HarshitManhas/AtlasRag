"""Regression checks for service boundaries and truthful evaluation results."""

import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, Mock

import httpx
import pytest
from botocore.exceptions import ClientError

from atlas_rag.api.v1 import health
from atlas_rag.config import Settings
from atlas_rag.core.generation.llm import LLMClient
from atlas_rag.core.ingestion.pipeline import IngestionPipeline
from atlas_rag.core.retrieval.dense import DenseRetriever
from atlas_rag.core.retrieval.pipeline import RetrievalPipeline
from atlas_rag.core.retrieval.sparse import SparseRetriever
from atlas_rag.evaluation.runner import EvaluationRunner
from atlas_rag.schemas.table import SQLResult
from atlas_rag.services import storage


@pytest.mark.asyncio
async def test_empty_search_scope_never_contacts_backends():
    for component in (DenseRetriever(), SparseRetriever()):
        assert await component.search("fixture", document_ids=[]) == []
    assert (await RetrievalPipeline().retrieve("fixture", document_ids=[])).results == []


@pytest.mark.asyncio
async def test_total_search_outage_is_not_an_empty_result(monkeypatch):
    pipeline = RetrievalPipeline()
    monkeypatch.setattr(pipeline.dense, "search", AsyncMock(side_effect=OSError("down")))
    monkeypatch.setattr(pipeline.sparse, "search", AsyncMock(side_effect=OSError("down")))
    with pytest.raises(RuntimeError, match="Both search backends failed"):
        await pipeline.retrieve("fixture")


@pytest.mark.asyncio
@pytest.mark.parametrize("available", [True, False])
async def test_readiness_reports_dependency_failures(monkeypatch, available):
    connection = AsyncMock()

    @asynccontextmanager
    async def connect():
        yield connection

    monkeypatch.setattr(health, "engine", Mock(connect=connect))
    redis = MagicMock()
    redis.__aenter__ = AsyncMock(return_value=redis)
    redis.__aexit__ = AsyncMock(return_value=None)
    redis.ping = AsyncMock()
    monkeypatch.setattr(health.Redis, "from_url", Mock(return_value=redis))
    s3 = MagicMock()
    s3.__enter__.return_value = s3
    monkeypatch.setattr(health, "storage_client", lambda: s3)
    response = httpx.Response(200, request=httpx.Request("GET", "http://fixture"))
    monkeypatch.setattr(httpx.AsyncClient, "get", AsyncMock(return_value=response))
    if not available:
        redis.ping.side_effect = OSError("private connection detail")
    result = await health.readiness_check()
    assert result.status_code == (200 if available else 503)
    data = json.loads(result.body)
    assert data["dependencies"]["broker"] == ("ready" if available else "unavailable")
    assert "private connection detail" not in result.body.decode()


def test_storage_creates_bucket_and_closes_download(monkeypatch):
    client = MagicMock()
    client.__enter__.return_value = client
    client.head_bucket.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadBucket")
    monkeypatch.setattr(storage, "storage_client", lambda: client)
    storage.store_document("key", b"fixture", "text/plain")
    client.create_bucket.assert_called_once()
    assert client.put_object.call_args.kwargs["Body"] == b"fixture"
    body = client.get_object.return_value["Body"]
    body.read.return_value = b"fixture"
    assert storage.load_document("key") == b"fixture"
    body.close.assert_called_once()
    client.head_bucket.side_effect = ClientError({"Error": {"Code": "403"}}, "HeadBucket")
    client.put_object.reset_mock()
    with pytest.raises(ClientError):
        storage.store_document("key", b"fixture", "text/plain")
    client.put_object.assert_not_called()


@pytest.mark.asyncio
async def test_partial_bulk_failure_cannot_mark_ingestion_ready(monkeypatch):
    from atlas_rag.core.ingestion.chunker import Chunk

    es = AsyncMock()
    es.indices.exists.return_value = True
    es.bulk.return_value = {"errors": True}
    monkeypatch.setattr("elasticsearch.AsyncElasticsearch", Mock(return_value=es))
    with pytest.raises(RuntimeError, match="rejected"):
        await IngestionPipeline()._index_keywords("doc", [Chunk(content="fixture", chunk_index=0)])
    es.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_benchmark_scores_drop_when_components_fail(monkeypatch):
    runner = EvaluationRunner()
    baseline = await runner.run()
    monkeypatch.setattr(
        runner.sql_executor, "execute", Mock(return_value=SQLResult(error="broken SQL"))
    )
    monkeypatch.setattr(runner.citation_extractor, "extract_citations", Mock(return_value=("", [])))
    result = await runner.run()
    assert result.table_sql_execution_success == 0
    assert result.table_numerical_accuracy == 0
    assert result.citation_recall_score == 0
    assert result.passed_tests < baseline.passed_tests
    assert result.overall_score < baseline.overall_score


@pytest.mark.asyncio
async def test_llm_uses_configured_key_without_exporting_environment(monkeypatch):
    import litellm

    response = Mock()
    response.choices = [Mock(message=Mock(content="answer"), finish_reason="stop")]
    response.usage = None
    completion = AsyncMock(return_value=response)
    monkeypatch.setattr(litellm, "acompletion", completion)
    settings = Settings(_env_file=None, openai_api_key="test-key")
    result = await LLMClient(settings).complete([{"role": "user", "content": "fixture"}])
    assert result.content == "answer"
    assert completion.call_args.kwargs["api_key"] == "test-key"
