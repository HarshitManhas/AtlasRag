"""Health check endpoints."""

import asyncio

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy import text

from atlas_rag.config import get_settings
from atlas_rag.database import engine
from atlas_rag.services.storage import storage_client

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Basic health check."""
    return {"status": "healthy", "service": "AtlasRAG"}


@router.get("/health/ready")
async def readiness_check() -> JSONResponse:
    """Readiness check — verifies all dependencies are available."""
    settings = get_settings()

    async def database():
        async with engine.connect() as connection:
            await connection.execute(text("SELECT id FROM users LIMIT 1"))

    async def redis(url):
        async with Redis.from_url(url, socket_connect_timeout=3, socket_timeout=3) as client:
            await client.ping()

    def storage():
        with storage_client() as client:
            client.list_buckets()

    async with httpx.AsyncClient(timeout=3) as client:

        async def http(url):
            response = await client.get(url)
            response.raise_for_status()

        checks = {
            "database": database(),
            "redis": redis(settings.redis_url),
            "broker": redis(settings.celery_broker_url),
            "result_backend": redis(settings.celery_result_backend),
            "qdrant": http(f"{settings.qdrant_url.rstrip('/')}/readyz"),
            "elasticsearch": http(f"{settings.elasticsearch_url.rstrip('/')}/_cluster/health"),
            "storage": asyncio.to_thread(storage),
        }
        results = await asyncio.gather(
            *(asyncio.wait_for(check, timeout=5) for check in checks.values()),
            return_exceptions=True,
        )
    dependencies = {
        name: "unavailable" if isinstance(result, BaseException) else "ready"
        for name, result in zip(checks, results, strict=True)
    }
    ready = all(value == "ready" for value in dependencies.values())
    return JSONResponse(
        {"status": "ready" if ready else "not_ready", "dependencies": dependencies},
        status_code=200 if ready else 503,
    )
