"""FastAPI application entry point for AtlasRAG."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse

from atlas_rag.config import get_settings
from atlas_rag.database import close_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup and shutdown hooks."""
    settings = get_settings()
    logger.info("Starting AtlasRAG (%s)", settings.environment)
    yield
    logger.info("Shutting down AtlasRAG")
    await close_db()


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI app."""
    settings = get_settings()

    app = FastAPI(
        title="AtlasRAG",
        description="Enterprise-Grade Hybrid Retrieval-Augmented Generation Platform",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.environment == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    from atlas_rag.api.v1.documents import router as documents_router
    from atlas_rag.api.v1.health import router as health_router
    from atlas_rag.api.v1.query import router as query_router
    from atlas_rag.api.v1.users import router as users_router

    app.include_router(health_router)
    app.include_router(users_router, prefix="/api/v1")
    app.include_router(documents_router, prefix="/api/v1")
    app.include_router(query_router, prefix="/api/v1")

    # Serve Web UI
    static_file = Path(__file__).parent / "static" / "index.html"

    @app.get("/", include_in_schema=False)
    @app.get("/ui", include_in_schema=False)
    async def serve_ui() -> FileResponse:
        return FileResponse(static_file)

    return app


app = create_app()
