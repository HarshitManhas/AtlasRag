"""Shared test fixtures for AtlasRAG."""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

# Configure isolation before test modules import the application engine.
_test_directory = tempfile.TemporaryDirectory(prefix="atlas-tests-")
os.environ["DATABASE_URL"] = (
    "sqlite+aiosqlite:///" + (Path(_test_directory.name) / "test.db").as_posix()
)
os.environ["DEBUG"] = "false"


@pytest.fixture(scope="session", autouse=True)
def database():
    from atlas_rag.database import close_db, init_db

    asyncio.run(init_db())
    yield
    asyncio.run(close_db())
    _test_directory.cleanup()
