import asyncio
import os
import sys

# Ensure src is in sys.path
sys.path.insert(0, os.path.abspath("src"))

from atlas_rag.config import get_settings
from atlas_rag.database import init_db, engine, Base
import atlas_rag.models

async def main():
    print(f"Engine URL: {engine.url}")
    print(f"Tables in Base.metadata: {list(Base.metadata.tables.keys())}")
    await init_db()
    print("Database schema successfully created!")

if __name__ == "__main__":
    asyncio.run(main())
