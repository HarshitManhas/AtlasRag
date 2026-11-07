"""Ingestion Pipeline — orchestrates document upload → parse → chunk → embed → index.

This is the top-level entry point for all document processing in AtlasRAG.
It coordinates the parser, chunker, table extractor, embedder, and vector/keyword
stores to fully ingest a document and make it searchable.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import closing
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from atlas_rag.config import Settings, get_settings
from atlas_rag.core.ingestion.chunker import Chunk as ChunkData
from atlas_rag.core.ingestion.chunker import TextChunker
from atlas_rag.core.ingestion.embedder import EmbeddingGenerator
from atlas_rag.core.ingestion.parser import DocumentParser
from atlas_rag.core.table_analysis.pipeline import TableAnalysisPipeline
from atlas_rag.models.chunk import Chunk
from atlas_rag.models.document import Document, DocumentStatus

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end document ingestion: parse → chunk → embed → index.

    Orchestrates the full ingestion workflow:
      1. Parse document (PDF/DOCX/HTML) into pages
      2. Extract tables from PDFs
      3. Chunk text into retrieval segments
      4. Generate embeddings for each chunk
      5. Index chunks in Qdrant (vector) and Elasticsearch (BM25)
      6. Persist chunk metadata to PostgreSQL
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.parser = DocumentParser()
        self.chunker = TextChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.embedder = EmbeddingGenerator(self.settings)
        self.table_pipeline = TableAnalysisPipeline(self.settings)

    async def ingest_document(
        self,
        document: Document,
        file_content: bytes,
        db: AsyncSession,
    ) -> dict[str, Any]:
        """Ingest a document: parse, chunk, embed, and index.

        Args:
            document: The Document ORM object (already created in DB).
            file_content: Raw file bytes.
            db: Async database session.

        Returns:
            Dict with ingestion statistics.
        """
        doc_id = str(document.id)
        filename = document.filename
        mime_type = document.mime_type

        logger.info("Starting ingestion for document %s (%s)", doc_id, filename)

        try:
            # Update status
            document.status = DocumentStatus.PROCESSING
            document.error_message = None
            document.table_count = 0
            await db.flush()

            # Step 1: Parse document
            parsed = self.parser.parse(file_content, filename, mime_type)
            document.page_count = parsed.total_pages

            # Step 2: Extract tables (PDF only)
            tables = []
            if mime_type == "application/pdf":
                tables = await self.table_pipeline.extract_and_store(
                    pdf_bytes=file_content,
                    filename=filename,
                    document_id=doc_id,
                    db=db,
                )
                document.table_count = len(tables)

            # Step 3: Chunk text
            text_chunks = self.chunker.chunk_document(parsed)

            # Also create chunks from table text representations
            for table in tables:
                if table.text_representation:
                    text_chunks.append(
                        ChunkData(
                            content=table.text_representation,
                            chunk_index=len(text_chunks),
                            page_number=table.source.page,
                            section_header=table.title or f"Table {table.table_id}",
                            metadata={"type": "table", "table_id": table.table_id},
                        )
                    )

            document.chunk_count = len(text_chunks)

            # Step 4: Generate embeddings
            chunk_texts = [c.content for c in text_chunks]
            embeddings = self.embedder.embed_batch(chunk_texts)

            # Step 5: Index in vector store (Qdrant)
            point_ids = await self._index_vectors(doc_id, text_chunks, embeddings)

            # Step 6: Index in keyword store (Elasticsearch)
            await self._index_keywords(doc_id, text_chunks)

            # Step 7: Persist chunk metadata to PostgreSQL
            for i, chunk_data in enumerate(text_chunks):
                chunk = Chunk(
                    id=uuid.uuid4(),
                    document_id=document.id,
                    content=chunk_data.content,
                    chunk_index=chunk_data.chunk_index,
                    page_number=chunk_data.page_number,
                    section_header=chunk_data.section_header,
                    embedding_id=point_ids[i] if i < len(point_ids) else None,
                    token_count=chunk_data.token_count,
                    metadata_=chunk_data.metadata,
                )
                db.add(chunk)

            # Mark as ready
            document.status = DocumentStatus.READY
            await db.flush()

            stats = {
                "document_id": doc_id,
                "filename": filename,
                "pages": parsed.total_pages,
                "chunks": len(text_chunks),
                "tables": len(tables),
                "status": "ready",
            }
            logger.info("Ingestion complete for %s: %s", doc_id, stats)
            return stats

        except Exception as e:
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)
            await db.flush()
            logger.exception("Ingestion failed for document %s", doc_id)
            raise

    async def _index_vectors(
        self,
        doc_id: str,
        chunks: list[ChunkData],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Index chunk embeddings in Qdrant."""
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        with closing(QdrantClient(url=self.settings.qdrant_url)) as client:
            collection = self.settings.qdrant_collection

            # Ensure collection exists
            collections = client.get_collections().collections
            if not any(c.name == collection for c in collections):
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=self.settings.embedding_dimension,
                        distance=Distance.COSINE,
                    ),
                )

            # Build points
            point_ids: list[str] = []
            points: list[PointStruct] = []

            for chunk, embedding in zip(chunks, embeddings, strict=True):
                point_id = str(uuid.uuid5(uuid.UUID(doc_id), str(chunk.chunk_index)))
                point_ids.append(point_id)
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "document_id": doc_id,
                            "chunk_index": chunk.chunk_index,
                            "page_number": chunk.page_number,
                            "section_header": chunk.section_header,
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                        },
                    )
                )

            # Upsert in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                client.upsert(collection_name=collection, points=batch)

            logger.info("Indexed %d vectors in Qdrant for document %s", len(points), doc_id)
            return point_ids

    async def _index_keywords(
        self,
        doc_id: str,
        chunks: list[ChunkData],
    ) -> None:
        """Index chunk text in Elasticsearch for BM25 keyword search."""
        from elasticsearch import AsyncElasticsearch

        es = AsyncElasticsearch(self.settings.elasticsearch_url)
        index = self.settings.elasticsearch_index

        try:
            # Ensure index exists
            if not await es.indices.exists(index=index):
                await es.indices.create(
                    index=index,
                    body={
                        "settings": {
                            "number_of_shards": 1,
                            "number_of_replicas": 0,
                            "analysis": {
                                "analyzer": {
                                    "text_analyzer": {
                                        "type": "standard",
                                        "stopwords": "_english_",
                                    }
                                }
                            },
                        },
                        "mappings": {
                            "properties": {
                                "document_id": {"type": "keyword"},
                                "chunk_index": {"type": "integer"},
                                "content": {
                                    "type": "text",
                                    "analyzer": "text_analyzer",
                                },
                                "page_number": {"type": "integer"},
                                "section_header": {"type": "text"},
                            }
                        },
                    },
                )

            # Bulk index
            actions = []
            for chunk in chunks:
                actions.append({"index": {"_index": index, "_id": f"{doc_id}:{chunk.chunk_index}"}})
                actions.append(
                    {
                        "document_id": doc_id,
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "page_number": chunk.page_number,
                        "section_header": chunk.section_header,
                    }
                )

            if actions:
                response = await es.bulk(body=actions, refresh=True)
                if response.get("errors"):
                    raise RuntimeError("Elasticsearch rejected one or more document chunks")
                logger.info(
                    "Indexed %d chunks in Elasticsearch for document %s",
                    len(chunks),
                    doc_id,
                )
        finally:
            await es.close()

    async def remove_indexes(self, doc_id: str) -> None:
        """Remove all indexed chunks before deleting their source document."""
        from elasticsearch import AsyncElasticsearch
        from qdrant_client import QdrantClient
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        with closing(QdrantClient(url=self.settings.qdrant_url)) as client:
            if client.collection_exists(self.settings.qdrant_collection):
                client.delete(
                    collection_name=self.settings.qdrant_collection,
                    points_selector=FilterSelector(
                        filter=Filter(
                            must=[
                                FieldCondition(key="document_id", match=MatchValue(value=doc_id)),
                            ]
                        )
                    ),
                    wait=True,
                )
        async with AsyncElasticsearch(self.settings.elasticsearch_url) as es:
            if await es.indices.exists(index=self.settings.elasticsearch_index):
                response = await es.delete_by_query(
                    index=self.settings.elasticsearch_index,
                    query={"term": {"document_id": doc_id}},
                    refresh=True,
                )
                if response.get("failures") or response.get("timed_out"):
                    raise RuntimeError("Elasticsearch document cleanup failed")
