"""Embedding Generator — produces dense vector embeddings for text chunks.

Supports:
  - Local models via sentence-transformers
  - Remote API models via LiteLLM (OpenAI, Cohere, etc.)
  - Batch processing for efficiency
  - Caching to avoid re-embedding identical text
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from atlas_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates dense vector embeddings for text chunks.

    Uses sentence-transformers for local models or LiteLLM for API-based
    embedding models. Supports batch encoding for throughput.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any = None
        self._cache: dict[str, list[float]] = {}

    @property
    def model(self) -> Any:
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.settings.embedding_dimension

    def _load_model(self) -> Any:
        """Load the embedding model based on configuration."""
        model_name = self.settings.embedding_model

        if model_name.startswith("sentence-transformers/") or "/" in model_name:
            # Local sentence-transformers model
            try:
                from sentence_transformers import SentenceTransformer

                model_path = model_name.replace("sentence-transformers/", "")
                model = SentenceTransformer(model_path)
                logger.info("Loaded sentence-transformers model: %s", model_path)
                return model
            except ImportError:
                logger.error("sentence-transformers not installed")
                raise
        else:
            # API-based model (OpenAI, etc.) — handled in embed methods
            logger.info("Using API-based embedding model: %s", model_name)
            return None

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        # Check cache
        cache_key = self._cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = self._encode_single(text)
        self._cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process in each batch.

        Returns:
            List of embedding vectors in the same order as input texts.
        """
        # Check cache for all texts
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Encode uncached texts in batches
        if uncached_texts:
            all_embeddings = self._encode_batch(uncached_texts, batch_size)
            for idx, embedding in zip(uncached_indices, all_embeddings, strict=True):
                results[idx] = embedding
                self._cache[self._cache_key(texts[idx])] = embedding

        return [r for r in results if r is not None]

    def _encode_single(self, text: str) -> list[float]:
        """Encode a single text using the loaded model."""
        model = self.model

        if model is not None:
            # sentence-transformers
            embedding = model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        else:
            # API-based encoding via LiteLLM
            return self._encode_via_api([text])[0]

    def _encode_batch(self, texts: list[str], batch_size: int) -> list[list[float]]:
        """Encode a batch of texts."""
        model = self.model

        if model is not None:
            # sentence-transformers
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 100,
            )
            return [e.tolist() for e in embeddings]
        else:
            # API-based encoding
            all_embeddings: list[list[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_embeddings = self._encode_via_api(batch)
                all_embeddings.extend(batch_embeddings)
            return all_embeddings

    def _encode_via_api(self, texts: list[str]) -> list[list[float]]:
        """Encode texts using LiteLLM embedding API."""
        import litellm

        response = litellm.embedding(
            model=self.settings.embedding_model,
            input=texts,
        )

        return [item["embedding"] for item in response.data]

    @staticmethod
    def _cache_key(text: str) -> str:
        """Generate a cache key for a text string."""
        return hashlib.md5(text.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()
