"""Embedding service: external API (OpenAI / Voyage) or BM25-only fallback.

When no embedding API key is configured, uses zero vectors for ChromaDB
storage and relies on BM25 keyword search + LLM reranking for retrieval.
No local models, no extra API keys needed.
"""
import logging
from typing import Optional

import numpy as np

from app.config import (
    get_openai_key, get_voyage_key, get_deepseek_key,
    OPENAI_EMBEDDING_MODEL, VOYAGE_EMBEDDING_MODEL,
    EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)

_embedder_instance: Optional["Embedder"] = None


class Embedder:
    """Cloud embedding or BM25-only fallback. Zero local dependencies."""

    def __init__(self):
        self._backend: Optional[str] = None
        self._client = None
        self._dim = EMBEDDING_DIM

    def _ensure_client(self):
        if self._backend is not None:
            return

        # Voyage
        if get_voyage_key():
            self._backend = "voyage"
            self._dim = 1024
            logger.info("Embedding: Voyage AI")
            return

        # OpenAI
        if get_openai_key():
            from openai import AsyncOpenAI
            self._backend = "openai"
            self._client = AsyncOpenAI(api_key=get_openai_key())
            self._dim = 1536 if "ada" in OPENAI_EMBEDDING_MODEL else 1024
            logger.info(f"Embedding: OpenAI ({OPENAI_EMBEDDING_MODEL})")
            return

        # DeepSeek — no embedding API, fall through to BM25-only
        self._backend = "bm25"
        self._dim = 1024
        logger.info("Embedding: BM25-only (no embedding API key configured)")

    async def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        self._ensure_client()

        if self._backend == "openai":
            return await self._openai_encode(texts)
        elif self._backend == "voyage":
            return await self._voyage_encode(texts, is_query)
        elif self._backend == "bm25":
            # Return zero vectors — ChromaDB needs something, but BM25 does the real work
            return np.zeros((len(texts), self._dim), dtype=np.float32)
        else:
            return np.zeros((len(texts), self._dim), dtype=np.float32)

    async def encode_query(self, query: str) -> np.ndarray:
        return await self.encode([query], is_query=True)

    async def encode_documents(self, texts: list[str]) -> np.ndarray:
        return await self.encode(texts, is_query=False)

    async def _openai_encode(self, texts: list[str]) -> np.ndarray:
        resp = await self._client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL, input=texts,
        )
        return np.array([d.embedding for d in resp.data], dtype=np.float32)

    async def _voyage_encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {get_voyage_key()}"},
                json={
                    "model": VOYAGE_EMBEDDING_MODEL,
                    "input": texts,
                    "input_type": "query" if is_query else "document",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return np.array([d["embedding"] for d in data["data"]], dtype=np.float32)

    @property
    def dimension(self) -> int:
        self._ensure_client()
        return self._dim

    @property
    def has_semantic(self) -> bool:
        """Whether we have real semantic search (vs BM25-only)."""
        self._ensure_client()
        return self._backend in ("openai", "voyage")


def get_embedder() -> Embedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
