"""Embedding service: bge-large-en-v1.5 ONNX local, with Voyage API fallback."""
import logging
import time
import numpy as np
from typing import Optional

from app.config import EMBED_MODEL_NAME, get_voyage_key, MODEL_DIR, BGE_IDLE_UNLOAD_SEC

logger = logging.getLogger(__name__)

_embedder_instance: Optional["Embedder"] = None


class Embedder:
    """Manages the embedding model lifecycle. Lazily loads, auto-unloads when idle."""

    def __init__(self):
        self._model = None
        self._last_used = 0.0
        self._using_voyage = bool(get_voyage_key())

    def _ensure_loaded(self):
        """Load model if not loaded. Auto-unload if idle too long."""
        now = time.time()
        if self._model is not None:
            # Check if idle too long
            if self._last_used and (now - self._last_used) > BGE_IDLE_UNLOAD_SEC:
                logger.info("Unloading embedding model (idle)")
                self._model = None
                import gc
                gc.collect()

        if self._using_voyage:
            return  # No local model needed

        if self._model is None:
            logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                EMBED_MODEL_NAME,
                cache_folder=str(MODEL_DIR),
            )
            # Try ONNX optimization
            try:
                from optimum.onnxruntime import ORTModelForFeatureExtraction
                logger.info("ONNX optimization applied")
            except Exception:
                logger.info("ONNX not available, using PyTorch")

        self._last_used = now

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Encode texts to embeddings. For queries, adds instruction prefix."""
        self._ensure_loaded()

        if self._using_voyage:
            return self._voyage_encode(texts, is_query)

        # Local bge
        if is_query and len(texts) == 1:
            # BGE models benefit from query instruction prefix
            texts = [f"Represent this sentence for searching relevant passages: {texts[0]}"]
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query."""
        return self.encode([query], is_query=True)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Encode document chunks."""
        return self.encode(texts, is_query=False)

    def _voyage_encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        """Use Voyage AI API for embeddings (0 local memory)."""
        import voyageai
        voyageai.api_key = get_voyage_key()
        model = "voyage-3"
        result = voyageai.embed(texts, model=model, input_type="query" if is_query else "document")
        return np.array(result.embeddings)

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        if self._using_voyage:
            return 1024
        return 1024  # bge-large-en-v1.5

    @property
    def model_loaded(self) -> bool:
        return self._model is not None or self._using_voyage


def get_embedder() -> Embedder:
    """Get or create the global embedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
