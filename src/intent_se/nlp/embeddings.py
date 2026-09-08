from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from intent_se.config import NLPConfig

__all__ = ["SentenceEmbedder"]


class SentenceEmbedder:
    def __init__(
        self,
        config: NLPConfig | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.cfg = config or NLPConfig()
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None

    @property
    def model(self):
        """The underlying ``SentenceTransformer``, loaded on first access."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Sentence embeddings need the 'sentence-transformers' package.\n"
                    "Install it with: pip install '.[nlp]'"
                ) from exc
            self._model = SentenceTransformer(self.cfg.embedding_model)
        return self._model

    @property
    def dim(self) -> int:
        """Embedding dimensionality (768 for all-mpnet-base-v2)."""
        return self.cfg.embedding_dim


    def _cache_path(self, sentences: Sequence[str]) -> Path | None:
        """Deterministic cache filename derived from the model and corpus."""
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256()
        digest.update(self.cfg.embedding_model.encode())
        digest.update(str(self.cfg.normalize_embeddings).encode())
        for s in sentences:
            digest.update(s.encode())
            digest.update(b"\x00")
        return self.cache_dir / f"emb_{digest.hexdigest()[:16]}.npy"

    def encode(
        self,
        sentences: Sequence[str],
        batch_size: int = 32,
        show_progress: bool = False,
        use_cache: bool = True,
    ) -> np.ndarray:
        """Embed a list of sentences.
        """
        sentences = list(sentences)
        if not sentences:
            return np.empty((0, self.dim), dtype=np.float32)

        cache = self._cache_path(sentences) if use_cache else None
        if cache is not None and cache.exists():
            return np.load(cache)

        vectors = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.cfg.normalize_embeddings,
        ).astype(np.float32)

        if cache is not None:
            np.save(cache, vectors)

        return vectors

    def encode_one(self, sentence: str) -> np.ndarray:
        """Embed a single sentence, returning a 1-D vector of shape ``(dim,)``."""
        return self.encode([sentence], use_cache=False)[0]
