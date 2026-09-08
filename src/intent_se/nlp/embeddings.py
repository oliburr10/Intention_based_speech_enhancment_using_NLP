"""Sentence embeddings.

Turns a complaint sentence into a fixed-length vector that captures its
*meaning* rather than its words. "The background noise is overwhelming" and
"there is too much ambient sound" share almost no vocabulary but should land
close together; "the noise is too loud" and "I cannot hear what anyone is
saying" should not, even though both describe a listening difficulty.

This is what makes the classification problem tractable. If the embedding model
organises the six classes into distinct clusters, the classifier only has to
find boundaries between them -- a far easier job than classifying raw text.

Model choice
------------
``all-mpnet-base-v2`` (768-dim) over ``all-MiniLM-L6-v2`` (384-dim). MiniLM is
smaller and faster, but mpnet scores higher on Sentence-BERT semantic-similarity
benchmarks, and that matters here because several sentences sit near class
boundaries -- "I can't make out what people are saying" could plausibly be
``TOO_NOISY`` or ``SPEECH_UNCLEAR``. Inference speed is not a bottleneck for a
proof-of-concept, so the higher-quality model wins.

Embeddings are L2-normalised, so only direction matters and cosine similarity
reduces to the dot product.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from intent_se.config import NLPConfig

__all__ = ["SentenceEmbedder"]


class SentenceEmbedder:
    """Wrapper around a Sentence-BERT model with on-disk caching.

    The model is loaded lazily on first use, so importing this module stays
    cheap and the 110M-parameter download only happens when embeddings are
    actually needed.

    Parameters
    ----------
    config:
        NLP configuration selecting the model name and normalisation.
    cache_dir:
        Directory for cached embedding arrays. ``None`` disables caching.

    Examples
    --------
    >>> embedder = SentenceEmbedder()                        # doctest: +SKIP
    >>> vecs = embedder.encode(["the noise is too loud"])    # doctest: +SKIP
    >>> vecs.shape                                           # doctest: +SKIP
    (1, 768)
    """

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

    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------

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

        Parameters
        ----------
        sentences:
            Sentences to embed.
        batch_size:
            Encoder batch size.
        show_progress:
            Show the sentence-transformers progress bar.
        use_cache:
            Read from and write to the on-disk cache when ``cache_dir`` is set.
            Embedding 1106 sentences takes a few seconds on GPU but the better
            part of a minute on CPU, so caching matters during development.

        Returns
        -------
        np.ndarray
            Array of shape ``(len(sentences), dim)``, L2-normalised if
            ``config.normalize_embeddings`` is set.
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
