"""
Pluggable text embedding backends for the "semantic" half of hybrid retrieval.

Default backend (`LSAEmbedder`) is 100% local: TF-IDF followed by truncated SVD
(latent semantic analysis). It needs no model download and no API key, which
keeps this repo fully reproducible offline. `Embedder` is an abstract interface
so a transformer-based backend (sentence-transformers, OpenAI embeddings, etc.)
can be dropped in without touching any retrieval code -- see
`OpenAIEmbedder` below for the shape that swap would take.
"""
from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class Embedder(ABC):
    """Common interface every embedding backend implements."""

    @abstractmethod
    def fit(self, texts: List[str]) -> "Embedder":
        ...

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (n, dim) L2-normalized float32 matrix."""
        ...

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "Embedder":
        with open(path, "rb") as f:
            return pickle.load(f)


class LSAEmbedder(Embedder):
    """TF-IDF -> truncated SVD dense embeddings. No network access required."""

    def __init__(self, n_components: int = 256, max_features: int = 60_000):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english",
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False

    def fit(self, texts: List[str]) -> "LSAEmbedder":
        tfidf = self.vectorizer.fit_transform(texts)
        # SVD components can't exceed min(n_samples, n_features) - 1
        n_comp = min(self.n_components, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        if n_comp != self.svd.n_components:
            self.svd = TruncatedSVD(n_components=max(n_comp, 2), random_state=42)
        self.svd.fit(tfidf)
        self._fitted = True
        return self

    def embed(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Embedder.fit() must be called before embed().")
        tfidf = self.vectorizer.transform(texts)
        dense = self.svd.transform(tfidf)
        return normalize(dense).astype(np.float32)


class OpenAIEmbedder(Embedder):
    """
    Optional real-embedding backend. Requires `OPENAI_API_KEY` and network access
    (both unavailable in the sandboxed environment this repo was built in, so this
    class is untested end-to-end here -- it documents the intended integration
    point rather than being the default path). Swap it in via `get_embedder()`
    below once you have a key and outbound network access.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError(
                    "pip install openai to use OpenAIEmbedder"
                ) from e
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Set OPENAI_API_KEY to use OpenAIEmbedder")
            self._client = OpenAI(api_key=api_key)
        return self._client

    def fit(self, texts: List[str]) -> "OpenAIEmbedder":
        # Stateless remote embeddings -- nothing to fit locally.
        return self

    def embed(self, texts: List[str]) -> np.ndarray:
        client = self._client_or_raise()
        out = client.embeddings.create(model=self.model, input=texts)
        vecs = np.array([d.embedding for d in out.data], dtype=np.float32)
        return normalize(vecs)


def get_embedder(backend: str = "lsa", **kwargs) -> Embedder:
    if backend == "lsa":
        return LSAEmbedder(**kwargs)
    if backend == "openai":
        return OpenAIEmbedder(**kwargs)
    raise ValueError(f"Unknown embedding backend: {backend}")
