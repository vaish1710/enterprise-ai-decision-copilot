"""
Second-stage reranker for the hybrid retriever's candidate list.

`FeatureReranker` is a small learning-to-rank model: it computes several
cheap, interpretable features per (query, chunk) pair -- BM25 score, dense
cosine similarity, exact-term overlap, title match, phrase match -- and
combines them with a logistic regression trained on the labeled eval set
(`eval/retrieval_eval_set.json`). This is the classic "feature-based
reranker" pattern teams reach for before justifying a transformer
cross-encoder in production; `CrossEncoderReranker` documents the drop-in
replacement for when a GPU/transformer budget is available.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from .retrieval import RetrievedChunk, tokenize


def _features(query: str, chunk_text: str, title: str, bm25_score: float, dense_score: float) -> np.ndarray:
    q_tokens = set(tokenize(query))
    c_tokens = set(tokenize(chunk_text))
    t_tokens = set(tokenize(title))

    overlap = len(q_tokens & c_tokens)
    jaccard = overlap / max(1, len(q_tokens | c_tokens))
    title_overlap = len(q_tokens & t_tokens) / max(1, len(q_tokens))
    phrase_hit = 1.0 if query.lower().strip() and query.lower() in chunk_text.lower() else 0.0
    length_norm = min(len(chunk_text.split()), 400) / 400.0

    return np.array([
        bm25_score, dense_score, overlap, jaccard, title_overlap, phrase_hit, length_norm,
    ], dtype=np.float32)


FEATURE_NAMES = ["bm25_score", "dense_score", "term_overlap", "jaccard",
                  "title_overlap", "phrase_hit", "length_norm"]


class FeatureReranker:
    def __init__(self):
        self.model: Optional[LogisticRegression] = None
        # Reasonable fixed weights used until `fit()` is called with labels.
        self._fallback_weights = np.array([0.35, 0.35, 0.10, 0.10, 0.15, 0.20, 0.02])

    def _feature_matrix(self, query: str, candidates: List[RetrievedChunk],
                         bm25_scores: dict, dense_scores: dict) -> np.ndarray:
        rows = []
        for c in candidates:
            rows.append(_features(
                query, c.text, c.title,
                bm25_scores.get(c.chunk_id, 0.0),
                dense_scores.get(c.chunk_id, 0.0),
            ))
        return np.stack(rows)

    def fit(self, training_rows: List[dict]) -> "FeatureReranker":
        """training_rows: [{"features": [...], "label": 0/1}, ...]"""
        X = np.array([r["features"] for r in training_rows], dtype=np.float32)
        y = np.array([r["label"] for r in training_rows], dtype=np.int32)
        if len(set(y.tolist())) < 2:
            return self  # not enough signal to fit; keep fallback weights
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.model.fit(X, y)
        return self

    def score(self, query: str, candidates: List[RetrievedChunk],
              bm25_scores: dict, dense_scores: dict) -> List[RetrievedChunk]:
        if not candidates:
            return []
        X = self._feature_matrix(query, candidates, bm25_scores, dense_scores)
        if self.model is not None:
            probs = self.model.predict_proba(X)[:, 1]
        else:
            Xn = (X - X.min(axis=0)) / (np.ptp(X, axis=0) + 1e-9)
            probs = Xn @ self._fallback_weights

        order = np.argsort(-probs)
        out = []
        for rank, idx in enumerate(order):
            c = candidates[idx]
            out.append(RetrievedChunk(
                chunk_id=c.chunk_id, doc_id=c.doc_id, title=c.title, text=c.text,
                score=float(probs[idx]), rank=rank, source="reranked",
            ))
        return out

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "FeatureReranker":
        with open(path, "rb") as f:
            return pickle.load(f)


class CrossEncoderReranker:
    """
    Documents the swap-in point for a real transformer cross-encoder
    (e.g. sentence-transformers `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`).
    Not used by default because model downloads are blocked in the sandboxed
    environment this repo was authored in -- wire this up once you have
    outbound network access and a few hundred MB for the model weights.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, query: str, candidates: List[RetrievedChunk], **_) -> List[RetrievedChunk]:
        model = self._load()
        pairs = [(query, c.text) for c in candidates]
        scores = model.predict(pairs)
        order = np.argsort(-scores)
        out = []
        for rank, idx in enumerate(order):
            c = candidates[idx]
            out.append(RetrievedChunk(
                chunk_id=c.chunk_id, doc_id=c.doc_id, title=c.title, text=c.text,
                score=float(scores[idx]), rank=rank, source="reranked",
            ))
        return out
