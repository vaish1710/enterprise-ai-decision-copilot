"""
Hybrid retrieval: BM25 (sparse/lexical) + dense (semantic) search, fused with
Reciprocal Rank Fusion (RRF). RRF is the same fusion technique used in
production hybrid-search systems (e.g. Elastic, Azure AI Search, Weaviate) --
it needs no score calibration between the two retrievers, which is what makes
combining a lexical and a dense retriever tractable in practice.
"""
from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import Embedder, get_embedder
from .query_understanding import normalize_query


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    rank: int
    source: str  # "bm25", "dense", or "hybrid"


class BM25Retriever:
    def __init__(self, chunks: List[dict]):
        self.chunks = chunks
        self._tokenized = [tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 20) -> List[RetrievedChunk]:
        query = normalize_query(query)
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:top_k]
        out = []
        for rank, idx in enumerate(order):
            c = self.chunks[idx]
            out.append(RetrievedChunk(
                chunk_id=c["chunk_id"], doc_id=c["doc_id"], title=c["title"],
                text=c["text"], score=float(scores[idx]), rank=rank, source="bm25",
            ))
        return out


class DenseRetriever:
    def __init__(self, chunks: List[dict], embedder: Optional[Embedder] = None):
        self.chunks = chunks
        self.embedder = embedder or get_embedder("lsa")
        texts = [c["text"] for c in chunks]
        self.embedder.fit(texts)
        self.matrix = self.embedder.embed(texts)  # (n, dim), L2-normalized

    def search(self, query: str, top_k: int = 20) -> List[RetrievedChunk]:
        query = normalize_query(query)
        qvec = self.embedder.embed([query])[0]
        sims = self.matrix @ qvec  # cosine sim since both normalized
        order = np.argsort(-sims)[:top_k]
        out = []
        for rank, idx in enumerate(order):
            c = self.chunks[idx]
            out.append(RetrievedChunk(
                chunk_id=c["chunk_id"], doc_id=c["doc_id"], title=c["title"],
                text=c["text"], score=float(sims[idx]), rank=rank, source="dense",
            ))
        return out


class HybridRetriever:
    """Fuses BM25 and dense rankings with Reciprocal Rank Fusion."""

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int = 60):
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int = 10, candidate_pool: int = 50) -> List[RetrievedChunk]:
        bm25_hits = self.bm25.search(query, top_k=candidate_pool)
        dense_hits = self.dense.search(query, top_k=candidate_pool)

        fused: Dict[str, float] = {}
        meta: Dict[str, dict] = {}
        for hits in (bm25_hits, dense_hits):
            for h in hits:
                fused[h.chunk_id] = fused.get(h.chunk_id, 0.0) + 1.0 / (self.rrf_k + h.rank + 1)
                meta[h.chunk_id] = {"doc_id": h.doc_id, "title": h.title, "text": h.text}

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        out = []
        for rank, (cid, score) in enumerate(ranked):
            m = meta[cid]
            out.append(RetrievedChunk(
                chunk_id=cid, doc_id=m["doc_id"], title=m["title"], text=m["text"],
                score=score, rank=rank, source="hybrid",
            ))
        return out


def load_chunks(data_dir: Path) -> List[dict]:
    chunks = []
    with open(data_dir / "chunks.jsonl") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_indexes(data_dir: Path):
    chunks = load_chunks(data_dir)
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)
    hybrid = HybridRetriever(bm25, dense)
    return chunks, bm25, dense, hybrid
