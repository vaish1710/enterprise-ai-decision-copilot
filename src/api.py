"""
FastAPI service exposing the two halves of the copilot:

  POST /search  -- hybrid (BM25 + dense) retrieval over the document corpus,
                    reranked with the trained FeatureReranker.
  POST /ask     -- natural-language business question -> validated SQL ->
                    executed result, with confidence-based fallback.
  GET  /healthz -- liveness/readiness probe.

Indexes and the DB connection are built once at startup (see `lifespan`) and
reused across requests -- rebuilding the TF-IDF/BM25 index per request would
dominate latency.
"""
from __future__ import annotations

import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .retrieval import build_indexes
from .reranker import FeatureReranker
from .nl2sql.translator import TemplateTranslator
from .nl2sql.validator import answer_question

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["chunks"], STATE["bm25"], STATE["dense"], STATE["hybrid"] = build_indexes(DATA_DIR)
    reranker_path = RESULTS_DIR / "reranker.pkl"
    STATE["reranker"] = FeatureReranker.load(reranker_path) if reranker_path.exists() else FeatureReranker()
    STATE["translator"] = TemplateTranslator()
    STATE["db"] = sqlite3.connect(DATA_DIR / "business.db", check_same_thread=False)
    yield
    STATE["db"].close()


app = FastAPI(
    title="Enterprise AI Decision Intelligence Copilot",
    description="Hybrid RAG search + validated NL-to-SQL over a synthetic enterprise knowledge base.",
    version="1.0.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    rerank: bool = True


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]
    latency_ms: float


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    status: str
    sql: Optional[str]
    confidence: float
    columns: list[str]
    rows: list[list]
    message: str
    latency_ms: float


@app.get("/healthz")
def healthz():
    return {"status": "ok", "n_chunks": len(STATE.get("chunks", []))}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    t0 = time.time()
    hybrid = STATE["hybrid"]
    pool = hybrid.search(req.query, top_k=max(30, req.top_k), candidate_pool=30)

    if req.rerank:
        bm25_pool = STATE["bm25"].search(req.query, top_k=30)
        dense_pool = STATE["dense"].search(req.query, top_k=30)
        bm25_scores = {h.chunk_id: h.score for h in bm25_pool}
        dense_scores = {h.chunk_id: h.score for h in dense_pool}
        results = STATE["reranker"].score(req.query, pool, bm25_scores, dense_scores)[:req.top_k]
    else:
        results = pool[:req.top_k]

    latency_ms = (time.time() - t0) * 1000
    return SearchResponse(
        query=req.query,
        results=[SearchHit(chunk_id=r.chunk_id, doc_id=r.doc_id, title=r.title, text=r.text, score=r.score)
                  for r in results],
        latency_ms=round(latency_ms, 2),
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t0 = time.time()
    translation = STATE["translator"].translate(req.question)
    outcome = answer_question(translation, STATE["db"])
    latency_ms = (time.time() - t0) * 1000
    return AskResponse(
        question=req.question, status=outcome.status, sql=outcome.sql,
        confidence=outcome.confidence, columns=outcome.columns, rows=outcome.rows,
        message=outcome.message, latency_ms=round(latency_ms, 2),
    )
