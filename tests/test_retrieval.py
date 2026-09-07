from pathlib import Path

from src.retrieval import build_indexes, BM25Retriever, DenseRetriever, HybridRetriever

ROOT = Path(__file__).resolve().parents[1]


def test_indexes_build_and_return_hits():
    chunks, bm25, dense, hybrid = build_indexes(ROOT / "data")
    assert len(chunks) > 1000

    hits = bm25.search("case reference", top_k=5)
    assert len(hits) == 5
    assert all(h.source == "bm25" for h in hits)

    hits = dense.search("case reference", top_k=5)
    assert len(hits) == 5
    assert all(h.source == "dense" for h in hits)

    hits = hybrid.search("case reference", top_k=5)
    assert len(hits) == 5
    assert all(h.source == "hybrid" for h in hits)


def test_bm25_finds_exact_case_id():
    chunks, bm25, dense, hybrid = build_indexes(ROOT / "data")
    # pick a real case_id straight out of the indexed chunks
    sample = next(c for c in chunks if "case reference" in c["text"])
    import re
    m = re.search(r"case reference ([A-Z]{2,4}-\d+)", sample["text"])
    assert m, "fixture chunk should contain a case reference"
    case_id = m.group(1)

    hits = bm25.search(f"What's the status of case {case_id}?", top_k=5)
    assert any(case_id in h.text for h in hits), "BM25 should retrieve the exact case_id chunk in top 5"


def test_hybrid_returns_ranked_unique_chunks():
    chunks, bm25, dense, hybrid = build_indexes(ROOT / "data")
    hits = hybrid.search("customer refund approval process", top_k=10)
    ids = [h.chunk_id for h in hits]
    assert len(ids) == len(set(ids)), "hybrid results should not contain duplicate chunks"
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), "results should be sorted by fused score"
