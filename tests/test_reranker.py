from pathlib import Path

from src.retrieval import build_indexes
from src.reranker import FeatureReranker

ROOT = Path(__file__).resolve().parents[1]


def test_reranker_fallback_scores_without_training():
    chunks, bm25, dense, hybrid = build_indexes(ROOT / "data")
    query = "expense reimbursement policy"
    candidates = hybrid.search(query, top_k=10, candidate_pool=30)
    bm25_scores = {h.chunk_id: h.score for h in bm25.search(query, top_k=30)}
    dense_scores = {h.chunk_id: h.score for h in dense.search(query, top_k=30)}

    reranker = FeatureReranker()  # untrained -> uses fallback linear weights
    reranked = reranker.score(query, candidates, bm25_scores, dense_scores)

    assert len(reranked) == len(candidates)
    scores = [r.score for r in reranked]
    assert scores == sorted(scores, reverse=True)


def test_reranker_can_fit_on_labeled_rows():
    training_rows = [
        {"features": [0.9, 0.8, 5, 0.5, 0.3, 1.0, 0.4], "label": 1},
        {"features": [0.1, 0.1, 0, 0.0, 0.0, 0.0, 0.2], "label": 0},
        {"features": [0.85, 0.7, 4, 0.4, 0.2, 1.0, 0.5], "label": 1},
        {"features": [0.05, 0.2, 1, 0.05, 0.0, 0.0, 0.1], "label": 0},
    ]
    reranker = FeatureReranker().fit(training_rows)
    assert reranker.model is not None
