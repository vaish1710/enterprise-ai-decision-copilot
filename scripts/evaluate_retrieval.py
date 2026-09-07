"""
Evaluates retrieval quality: BM25-only vs dense(vector)-only vs hybrid (RRF)
vs hybrid+reranked, using Precision@5 on the labeled eval set built by
generate_eval_sets.py. A query counts a top-5 hit if the retrieved chunk_id
is in the query's ground-truth relevant_chunk_ids set.

The eval set is split 70/30: the 70% trains the feature reranker, the 30%
(held out, never seen by the reranker) is what all four methods are scored
on -- so the reranker's number isn't inflated by fitting on its own test set.

Results are also broken out by query_type ("lookup" vs "conceptual") to show
*why* hybrid beats either retriever alone: BM25 wins lookup queries, dense
wins conceptual/paraphrased queries, hybrid does well on both.
"""
from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path

from src.retrieval import build_indexes, RetrievedChunk
from src.reranker import FeatureReranker, _features


def precision_at_k(hits: list[RetrievedChunk], relevant_chunk_ids: set, k: int = 5) -> float:
    """
    Hit-rate@k: 1.0 if at least one of the top-k retrieved chunks is relevant,
    else 0.0 -- averaged over queries, this is what most RAG write-ups report
    as "Precision@k" (a query either "found the answer in the top k" or it
    didn't; most eval queries here have exactly one truly relevant document,
    so a strict per-slot precision would be capped at 1/k even for a perfect
    retriever).
    """
    top_k = hits[:k]
    if not top_k:
        return 0.0
    return 1.0 if any(h.chunk_id in relevant_chunk_ids for h in top_k) else 0.0


def build_reranker_training_rows(train_rows, bm25, dense, top_pool=30):
    training = []
    for row in train_rows:
        query = row["query"]
        relevant = set(row["relevant_chunk_ids"])
        bm25_hits = bm25.search(query, top_k=top_pool)
        dense_hits = dense.search(query, top_k=top_pool)
        bm25_scores = {h.chunk_id: h.score for h in bm25_hits}
        dense_scores = {h.chunk_id: h.score for h in dense_hits}
        candidates = {h.chunk_id: h for h in bm25_hits + dense_hits}.values()
        for c in candidates:
            label = 1 if c.chunk_id in relevant else 0
            feats = _features(query, c.text, c.title,
                               bm25_scores.get(c.chunk_id, 0.0),
                               dense_scores.get(c.chunk_id, 0.0))
            training.append({"features": feats.tolist(), "label": label})
    return training


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    eval_path = root / "eval" / "retrieval_eval_set.json"
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    print("Loading corpus and building indexes (BM25 + LSA dense)...")
    t0 = time.time()
    chunks, bm25, dense, hybrid = build_indexes(data_dir)
    print(f"  {len(chunks)} chunks indexed in {time.time()-t0:.1f}s")

    with open(eval_path) as f:
        eval_rows = json.load(f)

    rng = random.Random(11)
    rng.shuffle(eval_rows)
    split = int(len(eval_rows) * 0.7)
    train_rows, test_rows = eval_rows[:split], eval_rows[split:]

    print(f"Training reranker on {len(train_rows)} queries, evaluating on held-out {len(test_rows)}...")
    training_data = build_reranker_training_rows(train_rows, bm25, dense)
    reranker = FeatureReranker().fit(training_data)
    reranker.save(root / "results" / "reranker.pkl")

    methods = {"bm25_only": [], "dense_only": [], "hybrid_rrf": [], "hybrid_reranked": []}
    by_type = defaultdict(lambda: defaultdict(list))
    latencies = {"hybrid_rrf": [], "hybrid_reranked": []}

    for row in test_rows:
        query = row["query"]
        relevant = set(row["relevant_chunk_ids"])
        qtype = row["query_type"]

        t0 = time.time()
        bm25_hits = bm25.search(query, top_k=5)
        p = precision_at_k(bm25_hits, relevant)
        methods["bm25_only"].append(p); by_type[qtype]["bm25_only"].append(p)

        dense_hits = dense.search(query, top_k=5)
        p = precision_at_k(dense_hits, relevant)
        methods["dense_only"].append(p); by_type[qtype]["dense_only"].append(p)

        pool_bm25 = bm25.search(query, top_k=30)
        pool_dense = dense.search(query, top_k=30)
        hybrid_hits = hybrid.search(query, top_k=5, candidate_pool=30)
        t_hybrid = time.time() - t0
        latencies["hybrid_rrf"].append(t_hybrid)
        p = precision_at_k(hybrid_hits, relevant)
        methods["hybrid_rrf"].append(p); by_type[qtype]["hybrid_rrf"].append(p)

        t1 = time.time()
        hybrid_pool = hybrid.search(query, top_k=30, candidate_pool=30)
        bm25_scores = {h.chunk_id: h.score for h in pool_bm25}
        dense_scores = {h.chunk_id: h.score for h in pool_dense}
        reranked = reranker.score(query, hybrid_pool, bm25_scores, dense_scores)
        t_reranked = t_hybrid + (time.time() - t1)
        latencies["hybrid_reranked"].append(t_reranked)
        p = precision_at_k(reranked, relevant, k=5)
        methods["hybrid_reranked"].append(p); by_type[qtype]["hybrid_reranked"].append(p)

    summary = {name: sum(vals) / len(vals) for name, vals in methods.items()}
    summary_by_type = {
        qtype: {name: round(sum(vals) / len(vals), 4) for name, vals in m.items()}
        for qtype, m in by_type.items()
    }
    improvement_vs_vector_only = (
        (summary["hybrid_reranked"] - summary["dense_only"]) / summary["dense_only"] * 100
        if summary["dense_only"] > 0 else float("inf")
    )

    report = {
        "n_test_queries": len(test_rows),
        "n_chunks_indexed": len(chunks),
        "precision_at_5": {k: round(v, 4) for k, v in summary.items()},
        "precision_at_5_by_query_type": summary_by_type,
        "improvement_hybrid_reranked_vs_vector_only_pct": round(improvement_vs_vector_only, 1),
        "avg_latency_sec": {k: round(sum(v) / len(v), 4) for k, v in latencies.items()},
    }

    with open(results_dir / "retrieval_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
