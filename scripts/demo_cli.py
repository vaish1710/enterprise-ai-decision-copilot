"""
Interactive command-line demo -- no server needed.

    python3 scripts/demo_cli.py search "vendor onboarding process"
    python3 scripts/demo_cli.py ask "How many orders were placed in EMEA in 2025?"
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from src.retrieval import build_indexes
from src.reranker import FeatureReranker
from src.nl2sql.translator import TemplateTranslator
from src.nl2sql.validator import answer_question

ROOT = Path(__file__).resolve().parents[1]


def cmd_search(query: str, top_k: int = 5):
    chunks, bm25, dense, hybrid = build_indexes(ROOT / "data")
    reranker_path = ROOT / "results" / "reranker.pkl"
    reranker = FeatureReranker.load(reranker_path) if reranker_path.exists() else FeatureReranker()

    pool = hybrid.search(query, top_k=30, candidate_pool=30)
    bm25_scores = {h.chunk_id: h.score for h in bm25.search(query, top_k=30)}
    dense_scores = {h.chunk_id: h.score for h in dense.search(query, top_k=30)}
    results = reranker.score(query, pool, bm25_scores, dense_scores)[:top_k]

    print(f"\nQuery: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r.title}  (score={r.score:.3f}, chunk={r.chunk_id})")
        snippet = r.text.replace("\n", " ")[:220]
        print(f"    {snippet}...\n")


def cmd_ask(question: str):
    conn = sqlite3.connect(ROOT / "data" / "business.db")
    translator = TemplateTranslator()
    translation = translator.translate(question)
    outcome = answer_question(translation, conn)

    print(f"\nQuestion: {question}")
    print(f"Status: {outcome.status}  (confidence={outcome.confidence:.2f})")
    if outcome.sql:
        print(f"SQL: {outcome.sql}\nParams: {translation.params}")
    if outcome.status == "answered":
        print(f"Columns: {outcome.columns}")
        for row in outcome.rows[:10]:
            print(f"  {row}")
        if len(outcome.rows) > 10:
            print(f"  ... ({len(outcome.rows) - 10} more rows)")
    else:
        print(f"Message: {outcome.message}")
    conn.close()


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("search", "ask"):
        print(__doc__)
        sys.exit(1)
    cmd, text = sys.argv[1], " ".join(sys.argv[2:])
    if cmd == "search":
        cmd_search(text)
    else:
        cmd_ask(text)


if __name__ == "__main__":
    main()
