"""
One-command reproduction of every generated artifact + metric in this repo:

    python3 scripts/run_pipeline.py

Regenerates the synthetic document corpus, the synthetic business SQLite DB,
both labeled eval sets, then runs both evaluation scripts. Everything is
seeded, so results are stable run to run. Takes well under a minute.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ("Generating synthetic document corpus", [sys.executable, "src/corpus.py"]),
    ("Building synthetic business SQLite DB", [sys.executable, "-m", "src.nl2sql.schema"]),
    ("Building retrieval eval set", [sys.executable, "scripts/generate_eval_sets.py"]),
    ("Building NL2SQL eval set", [sys.executable, "scripts/generate_nl2sql_eval.py"]),
    ("Evaluating retrieval (BM25 / dense / hybrid / reranked)", [sys.executable, "scripts/evaluate_retrieval.py"]),
    ("Evaluating NL2SQL execution accuracy", [sys.executable, "scripts/evaluate_nl2sql.py"]),
]


def main():
    for label, cmd in STEPS:
        print(f"\n=== {label} ===")
        t0 = time.time()
        result = subprocess.run(cmd, cwd=ROOT, env={"PYTHONPATH": str(ROOT), **_env()})
        if result.returncode != 0:
            print(f"FAILED: {label}")
            sys.exit(result.returncode)
        print(f"  done in {time.time()-t0:.1f}s")

    print("\nAll done. See results/retrieval_metrics.json and results/nl2sql_metrics.json.")


def _env():
    import os
    return dict(os.environ)


if __name__ == "__main__":
    main()
