"""
Measures NL-to-SQL execution accuracy: for in-scope questions, does the
translator's generated SQL execute and return the same result set as the
ground-truth SQL? For out-of-scope questions, does the system correctly
decline (status "clarify"/"error") instead of confidently returning a wrong
answer?

Also reports average response time per question and a confusion breakdown
(answered-correct / answered-wrong / declined-when-should-answer /
declined-correctly / hallucinated-on-out-of-scope).
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from src.nl2sql.translator import TemplateTranslator
from src.nl2sql.validator import answer_question, execute_sql


def results_match(a, b) -> bool:
    def norm(rows):
        return sorted(tuple(r) for r in rows)
    return norm(a) == norm(b)


def main():
    root = Path(__file__).resolve().parents[1]
    db_path = root / "data" / "business.db"
    eval_path = root / "eval" / "nl2sql_eval_set.json"
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    conn = sqlite3.connect(db_path)
    translator = TemplateTranslator()

    with open(eval_path) as f:
        rows = json.load(f)

    counts = {
        "answered_correct": 0, "answered_wrong": 0,
        "declined_in_scope": 0,  # false negative: should have answered
        "declined_out_of_scope_correctly": 0,
        "hallucinated_out_of_scope": 0,
        "error_in_scope": 0,
    }
    latencies = []
    detail = []

    for row in rows:
        q = row["question"]
        t0 = time.time()
        translation = translator.translate(q)
        outcome = answer_question(translation, conn)
        latencies.append(time.time() - t0)

        record = {
            "question": q, "in_scope": row["in_scope"],
            "predicted_intent": translation.intent, "expected_intent": row["expected_intent"],
            "confidence": translation.confidence, "status": outcome.status,
        }

        if row["in_scope"]:
            if outcome.status == "answered":
                ok, expected_result = execute_sql(conn, row["expected_sql"], row["expected_params"])
                exp_cols, exp_rows = expected_result if ok else ([], [])
                if results_match(outcome.rows, exp_rows):
                    counts["answered_correct"] += 1
                    record["result"] = "answered_correct"
                else:
                    counts["answered_wrong"] += 1
                    record["result"] = "answered_wrong"
            elif outcome.status == "clarify":
                counts["declined_in_scope"] += 1
                record["result"] = "declined_in_scope"
            else:
                counts["error_in_scope"] += 1
                record["result"] = "error_in_scope"
        else:
            if outcome.status == "answered":
                counts["hallucinated_out_of_scope"] += 1
                record["result"] = "hallucinated_out_of_scope"
            else:
                counts["declined_out_of_scope_correctly"] += 1
                record["result"] = "declined_out_of_scope_correctly"

        detail.append(record)

    n_in_scope = sum(1 for r in rows if r["in_scope"])
    n_out_of_scope = len(rows) - n_in_scope
    execution_accuracy = counts["answered_correct"] / n_in_scope if n_in_scope else 0.0
    abstention_precision = (
        counts["declined_out_of_scope_correctly"] / n_out_of_scope if n_out_of_scope else 0.0
    )

    report = {
        "n_questions": len(rows),
        "n_in_scope": n_in_scope,
        "n_out_of_scope": n_out_of_scope,
        "execution_accuracy_in_scope": round(execution_accuracy, 4),
        "abstention_correct_rate_out_of_scope": round(abstention_precision, 4),
        "counts": counts,
        "avg_response_time_sec": round(sum(latencies) / len(latencies), 5),
        "p95_response_time_sec": round(sorted(latencies)[int(len(latencies) * 0.95)], 5),
    }

    with open(results_dir / "nl2sql_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(results_dir / "nl2sql_eval_detail.json", "w") as f:
        json.dump(detail, f, indent=2)

    print(json.dumps(report, indent=2))
    conn.close()


if __name__ == "__main__":
    main()
