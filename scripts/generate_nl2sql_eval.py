"""
Builds a labeled NL-to-SQL evaluation set: for each question we already know
the ground-truth (intent, slots) it was generated from, so the ground-truth
SQL is just that intent's builder function called on those slots -- no need
to hand-author 150 SQL strings. "Correct" means the translator's SQL executes
and returns the *same result set* as the ground-truth SQL, not just that it
executes without error.

Includes a mix of:
  - directly-templated questions (use the translator's own trigger phrasing)
  - paraphrased questions (synonyms the translator was NOT built around, to
    honestly test generalization rather than only testing template recall)
  - out-of-scope questions with no ground-truth SQL, where the "correct"
    outcome is the system declining to answer rather than guessing
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.nl2sql.schema import REGIONS, SEGMENTS, CATEGORIES, ORDER_STATUSES
from src.nl2sql.translator import Slots, _sql_total_revenue, _sql_revenue_by_region, \
    _sql_revenue_by_category, _sql_top_products, _sql_top_customers, _sql_order_count, \
    _sql_avg_order_value, _sql_customer_count, _sql_product_count

RANDOM_SEED = 21

# (intent_name, builder, question templates -- {slot} placeholders)
SPECS = [
    ("total_revenue", _sql_total_revenue, [
        "What was total revenue in {region}?",
        "How much revenue did we make from {category} in {year}?",
        "What's our total sales figure for {year}?",
        "Total revenue, {region}, {year}?",
    ]),
    ("revenue_by_region", _sql_revenue_by_region, [
        "Show revenue by region for {year}.",
        "What's the sales by region breakdown?",
        "Give me revenue per region in {year}.",
    ]),
    ("revenue_by_category", _sql_revenue_by_category, [
        "Show revenue by category for {year}.",
        "What's the sales by category breakdown in {year}?",
        "Revenue per category, please.",
    ]),
    ("top_products", _sql_top_products, [
        "What are the top {top_n} products by revenue in {category}?",
        "Show me the top {top_n} best-selling products in {region}.",
        "Which products made the most revenue in {year}?",
    ]),
    ("top_customers", _sql_top_customers, [
        "Who are the top {top_n} customers by spend in {region}?",
        "Show the top {top_n} highest spend customers.",
        "Which customers spent the most in {year}?",
    ]),
    ("order_count", _sql_order_count, [
        "How many orders were placed in {region}?",
        "Number of {status} orders in {year}?",
        "How many orders came from {region} in {year}?",
    ]),
    ("avg_order_value", _sql_avg_order_value, [
        "What's the average order value in {region}?",
        "Average order value for {segment} customers?",
        "What is the mean order value in {year}?",
    ]),
    ("customer_count", _sql_customer_count, [
        "How many customers are in the {segment} segment?",
        "Number of customers in {region}?",
        "How many {segment} customers do we have in {region}?",
    ]),
    ("product_count", _sql_product_count, [
        "How many products are in {category}?",
        "Number of products in the {category} category?",
    ]),
]

OUT_OF_SCOPE = [
    "What's the weather like in the APAC region?",
    "Can you recommend a good marketing slogan?",
    "Who is our CEO?",
    "What's the meaning of life?",
    "Draft an email apologizing for a late shipment.",
    "What's our stock price today?",
    "Summarize our HR policy on parental leave.",
    "How do I reset my VPN password?",
    "What's the capital of France?",
    "Write a haiku about our best-selling product.",
]


import re

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def build_eval_set(n_per_intent: int = 16, seed: int = RANDOM_SEED) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for intent_name, builder, templates in SPECS:
        for _ in range(n_per_intent):
            template = rng.choice(templates)
            placeholders = set(_PLACEHOLDER_RE.findall(template))

            # Generate a concrete value ONLY for placeholders this specific
            # template actually references, so the question text never
            # contains a slot value the "ground truth" doesn't know about
            # (that mismatch was silently corrupting expected SQL below).
            fill_values = {}
            mentioned = Slots()
            if "region" in placeholders:
                v = rng.choice(REGIONS); fill_values["region"] = v; mentioned.region = v
            if "category" in placeholders:
                v = rng.choice(CATEGORIES); fill_values["category"] = v; mentioned.category = v
            if "segment" in placeholders:
                v = rng.choice(SEGMENTS); fill_values["segment"] = v; mentioned.segment = v
            if "status" in placeholders:
                v = rng.choice(ORDER_STATUSES); fill_values["status"] = v; mentioned.status = v
            if "year" in placeholders:
                v = rng.choice([2023, 2024, 2025, 2026]); fill_values["year"] = v; mentioned.year = v
            if "top_n" in placeholders:
                v = rng.choice([3, 5, 10]); fill_values["top_n"] = v; mentioned.top_n = v

            question = template.format(**fill_values)
            sql, params = builder(mentioned)
            rows.append({
                "question": question,
                "expected_intent": intent_name,
                "expected_sql": sql,
                "expected_params": params,
                "in_scope": True,
            })

    for q in OUT_OF_SCOPE:
        rows.append({
            "question": q, "expected_intent": None, "expected_sql": None,
            "expected_params": [], "in_scope": False,
        })

    rng.shuffle(rows)
    return rows


def main():
    root = Path(__file__).resolve().parents[1]
    eval_dir = root / "eval"
    eval_dir.mkdir(exist_ok=True)
    rows = build_eval_set()
    with open(eval_dir / "nl2sql_eval_set.json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} NL2SQL eval questions "
          f"({sum(r['in_scope'] for r in rows)} in-scope, "
          f"{sum(not r['in_scope'] for r in rows)} out-of-scope) -> eval/nl2sql_eval_set.json")


if __name__ == "__main__":
    main()
