"""
Template-based natural-language-to-SQL translator.

No LLM API key is available in the sandboxed environment this repo was built
in, so the default backend here is a deterministic, fully-offline semantic
parser: it classifies the question into one of a fixed set of business
"intents" (total revenue, top-N products, order counts, ...), extracts slot
values (region/category/segment/status/year/top_n) by scanning for known
schema values and light regex, and renders a parameterized SQL template.
Every generated query is parameterized (no string-built literals from user
text go directly into SQL), so this is not vulnerable to SQL injection the
way a naive "glue the question into a prompt and trust the LLM" pipeline can
be if the LLM's output isn't validated downstream.

`LLMTranslator` documents the drop-in swap for a real LLM-backed translator
(OpenAI/Anthropic) for open-ended questions the template set doesn't cover --
see validator.py for how confidence-based fallback between the two would work.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from .schema import REGIONS, SEGMENTS, CATEGORIES, ORDER_STATUSES, SCHEMA_DESCRIPTION

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20,
}


def _find_value(question: str, values: list[str]) -> Optional[str]:
    q = question.lower()
    # match longest values first so "North America" wins over a shorter substring
    for v in sorted(values, key=len, reverse=True):
        if v.lower() in q:
            return v
    return None


def _find_year(question: str) -> Optional[int]:
    m = re.search(r"\b(20\d{2})\b", question)
    return int(m.group(1)) if m else None


def _find_top_n(question: str) -> Optional[int]:
    m = re.search(r"\btop\s+(\d+)\b", question.lower())
    if m:
        return int(m.group(1))
    m = re.search(r"\btop\s+(\w+)\b", question.lower())
    if m and m.group(1) in NUMBER_WORDS:
        return NUMBER_WORDS[m.group(1)]
    return None


@dataclass
class Slots:
    region: Optional[str] = None
    category: Optional[str] = None
    segment: Optional[str] = None
    status: Optional[str] = None
    year: Optional[int] = None
    top_n: Optional[int] = None

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


def extract_slots(question: str) -> Slots:
    return Slots(
        region=_find_value(question, REGIONS),
        category=_find_value(question, CATEGORIES),
        segment=_find_value(question, SEGMENTS),
        status=_find_value(question, ORDER_STATUSES),
        year=_find_year(question),
        top_n=_find_top_n(question),
    )


@dataclass
class Intent:
    name: str
    keywords: list[str]                       # any of these boosts intent score
    required_groups: list[set[str]]            # AND across groups, OR within a group
    build_sql: Callable[[Slots], str]
    uses_slots: list[str] = field(default_factory=list)  # slots this intent can use

    def required_satisfied(self, q: str) -> bool:
        return all(any(phrase in q for phrase in group) for group in self.required_groups)


# A query has to look like a request for information before any intent is
# even considered -- this keeps domain nouns that show up in unrelated
# sentences ("...APAC region", "...best-selling product" in a creative-writing
# ask) from accidentally triggering a confident SQL answer.
_INTERROGATIVE_MARKERS = [
    "what", "how many", "how much", "which", "who", "show", "give me",
    "list", "number of", "count of", "average", "total", "top", "pull up",
]


def _looks_like_a_question(q: str) -> bool:
    return any(m in q for m in _INTERROGATIVE_MARKERS)


def _where_orders(slots: Slots, alias: str = "o") -> tuple[str, list]:
    clauses, params = [], []
    if slots.region:
        clauses.append(f"{alias}.region = ?"); params.append(slots.region)
    if slots.status:
        clauses.append(f"{alias}.status = ?"); params.append(slots.status)
    else:
        clauses.append(f"{alias}.status = 'completed'")
    if slots.year:
        clauses.append(f"strftime('%Y', {alias}.order_date) = ?"); params.append(str(slots.year))
    return (" AND ".join(clauses) if clauses else "1=1"), params


def _sql_total_revenue(slots: Slots) -> tuple[str, list]:
    where, params = _where_orders(slots)
    extra_join, extra_where = "", ""
    if slots.category:
        extra_join = "JOIN products p ON p.product_id = oi.product_id"
        extra_where = " AND p.category = ?"
        params.append(slots.category)
    sql = (
        "SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_revenue "
        "FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
        f"{extra_join} WHERE {where}{extra_where}"
    )
    return sql, params


def _sql_revenue_by_region(slots: Slots) -> tuple[str, list]:
    where, params = _where_orders(slots)
    sql = (
        "SELECT o.region, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
        "FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
        f"WHERE {where} GROUP BY o.region ORDER BY revenue DESC"
    )
    return sql, params


def _sql_revenue_by_category(slots: Slots) -> tuple[str, list]:
    where, params = _where_orders(slots)
    sql = (
        "SELECT p.category, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
        "FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
        "JOIN products p ON p.product_id = oi.product_id "
        f"WHERE {where} GROUP BY p.category ORDER BY revenue DESC"
    )
    return sql, params


def _sql_top_products(slots: Slots) -> tuple[str, list]:
    where, params = _where_orders(slots)
    extra_where = ""
    if slots.category:
        extra_where = " AND p.category = ?"
        params.append(slots.category)
    limit = slots.top_n or 10
    sql = (
        "SELECT p.name, p.category, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
        "FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
        "JOIN products p ON p.product_id = oi.product_id "
        f"WHERE {where}{extra_where} GROUP BY p.product_id ORDER BY revenue DESC LIMIT {limit}"
    )
    return sql, params


def _sql_top_customers(slots: Slots) -> tuple[str, list]:
    where, params = _where_orders(slots)
    limit = slots.top_n or 10
    sql = (
        "SELECT c.name, c.segment, c.region, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS spend "
        "FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
        "JOIN customers c ON c.customer_id = o.customer_id "
        f"WHERE {where} GROUP BY c.customer_id ORDER BY spend DESC LIMIT {limit}"
    )
    return sql, params


def _sql_order_count(slots: Slots) -> tuple[str, list]:
    where, params = _where_orders(slots)
    sql = f"SELECT COUNT(*) AS order_count FROM orders o WHERE {where}"
    return sql, params


def _sql_avg_order_value(slots: Slots) -> tuple[str, list]:
    where, params = _where_orders(slots)
    join_seg = ""
    group_by = ""
    select_seg = ""
    if slots.segment:
        join_seg = "JOIN customers c ON c.customer_id = o.customer_id"
        params_extra = slots.segment
    sql_order_totals = (
        "SELECT o.order_id, o.region, o.customer_id, SUM(oi.quantity * oi.unit_price) AS order_total "
        "FROM order_items oi JOIN orders o ON o.order_id = oi.order_id "
        f"WHERE {where} GROUP BY o.order_id"
    )
    if slots.segment:
        sql = (
            "SELECT ROUND(AVG(t.order_total), 2) AS avg_order_value FROM "
            f"({sql_order_totals}) t JOIN customers c ON c.customer_id = t.customer_id "
            "WHERE c.segment = ?"
        )
        params = params + [slots.segment]
    else:
        sql = f"SELECT ROUND(AVG(t.order_total), 2) AS avg_order_value FROM ({sql_order_totals}) t"
    return sql, params


def _sql_customer_count(slots: Slots) -> tuple[str, list]:
    clauses, params = [], []
    if slots.segment:
        clauses.append("segment = ?"); params.append(slots.segment)
    if slots.region:
        clauses.append("region = ?"); params.append(slots.region)
    where = " AND ".join(clauses) if clauses else "1=1"
    sql = f"SELECT COUNT(*) AS customer_count FROM customers WHERE {where}"
    return sql, params


def _sql_product_count(slots: Slots) -> tuple[str, list]:
    clauses, params = [], []
    if slots.category:
        clauses.append("category = ?"); params.append(slots.category)
    where = " AND ".join(clauses) if clauses else "1=1"
    sql = f"SELECT COUNT(*) AS product_count FROM products WHERE {where}"
    return sql, params


INTENTS: list[Intent] = [
    Intent(
        "top_products",
        ["best-selling", "best selling", "highest revenue product", "products by revenue", "made the most"],
        [{"product"}, {"top", "best-selling", "best selling", "highest revenue", "most revenue", "made the most"}],
        lambda s: _sql_top_products(s), ["region", "category", "year", "top_n", "status"],
    ),
    Intent(
        "top_customers",
        ["top customer", "highest spend", "biggest customer", "customers by spend"],
        [{"customer"}, {"top", "highest spend", "biggest", "spent the most", "most spend"}],
        lambda s: _sql_top_customers(s), ["region", "year", "top_n", "status"],
    ),
    Intent(
        "revenue_by_region",
        ["revenue by region", "revenue per region", "sales by region"],
        [{"revenue", "sales"}, {"region"}, {"by region", "per region", "region breakdown"}],
        lambda s: _sql_revenue_by_region(s), ["year", "status"],
    ),
    Intent(
        "revenue_by_category",
        ["revenue by category", "sales by category", "revenue per category"],
        [{"revenue", "sales"}, {"category"}, {"by category", "per category", "category breakdown"}],
        lambda s: _sql_revenue_by_category(s), ["year", "status"],
    ),
    Intent(
        "total_revenue",
        ["total revenue", "how much revenue", "total sales", "sales figure"],
        [{"revenue", "sales"}],
        lambda s: _sql_total_revenue(s), ["region", "category", "year", "status"],
    ),
    Intent(
        "order_count",
        ["how many orders", "number of orders", "order count"],
        [{"order"}, {"how many", "number of", "count"}],
        lambda s: _sql_order_count(s), ["region", "status", "year"],
    ),
    Intent(
        "avg_order_value",
        ["average order value", "avg order value", "mean order value"],
        [{"order value"}],
        lambda s: _sql_avg_order_value(s), ["region", "segment", "year", "status"],
    ),
    Intent(
        "customer_count",
        ["how many customers", "number of customers", "customer count"],
        [{"customer"}, {"how many", "number of", "count"}],
        lambda s: _sql_customer_count(s), ["segment", "region"],
    ),
    Intent(
        "product_count",
        ["how many products", "number of products", "product count"],
        [{"product"}, {"how many", "number of", "count"}],
        lambda s: _sql_product_count(s), ["category"],
    ),
]


@dataclass
class TranslationResult:
    question: str
    intent: Optional[str]
    sql: Optional[str]
    params: list
    confidence: float
    slots: dict
    notes: str = ""


def classify_intent(question: str) -> tuple[Optional[Intent], float]:
    q = question.lower()
    if not _looks_like_a_question(q):
        return None, 0.0

    candidates = []
    for intent in INTENTS:
        if not intent.required_satisfied(q):
            continue
        # Required-group specificity: an intent whose required groups spell
        # out more distinct concepts (e.g. "customer" AND "how many") is a
        # more specific match than one satisfied by a single generic group,
        # so it should win ties instead of whichever intent happens to be
        # listed first.
        specificity = len(intent.required_groups)
        bonus = sum(1 for kw in intent.keywords if kw in q)
        score = specificity * 10 + bonus
        candidates.append((score, intent))

    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda c: -c[0])
    best_score, best = candidates[0]
    # If two+ intents are equally specific and tied on bonus keywords, we
    # genuinely can't disambiguate -- that should show up as lower
    # confidence rather than an arbitrary pick.
    tied = sum(1 for s, _ in candidates if s == best_score)
    max_possible = len(best.keywords) + 1 + best_score
    conf = min(1.0, (best_score + sum(1 for kw in best.keywords if kw in q)) / max_possible + 0.5)
    if tied > 1:
        conf *= 0.6
    return best, conf


class TemplateTranslator:
    """Deterministic, offline NL->SQL translator (default backend)."""

    def translate(self, question: str) -> TranslationResult:
        intent, intent_conf = classify_intent(question)
        slots = extract_slots(question)

        if intent is None:
            return TranslationResult(
                question=question, intent=None, sql=None, params=[],
                confidence=0.0, slots=slots.as_dict(),
                notes="No matching intent template found.",
            )

        sql, params = intent.build_sql(slots)

        # Confidence blends how well the question matched the intent's
        # trigger phrases with how many of the intent's *usable* slots we
        # were actually able to resolve from the question text.
        usable = [s for s in intent.uses_slots if s not in ("top_n", "status")]
        resolved = sum(1 for s in usable if getattr(slots, s) is not None)
        slot_conf = 0.5 + 0.5 * (resolved / len(usable)) if usable else 1.0
        confidence = round(0.6 * intent_conf + 0.4 * slot_conf, 3)

        return TranslationResult(
            question=question, intent=intent.name, sql=sql, params=params,
            confidence=confidence, slots=slots.as_dict(),
        )


class LLMTranslator:
    """
    Documents the swap-in point for an OpenAI/Anthropic-backed translator that
    handles open-ended questions outside the template set. Requires
    `OPENAI_API_KEY` and outbound network access, neither available in the
    sandboxed environment this repo was authored in, so this is not exercised
    by the test suite -- `TemplateTranslator` is the default, fully-offline
    path that the eval numbers in results/nl2sql_metrics.json reflect.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def translate(self, question: str) -> TranslationResult:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("pip install openai to use LLMTranslator") from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY to use LLMTranslator")
        client = OpenAI(api_key=api_key)
        prompt = (
            "You translate business questions into a single read-only SQLite "
            f"SELECT query against this schema:\n{SCHEMA_DESCRIPTION}\n\n"
            "Return ONLY the SQL, no commentary.\n\nQuestion: " + question
        )
        resp = client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], temperature=0,
        )
        sql = resp.choices[0].message.content.strip().strip("`").strip()
        return TranslationResult(
            question=question, intent="llm", sql=sql, params=[],
            confidence=0.75, slots={}, notes="Generated by LLMTranslator; validate before executing.",
        )
