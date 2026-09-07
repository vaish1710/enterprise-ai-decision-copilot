import sqlite3
from pathlib import Path

import pytest

from src.nl2sql.translator import TemplateTranslator
from src.nl2sql.validator import answer_question, execute_sql

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def conn():
    c = sqlite3.connect(ROOT / "data" / "business.db")
    yield c
    c.close()


@pytest.fixture(scope="module")
def translator():
    return TemplateTranslator()


def test_total_revenue_executes_and_returns_one_row(translator, conn):
    t = translator.translate("What was total revenue in North America?")
    assert t.intent == "total_revenue"
    outcome = answer_question(t, conn)
    assert outcome.status == "answered"
    assert len(outcome.rows) == 1
    assert outcome.rows[0][0] is not None


def test_top_products_respects_top_n_and_category(translator, conn):
    t = translator.translate("What are the top 3 products by revenue in Electronics?")
    assert t.intent == "top_products"
    outcome = answer_question(t, conn)
    assert outcome.status == "answered"
    assert len(outcome.rows) <= 3
    assert all(row[1] == "Electronics" for row in outcome.rows)


def test_customer_count_vs_top_customers_disambiguation(translator, conn):
    # regression test: these two used to collide because both templates
    # required only the word "customer" -- see translator.py's required_groups.
    count_q = translator.translate("How many Enterprise customers do we have in North America?")
    assert count_q.intent == "customer_count"

    top_q = translator.translate("Who are the top 5 customers by spend in North America?")
    assert top_q.intent == "top_customers"


def test_out_of_scope_question_is_not_confidently_answered(translator, conn):
    t = translator.translate("What's the weather like in the APAC region?")
    outcome = answer_question(t, conn)
    assert outcome.status in ("clarify", "error")


def test_non_select_sql_is_refused(conn):
    ok, msg = execute_sql(conn, "DELETE FROM customers", [])
    assert ok is False
    assert "select" in msg.lower() or "refus" in msg.lower()


def test_low_confidence_falls_back_to_clarify(translator, conn):
    t = translator.translate("asdkfj alksdjf laksjdf")
    outcome = answer_question(t, conn)
    assert outcome.status == "clarify"
