"""
Executes and validates translator output against the SQLite database, with
confidence-based fallback: low-confidence or failed-execution translations
never get silently returned as an answer -- the caller gets a clarification
request instead. This is the "validation with confidence-based fallback"
half of the NL-to-SQL pipeline.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .translator import TranslationResult

READ_ONLY_PREFIXES = ("select",)
CONFIDENCE_THRESHOLD = 0.55


@dataclass
class QueryOutcome:
    question: str
    sql: Optional[str]
    confidence: float
    status: str  # "answered", "clarify", "error"
    rows: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    message: str = ""


def _is_read_only(sql: str) -> bool:
    return sql.strip().lower().startswith(READ_ONLY_PREFIXES)


def execute_sql(conn: sqlite3.Connection, sql: str, params: list) -> tuple[bool, Any]:
    if not _is_read_only(sql):
        return False, "Refusing to execute a non-SELECT statement."
    try:
        cur = conn.execute(sql, params)
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall()
        return True, (cols, rows)
    except sqlite3.Error as e:
        return False, str(e)


def answer_question(
    translation: TranslationResult,
    conn: sqlite3.Connection,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> QueryOutcome:
    if translation.sql is None or translation.confidence < confidence_threshold:
        return QueryOutcome(
            question=translation.question, sql=translation.sql,
            confidence=translation.confidence, status="clarify",
            message=(
                "I'm not confident enough in a generated query for this question "
                f"(confidence={translation.confidence:.2f}, threshold={confidence_threshold}). "
                "Could you rephrase it, e.g. naming a region, category, segment, or year explicitly?"
            ),
        )

    ok, result = execute_sql(conn, translation.sql, translation.params)
    if not ok:
        return QueryOutcome(
            question=translation.question, sql=translation.sql,
            confidence=translation.confidence, status="error",
            message=f"Generated SQL failed validation/execution: {result}",
        )

    cols, rows = result
    return QueryOutcome(
        question=translation.question, sql=translation.sql,
        confidence=translation.confidence, status="answered",
        rows=[list(r) for r in rows], columns=cols,
    )
