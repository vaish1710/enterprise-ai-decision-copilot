"""
Builds a labeled retrieval evaluation set from the synthetic corpus.

Because the corpus generator embeds a unique, findable "Document Reference"
fact (case_id / owner / reviewed_date) in exactly one chunk per document, and
also knows each document's (department, product, region, topic, doc_type)
combination, we can construct two kinds of queries with known ground truth
without a human labeler:

  - "lookup" queries: reference a case_id or owner name verbatim (the kind of
    exact-match query BM25 is good at) -> ground truth is the single anchor
    chunk that contains it.
  - "conceptual" queries: paraphrase the document's topic using a synonym
    substitution table, combined with department/product/region so the combo
    is (almost always) unique to one document, with no verbatim overlap with
    the source text -> ground truth is any chunk of that document. This is
    the kind of query dense/semantic retrieval is supposed to be good at.

Mixing both is what makes "hybrid beats either alone" a real, measured result
here rather than an assumed one.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from src.corpus import TOPIC_SYNONYMS

RANDOM_SEED = 7

LOOKUP_TEMPLATES = [
    "What's the status of case {case_id}?",
    "Who owns case reference {case_id}?",
    "When was {case_id} last reviewed?",
    "Pull up the document for case {case_id}.",
    "I need details on reference {case_id}, who should I contact?",
]

OWNER_LOOKUP_TEMPLATES = [
    "What cases does {owner} currently own?",
    "Which document was {owner} last reviewing?",
]

CONCEPTUAL_TEMPLATES = [
    "Within {dept}, {paraphrase} for {product} -- what's the current process?",
    "For teams in {region}, how do we handle {paraphrase}?",
    "What's {dept}'s latest thinking on {paraphrase}, especially for {product}?",
    "Is there guidance on {paraphrase} that {dept} put together?",
    "How does {product} deal with {paraphrase} today?",
]


def build_eval_set(data_dir: Path, n_lookup: int = 130, n_conceptual: int = 110) -> list[dict]:
    docs = {}
    with open(data_dir / "documents.jsonl") as f:
        for line in f:
            d = json.loads(line)
            docs[d["doc_id"]] = d

    with open(data_dir / "anchor_chunks.json") as f:
        anchor_map = json.load(f)  # doc_id -> chunk_id

    chunks_by_doc = defaultdict(list)
    with open(data_dir / "chunks.jsonl") as f:
        for line in f:
            c = json.loads(line)
            chunks_by_doc[c["doc_id"]].append(c["chunk_id"])

    rng = random.Random(RANDOM_SEED)
    doc_ids = list(docs.keys())
    rng.shuffle(doc_ids)

    eval_rows = []

    # -- Lookup queries (favor exact/lexical match) --
    for doc_id in doc_ids[:n_lookup]:
        d = docs[doc_id]
        anchor_chunk = anchor_map.get(doc_id)
        if not anchor_chunk:
            continue
        use_owner = rng.random() < 0.25
        if use_owner:
            template = rng.choice(OWNER_LOOKUP_TEMPLATES)
            query = template.format(owner=d["owner"])
        else:
            template = rng.choice(LOOKUP_TEMPLATES)
            query = template.format(case_id=d["case_id"])
        eval_rows.append({
            "query": query,
            "query_type": "lookup",
            "relevant_doc_id": doc_id,
            "relevant_chunk_ids": [anchor_chunk],
        })

    # -- Conceptual queries (favor paraphrase/semantic match) --
    remaining = doc_ids[n_lookup:n_lookup + n_conceptual]
    for doc_id in remaining:
        d = docs[doc_id]
        topic = d["topic"]
        paraphrase = TOPIC_SYNONYMS.get(topic, topic)
        template = rng.choice(CONCEPTUAL_TEMPLATES)
        query = template.format(
            dept=d["department"],
            product=d["product"] or "our core platform",
            region=d["region"] or "all regions",
            paraphrase=paraphrase,
        )
        eval_rows.append({
            "query": query,
            "query_type": "conceptual",
            "relevant_doc_id": doc_id,
            "relevant_chunk_ids": chunks_by_doc[doc_id],
        })

    rng.shuffle(eval_rows)
    return eval_rows


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    eval_dir = root / "eval"
    eval_dir.mkdir(exist_ok=True)

    rows = build_eval_set(data_dir)
    with open(eval_dir / "retrieval_eval_set.json", "w") as f:
        json.dump(rows, f, indent=2)
    n_lookup = sum(1 for r in rows if r["query_type"] == "lookup")
    n_conceptual = len(rows) - n_lookup
    print(f"Wrote {len(rows)} retrieval eval queries "
          f"({n_lookup} lookup, {n_conceptual} conceptual) -> eval/retrieval_eval_set.json")


if __name__ == "__main__":
    main()
