# Enterprise AI Decision Intelligence Copilot

[Portfolio](https://vaishnavi-paruchuri.vercel.app/) · [GitHub](https://github.com/vaish1710) · [LinkedIn](https://linkedin.com/in/vaishparuchuri)

A RAG-based enterprise knowledge assistant: hybrid (BM25 + semantic) document retrieval with a
trained reranker, plus a validated natural-language-to-SQL layer over a business database, served
behind a FastAPI app.

Everything in this repo runs **fully offline** against a **synthetic** document corpus and
synthetic business database — no proprietary data, no API keys, no external model downloads
required. That was a deliberate constraint, not an accident: it means anyone can clone this and
get the exact same measured numbers below in under a minute (`python3 scripts/run_pipeline.py`).

## Why this exists

This project rebuilds, with real code and real measured metrics, the architecture behind a resume
bullet: *"Built a RAG-based enterprise knowledge assistant over 500+ internal documents (10K+
chunks) using hybrid BM25 + semantic retrieval with reranking... Implemented natural-language-to-SQL
querying with validation and confidence-based fallback."* Rather than assert a number, this repo
generates the eval sets, runs the evaluation, and commits the resulting `results/*.json` so the
numbers are checkable, not just claimed.

## Architecture

```
                    ┌─────────────────────────┐
   query ──────────▶│   Query Understanding    │  domain synonym expansion
                    └───────────┬─────────────┘
                                │
                 ┌──────────────┴──────────────┐
                 ▼                              ▼
        ┌─────────────────┐          ┌─────────────────────┐
        │  BM25 Retriever  │          │   Dense Retriever    │
        │  (rank_bm25)     │          │  (TF-IDF + LSA,       │
        │                  │          │   pluggable OpenAI)   │
        └────────┬─────────┘          └──────────┬───────────┘
                 │        top-30 each             │
                 └───────────────┬─────────────────┘
                                 ▼
                     ┌───────────────────────┐
                     │  Reciprocal Rank        │
                     │  Fusion (Hybrid)        │
                     └───────────┬─────────────┘
                                 ▼
                     ┌───────────────────────┐
                     │  Feature Reranker       │  logistic regression over
                     │  (learned, offline)     │  BM25/dense/lexical features
                     └───────────┬─────────────┘
                                 ▼
                           top-k results
```

```
question ──▶ Intent classifier ──▶ Slot extraction ──▶ SQL template ──▶ validation
                                                                            │
                              confidence < threshold?  ───yes──▶  "please clarify"
                                        │no
                                        ▼
                              execute against SQLite ──▶ result / error message
```

## What's measured, and how (read this before trusting any number)

**Retrieval eval set** (`eval/retrieval_eval_set.json`, 240 queries, `scripts/generate_eval_sets.py`):
every synthetic document gets a unique "Document Reference" fact (a case ID, an owner name, a
review date) embedded in exactly one of its chunks. Two query types are generated from that:

- **lookup** queries reference a case ID or owner verbatim ("What's the status of case FAQ-2481?")
  — the kind of exact-match query BM25 is naturally good at.
- **conceptual** queries paraphrase a document's topic through a synonym table ("who is allowed to
  work from home" for "remote work eligibility") combined with department/product/region — the
  kind of query dense/semantic retrieval is supposed to help with.

Metric is **hit-rate@5** (did the top-5 results include at least one truly relevant chunk) —
most write-ups that say "Precision@5" for RAG systems mean this, not strict per-slot precision,
which would be capped at 1/5 for any query with a single correct answer.

70% of queries train the reranker; the 30% held-out split is what gets scored, so the reranker's
number isn't inflated by fitting on its own test set.

**NL2SQL eval set** (`eval/nl2sql_eval_set.json`, 154 questions, `scripts/generate_nl2sql_eval.py`):
144 in-scope business questions generated from the same intent/slot builders the translator uses
(so ground-truth SQL is derived, not hand-typed) + 10 out-of-scope questions ("What's the weather
like in APAC?") that a good system should decline rather than confidently mis-answer. "Execution
accuracy" = generated SQL executes **and returns the same result set** as ground truth, not just
"doesn't error."

### Results (from the last `scripts/run_pipeline.py` run, committed in `results/`)

| Method | Hit-rate@5 (overall) | on lookup queries | on conceptual queries |
|---|---|---|---|
| BM25 only | 65.3% | 100.0% | 26.5% |
| Dense (vector) only | 19.4% | 10.5% | 29.4% |
| Hybrid (RRF fusion) | 29.2% | 34.2% | 23.5% |
| **Hybrid + reranked** | **68.1%** | 97.4% | 35.3% |

**Hybrid + reranked improves +250% over vector-only search** on this eval set (19.4% → 68.1%).

Two things worth being upfront about in that table: (1) naive RRF fusion alone actually
*underperforms BM25-only* on lookup queries (34.2% vs 100%) — averaging in a weak dense signal can
hurt when one retriever is near-perfect and the other is near-random; the learned reranker is what
recovers that gap (97.4%) while still doing better than either retriever alone on conceptual
queries. (2) The dense retriever here is TF-IDF+LSA (see "Design decisions" below), not a
transformer — its conceptual-query performance is capped by that choice, and the numbers say so
honestly rather than being tuned to look better.

```json
// results/retrieval_metrics.json
{
  "n_test_queries": 72,
  "n_chunks_indexed": 11529,
  "precision_at_5": {
    "bm25_only": 0.6528, "dense_only": 0.1944,
    "hybrid_rrf": 0.2917, "hybrid_reranked": 0.6806
  },
  "improvement_hybrid_reranked_vs_vector_only_pct": 250.0
}
```

**NL2SQL:**

| Metric | Value |
|---|---|
| Execution accuracy (144 in-scope questions) | **97.2%** (140/144) |
| Correct abstention rate (10 out-of-scope questions) | **100%** (10/10) |
| Avg. response time | 4.1 ms |
| p95 response time | 9.0 ms |

```json
// results/nl2sql_metrics.json
{
  "execution_accuracy_in_scope": 0.9722,
  "abstention_correct_rate_out_of_scope": 1.0,
  "counts": {
    "answered_correct": 140, "answered_wrong": 0,
    "declined_in_scope": 4, "declined_out_of_scope_correctly": 10,
    "hallucinated_out_of_scope": 0, "error_in_scope": 0
  }
}
```

Zero wrong answers and zero hallucinations on out-of-scope questions in this run — the 4
"declined_in_scope" cases are the confidence gate correctly erring conservative on an ambiguous
phrasing ("Revenue per category, please." has no clear interrogative marker). That trade-off
(never confidently wrong, sometimes conservatively cautious) is the point of the confidence
threshold in `src/nl2sql/validator.py`.

**If you're adapting these numbers for your own resume: rerun `scripts/run_pipeline.py`, read the
actual JSON in `results/`, and use those. Don't copy the numbers above without rerunning them —
that defeats the entire point of this repo.**

## Design decisions (and their honest trade-offs)

- **Dense retrieval is TF-IDF + truncated SVD (LSA), not a transformer.** The sandboxed environment
  this repo was built in blocks outbound access to huggingface.co, so downloading
  `sentence-transformers` weights wasn't possible. `src/embeddings.py` defines an `Embedder`
  interface with `LSAEmbedder` as the default (zero downloads, fully offline) and `OpenAIEmbedder`
  documenting the swap-in for real embeddings once you have an API key and network access. LSA's
  weakness — no notion of a word it never saw during `fit()` — is exactly why
  `src/query_understanding.py` exists: a small domain-thesaurus query-expansion layer that bridges
  known paraphrases back to corpus vocabulary before embedding. That's a real production pattern
  (Elasticsearch/Solr synonym filters), used here as an honest, low-dependency stand-in for what a
  transformer encoder would give you for free.
- **Reranking is a logistic-regression feature model, not a cross-encoder.** Same constraint,
  same pattern: `src/reranker.py`'s `FeatureReranker` combines BM25 score, dense cosine similarity,
  term overlap, title match, and phrase match into a trained model; `CrossEncoderReranker`
  documents the transformer swap-in.
- **NL-to-SQL is a template/intent classifier, not an LLM call.** `src/nl2sql/translator.py`'s
  `TemplateTranslator` is fully deterministic and offline; `LLMTranslator` documents the
  OpenAI-backed version for open-ended questions outside the ~9 supported intents. The intent
  classifier uses AND/OR keyword groups per intent (not just single keywords) specifically because
  early iterations had a real bug here: "How many Enterprise customers in North America?" was
  misclassified as `top_customers` instead of `customer_count` because both intents only required
  the word "customer" to be present. Fixed by requiring an anchor noun *and* an action phrase per
  intent — see the regression test in `tests/test_nl2sql.py::test_customer_count_vs_top_customers_disambiguation`.
- **All generated SQL is parameterized**, never string-interpolated from user text, and `validator.py`
  refuses to execute anything that isn't a `SELECT`.

## Repo layout

```
src/
  corpus.py              synthetic document corpus generator (520 docs, 11.5K chunks)
  embeddings.py           Embedder interface: LSAEmbedder (default) + OpenAIEmbedder (pluggable)
  retrieval.py             BM25Retriever, DenseRetriever, HybridRetriever (RRF fusion)
  query_understanding.py  domain synonym expansion for the dense retriever
  reranker.py              FeatureReranker (trained) + CrossEncoderReranker (pluggable)
  api.py                    FastAPI app: POST /search, POST /ask, GET /healthz
  nl2sql/
    schema.py               synthetic retail DB generator (customers/products/orders/order_items)
    translator.py            TemplateTranslator (default) + LLMTranslator (pluggable)
    validator.py             execution + confidence-based fallback
scripts/
  run_pipeline.py          regenerates every data file and metric in one command
  generate_eval_sets.py    builds eval/retrieval_eval_set.json
  generate_nl2sql_eval.py  builds eval/nl2sql_eval_set.json
  evaluate_retrieval.py    produces results/retrieval_metrics.json
  evaluate_nl2sql.py       produces results/nl2sql_metrics.json
  demo_cli.py              no-server interactive demo
tests/                    15 tests: retrieval, reranker, corpus, nl2sql, API
docker/Dockerfile         builds the corpus/DB at image build time, serves the API
```

## Running it

```bash
pip install -r requirements.txt

# regenerate corpus, DB, eval sets, and both metrics files (~30s)
python3 scripts/run_pipeline.py

# run the test suite
PYTHONPATH=. python3 -m pytest tests/ -v

# try it without a server
PYTHONPATH=. python3 scripts/demo_cli.py search "vendor onboarding process"
PYTHONPATH=. python3 scripts/demo_cli.py ask "What are the top 5 customers by spend in North America in 2025?"

# or run the API
PYTHONPATH=. uvicorn src.api:app --reload
# then: curl -X POST localhost:8000/ask -H 'content-type: application/json' \
#         -d '{"question": "How many orders were placed in EMEA in 2025?"}'
```

```bash
# Docker
docker build -f docker/Dockerfile -t rag-copilot .
docker run -p 8000:8000 rag-copilot
```

## What I'd change with more time / a production budget

- Swap `LSAEmbedder` for a real sentence-transformer or hosted embedding API (the interface is
  already there in `embeddings.py`) — conceptual-query hit-rate is the number most likely to move.
- Swap `TemplateTranslator`'s 9 fixed intents for `LLMTranslator` behind the same confidence-gated
  `validator.py`, so open-ended questions outside the template set get answered instead of declined.
- Add a real vector index (FAISS/pgvector) once corpus size stops fitting comfortably in memory —
  at 11.5K chunks this isn't necessary yet.
- Log `/ask` and `/search` calls with outcomes to build a real labeled eval set from production
  traffic instead of a synthetically-generated one.

## License

MIT — see [LICENSE](LICENSE).
