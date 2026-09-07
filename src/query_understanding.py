"""
Lightweight query-normalization layer: maps common paraphrases of domain
topics back to canonical terminology before they hit either retriever.

This is the "domain thesaurus" pattern real enterprise search systems lean on
(e.g. Elasticsearch/Solr synonym filters) when a query uses different words
than the source documents. It matters a lot here specifically because our
default dense backend (`LSAEmbedder` in embeddings.py) is TF-IDF-based: it has
no notion of a word it never saw during `fit()`, so an out-of-vocabulary
paraphrase like "who is allowed to work from home" embeds as a near-zero
vector against a corpus that only ever says "remote work eligibility". A
transformer embedding model (see `OpenAIEmbedder`) would generalize across
that gap on its own; this module is the low-dependency stand-in for that
capability, built from the same synonym table used to author the eval set's
"conceptual" queries.
"""
from __future__ import annotations

import re
from typing import Dict

from .corpus import TOPIC_SYNONYMS

# paraphrase phrase -> canonical topic phrase, longest-phrase-first so we
# don't do a partial substring replacement inside a longer paraphrase.
_REVERSE: Dict[str, str] = {v.lower(): k for k, v in TOPIC_SYNONYMS.items()}
_ORDERED_PHRASES = sorted(_REVERSE.keys(), key=len, reverse=True)


def normalize_query(text: str) -> str:
    """Append canonical topic terms alongside any recognized paraphrase.

    We *append* rather than replace so exact user wording is still available
    to the lexical retriever, while the canonical vocabulary the corpus was
    generated with becomes available to the TF-IDF-based dense retriever.
    """
    lowered = text.lower()
    additions = []
    for phrase in _ORDERED_PHRASES:
        if phrase in lowered:
            additions.append(_REVERSE[phrase])
    if not additions:
        return text
    return text + " " + " ".join(additions)
