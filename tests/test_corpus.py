from src.corpus import generate_corpus, chunk_document, _gen_document


def test_generation_is_deterministic():
    docs_a = generate_corpus(20)
    docs_b = generate_corpus(20)
    assert [d.text for d in docs_a] == [d.text for d in docs_b]


def test_every_document_has_a_unique_case_id():
    docs = generate_corpus(100)
    case_ids = [d.case_id for d in docs]
    assert len(case_ids) == len(set(case_ids))


def test_chunking_preserves_all_text_and_anchor_paragraph():
    doc = _gen_document(3)
    chunks = chunk_document(doc)
    assert len(chunks) >= 1
    joined = "\n\n".join(c.text for c in chunks)
    assert doc.case_id in joined
    assert any("## Document Reference" in c.text for c in chunks)
