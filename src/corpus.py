"""
Synthetic enterprise document corpus generator.

Generates a realistic-looking internal knowledge base (HR policies, engineering
runbooks, product FAQs, finance reports, meeting notes, compliance docs) that we
use to demonstrate and evaluate the hybrid retrieval + reranking pipeline without
depending on any real company's proprietary documents.

Everything here is templated + randomized (seeded for reproducibility) -- no
external data or network calls required.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

RANDOM_SEED = 42

FIRST_NAMES = [
    "Maria", "James", "Wei", "Fatima", "Liam", "Aisha", "Noah", "Priya", "Lucas",
    "Sofia", "Daniel", "Nia", "Yusuf", "Elena", "Marcus", "Hana", "Diego", "Ines",
    "Kwame", "Grace", "Ravi", "Chloe", "Omar", "Zara", "Tomas", "Amara", "Felix",
    "Leila", "Victor", "Mei",
]
LAST_NAMES = [
    "Alvarez", "Chen", "Okafor", "Rossi", "Nakamura", "Silva", "Kowalski", "Haddad",
    "Johansson", "Kim", "Dubois", "Osei", "Petrov", "Nakamoto", "Andrade", "Farooq",
    "Lindgren", "Moretti", "Adeyemi", "Novak", "Reyes", "Kaur", "Ibrahim", "Costa",
    "Larsson", "Mensah", "Vargas", "Suzuki", "Weiss", "Duarte",
]

# Synonym substitutions used to paraphrase topics in eval queries so BM25
# can't win purely on verbatim substring overlap with the source text.
TOPIC_SYNONYMS = {
    "remote work eligibility": "who is allowed to work from home",
    "expense reimbursement": "getting paid back for business expenses",
    "paid time off accrual": "how vacation days build up",
    "data retention": "how long we keep customer data",
    "vendor onboarding": "bringing a new supplier on board",
    "password rotation": "how often passwords must change",
    "on-call compensation": "extra pay for being on call",
    "customer refund approval": "sign-off needed to refund a customer",
    "code review requirements": "rules for reviewing code before merge",
    "incident escalation": "when and how to escalate an incident",
    "contractor access provisioning": "granting system access to contractors",
    "parental leave": "time off for new parents",
    "travel booking": "how to book business travel",
    "equipment procurement": "ordering new equipment",
    "performance review cadence": "how often performance reviews happen",
    "monthly active users": "MAU trend",
    "gross margin": "profit margin",
    "churn rate": "customer cancellation rate",
    "average handle time": "how long support tickets take to resolve",
    "SKU-level forecast accuracy": "how accurate product-level demand forecasts are",
    "P95 API latency": "slowest 5% of API response times",
    "ticket backlog": "the pile of unresolved support tickets",
    "net revenue retention": "revenue kept and expanded from existing customers",
    "on-time delivery rate": "percent of orders delivered on schedule",
    "defect escape rate": "bugs that slip past QA into production",
}

DEPARTMENTS = [
    "Human Resources", "Engineering", "Finance", "Sales", "Customer Support",
    "Legal & Compliance", "Product", "Data & Analytics", "Security", "Operations",
]

PRODUCTS = [
    "Atlas Billing", "Nimbus Data Platform", "Beacon CRM", "Forge API Gateway",
    "Vantage Analytics Suite", "Lighthouse Support Desk", "Compass Payments",
    "Relay Notifications", "Anchor Auth", "Horizon Reporting",
]

REGIONS = ["North America", "EMEA", "APAC", "LATAM"]

DOC_TYPES = [
    "policy", "runbook", "faq", "meeting_notes", "quarterly_report",
    "onboarding_guide", "incident_postmortem", "vendor_contract_summary",
    "product_spec", "compliance_memo",
]

# --- Small building blocks used to assemble longer, varied documents ----------

POLICY_TOPICS = [
    "remote work eligibility", "expense reimbursement", "paid time off accrual",
    "data retention", "vendor onboarding", "password rotation", "on-call compensation",
    "customer refund approval", "code review requirements", "incident escalation",
    "contractor access provisioning", "parental leave", "travel booking",
    "equipment procurement", "performance review cadence",
]

METRICS = [
    "monthly active users", "gross margin", "churn rate", "average handle time",
    "SKU-level forecast accuracy", "P95 API latency", "ticket backlog",
    "net revenue retention", "on-time delivery rate", "defect escape rate",
]

def _rng(seed_offset: int) -> random.Random:
    return random.Random(RANDOM_SEED + seed_offset)


def _lorem_sentence(rng: random.Random, subject: str) -> str:
    templates = [
        "The {subject} process is owned by {dept} and reviewed on a {cadence} basis.",
        "Any change to {subject} must be approved by the {dept} lead before rollout.",
        "As of {quarter}, {subject} is tracked in the {product} dashboard.",
        "Teams outside {region} should route {subject} questions to the regional POC.",
        "A recurring audit of {subject} is scheduled every {cadence} to catch drift.",
        "The current SLA for {subject} is {number} business days.",
        "{dept} flagged {subject} as a top risk area in the last planning cycle.",
        "Automation reduced manual effort on {subject} by roughly {pct}% last quarter.",
        "Exceptions to the {subject} rule require a written justification filed with {dept}.",
        "{product} now surfaces {subject} metrics directly to stakeholders, removing the old manual report.",
    ]
    t = rng.choice(templates)
    return t.format(
        subject=subject,
        dept=rng.choice(DEPARTMENTS),
        cadence=rng.choice(["weekly", "biweekly", "monthly", "quarterly"]),
        quarter=f"Q{rng.randint(1,4)} {rng.randint(2023,2026)}",
        product=rng.choice(PRODUCTS),
        region=rng.choice(REGIONS),
        number=rng.randint(1, 15),
        pct=rng.randint(8, 55),
    )


def _make_paragraph(rng: random.Random, subject: str, n_sentences: int) -> str:
    return " ".join(_lorem_sentence(rng, subject) for _ in range(n_sentences))


@dataclass
class Document:
    doc_id: str
    title: str
    doc_type: str
    department: str
    product: str | None
    region: str | None
    created: str
    text: str
    topic: str = ""
    case_id: str = ""
    owner: str = ""
    reviewed_date: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    doc_type: str
    department: str
    position: int
    text: str

    def to_dict(self):
        return asdict(self)


def _gen_document(i: int) -> Document:
    rng = _rng(i)
    doc_type = rng.choice(DOC_TYPES)
    dept = rng.choice(DEPARTMENTS)
    product = rng.choice(PRODUCTS) if rng.random() < 0.6 else None
    region = rng.choice(REGIONS) if rng.random() < 0.5 else None
    topic = rng.choice(POLICY_TOPICS + METRICS)
    quarter = f"Q{rng.randint(1,4)} {rng.randint(2023,2026)}"

    title_templates = {
        "policy": f"{topic.title()} Policy",
        "runbook": f"Runbook: Handling {topic.title()} Escalations for {product or 'the platform'}",
        "faq": f"{product or dept} FAQ: {topic.title()}",
        "meeting_notes": f"{dept} Sync Notes - {quarter}",
        "quarterly_report": f"{dept} {quarter} Business Review",
        "onboarding_guide": f"New Hire Onboarding Guide - {dept}",
        "incident_postmortem": f"Postmortem: {product or 'Platform'} {topic.title()} Incident",
        "vendor_contract_summary": f"Vendor Contract Summary - {product or 'Service'} Renewal",
        "product_spec": f"{product or 'New Feature'} Product Spec",
        "compliance_memo": f"Compliance Memo: {topic.title()}",
    }
    title = title_templates[doc_type]

    owner = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    case_id = f"{doc_type[:3].upper()}-{2000 + i}"
    reviewed_date = f"{rng.randint(1,28):02d}-{rng.choice(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])}-{rng.randint(2024,2026)}"

    n_paragraphs = rng.randint(18, 34)
    paragraphs = []
    section_names = ["Overview", "Background", "Current State", "Process", "Metrics",
                      "Risks & Mitigations", "Ownership", "Next Steps", "References", "Appendix"]

    # Unique, findable anchor paragraph -- every document gets exactly one of
    # these, always right after Overview, so downstream eval queries can
    # target a single ground-truth chunk instead of "any chunk in this doc".
    anchor_paragraph = (
        f"## Document Reference\n"
        f"This {doc_type.replace('_', ' ')} on {topic} is filed under case reference "
        f"{case_id}. The document owner of record is {owner} ({dept}), last reviewed on "
        f"{reviewed_date}."
    )

    for p in range(n_paragraphs):
        section = section_names[p % len(section_names)]
        n_sent = rng.randint(3, 6)
        body = _make_paragraph(rng, topic, n_sent)
        paragraphs.append(f"## {section}\n{body}")
        if p == 0:
            paragraphs.append(anchor_paragraph)

    text = f"# {title}\n\n" + "\n\n".join(paragraphs)
    return Document(
        doc_id=f"doc-{i:04d}",
        title=title,
        doc_type=doc_type,
        department=dept,
        product=product,
        region=region,
        created=f"{quarter}",
        text=text,
        topic=topic,
        case_id=case_id,
        owner=owner,
        reviewed_date=reviewed_date,
    )


def generate_corpus(n_docs: int = 520) -> List[Document]:
    return [_gen_document(i) for i in range(n_docs)]


def chunk_document(doc: Document, target_words: int = 110) -> List[Chunk]:
    """Split a document into chunks by paragraph, packing ~target_words per chunk."""
    paras = [p for p in doc.text.split("\n\n") if p.strip()]
    chunks: List[Chunk] = []
    buf: List[str] = []
    buf_words = 0
    pos = 0
    for para in paras:
        words = len(para.split())
        if buf and buf_words + words > target_words:
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}-c{pos:03d}", doc_id=doc.doc_id, title=doc.title,
                doc_type=doc.doc_type, department=doc.department, position=pos,
                text="\n\n".join(buf),
            ))
            pos += 1
            buf, buf_words = [], 0
        buf.append(para)
        buf_words += words
    if buf:
        chunks.append(Chunk(
            chunk_id=f"{doc.doc_id}-c{pos:03d}", doc_id=doc.doc_id, title=doc.title,
            doc_type=doc.doc_type, department=doc.department, position=pos,
            text="\n\n".join(buf),
        ))
    return chunks


def build_and_save(out_dir: Path, n_docs: int = 520) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = generate_corpus(n_docs)
    all_chunks: List[Chunk] = []
    for d in docs:
        all_chunks.extend(chunk_document(d))

    with open(out_dir / "documents.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d.to_dict()) + "\n")

    with open(out_dir / "chunks.jsonl", "w") as f:
        for c in all_chunks:
            f.write(json.dumps(c.to_dict()) + "\n")

    # Record which chunk holds each document's unique "Document Reference"
    # anchor paragraph (case_id/owner/reviewed_date), so eval-set generation
    # can build precise, single-chunk-answerable lookup queries.
    anchor_map = {}
    for c in all_chunks:
        if "## Document Reference" in c.text:
            anchor_map[c.doc_id] = c.chunk_id
    with open(out_dir / "anchor_chunks.json", "w") as f:
        json.dump(anchor_map, f, indent=2)

    stats = {"n_documents": len(docs), "n_chunks": len(all_chunks), "n_anchors": len(anchor_map)}
    with open(out_dir / "corpus_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    stats = build_and_save(Path(__file__).resolve().parents[1] / "data", n_docs=520)
    print(stats)
