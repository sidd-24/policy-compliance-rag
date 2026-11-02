"""Tests for chunking, citations, the refusal gate and the evaluation metrics. No API key or network needed."""
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag.chain import NOT_FOUND, answer_question, parse_citations
from rag.embeddings import build_vectorstore
from rag.evaluation import facts_present, retrieval_metrics, score_answer, threshold_sweep
from rag.loader import chunk_document, load_and_chunk, read_pages

POLICY = Path("data/policies/IS-01_Information_Security_Policy.pdf")


# ─── Loading and chunking ──────────────────────────────────────────────────────

def test_sections_carry_over_page_breaks():
    pages = ["Policy\n1. Scope\nApplies to everyone.\n2. Passwords\nAt least 14 characters.",
             "MFA is required for VPN.\n3. Backup\nDaily backups."]
    chunks = chunk_document("Test Policy", pages, "test.pdf")
    by_text = {c.page_content.split("\n", 1)[1]: c.metadata for c in chunks if "\n" in c.page_content}
    assert by_text["MFA is required for VPN."] == {"source": "test.pdf", "title": "Test Policy", "page": 2,
                                                  "section": "2. Passwords", "chunk_id": "test.pdf#3"}
    assert by_text["Daily backups."]["section"] == "3. Backup"
    # The heading is repeated in the chunk text, so the chunk is retrievable by topic
    assert any(c.page_content.startswith("Test Policy - 2. Passwords\nAt least 14") for c in chunks)


def test_repeated_footer_and_page_numbers_are_removed():
    title, pages = read_pages(POLICY, POLICY.name)
    assert title == "Information Security Policy"
    assert not any("Internal" in line or line.startswith("Page ") for p in pages for line in p.splitlines())


def test_fixed_chunks_record_every_section_they_overlap():
    chunks = load_and_chunk(POLICY, by_section=False)
    assert any("|" in c.metadata["sections_covered"] for c in chunks)


def test_new_upload_does_not_mix_with_previous_index():
    embeddings = DeterministicFakeEmbedding(size=32)
    first = build_vectorstore([Document(page_content="old policy", metadata={"source": "a"})], embeddings)
    second = build_vectorstore([Document(page_content="new policy", metadata={"source": "b"})], embeddings)
    assert first._collection.count() == 1 and second._collection.count() == 1


# ─── Citations and refusal ─────────────────────────────────────────────────────

def test_parse_citations_keeps_valid_numbers_in_order():
    assert parse_citations("Use MFA [2]. Also [2][5] and [1, 3].", n_sources=3) == [2, 1, 3]
    assert parse_citations("No sources here.", n_sources=3) == []
    assert parse_citations("Within 72 hours【1】 and 24 hours【2†L4-L6】.", n_sources=3) == [1, 2]


def doc(section, text="..."):
    return Document(page_content=f"Policy - {section}\n{text}",
                    metadata={"source": "IS-01.pdf", "title": "Policy", "page": 1, "section": section})


class StubStore:
    def __init__(self, hits):
        self.hits = hits

    def similarity_search_with_relevance_scores(self, question, k):
        return self.hits[:k]


class StubLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def invoke(self, messages):
        self.calls += 1
        return type("Reply", (), {"content": self.reply})()


def test_low_relevance_is_refused_without_calling_the_llm():
    llm = StubLLM("anything")
    result = answer_question("Vacation days?", StubStore([(doc("1. Scope"), 0.2)]), llm, min_relevance=0.35)
    assert result["status"] == "refused" and llm.calls == 0 and "0.20" in result["reason"]


def test_not_found_from_the_llm_is_a_refusal():
    result = answer_question("Guest Wi-Fi encryption?", StubStore([(doc("7. Network"), 0.5)]), StubLLM(NOT_FOUND))
    assert result["status"] == "refused" and result["citations"] == []


def test_cited_answer_returns_the_cited_sources():
    store = StubStore([(doc("4. Passwords", "At least 14 characters."), 0.7), (doc("5. Endpoint"), 0.5)])
    result = answer_question("Password length?", store, StubLLM("At least 14 characters [1]."))
    assert result["status"] == "answered"
    assert [(c["section"], c["excerpt"]) for c in result["citations"]] == [("4. Passwords", "At least 14 characters.")]


def test_answer_without_citation_is_flagged():
    result = answer_question("Password length?", StubStore([(doc("4. Passwords"), 0.7)]), StubLLM("14 characters."))
    assert result["status"] == "uncited" and result["reason"]


# ─── Evaluation metrics ────────────────────────────────────────────────────────

ITEMS = [{"id": "Q1", "source": "IS-01", "section": "4", "must_include": [["14"]]},
         {"id": "N1", "source": None}]


def test_retrieval_metrics_and_citation_label():
    spanning = Document(page_content="", metadata={"source": "IS-01.pdf", "section": "3. Access",
                                                   "sections_covered": "3. Access | 4. Passwords"})
    metrics = retrieval_metrics(ITEMS, [[(spanning, 0.6)], []])
    assert metrics["hit_at_1"] == 1.0                 # the chunk contains section 4...
    assert metrics["citation_label_at_1"] == 0.0      # ...but its citation would say section 3


def test_threshold_sweep():
    sweep = threshold_sweep(ITEMS, [0.5, 0.3], [0.25, 0.4])
    assert sweep == [{"threshold": 0.25, "answerable_passed": 1.0, "unanswerable_refused": 0.0},
                     {"threshold": 0.4, "answerable_passed": 1.0, "unanswerable_refused": 1.0}]


@pytest.mark.parametrize("answer, expected", [
    ("At least 14 characters.", True), ("At least 14 characters.", True), ("At least twelve characters.", False)])
def test_facts_present(answer, expected):
    assert facts_present(answer, [["14", "fourteen"]]) is expected


def test_score_answer():
    cited = {"source": "IS-01.pdf", "section": "4. Passwords"}
    good = score_answer(ITEMS[0], {"status": "answered", "answer": "14 characters [1]", "citations": [cited]})
    assert good["correct"] and good["cited_expected"]
    refused = score_answer(ITEMS[1], {"status": "refused", "answer": "", "citations": []})
    assert refused["correct"]
