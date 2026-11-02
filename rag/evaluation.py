"""
Metrics for the evaluation set in data/eval/questions.json.

Each answerable question names the document and section that answer it and
the key facts the answer must contain. Unanswerable questions have no source
and should be refused.
"""
import unicodedata


def is_expected(metadata: dict, item: dict, label_only: bool = False) -> bool:
    """
    True if a chunk comes from the document and section that answer the question.
    label_only=True checks only the section a citation would show, not every
    section a fixed-size chunk happens to overlap.
    """
    sections = metadata["section"] if label_only else (metadata.get("sections_covered") or metadata["section"])
    return (metadata["source"].startswith(item["source"])
            and any(s.strip().startswith(f"{item['section']}.") for s in sections.split("|")))


def retrieval_metrics(items: list[dict], hits_per_item: list[list[tuple]]) -> dict:
    """
    Hit@1, Hit@k and MRR over the answerable questions.
    hits_per_item holds the retrieved (document, score) pairs for each item.
    """
    ranks, labels = [], []
    for item, hits in zip(items, hits_per_item):
        if not item["source"]:
            continue
        rank = next((i for i, (doc, _) in enumerate(hits, start=1) if is_expected(doc.metadata, item)), None)
        ranks.append(rank)
        labels.append(bool(hits) and is_expected(hits[0][0].metadata, item, label_only=True))
    n = len(ranks)
    return {
        "questions": n,
        "hit_at_1": sum(r == 1 for r in ranks) / n,
        "hit_at_k": sum(r is not None for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "citation_label_at_1": sum(labels) / n,
    }


def threshold_sweep(items: list[dict], top_scores: list[float], thresholds: list[float]) -> list[dict]:
    """
    For each relevance threshold: the share of answerable questions that pass
    the refusal gate and the share of unanswerable questions it refuses.
    """
    answerable = [s for item, s in zip(items, top_scores) if item["source"]]
    unanswerable = [s for item, s in zip(items, top_scores) if not item["source"]]
    return [{
        "threshold": t,
        "answerable_passed": sum(s >= t for s in answerable) / len(answerable),
        "unanswerable_refused": sum(s < t for s in unanswerable) / len(unanswerable),
    } for t in thresholds]


def facts_present(answer: str, must_include: list[list[str]]) -> bool:
    """
    Every group of alternatives must have at least one match (case-insensitive).
    NFKC turns non-breaking spaces ("14 days") into normal spaces first.
    """
    text = unicodedata.normalize("NFKC", answer).lower()
    return all(any(option.lower() in text for option in group) for group in must_include)


def score_answer(item: dict, result: dict) -> dict:
    """Checks one end-to-end answer against the expected facts and source."""
    if not item["source"]:
        return {"id": item["id"], "answerable": False, "correct": result["status"] == "refused",
                "status": result["status"], "cited_expected": None}
    cited_expected = any(is_expected(c, item) for c in result["citations"])
    correct = result["status"] == "answered" and facts_present(result["answer"], item["must_include"])
    return {"id": item["id"], "answerable": True, "correct": correct, "status": result["status"],
            "cited_expected": cited_expected}


def summarise_answers(scores: list[dict]) -> dict:
    answerable = [s for s in scores if s["answerable"]]
    unanswerable = [s for s in scores if not s["answerable"]]
    answered = [s for s in answerable if s["status"] == "answered"]
    return {
        "answer_accuracy": sum(s["correct"] for s in answerable) / len(answerable),
        "citation_accuracy": sum(s["cited_expected"] for s in answered) / len(answered) if answered else 0.0,
        "answered_without_citation": sum(s["status"] == "uncited" for s in scores),
        "false_refusals": sum(s["status"] == "refused" for s in answerable),
        "correct_refusals": sum(s["correct"] for s in unanswerable) / len(unanswerable),
    }
