"""
Evaluates retrieval, the refusal gate and (optionally) end-to-end answers
on the sample policies and data/eval/questions.json.

    python evaluate.py          # retrieval and refusal gate, no API key needed
    python evaluate.py --llm    # also answers every question with Groq (needs GROQ_API_KEY)

Writes results/eval_report.md and results/eval_results.json.
"""
import argparse
import json
import os
import time
from pathlib import Path

from rag.chain import MIN_RELEVANCE, TOP_K, answer_question, build_llm, retrieve
from rag.embeddings import build_vectorstore, get_embeddings
from rag.evaluation import retrieval_metrics, score_answer, summarise_answers, threshold_sweep
from rag.loader import load_and_chunk

POLICY_DIR = Path("data/policies")
QUESTIONS = Path("data/eval/questions.json")
RESULTS = Path("results")
THRESHOLDS = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def build_index(by_section: bool, embeddings):
    chunks = [c for pdf in sorted(POLICY_DIR.glob("*.pdf")) for c in load_and_chunk(pdf, by_section=by_section)]
    return build_vectorstore(chunks, embeddings), len(chunks)


def pct(x):
    return f"{x:.0%}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", action="store_true", help="also evaluate end-to-end answers with Groq")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    args = parser.parse_args()

    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    embeddings = get_embeddings()
    report = {"questions": len(items), "answerable": sum(bool(i["source"]) for i in items), "top_k": TOP_K}
    lines = ["# Evaluation report", "",
             f"{report['questions']} questions on the sample policies: {report['answerable']} answerable, "
             f"{report['questions'] - report['answerable']} not covered by any policy.", ""]

    # 1. Retrieval: fixed-size chunks vs section-aware chunks
    lines += ["## Retrieval", "", f"| Chunking | Chunks | Hit@1 | Hit@{TOP_K} | MRR | Top-1 citation shows the right section |",
              "|---|---|---|---|---|---|"]
    stores = {}
    for name, by_section in [("Fixed 800 characters", False), ("Section-aware", True)]:
        store, n_chunks = build_index(by_section, embeddings)
        hits = [retrieve(store, item["question"], TOP_K) for item in items]
        metrics = retrieval_metrics(items, hits)
        report[f"retrieval_{'section' if by_section else 'fixed'}"] = {"chunks": n_chunks, **metrics}
        lines.append(f"| {name} | {n_chunks} | {pct(metrics['hit_at_1'])} | {pct(metrics['hit_at_k'])} | {metrics['mrr']:.2f} | {pct(metrics['citation_label_at_1'])} |")
        stores[by_section] = (store, hits)

    # 2. Refusal gate on the section-aware index
    store, hits = stores[True]
    top_scores = [h[0][1] if h else 0.0 for h in hits]
    sweep = threshold_sweep(items, top_scores, THRESHOLDS)
    report["threshold_sweep"] = sweep
    report["top_scores"] = {item["id"]: round(s, 3) for item, s in zip(items, top_scores)}
    lines += ["", "## Refusal gate (best-match relevance threshold)", "",
              "| Threshold | Answerable passed | Unanswerable refused |", "|---|---|---|"]
    for row in sweep:
        marker = " (used)" if abs(row["threshold"] - MIN_RELEVANCE) < 1e-9 else ""
        lines.append(f"| {row['threshold']:.2f}{marker} | {pct(row['answerable_passed'])} | {pct(row['unanswerable_refused'])} |")

    # 3. End-to-end answers
    if args.llm:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise SystemExit("Set GROQ_API_KEY to run the end-to-end evaluation.")
        llm = build_llm(api_key, args.model)
        details = []
        for item in items:
            result = answer_question(item["question"], store, llm)
            details.append({**score_answer(item, result), "question": item["question"],
                            "answer": result["answer"], "top_score": result["top_score"],
                            "citations": [f"{c['source']} p.{c['page']} {c['section']}" for c in result["citations"]]})
            print(f"{item['id']}: {result['status']:9} {'ok' if details[-1]['correct'] else 'WRONG'}")
            time.sleep(1)  # stay under the free-tier rate limit
        summary = summarise_answers(details)
        report["end_to_end"] = {"model": args.model, **summary}
        report["details"] = details
        lines += ["", f"## End-to-end answers ({args.model})", "",
                  "| Metric | Result |", "|---|---|",
                  f"| Answer contains the expected facts | {pct(summary['answer_accuracy'])} |",
                  f"| Answer cites the expected section | {pct(summary['citation_accuracy'])} |",
                  f"| Unanswerable questions refused | {pct(summary['correct_refusals'])} |",
                  f"| Answerable questions wrongly refused | {summary['false_refusals']} |",
                  f"| Answers without any citation | {summary['answered_without_citation']} |",
                  "", "### Misses", ""]
        misses = [d for d in details if not d["correct"]]
        lines += [f"- **{d['id']}** {d['question']} -> {d['status']}: {d['answer'][:160]}" for d in misses] or ["None."]

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
