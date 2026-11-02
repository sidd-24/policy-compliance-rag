# 📄 Policy & Compliance Q&A (RAG)

Ask questions about company policies in plain language. Every answer cites the **document, page and section** it comes from. If the policies don't cover a question, the assistant says so instead of guessing.

For compliance questions, an answer you can't check is worse than no answer. This project therefore puts most of its effort into grounding and evaluation, rather than a single PDF chat.

## What it does

- **Citations:** sources are given to the LLM numbered, and every statement must cite them ([1], [2]). The UI shows each cited passage with its document, page and section. An answer that cites no source is flagged.
- **Refusal:** there are two stages.
  1. **Relevance gate.** If no passage is relevant enough, the question never reaches the LLM.
  2. **NOT_FOUND rule.** If the passages are related but don't contain the answer, the model must reply `NOT_FOUND`.
- **Section-aware chunking:** chunks never cross a section boundary and carry their section heading. A citation therefore points to the right section, for example "§6 Patch and Vulnerability Management".
- **Evaluation set:** 33 questions on the sample policies, measuring retrieval, the refusal gate and end-to-end answers.
- **Sample policies:** four policies of a fictional manufacturer, written for this project ([`data/policies/`](data/policies/)):
  - Information Security Policy (ISO 27001 style)
  - Acceptable Use and Generative AI Policy
  - Data Protection and Retention Policy (GDPR)
  - Incident Response Procedure (including NIS2 reporting)

## How it works

```
PDFs ─► text per page (headers/footers removed) ─► split by section ─► local embeddings ─► Chroma
                                                                                         │
Question ─► embed ─► top-4 sections ─► best relevance < 0.35? ─► yes: refuse ◄───────────┘
                                              │ no
                                              ▼
                    numbered sources ─► Groq LLM ─► NOT_FOUND? ─► refuse
                                                         │ no
                                                         ▼
                                       answer + [n] citations ─► document, page, section, quote
```

## Evaluation

[`data/eval/questions.json`](data/eval/questions.json) has 33 questions:

- **25 answerable.** Each has the document and section that hold the answer, and the key facts the answer must contain.
- **8 not covered by any policy.** Four are clearly off-topic, such as vacation days. Four are near misses that share words with real sections, such as "Which encryption standard does the guest Wi-Fi use?".

Results from `python evaluate.py` (full report in [`results/eval_report.md`](results/eval_report.md)):

| Chunking | Hit@1 | Hit@4 | Top-1 citation shows the right section |
|---|---|---|---|
| Fixed 800 characters | 96% | 100% | 40% |
| **Section-aware** | 96% | 100% | **96%** |

On these documents, both strategies find the right passage. With fixed-size chunks, though, a chunk often spans two sections. The citation then names the section where the chunk starts, which is wrong 60% of the time. Section-aware chunking fixes the citation without losing retrieval quality.

| Relevance threshold | Answerable questions passed | Unanswerable questions refused |
|---|---|---|
| 0.30 | 100% | 50% |
| **0.35 (used)** | 100% | 50% |
| 0.45 | 96% | 62% |
| 0.50 | 80% | 75% |

The gate refuses every clearly off-topic question (best match 0.21 to 0.28). The weakest answerable question scores 0.45. The near misses score 0.43 to 0.54, the same range as real questions, so no threshold can separate them. They are left to the `NOT_FOUND` rule in the prompt. Raising the threshold to catch them would start refusing real questions.

End-to-end results from `python evaluate.py --llm` (needs a Groq key), using `openai/gpt-oss-120b`:

| Metric | Result |
|---|---|
| Answer contains the expected facts | 25 / 25 |
| Answer cites the expected section | 25 / 25 |
| Unanswerable questions refused | 8 / 8 (4 by the relevance gate, 4 by `NOT_FOUND`) |
| Answerable questions wrongly refused | 0 |
| Answers without any citation | 0 |

The two refusal stages split the work as intended. The gate stopped the four off-topic questions. The model replied `NOT_FOUND` for the four near misses, which no threshold could catch.

**Read these numbers with care.** I wrote the policies and the questions myself. The documents are short and clearly structured, and this is one run. The results show that the pipeline works as designed. They do not predict accuracy on a real policy library, which would need its own question set. `evaluate.py` is built so that one can be dropped in.

> **Bugs found by the evaluation, all fixed.**
> - **Mixed uploads.** The Chroma collection had a fixed name, so a second upload in the same session was mixed with the first. Answers could cite the wrong document.
> - **Inconsistent search.** Chroma's default HNSW settings sometimes missed the best passage, so the same question got different scores on different runs (e.g. 0.50 vs 0.31). The index is now built single-threaded with a wide search. Results are identical across runs and match an exact search.
> - **Hidden citations.** gpt-oss writes citations as `【1】` instead of `[1]`. The parser missed them, so 7 correctly cited answers were flagged as uncited, and their sources were hidden in the UI. The first end-to-end run scored 44% on facts, mostly for this reason. The fact check also failed on the non-breaking spaces gpt-oss puts in "14 days".
> - **Retired model.** The original default model, `llama-3.3-70b-versatile`, has been retired by Groq and now returns a 404.

## Stack

| Layer | Tool |
|---|---|
| LLM | Groq (`openai/gpt-oss-120b` by default) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, no API) |
| Vector store | ChromaDB (in-memory, cosine) |
| Orchestration | LangChain |
| UI | Streamlit |

## Setup

```bash
git clone https://github.com/sidd-24/policy-compliance-rag
cd policy-compliance-rag
pip install -r requirements.txt
export GROQ_API_KEY="your_groq_api_key"     # free key at console.groq.com

streamlit run app.py          # sample policies are loaded by default; upload your own PDFs in the UI
python evaluate.py            # retrieval + refusal gate (no API key needed)
python -m pytest -q           # 15 tests, no API key or network needed
```

Headings are detected when they are numbered, such as `4. Passwords` or `3.2 Access reviews`. Most policy documents use this format.

## Project structure

```
├── app.py                        # Streamlit UI: upload or sample policies, cited answers
├── evaluate.py                   # Evaluation CLI, writes results/eval_report.md
├── rag/
│   ├── loader.py                 # PDF text, header/footer removal, section-aware chunking
│   ├── embeddings.py             # Local embeddings + Chroma index
│   ├── chain.py                  # Retrieval, refusal gate, cited answers
│   └── evaluation.py             # Retrieval, refusal and answer metrics
├── data/
│   ├── policies/                 # Sample policies (fictional company)
│   └── eval/questions.json       # Evaluation set
├── scripts/make_sample_policies.py
├── results/                      # Evaluation report
└── tests/
```

## Limitations

- The sample policies are short: 4 documents, 42 chunks. On a large policy library, retrieval is harder, and the threshold should be re-tuned with `evaluate.py` on your own documents.
- The answer check uses keywords. It catches a missing or wrong number, but not every badly worded answer.
- This answers questions about the documents it is given. It is not legal advice.
