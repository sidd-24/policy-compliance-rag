"""
Answers a question from the indexed policies, with citations, or refuses.

Two safeguards keep answers grounded:
1. Refusal gate: if no retrieved passage is relevant enough, the question
   is not sent to the LLM at all.
2. Citations: the LLM sees numbered sources, must cite them as [1], [2],
   and must reply NOT_FOUND when the sources do not answer the question.
   Answers without a valid citation are flagged as uncited.
"""
import re

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

TOP_K = 4
# Chosen with evaluate.py: about halfway between the best off-topic question
# (0.28) and the weakest answerable one (0.45) on the sample policies. It
# stops clearly off-topic questions; near misses score like real questions
# and are left to the NOT_FOUND rule.
MIN_RELEVANCE = 0.35

NOT_FOUND = "NOT_FOUND"
REFUSAL_TEXT = ("I could not find this in the loaded policies. Please check with the policy owner "
                "or the relevant department.")

SYSTEM_PROMPT = f"""You answer employee questions about company policies.
Rules:
- Use only the numbered sources below. Do not use general knowledge.
- After every sentence that uses a source, cite it in square brackets, e.g. [1] or [2][3].
- Quote numbers, deadlines and roles exactly as written in the sources.
- If the sources do not contain the answer, reply with exactly {NOT_FOUND} and nothing else.
- Answer in at most 5 short sentences."""


def build_llm(api_key: str, model: str = "openai/gpt-oss-120b") -> ChatGroq:
    return ChatGroq(api_key=api_key, model_name=model, temperature=0, max_retries=6)


def retrieve(vectorstore: Chroma, question: str, k: int = TOP_K) -> list[tuple]:
    """Top-k chunks with relevance scores (cosine similarity), best first."""
    return vectorstore.similarity_search_with_relevance_scores(question, k=k)


def format_sources(hits: list[tuple]) -> str:
    blocks = []
    for n, (doc, _) in enumerate(hits, start=1):
        m = doc.metadata
        blocks.append(f"[{n}] {m['title']}, page {m['page']}, section {m['section'] or 'n/a'}\n{doc.page_content}")
    return "\n\n".join(blocks)


def parse_citations(answer: str, n_sources: int) -> list[int]:
    """
    Source numbers cited in the answer, in order of first use. Numbers that do not exist are dropped.
    Accepts [1], [1, 2] and the full-width 【1】 that some models (e.g. gpt-oss) write instead.
    """
    cited = []
    for group in re.findall(r"[\[【](\d+(?:\s*,\s*\d+)*)[^\]】]*[\]】]", answer):
        for n in (int(x) for x in group.split(",")):
            if 1 <= n <= n_sources and n not in cited:
                cited.append(n)
    return cited


def citation(n: int, doc, score: float) -> dict:
    m = doc.metadata
    excerpt = doc.page_content.split("\n", 1)[-1]
    return {"n": n, "source": m["source"], "title": m["title"], "page": m["page"],
            "section": m["section"], "score": round(score, 3), "excerpt": excerpt}


def answer_question(question: str, vectorstore: Chroma, llm, k: int = TOP_K,
                    min_relevance: float = MIN_RELEVANCE) -> dict:
    """
    Returns a dict with:
      status     "answered", "refused" or "uncited"
      answer     the text to show
      citations  the sources the answer cites (document, page, section, excerpt)
      top_score  relevance of the best retrieved passage
      reason     why it was refused, if it was
    """
    hits = retrieve(vectorstore, question, k)
    top_score = hits[0][1] if hits else 0.0
    result = {"status": "refused", "answer": REFUSAL_TEXT, "citations": [], "top_score": round(top_score, 3)}

    if top_score < min_relevance:
        return {**result, "reason": f"No passage is relevant enough (best match {top_score:.2f} < {min_relevance})."}

    response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT),
                           HumanMessage(content=f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}")])
    text = response.content.strip()

    if text.upper().startswith(NOT_FOUND):
        return {**result, "reason": "The retrieved passages do not answer the question."}

    cited = parse_citations(text, len(hits))
    citations = [citation(n, *hits[n - 1]) for n in cited]
    status = "answered" if citations else "uncited"
    return {"status": status, "answer": text, "citations": citations, "top_score": round(top_score, 3),
            "reason": "" if citations else "The answer does not cite any source. Verify it before relying on it."}
