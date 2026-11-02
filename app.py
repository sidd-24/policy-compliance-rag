import os
from pathlib import Path

import streamlit as st
from rag.loader import load_and_chunk
from rag.embeddings import build_vectorstore
from rag.chain import answer_question, build_llm

SAMPLE_DIR = Path("data/policies")


st.set_page_config(
    page_title="Policy Q&A",
    page_icon="📄",
    layout="centered",
)

st.title("📄 Policy & Compliance Q&A")
st.caption("Ask questions about your policies. Every answer cites document, page and section, "
           "or says it is not covered.")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_... (or set GROQ_API_KEY)",
        help="Get a free key at console.groq.com",
    ) or os.getenv("GROQ_API_KEY", "")
    model = st.selectbox(
        "Model",
        ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
        index=0,
    )
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. PDFs are split by section\n"
        "2. Chunks are embedded locally\n"
        "3. Your question retrieves the most relevant sections\n"
        "4. If nothing is relevant enough, it refuses\n"
        "5. Otherwise Groq answers from the sources, with citations"
    )

use_samples = st.toggle("Use the sample policies (fictional company)", value=True)
uploaded_files = st.file_uploader("Or upload your own PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    documents = [(f.name, f) for f in uploaded_files]
elif use_samples:
    documents = [(p.name, p) for p in sorted(SAMPLE_DIR.glob("*.pdf"))]
else:
    documents = []

if documents and api_key:
    index_key = tuple((name, getattr(doc, "size", None)) for name, doc in documents)

    if st.session_state.get("index_key") != index_key:
        with st.spinner("Reading and indexing the documents..."):
            chunks = [c for name, doc in documents for c in load_and_chunk(doc, name=name)]
            st.session_state["vectorstore"] = build_vectorstore(chunks)
            st.session_state["index_key"] = index_key
            st.session_state["chat_history"] = []
            st.session_state["chunk_count"] = len(chunks)

        st.success(f"✅ Ready: {len(documents)} document(s), {st.session_state['chunk_count']} chunks indexed.")

elif documents and not api_key:
    st.warning("Please enter your Groq API key in the sidebar or set GROQ_API_KEY.")


def show_result(result):
    if result["status"] == "refused":
        st.info(result["answer"])
        st.caption(result["reason"])
        return
    st.write(result["answer"])
    if result["status"] == "uncited":
        st.warning(result["reason"])
    for c in result["citations"]:
        with st.expander(f"[{c['n']}] {c['title']}, page {c['page']}, section {c['section'] or 'n/a'}"):
            st.markdown(f"> {c['excerpt']}")
            st.caption(f"{c['source']} · relevance {c['score']:.2f}")


if "vectorstore" in st.session_state and api_key:
    st.markdown("---")
    st.subheader("Ask a question")

    for entry in st.session_state.get("chat_history", []):
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            show_result(entry["result"])

    question = st.chat_input("e.g. How quickly must critical vulnerabilities be patched?")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the policies..."):
                result = answer_question(question, st.session_state["vectorstore"], build_llm(api_key, model))
            show_result(result)

        st.session_state["chat_history"].append({"question": question, "result": result})
