import streamlit as st
from rag.loader import load_and_chunk
from rag.embeddings import build_vectorstore
from rag.chain import build_rag_chain, ask


st.set_page_config(
    page_title="PDF Q&A",
    page_icon="📄",
    layout="centered",
)

st.title("📄 PDF Q&A")
st.caption("Upload a PDF and ask anything about it. Powered by Groq + LangChain.")

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get a free key at console.groq.com",
    )
    model = st.selectbox(
        "Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0,
    )
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "1. Upload your PDF\n"
        "2. PDF is split into chunks\n"
        "3. Chunks are embedded locally\n"
        "4. Your question retrieves relevant chunks\n"
        "5. Groq LLM generates the answer"
    )

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file and api_key:
    file_key = uploaded_file.name + str(uploaded_file.size)

    if st.session_state.get("file_key") != file_key:
        with st.spinner("Reading and indexing your PDF..."):
            chunks = load_and_chunk(uploaded_file)
            vectorstore = build_vectorstore(chunks)
            chain = build_rag_chain(vectorstore, api_key, model)

            st.session_state["chain"] = chain
            st.session_state["file_key"] = file_key
            st.session_state["chat_history"] = []
            st.session_state["chunk_count"] = len(chunks)

        st.success(f"✅ Ready — {st.session_state['chunk_count']} chunks indexed.")

elif uploaded_file and not api_key:
    st.warning("Please enter your Groq API key in the sidebar to continue.")

if "chain" in st.session_state:
    st.markdown("---")
    st.subheader("Ask a question")

    for entry in st.session_state.get("chat_history", []):
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            if entry["source_pages"]:
                st.caption(f"Sources: page(s) {', '.join(map(str, entry['source_pages']))}")

    question = st.chat_input("Ask something about the document...")

    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(st.session_state["chain"], question)

            st.write(result["answer"])
            if result["source_pages"]:
                st.caption(f"Sources: page(s) {', '.join(map(str, result['source_pages']))}")

        st.session_state["chat_history"].append({
            "question": question,
            "answer": result["answer"],
            "source_pages": result["source_pages"],
        })