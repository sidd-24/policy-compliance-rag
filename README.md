cat > README.md << 'EOF'
# 📄 RAG-based PDF Q&A

Ask natural language questions about any PDF using a local embedding model and Groq as the LLM. No paid embeddings API required.

## Stack

| Layer | Tool |
|---|---|
| LLM | Groq (Llama 3 / Mixtral) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local) |
| Vector Store | ChromaDB (in-memory) |
| Orchestration | LangChain |
| UI | Streamlit |

## How it works
```
PDF → chunks → embeddings → ChromaDB
                                 ↑
Question → embed → similarity search → top-k chunks → Groq LLM → Answer
```

## Setup

1. Clone the repo
```bash
git clone https://github.com/sidd-24/rag-pdf-qa
cd rag-pdf-qa
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Get a free Groq API key at [console.groq.com](https://console.groq.com)

4. Run the app
```bash
streamlit run app.py
```

## Project Structure
```
rag-pdf-qa/
├── app.py              # Streamlit UI
├── rag/
│   ├── loader.py       # PDF loading and chunking
│   ├── embeddings.py   # HuggingFace embeddings + ChromaDB
│   └── chain.py        # RAG chain with Groq LLM
├── requirements.txt
└── .gitignore
```
EOF