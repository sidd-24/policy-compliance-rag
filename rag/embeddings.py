import uuid

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(chunks: list, embeddings=None) -> Chroma:
    # Every index gets its own collection. In-memory Chroma shares collections
    # by name within a process, so a fixed name mixed a new upload with the
    # chunks of the previous one.
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings or get_embeddings(),
        collection_name=f"policies_{uuid.uuid4().hex[:12]}",
        collection_metadata={
            # Cosine similarity, so relevance scores read as 0 (unrelated) to 1 (same meaning)
            "hnsw:space": "cosine",
            # The default HNSW settings sometimes missed the best passage and
            # gave different results per run. A single-threaded build is
            # deterministic, and a wide search is exact for policy-sized corpora.
            "hnsw:num_threads": 1,
            "hnsw:construction_ef": 200,
            "hnsw:search_ef": 200,
        },
    )
