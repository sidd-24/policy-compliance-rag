from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma


PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions based on the provided document context.
Be concise and accurate. If the answer is not in the context, say so clearly.

Context:
{context}

Question: {question}

Answer:
"""


def build_rag_chain(vectorstore: Chroma, api_key: str, model: str = "llama-3.3-70b-versatile") -> RetrievalQA:
    llm = ChatGroq(
        api_key=api_key,
        model_name=model,
        temperature=0.2,
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain


def ask(chain: RetrievalQA, question: str) -> dict:
    result = chain.invoke({"query": question})

    answer = result["result"]
    sources = list(set(
        doc.metadata.get("page", "?") + 1
        for doc in result["source_documents"]
        if "page" in doc.metadata
    ))
    sources.sort()

    return {"answer": answer, "source_pages": sources}