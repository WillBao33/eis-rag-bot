from __future__ import annotations
from typing import Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from .config import settings

def get_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(
        model=settings.MODEL_EMBEDDING,
        api_key=settings.OPENAI_API_KEY,
    )
    vs = Chroma(
        embedding_function=embeddings,
        persist_directory=settings.PERSIST_DIR,
    )
    return vs

def build_retriever():
    vs = get_vectorstore()

    try:
        base = vs.as_retriever(search_type="mmr", search_kwargs={"k": settings.TOP_K})
    except Exception as e:
        print("[retriever] Failed to build MMR retriever, falling back to default:", e)

    mq = None
    try:
        from langchain.retrievers.multi_query import MultiQueryRetriever
        llm = ChatOpenAI(
            model=settings.MODEL_CHAT,
            temperature=0,
            api_key=settings.OPENAI_API_KEY,
            )
        mq = MultiQueryRetriever.from_llm(retriever=base, llm=llm)
    except Exception as e:
        print("[retriever] MultiQuery unavailable, using base retriever:", e)

    try:
        if settings.COHERE_API_KEY and settings.COHERE_API_KEY.strip():
            from langchain.retrievers.document_compressors import CohereRerank
            from langchain.retrievers import ContextualCompressionRetriever
            compressor = CohereRerank(
                model=settings.RERANK_MODEL,
                top_n=settings.RERANK_TOP_N,
                cohere_api_key=settings.COHERE_API_KEY,
            )
            reranked = ContextualCompressionRetriever(base_retriever=mq, base_compressor=compressor)
            return reranked
    except Exception as e:
        print("[retriever] Cohere rerank unavailable, continuing without rerank:", e)

    return mq