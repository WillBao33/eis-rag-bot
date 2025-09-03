from typing import List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm import get_llm, SYSTEM_PROMPT
from .retriever import build_retriever

QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("user", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer briefly"),
])

def format_context(docs):
    """Compact context block with numbered snippets and filename for citations."""
    blocks = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "context")
        blocks.append(f"[{i}] ({src})\n{d.page_content}")
    return "\n\n".join(blocks)

def build_chain():
    retriever = build_retriever()
    llm = get_llm()
    parser = StrOutputParser()

    def run(question: str):
        docs = retriever.invoke(question)
        context = format_context(docs)
        chain = QUESTION_PROMPT | llm | parser
        answer = chain.invoke({"context": context, "question": question})

        return answer, docs

    return run