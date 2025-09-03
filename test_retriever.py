from eis_rag.retriever import build_retriever

retriever = build_retriever()
print("retriever type:", type(retriever))

query = "What services does the studio provide?"
docs = retriever.get_relevant_documents(query)

print(f"Retrieved {len(docs)} docs for query: {query}")
for i, d in enumerate(docs, 1):
    print(f"\n--- doc {i} ---")
    print("source:", d.metadata.get("source"))
    print(d.page_content[:300], "...")