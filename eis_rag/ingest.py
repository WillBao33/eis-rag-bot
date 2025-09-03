from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from .config import settings

def load_markdown(data_dir: str = "data"):
    loader = DirectoryLoader(data_dir,
                             glob="**/*.md",
                             loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"},
                             )
    return loader.load()

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)

def build_vectorstore(splits):
    embeddings = OpenAIEmbeddings(
        model=settings.MODEL_EMBEDDING,
        api_key=settings.OPENAI_API_KEY,
    )
    vs = Chroma.from_documents(splits, embeddings, persist_directory=settings.PERSIST_DIR)
    return vs

def ingest(data_dir: str = "data"):
    docs = load_markdown(data_dir)
    splits = split_docs(docs)
    vs = build_vectorstore(splits)
    return len(docs), len(splits)

if __name__ == "__main__":
    n_docs, n_chunks = ingest("data")
    print(f"Ingested {n_docs} documents into {n_chunks} chunks.")