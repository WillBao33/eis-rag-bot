from eis_rag.config import settings
print("OPENAI key loaded:", "*" * max(0, len(settings.OPENAI_API_KEY)-8) + settings.OPENAI_API_KEY[-8:])
print("Embedding model:", settings.MODEL_EMBEDDING)
print("Chat model:", settings.MODEL_CHAT)

# import core deps to confirm install
import gradio, chromadb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
print("Imports OK")