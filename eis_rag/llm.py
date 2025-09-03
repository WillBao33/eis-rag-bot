from langchain_openai import ChatOpenAI
from .config import settings

SYSTEM_PROMPT = (
    "You are a helpful assistant for the Ontario Tech University Engineering Innovation Studio (EIS). "
    "Answer ONLY using the provided context. If the answer is not present in the context, say you don't know "
    "and suggest asking studio staff or checking the handbook. "
    "When asked broadly about services/capabilities, list core services first "
    "(e.g., 3D printing, electronics benches, computer workstations, mentorship)."
)

def get_llm():
    return ChatOpenAI(
        model=settings.MODEL_CHAT,
        temperature=0.2,
        api_key=settings.OPENAI_API_KEY,
    )