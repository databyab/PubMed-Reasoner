from .base import MockLLM, GroqLLM
from ..config import settings

def create_llm():
    if settings.llm_mode == "mock": return MockLLM()
    return GroqLLM(settings.groq_api_key or "",settings.groq_model,settings.request_timeout)

