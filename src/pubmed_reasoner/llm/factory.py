from .base import MockLLM, GroqLLM, GeminiLLM
from ..config import settings

def create_llm():
    if settings.llm_mode == "mock": return MockLLM()
    if settings.llm_mode == "gemini":
        return GeminiLLM(settings.gemini_api_key or "",settings.gemini_model,settings.request_timeout)
    return GroqLLM(settings.groq_api_key or "",settings.groq_model,settings.request_timeout)

