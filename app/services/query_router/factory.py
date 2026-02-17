from .base import BaseQueryRouter
from .llm_router import LLMQueryRouter

def get_query_router(provider: str = "llm") -> BaseQueryRouter:
    """
    Factory to get the query router instance.
    Currently only supports 'llm' provider.
    """
    if provider == "llm":
        return LLMQueryRouter()
    else:
        # Fallback or raise error
        return LLMQueryRouter()
