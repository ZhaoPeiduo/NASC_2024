from functools import lru_cache
from backend.config import settings
from backend.services.llm.base import LLMProvider


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider:
    if settings.llm_provider == "openrouter":
        from backend.services.llm.openrouter import OpenRouterProvider

        return OpenRouterProvider(
            api_key=settings.openrouter_api_key,
            model=settings.openrouter_model,
        )
    elif settings.llm_provider == "local":
        from backend.services.llm.local import LocalModelProvider

        return LocalModelProvider(model_name=settings.local_model_name)
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}")
