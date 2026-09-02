"""
Provider-agnostic LLM client abstraction used by ingestion/metadata_gen.py.

metadata_gen.py talks to `self._llm: LLMClient` and never imports a provider
SDK directly. Today only Gemini (google-genai / Vertex AI) is implemented;
adding openai / anthropic / mistralai / openrouter means writing a new
LLMClient subclass (see gemini.py for the shape) and registering it in
`_PROVIDERS` below — no caller changes.
"""

from __future__ import annotations

from typing import Any

from .base import (
    LLMAPIError,
    LLMClient,
    LLMConnectivityError,
    LLMError,
    LLMGenerationError,
    LLMModelUnavailableError,
    log_llm_error,
)

# provider name -> "module:ClassName", imported lazily so e.g. an unused
# provider's SDK is never required to be installed.
_PROVIDERS: dict[str, str] = {
    "gemini": "ingestion.llm.gemini:GeminiClient",
    # "openai": "ingestion.llm.openai_client:OpenAIClient",
    # "anthropic": "ingestion.llm.anthropic_client:AnthropicClient",
    # "mistralai": "ingestion.llm.mistral_client:MistralClient",
    # "openrouter": "ingestion.llm.openrouter_client:OpenRouterClient",
}


def build_llm_client(provider: str, **kwargs: Any) -> LLMClient:
    """Construct an LLMClient for `provider`.

    Args:
        provider: One of the keys in `_PROVIDERS` (only "gemini" today).
        **kwargs: Forwarded to the provider's constructor.

    Returns:
        A ready-to-use LLMClient.

    Raises:
        ValueError: If `provider` is unknown.
    """
    key = (provider or "").strip().lower()
    target = _PROVIDERS.get(key)
    if target is None:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Known: {sorted(_PROVIDERS)}"
        )
    module_path, class_name = target.split(":")
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


__all__ = [
    "LLMClient",
    "LLMError",
    "LLMConnectivityError",
    "LLMAPIError",
    "LLMGenerationError",
    "LLMModelUnavailableError",
    "log_llm_error",
    "build_llm_client",
]
