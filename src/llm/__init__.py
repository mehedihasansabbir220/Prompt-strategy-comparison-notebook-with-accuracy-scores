"""LLM provider adapters.

The evaluation layer imports only the neutral contract from :mod:`src.llm.base`;
:class:`~src.llm.gemini.GeminiProvider` is one implementation of it.

``GeminiProvider`` is re-exported lazily (PEP 562). Importing it eagerly here
would load the module while ``python -m src.llm.gemini`` is starting it, which
emits a ``RuntimeWarning`` about double execution — and would pull the SDK into
every import of this package, including test runs that never use it.
"""

from typing import TYPE_CHECKING, Any

from src.llm.base import (
    LLMAuthError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    LLMTransientError,
    UsageMetadata,
)

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from src.llm.gemini import GeminiProvider

__all__ = [
    "GeminiProvider",
    "LLMAuthError",
    "LLMError",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMResponse",
    "LLMTransientError",
    "UsageMetadata",
]


def __getattr__(name: str) -> Any:
    """Resolve ``GeminiProvider`` on first access instead of at import time."""
    if name == "GeminiProvider":
        from src.llm.gemini import GeminiProvider

        return GeminiProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
