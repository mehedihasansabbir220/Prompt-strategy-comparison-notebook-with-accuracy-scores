"""Provider-neutral contract for LLM access.

Nothing in this module imports a vendor SDK. The evaluation layer depends only
on :class:`LLMResponse` and the :class:`LLMProvider` protocol, so a second
provider can be added later by writing one adapter — the runner, metrics and
error analysis never change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from src.prompts.base import Prompt


class LLMError(RuntimeError):
    """Base class for every provider failure."""


class LLMAuthError(LLMError):
    """Credentials are missing, invalid, or lack permission.

    Raised rather than returned: a bad key fails every call, so a benchmark
    should stop immediately instead of burning through the sample set.
    """


class LLMRateLimitError(LLMError):
    """The provider rejected the call for quota or rate reasons."""


class LLMTransientError(LLMError):
    """A server-side or network failure that may succeed on retry."""


@dataclass(frozen=True, slots=True)
class UsageMetadata:
    """Token accounting for one call, as reported by the provider.

    Every field is optional: usage reporting is a provider courtesy, not a
    guarantee. Missing values stay ``None`` and are reported as unavailable —
    never silently replaced with zero, which would understate real cost.
    """

    prompt_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def is_available(self) -> bool:
        """Whether the provider reported any usage at all."""
        return any(
            value is not None
            for value in (self.prompt_tokens, self.output_tokens, self.total_tokens)
        )

    def to_dict(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """The complete outcome of one generation call, successful or not.

    A failure is a value, not an exception: the benchmark must record an API
    failure against a sample and carry on, so ``generate`` returns this object
    in both cases. Only unrecoverable misconfiguration raises.
    """

    #: Raw text exactly as returned. Stored unmodified so a prediction can
    #: always be traced back to what the model actually said.
    text: str

    #: Whether usable text came back.
    success: bool

    #: Wall-clock duration of the call, including retries.
    latency_seconds: float

    #: Model identifier that served the request.
    model: str

    usage: UsageMetadata = UsageMetadata()

    #: Provider's stop reason (e.g. ``STOP``, ``MAX_TOKENS``, ``SAFETY``).
    finish_reason: str | None = None

    #: Populated only on failure.
    error_type: str | None = None
    error_message: str | None = None

    #: Number of attempts made, including the successful one.
    attempts: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Flat, JSON-safe view for ``predictions.csv`` and run metadata."""
        return {
            "response_text": self.text,
            "success": self.success,
            "latency_seconds": round(self.latency_seconds, 4),
            "model": self.model,
            "finish_reason": self.finish_reason,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "attempts": self.attempts,
            **self.usage.to_dict(),
        }


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface the evaluation layer depends on."""

    @property
    def model(self) -> str:
        """Identifier of the model this provider calls."""

    def generate(self, prompt: Prompt | str) -> LLMResponse:
        """Send one prompt and return its outcome."""

    def describe(self) -> dict[str, Any]:
        """Return the generation settings, for experiment metadata."""
