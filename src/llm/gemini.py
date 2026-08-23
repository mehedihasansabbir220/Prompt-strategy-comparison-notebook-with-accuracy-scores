"""Gemini provider adapter.

Wraps the ``google-genai`` SDK behind the provider-neutral contract in
:mod:`src.llm.base`. Responsibilities: build the client from an environment-held
key, send one prompt, time it, capture usage, classify and retry failures, and
log what happened without ever revealing the key.

Run a real end-to-end check from the command line::

    python -m src.llm.gemini "The ending completely won me over."

That is the only code path in this module that contacts Google.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Final

from src.config import AppConfig, get_api_key
from src.llm.base import (
    LLMAuthError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    LLMTransientError,
    UsageMetadata,
)
from src.prompts.base import Prompt

logger = logging.getLogger(__name__)

#: Answers are one word or a small JSON object. Capping output keeps cost and
#: latency low and makes a runaway essay impossible. Held constant across every
#: strategy, so it is a controlled setting rather than a variable.
DEFAULT_MAX_OUTPUT_TOKENS: Final[int] = 64

#: Gemini models spend output tokens on internal "thinking" by default.
#: Disabling it (budget 0) keeps the comparison about *prompt* design: with
#: thinking on, a model-side reasoning process of unknown length would vary
#: between strategies and confound the result - and it consumes the output
#: budget, so short-answer prompts return MAX_TOKENS with no text at all.
#:
#: Support is model-dependent, which constrains the model choice: verified on
#: 2026-08-23, gemini-3.5-flash accepts budget 0, gemini-3.6-flash rejects it
#: with HTTP 400, and gemini-3.7-flash cannot disable thinking either. Set to
#: ``None`` to leave the provider default in place.
DEFAULT_THINKING_BUDGET: Final[int | None] = 0

DEFAULT_MAX_RETRIES: Final[int] = 3

#: Base for exponential backoff: 1s, 2s, 4s between attempts.
DEFAULT_RETRY_BASE_DELAY: Final[float] = 1.0

#: HTTP statuses worth retrying. 429 is quota; 5xx are server-side.
RETRYABLE_STATUS_CODES: Final[frozenset[int]] = frozenset({408, 429, 500, 502, 503, 504})
AUTH_STATUS_CODES: Final[frozenset[int]] = frozenset({401, 403})


class GeminiProvider:
    """Calls the Gemini API for one prompt at a time.

    The client is created lazily on first use, so constructing a provider needs
    no API key. That is what lets the unit tests build one with an injected fake
    client and never touch the network.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        client: Any | None = None,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_budget: int | None = DEFAULT_THINKING_BUDGET,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    ) -> None:
        """Configure the provider.

        Args:
            config: Experiment configuration supplying model and temperature.
                Defaults to :meth:`AppConfig.from_env`.
            client: Pre-built client. Injected by tests; production leaves it
                ``None`` so the key is read from the environment on first call.
            max_output_tokens: Hard cap on the answer length.
            thinking_budget: Output tokens the model may spend thinking, or
                ``None`` to leave the provider default.
            max_retries: Total attempts per call for retryable failures.
            retry_base_delay: Seconds for the first backoff, doubling each time.
        """
        self._config = config or AppConfig.from_env()
        self._client = client
        self.max_output_tokens = max_output_tokens
        self.thinking_budget = thinking_budget
        self.max_retries = max(1, max_retries)
        self.retry_base_delay = retry_base_delay

    # -- identity -----------------------------------------------------------

    @property
    def model(self) -> str:
        """Model identifier used for every call."""
        return self._config.model

    @property
    def temperature(self) -> float:
        return self._config.temperature

    def describe(self) -> dict[str, Any]:
        """Return the generation settings for experiment metadata.

        Contains no credentials — the key is never stored on this object.
        """
        return {
            "provider": "gemini",
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "thinking_budget": self.thinking_budget,
            "max_retries": self.max_retries,
        }

    # -- client -------------------------------------------------------------

    def _ensure_client(self) -> Any:
        """Return the client, building it from the environment on first use.

        Raises:
            LLMAuthError: If no usable key is configured.
        """
        if self._client is not None:
            return self._client

        from google import genai  # imported lazily: keeps unit tests SDK-free

        try:
            api_key = get_api_key()
        except Exception as exc:  # MissingAPIKeyError and any config failure
            raise LLMAuthError(str(exc)) from exc

        logger.info("Initialising Gemini client for model %r", self.model)
        self._client = genai.Client(api_key=api_key)
        return self._client

    def _build_request_config(self, prompt: Prompt) -> Any:
        """Assemble the generation config, identical for every strategy.

        The prompt's system instruction is the only per-strategy input; the
        decoding parameters are fixed by configuration.
        """
        from google.genai import types

        kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        if prompt.system_instruction:
            kwargs["system_instruction"] = prompt.system_instruction
        if self.thinking_budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        return types.GenerateContentConfig(**kwargs)

    # -- generation ---------------------------------------------------------

    def generate(self, prompt: Prompt | str) -> LLMResponse:
        """Send one prompt and return its outcome.

        API failures are returned as unsuccessful :class:`LLMResponse` objects
        rather than raised, so the benchmark can record the failure against the
        sample and continue. The single exception is authentication: a bad key
        fails every call, so it raises immediately instead of producing a run
        full of identical errors.

        Args:
            prompt: A :class:`Prompt`, or a bare string for quick manual checks.

        Returns:
            An :class:`LLMResponse` with text, latency, usage and status.

        Raises:
            LLMAuthError: If credentials are missing, invalid or unauthorised.
        """
        active = Prompt(user_text=prompt) if isinstance(prompt, str) else prompt
        client = self._ensure_client()
        request_config = self._build_request_config(active)

        started = time.perf_counter()
        last_error: LLMError | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                raw = client.models.generate_content(
                    model=self.model,
                    contents=active.user_text,
                    config=request_config,
                )
            except Exception as exc:  # noqa: BLE001 - classified immediately below
                error = self._classify_error(exc)
                if isinstance(error, LLMAuthError):
                    logger.error("Gemini authentication failed: %s", error)
                    raise error from exc
                last_error = error
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Gemini call failed (%s, attempt %d/%d); retrying in %.1fs",
                        type(error).__name__,
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                break

            elapsed = time.perf_counter() - started
            return self._build_response(raw, elapsed, attempt)

        elapsed = time.perf_counter() - started
        logger.error(
            "Gemini call failed after %d attempt(s) in %.2fs: %s",
            self.max_retries,
            elapsed,
            last_error,
        )
        return LLMResponse(
            text="",
            success=False,
            latency_seconds=elapsed,
            model=self.model,
            error_type=type(last_error).__name__ if last_error else "LLMError",
            error_message=str(last_error) if last_error else "unknown failure",
            attempts=self.max_retries,
        )

    # -- response handling --------------------------------------------------

    def _build_response(self, raw: Any, elapsed: float, attempts: int) -> LLMResponse:
        """Convert a raw SDK response into an :class:`LLMResponse`.

        An empty body is treated as a failure, not as an empty prediction: it
        usually means a safety block or a truncation, and must be counted as an
        API failure rather than scored as a wrong answer.
        """
        text = (getattr(raw, "text", None) or "").strip()
        usage = self._extract_usage(raw)
        finish_reason = self._extract_finish_reason(raw)

        if not text:
            logger.warning(
                "Gemini returned an empty response (finish_reason=%s) in %.2fs",
                finish_reason,
                elapsed,
            )
            return LLMResponse(
                text="",
                success=False,
                latency_seconds=elapsed,
                model=self.model,
                usage=usage,
                finish_reason=finish_reason,
                error_type="EmptyResponse",
                error_message=self._empty_response_message(finish_reason),
                attempts=attempts,
            )

        logger.debug(
            "Gemini responded in %.2fs (%s tokens total, finish_reason=%s)",
            elapsed,
            usage.total_tokens if usage.is_available else "unreported",
            finish_reason,
        )
        return LLMResponse(
            text=text,
            success=True,
            latency_seconds=elapsed,
            model=self.model,
            usage=usage,
            finish_reason=finish_reason,
            attempts=attempts,
        )

    @staticmethod
    def _empty_response_message(finish_reason: str | None) -> str:
        """Explain an empty body, naming the fix when the cause is knowable.

        ``MAX_TOKENS`` with no text means the model spent its whole output
        budget on internal thinking. That is a configuration problem, not a
        model failure, so the message says which knob to turn.
        """
        base = f"Model returned no text (finish_reason={finish_reason})"
        if finish_reason == "MAX_TOKENS":
            return (
                f"{base}. The output budget was consumed before any answer was "
                "emitted - raise max_output_tokens, or set thinking_budget=0 on "
                "a model that supports disabling it."
            )
        return base

    @staticmethod
    def _extract_usage(raw: Any) -> UsageMetadata:
        """Read token counts defensively; absent fields stay ``None``."""
        metadata = getattr(raw, "usage_metadata", None)
        if metadata is None:
            return UsageMetadata()

        def _read(*names: str) -> int | None:
            for name in names:
                value = getattr(metadata, name, None)
                if isinstance(value, int):
                    return value
            return None

        return UsageMetadata(
            prompt_tokens=_read("prompt_token_count"),
            output_tokens=_read("candidates_token_count", "output_token_count"),
            total_tokens=_read("total_token_count"),
        )

    @staticmethod
    def _extract_finish_reason(raw: Any) -> str | None:
        """Read the stop reason of the first candidate, if the SDK exposes one."""
        candidates = getattr(raw, "candidates", None) or []
        if not candidates:
            return None
        reason = getattr(candidates[0], "finish_reason", None)
        if reason is None:
            return None
        return getattr(reason, "name", None) or str(reason)

    @staticmethod
    def _classify_error(exc: Exception) -> LLMError:
        """Map an SDK exception onto the provider-neutral error hierarchy.

        Classification drives behaviour: auth errors abort the run, rate-limit
        and server errors are retried, everything else fails the single sample.
        """
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        message = str(exc)

        if isinstance(status, int):
            if status in AUTH_STATUS_CODES:
                return LLMAuthError(f"Authentication failed (HTTP {status})")
            if status == 429:
                return LLMRateLimitError(f"Rate limit or quota exceeded: {message}")
            if status in RETRYABLE_STATUS_CODES:
                return LLMTransientError(f"Transient provider error (HTTP {status})")
            return LLMError(f"Provider error (HTTP {status}): {message}")

        lowered = message.lower()
        if any(token in lowered for token in ("api key", "unauthenticated", "permission denied")):
            return LLMAuthError("Authentication failed: check GEMINI_API_KEY")
        if any(token in lowered for token in ("rate limit", "quota", "resource_exhausted")):
            return LLMRateLimitError(f"Rate limit or quota exceeded: {message}")
        if any(token in lowered for token in ("timeout", "timed out", "connection", "unavailable")):
            return LLMTransientError(f"Transient network error: {message}")
        return LLMError(f"Provider error: {message}")


# ---------------------------------------------------------------------------
# Manual smoke test (the only code path here that contacts Google)
# ---------------------------------------------------------------------------


def smoke_test(review: str | None = None, *, strategy: str = "zero_shot") -> LLMResponse:
    """Send exactly one real request and print a readable summary.

    Intended for manual verification that the key, model and network path all
    work. Never called by the test suite.

    Args:
        review: Review text to classify. A short default is used if omitted.
        strategy: Which prompt strategy to render the request with.

    Returns:
        The :class:`LLMResponse`, so it can also be used from a notebook cell.
    """
    from src.prompts import get_strategy

    text = review or "Slow in places, but the ending completely won me over."
    prompt = get_strategy(strategy).build_prompt(text)
    provider = GeminiProvider()

    print(f"model      : {provider.model}")
    print(f"strategy   : {strategy}")
    print(f"review     : {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"prompt size: {prompt.char_count} chars")
    print("-" * 60)

    response = provider.generate(prompt)

    print(f"success    : {response.success}")
    print(f"latency    : {response.latency_seconds:.2f}s")
    print(f"attempts   : {response.attempts}")
    print(f"finish     : {response.finish_reason}")
    print(f"usage      : {response.usage.to_dict() if response.usage.is_available else 'not reported'}")
    print(f"raw text   : {response.text!r}")
    if not response.success:
        print(f"error      : {response.error_type}: {response.error_message}")
    return response


def _main() -> int:
    """Entry point for ``python -m src.llm.gemini [review]``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Send one real request to the Gemini API to verify local setup."
    )
    parser.add_argument("review", nargs="?", help="Review text to classify.")
    parser.add_argument(
        "--strategy",
        default="zero_shot",
        help="Prompt strategy to use (default: zero_shot).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s | %(message)s",
    )

    try:
        response = smoke_test(args.review, strategy=args.strategy)
    except LLMAuthError as exc:
        print(f"\nAuthentication error: {exc}")
        return 2
    return 0 if response.success else 1


if __name__ == "__main__":
    raise SystemExit(_main())
