"""Tests for the Gemini provider adapter.

Not one test contacts Google. Every provider is constructed with an injected
fake client, retry backoff is monkeypatched to a no-op, and the API key is
removed from the environment so an accidental real call would fail loudly
rather than silently succeed.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from src.config import ENV_API_KEY, AppConfig
from src.llm import (
    GeminiProvider,
    LLMAuthError,
    LLMError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponse,
    LLMTransientError,
    UsageMetadata,
)
from src.prompts import get_strategy
from src.prompts.base import Prompt

REVIEW = "Slow in places, but the ending won me over."


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeModels:
    """Stands in for ``client.models``; records calls and replays scripted results."""

    def __init__(self, results: list[Any]) -> None:
        self._results = list(results)
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        result = self._results.pop(0) if self._results else self._results
        if isinstance(result, Exception):
            raise result
        return result


class FakeClient:
    """Minimal stand-in for ``genai.Client``."""

    def __init__(self, *results: Any) -> None:
        self.models = FakeModels(list(results))


def fake_sdk_response(
    text: str | None = "positive",
    *,
    prompt_tokens: int | None = 120,
    output_tokens: int | None = 2,
    total_tokens: int | None = 122,
    finish_reason: str | None = "STOP",
) -> SimpleNamespace:
    """Build an object shaped like a ``google-genai`` response."""
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=output_tokens,
        total_token_count=total_tokens,
    )
    candidate = SimpleNamespace(finish_reason=SimpleNamespace(name=finish_reason))
    return SimpleNamespace(text=text, usage_metadata=usage, candidates=[candidate])


class ApiError(Exception):
    """Stand-in for an SDK error carrying an HTTP status code."""

    def __init__(self, code: int, message: str = "api failure") -> None:
        super().__init__(message)
        self.code = code


@pytest.fixture(autouse=True)
def no_real_api(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee hermetic tests.

    Removing the environment variable is not enough on its own: ``get_api_key``
    also loads ``.env``, which on a developer machine holds a working key. Both
    doors are shut, so a test that accidentally builds a real client fails with
    an auth error instead of silently calling Google.
    """
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    monkeypatch.setattr("src.config.load_environment", lambda *a, **k: False)
    monkeypatch.setattr("src.llm.gemini.time.sleep", lambda _seconds: None)


@pytest.fixture
def config() -> AppConfig:
    return AppConfig(model="gemini-test-model", temperature=0.0)


def make_provider(*results: Any, config: AppConfig | None = None, **kwargs: Any) -> tuple[GeminiProvider, FakeClient]:
    client = FakeClient(*results)
    provider = GeminiProvider(config or AppConfig(model="gemini-test-model"), client=client, **kwargs)
    return provider, client


# ---------------------------------------------------------------------------
# Contract and configuration
# ---------------------------------------------------------------------------


class TestProviderContract:
    def test_satisfies_the_provider_protocol(self) -> None:
        provider, _ = make_provider()
        assert isinstance(provider, LLMProvider)

    def test_exposes_the_configured_model(self, config: AppConfig) -> None:
        provider, _ = make_provider(config=config)
        assert provider.model == "gemini-test-model"

    def test_describe_reports_generation_settings(self, config: AppConfig) -> None:
        provider, _ = make_provider(config=config)
        described = provider.describe()
        assert described["provider"] == "gemini"
        assert described["model"] == "gemini-test-model"
        assert described["temperature"] == 0.0
        assert "max_output_tokens" in described

    def test_describe_contains_no_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_API_KEY, "super-secret-key-value")
        provider, _ = make_provider()
        assert "super-secret-key-value" not in str(provider.describe())

    def test_construction_requires_no_api_key(self) -> None:
        # No key in the environment, yet building a provider must not fail.
        GeminiProvider(AppConfig(), client=FakeClient())


class TestRequestConstruction:
    def test_sends_model_and_prompt_text(self, config: AppConfig) -> None:
        provider, client = make_provider(fake_sdk_response(), config=config)
        provider.generate(Prompt(user_text="classify: positive or negative"))

        call = client.models.calls[0]
        assert call["model"] == "gemini-test-model"
        assert call["contents"] == "classify: positive or negative"

    def test_applies_the_configured_temperature(self) -> None:
        provider, client = make_provider(
            fake_sdk_response(), config=AppConfig(temperature=0.0)
        )
        provider.generate(Prompt(user_text="anything"))
        assert client.models.calls[0]["config"].temperature == 0.0

    def test_caps_output_tokens(self) -> None:
        provider, client = make_provider(fake_sdk_response(), max_output_tokens=16)
        provider.generate(Prompt(user_text="anything"))
        assert client.models.calls[0]["config"].max_output_tokens == 16

    def test_forwards_the_system_instruction_when_present(self) -> None:
        provider, client = make_provider(fake_sdk_response())
        provider.generate(
            Prompt(user_text="classify this", system_instruction="you are an annotator")
        )
        assert client.models.calls[0]["config"].system_instruction == "you are an annotator"

    def test_omits_the_system_instruction_when_absent(self) -> None:
        provider, client = make_provider(fake_sdk_response())
        provider.generate(Prompt(user_text="classify this"))
        assert client.models.calls[0]["config"].system_instruction is None

    def test_accepts_a_bare_string_for_manual_checks(self) -> None:
        provider, client = make_provider(fake_sdk_response())
        provider.generate("just a string prompt")
        assert client.models.calls[0]["contents"] == "just a string prompt"

    def test_works_with_a_real_strategy_prompt(self) -> None:
        provider, client = make_provider(fake_sdk_response())
        prompt = get_strategy("role_based").build_prompt(REVIEW)
        response = provider.generate(prompt)
        assert response.success
        assert REVIEW in client.models.calls[0]["contents"]
        assert client.models.calls[0]["config"].system_instruction is not None

    def test_settings_are_identical_across_strategies(self) -> None:
        """The controlled-experiment guarantee, at the provider boundary."""
        provider, client = make_provider(*[fake_sdk_response()] * 4)
        for name in ("zero_shot", "structured", "reasoning", "role_based"):
            provider.generate(get_strategy(name).build_prompt(REVIEW))

        settings = {
            (call["config"].temperature, call["config"].max_output_tokens, call["model"])
            for call in client.models.calls
        }
        assert len(settings) == 1


# ---------------------------------------------------------------------------
# Successful responses
# ---------------------------------------------------------------------------


class TestSuccessfulGeneration:
    def test_returns_the_raw_text(self) -> None:
        provider, _ = make_provider(fake_sdk_response(text="positive"))
        response = provider.generate(Prompt(user_text="x"))
        assert isinstance(response, LLMResponse)
        assert response.success is True
        assert response.text == "positive"
        assert response.error_type is None

    def test_strips_surrounding_whitespace_only(self) -> None:
        provider, _ = make_provider(fake_sdk_response(text='  {"sentiment": "positive"}\n'))
        assert provider.generate(Prompt(user_text="x")).text == '{"sentiment": "positive"}'

    def test_records_latency(self) -> None:
        provider, _ = make_provider(fake_sdk_response())
        response = provider.generate(Prompt(user_text="x"))
        assert response.latency_seconds >= 0.0

    def test_captures_usage_metadata(self) -> None:
        provider, _ = make_provider(
            fake_sdk_response(prompt_tokens=310, output_tokens=3, total_tokens=313)
        )
        usage = provider.generate(Prompt(user_text="x")).usage
        assert usage.prompt_tokens == 310
        assert usage.output_tokens == 3
        assert usage.total_tokens == 313
        assert usage.is_available is True

    def test_missing_usage_metadata_is_none_not_zero(self) -> None:
        raw = SimpleNamespace(text="positive", usage_metadata=None, candidates=[])
        provider, _ = make_provider(raw)
        usage = provider.generate(Prompt(user_text="x")).usage
        assert usage.to_dict() == {
            "prompt_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
        }
        assert usage.is_available is False

    def test_captures_the_finish_reason(self) -> None:
        provider, _ = make_provider(fake_sdk_response(finish_reason="STOP"))
        assert provider.generate(Prompt(user_text="x")).finish_reason == "STOP"

    def test_records_the_model_that_served_the_call(self, config: AppConfig) -> None:
        provider, _ = make_provider(fake_sdk_response(), config=config)
        assert provider.generate(Prompt(user_text="x")).model == "gemini-test-model"

    def test_response_serialises_for_storage(self) -> None:
        provider, _ = make_provider(fake_sdk_response())
        payload = provider.generate(Prompt(user_text="x")).to_dict()
        assert payload["response_text"] == "positive"
        assert payload["success"] is True
        assert payload["prompt_tokens"] == 120


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


class TestEmptyResponse:
    @pytest.mark.parametrize("text", [None, "", "   "])
    def test_empty_body_is_a_failure_not_an_empty_prediction(self, text: str | None) -> None:
        provider, _ = make_provider(fake_sdk_response(text=text, finish_reason="SAFETY"))
        response = provider.generate(Prompt(user_text="x"))
        assert response.success is False
        assert response.text == ""
        assert response.error_type == "EmptyResponse"
        assert "SAFETY" in (response.error_message or "")

    def test_max_tokens_failure_names_the_fix(self) -> None:
        provider, _ = make_provider(fake_sdk_response(text="", finish_reason="MAX_TOKENS"))
        message = provider.generate(Prompt(user_text="x")).error_message or ""
        assert "max_output_tokens" in message
        assert "thinking_budget" in message

    def test_usage_is_still_captured_on_an_empty_response(self) -> None:
        provider, _ = make_provider(fake_sdk_response(text=None, total_tokens=99))
        assert provider.generate(Prompt(user_text="x")).usage.total_tokens == 99


class TestErrorHandling:
    def test_api_failure_is_returned_not_raised(self) -> None:
        provider, _ = make_provider(ApiError(500), ApiError(500), ApiError(500))
        response = provider.generate(Prompt(user_text="x"))
        assert response.success is False
        assert response.text == ""
        assert response.error_type == "LLMTransientError"
        assert response.attempts == 3

    def test_auth_failure_raises_immediately(self) -> None:
        provider, client = make_provider(ApiError(403, "permission denied"))
        with pytest.raises(LLMAuthError):
            provider.generate(Prompt(user_text="x"))
        # Fails fast: no retries burned on a key that cannot work.
        assert len(client.models.calls) == 1

    def test_rate_limit_is_classified(self) -> None:
        provider, _ = make_provider(*[ApiError(429, "quota exceeded")] * 3)
        response = provider.generate(Prompt(user_text="x"))
        assert response.error_type == "LLMRateLimitError"

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 408])
    def test_server_errors_are_transient(self, status: int) -> None:
        provider, _ = make_provider(*[ApiError(status)] * 3)
        assert provider.generate(Prompt(user_text="x")).error_type == "LLMTransientError"

    def test_unknown_status_is_a_plain_error(self) -> None:
        provider, _ = make_provider(*[ApiError(418, "teapot")] * 3)
        assert provider.generate(Prompt(user_text="x")).error_type == "LLMError"

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("Invalid API key provided", LLMAuthError),
            ("Request timed out", LLMTransientError),
            ("RESOURCE_EXHAUSTED: quota", LLMRateLimitError),
            ("something entirely unexpected", LLMError),
        ],
    )
    def test_classifies_errors_without_a_status_code(
        self, message: str, expected: type[LLMError]
    ) -> None:
        assert type(GeminiProvider._classify_error(Exception(message))) is expected

    def test_failure_response_records_a_message(self) -> None:
        provider, _ = make_provider(*[ApiError(503, "backend unavailable")] * 3)
        response = provider.generate(Prompt(user_text="x"))
        assert response.error_message
        assert response.latency_seconds >= 0.0


class TestRetryBehaviour:
    def test_retries_then_succeeds(self) -> None:
        provider, client = make_provider(ApiError(503), fake_sdk_response(text="negative"))
        response = provider.generate(Prompt(user_text="x"))
        assert response.success is True
        assert response.text == "negative"
        assert response.attempts == 2
        assert len(client.models.calls) == 2

    def test_stops_at_max_retries(self) -> None:
        provider, client = make_provider(*[ApiError(503)] * 5, max_retries=2)
        response = provider.generate(Prompt(user_text="x"))
        assert response.success is False
        assert len(client.models.calls) == 2

    def test_backoff_grows_exponentially(self, monkeypatch: pytest.MonkeyPatch) -> None:
        delays: list[float] = []
        monkeypatch.setattr("src.llm.gemini.time.sleep", delays.append)
        provider, _ = make_provider(*[ApiError(503)] * 3, retry_base_delay=1.0)
        provider.generate(Prompt(user_text="x"))
        assert delays == [1.0, 2.0]

    def test_successful_first_attempt_does_not_sleep(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        delays: list[float] = []
        monkeypatch.setattr("src.llm.gemini.time.sleep", delays.append)
        provider, _ = make_provider(fake_sdk_response())
        provider.generate(Prompt(user_text="x"))
        assert delays == []


# ---------------------------------------------------------------------------
# Credential handling
# ---------------------------------------------------------------------------


class TestCredentialSafety:
    def test_missing_key_raises_auth_error_on_first_call(self) -> None:
        provider = GeminiProvider(AppConfig())  # no injected client
        with pytest.raises(LLMAuthError, match=ENV_API_KEY):
            provider.generate(Prompt(user_text="x"))

    def test_key_is_never_stored_on_the_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "super-secret-key-value"
        monkeypatch.setenv(ENV_API_KEY, secret)
        provider, _ = make_provider(fake_sdk_response())
        assert secret not in repr(provider)
        assert secret not in str(vars(provider))

    def test_key_never_appears_in_logs(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        secret = "super-secret-key-value"
        monkeypatch.setenv(ENV_API_KEY, secret)
        provider, _ = make_provider(*[ApiError(503)] * 3)
        with caplog.at_level(logging.DEBUG, logger="src.llm.gemini"):
            provider.generate(Prompt(user_text="x"))
        assert secret not in caplog.text

    def test_key_never_appears_in_a_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "super-secret-key-value"
        monkeypatch.setenv(ENV_API_KEY, secret)
        provider, _ = make_provider(ApiError(500, f"failure using key {secret[:4]}"))
        response = provider.generate(Prompt(user_text="x"))
        assert secret not in str(response.to_dict())


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


class TestUsageMetadata:
    def test_defaults_to_unavailable(self) -> None:
        assert UsageMetadata().is_available is False

    def test_partial_usage_counts_as_available(self) -> None:
        assert UsageMetadata(total_tokens=10).is_available is True


class TestLLMResponseValue:
    def test_is_immutable(self) -> None:
        response = LLMResponse(text="positive", success=True, latency_seconds=0.1, model="m")
        with pytest.raises(Exception):
            response.text = "negative"  # type: ignore[misc]

    def test_to_dict_rounds_latency(self) -> None:
        response = LLMResponse(
            text="positive", success=True, latency_seconds=0.123456789, model="m"
        )
        assert response.to_dict()["latency_seconds"] == 0.1235
