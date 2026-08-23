"""Tests for the configuration layer.

These tests never read the developer's real ``.env``: every test that touches
the environment either clears the relevant variables or passes
``load_env_file=False``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from src.config import (
    API_KEY_PLACEHOLDER,
    APP_NAME,
    DEFAULT_BENCHMARK_SAMPLE_SIZE,
    DEFAULT_DEV_SAMPLE_SIZE,
    DEFAULT_MODEL,
    DEFAULT_RANDOM_SEED,
    DEFAULT_STRATEGIES,
    DEFAULT_TEMPERATURE,
    ENV_API_KEY,
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    LABEL_UNKNOWN,
    MAX_TEMPERATURE,
    PROJECT_ROOT,
    SENTIMENT_LABELS,
    AppConfig,
    ConfigurationError,
    MissingAPIKeyError,
    PromptStrategy,
    get_api_key,
    get_config,
    has_api_key,
)

PROMPTBENCH_ENV_VARS = (
    "PROMPTBENCH_MODEL",
    "PROMPTBENCH_TEMPERATURE",
    "PROMPTBENCH_RANDOM_SEED",
    "PROMPTBENCH_DEV_SAMPLE_SIZE",
    "PROMPTBENCH_BENCHMARK_SAMPLE_SIZE",
    "PROMPTBENCH_RESULTS_DIR",
    "PROMPTBENCH_LOG_LEVEL",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Remove every PromptBench variable so tests start from module defaults."""
    for name in (*PROMPTBENCH_ENV_VARS, ENV_API_KEY):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# Label vocabulary
# ---------------------------------------------------------------------------


class TestLabels:
    def test_canonical_order_matches_imdb_encoding(self) -> None:
        # IMDb encodes 0 = negative, 1 = positive; confusion matrices depend on it.
        assert SENTIMENT_LABELS == (LABEL_NEGATIVE, LABEL_POSITIVE)
        assert SENTIMENT_LABELS.index(LABEL_NEGATIVE) == 0
        assert SENTIMENT_LABELS.index(LABEL_POSITIVE) == 1

    def test_unknown_is_not_a_sentiment_class(self) -> None:
        assert LABEL_UNKNOWN not in SENTIMENT_LABELS

    def test_labels_are_immutable(self) -> None:
        assert isinstance(SENTIMENT_LABELS, tuple)


# ---------------------------------------------------------------------------
# Prompt strategies
# ---------------------------------------------------------------------------


class TestPromptStrategy:
    def test_all_six_strategies_are_defined(self) -> None:
        assert len(PromptStrategy) == 6

    def test_members_are_unique(self) -> None:
        values = [s.value for s in PromptStrategy]
        assert len(set(values)) == len(values)

    def test_behaves_as_a_string(self) -> None:
        # StrEnum members can be written straight into DataFrames and JSON.
        assert PromptStrategy.ZERO_SHOT == "zero_shot"
        assert f"{PromptStrategy.FEW_SHOT}" == "few_shot"

    def test_default_order_starts_with_baseline_and_ends_with_combined(self) -> None:
        assert DEFAULT_STRATEGIES[0] is PromptStrategy.ZERO_SHOT
        assert DEFAULT_STRATEGIES[-1] is PromptStrategy.COMBINED
        assert set(DEFAULT_STRATEGIES) == set(PromptStrategy)


# ---------------------------------------------------------------------------
# Defaults and derived values
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_field_values(self) -> None:
        cfg = AppConfig()
        assert cfg.app_name == APP_NAME
        assert cfg.model == DEFAULT_MODEL
        assert cfg.temperature == DEFAULT_TEMPERATURE
        assert cfg.random_seed == DEFAULT_RANDOM_SEED
        assert cfg.dev_sample_size == DEFAULT_DEV_SAMPLE_SIZE
        assert cfg.benchmark_sample_size == DEFAULT_BENCHMARK_SAMPLE_SIZE
        assert cfg.labels == SENTIMENT_LABELS
        assert cfg.prompt_strategies == DEFAULT_STRATEGIES

    def test_temperature_defaults_to_zero_for_repeatability(self) -> None:
        assert AppConfig().temperature == 0.0

    def test_dev_sample_is_smaller_than_benchmark_sample(self) -> None:
        cfg = AppConfig()
        assert cfg.dev_sample_size < cfg.benchmark_sample_size

    def test_results_paths_are_derived_from_results_dir(self) -> None:
        cfg = AppConfig(results_dir=Path("/tmp/pb-results"))
        assert cfg.experiments_dir == Path("/tmp/pb-results/experiments")
        assert cfg.latest_dir == Path("/tmp/pb-results/latest")

    def test_default_results_dir_is_inside_the_repository(self) -> None:
        assert AppConfig().results_dir == PROJECT_ROOT / "results"


class TestImmutability:
    def test_config_cannot_be_mutated(self) -> None:
        cfg = AppConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.temperature = 1.0  # type: ignore[misc]

    def test_with_overrides_returns_a_new_validated_copy(self) -> None:
        cfg = AppConfig()
        variant = cfg.with_overrides(benchmark_sample_size=25)
        assert variant.benchmark_sample_size == 25
        assert cfg.benchmark_sample_size == DEFAULT_BENCHMARK_SAMPLE_SIZE
        assert variant.model == cfg.model

    def test_with_overrides_still_validates(self) -> None:
        with pytest.raises(ConfigurationError):
            AppConfig().with_overrides(temperature=99.0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("temperature", [-0.1, MAX_TEMPERATURE + 0.1, 100.0])
    def test_rejects_out_of_range_temperature(self, temperature: float) -> None:
        with pytest.raises(ConfigurationError, match="temperature"):
            AppConfig(temperature=temperature)

    @pytest.mark.parametrize("temperature", [0.0, 1.0, MAX_TEMPERATURE])
    def test_accepts_in_range_temperature(self, temperature: float) -> None:
        assert AppConfig(temperature=temperature).temperature == temperature

    @pytest.mark.parametrize("size", [0, -1])
    def test_rejects_non_positive_sample_sizes(self, size: int) -> None:
        with pytest.raises(ConfigurationError, match="benchmark_sample_size"):
            AppConfig(benchmark_sample_size=size)
        with pytest.raises(ConfigurationError, match="dev_sample_size"):
            AppConfig(dev_sample_size=size)

    def test_rejects_empty_model(self) -> None:
        with pytest.raises(ConfigurationError, match="model"):
            AppConfig(model="   ")

    def test_rejects_empty_app_name(self) -> None:
        with pytest.raises(ConfigurationError, match="app_name"):
            AppConfig(app_name="")

    def test_rejects_duplicate_strategies(self) -> None:
        duplicated = (PromptStrategy.ZERO_SHOT, PromptStrategy.ZERO_SHOT)
        with pytest.raises(ConfigurationError, match="repeat"):
            AppConfig(prompt_strategies=duplicated)

    def test_rejects_empty_strategy_list(self) -> None:
        with pytest.raises(ConfigurationError, match="prompt_strategies"):
            AppConfig(prompt_strategies=())

    def test_rejects_duplicate_labels(self) -> None:
        with pytest.raises(ConfigurationError, match="labels"):
            AppConfig(labels=(LABEL_POSITIVE, LABEL_POSITIVE))

    def test_rejects_unknown_log_level(self) -> None:
        with pytest.raises(ConfigurationError, match="log_level"):
            AppConfig(log_level="LOUD")


# ---------------------------------------------------------------------------
# Environment loading
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_falls_back_to_defaults_when_environment_is_empty(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        cfg = AppConfig.from_env(load_env_file=False)
        assert cfg == AppConfig()

    def test_reads_overrides_from_environment(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        clean_env.setenv("PROMPTBENCH_MODEL", "gemini-2.5-pro")
        clean_env.setenv("PROMPTBENCH_TEMPERATURE", "0.7")
        clean_env.setenv("PROMPTBENCH_RANDOM_SEED", "7")
        clean_env.setenv("PROMPTBENCH_BENCHMARK_SAMPLE_SIZE", "50")
        clean_env.setenv("PROMPTBENCH_DEV_SAMPLE_SIZE", "3")
        clean_env.setenv("PROMPTBENCH_LOG_LEVEL", "debug")

        cfg = AppConfig.from_env(load_env_file=False)

        assert cfg.model == "gemini-2.5-pro"
        assert cfg.temperature == 0.7
        assert cfg.random_seed == 7
        assert cfg.benchmark_sample_size == 50
        assert cfg.dev_sample_size == 3
        assert cfg.log_level == "DEBUG"  # normalised to upper case

    def test_blank_values_fall_back_to_defaults(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        clean_env.setenv("PROMPTBENCH_MODEL", "   ")
        assert AppConfig.from_env(load_env_file=False).model == DEFAULT_MODEL

    def test_values_are_stripped(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("PROMPTBENCH_MODEL", "  gemini-2.5-pro  ")
        assert AppConfig.from_env(load_env_file=False).model == "gemini-2.5-pro"

    def test_rejects_non_numeric_integer_variable(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        clean_env.setenv("PROMPTBENCH_RANDOM_SEED", "abc")
        with pytest.raises(ConfigurationError, match="PROMPTBENCH_RANDOM_SEED"):
            AppConfig.from_env(load_env_file=False)

    def test_rejects_non_numeric_float_variable(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        clean_env.setenv("PROMPTBENCH_TEMPERATURE", "hot")
        with pytest.raises(ConfigurationError, match="PROMPTBENCH_TEMPERATURE"):
            AppConfig.from_env(load_env_file=False)

    def test_environment_values_are_validated(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        clean_env.setenv("PROMPTBENCH_TEMPERATURE", "5.0")
        with pytest.raises(ConfigurationError, match="temperature"):
            AppConfig.from_env(load_env_file=False)

    def test_relative_results_dir_resolves_against_repo_root(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        clean_env.setenv("PROMPTBENCH_RESULTS_DIR", "tmp-results")
        cfg = AppConfig.from_env(load_env_file=False)
        assert cfg.results_dir == PROJECT_ROOT / "tmp-results"

    def test_absolute_results_dir_is_used_as_is(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        clean_env.setenv("PROMPTBENCH_RESULTS_DIR", "/tmp/pb-abs")
        cfg = AppConfig.from_env(load_env_file=False)
        assert cfg.results_dir == Path("/tmp/pb-abs")


# ---------------------------------------------------------------------------
# API key handling
# ---------------------------------------------------------------------------


class TestApiKey:
    def test_returns_key_from_environment(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv(ENV_API_KEY, "test-key-123")
        assert get_api_key(load_env_file=False) == "test-key-123"

    def test_strips_surrounding_whitespace(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        clean_env.setenv(ENV_API_KEY, "  test-key-123\n")
        assert get_api_key(load_env_file=False) == "test-key-123"

    def test_raises_when_unset(self, clean_env: pytest.MonkeyPatch) -> None:
        with pytest.raises(MissingAPIKeyError, match=ENV_API_KEY):
            get_api_key(load_env_file=False)

    def test_raises_when_blank(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv(ENV_API_KEY, "   ")
        with pytest.raises(MissingAPIKeyError):
            get_api_key(load_env_file=False)

    def test_raises_on_unedited_placeholder(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        clean_env.setenv(ENV_API_KEY, API_KEY_PLACEHOLDER)
        with pytest.raises(MissingAPIKeyError, match="placeholder"):
            get_api_key(load_env_file=False)

    def test_error_message_never_leaks_the_value(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        secret = "super-secret-key-value"
        clean_env.setenv(ENV_API_KEY, API_KEY_PLACEHOLDER)
        try:
            get_api_key(load_env_file=False)
        except MissingAPIKeyError as exc:
            assert secret not in str(exc)
            assert API_KEY_PLACEHOLDER in str(exc) or ENV_API_KEY in str(exc)

    def test_has_api_key_reports_without_raising(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        assert has_api_key(load_env_file=False) is False
        clean_env.setenv(ENV_API_KEY, "test-key-123")
        assert has_api_key(load_env_file=False) is True

    def test_key_is_not_a_config_field(self, clean_env: pytest.MonkeyPatch) -> None:
        secret = "super-secret-key-value"
        clean_env.setenv(ENV_API_KEY, secret)
        cfg = AppConfig.from_env(load_env_file=False)

        field_names = {f.name for f in dataclasses.fields(cfg)}
        assert not any("key" in name or "secret" in name for name in field_names)
        assert secret not in repr(cfg)
        assert secret not in str(cfg.to_dict())


# ---------------------------------------------------------------------------
# Serialisation and caching
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_to_dict_is_json_serialisable(self) -> None:
        import json

        payload = json.dumps(AppConfig().to_dict())
        assert "gemini" in payload

    def test_to_dict_contains_every_field(self) -> None:
        cfg = AppConfig()
        assert set(cfg.to_dict()) == {f.name for f in dataclasses.fields(cfg)}

    def test_to_dict_renders_strategies_as_plain_strings(self) -> None:
        strategies = AppConfig().to_dict()["prompt_strategies"]
        assert strategies == [
            "zero_shot",
            "few_shot",
            "role_based",
            "structured",
            "reasoning",
            "combined",
        ]
        assert all(type(s) is str for s in strategies)


class TestGetConfig:
    def test_returns_a_cached_instance(self, clean_env: pytest.MonkeyPatch) -> None:
        get_config.cache_clear()
        first = get_config()
        assert get_config() is first
        get_config.cache_clear()

    def test_cache_clear_picks_up_new_environment(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        get_config.cache_clear()
        clean_env.setenv("PROMPTBENCH_MODEL", "gemini-2.5-pro")
        assert get_config().model == "gemini-2.5-pro"
        get_config.cache_clear()
