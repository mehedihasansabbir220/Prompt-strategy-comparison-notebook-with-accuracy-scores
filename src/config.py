"""Typed, immutable configuration for PromptBench.

This module is the single source of truth for every experiment constant. It is
deliberately the *only* place that reads environment variables, so that:

* the Gemini API key is never hardcoded and never stored on a config object,
* an experiment's exact settings can be serialised into ``config.json`` later,
* tests can construct a configuration without touching the developer's ``.env``.

Nothing here performs I/O against the dataset or the LLM.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Literal

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ConfigurationError(RuntimeError):
    """Raised when configuration is missing, malformed, or out of range."""


class MissingAPIKeyError(ConfigurationError):
    """Raised when ``GEMINI_API_KEY`` is absent or still holds the placeholder."""


# ---------------------------------------------------------------------------
# Identity and filesystem anchors
# ---------------------------------------------------------------------------

#: Human-readable application name, used in logs, charts and the notebook header.
APP_NAME: Final[str] = "PromptBench"

#: Repository root, resolved from this file so paths work from any CWD
#: (notebook kernels start in ``notebooks/``, pytest starts in the root).
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

#: Default root for all persisted experiment output.
DEFAULT_RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Sentiment label vocabulary
# ---------------------------------------------------------------------------

LABEL_NEGATIVE: Final[str] = "negative"
LABEL_POSITIVE: Final[str] = "positive"

#: Sentinel for a model response that could not be parsed into a valid label.
#: It is NOT a sentiment class: it is excluded from precision/recall and
#: reported as its own reliability metric.
LABEL_UNKNOWN: Final[str] = "unknown"

#: Canonical class order. Fixed as (negative, positive) to mirror IMDb's own
#: integer encoding (0 = negative, 1 = positive), so confusion matrices are
#: laid out identically for every strategy and are directly comparable.
SENTIMENT_LABELS: Final[tuple[str, ...]] = (LABEL_NEGATIVE, LABEL_POSITIVE)

#: Static type for a valid ground-truth or predicted sentiment class.
SentimentLabel = Literal["negative", "positive"]


# ---------------------------------------------------------------------------
# Prompt strategies
# ---------------------------------------------------------------------------


class PromptStrategy(StrEnum):
    """The prompt strategies compared by the benchmark.

    ``StrEnum`` members compare equal to their string value, so a strategy can
    be used directly as a DataFrame column value or JSON field while still
    being typo-proof at call sites.
    """

    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    ROLE_BASED = "role_based"
    STRUCTURED = "structured"
    REASONING = "reasoning"
    COMBINED = "combined"


#: Default execution order for a benchmark run: baseline first, then single
#: techniques, then the composite — so the combined strategy is always read
#: against strategies that have already been measured.
DEFAULT_STRATEGIES: Final[tuple[PromptStrategy, ...]] = (
    PromptStrategy.ZERO_SHOT,
    PromptStrategy.FEW_SHOT,
    PromptStrategy.ROLE_BASED,
    PromptStrategy.STRUCTURED,
    PromptStrategy.REASONING,
    PromptStrategy.COMBINED,
)


# ---------------------------------------------------------------------------
# Environment defaults
# ---------------------------------------------------------------------------

ENV_API_KEY: Final[str] = "GEMINI_API_KEY"

#: Placeholder shipped in ``.env.example``; treated as "not configured" so a
#: user who copied the template but never edited it gets a clear error.
API_KEY_PLACEHOLDER: Final[str] = "your_api_key_here"

DEFAULT_MODEL: Final[str] = "gemini-2.5-flash"
DEFAULT_TEMPERATURE: Final[float] = 0.0
DEFAULT_RANDOM_SEED: Final[int] = 42
DEFAULT_DEV_SAMPLE_SIZE: Final[int] = 10
DEFAULT_BENCHMARK_SAMPLE_SIZE: Final[int] = 100
DEFAULT_LOG_LEVEL: Final[str] = "INFO"

#: Upper bound accepted for ``temperature``. Gemini's documented range is
#: [0.0, 2.0]; values outside it are a configuration mistake, not an experiment.
MAX_TEMPERATURE: Final[float] = 2.0

VALID_LOG_LEVELS: Final[tuple[str, ...]] = (
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
)


# ---------------------------------------------------------------------------
# Environment loading helpers
# ---------------------------------------------------------------------------


def load_environment(dotenv_path: Path | None = None, *, override: bool = False) -> bool:
    """Load ``.env`` into the process environment.

    Args:
        dotenv_path: Explicit ``.env`` location. Defaults to the repository root.
        override: If ``False`` (default) real environment variables win over the
            file, so CI secrets are never shadowed by a stale local ``.env``.

    Returns:
        ``True`` if a ``.env`` file was found and read, ``False`` otherwise.
        A missing file is not an error: the environment may be populated
        directly (CI, shell export, Docker).
    """
    path = dotenv_path or (PROJECT_ROOT / ".env")
    return load_dotenv(dotenv_path=path, override=override)


def _env_str(name: str, default: str) -> str:
    """Read a string variable, falling back to ``default`` when unset or blank."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_int(name: str, default: int) -> int:
    """Read an integer variable, raising ``ConfigurationError`` on bad input."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be an integer, got {raw.strip()!r}"
        ) from exc


def _env_float(name: str, default: float) -> float:
    """Read a float variable, raising ``ConfigurationError`` on bad input."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ConfigurationError(
            f"{name} must be a number, got {raw.strip()!r}"
        ) from exc


def _env_path(name: str, default: Path) -> Path:
    """Read a filesystem path variable, resolving relatives against the repo root."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    candidate = Path(raw.strip()).expanduser()
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


def get_api_key(*, load_env_file: bool = True) -> str:
    """Return the Gemini API key from the environment.

    The key is intentionally NOT a field on :class:`AppConfig`: configuration
    objects get logged, displayed in the notebook and serialised to
    ``config.json``, and a secret must never travel with them.

    Args:
        load_env_file: Load ``.env`` first. Disabled in tests for isolation.

    Returns:
        The API key, stripped of surrounding whitespace.

    Raises:
        MissingAPIKeyError: If the variable is unset, blank, or still the
            ``.env.example`` placeholder. The message never contains the value.
    """
    if load_env_file:
        load_environment()

    raw = os.getenv(ENV_API_KEY, "").strip()
    if not raw:
        raise MissingAPIKeyError(
            f"{ENV_API_KEY} is not set. Copy .env.example to .env and add your key "
            "(https://aistudio.google.com/apikey)."
        )
    if raw == API_KEY_PLACEHOLDER:
        raise MissingAPIKeyError(
            f"{ENV_API_KEY} still holds the .env.example placeholder. "
            "Replace it with a real key."
        )
    return raw


def has_api_key(*, load_env_file: bool = True) -> bool:
    """Return whether a usable API key is configured, without raising.

    Lets the notebook show a clear "API key not configured — benchmark cannot
    run" banner instead of crashing a cell.
    """
    try:
        get_api_key(load_env_file=load_env_file)
    except MissingAPIKeyError:
        return False
    return True


# ---------------------------------------------------------------------------
# Configuration object
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Immutable experiment configuration.

    Frozen so a running benchmark cannot mutate the settings it is being
    measured under — the stored ``config.json`` always describes what actually
    executed. Use :meth:`with_overrides` to derive a variant.
    """

    #: Application name, surfaced in logs, chart titles and the notebook header.
    app_name: str = APP_NAME

    #: Gemini model identifier. Held constant across every strategy in a run;
    #: recorded in metadata because hosted models change over time.
    model: str = DEFAULT_MODEL

    #: Decoding temperature. Defaults to 0.0 for the most repeatable output the
    #: provider offers. Constant across strategies — it is a controlled variable.
    temperature: float = DEFAULT_TEMPERATURE

    #: Seed for evaluation-subset sampling. The same seed and sample size always
    #: select the same IMDb reviews, so strategies are compared on identical data.
    random_seed: int = DEFAULT_RANDOM_SEED

    #: Small subset for smoke-testing the pipeline end to end before spending quota.
    dev_sample_size: int = DEFAULT_DEV_SAMPLE_SIZE

    #: Sample count for a real benchmark run. Reported alongside every metric,
    #: since it bounds how much a difference between strategies can be trusted.
    benchmark_sample_size: int = DEFAULT_BENCHMARK_SAMPLE_SIZE

    #: Valid sentiment classes in canonical (negative, positive) order.
    labels: tuple[str, ...] = SENTIMENT_LABELS

    #: Root directory for persisted runs. ``experiments/`` holds immutable runs.
    results_dir: Path = DEFAULT_RESULTS_DIR

    #: Strategies to execute, in order.
    prompt_strategies: tuple[PromptStrategy, ...] = field(default=DEFAULT_STRATEGIES)

    #: Logging verbosity for the run.
    log_level: str = DEFAULT_LOG_LEVEL

    # -- derived paths ------------------------------------------------------

    @property
    def experiments_dir(self) -> Path:
        """Directory holding one immutable subdirectory per experiment run."""
        return self.results_dir / "experiments"

    @property
    def latest_dir(self) -> Path:
        """Convenience location pointing at the most recent run's output."""
        return self.results_dir / "latest"

    # -- validation ---------------------------------------------------------

    def __post_init__(self) -> None:
        """Reject invalid configuration at construction time, not mid-benchmark."""
        if not self.app_name.strip():
            raise ConfigurationError("app_name must not be empty")
        if not self.model.strip():
            raise ConfigurationError("model must not be empty")
        if not 0.0 <= self.temperature <= MAX_TEMPERATURE:
            raise ConfigurationError(
                f"temperature must be within [0.0, {MAX_TEMPERATURE}], "
                f"got {self.temperature}"
            )
        if self.dev_sample_size < 1:
            raise ConfigurationError(
                f"dev_sample_size must be >= 1, got {self.dev_sample_size}"
            )
        if self.benchmark_sample_size < 1:
            raise ConfigurationError(
                f"benchmark_sample_size must be >= 1, got {self.benchmark_sample_size}"
            )
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels):
            raise ConfigurationError(
                f"labels must contain at least two unique values, got {self.labels}"
            )
        if not self.prompt_strategies:
            raise ConfigurationError("prompt_strategies must not be empty")
        if len(set(self.prompt_strategies)) != len(self.prompt_strategies):
            raise ConfigurationError(
                f"prompt_strategies must not repeat, got {self.prompt_strategies}"
            )
        if self.log_level.upper() not in VALID_LOG_LEVELS:
            raise ConfigurationError(
                f"log_level must be one of {VALID_LOG_LEVELS}, got {self.log_level!r}"
            )

    # -- constructors and views --------------------------------------------

    @classmethod
    def from_env(cls, *, load_env_file: bool = True) -> AppConfig:
        """Build a configuration from ``PROMPTBENCH_*`` variables.

        Every field falls back to the module default, so the project runs with
        no ``.env`` at all (aside from the API key, which is required only when
        the LLM is actually called).

        Args:
            load_env_file: Load ``.env`` first. Tests pass ``False`` so they
                never pick up the developer's local settings.
        """
        if load_env_file:
            load_environment()

        return cls(
            model=_env_str("PROMPTBENCH_MODEL", DEFAULT_MODEL),
            temperature=_env_float("PROMPTBENCH_TEMPERATURE", DEFAULT_TEMPERATURE),
            random_seed=_env_int("PROMPTBENCH_RANDOM_SEED", DEFAULT_RANDOM_SEED),
            dev_sample_size=_env_int(
                "PROMPTBENCH_DEV_SAMPLE_SIZE", DEFAULT_DEV_SAMPLE_SIZE
            ),
            benchmark_sample_size=_env_int(
                "PROMPTBENCH_BENCHMARK_SAMPLE_SIZE", DEFAULT_BENCHMARK_SAMPLE_SIZE
            ),
            results_dir=_env_path("PROMPTBENCH_RESULTS_DIR", DEFAULT_RESULTS_DIR),
            log_level=_env_str("PROMPTBENCH_LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        )

    def with_overrides(self, **changes: Any) -> AppConfig:
        """Return a validated copy with selected fields replaced.

        Used for one-off variants (e.g. a dev-sized run) without mutating or
        re-reading the environment.
        """
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view for experiment metadata and display.

        Contains no secrets by construction — the API key is not a field.
        """
        return {
            "app_name": self.app_name,
            "model": self.model,
            "temperature": self.temperature,
            "random_seed": self.random_seed,
            "dev_sample_size": self.dev_sample_size,
            "benchmark_sample_size": self.benchmark_sample_size,
            "labels": list(self.labels),
            "results_dir": str(self.results_dir),
            "prompt_strategies": [str(s) for s in self.prompt_strategies],
            "log_level": self.log_level,
        }


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Return the shared, cached configuration built from the environment.

    Cached rather than a module-level global: the object is frozen, so callers
    share an immutable value. Call ``get_config.cache_clear()`` to rebuild after
    changing environment variables (tests do this).
    """
    return AppConfig.from_env()
