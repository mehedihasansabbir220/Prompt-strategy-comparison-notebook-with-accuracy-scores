"""Benchmark runner: strategy x sample -> prediction.

The runner is the component that makes this a controlled experiment. It
receives *one* fixed evaluation set and *one* fixed set of demonstrations, then
sends every strategy through exactly the same samples in the same order with
the same provider settings. Nothing about the data changes between strategies —
only the prompt.

It records the outcome of every call, including the ones that fail. An API
failure or an unparseable response is data about the strategy, not a reason to
drop a sample, so the row count is always ``strategies x samples`` regardless of
what went wrong.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import pandas as pd
from tqdm.auto import tqdm

from src.config import LABEL_UNKNOWN, AppConfig, PromptStrategy
from src.dataset import (
    LABEL_COLUMN,
    REVIEW_COLUMN,
    SAMPLE_ID_COLUMN,
    FewShotExample,
    validate_prepared_dataset,
)
from src.evaluation.metrics import (
    best_strategy as select_best_strategy,
)
from src.evaluation.metrics import (
    evaluate_strategies,
    rank_strategies,
)
from src.llm.base import LLMAuthError, LLMProvider, LLMResponse
from src.prompts import PromptContext, PromptStrategyBase, build_strategies
from src.prompts.base import PromptError
from src.utils.parsing import parse_sentiment_response

logger = logging.getLogger(__name__)

#: Column order of ``predictions.csv``. The ten required fields come first, then
#: the operational columns used for cost and reliability analysis.
PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "experiment_id",
    "sample_id",
    "strategy",
    "review",
    "actual_label",
    "predicted_label",
    "raw_response",
    "latency_seconds",
    "success",
    "error",
    "prompt_chars",
    "prompt_tokens",
    "output_tokens",
    "total_tokens",
    "finish_reason",
    "attempts",
)


class RunnerError(RuntimeError):
    """Raised when a benchmark cannot be run as configured."""


@dataclass(frozen=True, slots=True)
class StrategyRunStats:
    """Per-strategy operational summary. Not a quality metric — those come later."""

    strategy: str
    samples: int
    successful_calls: int
    api_failures: int
    unknown_predictions: int
    runtime_seconds: float
    total_tokens: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "samples": self.samples,
            "successful_calls": self.successful_calls,
            "api_failures": self.api_failures,
            "unknown_predictions": self.unknown_predictions,
            "runtime_seconds": round(self.runtime_seconds, 3),
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Everything one benchmark run produced."""

    experiment_id: str
    predictions: pd.DataFrame
    config: dict[str, Any]
    stats: list[StrategyRunStats] = field(default_factory=list)
    runtime_seconds: float = 0.0
    experiment_dir: Path | None = None

    @property
    def stats_frame(self) -> pd.DataFrame:
        """Per-strategy operational summary as a DataFrame."""
        return pd.DataFrame([stat.to_dict() for stat in self.stats])


#: The metric the headline result is chosen on. F1 balances precision and
#: recall, so a strategy cannot top the table by abstaining on hard samples the
#: way it could on precision alone.
BEST_STRATEGY_METRIC: Final[str] = "f1_macro"


@dataclass(frozen=True, slots=True)
class ExperimentEvaluation:
    """Scored results for one experiment."""

    experiment_id: str
    metrics: pd.DataFrame
    summary: dict[str, Any]
    best_strategy: str | None


def build_experiment_summary(
    experiment_id: str,
    metrics: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    metric: str = BEST_STRATEGY_METRIC,
) -> dict[str, Any]:
    """Assemble ``summary.json`` from a scored metrics table.

    Every figure is read from ``metrics``, which is itself derived from stored
    predictions. Nothing is estimated, and ``best_strategy`` is ``None`` when
    the top score is tied — declaring a winner between indistinguishable
    strategies would overstate the result.
    """
    winner = select_best_strategy(metrics, metric=metric)
    ranked = rank_strategies(metrics, metric=metric)
    run_config = config or {}

    leaderboard = [
        {
            "rank": int(row["rank"]),
            "strategy": row["strategy"],
            "f1_macro": _round(row.get("f1_macro")),
            "accuracy": _round(row.get("accuracy")),
            "unknown_rate": _round(row.get("unknown_rate")),
        }
        for _, row in ranked.iterrows()
    ]
    winning_row = (
        metrics.loc[metrics["strategy"] == winner].iloc[0] if winner else None
    )

    return {
        "experiment_id": experiment_id,
        "timestamp": run_config.get("timestamp"),
        "dev_mode": run_config.get("dev_mode"),
        "model": run_config.get("provider", {}).get("model"),
        "temperature": run_config.get("provider", {}).get("temperature"),
        "dataset": {
            "name": run_config.get("dataset", {}).get("name"),
            "sample_count": run_config.get("dataset", {}).get("sample_count"),
            "random_seed": run_config.get("dataset", {}).get("random_seed"),
            "sample_id_checksum": run_config.get("dataset", {}).get(
                "sample_id_checksum"
            ),
        },
        "selection_metric": metric,
        "best_strategy": (
            None
            if winning_row is None
            else {
                "strategy": winner,
                "f1_macro": _round(winning_row.get("f1_macro")),
                "accuracy": _round(winning_row.get("accuracy")),
                "precision_macro": _round(winning_row.get("precision_macro")),
                "recall_macro": _round(winning_row.get("recall_macro")),
                "unknown_rate": _round(winning_row.get("unknown_rate")),
            }
        ),
        "best_strategy_note": (
            None
            if winner
            else f"No single best strategy: the top {metric} score is tied."
        ),
        "strategies_evaluated": int(len(metrics)),
        "totals": {
            "predictions": int(metrics["total_samples"].sum()),
            "correct": int(metrics["correct"].sum()),
            "incorrect": int(metrics["incorrect"].sum()),
            "unknown": int(metrics["unknown"].sum()),
            "api_failures": int(metrics["api_failures"].fillna(0).sum()),
            "runtime_seconds": _round(metrics["runtime_seconds"].sum()),
        },
        "ranking": leaderboard,
    }


def _round(value: Any, digits: int = 4) -> Any:
    """Round a numeric value for display, leaving ``None`` and text untouched."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


class BenchmarkRunner:
    """Executes prompt strategies over a fixed evaluation set."""

    def __init__(
        self,
        provider: LLMProvider,
        config: AppConfig | None = None,
        storage: Any | None = None,
    ) -> None:
        """Args:
        provider: Any object satisfying :class:`~src.llm.base.LLMProvider`.
            The runner never imports a vendor SDK, so swapping providers
            changes nothing here.
        config: Experiment configuration. Defaults to the environment.
        storage: :class:`~src.experiments.storage.ExperimentStorage` to persist
            into. ``None`` runs entirely in memory — used by tests and by
            exploratory calls that should not create a results directory.
        """
        self.provider = provider
        self.config = config or AppConfig.from_env()
        self.storage = storage

    # -- public API ---------------------------------------------------------

    def run(
        self,
        evaluation_set: pd.DataFrame,
        strategies: Sequence[PromptStrategy | str | PromptStrategyBase] | None = None,
        few_shot_examples: Sequence[FewShotExample] = (),
        *,
        dev_mode: bool = False,
        experiment_id: str | None = None,
        persist: bool = True,
        show_progress: bool = True,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> ExperimentResult:
        """Run every strategy over the same evaluation samples.

        Args:
            evaluation_set: The fixed subset from
                :func:`~src.dataset.create_fixed_evaluation_set`. Used verbatim
                by every strategy.
            strategies: Strategy names or instances. Defaults to the configured
                set, in configured order.
            few_shot_examples: Fixed demonstrations, shared by the strategies
                that need them.
            dev_mode: Use only the first ``config.dev_sample_size`` samples.
                Takes the *head* of the already-fixed set rather than resampling,
                so a dev run is a strict subset of the full run.
            experiment_id: Override the automatically allocated id.
            persist: Write ``config.json`` and ``predictions.csv``. Requires
                ``storage``.
            show_progress: Display a per-strategy terminal progress bar.
            progress_callback: Called as ``(strategy, completed, total)`` after
                every sample. Lets a GUI render its own progress without this
                module knowing anything about the front end.

        Returns:
            An :class:`ExperimentResult`.

        Raises:
            RunnerError: If the evaluation set is unusable, no strategies were
                given, or persistence was requested without storage.
            LLMAuthError: Propagated from the provider — a credential failure
                affects every call, so the run stops instead of recording
                hundreds of identical errors.
        """
        samples = self._prepare_samples(evaluation_set, dev_mode=dev_mode)
        active = self._resolve_strategies(strategies)
        context = PromptContext(few_shot_examples=tuple(few_shot_examples))
        self._check_requirements(active, context)

        run_id = self._allocate_experiment_id(experiment_id, persist=persist)
        started_at = datetime.now(timezone.utc)
        started = time.perf_counter()

        logger.info(
            "Experiment %s: %d strategy(ies) x %d sample(s) = %d call(s)%s",
            run_id,
            len(active),
            len(samples),
            len(active) * len(samples),
            " [dev mode]" if dev_mode else "",
        )

        records: list[dict[str, Any]] = []
        stats: list[StrategyRunStats] = []
        for strategy in active:
            strategy_records, strategy_stats = self._run_strategy(
                strategy, samples, context, run_id,
                show_progress=show_progress,
                progress_callback=progress_callback,
            )
            records.extend(strategy_records)
            stats.append(strategy_stats)

        runtime = time.perf_counter() - started
        predictions = pd.DataFrame(records, columns=list(PREDICTION_COLUMNS))
        config_payload = self._build_config(
            run_id,
            started_at=started_at,
            samples=samples,
            strategies=active,
            few_shot_examples=few_shot_examples,
            dev_mode=dev_mode,
            runtime=runtime,
            stats=stats,
        )

        experiment_dir: Path | None = None
        if persist and self.storage is not None:
            self.storage.save_config(run_id, config_payload)
            self.storage.save_predictions(run_id, predictions)
            experiment_dir = self.storage.experiment_path(run_id)

        logger.info(
            "Experiment %s finished in %.1fs: %d prediction(s), %d API failure(s), "
            "%d unparseable",
            run_id,
            runtime,
            len(predictions),
            sum(stat.api_failures for stat in stats),
            sum(stat.unknown_predictions for stat in stats),
        )
        return ExperimentResult(
            experiment_id=run_id,
            predictions=predictions,
            config=config_payload,
            stats=stats,
            runtime_seconds=runtime,
            experiment_dir=experiment_dir,
        )

    def evaluate(
        self, result: ExperimentResult, *, persist: bool = True
    ) -> ExperimentEvaluation:
        """Score a completed run and persist ``metrics.csv`` and ``summary.json``.

        Kept separate from :meth:`run` so scoring is an explicit, repeatable
        step: a stored ``predictions.csv`` can be re-scored at any time without
        re-issuing a single API call.

        Args:
            result: The run to score.
            persist: Write the two artefacts. Requires ``storage`` and an
                already-created experiment directory.

        Returns:
            An :class:`ExperimentEvaluation`.

        Raises:
            RunnerError: If persistence was requested without storage.
        """
        metrics = evaluate_strategies(
            result.predictions,
            runtime_by_strategy={
                stat.strategy: stat.runtime_seconds for stat in result.stats
            },
        )
        summary = build_experiment_summary(
            result.experiment_id, metrics, config=result.config
        )
        winner = (
            summary["best_strategy"]["strategy"] if summary["best_strategy"] else None
        )

        if persist:
            if self.storage is None:
                raise RunnerError("persist=True requires a storage backend")
            self.storage.save_metrics(result.experiment_id, metrics)
            self.storage.save_summary(result.experiment_id, summary)

        logger.info(
            "Experiment %s scored: best strategy by %s is %s",
            result.experiment_id,
            BEST_STRATEGY_METRIC,
            winner or "undecided (tie)",
        )
        return ExperimentEvaluation(
            experiment_id=result.experiment_id,
            metrics=metrics,
            summary=summary,
            best_strategy=winner,
        )

    # -- setup --------------------------------------------------------------

    def _prepare_samples(
        self, evaluation_set: pd.DataFrame, *, dev_mode: bool
    ) -> pd.DataFrame:
        """Validate the evaluation set and apply development mode."""
        try:
            validate_prepared_dataset(evaluation_set)
        except Exception as exc:
            raise RunnerError(f"Invalid evaluation set: {exc}") from exc

        if not dev_mode:
            return evaluation_set.reset_index(drop=True)

        limit = min(self.config.dev_sample_size, len(evaluation_set))
        logger.info("Development mode: using the first %d sample(s)", limit)
        return evaluation_set.head(limit).reset_index(drop=True)

    @staticmethod
    def _resolve_strategies(
        strategies: Sequence[PromptStrategy | str | PromptStrategyBase] | None,
    ) -> list[PromptStrategyBase]:
        """Accept names or instances and return instances, order preserved."""
        if strategies is None:
            return build_strategies()
        if len(strategies) == 0:
            raise RunnerError("At least one strategy is required")
        resolved: list[PromptStrategyBase] = []
        for item in strategies:
            if isinstance(item, PromptStrategyBase):
                resolved.append(item)
            else:
                resolved.extend(build_strategies([item]))
        return resolved

    @staticmethod
    def _check_requirements(
        strategies: Sequence[PromptStrategyBase], context: PromptContext
    ) -> None:
        """Fail before any tokens are spent if a strategy lacks its inputs."""
        missing = [
            str(strategy.name)
            for strategy in strategies
            if strategy.requires_examples and not context.few_shot_examples
        ]
        if missing:
            raise RunnerError(
                f"Strategy/strategies {missing} require few-shot examples, "
                "but none were supplied"
            )

    def _allocate_experiment_id(
        self, experiment_id: str | None, *, persist: bool
    ) -> str:
        """Reserve an id, creating its directory when the run will be stored."""
        if self.storage is None:
            if persist and experiment_id is None:
                raise RunnerError("persist=True requires a storage backend")
            return experiment_id or datetime.now().strftime("%Y-%m-%d_000")
        if not persist:
            return experiment_id or self.storage.next_experiment_id()
        return self.storage.create_experiment(experiment_id)

    # -- execution ----------------------------------------------------------

    def _run_strategy(
        self,
        strategy: PromptStrategyBase,
        samples: pd.DataFrame,
        context: PromptContext,
        experiment_id: str,
        *,
        show_progress: bool,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> tuple[list[dict[str, Any]], StrategyRunStats]:
        """Run one strategy across every sample, collecting records and stats."""
        records: list[dict[str, Any]] = []
        started = time.perf_counter()
        api_failures = 0
        unknown = 0
        token_total = 0
        tokens_reported = False

        rows = samples.itertuples(index=False)
        iterator = tqdm(
            rows,
            total=len(samples),
            desc=f"{str(strategy.name):<12}",
            disable=not show_progress,
            leave=False,
            unit="sample",
        )

        for row in iterator:
            record = self._run_sample(strategy, row, context, experiment_id)
            records.append(record)

            if not record["success"]:
                api_failures += 1
            if record["predicted_label"] == LABEL_UNKNOWN:
                unknown += 1
            if record["total_tokens"] is not None:
                token_total += int(record["total_tokens"])
                tokens_reported = True
            if progress_callback is not None:
                progress_callback(str(strategy.name), len(records), len(samples))

        runtime = time.perf_counter() - started
        stats = StrategyRunStats(
            strategy=str(strategy.name),
            samples=len(records),
            successful_calls=len(records) - api_failures,
            api_failures=api_failures,
            unknown_predictions=unknown,
            runtime_seconds=runtime,
            total_tokens=token_total if tokens_reported else None,
        )
        logger.info(
            "Strategy %s: %d sample(s) in %.1fs | %d API failure(s), %d unparseable",
            stats.strategy,
            stats.samples,
            runtime,
            api_failures,
            unknown,
        )
        return records, stats

    def _run_sample(
        self,
        strategy: PromptStrategyBase,
        row: Any,
        context: PromptContext,
        experiment_id: str,
    ) -> dict[str, Any]:
        """Build the prompt, call the provider, parse the answer, record it all."""
        sample_id = getattr(row, SAMPLE_ID_COLUMN)
        review = getattr(row, REVIEW_COLUMN)
        actual = getattr(row, LABEL_COLUMN)

        try:
            prompt = strategy.build_prompt(review, context)
        except PromptError as exc:
            # A prompt that cannot be built is a code-level failure for this
            # sample; record it rather than aborting the whole benchmark.
            logger.error("Prompt build failed for %s/%s: %s", strategy.name, sample_id, exc)
            return self._record(
                experiment_id, sample_id, strategy, review, actual,
                response=None, prompt_chars=0, error=f"PromptError: {exc}",
            )

        response = self.provider.generate(prompt)
        return self._record(
            experiment_id, sample_id, strategy, review, actual,
            response=response, prompt_chars=prompt.char_count,
        )

    @staticmethod
    def _record(
        experiment_id: str,
        sample_id: str,
        strategy: PromptStrategyBase,
        review: str,
        actual: str,
        *,
        response: LLMResponse | None,
        prompt_chars: int,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Assemble one prediction row.

        A failed call still produces a row, with ``predicted_label`` set to
        ``unknown``: the sample was asked and did not yield an answer, and
        dropping it would quietly shrink the denominator of every metric.
        """
        raw_text = response.text if response else ""
        predicted = (
            parse_sentiment_response(raw_text)
            if response and response.success
            else LABEL_UNKNOWN
        )
        if response is not None and not response.success:
            error = error or f"{response.error_type}: {response.error_message}"

        return {
            "experiment_id": experiment_id,
            "sample_id": sample_id,
            "strategy": str(strategy.name),
            "review": review,
            "actual_label": actual,
            "predicted_label": predicted,
            "raw_response": raw_text,
            "latency_seconds": round(response.latency_seconds, 4) if response else 0.0,
            "success": bool(response.success) if response else False,
            "error": error,
            "prompt_chars": prompt_chars,
            "prompt_tokens": response.usage.prompt_tokens if response else None,
            "output_tokens": response.usage.output_tokens if response else None,
            "total_tokens": response.usage.total_tokens if response else None,
            "finish_reason": response.finish_reason if response else None,
            "attempts": response.attempts if response else 0,
        }

    # -- metadata -----------------------------------------------------------

    def _build_config(
        self,
        experiment_id: str,
        *,
        started_at: datetime,
        samples: pd.DataFrame,
        strategies: Sequence[PromptStrategyBase],
        few_shot_examples: Sequence[FewShotExample],
        dev_mode: bool,
        runtime: float,
        stats: Sequence[StrategyRunStats],
    ) -> dict[str, Any]:
        """Assemble the metadata that makes the run reproducible."""
        sample_ids = samples[SAMPLE_ID_COLUMN].tolist()
        return {
            "experiment_id": experiment_id,
            "timestamp": started_at.isoformat(),
            "dev_mode": dev_mode,
            "runtime_seconds": round(runtime, 3),
            "provider": self.provider.describe(),
            "configuration": self.config.to_dict(),
            "dataset": {
                "name": "imdb",
                "sample_count": len(samples),
                "random_seed": self.config.random_seed,
                "class_distribution": samples[LABEL_COLUMN].value_counts().to_dict(),
                # Proves after the fact that every strategy saw the same rows,
                # and lets a re-run be checked against this one.
                "sample_id_checksum": checksum_sample_ids(sample_ids),
                "sample_ids": sample_ids,
            },
            "few_shot_examples": {
                "count": len(few_shot_examples),
                "labels": [example.label for example in few_shot_examples],
            },
            "prompt_strategies": [strategy.describe() for strategy in strategies],
            "run_stats": [stat.to_dict() for stat in stats],
        }


def checksum_sample_ids(sample_ids: Sequence[str]) -> str:
    """Return a short, stable digest of an evaluation set's sample ids.

    Two runs sharing a checksum were scored on exactly the same reviews, which
    is what makes their numbers comparable. Order-sensitive by design.
    """
    digest = hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest()
    return digest[:16]
