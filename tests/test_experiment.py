"""Tests for the benchmark runner and experiment storage.

The provider is always a scripted fake, so no test contacts Gemini. Storage
tests write into pytest's ``tmp_path``, never into the real ``results/``.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

import pandas as pd
import pytest

from src.config import LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_UNKNOWN, AppConfig
from src.dataset import FewShotExample
from src.evaluation.metrics import evaluate_strategies
from src.experiments.runner import (
    PREDICTION_COLUMNS,
    BenchmarkRunner,
    RunnerError,
    build_experiment_summary,
    checksum_sample_ids,
)
from src.experiments.storage import (
    CONFIG_FILENAME,
    METRICS_FILENAME,
    PREDICTIONS_FILENAME,
    SUMMARY_FILENAME,
    ExperimentStorage,
    StorageError,
)
from src.llm.base import LLMAuthError, LLMResponse, UsageMetadata

POS = LABEL_POSITIVE
NEG = LABEL_NEGATIVE
UNK = LABEL_UNKNOWN


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeProvider:
    """Scripted provider. Records every prompt it was given."""

    def __init__(self, *, replies: list[Any] | None = None, default: str = "positive") -> None:
        self._replies = list(replies or [])
        self._default = default
        self.prompts: list[Any] = []

    @property
    def model(self) -> str:
        return "fake-model"

    def describe(self) -> dict[str, Any]:
        return {"provider": "fake", "model": self.model, "temperature": 0.0}

    def generate(self, prompt: Any) -> LLMResponse:
        self.prompts.append(prompt)
        reply = self._replies.pop(0) if self._replies else self._default
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, LLMResponse):
            return reply
        return LLMResponse(
            text=reply,
            success=True,
            latency_seconds=0.01,
            model=self.model,
            usage=UsageMetadata(prompt_tokens=100, output_tokens=1, total_tokens=101),
            finish_reason="STOP",
        )


def failed_response(error_type: str = "LLMTransientError") -> LLMResponse:
    return LLMResponse(
        text="",
        success=False,
        latency_seconds=0.5,
        model="fake-model",
        error_type=error_type,
        error_message="backend unavailable",
        attempts=3,
    )


@pytest.fixture
def evaluation_set() -> pd.DataFrame:
    """Six samples, class-balanced, in the prepared schema."""
    return pd.DataFrame(
        {
            "sample_id": [f"test-{index:05d}" for index in range(6)],
            "review": [
                "A wonderful film.",
                "Utterly tedious.",
                "Charming and warm.",
                "I walked out early.",
                "Beautifully shot.",
                "A complete waste of time.",
            ],
            "label": [POS, NEG, POS, NEG, POS, NEG],
        }
    )


@pytest.fixture
def examples() -> tuple[FewShotExample, ...]:
    return (
        FewShotExample(review="Dull and predictable.", label=NEG),
        FewShotExample(review="Genuinely moving.", label=POS),
    )


@pytest.fixture
def storage(tmp_path: Any) -> ExperimentStorage:
    return ExperimentStorage(tmp_path / "experiments")


@pytest.fixture
def config() -> AppConfig:
    return AppConfig(model="fake-model", dev_sample_size=2)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class TestExperimentIds:
    def test_first_id_of_the_day(self, storage: ExperimentStorage) -> None:
        assert storage.next_experiment_id(today=date(2026, 8, 24)) == "2026-08-24_001"

    def test_ids_increment_within_a_day(self, storage: ExperimentStorage) -> None:
        day = date(2026, 8, 24)
        assert storage.create_experiment(today=day) == "2026-08-24_001"
        assert storage.create_experiment(today=day) == "2026-08-24_002"
        assert storage.next_experiment_id(today=day) == "2026-08-24_003"

    def test_ids_restart_on_a_new_day(self, storage: ExperimentStorage) -> None:
        storage.create_experiment(today=date(2026, 8, 24))
        assert storage.next_experiment_id(today=date(2026, 8, 25)) == "2026-08-25_001"

    def test_ids_are_never_reissued_after_a_deletion(
        self, storage: ExperimentStorage
    ) -> None:
        day = date(2026, 8, 24)
        first = storage.create_experiment(today=day)
        storage.create_experiment(today=day)
        storage.experiment_path(first).rmdir()
        # 001 is now free, but the counter still moves forward.
        assert storage.create_experiment(today=day) == "2026-08-24_003"

    def test_unrelated_directories_are_ignored(
        self, storage: ExperimentStorage
    ) -> None:
        storage.base_dir.mkdir(parents=True)
        (storage.base_dir / "scratch-notes").mkdir()
        assert storage.list_experiments() == []
        assert storage.next_experiment_id(today=date(2026, 8, 24)) == "2026-08-24_001"

    def test_malformed_id_is_rejected(self, storage: ExperimentStorage) -> None:
        with pytest.raises(StorageError, match="Malformed experiment id"):
            storage.experiment_path("../escape")


class TestStorageWrites:
    def test_creating_an_existing_experiment_is_refused(
        self, storage: ExperimentStorage
    ) -> None:
        created = storage.create_experiment(today=date(2026, 8, 24))
        with pytest.raises(StorageError, match="never overwritten"):
            storage.create_experiment(created)

    def test_config_round_trips(self, storage: ExperimentStorage) -> None:
        experiment_id = storage.create_experiment()
        storage.save_config(experiment_id, {"experiment_id": experiment_id, "seed": 42})
        assert storage.load_config(experiment_id)["seed"] == 42

    def test_config_is_human_readable_json(self, storage: ExperimentStorage) -> None:
        experiment_id = storage.create_experiment()
        path = storage.save_config(experiment_id, {"a": 1})
        assert path.name == CONFIG_FILENAME
        assert "\n" in path.read_text()  # indented, not a single line

    def test_non_serialisable_config_is_coerced_not_lost(
        self, storage: ExperimentStorage
    ) -> None:
        experiment_id = storage.create_experiment()
        storage.save_config(experiment_id, {"path": storage.base_dir})
        assert isinstance(storage.load_config(experiment_id)["path"], str)

    def test_predictions_round_trip(self, storage: ExperimentStorage) -> None:
        experiment_id = storage.create_experiment()
        frame = pd.DataFrame({"sample_id": ["s1"], "raw_response": ["positive"]})
        path = storage.save_predictions(experiment_id, frame)
        assert path.name == PREDICTIONS_FILENAME
        pd.testing.assert_frame_equal(storage.load_predictions(experiment_id), frame)

    def test_empty_raw_response_round_trips_as_empty_string(
        self, storage: ExperimentStorage
    ) -> None:
        experiment_id = storage.create_experiment()
        frame = pd.DataFrame({"sample_id": ["s1"], "raw_response": [""]})
        storage.save_predictions(experiment_id, frame)
        assert storage.load_predictions(experiment_id)["raw_response"].iloc[0] == ""

    def test_empty_predictions_are_refused(self, storage: ExperimentStorage) -> None:
        experiment_id = storage.create_experiment()
        with pytest.raises(StorageError, match="empty predictions"):
            storage.save_predictions(experiment_id, pd.DataFrame())

    def test_writing_to_a_missing_experiment_is_refused(
        self, storage: ExperimentStorage
    ) -> None:
        with pytest.raises(StorageError, match="does not exist"):
            storage.save_config("2026-08-24_001", {})

    def test_listing_is_sorted(self, storage: ExperimentStorage) -> None:
        day = date(2026, 8, 24)
        for _ in range(3):
            storage.create_experiment(today=day)
        assert storage.list_experiments() == [
            "2026-08-24_001",
            "2026-08-24_002",
            "2026-08-24_003",
        ]


# ---------------------------------------------------------------------------
# Runner — the controlled-experiment guarantees
# ---------------------------------------------------------------------------


class TestControlledExecution:
    def test_every_strategy_sees_the_same_samples(
        self, evaluation_set: pd.DataFrame, examples: Any, config: AppConfig
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), config)
        result = runner.run(
            evaluation_set,
            ["zero_shot", "few_shot", "structured"],
            examples,
            persist=False,
            experiment_id="2026-08-24_001",
            show_progress=False,
        )
        per_strategy = result.predictions.groupby("strategy")["sample_id"].apply(list)
        reference = per_strategy.iloc[0]
        assert all(ids == reference for ids in per_strategy)

    def test_row_count_is_strategies_times_samples(
        self, evaluation_set: pd.DataFrame, examples: Any, config: AppConfig
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), config)
        result = runner.run(
            evaluation_set, ["zero_shot", "role_based"], examples,
            persist=False, experiment_id="2026-08-24_001", show_progress=False,
        )
        assert len(result.predictions) == 2 * len(evaluation_set)

    def test_strategy_order_is_preserved(
        self, evaluation_set: pd.DataFrame, examples: Any, config: AppConfig
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), config)
        result = runner.run(
            evaluation_set, ["reasoning", "zero_shot", "combined"], examples,
            persist=False, experiment_id="2026-08-24_001", show_progress=False,
        )
        assert result.predictions["strategy"].drop_duplicates().tolist() == [
            "reasoning",
            "zero_shot",
            "combined",
        ]

    def test_each_strategy_sends_a_different_prompt(
        self, evaluation_set: pd.DataFrame, examples: Any, config: AppConfig
    ) -> None:
        provider = FakeProvider()
        runner = BenchmarkRunner(provider, config)
        runner.run(
            evaluation_set.head(1), ["zero_shot", "structured", "role_based"], examples,
            persist=False, experiment_id="2026-08-24_001", show_progress=False,
        )
        assert len({prompt.as_text() for prompt in provider.prompts}) == 3

    def test_ground_truth_is_carried_through_unchanged(
        self, evaluation_set: pd.DataFrame, config: AppConfig
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), config)
        result = runner.run(
            evaluation_set, ["zero_shot"], persist=False,
            experiment_id="2026-08-24_001", show_progress=False,
        )
        merged = result.predictions.merge(
            evaluation_set, on="sample_id", suffixes=("", "_source")
        )
        assert (merged["actual_label"] == merged["label"]).all()


class TestPredictionRecords:
    @pytest.fixture
    def result(self, evaluation_set: pd.DataFrame, config: AppConfig) -> Any:
        runner = BenchmarkRunner(FakeProvider(default="positive"), config)
        return runner.run(
            evaluation_set, ["zero_shot"], persist=False,
            experiment_id="2026-08-24_001", show_progress=False,
        )

    def test_contains_every_required_field(self, result: Any) -> None:
        required = (
            "experiment_id", "sample_id", "strategy", "review", "actual_label",
            "predicted_label", "raw_response", "latency_seconds", "success", "error",
        )
        assert list(result.predictions.columns)[: len(required)] == list(required)
        assert list(result.predictions.columns) == list(PREDICTION_COLUMNS)

    def test_experiment_id_is_stamped_on_every_row(self, result: Any) -> None:
        assert set(result.predictions["experiment_id"]) == {"2026-08-24_001"}

    def test_raw_response_is_preserved_verbatim(
        self, evaluation_set: pd.DataFrame, config: AppConfig
    ) -> None:
        provider = FakeProvider(replies=['{"sentiment": "negative"}'])
        runner = BenchmarkRunner(provider, config)
        result = runner.run(
            evaluation_set.head(1), ["structured"], persist=False,
            experiment_id="2026-08-24_001", show_progress=False,
        )
        row = result.predictions.iloc[0]
        assert row["raw_response"] == '{"sentiment": "negative"}'
        assert row["predicted_label"] == NEG

    def test_latency_and_usage_are_recorded(self, result: Any) -> None:
        row = result.predictions.iloc[0]
        assert row["latency_seconds"] == pytest.approx(0.01)
        assert row["prompt_tokens"] == 100
        assert row["total_tokens"] == 101

    def test_prompt_size_is_recorded(self, result: Any) -> None:
        assert (result.predictions["prompt_chars"] > 0).all()

    def test_successful_rows_have_no_error(self, result: Any) -> None:
        assert result.predictions["success"].all()
        assert result.predictions["error"].isna().all()


class TestFailureTracking:
    def test_api_failure_is_recorded_not_dropped(
        self, evaluation_set: pd.DataFrame, config: AppConfig
    ) -> None:
        provider = FakeProvider(replies=["positive", failed_response(), "negative"])
        runner = BenchmarkRunner(provider, config)
        result = runner.run(
            evaluation_set.head(3), ["zero_shot"], persist=False,
            experiment_id="2026-08-24_001", show_progress=False,
        )
        # The denominator is preserved: three samples asked, three rows stored.
        assert len(result.predictions) == 3
        failed = result.predictions[~result.predictions["success"]]
        assert len(failed) == 1
        assert failed.iloc[0]["predicted_label"] == UNK
        assert "LLMTransientError" in failed.iloc[0]["error"]

    def test_api_failures_are_counted_per_strategy(
        self, evaluation_set: pd.DataFrame, config: AppConfig
    ) -> None:
        provider = FakeProvider(replies=[failed_response(), failed_response(), "positive"])
        runner = BenchmarkRunner(provider, config)
        result = runner.run(
            evaluation_set.head(3), ["zero_shot"], persist=False,
            experiment_id="2026-08-24_001", show_progress=False,
        )
        stats = result.stats[0]
        assert stats.api_failures == 2
        assert stats.successful_calls == 1
        assert stats.samples == 3

    def test_unparseable_responses_are_counted(
        self, evaluation_set: pd.DataFrame, config: AppConfig
    ) -> None:
        provider = FakeProvider(replies=["positive", "I cannot decide", "maybe both"])
        runner = BenchmarkRunner(provider, config)
        result = runner.run(
            evaluation_set.head(3), ["zero_shot"], persist=False,
            experiment_id="2026-08-24_001", show_progress=False,
        )
        assert result.stats[0].unknown_predictions == 2
        assert result.stats[0].api_failures == 0  # the calls succeeded

    def test_unparseable_is_distinct_from_api_failure(
        self, evaluation_set: pd.DataFrame, config: AppConfig
    ) -> None:
        provider = FakeProvider(replies=["nonsense", failed_response()])
        runner = BenchmarkRunner(provider, config)
        result = runner.run(
            evaluation_set.head(2), ["zero_shot"], persist=False,
            experiment_id="2026-08-24_001", show_progress=False,
        )
        stats = result.stats[0]
        assert stats.unknown_predictions == 2  # both end up unlabelled
        assert stats.api_failures == 1         # but only one call failed

    def test_auth_failure_aborts_the_run(
        self, evaluation_set: pd.DataFrame, config: AppConfig
    ) -> None:
        provider = FakeProvider(replies=["positive", LLMAuthError("bad key")])
        runner = BenchmarkRunner(provider, config)
        with pytest.raises(LLMAuthError):
            runner.run(
                evaluation_set, ["zero_shot"], persist=False,
                experiment_id="2026-08-24_001", show_progress=False,
            )

    def test_runtime_is_tracked_per_strategy_and_overall(
        self, evaluation_set: pd.DataFrame, examples: Any, config: AppConfig
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), config)
        result = runner.run(
            evaluation_set.head(2), ["zero_shot", "few_shot"], examples,
            persist=False, experiment_id="2026-08-24_001", show_progress=False,
        )
        assert all(stat.runtime_seconds >= 0 for stat in result.stats)
        assert result.runtime_seconds >= 0


class TestDevelopmentMode:
    def test_limits_the_sample_count(
        self, evaluation_set: pd.DataFrame, config: AppConfig
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), config)  # dev_sample_size=2
        result = runner.run(
            evaluation_set, ["zero_shot"], dev_mode=True, persist=False,
            experiment_id="2026-08-24_001", show_progress=False,
        )
        assert len(result.predictions) == 2

    def test_uses_the_head_so_a_dev_run_is_a_subset(
        self, evaluation_set: pd.DataFrame, config: AppConfig
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), config)
        dev = runner.run(
            evaluation_set, ["zero_shot"], dev_mode=True, persist=False,
            experiment_id="2026-08-24_001", show_progress=False,
        )
        full = runner.run(
            evaluation_set, ["zero_shot"], dev_mode=False, persist=False,
            experiment_id="2026-08-24_002", show_progress=False,
        )
        assert set(dev.predictions["sample_id"]) <= set(full.predictions["sample_id"])
        assert dev.predictions["sample_id"].tolist() == (
            full.predictions["sample_id"].tolist()[:2]
        )

    def test_is_recorded_in_the_metadata(
        self, evaluation_set: pd.DataFrame, config: AppConfig
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), config)
        result = runner.run(
            evaluation_set, ["zero_shot"], dev_mode=True, persist=False,
            experiment_id="2026-08-24_001", show_progress=False,
        )
        assert result.config["dev_mode"] is True
        assert result.config["dataset"]["sample_count"] == 2

    def test_never_exceeds_the_available_samples(
        self, evaluation_set: pd.DataFrame
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), AppConfig(dev_sample_size=999))
        result = runner.run(
            evaluation_set, ["zero_shot"], dev_mode=True, persist=False,
            experiment_id="2026-08-24_001", show_progress=False,
        )
        assert len(result.predictions) == len(evaluation_set)


class TestRunnerValidation:
    def test_rejects_an_invalid_evaluation_set(self, config: AppConfig) -> None:
        runner = BenchmarkRunner(FakeProvider(), config)
        with pytest.raises(RunnerError, match="Invalid evaluation set"):
            runner.run(
                pd.DataFrame({"wrong": [1]}), ["zero_shot"], persist=False,
                experiment_id="2026-08-24_001", show_progress=False,
            )

    def test_rejects_an_empty_strategy_list(
        self, evaluation_set: pd.DataFrame, config: AppConfig
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), config)
        with pytest.raises(RunnerError, match="At least one strategy"):
            runner.run(
                evaluation_set, [], persist=False,
                experiment_id="2026-08-24_001", show_progress=False,
            )

    def test_fails_before_spending_tokens_when_examples_are_missing(
        self, evaluation_set: pd.DataFrame, config: AppConfig
    ) -> None:
        provider = FakeProvider()
        runner = BenchmarkRunner(provider, config)
        with pytest.raises(RunnerError, match="require few-shot examples"):
            runner.run(
                evaluation_set, ["few_shot"], persist=False,
                experiment_id="2026-08-24_001", show_progress=False,
            )
        assert provider.prompts == []  # not a single call was made

    def test_persist_without_storage_is_refused(
        self, evaluation_set: pd.DataFrame, config: AppConfig
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), config)
        with pytest.raises(RunnerError, match="requires a storage backend"):
            runner.run(evaluation_set, ["zero_shot"], show_progress=False)

    def test_accepts_strategy_instances_as_well_as_names(
        self, evaluation_set: pd.DataFrame, config: AppConfig
    ) -> None:
        from src.prompts import get_strategy

        runner = BenchmarkRunner(FakeProvider(), config)
        result = runner.run(
            evaluation_set.head(1), [get_strategy("zero_shot")], persist=False,
            experiment_id="2026-08-24_001", show_progress=False,
        )
        assert result.predictions["strategy"].iloc[0] == "zero_shot"


# ---------------------------------------------------------------------------
# Persistence and metadata
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_writes_both_artefacts(
        self,
        evaluation_set: pd.DataFrame,
        examples: Any,
        config: AppConfig,
        storage: ExperimentStorage,
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), config, storage=storage)
        result = runner.run(
            evaluation_set, ["zero_shot", "few_shot"], examples, show_progress=False
        )
        directory = storage.experiment_path(result.experiment_id)
        assert (directory / CONFIG_FILENAME).exists()
        assert (directory / PREDICTIONS_FILENAME).exists()
        assert result.experiment_dir == directory

    def test_two_runs_do_not_overwrite_each_other(
        self, evaluation_set: pd.DataFrame, config: AppConfig, storage: ExperimentStorage
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), config, storage=storage)
        first = runner.run(evaluation_set, ["zero_shot"], show_progress=False)
        second = runner.run(evaluation_set, ["zero_shot"], show_progress=False)
        assert first.experiment_id != second.experiment_id
        assert len(storage.list_experiments()) == 2
        assert storage.load_predictions(first.experiment_id) is not None

    def test_stored_predictions_match_the_returned_frame(
        self, evaluation_set: pd.DataFrame, config: AppConfig, storage: ExperimentStorage
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), config, storage=storage)
        result = runner.run(evaluation_set, ["zero_shot"], show_progress=False)
        stored = storage.load_predictions(result.experiment_id)
        assert len(stored) == len(result.predictions)
        assert stored["sample_id"].tolist() == result.predictions["sample_id"].tolist()

    def test_run_without_persistence_writes_nothing(
        self, evaluation_set: pd.DataFrame, config: AppConfig, storage: ExperimentStorage
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), config, storage=storage)
        runner.run(evaluation_set, ["zero_shot"], persist=False, show_progress=False)
        assert storage.list_experiments() == []


class TestExperimentMetadata:
    @pytest.fixture
    def config_payload(
        self,
        evaluation_set: pd.DataFrame,
        examples: Any,
        config: AppConfig,
        storage: ExperimentStorage,
    ) -> dict[str, Any]:
        runner = BenchmarkRunner(FakeProvider(), config, storage=storage)
        result = runner.run(
            evaluation_set, ["zero_shot", "few_shot"], examples, show_progress=False
        )
        return storage.load_config(result.experiment_id)

    def test_records_the_reproducibility_fields(
        self, config_payload: dict[str, Any]
    ) -> None:
        assert config_payload["experiment_id"]
        assert config_payload["timestamp"]
        assert config_payload["provider"]["model"] == "fake-model"
        assert config_payload["configuration"]["temperature"] == 0.0
        assert config_payload["dataset"]["random_seed"] == 42
        assert config_payload["dataset"]["sample_count"] == 6

    def test_records_the_strategies_with_their_hypotheses(
        self, config_payload: dict[str, Any]
    ) -> None:
        strategies = config_payload["prompt_strategies"]
        assert [entry["name"] for entry in strategies] == ["zero_shot", "few_shot"]
        assert all(entry["hypothesis"] for entry in strategies)

    def test_records_the_evaluation_sample_ids_and_checksum(
        self, config_payload: dict[str, Any]
    ) -> None:
        dataset = config_payload["dataset"]
        assert len(dataset["sample_ids"]) == 6
        assert dataset["sample_id_checksum"] == checksum_sample_ids(dataset["sample_ids"])

    def test_records_operational_stats(self, config_payload: dict[str, Any]) -> None:
        stats = config_payload["run_stats"]
        assert len(stats) == 2
        assert all("api_failures" in entry for entry in stats)

    def test_config_contains_no_credentials(
        self, config_payload: dict[str, Any]
    ) -> None:
        serialised = json.dumps(config_payload).lower()
        assert "api_key" not in serialised
        assert "gemini_api_key" not in serialised


class TestChecksum:
    def test_same_samples_give_the_same_checksum(self) -> None:
        assert checksum_sample_ids(["a", "b"]) == checksum_sample_ids(["a", "b"])

    def test_different_samples_differ(self) -> None:
        assert checksum_sample_ids(["a", "b"]) != checksum_sample_ids(["a", "c"])

    def test_order_matters(self) -> None:
        assert checksum_sample_ids(["a", "b"]) != checksum_sample_ids(["b", "a"])


# ---------------------------------------------------------------------------
# End-to-end with the real evaluation modules
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_predictions_can_be_scored_and_error_analysed(
        self, evaluation_set: pd.DataFrame, config: AppConfig
    ) -> None:
        from src.evaluation.errors import extract_errors
        from src.evaluation.metrics import compute_prediction_counts

        # Four correct, one wrong, one unparseable.
        replies = ["positive", "negative", "positive", "negative", "negative", "hmm"]
        runner = BenchmarkRunner(FakeProvider(replies=replies), config)
        result = runner.run(
            evaluation_set, ["zero_shot"], persist=False,
            experiment_id="2026-08-24_001", show_progress=False,
        )

        counts = compute_prediction_counts(
            result.predictions["actual_label"], result.predictions["predicted_label"]
        )
        assert counts["total_samples"] == 6
        assert counts["correct"] == 4
        assert counts["unknown"] == 1

        errors = extract_errors(result.predictions)
        assert len(errors) == counts["incorrect"] == 2


# ---------------------------------------------------------------------------
# Evaluation step (Step 9)
# ---------------------------------------------------------------------------


class ScriptedProvider:
    """Returns a fixed reply per strategy, identified by the prompt's shape."""

    model = "fake-model"

    def __init__(self, scripts: dict[str, list[str]]) -> None:
        self._scripts = scripts
        self._counts: dict[str, int] = {}

    def describe(self) -> dict[str, Any]:
        return {"provider": "fake", "model": self.model, "temperature": 0.0}

    @staticmethod
    def _identify(prompt: Any) -> str:
        if '"sentiment"' in prompt.user_text:
            return "structured"
        if "silently" in prompt.user_text:
            return "reasoning"
        return "zero_shot"

    def generate(self, prompt: Any) -> LLMResponse:
        name = self._identify(prompt)
        index = self._counts.get(name, 0)
        self._counts[name] = index + 1
        return LLMResponse(
            text=self._scripts[name][index],
            success=True,
            latency_seconds=0.1,
            model=self.model,
            usage=UsageMetadata(prompt_tokens=100, output_tokens=2, total_tokens=102),
            finish_reason="STOP",
        )


@pytest.fixture
def eight_samples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"test-{index:05d}" for index in range(8)],
            "review": [f"review number {index}" for index in range(8)],
            "label": [POS] * 4 + [NEG] * 4,
        }
    )


@pytest.fixture
def scored_run(
    eight_samples: pd.DataFrame, config: AppConfig, storage: ExperimentStorage
) -> tuple[Any, Any, ExperimentStorage]:
    """A complete mocked experiment, run then scored.

    Ground truth       : P P P P N N N N
    zero_shot  replies : P P N N N N N N   -> 6/8 correct, no unknowns
    structured replies : P P P P N N N N   -> 8/8 correct
    reasoning  replies : two unparseable   -> 6/8 correct, 2 unknown
    """
    scripts = {
        "zero_shot": [POS, POS, NEG, NEG, NEG, NEG, NEG, NEG],
        "structured": [POS, POS, POS, POS, NEG, NEG, NEG, NEG],
        "reasoning": [POS, POS, "hmm, hard to say", POS, NEG, NEG, NEG, "who knows"],
    }
    runner = BenchmarkRunner(ScriptedProvider(scripts), config, storage=storage)
    result = runner.run(
        eight_samples, ["zero_shot", "structured", "reasoning"], show_progress=False
    )
    return result, runner.evaluate(result), storage


class TestEvaluationArtefacts:
    def test_writes_metrics_and_summary(self, scored_run: Any) -> None:
        result, _, storage = scored_run
        directory = storage.experiment_path(result.experiment_id)
        assert sorted(path.name for path in directory.iterdir()) == [
            CONFIG_FILENAME,
            METRICS_FILENAME,
            PREDICTIONS_FILENAME,
            SUMMARY_FILENAME,
        ]

    def test_metrics_round_trip(self, scored_run: Any) -> None:
        result, evaluation, storage = scored_run
        stored = storage.load_metrics(result.experiment_id)
        assert stored["strategy"].tolist() == evaluation.metrics["strategy"].tolist()
        assert stored["f1_macro"].tolist() == pytest.approx(
            evaluation.metrics["f1_macro"].tolist()
        )

    def test_summary_round_trips(self, scored_run: Any) -> None:
        result, evaluation, storage = scored_run
        assert storage.load_summary(result.experiment_id) == evaluation.summary

    def test_evaluation_without_persistence_writes_nothing(
        self, eight_samples: pd.DataFrame, config: AppConfig, storage: ExperimentStorage
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), config, storage=storage)
        result = runner.run(
            eight_samples, ["zero_shot"], persist=False,
            experiment_id="2026-08-24_001", show_progress=False,
        )
        runner.evaluate(result, persist=False)
        assert storage.list_experiments() == []

    def test_persisting_without_storage_is_refused(
        self, eight_samples: pd.DataFrame, config: AppConfig
    ) -> None:
        runner = BenchmarkRunner(FakeProvider(), config)
        result = runner.run(
            eight_samples, ["zero_shot"], persist=False,
            experiment_id="2026-08-24_001", show_progress=False,
        )
        with pytest.raises(RunnerError, match="requires a storage backend"):
            runner.evaluate(result)


class TestScoredMetrics:
    def test_one_row_per_strategy_in_execution_order(self, scored_run: Any) -> None:
        _, evaluation, _ = scored_run
        assert evaluation.metrics["strategy"].tolist() == [
            "zero_shot",
            "structured",
            "reasoning",
        ]

    def test_hand_verified_numbers(self, scored_run: Any) -> None:
        _, evaluation, _ = scored_run
        metrics = evaluation.metrics.set_index("strategy")

        assert metrics.loc["zero_shot", "correct"] == 6
        assert metrics.loc["zero_shot", "accuracy"] == 0.75
        assert metrics.loc["zero_shot", "f1_macro"] == pytest.approx(0.7333, abs=1e-4)

        assert metrics.loc["structured", "accuracy"] == 1.0
        assert metrics.loc["structured", "f1_macro"] == 1.0

        assert metrics.loc["reasoning", "unknown"] == 2
        assert metrics.loc["reasoning", "unknown_rate"] == 0.25
        assert metrics.loc["reasoning", "accuracy"] == 0.75
        assert metrics.loc["reasoning", "f1_macro"] == pytest.approx(0.8571, abs=1e-4)

    def test_average_latency_is_recorded(self, scored_run: Any) -> None:
        _, evaluation, _ = scored_run
        assert evaluation.metrics["avg_latency_seconds"].tolist() == pytest.approx(
            [0.1, 0.1, 0.1]
        )

    def test_metrics_agree_with_the_stored_predictions(self, scored_run: Any) -> None:
        """metrics.csv must be re-derivable from predictions.csv alone."""
        result, evaluation, storage = scored_run
        recomputed = evaluate_strategies(storage.load_predictions(result.experiment_id))
        pd.testing.assert_series_equal(
            recomputed["f1_macro"], evaluation.metrics["f1_macro"]
        )
        pd.testing.assert_series_equal(
            recomputed["correct"], evaluation.metrics["correct"]
        )


class TestSummaryContent:
    def test_identifies_the_best_strategy_by_f1(self, scored_run: Any) -> None:
        _, evaluation, _ = scored_run
        assert evaluation.summary["selection_metric"] == "f1_macro"
        assert evaluation.summary["best_strategy"]["strategy"] == "structured"
        assert evaluation.best_strategy == "structured"

    def test_ranking_is_ordered_by_f1(self, scored_run: Any) -> None:
        _, evaluation, _ = scored_run
        ranking = evaluation.summary["ranking"]
        assert [entry["strategy"] for entry in ranking] == [
            "structured",
            "reasoning",
            "zero_shot",
        ]
        assert [entry["rank"] for entry in ranking] == [1, 2, 3]

    def test_ranking_exposes_the_unknown_rate(self, scored_run: Any) -> None:
        # F1 alone hides abstention; the rate sits beside it so a strategy
        # cannot look strong purely by declining hard samples.
        _, evaluation, _ = scored_run
        reasoning = next(
            entry
            for entry in evaluation.summary["ranking"]
            if entry["strategy"] == "reasoning"
        )
        assert reasoning["unknown_rate"] == 0.25

    def test_totals_reconcile_with_the_predictions(self, scored_run: Any) -> None:
        result, evaluation, _ = scored_run
        totals = evaluation.summary["totals"]
        assert totals["predictions"] == len(result.predictions)
        assert totals["correct"] + totals["incorrect"] == totals["predictions"]
        assert totals["unknown"] == 2
        assert totals["api_failures"] == 0

    def test_carries_the_reproducibility_fields(self, scored_run: Any) -> None:
        result, evaluation, _ = scored_run
        summary = evaluation.summary
        assert summary["experiment_id"] == result.experiment_id
        assert summary["model"] == "fake-model"
        assert summary["temperature"] == 0.0
        assert summary["dataset"]["sample_count"] == 8
        assert summary["dataset"]["random_seed"] == 42
        assert summary["dataset"]["sample_id_checksum"] == (
            result.config["dataset"]["sample_id_checksum"]
        )

    def test_refuses_to_name_a_winner_on_a_tie(
        self, eight_samples: pd.DataFrame, config: AppConfig
    ) -> None:
        # Both strategies score identically, so no single best exists.
        identical = [POS, POS, POS, POS, NEG, NEG, NEG, NEG]
        provider = ScriptedProvider(
            {"zero_shot": identical, "structured": identical, "reasoning": identical}
        )
        runner = BenchmarkRunner(provider, config)
        result = runner.run(
            eight_samples, ["zero_shot", "structured"], persist=False,
            experiment_id="2026-08-24_001", show_progress=False,
        )
        evaluation = runner.evaluate(result, persist=False)
        assert evaluation.best_strategy is None
        assert evaluation.summary["best_strategy"] is None
        assert "tied" in evaluation.summary["best_strategy_note"]

    def test_summary_is_json_serialisable(self, scored_run: Any) -> None:
        _, evaluation, _ = scored_run
        assert json.loads(json.dumps(evaluation.summary))

    def test_summary_contains_no_review_text_or_credentials(
        self, scored_run: Any
    ) -> None:
        _, evaluation, _ = scored_run
        serialised = json.dumps(evaluation.summary).lower()
        assert "api_key" not in serialised
        assert "review number" not in serialised


class TestRescoringStoredResults:
    def test_a_stored_experiment_can_be_rescored_without_api_calls(
        self, scored_run: Any
    ) -> None:
        result, evaluation, storage = scored_run
        stored_predictions = storage.load_predictions(result.experiment_id)

        rescored = evaluate_strategies(stored_predictions)
        summary = build_experiment_summary(
            result.experiment_id, rescored, config=storage.load_config(result.experiment_id)
        )
        assert summary["best_strategy"]["strategy"] == (
            evaluation.summary["best_strategy"]["strategy"]
        )
        assert summary["totals"]["correct"] == evaluation.summary["totals"]["correct"]
