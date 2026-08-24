"""Tests for the command-line interface.

Argument parsing is tested directly. Execution is tested with the provider and
the dataset loader both replaced, so **no test makes an API call or downloads
anything**, and no test writes into the real ``results/`` directory.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src import cli  # noqa: E402
from src.config import ENV_API_KEY, LABEL_NEGATIVE, LABEL_POSITIVE, PromptStrategy  # noqa: E402
from src.experiments.storage import ExperimentStorage  # noqa: E402
from src.llm.base import LLMResponse, UsageMetadata  # noqa: E402

POS = LABEL_POSITIVE
NEG = LABEL_NEGATIVE


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    def test_requires_a_subcommand(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([])

    @pytest.mark.parametrize("command", ["dev", "benchmark"])
    def test_both_commands_parse(self, command: str) -> None:
        assert cli.build_parser().parse_args([command]).command == command

    def test_sample_size_is_configurable(self) -> None:
        assert cli.build_parser().parse_args(["benchmark", "--samples", "100"]).samples == 100

    def test_experiment_id_is_configurable(self) -> None:
        args = cli.build_parser().parse_args(
            ["benchmark", "--experiment-id", "2026-08-24_007"]
        )
        assert args.experiment_id == "2026-08-24_007"

    def test_strategies_accept_several_names(self) -> None:
        args = cli.build_parser().parse_args(
            ["benchmark", "--strategies", "zero_shot", "combined"]
        )
        assert args.strategies == ["zero_shot", "combined"]

    def test_unknown_strategy_is_rejected(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["benchmark", "--strategies", "chain_of_thought"])

    def test_defaults_are_conservative(self) -> None:
        args = cli.build_parser().parse_args(["benchmark"])
        assert args.samples is None       # falls back to configuration
        assert args.strategies is None    # all strategies
        assert args.no_save is False
        assert args.no_charts is False
        assert args.yes is False

    def test_flags_parse(self) -> None:
        args = cli.build_parser().parse_args(
            ["dev", "--no-charts", "--no-save", "--yes", "--quiet"]
        )
        assert (args.no_charts, args.no_save, args.yes, args.quiet) == (True, True, True, True)

    def test_model_and_temperature_parse(self) -> None:
        args = cli.build_parser().parse_args(
            ["benchmark", "--model", "gemini-x", "--temperature", "0.7"]
        )
        assert args.model == "gemini-x"
        assert args.temperature == 0.7


class TestConfigResolution:
    def test_dev_uses_the_small_sample_size(self) -> None:
        args = cli.build_parser().parse_args(["dev"])
        config = cli.resolve_config(args)
        assert config.benchmark_sample_size == config.dev_sample_size

    def test_explicit_samples_win_over_dev_default(self) -> None:
        args = cli.build_parser().parse_args(["dev", "--samples", "25"])
        assert cli.resolve_config(args).benchmark_sample_size == 25

    def test_overrides_are_applied(self) -> None:
        args = cli.build_parser().parse_args(
            ["benchmark", "--model", "gemini-x", "--temperature", "0.4", "--seed", "7"]
        )
        config = cli.resolve_config(args)
        assert config.model == "gemini-x"
        assert config.temperature == 0.4
        assert config.random_seed == 7

    def test_strategy_subset_is_applied_in_order(self) -> None:
        args = cli.build_parser().parse_args(
            ["benchmark", "--strategies", "combined", "zero_shot"]
        )
        assert cli.resolve_config(args).prompt_strategies == (
            PromptStrategy.COMBINED,
            PromptStrategy.ZERO_SHOT,
        )

    def test_no_overrides_leaves_configuration_untouched(self) -> None:
        from src.config import AppConfig

        args = cli.build_parser().parse_args(["benchmark"])
        assert cli.resolve_config(args) == AppConfig.from_env()


class TestConfirmation:
    def test_small_runs_do_not_ask(self) -> None:
        assert cli.confirm_run(10, assume_yes=False) is True

    def test_yes_flag_skips_the_prompt(self) -> None:
        assert cli.confirm_run(100_000, assume_yes=True) is True

    def test_non_interactive_large_run_is_refused(self) -> None:
        # A piped session cannot answer, so it must not block or assume yes.
        assert cli.confirm_run(100_000, assume_yes=False, stream=io.StringIO()) is False


# ---------------------------------------------------------------------------
# Mocked execution
# ---------------------------------------------------------------------------


class FakeProvider:
    """Answers every prompt correctly, without touching the network."""

    model = "fake-model"

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.calls = 0

    def describe(self) -> dict[str, Any]:
        return {"provider": "fake", "model": self.model, "temperature": 0.0}

    def generate(self, prompt: Any) -> LLMResponse:
        self.calls += 1
        # The synthetic reviews carry their own label in the text.
        text = POS if "wonderful" in prompt.user_text else NEG
        return LLMResponse(
            text=text, success=True, latency_seconds=0.01, model=self.model,
            usage=UsageMetadata(prompt_tokens=100, output_tokens=1, total_tokens=101),
            finish_reason="STOP",
        )


def fake_raw_split(_split: str, **_kwargs: Any) -> pd.DataFrame:
    """Stand-in for the Hugging Face loader: 12 balanced synthetic reviews."""
    return pd.DataFrame(
        {
            "text": [
                f"A wonderful film, number {index}." if index % 2 else
                f"A dreadful film, number {index}."
                for index in range(12)
            ],
            "label": [index % 2 for index in range(12)],
        }
    )


@pytest.fixture
def mocked_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    """Replace the network-facing pieces and redirect results into tmp_path."""
    monkeypatch.setenv(ENV_API_KEY, "test-key-not-real")
    monkeypatch.setenv("PROMPTBENCH_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("PROMPTBENCH_DEV_SAMPLE_SIZE", "4")
    monkeypatch.setenv("PROMPTBENCH_BENCHMARK_SAMPLE_SIZE", "6")

    provider = FakeProvider()
    monkeypatch.setattr(cli, "GeminiProvider", lambda *a, **k: provider)
    monkeypatch.setattr(cli, "load_imdb_dataset", fake_raw_split)
    return {"provider": provider, "results": tmp_path / "results"}


class TestMockedExecution:
    def test_dev_run_completes_successfully(self, mocked_cli: dict[str, Any]) -> None:
        assert cli.main(["dev", "--yes", "--no-charts"]) == cli.EXIT_OK

    def test_dev_run_uses_the_small_sample_size(
        self, mocked_cli: dict[str, Any]
    ) -> None:
        cli.main(["dev", "--yes", "--no-charts", "--strategies", "zero_shot"])
        # 4 dev samples x 1 strategy
        assert mocked_cli["provider"].calls == 4

    def test_benchmark_run_uses_the_full_sample_size(
        self, mocked_cli: dict[str, Any]
    ) -> None:
        cli.main(["benchmark", "--yes", "--no-charts", "--strategies", "zero_shot"])
        assert mocked_cli["provider"].calls == 6

    def test_sample_override_is_honoured(self, mocked_cli: dict[str, Any]) -> None:
        cli.main(["benchmark", "--yes", "--no-charts", "--samples", "2",
                  "--strategies", "zero_shot", "few_shot"])
        assert mocked_cli["provider"].calls == 4

    def test_writes_the_four_artefacts(self, mocked_cli: dict[str, Any]) -> None:
        cli.main(["dev", "--yes", "--no-charts", "--strategies", "zero_shot"])
        storage = ExperimentStorage(mocked_cli["results"] / "experiments")
        experiments = storage.list_experiments()
        assert len(experiments) == 1
        directory = storage.experiment_path(experiments[0])
        assert sorted(path.name for path in directory.iterdir()) == [
            "config.json", "metrics.csv", "predictions.csv", "summary.json",
        ]

    def test_renders_charts_by_default(self, mocked_cli: dict[str, Any]) -> None:
        cli.main(["dev", "--yes", "--strategies", "zero_shot", "structured"])
        storage = ExperimentStorage(mocked_cli["results"] / "experiments")
        charts = storage.experiment_path(storage.list_experiments()[0]) / "charts"
        assert charts.is_dir()
        assert {path.suffix for path in charts.iterdir()} == {".png"}

    def test_no_save_writes_nothing(self, mocked_cli: dict[str, Any]) -> None:
        assert cli.main(["dev", "--yes", "--no-save", "--strategies", "zero_shot"]) == cli.EXIT_OK
        assert not (mocked_cli["results"] / "experiments").exists()

    def test_explicit_experiment_id_is_used(self, mocked_cli: dict[str, Any]) -> None:
        cli.main(["dev", "--yes", "--no-charts", "--strategies", "zero_shot",
                  "--experiment-id", "2026-08-24_042"])
        storage = ExperimentStorage(mocked_cli["results"] / "experiments")
        assert storage.list_experiments() == ["2026-08-24_042"]

    def test_two_runs_do_not_overwrite_each_other(
        self, mocked_cli: dict[str, Any]
    ) -> None:
        cli.main(["dev", "--yes", "--no-charts", "--strategies", "zero_shot"])
        cli.main(["dev", "--yes", "--no-charts", "--strategies", "zero_shot"])
        storage = ExperimentStorage(mocked_cli["results"] / "experiments")
        assert len(storage.list_experiments()) == 2

    def test_every_strategy_sees_the_same_samples(
        self, mocked_cli: dict[str, Any]
    ) -> None:
        cli.main(["dev", "--yes", "--no-charts",
                  "--strategies", "zero_shot", "few_shot", "combined"])
        storage = ExperimentStorage(mocked_cli["results"] / "experiments")
        predictions = storage.load_predictions(storage.list_experiments()[0])
        per_strategy = predictions.groupby("strategy")["sample_id"].apply(list)
        assert all(ids == per_strategy.iloc[0] for ids in per_strategy)

    def test_summary_is_printed(
        self, mocked_cli: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["dev", "--yes", "--no-charts", "--strategies", "zero_shot"])
        output = capsys.readouterr().out
        assert "Experiment 2026-" in output
        assert "Best by F1" in output
        assert "accuracy" in output
        # The caveat about F1 and abstention must reach the reader.
        assert "unparseable answer" in output

    def test_summary_reports_where_results_were_written(
        self, mocked_cli: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["dev", "--yes", "--no-charts", "--strategies", "zero_shot"])
        assert "predictions.csv" in capsys.readouterr().out

    def test_no_save_run_says_nothing_was_written(
        self, mocked_cli: dict[str, Any], capsys: pytest.CaptureFixture[str]
    ) -> None:
        cli.main(["dev", "--yes", "--no-save", "--strategies", "zero_shot"])
        assert "nothing written" in capsys.readouterr().out


class TestFailureModes:
    def test_missing_api_key_exits_with_a_config_code(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.delenv(ENV_API_KEY, raising=False)
        monkeypatch.setattr("src.config.load_environment", lambda *a, **k: False)
        monkeypatch.setenv("PROMPTBENCH_RESULTS_DIR", str(tmp_path))
        assert cli.main(["dev", "--yes"]) == cli.EXIT_CONFIG
        assert "GEMINI_API_KEY" in capsys.readouterr().err

    def test_no_api_call_is_made_without_a_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv(ENV_API_KEY, raising=False)
        monkeypatch.setattr("src.config.load_environment", lambda *a, **k: False)
        monkeypatch.setenv("PROMPTBENCH_RESULTS_DIR", str(tmp_path))
        provider = FakeProvider()
        monkeypatch.setattr(cli, "GeminiProvider", lambda *a, **k: provider)
        cli.main(["dev", "--yes"])
        assert provider.calls == 0

    def test_refused_confirmation_aborts_before_calling(
        self, mocked_cli: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli, "confirm_run", lambda *a, **k: False)
        assert cli.main(["benchmark"]) == cli.EXIT_FAILURE
        assert mocked_cli["provider"].calls == 0

    def test_dataset_failure_is_reported_cleanly(
        self, mocked_cli: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from src.dataset import DatasetError

        def boom(*_args: Any, **_kwargs: Any) -> None:
            raise DatasetError("hub unreachable")

        monkeypatch.setattr(cli, "load_imdb_dataset", boom)
        assert cli.main(["dev", "--yes"]) == cli.EXIT_FAILURE
        assert "hub unreachable" in capsys.readouterr().err

    def test_auth_failure_exits_with_a_config_code(
        self, mocked_cli: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.llm.base import LLMAuthError

        class Unauthorised(FakeProvider):
            def generate(self, prompt: Any) -> LLMResponse:
                raise LLMAuthError("invalid key")

        monkeypatch.setattr(cli, "GeminiProvider", lambda *a, **k: Unauthorised())
        assert cli.main(["dev", "--yes", "--no-charts"]) == cli.EXIT_CONFIG


class TestEntryPointShim:
    def test_top_level_module_exposes_main(self) -> None:
        import promptbench

        assert promptbench.main is cli.main
