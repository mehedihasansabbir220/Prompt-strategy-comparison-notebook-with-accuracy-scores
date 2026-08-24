"""Command-line interface for running PromptBench experiments.

Two subcommands, one pipeline::

    python -m promptbench dev                    # small smoke run
    python -m promptbench benchmark --samples 100

Both walk the same eleven steps — load configuration, load the dataset, draw
the fixed evaluation subset and the fixed demonstrations, initialise the
provider, run every strategy, save predictions, score them, save the results,
render the charts, and print a summary.

The CLI holds no evaluation logic of its own. It wires together the same
modules the notebook and the dashboard use, so a run started here is
indistinguishable from one started anywhere else.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import AppConfig, ConfigurationError, PromptStrategy, has_api_key
from src.dataset import (
    DatasetError,
    FewShotExample,
    create_few_shot_examples,
    create_fixed_evaluation_set,
    load_imdb_dataset,
    prepare_dataset,
)
from src.experiments.runner import BenchmarkRunner, ExperimentEvaluation, ExperimentResult
from src.experiments.storage import ExperimentStorage, StorageError
from src.llm.base import LLMAuthError
from src.llm.gemini import GeminiProvider
from src.visualization.charts import render_all_charts

logger = logging.getLogger("promptbench")

#: Above this many calls an interactive run asks for confirmation first, so a
#: mistyped ``--samples`` cannot quietly spend an afternoon of quota.
CONFIRMATION_THRESHOLD: int = 120

#: Default demonstrations per class for the few-shot family.
DEFAULT_EXAMPLES_PER_CLASS: int = 2

EXIT_OK: int = 0
EXIT_FAILURE: int = 1
EXIT_CONFIG: int = 2
EXIT_INTERRUPTED: int = 130


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Exposed separately so the parser can be tested without running anything.
    """
    parser = argparse.ArgumentParser(
        prog="promptbench",
        description="Compare prompt strategies on a controlled sentiment benchmark.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--samples", type=int, default=None,
            help="Evaluation samples per strategy (default: from configuration).",
        )
        target.add_argument(
            "--strategies", nargs="+", default=None,
            metavar="NAME", choices=[str(member) for member in PromptStrategy],
            help="Strategies to compare (default: all, in configured order).",
        )
        target.add_argument("--seed", type=int, default=None, help="Sampling seed.")
        target.add_argument("--model", default=None, help="Model identifier.")
        target.add_argument(
            "--temperature", type=float, default=None, help="Decoding temperature."
        )
        target.add_argument(
            "--experiment-id", default=None,
            help="Use this id instead of allocating the next one (YYYY-MM-DD_NNN).",
        )
        target.add_argument(
            "--examples-per-class", type=int, default=DEFAULT_EXAMPLES_PER_CLASS,
            help=f"Few-shot demonstrations per class (default: {DEFAULT_EXAMPLES_PER_CLASS}).",
        )
        target.add_argument(
            "--no-charts", action="store_true", help="Skip chart rendering."
        )
        target.add_argument(
            "--no-save", action="store_true",
            help="Run without writing anything to results/.",
        )
        target.add_argument(
            "--yes", "-y", action="store_true",
            help="Skip the confirmation prompt for a large run.",
        )
        target.add_argument(
            "--quiet", action="store_true", help="Only print the final summary."
        )
        target.add_argument("--verbose", action="store_true", help="Debug logging.")

    development = subparsers.add_parser(
        "dev", help="Small smoke run to check the pipeline end to end."
    )
    add_common(development)

    benchmark = subparsers.add_parser(
        "benchmark", help="Full benchmark run over every strategy."
    )
    add_common(benchmark)

    return parser


def resolve_config(args: argparse.Namespace) -> AppConfig:
    """Apply command-line overrides on top of the environment configuration.

    Development mode substitutes the small ``dev_sample_size`` unless the user
    asked for a specific ``--samples``.
    """
    config = AppConfig.from_env()
    overrides: dict[str, Any] = {}

    if args.model:
        overrides["model"] = args.model
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.seed is not None:
        overrides["random_seed"] = args.seed

    if args.samples is not None:
        overrides["benchmark_sample_size"] = args.samples
    elif args.command == "dev":
        overrides["benchmark_sample_size"] = config.dev_sample_size

    if args.strategies:
        overrides["prompt_strategies"] = tuple(
            PromptStrategy(name) for name in args.strategies
        )

    return config.with_overrides(**overrides) if overrides else config


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def prepare_data(
    config: AppConfig, *, examples_per_class: int
) -> tuple[pd.DataFrame, tuple[FewShotExample, ...]]:
    """Steps 2-4: load IMDb, draw the fixed subset, draw the fixed demonstrations.

    Demonstrations come from the *train* split and exclude every evaluation
    sample, so no graded review can appear inside a prompt.
    """
    logger.info("Loading IMDb test split")
    evaluation_pool = prepare_dataset(load_imdb_dataset("test"), split="test")
    evaluation_set = create_fixed_evaluation_set(
        evaluation_pool,
        sample_size=config.benchmark_sample_size,
        seed=config.random_seed,
    )

    logger.info("Loading IMDb train split for demonstrations")
    demonstration_pool = prepare_dataset(load_imdb_dataset("train"), split="train")
    examples = tuple(
        create_few_shot_examples(
            demonstration_pool,
            examples_per_class=examples_per_class,
            seed=config.random_seed,
            exclude_sample_ids=evaluation_set["sample_id"],
        )
    )
    return evaluation_set, examples


def confirm_run(calls: int, *, assume_yes: bool, stream: Any = sys.stdin) -> bool:
    """Ask before a large run. Returns whether to proceed.

    Skipped for small runs and whenever ``--yes`` is given. A non-interactive
    session without ``--yes`` refuses rather than blocking on input forever.
    """
    if assume_yes or calls <= CONFIRMATION_THRESHOLD:
        return True
    if not hasattr(stream, "isatty") or not stream.isatty():
        print(
            f"This run makes {calls:,} API calls. Re-run with --yes to confirm "
            "in a non-interactive session.",
            file=sys.stderr,
        )
        return False
    answer = input(f"This run makes {calls:,} API calls. Continue? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def print_summary(
    result: ExperimentResult,
    evaluation: ExperimentEvaluation,
    chart_paths: dict[str, Path],
    *,
    saved: bool,
) -> None:
    """Step 11: print a concise, honest summary of what just happened."""
    summary = evaluation.summary
    totals = summary.get("totals", {})
    metrics = evaluation.metrics

    print()
    print("=" * 76)
    print(f"  Experiment {result.experiment_id}")
    print("=" * 76)
    print(f"  Model        {summary.get('model')}   temperature {summary.get('temperature')}")
    dataset = summary.get("dataset", {})
    print(
        f"  Dataset      {dataset.get('name')} · {dataset.get('sample_count')} samples "
        f"· seed {dataset.get('random_seed')} · checksum {dataset.get('sample_id_checksum')}"
    )
    print(
        f"  Totals       {totals.get('predictions', 0):,} predictions · "
        f"{totals.get('correct', 0):,} correct · {totals.get('unknown', 0):,} unparseable "
        f"· {totals.get('api_failures', 0):,} API failures"
    )
    print(f"  Runtime      {result.runtime_seconds:.1f}s")
    print("-" * 76)

    header = f"  {'strategy':<13}{'accuracy':>10}{'F1':>8}{'errors':>9}{'unknown':>9}{'latency':>10}"
    print(header)
    for _, row in metrics.iterrows():
        latency = row.get("avg_latency_seconds")
        print(
            f"  {str(row['strategy']):<13}"
            f"{row['accuracy']:>9.1%}"
            f"{row['f1_macro']:>8.3f}"
            f"{row['error_rate']:>9.1%}"
            f"{row['unknown_rate']:>9.1%}"
            f"{('—' if pd.isna(latency) or latency is None else f'{latency:.2f}s'):>10}"
        )
    print("-" * 76)

    best = summary.get("best_strategy")
    if best:
        print(
            f"  Best by F1   {best['strategy']} "
            f"(F1 {best['f1_macro']:.3f}, accuracy {best['accuracy']:.1%}, "
            f"unparseable {best['unknown_rate']:.1%})"
        )
    else:
        print(f"  Best by F1   {summary.get('best_strategy_note')}")
    print(
        "  Note         F1 alone can flatter a strategy that abstains: an "
        "unparseable answer\n"
        "               costs recall but never precision. Read F1, accuracy and "
        "unknown together."
    )

    if saved and result.experiment_dir:
        print("-" * 76)
        print(f"  Saved to     {result.experiment_dir}")
        for name in ("config.json", "predictions.csv", "metrics.csv", "summary.json"):
            print(f"               {name}")
        for path in chart_paths.values():
            print(f"               charts/{path.name}")
    else:
        print("-" * 76)
        print("  Saved to     nothing written (--no-save)")
    print("=" * 76)
    print()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_experiment(args: argparse.Namespace) -> int:
    """Execute one experiment end to end. Returns a process exit code."""
    # 1. Configuration.
    config = resolve_config(args)
    persist = not args.no_save

    if not has_api_key():
        print(
            "No usable GEMINI_API_KEY. Copy .env.example to .env and add your key "
            "from https://aistudio.google.com/apikey",
            file=sys.stderr,
        )
        return EXIT_CONFIG

    strategies = list(config.prompt_strategies)
    calls = len(strategies) * config.benchmark_sample_size
    if not args.quiet:
        print(
            f"{args.command} run · {len(strategies)} strategies × "
            f"{config.benchmark_sample_size} samples = {calls:,} calls · "
            f"model {config.model} · seed {config.random_seed}"
        )
    if not confirm_run(calls, assume_yes=args.yes):
        print("Aborted.", file=sys.stderr)
        return EXIT_FAILURE

    # 2-4. Data.
    evaluation_set, examples = prepare_data(
        config, examples_per_class=args.examples_per_class
    )

    # 5. Provider and storage.
    provider = GeminiProvider(config)
    storage = ExperimentStorage(config.experiments_dir)
    runner = BenchmarkRunner(provider, config, storage=storage)

    # 6-7. Run every strategy over the identical samples; predictions are saved.
    result = runner.run(
        evaluation_set,
        strategies,
        examples,
        experiment_id=args.experiment_id,
        persist=persist,
        show_progress=not args.quiet,
    )

    # 8-9. Score and save metrics.csv + summary.json.
    evaluation = runner.evaluate(result, persist=persist)

    # 10. Charts.
    chart_paths: dict[str, Path] = {}
    if not args.no_charts and persist and result.experiment_dir:
        chart_paths = render_all_charts(
            evaluation.metrics,
            result.predictions,
            output_dir=result.experiment_dir / "charts",
        )

    # 11. Summary.
    print_summary(result, evaluation, chart_paths, saved=persist)
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than raising."""
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else (
            logging.WARNING if args.quiet else logging.INFO
        ),
        format="%(levelname)-8s %(name)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    try:
        return run_experiment(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except LLMAuthError as error:
        print(f"Authentication failed: {error}", file=sys.stderr)
        return EXIT_CONFIG
    except (ConfigurationError, DatasetError, StorageError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
