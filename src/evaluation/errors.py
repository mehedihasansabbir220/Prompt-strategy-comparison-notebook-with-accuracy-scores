"""Error extraction and cross-strategy error comparison.

Every incorrect prediction is preserved with the full context needed to judge
it later: which sample, which strategy, the review itself, both labels, and the
raw model response that produced the prediction. Keeping the raw response is
what makes the analysis verifiable — a claim about *why* a strategy failed can
always be checked against what the model actually said.

An ``unknown`` prediction is an error too, of a different kind, and is labelled
as such rather than being merged with genuine misclassifications.

No LLM is contacted here.
"""

from __future__ import annotations

import logging
from typing import Any, Final

import pandas as pd

from src.config import LABEL_UNKNOWN

logger = logging.getLogger(__name__)

#: Columns required on an input predictions frame.
REQUIRED_PREDICTION_COLUMNS: Final[tuple[str, ...]] = (
    "sample_id",
    "strategy",
    "review",
    "actual_label",
    "predicted_label",
    "raw_response",
)

#: Column order of ``errors.csv``. The six required fields come first;
#: ``error_type`` is derived, and separates the two failure modes.
ERROR_COLUMNS: Final[tuple[str, ...]] = (*REQUIRED_PREDICTION_COLUMNS, "error_type")

#: The model produced a readable label, and it was the wrong one.
ERROR_MISCLASSIFICATION: Final[str] = "misclassification"

#: The model's response could not be parsed into a label at all. A format or
#: reliability failure, not a judgement failure — counted separately because the
#: two call for completely different fixes.
ERROR_UNPARSEABLE: Final[str] = "unparseable"


class ErrorAnalysisError(ValueError):
    """Raised when a predictions frame cannot be analysed as given."""


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ErrorAnalysisError(
            f"Predictions frame is missing column(s) {missing}. "
            f"Found: {list(frame.columns)}"
        )


def classify_error(actual: str, predicted: str) -> str | None:
    """Return the error type for one prediction, or ``None`` if it was correct."""
    if predicted == actual:
        return None
    return ERROR_UNPARSEABLE if predicted == LABEL_UNKNOWN else ERROR_MISCLASSIFICATION


def extract_errors(
    predictions: pd.DataFrame, *, strategy: str | None = None
) -> pd.DataFrame:
    """Return every row where the prediction does not match the ground truth.

    Args:
        predictions: Frame containing at least :data:`REQUIRED_PREDICTION_COLUMNS`.
        strategy: Optional filter, to analyse a single strategy in isolation.

    Returns:
        A frame with :data:`ERROR_COLUMNS`, ordered by strategy then sample id
        so two runs produce diffable files. Empty (with the correct columns) when
        a strategy made no errors — an empty result is a real finding, not a
        missing one.

    Raises:
        ErrorAnalysisError: If required columns are absent.
    """
    _require_columns(predictions, REQUIRED_PREDICTION_COLUMNS)

    working = predictions
    if strategy is not None:
        working = working[working["strategy"] == strategy]

    if working.empty:
        return pd.DataFrame(columns=list(ERROR_COLUMNS))

    incorrect = working[working["actual_label"] != working["predicted_label"]].copy()
    if incorrect.empty:
        return pd.DataFrame(columns=list(ERROR_COLUMNS))

    incorrect["error_type"] = [
        classify_error(actual, predicted)
        for actual, predicted in zip(
            incorrect["actual_label"], incorrect["predicted_label"], strict=True
        )
    ]

    errors = (
        incorrect[list(ERROR_COLUMNS)]
        .sort_values(["strategy", "sample_id"])
        .reset_index(drop=True)
    )
    logger.info(
        "Extracted %d error(s)%s: %s",
        len(errors),
        f" for strategy {strategy!r}" if strategy else "",
        errors["error_type"].value_counts().to_dict(),
    )
    return errors


def summarize_errors_by_strategy(errors: pd.DataFrame) -> pd.DataFrame:
    """Count each failure mode per strategy.

    Answers "does this strategy fail by misjudging, or by being unreadable?",
    which the headline accuracy alone cannot distinguish.

    Returns:
        One row per strategy with ``misclassification``, ``unparseable`` and
        ``total_errors`` columns.
    """
    columns = ["strategy", ERROR_MISCLASSIFICATION, ERROR_UNPARSEABLE, "total_errors"]
    if errors.empty:
        return pd.DataFrame(columns=columns)

    _require_columns(errors, ("strategy", "error_type"))
    counts = (
        errors.groupby(["strategy", "error_type"]).size().unstack(fill_value=0)
    )
    for error_type in (ERROR_MISCLASSIFICATION, ERROR_UNPARSEABLE):
        if error_type not in counts.columns:
            counts[error_type] = 0

    counts["total_errors"] = (
        counts[ERROR_MISCLASSIFICATION] + counts[ERROR_UNPARSEABLE]
    )
    return counts.reset_index()[columns]


def find_shared_errors(
    errors: pd.DataFrame, *, strategy_count: int | None = None
) -> pd.DataFrame:
    """Find samples that *every* strategy got wrong.

    These are the most informative rows in the whole experiment: a sample no
    prompt design rescues is evidence about the *data* — genuine ambiguity or a
    questionable gold label — rather than about any strategy.

    Args:
        errors: Combined error frame across strategies.
        strategy_count: How many strategies ran. Defaults to the number of
            distinct strategies present in ``errors``, which is only correct
            when every strategy made at least one error; pass it explicitly to
            be exact.

    Returns:
        One row per shared sample with the review, its true label, and the count
        of strategies that failed it.
    """
    columns = ["sample_id", "review", "actual_label", "failed_strategies"]
    if errors.empty:
        return pd.DataFrame(columns=columns)

    _require_columns(errors, ("sample_id", "strategy", "review", "actual_label"))
    total_strategies = strategy_count or errors["strategy"].nunique()

    grouped = (
        errors.groupby("sample_id")
        .agg(
            review=("review", "first"),
            actual_label=("actual_label", "first"),
            failed_strategies=("strategy", "nunique"),
        )
        .reset_index()
    )
    shared = grouped[grouped["failed_strategies"] >= total_strategies]
    return shared.sort_values("sample_id").reset_index(drop=True)[columns]


def unique_errors(errors: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Find samples that only ``strategy`` got wrong.

    The counterpart to :func:`find_shared_errors`: these rows isolate what a
    particular prompt design uniquely breaks.
    """
    if errors.empty:
        return pd.DataFrame(columns=list(ERROR_COLUMNS))

    _require_columns(errors, ("sample_id", "strategy"))
    failure_counts = errors.groupby("sample_id")["strategy"].nunique()
    only_once = set(failure_counts[failure_counts == 1].index)
    mask = (errors["strategy"] == strategy) & errors["sample_id"].isin(only_once)
    return errors[mask].reset_index(drop=True)


def error_analysis_summary(errors: pd.DataFrame) -> dict[str, Any]:
    """Return headline error-analysis figures for the notebook and ``summary.json``."""
    if errors.empty:
        return {
            "total_errors": 0,
            "misclassifications": 0,
            "unparseable": 0,
            "strategies_with_errors": 0,
            "samples_with_errors": 0,
        }

    _require_columns(errors, ("error_type", "strategy", "sample_id"))
    counts = errors["error_type"].value_counts()
    return {
        "total_errors": int(len(errors)),
        "misclassifications": int(counts.get(ERROR_MISCLASSIFICATION, 0)),
        "unparseable": int(counts.get(ERROR_UNPARSEABLE, 0)),
        "strategies_with_errors": int(errors["strategy"].nunique()),
        "samples_with_errors": int(errors["sample_id"].nunique()),
    }
