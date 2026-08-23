"""Classification metrics for one strategy's predictions.

Every number here comes from real predictions; nothing is estimated or filled
in. The module's central concern is the honest treatment of ``unknown``.

Policy for unknown predictions
------------------------------
An ``unknown`` prediction means the model's response could not be parsed into a
label. It is **never silently dropped**. It is treated as follows, consistently
everywhere:

* **Accuracy** counts it as *incorrect*. A response nobody can read is not a
  correct classification, so the headline ``accuracy`` is the share of samples
  the strategy actually got right out of everything it was asked.
* **Precision / recall / F1** are computed over the two real classes only. An
  unknown is therefore "no prediction was made": it costs recall on the true
  class but does not pollute precision on either. This is the standard
  treatment for an abstention and is why the two accuracy figures can differ.
* **``accuracy_on_resolved``** is reported alongside as a secondary figure: the
  accuracy over only those samples that produced a readable label. Comparing it
  with ``accuracy`` separates *judgement* quality from *format* reliability.
* **``unknown_count`` and ``unknown_rate``** are reported as first-class
  metrics, because format compliance is one of the things this benchmark
  measures.

No LLM is contacted here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import LABEL_UNKNOWN, SENTIMENT_LABELS

logger = logging.getLogger(__name__)

#: Column order of the per-strategy metrics table (``metrics.csv``).
METRICS_COLUMNS: tuple[str, ...] = (
    "strategy",
    "total_samples",
    "correct",
    "incorrect",
    "unknown",
    "resolved",
    "accuracy",
    "accuracy_on_resolved",
    "error_rate",
    "unknown_rate",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_negative",
    "recall_negative",
    "f1_negative",
    "support_negative",
    "precision_positive",
    "recall_positive",
    "f1_positive",
    "support_positive",
    "api_failures",
    "runtime_seconds",
)


class MetricsError(ValueError):
    """Raised when predictions cannot be scored as given."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _as_list(values: Sequence[str] | pd.Series) -> list[str]:
    """Normalise any accepted input container to a plain list of strings."""
    if isinstance(values, pd.Series):
        return [str(value) for value in values.tolist()]
    return [str(value) for value in values]


def validate_predictions(
    y_true: Sequence[str] | pd.Series, y_pred: Sequence[str] | pd.Series
) -> tuple[list[str], list[str]]:
    """Check that two label sequences can be scored against each other.

    Ground truth must contain only real classes: an ``unknown`` in ``y_true``
    would mean the dataset itself is unlabelled, which is a data bug rather
    than a model failure.

    Raises:
        MetricsError: On empty input, length mismatch, or an unrecognised label.
    """
    actual = _as_list(y_true)
    predicted = _as_list(y_pred)

    if not actual:
        raise MetricsError("Cannot compute metrics on an empty prediction set")
    if len(actual) != len(predicted):
        raise MetricsError(
            f"Length mismatch: {len(actual)} ground-truth labels vs "
            f"{len(predicted)} predictions"
        )

    invalid_true = sorted(set(actual) - set(SENTIMENT_LABELS))
    if invalid_true:
        raise MetricsError(
            f"Ground-truth labels must be one of {list(SENTIMENT_LABELS)}; "
            f"found {invalid_true}"
        )

    allowed_pred = set(SENTIMENT_LABELS) | {LABEL_UNKNOWN}
    invalid_pred = sorted(set(predicted) - allowed_pred)
    if invalid_pred:
        raise MetricsError(
            f"Predictions must be one of {sorted(allowed_pred)}; found {invalid_pred}"
        )

    return actual, predicted


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def compute_prediction_counts(
    y_true: Sequence[str] | pd.Series, y_pred: Sequence[str] | pd.Series
) -> dict[str, Any]:
    """Count outcomes and derive the rate metrics.

    Returns:
        ``total_samples``, ``correct``, ``incorrect``, ``unknown``, ``resolved``,
        ``accuracy``, ``accuracy_on_resolved``, ``error_rate``, ``unknown_rate``.

        ``correct + incorrect == total_samples`` always holds, and every unknown
        is inside ``incorrect`` — the two never double-count.
    """
    actual, predicted = validate_predictions(y_true, y_pred)

    total = len(actual)
    unknown = sum(1 for label in predicted if label == LABEL_UNKNOWN)
    correct = sum(1 for a, p in zip(actual, predicted, strict=True) if a == p)
    resolved = total - unknown
    correct_resolved = sum(
        1
        for a, p in zip(actual, predicted, strict=True)
        if p != LABEL_UNKNOWN and a == p
    )

    return {
        "total_samples": total,
        "correct": correct,
        "incorrect": total - correct,
        "unknown": unknown,
        "resolved": resolved,
        "accuracy": correct / total,
        # None, not 0.0: with nothing resolved the figure is undefined, and a
        # zero would read as "got everything wrong" instead of "no data".
        "accuracy_on_resolved": (correct_resolved / resolved) if resolved else None,
        "error_rate": (total - correct) / total,
        "unknown_rate": unknown / total,
    }


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: Sequence[str] | pd.Series, y_pred: Sequence[str] | pd.Series
) -> dict[str, Any]:
    """Compute per-class and macro precision, recall and F1.

    Scored over the two real classes only; see the module docstring for how
    ``unknown`` is handled. ``zero_division=0`` keeps a class that was never
    predicted at 0.0 rather than raising or emitting NaN.
    """
    actual, predicted = validate_predictions(y_true, y_pred)

    per_class = precision_recall_fscore_support(
        actual,
        predicted,
        labels=list(SENTIMENT_LABELS),
        zero_division=0,
    )
    precision, recall, f1, support = per_class

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=list(SENTIMENT_LABELS),
        average="macro",
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
    }
    for index, label in enumerate(SENTIMENT_LABELS):
        metrics[f"precision_{label}"] = float(precision[index])
        metrics[f"recall_{label}"] = float(recall[index])
        metrics[f"f1_{label}"] = float(f1[index])
        metrics[f"support_{label}"] = int(support[index])
    return metrics


def confusion_matrix_frame(
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
    *,
    include_unknown: bool = True,
) -> pd.DataFrame:
    """Build a labelled confusion matrix as a DataFrame.

    Rows are actual classes, columns are predicted. With ``include_unknown``
    the ``unknown`` column is always present — even when it is all zeros — so
    every strategy's matrix has the same shape and they can be compared and
    plotted side by side.

    Args:
        include_unknown: Add the ``unknown`` prediction column. Setting this to
            ``False`` produces the conventional 2x2 matrix and **discards**
            unknown predictions from the counts, so the totals no longer sum to
            the sample count.
    """
    actual, predicted = validate_predictions(y_true, y_pred)

    predicted_labels = list(SENTIMENT_LABELS)
    if include_unknown:
        predicted_labels.append(LABEL_UNKNOWN)

    matrix = confusion_matrix(
        actual, predicted, labels=list(SENTIMENT_LABELS) + [LABEL_UNKNOWN]
    )
    frame = pd.DataFrame(
        matrix,
        index=list(SENTIMENT_LABELS) + [LABEL_UNKNOWN],
        columns=list(SENTIMENT_LABELS) + [LABEL_UNKNOWN],
    )
    # Ground truth never contains 'unknown', so that row carries no information.
    frame = frame.loc[list(SENTIMENT_LABELS), predicted_labels]
    frame.index.name = "actual"
    frame.columns.name = "predicted"
    return frame


def classification_report_frame(
    y_true: Sequence[str] | pd.Series, y_pred: Sequence[str] | pd.Series
) -> pd.DataFrame:
    """Return scikit-learn's classification report as a tidy DataFrame.

    Restricted to the two real classes, consistent with the module's unknown
    policy. Rounded for display; the unrounded values live in
    :func:`compute_classification_metrics`.
    """
    actual, predicted = validate_predictions(y_true, y_pred)

    report = classification_report(
        actual,
        predicted,
        labels=list(SENTIMENT_LABELS),
        output_dict=True,
        zero_division=0,
    )
    frame = pd.DataFrame(report).transpose()
    if "support" in frame.columns:
        frame["support"] = frame["support"].astype(int)
    frame.index.name = "class"
    return frame.round(4)


# ---------------------------------------------------------------------------
# Aggregate view
# ---------------------------------------------------------------------------


def evaluate_predictions(
    y_true: Sequence[str] | pd.Series,
    y_pred: Sequence[str] | pd.Series,
    *,
    strategy: str | None = None,
    api_failures: int = 0,
    runtime_seconds: float | None = None,
) -> dict[str, Any]:
    """Score one strategy and return a single flat metrics record.

    This is one row of ``metrics.csv``.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels, possibly containing ``unknown``.
        strategy: Strategy name, carried through for the results table.
        api_failures: Calls that never produced a response. Recorded as an
            operational metric; the samples themselves still appear as unknown
            predictions, so counts stay consistent.
        runtime_seconds: Wall-clock time for the strategy's run.

    Returns:
        A dict whose keys follow :data:`METRICS_COLUMNS`.
    """
    record: dict[str, Any] = {"strategy": strategy}
    record.update(compute_prediction_counts(y_true, y_pred))
    record.update(compute_classification_metrics(y_true, y_pred))
    record["api_failures"] = api_failures
    record["runtime_seconds"] = runtime_seconds

    if record["unknown"]:
        logger.info(
            "Strategy %s produced %d unparseable response(s) out of %d (%.1f%%)",
            strategy,
            record["unknown"],
            record["total_samples"],
            record["unknown_rate"] * 100,
        )
    return {key: record.get(key) for key in METRICS_COLUMNS}


def build_metrics_table(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Assemble per-strategy records into the comparison table.

    Rows keep the order they were evaluated in — the configured strategy order,
    baseline first — rather than being re-sorted by score, so the table reads as
    an experiment log rather than a leaderboard.
    """
    if not records:
        return pd.DataFrame(columns=list(METRICS_COLUMNS))
    frame = pd.DataFrame(list(records))
    for column in METRICS_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[list(METRICS_COLUMNS)].reset_index(drop=True)


def best_strategy(metrics: pd.DataFrame, *, metric: str = "accuracy") -> str | None:
    """Return the highest-scoring strategy name, or ``None`` if there is a tie.

    Refusing to break a tie is deliberate: with a modest sample size two
    strategies can be genuinely indistinguishable, and silently picking one
    would overstate the result.
    """
    if metrics.empty or metric not in metrics.columns:
        return None
    scores = metrics[metric].dropna()
    if scores.empty:
        return None
    top = scores.max()
    winners = metrics.loc[metrics[metric] == top, "strategy"].tolist()
    if len(winners) != 1:
        logger.info("No single best strategy on %r: tie between %s", metric, winners)
        return None
    return str(winners[0])
