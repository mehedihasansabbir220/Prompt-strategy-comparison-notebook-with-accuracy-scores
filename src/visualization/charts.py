"""Presentation-ready charts for benchmark results.

Every chart is built from a stored experiment — ``metrics.csv`` for the
comparisons and ``predictions.csv`` for the confusion matrices. Nothing is
hardcoded and nothing is estimated: if a figure is not in the data, it does not
appear on the chart.

Design rules applied throughout (they are what makes the set read as one system):

* **Consistent order and naming.** Strategies keep their execution order — the
  configured order, baseline first — in every chart, and are labelled by one
  shared formatter, so the reader can scan across figures.
* **Colour by job.** Single-measure charts use one hue; multi-series charts draw
  from a fixed categorical order; the confusion matrix uses one sequential ramp
  light-to-dark. Hues are never cycled or reassigned by rank.
* **Values are printed on the marks.** Three of the palette slots sit below the
  3:1 contrast floor on a light surface, so direct labels are not decoration
  here — they are the accessibility fallback that keeps the chart readable
  without colour.
* **Recessive chrome.** Hairline solid gridlines one shade off the surface, no
  dashes, no chartjunk, generous padding.

No LLM is contacted and no metric is computed here; this module only draws.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from src.config import LABEL_UNKNOWN, SENTIMENT_LABELS
from src.evaluation.metrics import confusion_matrix_frame

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

SURFACE: Final[str] = "#fcfcfb"
TEXT_PRIMARY: Final[str] = "#0b0b0b"
TEXT_SECONDARY: Final[str] = "#52514e"
TEXT_MUTED: Final[str] = "#7a7973"
GRID: Final[str] = "#e8e7e3"

#: Fixed categorical order. Validated for adjacent-pair colour-vision
#: separation on this surface (worst adjacent CVD dE 9.2, normal-vision 27.6).
#: Assigned in order and never cycled: a fourth series would be folded or
#: faceted rather than given a generated hue.
CATEGORICAL: Final[tuple[str, ...]] = ("#2a78d6", "#eb6834", "#1baf7a")

#: Single hue for one-measure charts.
PRIMARY: Final[str] = "#2a78d6"

#: Emphasis pair: the highlighted mark keeps the hue, the rest recede. Used
#: only to draw the eye to the best strategy, never to encode a value.
MUTED_MARK: Final[str] = "#c9cdd3"

#: Sequential blue ramp, light to dark, for magnitude in the confusion matrix.
SEQUENTIAL_STEPS: Final[tuple[str, ...]] = (
    "#f4f8fe",
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)

SEQUENTIAL_CMAP: Final[LinearSegmentedColormap] = LinearSegmentedColormap.from_list(
    "promptbench_blues", list(SEQUENTIAL_STEPS)
)

DEFAULT_DPI: Final[int] = 200


class ChartError(ValueError):
    """Raised when a chart cannot be drawn from the data supplied."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def format_strategy_name(name: str) -> str:
    """Render a strategy identifier for display, consistently everywhere.

    ``zero_shot`` becomes ``Zero-shot``. One formatter for all charts so the
    same strategy never appears under two different labels.
    """
    return str(name).replace("_", "-").capitalize()


def _require(frame: pd.DataFrame, columns: Sequence[str], *, what: str) -> None:
    """Fail loudly rather than draw a chart with silently missing series."""
    if frame is None or frame.empty:
        raise ChartError(f"Cannot draw {what}: no data")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ChartError(
            f"Cannot draw {what}: missing column(s) {missing}. "
            f"Found: {list(frame.columns)}"
        )


def _new_figure(rows: int, *, width: float = 9.0, row_height: float = 0.55) -> tuple[Figure, Any]:
    """Create a figure whose height grows with the number of strategies."""
    height = max(3.2, 2.0 + rows * row_height)
    figure, axes = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    return figure, axes


def _style_value_axis(axes: Any, *, axis: str = "x") -> None:
    """Apply recessive chrome: hairline solid grid on the value axis only."""
    axes.grid(axis=axis, color=GRID, linewidth=0.8, linestyle="-", zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=9)


def _title(axes: Any, title: str, subtitle: str | None = None) -> None:
    """Set a plain-language title, with the reading of the chart underneath."""
    axes.set_title(title, fontsize=13, fontweight="bold", color=TEXT_PRIMARY, loc="left", pad=18)
    if subtitle:
        axes.text(
            0.0, 1.02, subtitle, transform=axes.transAxes,
            fontsize=9.5, color=TEXT_SECONDARY, ha="left", va="bottom",
        )


def save_figure(figure: Figure, output_path: Path | str, *, dpi: int = DEFAULT_DPI) -> Path:
    """Write a figure to PNG, creating parent directories as needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path, dpi=dpi, bbox_inches="tight", facecolor=figure.get_facecolor()
    )
    logger.info("Saved chart %s", path)
    return path


def _finish(figure: Figure, output_path: Path | str | None) -> Figure:
    """Tighten layout and optionally persist."""
    figure.tight_layout()
    if output_path is not None:
        save_figure(figure, output_path)
    return figure


# ---------------------------------------------------------------------------
# 1. Accuracy comparison
# ---------------------------------------------------------------------------


def plot_accuracy_comparison(
    metrics: pd.DataFrame,
    *,
    metric: str = "accuracy",
    output_path: Path | str | None = None,
    highlight_best: bool = True,
) -> Figure:
    """Horizontal bars comparing one headline measure across strategies.

    Horizontal because strategy names are long; a single measure, so one hue
    and no legend — the title names the series.

    Args:
        metrics: The ``metrics.csv`` table.
        metric: Column to plot. Defaults to ``accuracy``.
        output_path: Where to write the PNG. ``None`` returns the figure only.
        highlight_best: Draw the leading bar in the accent hue and let the rest
            recede. Emphasis only — the bar's *length* still carries the value.
    """
    _require(metrics, ["strategy", metric], what="accuracy comparison")

    frame = metrics.dropna(subset=[metric])
    if frame.empty:
        raise ChartError(f"Cannot draw accuracy comparison: {metric!r} has no values")

    labels = [format_strategy_name(name) for name in frame["strategy"]]
    values = frame[metric].astype(float).tolist()
    best = max(values) if highlight_best else None
    colors = [
        PRIMARY if (best is not None and value == best) else MUTED_MARK
        for value in values
    ]
    if not highlight_best:
        colors = [PRIMARY] * len(values)

    figure, axes = _new_figure(len(frame))
    positions = range(len(frame))
    axes.barh(
        list(positions), values, color=colors, height=0.62,
        edgecolor=SURFACE, linewidth=1.5, zorder=2,
    )

    axes.set_yticks(list(positions))
    axes.set_yticklabels(labels, fontsize=10, color=TEXT_PRIMARY)
    axes.invert_yaxis()  # keep execution order reading top-to-bottom
    axes.set_xlim(0, 1.0)
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    axes.set_xlabel(metric.replace("_", " ").title(), fontsize=10, color=TEXT_SECONDARY)
    _style_value_axis(axes)

    for index, value in zip(positions, values, strict=True):
        axes.text(
            value + 0.012, index, f"{value:.1%}",
            va="center", ha="left", fontsize=9.5, color=TEXT_PRIMARY,
        )

    _title(
        axes,
        f"{metric.replace('_', ' ').title()} by prompt strategy",
        f"Same model, same {int(frame['total_samples'].iloc[0])} reviews, same settings — only the prompt changes"
        if "total_samples" in frame.columns
        else None,
    )
    return _finish(figure, output_path)


# ---------------------------------------------------------------------------
# 2. Precision / recall / F1
# ---------------------------------------------------------------------------


def plot_metric_comparison(
    metrics: pd.DataFrame,
    *,
    columns: Sequence[str] = ("precision_macro", "recall_macro", "f1_macro"),
    output_path: Path | str | None = None,
) -> Figure:
    """Grouped bars comparing precision, recall and F1 across strategies.

    Three series, so three categorical hues assigned in fixed order, a legend,
    and a value printed on every bar.
    """
    _require(metrics, ["strategy", *columns], what="metric comparison")
    if len(columns) > len(CATEGORICAL):
        raise ChartError(
            f"At most {len(CATEGORICAL)} series can be drawn together; got {len(columns)}"
        )

    labels = [format_strategy_name(name) for name in metrics["strategy"]]
    group_count = len(columns)
    bar_height = 0.78 / group_count

    figure, axes = _new_figure(len(metrics), row_height=0.78)
    for series_index, column in enumerate(columns):
        offsets = [
            index + (series_index - (group_count - 1) / 2) * bar_height
            for index in range(len(metrics))
        ]
        values = metrics[column].astype(float).tolist()
        axes.barh(
            offsets, values, height=bar_height * 0.88,
            color=CATEGORICAL[series_index],
            label=column.replace("_macro", "").title(),
            edgecolor=SURFACE, linewidth=1.2, zorder=2,
        )
        for offset, value in zip(offsets, values, strict=True):
            axes.text(
                value + 0.01, offset, f"{value:.2f}",
                va="center", ha="left", fontsize=8, color=TEXT_SECONDARY,
            )

    axes.set_yticks(range(len(metrics)))
    axes.set_yticklabels(labels, fontsize=10, color=TEXT_PRIMARY)
    axes.invert_yaxis()
    axes.set_xlim(0, 1.08)
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes.set_xlabel("Macro-averaged score", fontsize=10, color=TEXT_SECONDARY)
    _style_value_axis(axes)

    # Below the plot, not above it: an upper-right legend collides with the
    # subtitle as soon as the subtitle is long enough to be useful.
    legend = axes.legend(
        loc="upper center", frameon=False, fontsize=9, ncol=group_count,
        bbox_to_anchor=(0.5, -0.09),
    )
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    _title(
        axes,
        "Precision, recall and F1 by prompt strategy",
        "Macro-averaged over both classes; unparseable answers cost recall, not precision",
    )
    return _finish(figure, output_path)


# ---------------------------------------------------------------------------
# 3. Error rate
# ---------------------------------------------------------------------------


def plot_error_rate_comparison(
    metrics: pd.DataFrame, *, output_path: Path | str | None = None
) -> Figure:
    """Stacked bars splitting each strategy's error rate into its two causes.

    A single error-rate bar hides the distinction that matters most here: a
    wrong label and an unreadable answer are different failures with different
    fixes. The segments sum to the error rate, and the total is printed.
    """
    _require(metrics, ["strategy", "error_rate", "unknown_rate"], what="error rate comparison")

    unknown = metrics["unknown_rate"].astype(float)
    total = metrics["error_rate"].astype(float)
    # Derived, not assumed: everything wrong that was still readable.
    misclassified = (total - unknown).clip(lower=0.0)

    labels = [format_strategy_name(name) for name in metrics["strategy"]]
    positions = list(range(len(metrics)))

    figure, axes = _new_figure(len(metrics))
    axes.barh(
        positions, misclassified, height=0.62, color=CATEGORICAL[0],
        label="Misclassified", edgecolor=SURFACE, linewidth=1.5, zorder=2,
    )
    axes.barh(
        positions, unknown, left=misclassified, height=0.62, color=CATEGORICAL[1],
        label="Unparseable", edgecolor=SURFACE, linewidth=1.5, zorder=2,
    )

    for index, value in zip(positions, total, strict=True):
        axes.text(
            value + 0.008, index, f"{value:.1%}",
            va="center", ha="left", fontsize=9.5, color=TEXT_PRIMARY,
        )

    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=10, color=TEXT_PRIMARY)
    axes.invert_yaxis()
    axes.set_xlim(0, max(float(total.max()) * 1.25, 0.05))
    axes.set_xlabel("Share of samples", fontsize=10, color=TEXT_SECONDARY)
    axes.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    _style_value_axis(axes)

    legend = axes.legend(
        loc="upper center", frameon=False, fontsize=9, ncol=2,
        bbox_to_anchor=(0.5, -0.09),
    )
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    _title(
        axes,
        "Error rate by prompt strategy",
        "Split by cause: a wrong label, or an answer that could not be parsed at all",
    )
    return _finish(figure, output_path)


# ---------------------------------------------------------------------------
# 4. Latency
# ---------------------------------------------------------------------------


def plot_latency_comparison(
    metrics: pd.DataFrame,
    *,
    column: str = "avg_latency_seconds",
    output_path: Path | str | None = None,
) -> Figure:
    """Horizontal bars comparing mean response latency per call.

    Cost, not quality: a strategy that wins by a point but takes twice as long
    is a different trade-off, and the reader should be able to see it.
    """
    _require(metrics, ["strategy", column], what="latency comparison")

    frame = metrics.dropna(subset=[column])
    if frame.empty:
        raise ChartError(
            f"Cannot draw latency comparison: {column!r} was not recorded in this run"
        )

    labels = [format_strategy_name(name) for name in frame["strategy"]]
    values = frame[column].astype(float).tolist()
    positions = list(range(len(frame)))

    figure, axes = _new_figure(len(frame))
    axes.barh(
        positions, values, height=0.62, color=PRIMARY,
        edgecolor=SURFACE, linewidth=1.5, zorder=2,
    )
    headroom = max(values) * 0.16 if max(values) else 0.1
    for index, value in zip(positions, values, strict=True):
        axes.text(
            value + headroom * 0.15, index, f"{value:.2f}s",
            va="center", ha="left", fontsize=9.5, color=TEXT_PRIMARY,
        )

    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=10, color=TEXT_PRIMARY)
    axes.invert_yaxis()
    axes.set_xlim(0, max(values) + headroom)
    axes.set_xlabel("Mean latency per call (seconds)", fontsize=10, color=TEXT_SECONDARY)
    _style_value_axis(axes)

    _title(
        axes,
        "Response latency by prompt strategy",
        "Mean wall-clock time per API call, including any retries",
    )
    return _finish(figure, output_path)


# ---------------------------------------------------------------------------
# 5. Confusion matrices
# ---------------------------------------------------------------------------


def _draw_matrix(axes: Any, matrix: pd.DataFrame, title: str, *, vmax: int) -> None:
    """Render one confusion matrix onto an axis with a shared colour scale."""
    sns.heatmap(
        matrix, ax=axes, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=vmax,
        annot=True, fmt="d", annot_kws={"fontsize": 10},
        linewidths=2, linecolor=SURFACE, cbar=False, square=False,
    )
    axes.set_title(title, fontsize=11, fontweight="bold", color=TEXT_PRIMARY, pad=8)
    axes.set_xlabel("Predicted", fontsize=9, color=TEXT_SECONDARY)
    axes.set_ylabel("Actual", fontsize=9, color=TEXT_SECONDARY)
    axes.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=9)
    axes.set_yticklabels(axes.get_yticklabels(), rotation=0)


def plot_confusion_matrix(
    predictions: pd.DataFrame,
    strategy: str,
    *,
    output_path: Path | str | None = None,
) -> Figure:
    """Confusion matrix for one strategy, including the ``unknown`` column.

    The unknown column is always present, even when empty, so matrices from
    different strategies have identical shape and can be read side by side.
    """
    _require(predictions, ["strategy", "actual_label", "predicted_label"], what="confusion matrix")

    rows = predictions[predictions["strategy"] == strategy]
    if rows.empty:
        raise ChartError(f"No predictions found for strategy {strategy!r}")

    matrix = confusion_matrix_frame(rows["actual_label"], rows["predicted_label"])
    figure, axes = plt.subplots(figsize=(5.4, 3.4))
    figure.patch.set_facecolor(SURFACE)
    _draw_matrix(axes, matrix, f"{format_strategy_name(strategy)} — confusion matrix",
                 vmax=int(matrix.to_numpy().max()) or 1)
    return _finish(figure, output_path)


def plot_confusion_matrices(
    predictions: pd.DataFrame,
    *,
    strategies: Sequence[str] | None = None,
    columns: int = 3,
    output_path: Path | str | None = None,
) -> Figure:
    """Small multiples: one confusion matrix per strategy on a shared scale.

    A shared colour scale is what makes the panels comparable — per-panel
    scaling would make an unbalanced strategy look identical to a balanced one.
    """
    _require(predictions, ["strategy", "actual_label", "predicted_label"], what="confusion matrices")

    names = list(strategies or predictions["strategy"].drop_duplicates())
    matrices = {}
    for name in names:
        rows = predictions[predictions["strategy"] == name]
        if rows.empty:
            raise ChartError(f"No predictions found for strategy {name!r}")
        matrices[name] = confusion_matrix_frame(
            rows["actual_label"], rows["predicted_label"]
        )

    shared_max = max(int(matrix.to_numpy().max()) for matrix in matrices.values()) or 1
    column_count = min(columns, len(names))
    row_count = -(-len(names) // column_count)  # ceiling division

    # Reserve a fixed header band in inches, then position the two header lines
    # inside it. Fractional y-coordinates alone drift with the grid height and
    # make the title and subtitle collide at some row counts.
    header_inches = 0.95
    panel_height = 3.1 * row_count
    figure_height = panel_height + header_inches
    figure, axes_grid = plt.subplots(
        row_count, column_count,
        figsize=(4.6 * column_count, figure_height),
        squeeze=False,
    )
    figure.patch.set_facecolor(SURFACE)

    for index, name in enumerate(names):
        axes = axes_grid[index // column_count][index % column_count]
        _draw_matrix(axes, matrices[name], format_strategy_name(name), vmax=shared_max)

    for empty_index in range(len(names), row_count * column_count):
        axes_grid[empty_index // column_count][empty_index % column_count].axis("off")

    figure.suptitle(
        "Confusion matrices by prompt strategy",
        fontsize=13, fontweight="bold", color=TEXT_PRIMARY,
        x=0.01, y=1 - 0.30 / figure_height, ha="left",
    )
    figure.text(
        0.01, 1 - 0.62 / figure_height,
        f"Rows are the true label, columns the prediction. "
        f"'{LABEL_UNKNOWN}' means the response could not be parsed. Shared colour scale.",
        fontsize=9.5, color=TEXT_SECONDARY, ha="left", va="top",
    )
    figure.tight_layout(rect=(0, 0, 1, 1 - header_inches / figure_height))
    if output_path is not None:
        save_figure(figure, output_path)
    return figure


# ---------------------------------------------------------------------------
# Rendering a whole experiment
# ---------------------------------------------------------------------------

#: Filenames written by :func:`render_all_charts`.
CHART_FILENAMES: Final[dict[str, str]] = {
    "accuracy": "accuracy_comparison.png",
    "metrics": "precision_recall_f1.png",
    "error_rate": "error_rate_comparison.png",
    "latency": "latency_comparison.png",
    "confusion": "confusion_matrices.png",
}


def render_all_charts(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame | None = None,
    *,
    output_dir: Path | str,
    close_figures: bool = True,
) -> dict[str, Path]:
    """Draw and save the full chart set for one experiment.

    A chart whose input is absent is skipped with a logged reason rather than
    drawn from placeholder values — an empty panel is honest, an invented one
    is not.

    Args:
        metrics: The experiment's ``metrics.csv``.
        predictions: The experiment's ``predictions.csv``. Required only for
            the confusion matrices.
        output_dir: Directory to write PNGs into; created if absent.
        close_figures: Release figures after saving. Leave ``True`` for batch
            rendering; set ``False`` in a notebook to keep them displayable.

    Returns:
        Mapping of chart key to the PNG path written.
    """
    directory = Path(output_dir)
    written: dict[str, Path] = {}

    jobs: list[tuple[str, Any]] = [
        ("accuracy", lambda: plot_accuracy_comparison(metrics)),
        ("metrics", lambda: plot_metric_comparison(metrics)),
        ("error_rate", lambda: plot_error_rate_comparison(metrics)),
        ("latency", lambda: plot_latency_comparison(metrics)),
    ]
    if predictions is not None and not predictions.empty:
        jobs.append(("confusion", lambda: plot_confusion_matrices(predictions)))

    for key, build in jobs:
        try:
            figure = build()
        except ChartError as exc:
            logger.warning("Skipping %s chart: %s", key, exc)
            continue
        written[key] = save_figure(figure, directory / CHART_FILENAMES[key])
        if close_figures:
            plt.close(figure)

    logger.info("Rendered %d chart(s) into %s", len(written), directory)
    return written


def render_experiment_charts(
    experiment_id: str,
    storage: Any,
    *,
    output_dir: Path | str | None = None,
    close_figures: bool = True,
) -> dict[str, Path]:
    """Render charts straight from a stored experiment.

    Reads ``metrics.csv`` and ``predictions.csv`` back from disk, so the charts
    are drawn from exactly the artefacts a reader of the repository would see —
    never from in-memory state that might differ.

    Args:
        experiment_id: The run to chart.
        storage: An :class:`~src.experiments.storage.ExperimentStorage`.
        output_dir: Defaults to ``<experiment>/charts/``.
    """
    metrics = storage.load_metrics(experiment_id)
    try:
        predictions = storage.load_predictions(experiment_id)
    except Exception as exc:  # noqa: BLE001 - predictions are optional here
        logger.warning("No predictions for %s (%s); skipping confusion matrices",
                       experiment_id, exc)
        predictions = None

    target = Path(output_dir) if output_dir else (
        storage.experiment_path(experiment_id) / "charts"
    )
    return render_all_charts(
        metrics, predictions, output_dir=target, close_figures=close_figures
    )
