"""Tests for the visualization layer.

Charts are tested for the things that can silently go wrong: drawing from the
wrong column, inventing data that is not there, losing a strategy, or writing
no file. Pixel appearance is not asserted — that was checked by rendering and
looking at the output.

All input is mocked. The Agg backend is forced so nothing tries to open a window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.config import LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_UNKNOWN  # noqa: E402
from src.visualization.charts import (  # noqa: E402
    CATEGORICAL,
    CHART_FILENAMES,
    SEQUENTIAL_STEPS,
    ChartError,
    format_strategy_name,
    plot_accuracy_comparison,
    plot_confusion_matrices,
    plot_confusion_matrix,
    plot_error_rate_comparison,
    plot_latency_comparison,
    plot_metric_comparison,
    render_all_charts,
    render_experiment_charts,
    save_figure,
)

POS = LABEL_POSITIVE
NEG = LABEL_NEGATIVE
UNK = LABEL_UNKNOWN


@pytest.fixture(autouse=True)
def close_figures() -> Any:
    """Release every figure a test creates, so the run cannot leak memory."""
    yield
    plt.close("all")


@pytest.fixture
def metrics() -> pd.DataFrame:
    """A three-strategy metrics table with hand-chosen, distinguishable values."""
    return pd.DataFrame(
        {
            "strategy": ["zero_shot", "few_shot", "combined"],
            "total_samples": [40, 40, 40],
            "correct": [24, 36, 39],
            "incorrect": [16, 4, 1],
            "unknown": [8, 0, 0],
            "accuracy": [0.60, 0.90, 0.975],
            "precision_macro": [0.75, 0.90, 0.98],
            "recall_macro": [0.60, 0.90, 0.975],
            "f1_macro": [0.667, 0.90, 0.975],
            "error_rate": [0.40, 0.10, 0.025],
            "unknown_rate": [0.20, 0.00, 0.00],
            "avg_latency_seconds": [1.05, 1.30, 2.10],
        }
    )


@pytest.fixture
def predictions() -> pd.DataFrame:
    """Two strategies over four samples, one with an unparseable answer."""
    rows = []
    scripts = {
        "zero_shot": [POS, NEG, UNK, NEG],
        "combined": [POS, NEG, POS, NEG],
    }
    for strategy, predicted in scripts.items():
        for index, prediction in enumerate(predicted):
            rows.append(
                {
                    "strategy": strategy,
                    "sample_id": f"test-{index:05d}",
                    "actual_label": [POS, NEG, POS, NEG][index],
                    "predicted_label": prediction,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Naming and palette
# ---------------------------------------------------------------------------


class TestNaming:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("zero_shot", "Zero-shot"),
            ("few_shot", "Few-shot"),
            ("role_based", "Role-based"),
            ("structured", "Structured"),
            ("combined", "Combined"),
        ],
    )
    def test_names_are_formatted_consistently(self, raw: str, expected: str) -> None:
        assert format_strategy_name(raw) == expected

    def test_the_same_formatter_is_used_by_every_chart(
        self, metrics: pd.DataFrame, predictions: pd.DataFrame
    ) -> None:
        expected = {format_strategy_name(name) for name in metrics["strategy"]}
        for figure in (
            plot_accuracy_comparison(metrics),
            plot_metric_comparison(metrics),
            plot_error_rate_comparison(metrics),
            plot_latency_comparison(metrics),
        ):
            labels = {text.get_text() for text in figure.axes[0].get_yticklabels()}
            assert labels == expected


class TestPalette:
    def test_categorical_slots_are_distinct(self) -> None:
        assert len(set(CATEGORICAL)) == len(CATEGORICAL)

    def test_sequential_ramp_is_ordered_light_to_dark(self) -> None:
        def luminance(hex_color: str) -> int:
            return sum(int(hex_color[index : index + 2], 16) for index in (1, 3, 5))

        values = [luminance(step) for step in SEQUENTIAL_STEPS]
        assert values == sorted(values, reverse=True)


# ---------------------------------------------------------------------------
# Chart content
# ---------------------------------------------------------------------------


class TestAccuracyChart:
    def test_draws_one_bar_per_strategy(self, metrics: pd.DataFrame) -> None:
        axes = plot_accuracy_comparison(metrics).axes[0]
        assert len(axes.patches) == len(metrics)

    def test_bar_lengths_are_the_actual_values(self, metrics: pd.DataFrame) -> None:
        axes = plot_accuracy_comparison(metrics).axes[0]
        widths = sorted(patch.get_width() for patch in axes.patches)
        assert widths == pytest.approx(sorted(metrics["accuracy"].tolist()))

    def test_values_are_printed_on_the_chart(self, metrics: pd.DataFrame) -> None:
        axes = plot_accuracy_comparison(metrics).axes[0]
        printed = {text.get_text() for text in axes.texts}
        assert {"60.0%", "90.0%", "97.5%"} <= printed

    def test_title_and_axis_label_are_present(self, metrics: pd.DataFrame) -> None:
        # Titles are left-aligned, so they live under loc="left".
        axes = plot_accuracy_comparison(metrics).axes[0]
        assert "Accuracy" in axes.get_title(loc="left")
        assert axes.get_xlabel()

    def test_best_strategy_is_emphasised(self, metrics: pd.DataFrame) -> None:
        axes = plot_accuracy_comparison(metrics).axes[0]
        # Exactly one bar carries the accent hue: the highest value.
        accented = [
            patch for patch in axes.patches
            if patch.get_facecolor()[:3] != plt.matplotlib.colors.to_rgb("#c9cdd3")
        ]
        assert len(accented) == 1
        assert accented[0].get_width() == pytest.approx(metrics["accuracy"].max())

    def test_emphasis_can_be_disabled(self, metrics: pd.DataFrame) -> None:
        axes = plot_accuracy_comparison(metrics, highlight_best=False).axes[0]
        colors = {patch.get_facecolor() for patch in axes.patches}
        assert len(colors) == 1

    def test_can_plot_another_metric(self, metrics: pd.DataFrame) -> None:
        axes = plot_accuracy_comparison(metrics, metric="f1_macro").axes[0]
        assert "F1" in axes.get_title(loc="left")

    def test_missing_column_is_reported(self, metrics: pd.DataFrame) -> None:
        with pytest.raises(ChartError, match="missing column"):
            plot_accuracy_comparison(metrics.drop(columns=["accuracy"]))

    def test_empty_input_is_refused(self) -> None:
        with pytest.raises(ChartError, match="no data"):
            plot_accuracy_comparison(pd.DataFrame())


class TestMetricComparisonChart:
    def test_draws_three_series_per_strategy(self, metrics: pd.DataFrame) -> None:
        axes = plot_metric_comparison(metrics).axes[0]
        assert len(axes.patches) == 3 * len(metrics)

    def test_has_a_legend_naming_every_series(self, metrics: pd.DataFrame) -> None:
        axes = plot_metric_comparison(metrics).axes[0]
        entries = {text.get_text() for text in axes.get_legend().get_texts()}
        assert entries == {"Precision", "Recall", "F1"}

    def test_series_use_the_fixed_categorical_order(self, metrics: pd.DataFrame) -> None:
        axes = plot_metric_comparison(metrics).axes[0]
        used = [container.patches[0].get_facecolor() for container in axes.containers]
        expected = [plt.matplotlib.colors.to_rgba(color) for color in CATEGORICAL[:3]]
        assert used == expected

    def test_every_bar_is_labelled_with_its_value(self, metrics: pd.DataFrame) -> None:
        axes = plot_metric_comparison(metrics).axes[0]
        # Count only the numeric labels; the subtitle is an axes text too.
        numeric = [
            text for text in axes.texts
            if text.get_text().replace(".", "", 1).isdigit()
        ]
        assert len(numeric) == 3 * len(metrics)

    def test_refuses_more_series_than_the_palette_allows(
        self, metrics: pd.DataFrame
    ) -> None:
        frame = metrics.assign(extra_a=0.5, extra_b=0.5)
        with pytest.raises(ChartError, match="At most"):
            plot_metric_comparison(
                frame,
                columns=("precision_macro", "recall_macro", "f1_macro", "extra_a", "extra_b"),
            )


class TestErrorRateChart:
    def test_segments_sum_to_the_error_rate(self, metrics: pd.DataFrame) -> None:
        axes = plot_error_rate_comparison(metrics).axes[0]
        misclassified, unparseable = axes.containers
        for index in range(len(metrics)):
            total = (
                misclassified.patches[index].get_width()
                + unparseable.patches[index].get_width()
            )
            assert total == pytest.approx(metrics["error_rate"].iloc[index])

    def test_unparseable_segment_uses_the_unknown_rate(
        self, metrics: pd.DataFrame
    ) -> None:
        axes = plot_error_rate_comparison(metrics).axes[0]
        _, unparseable = axes.containers
        widths = [patch.get_width() for patch in unparseable.patches]
        assert widths == pytest.approx(metrics["unknown_rate"].tolist())

    def test_totals_are_printed(self, metrics: pd.DataFrame) -> None:
        axes = plot_error_rate_comparison(metrics).axes[0]
        printed = {text.get_text() for text in axes.texts}
        assert {"40.0%", "10.0%", "2.5%"} <= printed

    def test_legend_names_both_causes(self, metrics: pd.DataFrame) -> None:
        axes = plot_error_rate_comparison(metrics).axes[0]
        entries = {text.get_text() for text in axes.get_legend().get_texts()}
        assert entries == {"Misclassified", "Unparseable"}


class TestLatencyChart:
    def test_bar_lengths_are_the_recorded_latencies(self, metrics: pd.DataFrame) -> None:
        axes = plot_latency_comparison(metrics).axes[0]
        widths = sorted(patch.get_width() for patch in axes.patches)
        assert widths == pytest.approx(sorted(metrics["avg_latency_seconds"].tolist()))

    def test_values_are_printed_with_units(self, metrics: pd.DataFrame) -> None:
        axes = plot_latency_comparison(metrics).axes[0]
        assert {"1.05s", "1.30s", "2.10s"} <= {text.get_text() for text in axes.texts}

    def test_unrecorded_latency_is_refused_not_faked(
        self, metrics: pd.DataFrame
    ) -> None:
        # A run without latency data must produce no chart rather than zeros.
        with pytest.raises(ChartError, match="not recorded"):
            plot_latency_comparison(metrics.assign(avg_latency_seconds=None))


class TestConfusionMatrixCharts:
    def test_single_matrix_has_the_unknown_column(
        self, predictions: pd.DataFrame
    ) -> None:
        axes = plot_confusion_matrix(predictions, "zero_shot").axes[0]
        labels = [text.get_text() for text in axes.get_xticklabels()]
        assert labels == [NEG, POS, UNK]

    def test_cell_annotations_are_the_actual_counts(
        self, predictions: pd.DataFrame
    ) -> None:
        axes = plot_confusion_matrix(predictions, "zero_shot").axes[0]
        annotated = sorted(int(text.get_text()) for text in axes.texts)
        # zero_shot: 1 correct positive, 1 unknown positive, 2 correct negatives
        assert sum(annotated) == 4

    def test_unknown_strategy_is_refused(self, predictions: pd.DataFrame) -> None:
        with pytest.raises(ChartError, match="No predictions found"):
            plot_confusion_matrix(predictions, "nonexistent")

    def test_small_multiples_draw_one_panel_per_strategy(
        self, predictions: pd.DataFrame
    ) -> None:
        figure = plot_confusion_matrices(predictions)
        drawn = [axes for axes in figure.axes if axes.get_title()]
        assert len(drawn) == 2

    def test_panels_share_one_colour_scale(self, predictions: pd.DataFrame) -> None:
        figure = plot_confusion_matrices(predictions)
        limits = {
            axes.collections[0].get_clim()
            for axes in figure.axes
            if axes.collections
        }
        assert len(limits) == 1

    def test_panel_titles_use_the_shared_formatter(
        self, predictions: pd.DataFrame
    ) -> None:
        figure = plot_confusion_matrices(predictions)
        titles = {axes.get_title() for axes in figure.axes if axes.get_title()}
        assert titles == {"Zero-shot", "Combined"}


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


class TestSaving:
    def test_save_figure_writes_a_png(self, metrics: pd.DataFrame, tmp_path: Path) -> None:
        path = save_figure(plot_accuracy_comparison(metrics), tmp_path / "a" / "chart.png")
        assert path.exists()
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_output_path_argument_saves_directly(
        self, metrics: pd.DataFrame, tmp_path: Path
    ) -> None:
        target = tmp_path / "accuracy.png"
        plot_accuracy_comparison(metrics, output_path=target)
        assert target.exists()

    def test_render_all_charts_writes_the_full_set(
        self, metrics: pd.DataFrame, predictions: pd.DataFrame, tmp_path: Path
    ) -> None:
        written = render_all_charts(metrics, predictions, output_dir=tmp_path)
        assert set(written) == set(CHART_FILENAMES)
        for key, path in written.items():
            assert path.name == CHART_FILENAMES[key]
            assert path.exists()

    def test_charts_without_input_are_skipped_not_faked(
        self, metrics: pd.DataFrame, tmp_path: Path
    ) -> None:
        # No latency recorded and no predictions supplied: those two are absent,
        # and nothing is drawn from placeholder values.
        written = render_all_charts(
            metrics.assign(avg_latency_seconds=None), None, output_dir=tmp_path
        )
        assert "latency" not in written
        assert "confusion" not in written
        assert "accuracy" in written

    def test_renders_straight_from_a_stored_experiment(
        self, metrics: pd.DataFrame, predictions: pd.DataFrame, tmp_path: Path
    ) -> None:
        from src.experiments.storage import ExperimentStorage

        storage = ExperimentStorage(tmp_path / "experiments")
        experiment_id = storage.create_experiment()
        storage.save_metrics(experiment_id, metrics)
        storage.save_predictions(experiment_id, predictions)

        written = render_experiment_charts(experiment_id, storage)
        expected_dir = storage.experiment_path(experiment_id) / "charts"
        assert set(written) == set(CHART_FILENAMES)
        assert all(path.parent == expected_dir for path in written.values())
