"""Tests for metrics and error analysis.

All predictions are constructed by hand so every expected number can be
verified by counting, not by trusting the implementation. The unknown-handling
policy is pinned down explicitly: it is the one place where a silent change
would alter every published result.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.config import LABEL_NEGATIVE, LABEL_POSITIVE, LABEL_UNKNOWN
from src.evaluation.errors import (
    ERROR_COLUMNS,
    ERROR_MISCLASSIFICATION,
    ERROR_UNPARSEABLE,
    ErrorAnalysisError,
    classify_error,
    error_analysis_summary,
    extract_errors,
    find_shared_errors,
    summarize_errors_by_strategy,
    unique_errors,
)
from src.evaluation.metrics import (
    METRICS_COLUMNS,
    evaluate_strategies,
    rank_strategies,
    MetricsError,
    best_strategy,
    build_metrics_table,
    classification_report_frame,
    compute_classification_metrics,
    compute_prediction_counts,
    confusion_matrix_frame,
    evaluate_predictions,
    validate_predictions,
)

POS = LABEL_POSITIVE
NEG = LABEL_NEGATIVE
UNK = LABEL_UNKNOWN


# ---------------------------------------------------------------------------
# Hand-built fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def perfect() -> tuple[list[str], list[str]]:
    return [POS, NEG, POS, NEG], [POS, NEG, POS, NEG]


@pytest.fixture
def mixed() -> tuple[list[str], list[str]]:
    """10 samples: 7 correct, 2 misclassified, 1 unparseable.

    actual    : P P P P P N N N N N
    predicted : P P P P N N N N P U
    correct   : 1 2 3 4 . 5 6 7 . .
    """
    actual = [POS] * 5 + [NEG] * 5
    predicted = [POS, POS, POS, POS, NEG, NEG, NEG, NEG, POS, UNK]
    return actual, predicted


@pytest.fixture
def predictions_frame() -> pd.DataFrame:
    """Two strategies over three samples, with one error of each kind."""
    return pd.DataFrame(
        {
            "sample_id": ["test-001", "test-002", "test-003"] * 2,
            "strategy": ["zero_shot"] * 3 + ["structured"] * 3,
            "review": ["great film", "dull film", "odd film"] * 2,
            "actual_label": [POS, NEG, POS] * 2,
            "predicted_label": [POS, POS, UNK, POS, NEG, NEG],
            "raw_response": [
                "positive",
                "positive",
                "I cannot decide",
                '{"sentiment": "positive"}',
                '{"sentiment": "negative"}',
                '{"sentiment": "negative"}',
            ],
        }
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_accepts_lists_and_series(self) -> None:
        actual, predicted = validate_predictions(pd.Series([POS]), [POS])
        assert actual == [POS] and predicted == [POS]

    def test_rejects_empty_input(self) -> None:
        with pytest.raises(MetricsError, match="empty"):
            compute_prediction_counts([], [])

    def test_rejects_length_mismatch(self) -> None:
        with pytest.raises(MetricsError, match="Length mismatch"):
            compute_prediction_counts([POS, NEG], [POS])

    def test_rejects_unknown_in_ground_truth(self) -> None:
        # An unlabelled sample is a data bug, not a model failure.
        with pytest.raises(MetricsError, match="Ground-truth"):
            compute_prediction_counts([POS, UNK], [POS, POS])

    def test_rejects_invalid_predicted_label(self) -> None:
        with pytest.raises(MetricsError, match="Predictions must be"):
            compute_prediction_counts([POS], ["neutral"])

    def test_allows_unknown_in_predictions(self) -> None:
        assert compute_prediction_counts([POS], [UNK])["unknown"] == 1


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


class TestPredictionCounts:
    def test_perfect_predictions(self, perfect: tuple[list[str], list[str]]) -> None:
        counts = compute_prediction_counts(*perfect)
        assert counts["total_samples"] == 4
        assert counts["correct"] == 4
        assert counts["incorrect"] == 0
        assert counts["unknown"] == 0
        assert counts["accuracy"] == 1.0
        assert counts["error_rate"] == 0.0

    def test_hand_counted_mixed_case(self, mixed: tuple[list[str], list[str]]) -> None:
        counts = compute_prediction_counts(*mixed)
        assert counts["total_samples"] == 10
        assert counts["correct"] == 7
        assert counts["incorrect"] == 3
        assert counts["unknown"] == 1
        assert counts["resolved"] == 9
        assert counts["accuracy"] == 0.7
        assert counts["error_rate"] == pytest.approx(0.3)
        assert counts["unknown_rate"] == 0.1

    def test_counts_always_reconcile(self, mixed: tuple[list[str], list[str]]) -> None:
        counts = compute_prediction_counts(*mixed)
        assert counts["correct"] + counts["incorrect"] == counts["total_samples"]
        assert counts["resolved"] + counts["unknown"] == counts["total_samples"]

    def test_all_wrong(self) -> None:
        counts = compute_prediction_counts([POS, NEG], [NEG, POS])
        assert counts["accuracy"] == 0.0
        assert counts["correct"] == 0


class TestUnknownPolicy:
    """The documented treatment of unparseable predictions."""

    def test_unknown_counts_as_incorrect_in_accuracy(self) -> None:
        counts = compute_prediction_counts([POS, POS], [POS, UNK])
        assert counts["accuracy"] == 0.5
        assert counts["incorrect"] == 1

    def test_unknown_is_never_silently_dropped(self) -> None:
        counts = compute_prediction_counts([POS, POS, POS], [POS, UNK, UNK])
        # Denominator is still 3, not the 1 resolved sample.
        assert counts["total_samples"] == 3
        assert counts["accuracy"] == pytest.approx(1 / 3)

    def test_accuracy_on_resolved_excludes_unknown(self) -> None:
        counts = compute_prediction_counts([POS, POS, POS], [POS, POS, UNK])
        assert counts["accuracy"] == pytest.approx(2 / 3)
        assert counts["accuracy_on_resolved"] == 1.0

    def test_accuracy_on_resolved_is_none_when_nothing_resolved(self) -> None:
        # None, not 0.0: undefined is not the same as "got everything wrong".
        counts = compute_prediction_counts([POS, NEG], [UNK, UNK])
        assert counts["accuracy_on_resolved"] is None
        assert counts["accuracy"] == 0.0

    def test_unknown_hurts_recall_but_not_precision(self) -> None:
        # Two positives: one predicted correctly, one unparseable.
        metrics = compute_classification_metrics([POS, POS], [POS, UNK])
        assert metrics["precision_positive"] == 1.0   # no wrong positive claims
        assert metrics["recall_positive"] == 0.5      # one true positive missed

    def test_unknown_rate_is_reported(self) -> None:
        counts = compute_prediction_counts([POS] * 4, [POS, UNK, UNK, POS])
        assert counts["unknown_rate"] == 0.5


# ---------------------------------------------------------------------------
# Precision / recall / F1
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    def test_perfect_scores(self, perfect: tuple[list[str], list[str]]) -> None:
        metrics = compute_classification_metrics(*perfect)
        for key in ("precision_macro", "recall_macro", "f1_macro"):
            assert metrics[key] == 1.0

    def test_hand_computed_values(self) -> None:
        # actual   : P P N N
        # predicted: P N N N
        # positive: TP=1, FP=0, FN=1  -> precision 1.0, recall 0.5, F1 0.667
        # negative: TP=2, FP=1, FN=0  -> precision 0.667, recall 1.0, F1 0.8
        metrics = compute_classification_metrics([POS, POS, NEG, NEG], [POS, NEG, NEG, NEG])
        assert metrics["precision_positive"] == 1.0
        assert metrics["recall_positive"] == 0.5
        assert metrics["f1_positive"] == pytest.approx(2 / 3, abs=1e-4)
        assert metrics["precision_negative"] == pytest.approx(2 / 3, abs=1e-4)
        assert metrics["recall_negative"] == 1.0
        assert metrics["f1_negative"] == pytest.approx(0.8)

    def test_macro_is_the_unweighted_mean_of_the_classes(self) -> None:
        metrics = compute_classification_metrics([POS, POS, NEG, NEG], [POS, NEG, NEG, NEG])
        expected = (metrics["precision_positive"] + metrics["precision_negative"]) / 2
        assert metrics["precision_macro"] == pytest.approx(expected)

    def test_support_counts_ground_truth_not_predictions(self) -> None:
        metrics = compute_classification_metrics([POS] * 3 + [NEG], [UNK] * 4)
        assert metrics["support_positive"] == 3
        assert metrics["support_negative"] == 1

    def test_never_predicted_class_scores_zero_not_nan(self) -> None:
        metrics = compute_classification_metrics([POS, NEG], [POS, POS])
        assert metrics["precision_negative"] == 0.0
        assert metrics["f1_negative"] == 0.0

    def test_all_values_are_plain_floats(self, mixed: tuple[list[str], list[str]]) -> None:
        metrics = compute_classification_metrics(*mixed)
        assert all(
            isinstance(value, (float, int)) for value in metrics.values()
        )


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------


class TestConfusionMatrix:
    def test_shape_and_labels(self, mixed: tuple[list[str], list[str]]) -> None:
        matrix = confusion_matrix_frame(*mixed)
        assert list(matrix.index) == [NEG, POS]
        assert list(matrix.columns) == [NEG, POS, UNK]
        assert matrix.index.name == "actual"
        assert matrix.columns.name == "predicted"

    def test_hand_counted_cells(self, mixed: tuple[list[str], list[str]]) -> None:
        matrix = confusion_matrix_frame(*mixed)
        assert matrix.loc[POS, POS] == 4
        assert matrix.loc[POS, NEG] == 1
        assert matrix.loc[NEG, NEG] == 3
        assert matrix.loc[NEG, POS] == 1
        assert matrix.loc[NEG, UNK] == 1
        assert matrix.loc[POS, UNK] == 0

    def test_totals_match_the_sample_count(self, mixed: tuple[list[str], list[str]]) -> None:
        assert confusion_matrix_frame(*mixed).to_numpy().sum() == 10

    def test_unknown_column_present_even_when_empty(
        self, perfect: tuple[list[str], list[str]]
    ) -> None:
        # Consistent shape across strategies keeps matrices comparable.
        matrix = confusion_matrix_frame(*perfect)
        assert UNK in matrix.columns
        assert matrix[UNK].sum() == 0

    def test_excluding_unknown_drops_those_counts(
        self, mixed: tuple[list[str], list[str]]
    ) -> None:
        matrix = confusion_matrix_frame(*mixed, include_unknown=False)
        assert list(matrix.columns) == [NEG, POS]
        assert matrix.to_numpy().sum() == 9  # the unparseable sample is excluded

    def test_never_has_an_unknown_row(self, mixed: tuple[list[str], list[str]]) -> None:
        assert UNK not in confusion_matrix_frame(*mixed).index


class TestClassificationReport:
    def test_contains_both_classes(self, mixed: tuple[list[str], list[str]]) -> None:
        report = classification_report_frame(*mixed)
        assert POS in report.index
        assert NEG in report.index

    def test_has_the_standard_columns(self, mixed: tuple[list[str], list[str]]) -> None:
        report = classification_report_frame(*mixed)
        for column in ("precision", "recall", "f1-score", "support"):
            assert column in report.columns

    def test_support_is_integer(self, mixed: tuple[list[str], list[str]]) -> None:
        report = classification_report_frame(*mixed)
        assert report.loc[POS, "support"] == 5


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestEvaluatePredictions:
    def test_returns_every_metrics_column(self, mixed: tuple[list[str], list[str]]) -> None:
        record = evaluate_predictions(*mixed, strategy="zero_shot")
        assert list(record) == list(METRICS_COLUMNS)

    def test_carries_operational_metrics_through(
        self, mixed: tuple[list[str], list[str]]
    ) -> None:
        record = evaluate_predictions(
            *mixed, strategy="zero_shot", api_failures=2, runtime_seconds=12.5
        )
        assert record["strategy"] == "zero_shot"
        assert record["api_failures"] == 2
        assert record["runtime_seconds"] == 12.5

    def test_defaults_are_explicit_not_invented(
        self, mixed: tuple[list[str], list[str]]
    ) -> None:
        record = evaluate_predictions(*mixed)
        assert record["strategy"] is None
        assert record["api_failures"] == 0
        assert record["runtime_seconds"] is None


class TestMetricsTable:
    def test_builds_one_row_per_strategy(self, mixed: tuple[list[str], list[str]]) -> None:
        records = [
            evaluate_predictions(*mixed, strategy="zero_shot"),
            evaluate_predictions(*mixed, strategy="few_shot"),
        ]
        table = build_metrics_table(records)
        assert len(table) == 2
        assert list(table.columns) == list(METRICS_COLUMNS)

    def test_preserves_evaluation_order(self, mixed: tuple[list[str], list[str]]) -> None:
        names = ["zero_shot", "few_shot", "combined"]
        table = build_metrics_table(
            [evaluate_predictions(*mixed, strategy=name) for name in names]
        )
        assert table["strategy"].tolist() == names

    def test_empty_input_gives_an_empty_typed_frame(self) -> None:
        table = build_metrics_table([])
        assert table.empty
        assert list(table.columns) == list(METRICS_COLUMNS)


class TestBestStrategy:
    def test_returns_the_top_scorer(self) -> None:
        table = pd.DataFrame(
            {"strategy": ["zero_shot", "few_shot"], "accuracy": [0.7, 0.9]}
        )
        assert best_strategy(table) == "few_shot"

    def test_refuses_to_break_a_tie(self) -> None:
        table = pd.DataFrame(
            {"strategy": ["zero_shot", "few_shot"], "accuracy": [0.9, 0.9]}
        )
        assert best_strategy(table) is None

    def test_supports_other_metrics(self) -> None:
        table = pd.DataFrame(
            {
                "strategy": ["a", "b"],
                "accuracy": [0.9, 0.5],
                "f1_macro": [0.4, 0.8],
            }
        )
        assert best_strategy(table, metric="f1_macro") == "b"

    def test_empty_table_has_no_winner(self) -> None:
        assert best_strategy(pd.DataFrame(columns=["strategy", "accuracy"])) is None


# ---------------------------------------------------------------------------
# Error extraction
# ---------------------------------------------------------------------------


class TestClassifyError:
    def test_correct_prediction_is_not_an_error(self) -> None:
        assert classify_error(POS, POS) is None

    def test_wrong_label_is_a_misclassification(self) -> None:
        assert classify_error(POS, NEG) == ERROR_MISCLASSIFICATION

    def test_unknown_is_unparseable(self) -> None:
        assert classify_error(POS, UNK) == ERROR_UNPARSEABLE


class TestExtractErrors:
    def test_extracts_only_incorrect_rows(self, predictions_frame: pd.DataFrame) -> None:
        errors = extract_errors(predictions_frame)
        assert len(errors) == 3  # by hand: zero_shot s2 + s3, structured s3
        assert (errors["actual_label"] != errors["predicted_label"]).all()

    def test_includes_every_required_field(self, predictions_frame: pd.DataFrame) -> None:
        errors = extract_errors(predictions_frame)
        assert list(errors.columns) == list(ERROR_COLUMNS)

    def test_preserves_the_raw_response(self, predictions_frame: pd.DataFrame) -> None:
        errors = extract_errors(predictions_frame, strategy="zero_shot")
        unparseable = errors[errors["error_type"] == ERROR_UNPARSEABLE]
        assert unparseable["raw_response"].iloc[0] == "I cannot decide"

    def test_labels_both_failure_modes(self, predictions_frame: pd.DataFrame) -> None:
        errors = extract_errors(predictions_frame)
        assert set(errors["error_type"]) == {ERROR_MISCLASSIFICATION, ERROR_UNPARSEABLE}

    def test_filters_by_strategy(self, predictions_frame: pd.DataFrame) -> None:
        errors = extract_errors(predictions_frame, strategy="structured")
        assert set(errors["strategy"]) == {"structured"}
        assert len(errors) == 1

    def test_is_sorted_for_diffable_output(self, predictions_frame: pd.DataFrame) -> None:
        errors = extract_errors(predictions_frame)
        assert errors[["strategy", "sample_id"]].values.tolist() == sorted(
            errors[["strategy", "sample_id"]].values.tolist()
        )

    def test_no_errors_returns_an_empty_typed_frame(self) -> None:
        perfect = pd.DataFrame(
            {
                "sample_id": ["s1"],
                "strategy": ["zero_shot"],
                "review": ["great"],
                "actual_label": [POS],
                "predicted_label": [POS],
                "raw_response": ["positive"],
            }
        )
        errors = extract_errors(perfect)
        assert errors.empty
        assert list(errors.columns) == list(ERROR_COLUMNS)

    def test_missing_columns_are_reported(self) -> None:
        with pytest.raises(ErrorAnalysisError, match="missing column"):
            extract_errors(pd.DataFrame({"sample_id": ["s1"]}))

    def test_unknown_strategy_filter_yields_nothing(
        self, predictions_frame: pd.DataFrame
    ) -> None:
        assert extract_errors(predictions_frame, strategy="nope").empty


class TestErrorComparison:
    def test_summary_counts_each_failure_mode(
        self, predictions_frame: pd.DataFrame
    ) -> None:
        summary = summarize_errors_by_strategy(extract_errors(predictions_frame))
        zero_shot = summary[summary["strategy"] == "zero_shot"].iloc[0]
        assert zero_shot[ERROR_MISCLASSIFICATION] == 1
        assert zero_shot[ERROR_UNPARSEABLE] == 1
        assert zero_shot["total_errors"] == 2

    def test_summary_fills_absent_failure_modes_with_zero(
        self, predictions_frame: pd.DataFrame
    ) -> None:
        summary = summarize_errors_by_strategy(extract_errors(predictions_frame))
        structured = summary[summary["strategy"] == "structured"].iloc[0]
        assert structured[ERROR_UNPARSEABLE] == 0

    def test_shared_errors_are_samples_every_strategy_failed(self) -> None:
        errors = pd.DataFrame(
            {
                "sample_id": ["s1", "s1", "s2"],
                "strategy": ["a", "b", "a"],
                "review": ["hard review", "hard review", "easy review"],
                "actual_label": [POS, POS, NEG],
            }
        )
        shared = find_shared_errors(errors, strategy_count=2)
        assert shared["sample_id"].tolist() == ["s1"]
        assert shared["failed_strategies"].iloc[0] == 2

    def test_shared_errors_respects_explicit_strategy_count(self) -> None:
        errors = pd.DataFrame(
            {
                "sample_id": ["s1", "s1"],
                "strategy": ["a", "b"],
                "review": ["hard", "hard"],
                "actual_label": [POS, POS],
            }
        )
        # Six strategies ran; only two failed this sample, so it is not shared.
        assert find_shared_errors(errors, strategy_count=6).empty

    def test_unique_errors_isolate_one_strategy(
        self, predictions_frame: pd.DataFrame
    ) -> None:
        errors = extract_errors(predictions_frame)
        # test-002 is failed by zero_shot alone; test-003 is failed by both,
        # so it belongs to neither strategy's unique set.
        assert unique_errors(errors, "zero_shot")["sample_id"].tolist() == ["test-002"]
        assert unique_errors(errors, "structured").empty

    def test_summary_of_an_empty_frame_is_zeroed(self) -> None:
        summary = error_analysis_summary(pd.DataFrame())
        assert summary["total_errors"] == 0
        assert summary["unparseable"] == 0

    def test_summary_reports_headline_figures(
        self, predictions_frame: pd.DataFrame
    ) -> None:
        summary = error_analysis_summary(extract_errors(predictions_frame))
        assert summary["total_errors"] == 3
        assert summary["misclassifications"] == 2
        assert summary["unparseable"] == 1
        assert summary["strategies_with_errors"] == 2


# ---------------------------------------------------------------------------
# Metrics and error analysis must agree
# ---------------------------------------------------------------------------


class TestConsistencyBetweenModules:
    def test_error_count_matches_the_metrics_incorrect_count(
        self, predictions_frame: pd.DataFrame
    ) -> None:
        for strategy in predictions_frame["strategy"].unique():
            subset = predictions_frame[predictions_frame["strategy"] == strategy]
            counts = compute_prediction_counts(
                subset["actual_label"], subset["predicted_label"]
            )
            errors = extract_errors(predictions_frame, strategy=strategy)
            assert counts["incorrect"] == len(errors)

    def test_unparseable_count_matches_the_unknown_count(
        self, predictions_frame: pd.DataFrame
    ) -> None:
        for strategy in predictions_frame["strategy"].unique():
            subset = predictions_frame[predictions_frame["strategy"] == strategy]
            counts = compute_prediction_counts(
                subset["actual_label"], subset["predicted_label"]
            )
            errors = extract_errors(predictions_frame, strategy=strategy)
            unparseable = (errors["error_type"] == ERROR_UNPARSEABLE).sum()
            assert counts["unknown"] == unparseable


# ---------------------------------------------------------------------------
# Scoring a predictions frame (Step 9)
# ---------------------------------------------------------------------------


@pytest.fixture
def scored_frame() -> pd.DataFrame:
    """Three strategies over the same 8 samples, with hand-checkable outcomes.

    Ground truth       : P P P P N N N N
    zero_shot  predicts: P P N N N N N N   -> 6 correct, 0 unknown
    structured predicts: P P P P N N N N   -> 8 correct, 0 unknown
    reasoning  predicts: P P U P N N N U   -> 6 correct, 2 unknown
    """
    actual = [POS] * 4 + [NEG] * 4
    scripts = {
        "zero_shot": [POS, POS, NEG, NEG, NEG, NEG, NEG, NEG],
        "structured": [POS, POS, POS, POS, NEG, NEG, NEG, NEG],
        "reasoning": [POS, POS, UNK, POS, NEG, NEG, NEG, UNK],
    }
    rows = []
    for strategy, predicted in scripts.items():
        for index, (truth, prediction) in enumerate(zip(actual, predicted, strict=True)):
            rows.append(
                {
                    "strategy": strategy,
                    "sample_id": f"test-{index:05d}",
                    "actual_label": truth,
                    "predicted_label": prediction,
                    "latency_seconds": 0.1,
                    "success": prediction != UNK or strategy == "reasoning",
                }
            )
    return pd.DataFrame(rows)


class TestEvaluateStrategies:
    def test_one_row_per_strategy_in_execution_order(
        self, scored_frame: pd.DataFrame
    ) -> None:
        metrics = evaluate_strategies(scored_frame)
        assert metrics["strategy"].tolist() == ["zero_shot", "structured", "reasoning"]
        assert list(metrics.columns) == list(METRICS_COLUMNS)

    def test_hand_verified_zero_shot_row(self, scored_frame: pd.DataFrame) -> None:
        # positive: TP=2 FP=0 FN=2 -> P 1.000 R 0.500 F1 0.667
        # negative: TP=4 FP=2 FN=0 -> P 0.667 R 1.000 F1 0.800
        row = evaluate_strategies(scored_frame).set_index("strategy").loc["zero_shot"]
        assert row["correct"] == 6
        assert row["accuracy"] == 0.75
        assert row["precision_positive"] == 1.0
        assert row["recall_positive"] == 0.5
        assert row["precision_negative"] == pytest.approx(2 / 3, abs=1e-4)
        assert row["recall_negative"] == 1.0
        assert row["precision_macro"] == pytest.approx(0.8333, abs=1e-4)
        assert row["recall_macro"] == 0.75
        assert row["f1_macro"] == pytest.approx(0.7333, abs=1e-4)
        assert row["error_rate"] == 0.25
        assert row["unknown_rate"] == 0.0

    def test_hand_verified_perfect_row(self, scored_frame: pd.DataFrame) -> None:
        row = evaluate_strategies(scored_frame).set_index("strategy").loc["structured"]
        assert row["accuracy"] == 1.0
        assert row["f1_macro"] == 1.0
        assert row["incorrect"] == 0

    def test_hand_verified_row_with_unknowns(self, scored_frame: pd.DataFrame) -> None:
        # Two unparseable answers, one per class.
        # positive: TP=3 FP=0 FN=1 -> P 1.000 R 0.750 F1 0.857
        # negative: TP=3 FP=0 FN=1 -> P 1.000 R 0.750 F1 0.857
        row = evaluate_strategies(scored_frame).set_index("strategy").loc["reasoning"]
        assert row["unknown"] == 2
        assert row["unknown_rate"] == 0.25
        assert row["accuracy"] == 0.75
        assert row["accuracy_on_resolved"] == 1.0
        assert row["precision_macro"] == 1.0
        assert row["recall_macro"] == 0.75
        assert row["f1_macro"] == pytest.approx(0.8571, abs=1e-4)

    def test_average_latency_is_computed(self, scored_frame: pd.DataFrame) -> None:
        metrics = evaluate_strategies(scored_frame)
        assert metrics["avg_latency_seconds"].tolist() == pytest.approx([0.1] * 3)

    def test_runtime_falls_back_to_summed_latency(
        self, scored_frame: pd.DataFrame
    ) -> None:
        metrics = evaluate_strategies(scored_frame).set_index("strategy")
        assert metrics.loc["zero_shot", "runtime_seconds"] == pytest.approx(0.8)

    def test_injected_wall_clock_runtime_wins(self, scored_frame: pd.DataFrame) -> None:
        metrics = evaluate_strategies(
            scored_frame, runtime_by_strategy={"zero_shot": 42.0}
        ).set_index("strategy")
        assert metrics.loc["zero_shot", "runtime_seconds"] == 42.0
        # Strategies without an injected value still fall back.
        assert metrics.loc["structured", "runtime_seconds"] == pytest.approx(0.8)

    def test_api_failures_are_counted_from_the_success_column(self) -> None:
        frame = pd.DataFrame(
            {
                "strategy": ["zero_shot"] * 3,
                "actual_label": [POS, POS, NEG],
                "predicted_label": [POS, UNK, NEG],
                "success": [True, False, True],
            }
        )
        assert evaluate_strategies(frame)["api_failures"].iloc[0] == 1

    def test_optional_columns_are_reported_as_unavailable(self) -> None:
        # No latency column: report None rather than a plausible-looking zero.
        frame = pd.DataFrame(
            {
                "strategy": ["zero_shot"] * 2,
                "actual_label": [POS, NEG],
                "predicted_label": [POS, NEG],
            }
        )
        row = evaluate_strategies(frame).iloc[0]
        assert row["avg_latency_seconds"] is None
        assert row["runtime_seconds"] is None
        assert row["api_failures"] == 0

    def test_metrics_are_reproducible_from_predictions_alone(
        self, scored_frame: pd.DataFrame
    ) -> None:
        first = evaluate_strategies(scored_frame)
        second = evaluate_strategies(scored_frame)
        pd.testing.assert_frame_equal(first, second)

    def test_rejects_missing_columns(self) -> None:
        with pytest.raises(MetricsError, match="missing column"):
            evaluate_strategies(pd.DataFrame({"strategy": ["zero_shot"]}))

    def test_rejects_an_empty_frame(self) -> None:
        with pytest.raises(MetricsError, match="empty"):
            evaluate_strategies(
                pd.DataFrame(columns=["strategy", "actual_label", "predicted_label"])
            )


class TestRankStrategies:
    def test_orders_by_f1_best_first(self, scored_frame: pd.DataFrame) -> None:
        ranked = rank_strategies(evaluate_strategies(scored_frame))
        assert ranked["strategy"].tolist() == ["structured", "reasoning", "zero_shot"]
        assert ranked["rank"].tolist() == [1, 2, 3]

    def test_ties_share_a_rank(self) -> None:
        metrics = pd.DataFrame(
            {"strategy": ["a", "b", "c"], "f1_macro": [0.9, 0.9, 0.5]}
        )
        ranked = rank_strategies(metrics)
        assert ranked["rank"].tolist() == [1, 1, 3]

    def test_supports_another_metric(self, scored_frame: pd.DataFrame) -> None:
        ranked = rank_strategies(evaluate_strategies(scored_frame), metric="accuracy")
        assert ranked["strategy"].iloc[0] == "structured"

    def test_empty_input_is_handled(self) -> None:
        assert rank_strategies(pd.DataFrame()).empty
