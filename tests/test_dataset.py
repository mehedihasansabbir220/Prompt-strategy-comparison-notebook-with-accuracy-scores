"""Tests for the dataset layer.

Every test runs against a small in-memory sample. The single test that touches
:func:`load_imdb_dataset` mocks ``datasets.load_dataset``, so the suite never
downloads anything and never needs a network connection.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from src.config import LABEL_NEGATIVE, LABEL_POSITIVE, SENTIMENT_LABELS
from src.dataset import (
    IMDB_DATASET_NAME,
    LABEL_COLUMN,
    PREPARED_COLUMNS,
    RAW_LABEL_COLUMN,
    RAW_TEXT_COLUMN,
    REVIEW_COLUMN,
    SAMPLE_ID_COLUMN,
    DatasetError,
    FewShotExample,
    create_few_shot_examples,
    create_fixed_evaluation_set,
    load_imdb_dataset,
    normalize_labels,
    prepare_dataset,
    validate_labels,
    validate_prepared_dataset,
)

# ---------------------------------------------------------------------------
# Fixtures — small local samples, no downloads
# ---------------------------------------------------------------------------


def _raw_frame(rows: int = 20) -> pd.DataFrame:
    """Build a raw IMDb-shaped sample: alternating labels, HTML noise included."""
    return pd.DataFrame(
        {
            RAW_TEXT_COLUMN: [
                f"Review number {index}.<br /><br />It was "
                f"{'wonderful' if index % 2 else 'terrible'} in every way."
                for index in range(rows)
            ],
            RAW_LABEL_COLUMN: [index % 2 for index in range(rows)],
        }
    )


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    return _raw_frame()


@pytest.fixture
def prepared(raw_frame: pd.DataFrame) -> pd.DataFrame:
    return prepare_dataset(raw_frame, split="test")


# ---------------------------------------------------------------------------
# load_imdb_dataset
# ---------------------------------------------------------------------------


class _FakeHFDataset:
    """Stand-in for a Hugging Face ``Dataset``: only ``to_pandas`` is needed."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self._frame


class TestLoadImdbDataset:
    def test_returns_raw_frame_without_network(
        self, monkeypatch: pytest.MonkeyPatch, raw_frame: pd.DataFrame
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_load_dataset(name: str, **kwargs: Any) -> _FakeHFDataset:
            captured["name"] = name
            captured.update(kwargs)
            return _FakeHFDataset(raw_frame)

        monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

        result = load_imdb_dataset(split="train")

        assert captured["name"] == IMDB_DATASET_NAME
        assert captured["split"] == "train"
        assert list(result.columns) == [RAW_TEXT_COLUMN, RAW_LABEL_COLUMN]
        assert len(result) == len(raw_frame)

    def test_missing_columns_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        bad = pd.DataFrame({"content": ["x"], "sentiment": [1]})
        monkeypatch.setattr(
            "datasets.load_dataset", lambda *a, **k: _FakeHFDataset(bad)
        )
        with pytest.raises(DatasetError, match="missing required column"):
            load_imdb_dataset()

    def test_download_failure_is_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*args: Any, **kwargs: Any) -> None:
            raise ConnectionError("no route to host")

        monkeypatch.setattr("datasets.load_dataset", boom)
        with pytest.raises(DatasetError, match="Failed to load dataset"):
            load_imdb_dataset()


# ---------------------------------------------------------------------------
# normalize_labels
# ---------------------------------------------------------------------------


class TestNormalizeLabels:
    def test_maps_imdb_integers(self) -> None:
        frame = pd.DataFrame({RAW_LABEL_COLUMN: [0, 1, 1, 0]})
        result = normalize_labels(frame)
        assert list(result[LABEL_COLUMN]) == [
            LABEL_NEGATIVE,
            LABEL_POSITIVE,
            LABEL_POSITIVE,
            LABEL_NEGATIVE,
        ]

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("pos", LABEL_POSITIVE),
            ("POSITIVE", LABEL_POSITIVE),
            (" neg ", LABEL_NEGATIVE),
            ("Negative", LABEL_NEGATIVE),
        ],
    )
    def test_maps_string_aliases(self, raw: str, expected: str) -> None:
        frame = pd.DataFrame({RAW_LABEL_COLUMN: [raw]})
        assert normalize_labels(frame)[LABEL_COLUMN].iloc[0] == expected

    def test_does_not_mutate_the_input(self) -> None:
        frame = pd.DataFrame({RAW_LABEL_COLUMN: [0, 1]})
        before = frame.copy()
        normalize_labels(frame, target_column="normalised")
        pd.testing.assert_frame_equal(frame, before)

    def test_rejects_unknown_numeric_label(self) -> None:
        frame = pd.DataFrame({RAW_LABEL_COLUMN: [0, 7]})
        with pytest.raises(DatasetError, match="Unknown numeric label"):
            normalize_labels(frame)

    def test_rejects_unknown_string_label(self) -> None:
        frame = pd.DataFrame({RAW_LABEL_COLUMN: ["neutral"]})
        with pytest.raises(DatasetError, match="Unknown label"):
            normalize_labels(frame)

    def test_rejects_missing_values(self) -> None:
        frame = pd.DataFrame({RAW_LABEL_COLUMN: [0, None]})
        with pytest.raises(DatasetError, match="missing values"):
            normalize_labels(frame)

    def test_rejects_missing_column(self) -> None:
        with pytest.raises(DatasetError, match="missing required column"):
            normalize_labels(pd.DataFrame({"other": [1]}))


class TestValidateLabels:
    def test_accepts_project_vocabulary(self) -> None:
        validate_labels([LABEL_POSITIVE, LABEL_NEGATIVE])

    def test_rejects_anything_else(self) -> None:
        with pytest.raises(DatasetError, match="Unexpected label"):
            validate_labels([LABEL_POSITIVE, "unknown"])


# ---------------------------------------------------------------------------
# prepare_dataset
# ---------------------------------------------------------------------------


class TestPrepareDataset:
    def test_produces_the_expected_schema(self, prepared: pd.DataFrame) -> None:
        assert list(prepared.columns) == list(PREPARED_COLUMNS)
        assert list(prepared.index) == list(range(len(prepared)))

    def test_strips_html_line_breaks_and_extra_whitespace(
        self, prepared: pd.DataFrame
    ) -> None:
        joined = " ".join(prepared[REVIEW_COLUMN])
        assert "<br" not in joined
        assert "  " not in joined

    def test_sample_ids_are_unique_and_prefixed_by_split(
        self, raw_frame: pd.DataFrame
    ) -> None:
        result = prepare_dataset(raw_frame, split="train")
        assert result[SAMPLE_ID_COLUMN].is_unique
        assert all(sid.startswith("train-") for sid in result[SAMPLE_ID_COLUMN])
        assert result[SAMPLE_ID_COLUMN].iloc[0] == "train-00000"

    def test_sample_id_tracks_the_original_row_position(self) -> None:
        raw = pd.DataFrame(
            {
                RAW_TEXT_COLUMN: ["first review", "   ", "third review"],
                RAW_LABEL_COLUMN: [0, 1, 1],
            }
        )
        result = prepare_dataset(raw, split="test")
        # Row 1 is dropped as empty; row 2 keeps its original index in its id.
        assert list(result[SAMPLE_ID_COLUMN]) == ["test-00000", "test-00002"]

    def test_drops_empty_reviews(self) -> None:
        raw = pd.DataFrame(
            {RAW_TEXT_COLUMN: ["good film", "", "  "], RAW_LABEL_COLUMN: [1, 0, 0]}
        )
        assert len(prepare_dataset(raw)) == 1

    def test_drops_duplicate_reviews_keeping_the_first(self) -> None:
        raw = pd.DataFrame(
            {
                RAW_TEXT_COLUMN: ["same text", "same text", "other"],
                RAW_LABEL_COLUMN: [1, 1, 0],
            }
        )
        result = prepare_dataset(raw)
        assert len(result) == 2
        assert list(result[SAMPLE_ID_COLUMN]) == ["test-00000", "test-00002"]

    def test_labels_are_normalised_strings(self, prepared: pd.DataFrame) -> None:
        assert set(prepared[LABEL_COLUMN]) == set(SENTIMENT_LABELS)

    def test_raises_when_nothing_usable_remains(self) -> None:
        raw = pd.DataFrame({RAW_TEXT_COLUMN: ["", "   "], RAW_LABEL_COLUMN: [0, 1]})
        with pytest.raises(DatasetError, match="No usable rows"):
            prepare_dataset(raw)

    def test_rejects_missing_columns(self) -> None:
        with pytest.raises(DatasetError, match="missing required column"):
            prepare_dataset(pd.DataFrame({"text": ["a"]}))


class TestValidatePreparedDataset:
    def test_accepts_a_prepared_frame(self, prepared: pd.DataFrame) -> None:
        validate_prepared_dataset(prepared)

    def test_rejects_duplicate_sample_ids(self, prepared: pd.DataFrame) -> None:
        broken = pd.concat([prepared.head(1), prepared.head(1)])
        with pytest.raises(DatasetError, match="Duplicate sample_id"):
            validate_prepared_dataset(broken)

    def test_rejects_empty_frame(self) -> None:
        empty = pd.DataFrame(columns=list(PREPARED_COLUMNS))
        with pytest.raises(DatasetError, match="empty"):
            validate_prepared_dataset(empty)

    def test_rejects_blank_review(self, prepared: pd.DataFrame) -> None:
        broken = prepared.copy()
        broken.loc[0, REVIEW_COLUMN] = "   "
        with pytest.raises(DatasetError, match="empty review"):
            validate_prepared_dataset(broken)


# ---------------------------------------------------------------------------
# create_fixed_evaluation_set — the reproducibility guarantee
# ---------------------------------------------------------------------------


class TestFixedEvaluationSet:
    def test_same_seed_gives_identical_samples(self, prepared: pd.DataFrame) -> None:
        first = create_fixed_evaluation_set(prepared, sample_size=6, seed=42)
        second = create_fixed_evaluation_set(prepared, sample_size=6, seed=42)
        pd.testing.assert_frame_equal(first, second)

    def test_every_strategy_would_receive_the_same_rows(
        self, prepared: pd.DataFrame
    ) -> None:
        # Simulates six strategies each asking for "the evaluation set".
        draws = [
            create_fixed_evaluation_set(prepared, sample_size=8, seed=42)
            for _ in range(6)
        ]
        reference = list(draws[0][SAMPLE_ID_COLUMN])
        assert all(list(draw[SAMPLE_ID_COLUMN]) == reference for draw in draws)

    def test_different_seed_can_give_different_samples(
        self, prepared: pd.DataFrame
    ) -> None:
        first = create_fixed_evaluation_set(prepared, sample_size=6, seed=1)
        second = create_fixed_evaluation_set(prepared, sample_size=6, seed=999)
        assert list(first[SAMPLE_ID_COLUMN]) != list(second[SAMPLE_ID_COLUMN])

    def test_requested_size_is_honoured(self, prepared: pd.DataFrame) -> None:
        assert len(create_fixed_evaluation_set(prepared, sample_size=7, seed=42)) == 7

    def test_stratified_draw_is_class_balanced(self, prepared: pd.DataFrame) -> None:
        result = create_fixed_evaluation_set(prepared, sample_size=8, seed=42)
        counts = result[LABEL_COLUMN].value_counts().to_dict()
        assert counts == {LABEL_NEGATIVE: 4, LABEL_POSITIVE: 4}

    def test_odd_size_distributes_remainder_deterministically(
        self, prepared: pd.DataFrame
    ) -> None:
        result = create_fixed_evaluation_set(prepared, sample_size=5, seed=42)
        counts = result[LABEL_COLUMN].value_counts().to_dict()
        # Remainder goes to the first canonical label (negative).
        assert counts == {LABEL_NEGATIVE: 3, LABEL_POSITIVE: 2}

    def test_rows_are_sorted_by_sample_id(self, prepared: pd.DataFrame) -> None:
        result = create_fixed_evaluation_set(prepared, sample_size=8, seed=42)
        assert list(result[SAMPLE_ID_COLUMN]) == sorted(result[SAMPLE_ID_COLUMN])

    def test_unstratified_draw_is_still_reproducible(
        self, prepared: pd.DataFrame
    ) -> None:
        first = create_fixed_evaluation_set(
            prepared, sample_size=6, seed=42, stratified=False
        )
        second = create_fixed_evaluation_set(
            prepared, sample_size=6, seed=42, stratified=False
        )
        pd.testing.assert_frame_equal(first, second)

    def test_output_satisfies_the_prepared_contract(
        self, prepared: pd.DataFrame
    ) -> None:
        result = create_fixed_evaluation_set(prepared, sample_size=6, seed=42)
        validate_prepared_dataset(result)
        assert list(result.columns) == list(PREPARED_COLUMNS)

    def test_rejects_oversized_request(self, prepared: pd.DataFrame) -> None:
        with pytest.raises(DatasetError, match="exceeds available rows"):
            create_fixed_evaluation_set(prepared, sample_size=len(prepared) + 1, seed=42)

    def test_rejects_non_positive_size(self, prepared: pd.DataFrame) -> None:
        with pytest.raises(DatasetError, match="sample_size must be >= 1"):
            create_fixed_evaluation_set(prepared, sample_size=0, seed=42)

    def test_reports_insufficient_class_support(self) -> None:
        raw = pd.DataFrame(
            {
                RAW_TEXT_COLUMN: ["a good film", "another good film", "a third good one"],
                RAW_LABEL_COLUMN: [1, 1, 1],
            }
        )
        prepared_single_class = prepare_dataset(raw)
        with pytest.raises(DatasetError, match="Cannot draw"):
            create_fixed_evaluation_set(prepared_single_class, sample_size=2, seed=42)


# ---------------------------------------------------------------------------
# create_few_shot_examples — fixed and leak-free
# ---------------------------------------------------------------------------


class TestFewShotExamples:
    def test_same_seed_gives_identical_demonstrations(
        self, prepared: pd.DataFrame
    ) -> None:
        first = create_few_shot_examples(prepared, examples_per_class=2, seed=42)
        second = create_few_shot_examples(prepared, examples_per_class=2, seed=42)
        assert first == second

    def test_is_class_balanced(self, prepared: pd.DataFrame) -> None:
        examples = create_few_shot_examples(prepared, examples_per_class=3, seed=42)
        labels = [example.label for example in examples]
        assert labels.count(LABEL_NEGATIVE) == 3
        assert labels.count(LABEL_POSITIVE) == 3

    def test_labels_alternate_so_the_last_class_is_not_fixed(
        self, prepared: pd.DataFrame
    ) -> None:
        examples = create_few_shot_examples(prepared, examples_per_class=2, seed=42)
        assert [example.label for example in examples] == [
            LABEL_NEGATIVE,
            LABEL_POSITIVE,
            LABEL_NEGATIVE,
            LABEL_POSITIVE,
        ]

    def test_excluded_evaluation_samples_never_appear(
        self, prepared: pd.DataFrame
    ) -> None:
        evaluation = create_fixed_evaluation_set(prepared, sample_size=8, seed=42)
        examples = create_few_shot_examples(
            prepared,
            examples_per_class=2,
            seed=42,
            exclude_sample_ids=evaluation[SAMPLE_ID_COLUMN],
        )
        evaluation_reviews = set(evaluation[REVIEW_COLUMN])
        assert all(example.review not in evaluation_reviews for example in examples)

    def test_zero_examples_returns_empty_list(self, prepared: pd.DataFrame) -> None:
        assert create_few_shot_examples(prepared, examples_per_class=0, seed=42) == []

    def test_truncates_long_demonstrations(self) -> None:
        long_text = "word " * 500
        raw = pd.DataFrame(
            {
                RAW_TEXT_COLUMN: [long_text + "good", long_text + "bad"],
                RAW_LABEL_COLUMN: [1, 0],
            }
        )
        examples = create_few_shot_examples(
            prepare_dataset(raw), examples_per_class=1, seed=42, max_review_chars=100
        )
        assert all(len(example.review) <= 110 for example in examples)
        assert all(example.review.endswith("...") for example in examples)

    def test_truncation_can_be_disabled(self, prepared: pd.DataFrame) -> None:
        examples = create_few_shot_examples(
            prepared, examples_per_class=1, seed=42, max_review_chars=None
        )
        assert all(not example.review.endswith("...") for example in examples)

    def test_rejects_negative_count(self, prepared: pd.DataFrame) -> None:
        with pytest.raises(DatasetError, match="examples_per_class"):
            create_few_shot_examples(prepared, examples_per_class=-1, seed=42)

    def test_reports_insufficient_eligible_rows(self, prepared: pd.DataFrame) -> None:
        with pytest.raises(DatasetError, match="Cannot draw"):
            create_few_shot_examples(prepared, examples_per_class=50, seed=42)


class TestFewShotExampleValue:
    def test_is_immutable(self) -> None:
        example = FewShotExample(review="great film", label=LABEL_POSITIVE)
        with pytest.raises(Exception):
            example.label = LABEL_NEGATIVE  # type: ignore[misc]

    def test_rejects_empty_review(self) -> None:
        with pytest.raises(DatasetError, match="must not be empty"):
            FewShotExample(review="   ", label=LABEL_POSITIVE)

    def test_rejects_invalid_label(self) -> None:
        with pytest.raises(DatasetError, match="label must be one of"):
            FewShotExample(review="great film", label="neutral")
