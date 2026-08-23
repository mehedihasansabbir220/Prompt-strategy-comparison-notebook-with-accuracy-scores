"""IMDb dataset loading, normalisation and reproducible sampling.

Responsibilities of this module:

* fetch the IMDb sentiment dataset from Hugging Face,
* normalise its integer labels to the project vocabulary (``negative`` / ``positive``),
* clean the raw review text into a tidy ``pandas`` DataFrame,
* draw **one fixed evaluation subset** that every prompt strategy is scored on,
* draw **fixed few-shot demonstrations** that never overlap that subset.

The reproducibility guarantee is the point of this module: given the same
``(source split, seed, sample_size)`` the selected rows are always identical, so
a difference between two strategies can never be an artefact of different data.

No prompts are rendered and no LLM is contacted here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from src.config import LABEL_NEGATIVE, LABEL_POSITIVE, SENTIMENT_LABELS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Hugging Face dataset identifier. The canonical ``namespace/name`` form is
#: required: the bare ``"imdb"`` alias is rejected by datasets >= 5.0.
IMDB_DATASET_NAME: Final[str] = "stanfordnlp/imdb"

#: Column names as they arrive from Hugging Face.
RAW_TEXT_COLUMN: Final[str] = "text"
RAW_LABEL_COLUMN: Final[str] = "label"

#: Column names used everywhere downstream.
SAMPLE_ID_COLUMN: Final[str] = "sample_id"
REVIEW_COLUMN: Final[str] = "review"
LABEL_COLUMN: Final[str] = "label"

#: Schema of a prepared dataset, in order.
PREPARED_COLUMNS: Final[tuple[str, ...]] = (
    SAMPLE_ID_COLUMN,
    REVIEW_COLUMN,
    LABEL_COLUMN,
)

#: IMDb's own integer encoding. Fixed by the dataset, not chosen by us.
IMDB_LABEL_MAP: Final[dict[int, str]] = {0: LABEL_NEGATIVE, 1: LABEL_POSITIVE}

#: Accepted textual spellings, so a re-labelled or third-party CSV also loads.
STRING_LABEL_ALIASES: Final[dict[str, str]] = {
    "neg": LABEL_NEGATIVE,
    "negative": LABEL_NEGATIVE,
    "0": LABEL_NEGATIVE,
    "pos": LABEL_POSITIVE,
    "positive": LABEL_POSITIVE,
    "1": LABEL_POSITIVE,
}

#: IMDb review text contains literal HTML line breaks; they are markup noise,
#: not sentiment signal, and would otherwise be spent as prompt tokens.
HTML_LINE_BREAK_PATTERN: Final[str] = r"<br\s*/?>"


class DatasetError(RuntimeError):
    """Raised when a dataset is missing columns, malformed, or too small."""


@dataclass(frozen=True, slots=True)
class FewShotExample:
    """One fixed demonstration used by few-shot style prompts.

    A small immutable value rather than a DataFrame row: the prompt layer needs
    exactly a review and its label, and freezing it prevents a strategy from
    quietly editing shared demonstrations at render time.
    """

    review: str
    label: str

    def __post_init__(self) -> None:
        if not self.review.strip():
            raise DatasetError("Few-shot example review must not be empty")
        if self.label not in SENTIMENT_LABELS:
            raise DatasetError(
                f"Few-shot example label must be one of {SENTIMENT_LABELS}, "
                f"got {self.label!r}"
            )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require_columns(frame: pd.DataFrame, required: Sequence[str], *, context: str) -> None:
    """Raise :class:`DatasetError` if ``frame`` is missing any required column."""
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DatasetError(
            f"{context}: missing required column(s) {missing}. "
            f"Found: {list(frame.columns)}"
        )


def validate_labels(labels: Iterable[Any]) -> None:
    """Raise :class:`DatasetError` unless every label is a valid sentiment class.

    Called after normalisation. A label outside the vocabulary means predictions
    would be scored against a class the model was never asked to produce.
    """
    unexpected = sorted({str(label) for label in labels} - set(SENTIMENT_LABELS))
    if unexpected:
        raise DatasetError(
            f"Unexpected label(s) {unexpected}; expected only {list(SENTIMENT_LABELS)}"
        )


def validate_prepared_dataset(frame: pd.DataFrame) -> None:
    """Assert that ``frame`` satisfies the prepared-dataset contract.

    Checks the schema, unique sample ids, non-empty reviews and valid labels —
    the four assumptions every later stage (prompting, scoring, error analysis)
    silently relies on.
    """
    _require_columns(frame, PREPARED_COLUMNS, context="Prepared dataset")

    if frame.empty:
        raise DatasetError("Prepared dataset is empty")
    if frame[SAMPLE_ID_COLUMN].duplicated().any():
        duplicates = frame.loc[frame[SAMPLE_ID_COLUMN].duplicated(), SAMPLE_ID_COLUMN]
        raise DatasetError(f"Duplicate sample_id values: {sorted(set(duplicates))[:5]}")
    if frame[REVIEW_COLUMN].isna().any() or (frame[REVIEW_COLUMN].str.strip() == "").any():
        raise DatasetError("Prepared dataset contains empty review text")

    validate_labels(frame[LABEL_COLUMN])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_imdb_dataset(
    split: str = "test",
    *,
    cache_dir: Path | str | None = None,
    dataset_name: str = IMDB_DATASET_NAME,
) -> pd.DataFrame:
    """Load one IMDb split from Hugging Face as a raw DataFrame.

    Args:
        split: ``"test"`` for evaluation samples, ``"train"`` for few-shot
            demonstrations. Keeping the two sources separate is what stops a
            demonstration from also being a graded sample.
        cache_dir: Optional Hugging Face cache location. ``None`` uses the
            default cache, so the download happens only once per machine.
        dataset_name: Overridable for tests and for drop-in alternative corpora.

    Returns:
        A DataFrame with the raw ``text`` and ``label`` columns, in dataset order.

    Raises:
        DatasetError: If the download fails or the expected columns are absent.
    """
    # Imported lazily so the rest of the module (and its tests) never pay the
    # cost of importing `datasets`, and so offline unit tests stay fast.
    from datasets import load_dataset

    logger.info("Loading dataset %r split=%r", dataset_name, split)
    try:
        raw = load_dataset(dataset_name, split=split, cache_dir=str(cache_dir) if cache_dir else None)
    except Exception as exc:  # noqa: BLE001 - surfaced with actionable context
        raise DatasetError(
            f"Failed to load dataset {dataset_name!r} split {split!r}: {exc}"
        ) from exc

    frame = raw.to_pandas()
    _require_columns(
        frame,
        (RAW_TEXT_COLUMN, RAW_LABEL_COLUMN),
        context=f"Raw dataset {dataset_name!r} split {split!r}",
    )
    logger.info("Loaded %d rows from %r split=%r", len(frame), dataset_name, split)
    return frame


# ---------------------------------------------------------------------------
# Normalisation and preparation
# ---------------------------------------------------------------------------


def normalize_labels(
    frame: pd.DataFrame,
    *,
    label_column: str = RAW_LABEL_COLUMN,
    target_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    """Map raw labels onto the project vocabulary ``negative`` / ``positive``.

    Handles IMDb's integers (0/1) and common textual spellings (``pos``/``neg``),
    so ground truth is expressed in exactly the words the model is asked to
    produce — no integer-to-string translation is needed at scoring time.

    Args:
        frame: Source DataFrame; never mutated.
        label_column: Column holding the raw labels.
        target_column: Column to write normalised labels into.

    Returns:
        A copy with ``target_column`` holding validated string labels.

    Raises:
        DatasetError: If a value cannot be mapped to a known class.
    """
    _require_columns(frame, (label_column,), context="normalize_labels")

    def _normalize(value: Any) -> str:
        if pd.isna(value):
            raise DatasetError("Label column contains missing values")
        if isinstance(value, bool):  # bool is an int subclass; reject it explicitly
            raise DatasetError(f"Unsupported boolean label {value!r}")
        if isinstance(value, (int, float)) and float(value).is_integer():
            mapped = IMDB_LABEL_MAP.get(int(value))
            if mapped is None:
                raise DatasetError(
                    f"Unknown numeric label {value!r}; expected one of "
                    f"{sorted(IMDB_LABEL_MAP)}"
                )
            return mapped
        mapped = STRING_LABEL_ALIASES.get(str(value).strip().lower())
        if mapped is None:
            raise DatasetError(
                f"Unknown label {value!r}; expected one of "
                f"{sorted(set(STRING_LABEL_ALIASES) | set(map(str, IMDB_LABEL_MAP)))}"
            )
        return mapped

    result = frame.copy()
    result[target_column] = [_normalize(value) for value in frame[label_column]]
    validate_labels(result[target_column])
    return result


def _clean_review_text(series: pd.Series) -> pd.Series:
    """Strip HTML line breaks and collapse whitespace in review text."""
    return (
        series.astype("string")
        .str.replace(HTML_LINE_BREAK_PATTERN, " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def prepare_dataset(
    frame: pd.DataFrame,
    *,
    split: str = "test",
    text_column: str = RAW_TEXT_COLUMN,
    label_column: str = RAW_LABEL_COLUMN,
) -> pd.DataFrame:
    """Turn a raw split into the tidy schema used by the rest of the project.

    Steps, in order: validate the raw schema, normalise labels, clean the text,
    drop empty reviews, drop duplicate reviews, assign a stable ``sample_id``.

    ``sample_id`` is ``"<split>-<original row index>"`` (e.g. ``test-00042``).
    Deriving it from the source position rather than from post-filter ordering
    means an error row in ``errors.csv`` can always be traced back to the exact
    IMDb record, and ids stay stable if filtering changes later.

    Args:
        frame: Raw DataFrame as returned by :func:`load_imdb_dataset`.
        split: Split name, used as the ``sample_id`` prefix.
        text_column: Name of the raw review-text column.
        label_column: Name of the raw label column.

    Returns:
        A DataFrame with columns ``(sample_id, review, label)`` and a clean
        ``RangeIndex``.

    Raises:
        DatasetError: If the raw schema is wrong, a label is unmappable, or no
            usable rows remain.
    """
    _require_columns(frame, (text_column, label_column), context="prepare_dataset")

    working = normalize_labels(frame, label_column=label_column, target_column=LABEL_COLUMN)
    working[REVIEW_COLUMN] = _clean_review_text(working[text_column])
    working[SAMPLE_ID_COLUMN] = [f"{split}-{position:05d}" for position in range(len(working))]

    before = len(working)
    working = working[working[REVIEW_COLUMN].notna() & (working[REVIEW_COLUMN] != "")]
    working = working.drop_duplicates(subset=REVIEW_COLUMN, keep="first")
    dropped = before - len(working)
    if dropped:
        logger.info("Dropped %d empty or duplicate review(s) from split=%r", dropped, split)

    if working.empty:
        raise DatasetError(f"No usable rows remain in split {split!r} after cleaning")

    prepared = working[list(PREPARED_COLUMNS)].reset_index(drop=True)
    prepared[REVIEW_COLUMN] = prepared[REVIEW_COLUMN].astype(str)
    validate_prepared_dataset(prepared)
    logger.info("Prepared %d rows for split=%r", len(prepared), split)
    return prepared


# ---------------------------------------------------------------------------
# Reproducible sampling
# ---------------------------------------------------------------------------


def _stratified_counts(sample_size: int, labels: Sequence[str]) -> dict[str, int]:
    """Split ``sample_size`` across classes as evenly as possible.

    Any remainder is handed out in canonical label order, so the allocation is
    deterministic rather than dependent on dictionary or sampling order.
    """
    base, remainder = divmod(sample_size, len(labels))
    return {
        label: base + (1 if index < remainder else 0)
        for index, label in enumerate(labels)
    }


def create_fixed_evaluation_set(
    frame: pd.DataFrame,
    *,
    sample_size: int,
    seed: int,
    stratified: bool = True,
) -> pd.DataFrame:
    """Draw the single evaluation subset that every strategy is scored on.

    Determinism comes from ``pandas.DataFrame.sample(random_state=seed)`` over a
    fixed input ordering: the same ``(frame, sample_size, seed)`` always yields
    the same rows. The result is sorted by ``sample_id`` so row *order* is also
    stable, which keeps ``predictions.csv`` diffable between runs.

    Args:
        frame: Prepared dataset (see :func:`prepare_dataset`).
        sample_size: Number of reviews to evaluate.
        seed: Random seed; recorded in experiment metadata.
        stratified: If ``True`` (default), draw an equal number per class so
            accuracy cannot be inflated by an unbalanced draw and per-class
            recall is computed on a comparable support.

    Returns:
        A DataFrame with the prepared schema, sorted by ``sample_id``.

    Raises:
        DatasetError: If ``sample_size`` is not positive, exceeds the available
            rows, or a class holds too few rows for a stratified draw.
    """
    validate_prepared_dataset(frame)

    if sample_size < 1:
        raise DatasetError(f"sample_size must be >= 1, got {sample_size}")
    if sample_size > len(frame):
        raise DatasetError(
            f"sample_size {sample_size} exceeds available rows ({len(frame)})"
        )

    if not stratified:
        selection = frame.sample(n=sample_size, random_state=seed)
    else:
        wanted = _stratified_counts(sample_size, SENTIMENT_LABELS)
        chunks: list[pd.DataFrame] = []
        for label, count in wanted.items():
            pool = frame[frame[LABEL_COLUMN] == label]
            if len(pool) < count:
                raise DatasetError(
                    f"Cannot draw {count} {label!r} sample(s): only {len(pool)} available"
                )
            if count:
                chunks.append(pool.sample(n=count, random_state=seed))
        selection = pd.concat(chunks)

    evaluation_set = (
        selection.sort_values(SAMPLE_ID_COLUMN).reset_index(drop=True)
    )
    validate_prepared_dataset(evaluation_set)
    logger.info(
        "Fixed evaluation set: %d samples (seed=%d, stratified=%s) | class counts: %s",
        len(evaluation_set),
        seed,
        stratified,
        evaluation_set[LABEL_COLUMN].value_counts().to_dict(),
    )
    return evaluation_set


def create_few_shot_examples(
    frame: pd.DataFrame,
    *,
    examples_per_class: int = 2,
    seed: int,
    exclude_sample_ids: Iterable[str] = (),
    max_review_chars: int | None = 600,
) -> list[FewShotExample]:
    """Select the fixed demonstrations shared by all few-shot style prompts.

    Three properties matter for a fair experiment:

    * **Fixed.** The same seed returns the same demonstrations, so few-shot and
      combined strategies are never advantaged by a luckier draw.
    * **Disjoint from evaluation.** ``exclude_sample_ids`` removes graded samples
      from the pool, so no strategy can see a test review inside its own prompt.
    * **Class-balanced and alternating.** Equal counts per class, returned in
      alternating label order, so the final demonstration is not always the same
      class — which would bias a model toward that label.

    Args:
        frame: Prepared dataset, normally the ``train`` split.
        examples_per_class: Demonstrations per sentiment class.
        seed: Random seed for the draw.
        exclude_sample_ids: Sample ids to keep out of the pool (the evaluation set).
        max_review_chars: Truncate long demonstrations to keep prompt length —
            and therefore cost — comparable across strategies. ``None`` disables
            truncation. Only demonstrations are truncated, never graded reviews.

    Returns:
        A list of :class:`FewShotExample`, alternating between classes.

    Raises:
        DatasetError: If ``examples_per_class`` is negative or a class lacks
            enough eligible rows.
    """
    validate_prepared_dataset(frame)

    if examples_per_class < 0:
        raise DatasetError(
            f"examples_per_class must be >= 0, got {examples_per_class}"
        )
    if examples_per_class == 0:
        return []
    if max_review_chars is not None and max_review_chars < 1:
        raise DatasetError(
            f"max_review_chars must be >= 1 or None, got {max_review_chars}"
        )

    excluded = set(exclude_sample_ids)
    pool = frame[~frame[SAMPLE_ID_COLUMN].isin(excluded)]
    if excluded:
        logger.info("Excluded %d evaluation sample(s) from the few-shot pool", len(excluded))

    per_class: dict[str, list[FewShotExample]] = {}
    for label in SENTIMENT_LABELS:
        candidates = pool[pool[LABEL_COLUMN] == label]
        if len(candidates) < examples_per_class:
            raise DatasetError(
                f"Cannot draw {examples_per_class} {label!r} demonstration(s): "
                f"only {len(candidates)} eligible row(s) available"
            )
        chosen = candidates.sample(n=examples_per_class, random_state=seed).sort_values(
            SAMPLE_ID_COLUMN
        )
        per_class[label] = [
            FewShotExample(review=_truncate(row.review, max_review_chars), label=label)
            for row in chosen.itertuples()
        ]

    # Interleave: negative, positive, negative, positive, ...
    examples: list[FewShotExample] = []
    for index in range(examples_per_class):
        for label in SENTIMENT_LABELS:
            examples.append(per_class[label][index])

    logger.info(
        "Fixed few-shot set: %d demonstration(s) (%d per class, seed=%d)",
        len(examples),
        examples_per_class,
        seed,
    )
    return examples


def _truncate(text: str, max_chars: int | None) -> str:
    """Shorten ``text`` at a word boundary, marking that it was cut."""
    if max_chars is None or len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit(" ", 1)[0].rstrip()
    return f"{clipped} ..."
