"""Shared helpers for the PromptBench Streamlit app.

Kept out of ``streamlit_app.py`` so the page file stays readable: this module
holds the caching, formatting and small presentational pieces, and the page
file holds the layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.config import AppConfig, has_api_key
from src.dataset import (
    create_few_shot_examples,
    create_fixed_evaluation_set,
    load_imdb_dataset,
    prepare_dataset,
)
from src.experiments.storage import ExperimentStorage

#: Cost per chart of a benchmark, used only for the pre-run estimate shown to
#: the user. Measured from real calls on 2026-08-24; clearly labelled as an
#: estimate everywhere it appears, and never mixed into a result.
SECONDS_PER_CALL_ESTIMATE: float = 1.4


@st.cache_data(show_spinner="Loading IMDb from Hugging Face…")
def load_split(split: str) -> pd.DataFrame:
    """Load and prepare one IMDb split. Cached for the session."""
    return prepare_dataset(load_imdb_dataset(split), split=split)


@st.cache_data(show_spinner=False)
def build_evaluation_set(split: str, sample_size: int, seed: int) -> pd.DataFrame:
    """Draw the fixed evaluation subset. Same inputs always give the same rows."""
    return create_fixed_evaluation_set(
        load_split(split), sample_size=sample_size, seed=seed
    )


@st.cache_data(show_spinner=False)
def build_examples(
    seed: int, examples_per_class: int, excluded: tuple[str, ...]
) -> list[Any]:
    """Draw the fixed few-shot demonstrations, excluding evaluation samples."""
    return create_few_shot_examples(
        load_split("train"),
        examples_per_class=examples_per_class,
        seed=seed,
        exclude_sample_ids=excluded,
    )


def get_storage(config: AppConfig) -> ExperimentStorage:
    """Return the experiment store for the configured results directory."""
    return ExperimentStorage(config.experiments_dir)


def api_key_banner() -> bool:
    """Render the credential status. Returns whether a key is usable."""
    if has_api_key():
        st.success("Gemini API key detected", icon=":material/key:")
        return True
    st.error(
        "No usable `GEMINI_API_KEY`. Copy `.env.example` to `.env` and add your key "
        "from https://aistudio.google.com/apikey — the dataset and prompt tabs work "
        "without it, but no model can be called.",
        icon=":material/key_off:",
    )
    return False


def metric_row(entries: list[tuple[str, str, str | None]]) -> None:
    """Render a row of headline metrics as ``(label, value, help)`` tuples."""
    columns = st.columns(len(entries))
    for column, (label, value, helptext) in zip(columns, entries, strict=True):
        column.metric(label, value, help=helptext)


def format_percent(value: float | None) -> str:
    """Format a rate for display, marking a missing value honestly."""
    return "—" if value is None or pd.isna(value) else f"{float(value):.1%}"


def strategy_label(name: str) -> str:
    """Human-readable strategy name, matching the charts."""
    return str(name).replace("_", "-").capitalize()


def download_button_for(
    frame: pd.DataFrame, label: str, filename: str, *, key: str | None = None
) -> None:
    """Offer a DataFrame as a CSV download.

    ``key`` must be unique when the same button is rendered from more than one
    place on the page.
    """
    st.download_button(
        label,
        frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        width="stretch",
        key=key,
    )


def estimate_runtime(strategy_count: int, sample_count: int) -> tuple[int, str]:
    """Return the call count and a rough wall-clock estimate for a planned run."""
    calls = strategy_count * sample_count
    seconds = calls * SECONDS_PER_CALL_ESTIMATE
    if seconds < 90:
        return calls, f"~{seconds:.0f} seconds"
    return calls, f"~{seconds / 60:.0f} minutes"


def experiment_dir_for(storage: ExperimentStorage, experiment_id: str) -> Path:
    return storage.experiment_path(experiment_id)
