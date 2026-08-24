"""Immutable, versioned storage for benchmark runs.

Each run owns a directory named by its experiment id::

    results/experiments/2026-08-24_001/
    ├── config.json
    ├── predictions.csv
    ├── metrics.csv
    └── summary.json

The central guarantee is that **a completed experiment is never overwritten**.
Ids are allocated by scanning what already exists, and creating a directory that
is already there is an error rather than a silent replacement — a benchmark
result that can be quietly clobbered is not evidence.

No LLM is contacted here.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date as date_type
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd

logger = logging.getLogger(__name__)

#: ``YYYY-MM-DD_NNN`` — sortable, human-readable, and unique per day.
EXPERIMENT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(\d{4}-\d{2}-\d{2})_(\d{3})$"
)

CONFIG_FILENAME: Final[str] = "config.json"
PREDICTIONS_FILENAME: Final[str] = "predictions.csv"
METRICS_FILENAME: Final[str] = "metrics.csv"
SUMMARY_FILENAME: Final[str] = "summary.json"


class StorageError(RuntimeError):
    """Raised when a run cannot be stored or read back."""


class ExperimentStorage:
    """Filesystem store for experiment runs, rooted at ``experiments/``."""

    def __init__(self, base_dir: Path | str) -> None:
        """Args:
        base_dir: Root directory holding one subdirectory per experiment,
            normally ``<results_dir>/experiments``. Created on demand.
        """
        self.base_dir = Path(base_dir)

    # -- identifiers --------------------------------------------------------

    def list_experiments(self) -> list[str]:
        """Return existing experiment ids, oldest first.

        Only well-formed ids are returned; unrelated directories are ignored
        so a stray folder cannot corrupt id allocation.
        """
        if not self.base_dir.exists():
            return []
        return sorted(
            entry.name
            for entry in self.base_dir.iterdir()
            if entry.is_dir() and EXPERIMENT_ID_PATTERN.match(entry.name)
        )

    def next_experiment_id(self, *, today: date_type | None = None) -> str:
        """Allocate the next unused id for the given day.

        Counts up from the highest sequence already present for that date, so
        an id is never reissued even if earlier runs were deleted out of order.
        """
        day = (today or datetime.now().date()).isoformat()
        used = [
            int(match.group(2))
            for name in self.list_experiments()
            if (match := EXPERIMENT_ID_PATTERN.match(name)) and match.group(1) == day
        ]
        return f"{day}_{max(used, default=0) + 1:03d}"

    def experiment_path(self, experiment_id: str) -> Path:
        """Return the directory for ``experiment_id`` without creating it."""
        if not EXPERIMENT_ID_PATTERN.match(experiment_id):
            raise StorageError(
                f"Malformed experiment id {experiment_id!r}; expected YYYY-MM-DD_NNN"
            )
        return self.base_dir / experiment_id

    # -- writing ------------------------------------------------------------

    def create_experiment(
        self, experiment_id: str | None = None, *, today: date_type | None = None
    ) -> str:
        """Create a fresh experiment directory and return its id.

        Raises:
            StorageError: If the directory already exists. Reusing an id would
                mix two runs' outputs in one folder.
        """
        new_id = experiment_id or self.next_experiment_id(today=today)
        path = self.experiment_path(new_id)
        if path.exists():
            raise StorageError(
                f"Experiment {new_id!r} already exists at {path}; "
                "completed experiments are never overwritten"
            )
        path.mkdir(parents=True)
        logger.info("Created experiment directory %s", path)
        return new_id

    def save_config(self, experiment_id: str, config: dict[str, Any]) -> Path:
        """Write ``config.json`` — the metadata that makes the run reproducible."""
        path = self._require_experiment(experiment_id) / CONFIG_FILENAME
        try:
            payload = json.dumps(config, indent=2, sort_keys=False, default=str)
        except (TypeError, ValueError) as exc:
            raise StorageError(f"Experiment config is not serialisable: {exc}") from exc
        path.write_text(payload, encoding="utf-8")
        logger.info("Saved %s", path)
        return path

    def save_predictions(self, experiment_id: str, predictions: pd.DataFrame) -> Path:
        """Write ``predictions.csv`` — one row per (strategy, sample)."""
        if predictions.empty:
            raise StorageError("Refusing to save an empty predictions frame")
        path = self._require_experiment(experiment_id) / PREDICTIONS_FILENAME
        predictions.to_csv(path, index=False)
        logger.info("Saved %s (%d rows)", path, len(predictions))
        return path

    def save_metrics(self, experiment_id: str, metrics: pd.DataFrame) -> Path:
        """Write ``metrics.csv`` — one scored row per strategy."""
        if metrics.empty:
            raise StorageError("Refusing to save an empty metrics frame")
        path = self._require_experiment(experiment_id) / METRICS_FILENAME
        metrics.to_csv(path, index=False)
        logger.info("Saved %s (%d strategies)", path, len(metrics))
        return path

    def save_summary(self, experiment_id: str, summary: dict[str, Any]) -> Path:
        """Write ``summary.json`` — the headline result of the run."""
        path = self._require_experiment(experiment_id) / SUMMARY_FILENAME
        try:
            payload = json.dumps(summary, indent=2, default=str)
        except (TypeError, ValueError) as exc:
            raise StorageError(f"Experiment summary is not serialisable: {exc}") from exc
        path.write_text(payload, encoding="utf-8")
        logger.info("Saved %s", path)
        return path

    # -- reading ------------------------------------------------------------

    def load_config(self, experiment_id: str) -> dict[str, Any]:
        """Read back a stored ``config.json``."""
        path = self._require_experiment(experiment_id) / CONFIG_FILENAME
        if not path.exists():
            raise StorageError(f"No {CONFIG_FILENAME} in experiment {experiment_id!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load_predictions(self, experiment_id: str) -> pd.DataFrame:
        """Read back a stored ``predictions.csv``.

        ``keep_default_na=False`` preserves empty raw responses as empty strings
        rather than turning them into NaN, so a failed call round-trips exactly
        as it was recorded.
        """
        path = self._require_experiment(experiment_id) / PREDICTIONS_FILENAME
        if not path.exists():
            raise StorageError(
                f"No {PREDICTIONS_FILENAME} in experiment {experiment_id!r}"
            )
        return pd.read_csv(path, keep_default_na=False, na_values=[])

    def load_metrics(self, experiment_id: str) -> pd.DataFrame:
        """Read back a stored ``metrics.csv``."""
        path = self._require_experiment(experiment_id) / METRICS_FILENAME
        if not path.exists():
            raise StorageError(f"No {METRICS_FILENAME} in experiment {experiment_id!r}")
        return pd.read_csv(path)

    def load_summary(self, experiment_id: str) -> dict[str, Any]:
        """Read back a stored ``summary.json``."""
        path = self._require_experiment(experiment_id) / SUMMARY_FILENAME
        if not path.exists():
            raise StorageError(f"No {SUMMARY_FILENAME} in experiment {experiment_id!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    # -- internals ----------------------------------------------------------

    def _require_experiment(self, experiment_id: str) -> Path:
        path = self.experiment_path(experiment_id)
        if not path.exists():
            raise StorageError(
                f"Experiment {experiment_id!r} does not exist at {path}; "
                "call create_experiment() first"
            )
        return path
