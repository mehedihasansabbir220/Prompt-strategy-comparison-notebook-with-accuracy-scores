"""PromptBench — interactive front end.

A local dashboard over the same ``src/`` modules the notebook and CLI use. It
adds no evaluation logic of its own: prompts, provider calls, parsing, metrics
and charts all come from the library, so what you see here is exactly what a
scripted run produces.

Launch with::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow `streamlit run app/streamlit_app.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ui import (  # noqa: E402
    api_key_banner,
    build_evaluation_set,
    build_examples,
    download_button_for,
    estimate_runtime,
    format_percent,
    get_storage,
    load_split,
    metric_row,
    strategy_label,
)
from src.config import AppConfig, PromptStrategy  # noqa: E402
from src.dataset import DatasetError  # noqa: E402
from src.evaluation.errors import (  # noqa: E402
    error_analysis_summary,
    extract_errors,
    find_shared_errors,
    summarize_errors_by_strategy,
)
from src.experiments.runner import BenchmarkRunner  # noqa: E402
from src.experiments.storage import StorageError  # noqa: E402
from src.llm.base import LLMAuthError  # noqa: E402
from src.llm.gemini import GeminiProvider  # noqa: E402
from src.prompts import PromptContext, describe_strategies, get_strategy  # noqa: E402
from src.utils.parsing import parse_sentiment_response  # noqa: E402
from src.visualization.charts import (  # noqa: E402
    plot_accuracy_comparison,
    plot_confusion_matrices,
    plot_error_rate_comparison,
    plot_latency_comparison,
    plot_metric_comparison,
)

st.set_page_config(
    page_title="PromptBench",
    page_icon=":material/science:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1400px; }
      h1 { font-size: 2.1rem !important; letter-spacing: -0.02em; }
      .pb-sub { color: #52514e; font-size: 1.02rem; margin-top: -0.6rem; }
      .pb-card { border: 1px solid #e8e7e3; border-radius: 10px;
                 padding: 0.9rem 1.1rem; background: #fcfcfb; height: 100%; }
      .pb-card h4 { margin: 0 0 0.35rem 0; font-size: 0.98rem; }
      .pb-card p { margin: 0.2rem 0; font-size: 0.86rem; color: #52514e; }
      .pb-tag { display:inline-block; font-size:0.72rem; letter-spacing:0.04em;
                text-transform:uppercase; color:#2a78d6; font-weight:600; }
      div[data-testid="stMetricValue"] { font-size: 1.7rem; }
      pre { font-size: 0.82rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

for key, default in (
    ("result", None),
    ("evaluation", None),
    ("single", None),
):
    st.session_state.setdefault(key, default)


# ---------------------------------------------------------------------------
# Sidebar — the controlled variables
# ---------------------------------------------------------------------------

base_config = AppConfig.from_env()

with st.sidebar:
    st.markdown("### Experiment settings")
    st.caption("Everything here is held constant across strategies.")

    model = st.text_input("Model", value=base_config.model)
    temperature = st.slider(
        "Temperature", 0.0, 2.0, float(base_config.temperature), 0.1,
        help="0.0 gives the most repeatable output the provider offers.",
    )
    sample_size = st.slider(
        "Evaluation samples", 4, 200, int(base_config.dev_sample_size), 2,
        help="The same fixed subset is used by every strategy.",
    )
    seed = st.number_input(
        "Random seed", value=int(base_config.random_seed), step=1,
        help="Same seed + same size always selects the same reviews.",
    )
    examples_per_class = st.slider(
        "Few-shot examples per class", 1, 4, 2,
        help="Drawn from the train split, never from the evaluation set.",
    )

    st.divider()
    st.markdown("### Strategies")
    selected = st.multiselect(
        "Compare",
        options=[str(member) for member in PromptStrategy],
        default=[str(member) for member in PromptStrategy],
        format_func=strategy_label,
        label_visibility="collapsed",
    )

    st.divider()
    key_available = api_key_banner()

config = base_config.with_overrides(
    model=model.strip() or base_config.model,
    temperature=temperature,
    benchmark_sample_size=int(sample_size),
    random_seed=int(seed),
)
storage = get_storage(config)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("PromptBench")
st.markdown(
    '<p class="pb-sub">An empirical study of prompt strategies for LLM sentiment '
    "classification — same model, same reviews, same settings; only the prompt changes.</p>",
    unsafe_allow_html=True,
)

overview_tab, prompt_tab, run_tab, results_tab, history_tab = st.tabs(
    ["Overview", "Prompt explorer", "Run benchmark", "Results", "Past experiments"]
)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

with overview_tab:
    st.subheader("Research question")
    st.info(
        "How much can prompt strategy affect an LLM's performance on sentiment "
        "classification when the model, dataset, test samples and evaluation "
        "methodology all remain controlled?",
        icon=":material/help:",
    )

    left, right = st.columns([2, 1])
    with left:
        st.markdown("#### Method")
        st.markdown(
            """
            1. Load IMDb and normalise labels to `positive` / `negative`.
            2. Draw **one fixed, seeded** evaluation subset — every strategy is scored on
               exactly these reviews.
            3. Render each strategy's prompt and call the model with **identical**
               generation settings.
            4. Parse every response with **one shared parser** into
               `positive` / `negative` / `unknown`.
            5. Score accuracy, precision, recall, F1 and a confusion matrix.
            6. Persist predictions, metrics, errors and metadata under an immutable
               experiment id.
            """
        )
    with right:
        st.markdown("#### Held constant")
        st.markdown(
            f"""
            - Model — `{config.model}`
            - Temperature — `{config.temperature}`
            - Seed — `{config.random_seed}`
            - Samples — `{config.benchmark_sample_size}`
            - Parser and metrics
            """
        )

    st.divider()
    st.subheader("Prompt strategies")
    st.caption(
        "Each hypothesis was written before any benchmark was run, so results can be "
        "read against a prediction rather than explained after the fact."
    )

    described = describe_strategies()
    for row_start in range(0, len(described), 3):
        for column, entry in zip(
            st.columns(3), described[row_start : row_start + 3], strict=False
        ):
            with column:
                st.markdown(
                    f"""<div class="pb-card">
                        <span class="pb-tag">Strategy</span>
                        <h4>{strategy_label(entry['name'])}</h4>
                        <p><b>What it does.</b> {entry['description']}</p>
                        <p><b>Hypothesis.</b> {entry['hypothesis']}</p>
                    </div>""",
                    unsafe_allow_html=True,
                )
        st.write("")


# ---------------------------------------------------------------------------
# Prompt explorer — single review, single call
# ---------------------------------------------------------------------------

with prompt_tab:
    st.subheader("Inspect a prompt, then classify one review")
    st.caption(
        "The cheapest way to see the difference between strategies: one review, "
        "one call, the exact prompt that was sent and the raw text that came back."
    )

    chooser, viewer = st.columns([1, 2])

    with chooser:
        strategy_name = st.selectbox(
            "Strategy", options=[str(member) for member in PromptStrategy],
            format_func=strategy_label,
        )
        source = st.radio(
            "Review source", ["From the evaluation set", "Type my own"], index=0
        )

        review_text = ""
        actual_label: str | None = None
        if source == "From the evaluation set":
            try:
                evaluation_set = build_evaluation_set(
                    "test", int(config.benchmark_sample_size), int(config.random_seed)
                )
                options = evaluation_set["sample_id"].tolist()
                chosen = st.selectbox("Sample", options)
                row = evaluation_set[evaluation_set["sample_id"] == chosen].iloc[0]
                review_text, actual_label = row["review"], row["label"]
                st.caption(f"True label: **{actual_label}**")
            except DatasetError as error:
                st.error(f"Could not load the dataset: {error}")
        else:
            review_text = st.text_area(
                "Review", height=180,
                value="The pacing dragged in the middle, but the final act completely won me over.",
            )

        run_single = st.button(
            "Classify this review", type="primary", width="stretch",
            disabled=not (key_available and review_text.strip()),
        )

    with viewer:
        if review_text.strip():
            try:
                examples = tuple(
                    build_examples(int(config.random_seed), int(examples_per_class), ())
                )
                context = PromptContext(few_shot_examples=examples)
                prompt = get_strategy(strategy_name).build_prompt(review_text, context)

                st.markdown("**Rendered prompt**")
                if prompt.system_instruction:
                    with st.expander("System instruction", expanded=False):
                        st.code(prompt.system_instruction, language="text")
                st.code(prompt.user_text, language="text")
                st.caption(f"{prompt.char_count:,} characters sent")
            except Exception as error:  # noqa: BLE001 - surfaced to the user
                st.error(f"Could not build the prompt: {error}")

    if run_single:
        with st.spinner("Calling Gemini…"):
            try:
                provider = GeminiProvider(config)
                response = provider.generate(prompt)
                st.session_state.single = {
                    "response": response,
                    "parsed": parse_sentiment_response(response.text),
                    "actual": actual_label,
                }
            except LLMAuthError as error:
                st.error(f"Authentication failed: {error}")

    single = st.session_state.single
    if single:
        st.divider()
        response, parsed, actual = single["response"], single["parsed"], single["actual"]

        verdict = "—"
        if actual:
            verdict = "correct" if parsed == actual else "wrong"
        metric_row(
            [
                ("Prediction", parsed, "Parsed from the raw response"),
                ("True label", actual or "—", None),
                ("Outcome", verdict, None),
                ("Latency", f"{response.latency_seconds:.2f}s", None),
                (
                    "Tokens",
                    str(response.usage.total_tokens or "—"),
                    "As reported by the provider",
                ),
            ]
        )
        st.markdown("**Raw model response**")
        st.code(response.text or "(empty)", language="text")
        if not response.success:
            st.warning(f"{response.error_type}: {response.error_message}")


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------

with run_tab:
    st.subheader("Run the benchmark")

    if not selected:
        st.warning("Select at least one strategy in the sidebar.")
    else:
        calls, estimate = estimate_runtime(len(selected), int(config.benchmark_sample_size))
        metric_row(
            [
                ("Strategies", str(len(selected)), None),
                ("Samples each", str(config.benchmark_sample_size), None),
                ("Total API calls", f"{calls:,}", None),
                ("Estimated time", estimate, "Rough estimate, not a measurement"),
            ]
        )
        st.caption(
            "Every strategy is scored on the identical fixed subset, so a difference "
            "between them cannot come from the data."
        )

        start = st.button(
            f"Run {calls:,} calls", type="primary", disabled=not key_available
        )

        if start:
            progress = st.progress(0.0, text="Preparing…")
            status = st.empty()
            try:
                evaluation_set = build_evaluation_set(
                    "test", int(config.benchmark_sample_size), int(config.random_seed)
                )
                examples = tuple(
                    build_examples(
                        int(config.random_seed),
                        int(examples_per_class),
                        tuple(evaluation_set["sample_id"]),
                    )
                )

                total_calls = len(selected) * len(evaluation_set)
                done = {"count": 0}
                started = time.perf_counter()

                def on_progress(strategy: str, completed: int, total: int) -> None:
                    done["count"] += 1
                    fraction = done["count"] / total_calls
                    elapsed = time.perf_counter() - started
                    progress.progress(
                        min(fraction, 1.0),
                        text=(
                            f"{strategy_label(strategy)} — {completed}/{total} "
                            f"· {done['count']}/{total_calls} calls · {elapsed:.0f}s elapsed"
                        ),
                    )

                runner = BenchmarkRunner(GeminiProvider(config), config, storage=storage)
                result = runner.run(
                    evaluation_set,
                    selected,
                    examples,
                    show_progress=False,
                    progress_callback=on_progress,
                )
                evaluation = runner.evaluate(result)

                st.session_state.result = result
                st.session_state.evaluation = evaluation
                progress.progress(1.0, text="Complete")
                status.success(
                    f"Experiment `{result.experiment_id}` finished in "
                    f"{result.runtime_seconds:.0f}s — open the **Results** tab.",
                    icon=":material/check_circle:",
                )
            except LLMAuthError as error:
                progress.empty()
                status.error(f"Authentication failed, run aborted: {error}")
            except (DatasetError, StorageError) as error:
                progress.empty()
                status.error(str(error))


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def render_results(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    summary: dict,
    *,
    key_prefix: str,
) -> None:
    """Render the full result view for one experiment.

    Args:
        key_prefix: Unique per call site. This view is rendered from two tabs at
            once, and Streamlit derives widget ids from their parameters — without
            distinct keys the two copies collide and the page raises.
    """
    best = summary.get("best_strategy")
    totals = summary.get("totals", {})

    if best:
        st.success(
            f"**Best strategy by F1: {strategy_label(best['strategy'])}** — "
            f"F1 {best['f1_macro']:.3f}, accuracy {best['accuracy']:.1%}, "
            f"unparseable {best['unknown_rate']:.1%}",
            icon=":material/trophy:",
        )
    else:
        st.info(
            summary.get("best_strategy_note")
            or "No single best strategy: the top score is tied.",
            icon=":material/balance:",
        )

    metric_row(
        [
            ("Predictions", f"{totals.get('predictions', 0):,}", None),
            ("Correct", f"{totals.get('correct', 0):,}", None),
            ("Unparseable", f"{totals.get('unknown', 0):,}", "Could not be parsed into a label"),
            ("API failures", f"{totals.get('api_failures', 0):,}", None),
            ("Runtime", f"{totals.get('runtime_seconds', 0):.0f}s", None),
        ]
    )

    st.caption(
        "F1 alone can flatter a strategy that abstains: an unparseable answer costs "
        "recall but never pollutes precision. Read F1, accuracy and the unparseable "
        "rate together."
    )

    display_columns = [
        "strategy", "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "error_rate", "unknown_rate", "avg_latency_seconds",
    ]
    table = metrics[[column for column in display_columns if column in metrics.columns]]
    st.dataframe(
        table.style.format(
            {
                "accuracy": "{:.1%}", "precision_macro": "{:.3f}",
                "recall_macro": "{:.3f}", "f1_macro": "{:.3f}",
                "error_rate": "{:.1%}", "unknown_rate": "{:.1%}",
                "avg_latency_seconds": "{:.2f}s",
            }
        ).background_gradient(subset=["f1_macro"], cmap="Blues"),
        width="stretch",
        hide_index=True,
    )

    st.divider()
    accuracy_column, metric_column = st.columns(2)
    with accuracy_column:
        st.pyplot(plot_accuracy_comparison(metrics), width="stretch")
    with metric_column:
        st.pyplot(plot_metric_comparison(metrics), width="stretch")

    error_column, latency_column = st.columns(2)
    with error_column:
        st.pyplot(plot_error_rate_comparison(metrics), width="stretch")
    with latency_column:
        try:
            st.pyplot(plot_latency_comparison(metrics), width="stretch")
        except Exception:  # noqa: BLE001 - latency is optional
            st.info("Latency was not recorded for this run.")

    st.pyplot(plot_confusion_matrices(predictions), width="stretch")

    st.divider()
    st.subheader("Error analysis")
    errors = extract_errors(predictions)
    if errors.empty:
        st.success("No errors — every strategy classified every sample correctly.")
        return

    overview = error_analysis_summary(errors)
    metric_row(
        [
            ("Total errors", str(overview["total_errors"]), None),
            ("Misclassified", str(overview["misclassifications"]), "Readable, but wrong"),
            ("Unparseable", str(overview["unparseable"]), "No label could be extracted"),
            ("Samples affected", str(overview["samples_with_errors"]), None),
        ]
    )

    breakdown, shared = st.columns([1, 1])
    with breakdown:
        st.markdown("**Failure mode by strategy**")
        st.dataframe(
            summarize_errors_by_strategy(errors), width="stretch", hide_index=True
        )
    with shared:
        st.markdown("**Samples every strategy got wrong**")
        universal = find_shared_errors(errors, strategy_count=predictions["strategy"].nunique())
        if universal.empty:
            st.caption("None — no sample defeated every strategy.")
        else:
            st.caption(
                "Evidence about the data — genuine ambiguity or a questionable gold "
                "label — rather than about any one strategy."
            )
            st.dataframe(
                universal.assign(review=universal["review"].str.slice(0, 160) + "…"),
                width="stretch", hide_index=True,
            )

    with st.expander(f"Browse all {len(errors)} errors"):
        chosen = st.multiselect(
            "Filter by strategy", sorted(errors["strategy"].unique()),
            default=sorted(errors["strategy"].unique()), format_func=strategy_label,
            key=f"{key_prefix}-error-filter",
        )
        st.dataframe(
            errors[errors["strategy"].isin(chosen)], width="stretch", hide_index=True
        )

    left, middle, right = st.columns(3)
    with left:
        download_button_for(metrics, "Download metrics.csv", "metrics.csv",
                            key=f"{key_prefix}-dl-metrics")
    with middle:
        download_button_for(predictions, "Download predictions.csv", "predictions.csv",
                            key=f"{key_prefix}-dl-predictions")
    with right:
        download_button_for(errors, "Download errors.csv", "errors.csv",
                            key=f"{key_prefix}-dl-errors")


with results_tab:
    evaluation = st.session_state.evaluation
    result = st.session_state.result
    if evaluation is None or result is None:
        st.info(
            "Not yet evaluated. Run a benchmark, or load a past experiment from the "
            "**Past experiments** tab.",
            icon=":material/hourglass_empty:",
        )
    else:
        st.caption(f"Experiment `{result.experiment_id}`")
        render_results(
            evaluation.metrics, result.predictions, evaluation.summary,
            key_prefix="current",
        )


# ---------------------------------------------------------------------------
# Past experiments
# ---------------------------------------------------------------------------

with history_tab:
    st.subheader("Past experiments")
    st.caption(
        "Completed runs are immutable — a new run never overwrites an old one, so "
        "results stay comparable over time."
    )

    experiments = storage.list_experiments()
    if not experiments:
        st.info("No experiments recorded yet.", icon=":material/folder_open:")
    else:
        chosen = st.selectbox("Experiment", list(reversed(experiments)))
        try:
            stored_config = storage.load_config(chosen)
            stored_metrics = storage.load_metrics(chosen)
            stored_predictions = storage.load_predictions(chosen)
            stored_summary = storage.load_summary(chosen)
        except StorageError as error:
            st.warning(f"This experiment is incomplete: {error}")
        else:
            dataset = stored_config.get("dataset", {})
            metric_row(
                [
                    ("Model", stored_config.get("provider", {}).get("model", "—"), None),
                    ("Samples", str(dataset.get("sample_count", "—")), None),
                    ("Seed", str(dataset.get("random_seed", "—")), None),
                    (
                        "Sample checksum",
                        str(dataset.get("sample_id_checksum", "—")),
                        "Two runs sharing a checksum were scored on identical reviews",
                    ),
                    ("Run at", str(stored_config.get("timestamp", "—"))[:19], None),
                ]
            )
            st.divider()
            render_results(
                stored_metrics, stored_predictions, stored_summary,
                key_prefix=f"stored-{chosen}",
            )
