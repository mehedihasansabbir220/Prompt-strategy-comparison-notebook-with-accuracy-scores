# PromptBench

**An Empirical Study of Prompt Strategies for LLM Sentiment Classification**

A local-first, reproducible evaluation harness that measures how much *prompt strategy alone*
changes an LLM's performance on a sentiment classification benchmark — with the model,
dataset, evaluation samples, decoding settings, and scoring methodology all held constant.

> **Status:** scaffolding complete (Step 1). No benchmark has been executed yet.
> All result tables below read **Not yet evaluated** until a real run is performed.

---

## Research question

> *How much can prompt strategy affect the performance of an LLM on sentiment classification
> when the model, dataset, test samples, and evaluation methodology remain controlled?*

Prompt engineering is frequently discussed anecdotally. This project treats it as a
measurable engineering variable: one benchmark, one model, one fixed sample set, six
prompt strategies, and identical scoring for all of them.

---

## Why this project exists

Most prompt-engineering material demonstrates a single prompt on a handful of cherry-picked
inputs. That tells you nothing about whether a technique *generalises*. PromptBench exists to
answer a narrower but honest question — on this task, on this model, with this sample set,
which strategies actually move the metrics, and by how much?

It is built as production-shaped engineering rather than a notebook script:
separated modules, typed interfaces, mocked tests, versioned experiment artifacts,
and a provider abstraction so a second LLM can be added without touching the evaluation layer.

---

## Methodology

A controlled experiment. Exactly one variable changes.

| Held constant | Varied |
| --- | --- |
| Dataset (IMDb, Hugging Face) | Prompt strategy |
| Fixed evaluation subset + ground-truth labels | |
| Model identifier | |
| Temperature and generation settings | |
| Response parsing and metric computation | |
| Random seed | |

**Procedure**

1. Load IMDb and normalise labels to `positive` / `negative`.
2. Draw one **fixed, seeded** evaluation subset. Every strategy is scored on *exactly* these samples.
3. For each strategy, render a prompt per sample and query the model with identical generation settings.
4. Parse each raw response into `positive`, `negative`, or `unknown` (unparseable) using one shared parser.
5. Compute accuracy, precision, recall, F1, and a confusion matrix per strategy.
6. Persist predictions, metrics, errors, and run metadata under an immutable experiment ID.

**Integrity rules**

- No fabricated numbers. Every metric traces back to a stored raw model response.
- Failed API calls and unparseable outputs are counted and reported — not silently dropped.
- Observed results and their interpretation are stated separately.
- No hidden chain-of-thought is requested or stored; only the final label and the raw response.

---

## Prompt strategies

| # | Strategy | Purpose | Hypothesis |
| --- | --- | --- | --- |
| 1 | **Zero-shot** | Baseline instruction with no demonstrations. | Establishes the floor that every other strategy must beat. |
| 2 | **Few-shot** | Fixed in-prompt demonstrations of the task. | Examples anchor the label space and reduce format drift. |
| 3 | **Role-based** | Assigns an expert classifier persona. | Role framing sharpens decision boundaries on ambiguous reviews. |
| 4 | **Structured-output** | Forces a machine-readable response schema. | Cuts unparseable responses, raising effective accuracy. |
| 5 | **Reasoning-aware** | Instructs careful consideration, returns only the final label. | Helps on mixed-sentiment reviews without exposing reasoning. |
| 6 | **Combined** | Merges the techniques that individually performed best. | Gains may or may not compose — this measures whether they do. |

Demonstrations used by the few-shot strategies are drawn from the *training* split and are
never members of the evaluation subset, so no strategy sees a test sample in its prompt.

---

## Architecture

```
promptbench/
├── src/
│   ├── config.py                 # Typed configuration; loads .env, holds experiment constants
│   ├── dataset.py                # IMDb loading, label normalisation, fixed seeded subset
│   ├── prompts/                  # One module per strategy, behind a shared base interface
│   │   ├── base.py               #   Strategy contract: name, purpose, hypothesis, render()
│   │   ├── zero_shot.py
│   │   ├── few_shot.py
│   │   ├── role_based.py
│   │   ├── structured.py
│   │   ├── reasoning.py
│   │   └── combined.py
│   ├── llm/
│   │   └── gemini.py             # Gemini adapter; provider-agnostic call signature
│   ├── evaluation/
│   │   ├── metrics.py            # Accuracy, precision, recall, F1, confusion matrix
│   │   └── errors.py             # Per-sample error records and cross-strategy comparison
│   ├── experiments/
│   │   ├── runner.py             # Benchmark loop: strategy x sample -> prediction
│   │   └── storage.py            # Experiment IDs, immutable run directories, metadata
│   ├── visualization/
│   │   └── charts.py             # Presentation-ready comparison charts
│   └── utils/
│       ├── parsing.py            # Raw response -> positive | negative | unknown
│       └── logging.py            # Structured logging; never logs secrets
├── notebooks/
│   └── prompt_strategy_comparison.ipynb   # Research interface; imports from src/
├── tests/                        # pytest; no test performs a real API call
├── results/
│   ├── experiments/<experiment_id>/       # Immutable, never overwritten
│   └── latest/                            # Convenience pointer to the newest run
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

The notebook is an interface, not an implementation — all logic lives in `src/` and is unit-tested.

---

## Evaluation metrics

**Classification quality:** accuracy, precision, recall, F1 (per class and macro-averaged),
confusion matrix per strategy.

**Operational metrics:** total samples, correct, incorrect, unknown/unparseable predictions,
API failures, wall-clock runtime, and token usage where the provider reports it.

Unknown predictions are reported as their own category. A strategy that returns prose instead
of a label is not "wrong by accident" — it has a reliability problem worth measuring separately.

---

## Results

**Not yet evaluated.**

Metric tables, per-strategy confusion matrices, and comparison charts are generated by the
notebook and written to `results/experiments/<experiment_id>/`. This section is populated from
a real run — never by hand.

---

## Error analysis

**Not yet evaluated.**

Every incorrect prediction is stored with `sample_id`, `strategy`, `review`, `actual_label`,
`predicted_label`, and `raw_response`, which supports:

- Samples that *every* strategy gets wrong (likely genuinely ambiguous or mislabelled).
- Samples a specific strategy uniquely fixes or uniquely breaks.
- Failure modes by type: wrong label vs. unparseable output vs. API failure.

---

## Reproducibility

Each run writes an immutable directory keyed by experiment ID (`YYYY-MM-DD_NNN`):

```
results/experiments/2026-08-23_001/
├── config.json        # model, temperature, seed, sample size, strategies, full configuration
├── predictions.csv    # one row per (strategy, sample): prediction, ground truth, raw response
├── metrics.csv        # per-strategy metric table
├── errors.csv         # every incorrect prediction with full context
└── summary.json       # headline numbers and run metadata
```

Historical experiments are never overwritten. Re-running with the same seed and sample size
reproduces the same evaluation subset; because LLM outputs are not guaranteed deterministic
even at temperature 0, predictions may vary slightly between runs — which is itself reported
rather than hidden.

---

## Run it locally

Everything runs on a normal laptop. **No GPU. No Colab. No cloud account** beyond a free
Gemini API key.

### Prerequisites

| Requirement | Notes |
| --- | --- |
| Python **3.11+** | Built and verified on **3.13**. Check with `python3 --version`. |
| A Gemini API key | Free tier is sufficient. Get one at <https://aistudio.google.com/apikey>. |
| ~500 MB disk | Dependencies plus the cached IMDb dataset. |

Steps 1–4 need **no API key** — you can clone, install, and run the full test suite offline.
The key is only required at Step 5, when the benchmark actually calls the model.

### 1. Get the code

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

Your shell prompt should now be prefixed with `(.venv)`. If you have several Python versions
installed, pin one explicitly — e.g. `python3.13 -m venv .venv`.

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the install resolved cleanly:

```bash
pip check          # expected: "No broken requirements found."
```

### 4. Run the test suite

```bash
pytest
```

The suite covers dataset processing, label normalisation, prompt generation, response parsing,
metric computation, and experiment storage. **No test makes a real API call** — LLM responses
are mocked — so this works with no key and costs nothing. Run it before anything else: if it
passes, your environment is correctly wired.

### 5. Configure your API key

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Open `.env` and replace the placeholder:

```dotenv
GEMINI_API_KEY=your_real_key_here
```

`.env` is listed in `.gitignore` and is never committed. The key is read from the environment
only — it is never hardcoded, printed, or written to logs or result files.

The same file also holds optional experiment defaults you can change without touching code:

```dotenv
PROMPTBENCH_MODEL=gemini-2.5-flash
PROMPTBENCH_TEMPERATURE=0.0
PROMPTBENCH_BENCHMARK_SAMPLE_SIZE=100
PROMPTBENCH_DEV_SAMPLE_SIZE=10
PROMPTBENCH_RANDOM_SEED=42
```

Start with the small `PROMPTBENCH_DEV_SAMPLE_SIZE` subset for your first run to confirm the
pipeline works end to end before spending quota on the full benchmark.

### 6. Launch the notebook

```bash
# one-off: make this venv selectable as a Jupyter kernel
python -m ipykernel install --user --name promptbench --display-name "Python (promptbench)"

jupyter lab notebooks/prompt_strategy_comparison.ipynb
```

In JupyterLab, select the **Python (promptbench)** kernel, then run the cells top to bottom.
The notebook is the main research interface — it walks through the research question,
dataset, prompt strategies, example prompts, a single-sample demonstration, the full
benchmark, metrics, charts, error analysis, findings, and limitations.

The first run downloads IMDb from Hugging Face (a one-time download, cached locally).

### 7. Read the results

Each benchmark run writes its own immutable directory:

```
results/experiments/<experiment_id>/     # e.g. 2026-08-23_001
├── config.json        # exact configuration used
├── predictions.csv    # every prediction with its raw response
├── metrics.csv        # per-strategy metric table
├── errors.csv         # every incorrect prediction with full context
└── summary.json       # headline numbers and run metadata
```

Earlier runs are never overwritten, so results stay comparable over time.

### Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the repository root. `pytest` is configured with `pythonpath = ["."]` in `pyproject.toml`. |
| Notebook imports fail but `pytest` passes | Wrong kernel selected — switch to **Python (promptbench)** (Step 6). |
| Missing / invalid API key error | `.env` is absent, or still holds `your_api_key_here`. Re-check Step 5. |
| Rate-limit or quota errors from Gemini | Lower `PROMPTBENCH_BENCHMARK_SAMPLE_SIZE`, or wait for the free-tier window to reset. Failed calls are counted and reported, not silently dropped. |
| IMDb download is slow or fails | It is a one-time cached download; re-run the cell to resume. |

---

## Limitations

- **Single model, single task.** Results describe Gemini on IMDb sentiment. They do not
  automatically transfer to other models, domains, or task types.
- **Sample size.** A modest evaluation subset keeps the run affordable; small differences
  between strategies may be within noise. Sample size is reported alongside every result.
- **Provider variance.** Model endpoints change over time; the model identifier and run date
  are recorded so results stay interpretable.
- **Binary sentiment.** IMDb reviews are long and often mixed; a two-class label cannot capture
  genuine ambiguity, which the error analysis surfaces directly.
- **Prompt space is unbounded.** These six strategies are a considered sample, not the optimum.

---

## Future improvements

- Additional providers behind the existing LLM interface for cross-model comparison.
- Statistical significance testing (bootstrap confidence intervals) on strategy differences.
- Cost-per-correct-prediction as a first-class metric alongside accuracy.
- Additional tasks to test whether strategy rankings hold beyond sentiment.

---

## License

MIT
# Prompt-strategy-comparison-notebook-with-accuracy-scores
