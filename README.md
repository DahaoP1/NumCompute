# NumCompute Stream

> **Looking for Assignment 2.1? [Open the original NumCompute README](numcompute/README.md)** for the existing APIs, examples, tests and benchmark results.

P_AI Assignment 2.2, 2026 S2 — Individual project

A streaming extension of NumCompute built with Python, NumPy and matplotlib. The aim is to process incoming CSV chunks, update preprocessing and statistics, train a single decision tree through buffered retraining, and track evaluation results over time.

**Current status: project structure initialised.** The files in `numcompute_stream/`, the streaming tests, notebook and benchmark scripts are placeholders. The streaming APIs and examples below describe the intended implementation and are not runnable yet.

## Project structure

```text
NumCompute/
├── README.md                         # Assignment 2.2 overview
├── pyproject.toml                    # Current packaging; streaming update pending
├── numcompute/                       # Existing Assignment 2.1 package
│   ├── README.md                     # Original documentation
│   └── ...
├── numcompute_stream/
│   ├── __init__.py
│   ├── io.py                         # Chunk-wise CSV input
│   ├── preprocessing.py              # Incremental scaling, filling and encoding
│   ├── tree.py                       # Decision tree with buffered retraining
│   ├── stream.py                     # StreamTrainer and per-chunk logs
│   ├── stats.py                      # Incremental statistics
│   ├── metrics.py                    # Cumulative and rolling-window metrics
│   ├── pipeline.py                   # Incremental transformation and prediction
│   ├── visualise.py                  # Reusable matplotlib plots
│   ├── sort_search.py                # Sorting/searching utilities
│   ├── rank.py                       # Ranking utilities
│   ├── optim.py                      # Numerical derivative utilities
│   ├── utils.py                      # Shared validation and batching
│   └── benchmarking.py               # Timing and result-comparison helpers
├── tests/
│   ├── test_stream_*.py              # Assignment 2.2 test placeholders
│   └── ...                           # Existing Assignment 2.1 tests
├── demo/
│   ├── stream_demo.ipynb             # Assignment 2.2 demo placeholder
│   ├── quickstart.ipynb              # Existing Assignment 2.1 demo
│   └── data/
│       └── sample.csv
└── benchmark/
    ├── stream_loops.py               # Streaming loop baselines placeholder
    ├── run_stream_benchmarks.py      # Streaming benchmark runner placeholder
    ├── loops.py                     # Existing batch baselines
    ├── run_benchmarks.py             # Existing batch runner
    └── results.md                   # Existing batch results
```

Existing functionality will be reused from `numcompute/` where suitable. Streaming state and new behaviour will be added in `numcompute_stream/`. The `tests/`, `demo/` and `benchmark/` folders are shared by both assignments.

## Environment setup

Use Python 3.10 or later. NumPy and matplotlib are the allowed numerical and plotting libraries; pytest is used for testing. External ML and data libraries such as scikit-learn, pandas and PyTorch are not used.

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate               # Windows; macOS/Linux: source .venv/bin/activate
python -m pip install numpy matplotlib pytest
python -m pip install ipython ipykernel  # Notebook display and kernel tools
```

The current `pyproject.toml` still packages the original `numcompute` only. Packaging and dependency metadata for `numcompute_stream` will be updated with the implementation. Keep the repository root on the Python import path during development.

## Streaming workflow

The intended workflow is:

1. Read labelled records from CSV in chunks using the custom I/O module.
2. Update preprocessing parameters with `partial_fit()` and transform the features.
3. Add labelled samples to the decision tree's internal buffer.
4. Retrain the tree when the buffer reaches `buffer_size`.
5. Record chunk metrics, cumulative accuracy and memory use.
6. Plot the recorded results with `visualise.py`.

For evaluation on unseen chunks, prediction and scoring should happen before training on those same records once a trained tree is available. The first prediction before any retraining, and any final incomplete buffer, will need explicitly documented behaviour.

### Buffered retraining

Each `partial_fit(X_chunk, y_chunk)` appends data to the buffer. Once the buffer contains at least `buffer_size` samples, the tree is rebuilt from scratch using the entire buffer, then the buffer is cleared. Between retrains, predictions use the most recently trained tree.

For example, with `buffer_size=500` and chunks of 200 samples:

| Chunk | Buffered samples after appending | Action |
| --- | ---: | --- |
| 1 | 200 | Keep the samples in the buffer. |
| 2 | 400 | Keep the samples in the buffer. |
| 3 | 600 | Train on all 600 samples, then clear the buffer. |
| 4 | 200 | Keep the new samples; continue using the last trained tree. |

## Planned API overview and example

These tables outline the assignment interfaces. Exact defaults and any additional options will be documented as the implementations are added. `X_chunk` has shape `(n_samples, n_features)` and `y_chunk` has shape `(n_samples,)`.

| Function or method | Parameters | Return | Purpose |
| --- | --- | --- | --- |
| `DecisionTreeClassifier(...)` | `max_depth`, `min_samples_split`, `max_features`, `buffer_size`; Gini or entropy criterion | A classifier instance | Configure a single classification tree and its retraining buffer. |
| `tree.fit(X, y)` | Features and matching labels | Fitted classifier | Build a tree from scratch on the supplied data. |
| `tree.partial_fit(X_chunk, y_chunk)` | A labelled data chunk | Updated classifier | Append samples and retrain when the buffer threshold is reached. |
| `tree.predict(X)` | Features compatible with the trained tree | Predicted labels, shape `(n_samples,)` | Predict with the most recently trained tree. |
| `processor.partial_fit(X_chunk)` | A feature chunk | Updated preprocessor | Update scaler statistics, imputation estimates or learned categories. |
| `processor.transform(X)` | Features compatible with the processor | Transformed NumPy array | Apply the current preprocessing state. |
| `pipe.partial_fit(X_chunk, y_chunk)` | A labelled data chunk | Updated pipeline | Update preprocessing and pass processed features to the final model. |
| `pipe.predict(X)` | New feature rows | Predicted labels | Transform the input and call the final classifier. |
| `StreamTrainer.fit_chunk(X, y)` | A training chunk | To be specified | Coordinate incremental training and logging. |
| `StreamTrainer.score_chunk(X, y)` | A chunk to evaluate | To be specified | Evaluate predictions and record chunk and cumulative results. |
| `update_stats(X_chunk)` | A numeric chunk | To be specified | Update streaming mean, variance, quantiles and histogram state. |
| `metric.update(y_true_chunk, y_pred_chunk)` | Matching true and predicted labels; scores for AUC | Updated metric state | Accumulate evaluation results, including rolling-window metrics. |
| `metric.result()` | None | Current score or confusion matrix | Read accumulated results without updating them. |
| `metric.reset()` | None | To be specified | Clear accumulated metric state. |

Conceptual example from the assignment requirements; run it after the streaming classes are implemented:

```python
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.pipeline import Pipeline

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
])

# stream supplies (X_chunk, y_chunk) pairs from the custom CSV reader.
for X_chunk, y_chunk in stream:
    pipe.partial_fit(X_chunk, y_chunk)
```

Scalers will maintain running statistics, the imputer will update its missing-value estimates, and OneHotEncoder will learn new categories incrementally. Consistent feature columns and scaling across buffered chunks must be handled explicitly. Metrics will accumulate confusion counts and AUC information over time; chunk accuracies must be weighted by sample count when calculating cumulative accuracy.

### Planned visualisation APIs

| Function | Parameters | Return | Purpose |
| --- | --- | --- | --- |
| `plot_metric_over_time()` | `metric_values`, `title`, `ylabel` | To be specified | Plot a metric such as accuracy across chunks. |
| `compare_models()` | `metric1`, `metric2`, `labels` | To be specified | Compare streaming results, for example trees using different buffer sizes. |
| `plot_predictions_vs_ground_truth()` | `y_true`, `y_pred` | To be specified | Compare predicted and actual labels for the latest chunk. |

The plotting functions will use matplotlib and support inline display or saving to a file.

## Demo

`demo/stream_demo.ipynb` is currently a placeholder. The completed notebook will load CSV data using `io.py`, process multiple chunks through `Pipeline.partial_fit()`, and log and visualise accuracy, errors and model comparisons. Predictions will come from the decision tree trained during the stream.

The original `demo/quickstart.ipynb` and its documentation remain available through the [Assignment 2.1 README](numcompute/README.md).

## Tests

The target is at least **30 unit tests covering the streaming functionality**. The new `test_stream_*.py` files currently contain no test cases; existing Assignment 2.1 tests remain separate from this target.

Planned coverage includes:

- Buffer sizes below, equal to and above the retraining threshold; buffer clearing and repeated retraining.
- Predictions between retrains and behaviour before the first trained tree exists.
- Incremental means and variances, zero-variance chunks, missing values and newly observed categories.
- Empty chunks, invalid types, shape mismatches, tied labels/splits and non-contiguous array views.
- Cumulative and rolling-window metrics, reset behaviour and changing class labels.
- Pipeline step order, reuse of learned state, plotting and benchmark result checks.

Once test cases are implemented, run from the project root:

```bash
python -m pytest tests -q                         # All collected tests
python -m pytest tests -q -k test_stream           # Streaming test files
python -m pytest tests/test_stream_tree.py -q     # Tree tests only
```

## Benchmarking and performance table

Streaming benchmark files are currently placeholders. The benchmark will compare Python-loop baselines with vectorised NumPy implementations on matching inputs, and compare different `buffer_size` values under the same stream.

| Comparison | Measurements | Status |
| --- | --- | --- |
| Python loops vs NumPy | Matching results, runtime and speedup | Pending implementation |
| Different buffer sizes | Accuracy over chunks, retraining time and retraining frequency | Pending implementation |
| Streaming resource use | Buffer/model memory footprint | Pending implementation |

Results will include dataset/chunk sizes, random seed, timing repetitions and environment notes. The old batch results in `benchmark/results.md` belong to Assignment 2.1; they are not streaming benchmark results. The new performance table will be filled only after the streaming benchmark runs successfully.
