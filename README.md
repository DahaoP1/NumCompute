# NumCompute Stream

> **Assignment 2.1: [Open the original NumCompute README](numcompute/README.md)** for the earlier package, APIs and batch benchmarks.

P_AI Assignment 2.2, 2026 S2 — Individual project

A streaming decision-tree framework built with plain Python, NumPy and matplotlib. It processes CSV chunks, updates preprocessing and statistics, retrains a single tree from a buffer, and records metrics over time. No external ML or data libraries are used.

## Project structure

```text
NumCompute/
├── README.md
├── pyproject.toml
├── numcompute/                   # Assignment 2.1; also provides CSV I/O
│   └── README.md
├── numcompute_stream/
│   ├── __init__.py
│   ├── preprocessing.py          # Incremental scaling, filling and encoding
│   ├── stats.py                  # Running moments and exact distribution queries
│   ├── tree.py                   # DecisionTreeClassifier and buffered retraining
│   ├── metrics.py                # Cumulative and sample-window metrics
│   ├── pipeline.py               # Transformer/Estimator protocols and Pipeline
│   ├── stream.py                 # StreamTrainer and chunk logs
│   ├── visualise.py              # Matplotlib plots
│   └── benchmarking.py           # Timing helpers and buffer-size comparison
├── tests/                        # Original and streaming tests
├── demo/
│   ├── stream_demo.ipynb         # Current end-to-end demo
│   ├── quickstart.ipynb          # Assignment 2.1 demo
│   ├── generate_sample.py
│   └── data/sample.csv
└── benchmark/
    ├── __init__.py
    ├── stream_loops.py
    ├── run_stream_benchmarks.py
    ├── stream_results.md
    ├── loops.py                  # Original batch baselines
    ├── run_benchmarks.py
    └── results.md
```

CSV reading is reused from `numcompute.io`. The original package remains available alongside the streaming extension.

## Installation

Requires Python 3.10+, NumPy and matplotlib. Python's standard-library modules need no separate installation. From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows; macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[test]"       # Both packages, benchmark runners and pytest
python -m pip install ipython ipykernel # Notebook tools
```

Open [demo/stream_demo.ipynb](demo/stream_demo.ipynb), choose this environment, restart the kernel and select **Run All**. The notebook supports the project root or `demo/` as its working directory.

## Quick start: test, then train

The tree needs a first full buffer before it can predict. After that, score each new chunk **before** training on those same rows. This measures performance on unseen data. Here age/income are inputs and city is the target; blank targets are excluded.

```python
import numpy as np
from numcompute.io import iter_csv_chunks
from numcompute_stream.preprocessing import SimpleImputer
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.metrics import F1

tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
pipe = Pipeline([
    ("impute", SimpleImputer(fill_value=0)),
    ("model", tree),
])
trainer = StreamTrainer(pipe, metrics={"f1": F1(average="macro")})
accuracy_curve = []

for raw in iter_csv_chunks("demo/data/sample.csv", chunk_size=200,
                           usecols=["age", "income", "city"]):
    labelled = raw[:, 2] != ""
    X = np.where(raw[labelled, :2] == "", "nan", raw[labelled, :2]).astype(float)
    y = raw[labelled, 2]

    score = np.nan  # Warm-up has no prediction yet.
    if tree.n_retrains_ and len(y):
        row = trainer.score_chunk(X, y)
        score = row["accuracy"]

    accuracy_curve.append(score)
    trainer.fit_chunk(X, y)

print(trainer.n_trained_samples_, trainer.n_scored_samples_)  # 2920, 2338
```

With complete 200-row chunks, `buffer_size=500` triggers training on the third chunk: **all 600 buffered rows** are used, then the buffer is cleared. Missing-target filtering can make chunks smaller. `fit(X, y)` instead uses all supplied training rows; buffer_size does not restrict fit. A replacement tree uses only its current buffer, not all historical data. A final incomplete buffer stays pending.

## API overview and examples

Unless stated otherwise, X has shape `(n, d)` and y has shape `(n,)`. `fit`, `partial_fit` and `update` return the same object (`self`), not intermediate data. Use `transform` for processed arrays, `predict` for predictions and `result` for metrics. Full validation and complexity details are in the docstrings.

### CSV I/O: reuse `numcompute.io`

| Function | Parameters | Return | Purpose |
| --- | --- | --- | --- |
| `read_header()` | `path`, `delimiter=","`, `encoding="utf-8-sig"` | List of column names | Read the CSV header. |
| `load_csv()` | `path`, `usecols=None`, `dtype=str`, `return_header=False`; shared CSV options | 2-D array; optionally `(data, names)` | Load selected columns in one call. |
| `iter_csv_chunks()` | `path`, positive `chunk_size`; shared CSV options | Iterator yielding arrays of at most chunk_size rows | Read new rows a chunk at a time. |

Shared options include `delimiter=","`, `has_header=True`, `usecols=None`, `dtype=str`, `on_missing="fill"` (`fill`, `skip`, `raise`), `fill_value=None`, `missing_values`, `comments=None` and `encoding="utf-8-sig"`. Missing float cells default to NaN; text cells default to an empty string. See the [original I/O documentation](numcompute/README.md#numcomputeio-csv-input) for all dtype and parsing rules.

### `preprocessing.py`

All four classes provide `fit(X)`, `partial_fit(X)`, `transform(X)` and `fit_transform(X)`. Fit starts again, while partial_fit adds the new chunk to learned state. Transform preserves row order and does not update parameters.

| Class | Parameters | Return | Purpose |
| --- | --- | --- | --- |
| `StandardScaler` | No constructor options | Transformed float array `(n, d)` | Use running column means and population standard deviations; zero scales become one. |
| `MinMaxScaler` | `feature_range=(0, 1)`, finite low < high | Transformed float array `(n, d)` | Update observed minima/maxima and scale to the chosen range; constant columns map to low. |
| `SimpleImputer` | `strategy=None`: mean, median or constant; `fill_value=None` | Filled numeric/text array `(n, d)` | Update missing-value estimates, or fill with a fixed number/string. |
| `OneHotEncoder` | `handle_unknown="error"`: error or ignore | Dense float indicators `(n, total_categories)` | Add new categories across chunks and encode categorical inputs. |

Numeric statistics ignore NaN per feature; scalers preserve NaN in their output. Imputer defaults to mean, or constant if fill_value is supplied. Unobserved numeric columns use the imputer's fallback (zero by default). Exact median filling keeps history. String constant filling and encoding treat `""`, `"nan"` and `"None"` as missing. Encoder categories stay sorted: new categories can change output width and positions, so re-encode old raw rows before combining outputs from different stages.

```python
from numcompute_stream.preprocessing import StandardScaler, OneHotEncoder

scaler = StandardScaler().partial_fit([[10.0], [20.0]])
scaler.partial_fit([[30.0]])
print(scaler.mean_)                         # [20.]; all three rows are included

encoder = OneHotEncoder(handle_unknown="ignore").partial_fit([["Sydney"]])
encoder.partial_fit([["Adelaide"]])
print(encoder.transform([["Sydney"]]))     # [[0., 1.]]
```

### `stats.py`

| Function or method | Parameters | Return | Purpose |
| --- | --- | --- | --- |
| `RunningStats()` | `skip_nan=True`, `store_data=True` | Empty accumulator | Choose NaN handling and whether to retain rows for exact queries. |
| `update()` / `partial_fit()` | New numeric chunk | self | Merge running moments, counts and extrema. |
| `update_stats()` | `X_chunk`, `stats=None` | Updated RunningStats | Update the supplied accumulator, or create one if omitted. |
| `mean`, `min`, `max`, `count_` | Properties; no parentheses | Arrays `(d,)` | Read per-column summaries and valid-value counts; n counts all received rows. |
| `var()` / `std()` | `ddof=0` | Arrays `(d,)` | Read population variance/std; ddof=1 gives sample estimates. |
| `quantile()` | `q` in [0,1], `axis=0`, `method="linear"` | Per-column quantiles, or flattened results with axis=None | Query exact quantiles of saved data. |
| `median()` | `axis=0` | Per-column or flattened median | Query the 0.5 quantile. |
| `histogram()` | `bins=10` or edges, `range=None`, `density=False`, `feature=None` | `(counts_or_density, bin_edges)` | Bin all saved values or one feature; NaNs are omitted. |
| `reset()` | None | self | Clear the current stream and keep constructor options. |

A 1-D stats input means many rows of one feature. Moments return `(d,)`. With axis=0, scalar q returns `(d,)`, and a q vector returns `(len(q), d)`. Axis=None combines all values. Quantile methods are linear, lower, higher, midpoint and nearest. skip_nan=False propagates NaN in affected features. Moments-only storage is O(d); exact quantiles, median and histograms require retained history, O(N*d).

```python
from numcompute_stream.stats import RunningStats, update_stats

summary = RunningStats()
for chunk in ([[1.0], [3.0]], [[5.0]]):
    update_stats(chunk, summary)  # Keep passing the same object.
print(summary.mean, summary.var())  # [3.], [2.666...]
print(summary.quantile([0.25, 0.5, 0.75]))
```

### `tree.py`

| Class or method | Parameters | Return | Purpose |
| --- | --- | --- | --- |
| `DecisionTreeClassifier()` | `max_depth=5`, `min_samples_split=2`, `max_features=None`, `buffer_size=500`, `criterion="gini"`, `random_state=0` | Classifier | Configure tree growth, feature selection and buffered retraining. |
| `fit()` | `X`, `y` | self | Train from scratch on all supplied rows; clear pending data and reset the retrain counter. |
| `partial_fit()` | `X_chunk`, `y_chunk` | self | Append rows; rebuild once when the threshold is reached, then clear the buffer. |
| `predict()` | `X` | Labels `(n,)` | Use the most recently trained tree. |

Criterion can be gini or entropy. max_depth=None allows unrestricted depth; max_features is None or a positive integer no greater than d. Numeric feature NaNs are supported; infinities and missing targets are rejected. Majority ties use the first sorted class; near-equal gains prefer the lower feature index, then the smaller threshold, then NaNs on the right. Predict requires fit or the first buffered retrain. Useful attributes are `classes_`, `n_buffered_samples_`, `n_retrains_` and `n_training_samples_`.

### `metrics.py`

Every metric class supports `update(y_true, y_pred) -> self`, `result()` and `reset() -> self`. ROC/AUC take y_score instead of y_pred. `window_size=None` accumulates all samples; a positive integer keeps the latest W samples, not chunks.

| Class | Parameters | Return from result() | Purpose |
| --- | --- | --- | --- |
| `Accuracy` | `labels=None`, `window_size=None` | Correct-prediction fraction, float | Evaluate sample-weighted accuracy. |
| `Precision`, `Recall`, `F1` | `average="binary"`, `pos_label=1`, `zero_division=0`, `labels=None`, `window_size=None` | Float, or `(k,)` with average=None | Evaluate per-class performance and combine it using the chosen average. |
| `ConfusionMatrix` | `labels=None`, `window_size=None` | Integer counts `(k, k)` | Accumulate true-row/predicted-column counts; labels_ gives their order. |
| `MSE` | `window_size=None` | Mean squared error, float | Accumulate numeric errors using sample counts. |
| `ROCCurve` | `pos_label=1`, `window_size=None` | `(fpr, tpr, thresholds)` arrays | Rank binary scores; tied scores cross a threshold together. |
| `AUC` | `pos_label=1`, `window_size=None` | ROC area, float | Compute exact area from all active labels/scores, not average chunk AUC. |

Averaging options are binary, macro, micro, weighted and None. Fixed labels preserve a chosen class order, including unobserved classes; otherwise active labels are sorted. zero_division supplies a finite value for undefined ratios. Empty updates do nothing; result before data raises RuntimeError. ROC/AUC also need both classes before result is defined. Exact cumulative ROC/AUC retain O(N) labels/scores; a window bounds this to O(W). Original batch functions such as accuracy, f1 and roc_curve are also available.

```python
from numcompute_stream.metrics import Accuracy, F1

accuracy = Accuracy(window_size=3)
f1 = F1(average="macro")
for truth, prediction in [([0, 1], [0, 0]), ([1, 1], [1, 1])]:
    accuracy.update(truth, prediction)
    f1.update(truth, prediction)
print(accuracy.result())  # 2/3: only the latest three samples
```

### `pipeline.py` and `stream.py`

| Class or method | Parameters | Return | Purpose |
| --- | --- | --- | --- |
| `Transformer` / `Estimator` | Protocols declaring fit/partial_fit plus transform/predict | Interface declarations | Describe required methods without forcing component inheritance. |
| `Pipeline()` | Named `(name, component)` steps | Pipeline | Chain transformers and an optional final estimator, storing component references. |
| `pipe.fit()` / `partial_fit()` | `X`, `y=None` | self | Fit afresh or update steps in order; only the final model receives y. |
| `pipe.transform()` / `fit_transform()` | `X`; fit_transform also accepts `y=None` | Processed array | Run the final transformer; fit_transform first starts a new fit. |
| `pipe.predict()` | `X` | Model predictions | Apply current preprocessing and call the final estimator without fitting. |
| `StreamTrainer()` | `model` or Pipeline, `metrics=None` mapping of named label metrics | Trainer | Coordinate model updates, evaluation and logs using existing components. |
| `trainer.fit_chunk()` | `X`, `y` | self | Train/update and log time, memory and buffer state; do not score training rows. |
| `trainer.score_chunk()` | `X`, `y` | Score-log dictionary; None for an empty chunk | Predict and update chunk/cumulative metrics without training. |

Pipeline keeps the original batch API; partial_fit explicitly requires an incremental method on every step. It does not rebuild an estimator's older transformed buffers. Use stable preprocessing, such as constant imputation, with buffered trees. If a fixed scaler is needed, fit it separately and transform chunks before a model-only pipeline. Expanding encoder columns also requires a downstream model that supports that schema change.

Trainer `logs_` includes event, row counts, accuracy, cumulative_accuracy, extra metrics, seconds, memory_bytes, buffered_samples and n_retrains. Fit events have no new chunk score. Memory is an estimate of retained model/metric objects and NumPy buffers, excluding input chunks/log history; it is not process RAM or peak memory. Timings exclude initial checks and memory/log copying. ROC/AUC require scores and are updated separately from the trainer's label-based metrics.

### `visualise.py`

| Function | Parameters | Return | Purpose |
| --- | --- | --- | --- |
| `plot_metric_over_time()` | `metric_values`, `title="Metric over time"`, `ylabel="Value"` | `(fig, ax)` | Plot a metric against one-based chunk positions. |
| `compare_models()` | `metric1`, `metric2`, `labels=("Model 1", "Model 2")`, optional title/ylabel | `(fig, ax)` | Compare equal-length histories aligned to the same input chunks. |
| `plot_predictions_vs_ground_truth()` | `y_true`, `y_pred`, optional title | `(fig, ax)` | Show true/predicted classes using a shared category axis. |

All accept `save_path=None` and `show=True`. NaN metric values leave gaps for unscored chunks; do not replace warm-up with zero accuracy. Label plots use category positions, not numerical distances between cities. Close returned figures when finished with repeated noninteractive plotting.

```python
from numcompute_stream.visualise import plot_metric_over_time

# Reuse the curve from the quick-start stream above.
fig, ax = plot_metric_over_time(accuracy_curve, title="Incoming-chunk accuracy", ylabel="Accuracy")
# Add save_path="plots/accuracy.png" to save; show=False disables display.
```

### `benchmarking.py`

| Function | Parameters | Return | Purpose |
| --- | --- | --- | --- |
| `time_function()` | `func`, `repeat=7`, `number=1`, `warmup=1` | Seconds per call `(repeat,)` | Repeat a callable and record its runtime. |
| `compare()` | `case`, `loop_func`, `vec_func`, repeats/count and tolerances | Dictionary with loop/NumPy median times and speedup | Check matching answers before timing. |
| `environment_info()` / `format_table()` | None / result rows | Environment dictionary / Markdown | Describe the machine and present function timings. |
| `benchmark_buffers()` | `X`, `y`, `buffer_sizes=(100,500,1000)`, `chunk_size=200`, `repeat=3`, `max_depth=5`, `random_state=42` | One result dictionary per buffer size | Compare fresh models on the same test-before-train scoring rows. |
| `format_buffer_table()` | Buffer result rows | Markdown | Present accuracy, scored counts, retrains, pending rows and times. |

The first four helpers are reused from the original package. The full runner passes five buffer sizes explicitly; the helper accepts other positive, distinct sizes too.

## Demo and sample data

[stream_demo.ipynb](demo/stream_demo.ipynb) follows one shared-data workflow: read/cache chunks, inspect preprocessing and stats, train trees, evaluate real predictions, build pipelines, run StreamTrainer, plot results and run benchmarks. The small CSV is cached for easy replay; the quick-start above shows direct iterator use. Model test rows are held out, and streaming scores are recorded before learning their rows.

`demo/data/sample.csv` has 3,000 fictional records from six cities (500 each before hiding labels). It contains 180 missing ages, 240 missing incomes and 80 blank city labels, plus repeated and unusually high incomes for edge-case demonstrations. `demo/generate_sample.py` regenerates it with seed 42. The legacy city_predict column serves Assignment 2.1; the streaming demo uses tree predictions. Its binary ROC example uses actual 0/1 Sydney predictions, so it is a coarse hard-label ROC, not a probability estimate.

## Tests and edge cases

The current suite passes **386 cases: 212 original and 174 streaming**, exceeding the 30-case streaming requirement. Related checks are grouped in test functions. Run from the root:

```bash
python -m pytest tests -q
python -m pytest tests -q -k test_stream
python -m pytest tests/test_stream_tree.py -q
```

Coverage includes empty chunks, invalid types/shapes, NaNs and all-missing columns, constant columns, duplicate categories, class/split ties, large numeric offsets, overflow rejection and non-contiguous arrays. Streaming checks cover uneven chunk sizes, threshold crossing, oversized chunks, buffer clearing, pending final data, prediction warm-up, reset, rolling sample windows, dynamic classes, pipeline order and log snapshots. Plot tests run headlessly and check file output. Benchmark tests check equal answers and shared scoring rows; they do not require one implementation to be faster.

## Benchmarking and performance table

```bash
python -m benchmark.run_stream_benchmarks
python -m benchmark.run_stream_benchmarks --quick
python -m benchmark.run_stream_benchmarks --output benchmark/my_results.md
```

The [full recorded report](benchmark/stream_results.md) includes sizes and environment notes. Below is that run: Python 3.12.8, NumPy 2.2.4, Windows 10, Intel64 Family 6 Model 151, 20 logical CPUs; seed 42 and five repeats. Times are medians and can change with hardware/load. Update cases start fresh; queries use prepared history. Python lists are prepared before timing, except conversion inside the loop split search. The table measures full implementations, including normal validation.

| Operation | NumCompute result | Baseline/result | Notes |
|---|---:|---:|---|
| stats.RunningStats n=50,000, d=6, chunk=500 | 11.629 ms | 37.187 ms (3.2x) | Per-cell Welford vs vectorised chunk merging; NaNs ignored per feature |
| preprocessing.StandardScaler n=50,000, d=6 | 13.531 ms | 99.804 ms (7.4x) | Fresh partial_fit sequence, then transform using final parameters |
| preprocessing.MinMaxScaler n=50,000, d=6 | 6.859 ms | 105.651 ms (15.4x) | Learn cumulative minima/maxima, then scale; same constant-column and NaN rules |
| preprocessing.SimpleImputer(mean) n=50,000, d=6 | 5.425 ms | 82.838 ms (15.3x) | Observed cells only; missing estimates updated across chunks |
| preprocessing.SimpleImputer(median) n=5,000, d=6 | 3.260 ms | 13.750 ms (4.2x) | Exact medians updated after every chunk on both sides; then fill using final medians |
| preprocessing.OneHotEncoder n=20,000, d=3 | 22.959 ms | 21.553 ms (0.9x) | Learn categories across chunks, then create the same sorted dense encoding; blanks/nan/None are missing |
| stats.histogram n=5,000, bins=20 | 0.056 ms | 0.846 ms (15.1x) | Query already saved chunks with identical fixed edges; NaNs skipped and last edge included |
| stats.quantile(linear) n=5,000 | 0.117 ms | 0.471 ms (4.0x) | Query already saved chunks; same linear quartiles; Python sorts while NumPy may use selection |
| metrics.MSE n=50,000 | 1.862 ms | 4.891 ms (2.6x) | Squared errors accumulated across chunks |
| metrics.AUC n=1,000 | 0.174 ms | 4.571 ms (26.3x) | Different algorithms: compare all positive-negative pairs vs sort scores once; same exact AUC |
| tree split search(gini) n=400, d=4 | 0.775 ms | 79.773 ms (103.0x) | Different algorithms: scan rows for each threshold vs cumulative counts; same depth-one fit and NaN rules; list conversion included |
| tree split search(entropy) n=400, d=4 | 0.731 ms | 78.569 ms (107.4x) | Same entropy criterion and zero-count rules; repeated row scans vs cumulative class counts |
| tree.predict n=50,000 | 2.228 ms | 20.121 ms (9.0x) | Same fitted tree; per-row traversal vs grouped NumPy traversal |

Accuracy/ConfusionMatrix speed comparisons are omitted because the current APIs perform extra confusion/schema work compared with simple counting loops. AUC, split search and quantile notes identify algorithm changes as well as vectorisation. Values below 1 are kept: OneHotEncoder is slightly slower in this recorded run.

The buffer comparison uses 6,000 rows, four features, 200-row chunks, max_depth=5 and seed 43 for input generation. All five sizes score the same final 2,000 rows after the largest buffer first trains. Total update time excludes prediction; mean retrain-call time includes validation and joining buffered rows. Pending final buffers are not flushed.

| Buffer size | Accuracy | Scored rows | Retrains | Pending | Total update time | Mean retrain call |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.7890 | 2000 | 30 | 0 | 162.811 ms | 5.427 ms |
| 500 | 0.8305 | 2000 | 10 | 0 | 97.074 ms | 9.689 ms |
| 1000 | 0.8485 | 2000 | 6 | 0 | 69.495 ms | 11.555 ms |
| 2000 | 0.8540 | 2000 | 3 | 0 | 50.668 ms | 16.837 ms |
| 4000 | 0.8600 | 2000 | 1 | 2000 | 25.663 ms | 25.514 ms |

The 4,000-row configuration ends with 2,000 pending rows: its last model is still the tree trained earlier. Larger buffers need longer warm-up and cost more per retrain, but retrain less often. These data-specific scores do not imply that increasing buffer size always improves accuracy. Assignment 2.1 timings remain separately available in [benchmark/results.md](benchmark/results.md).
