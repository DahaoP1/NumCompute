# NumCompute Stream

> **Assignment 2.1: [Open the original NumCompute README](numcompute/README.md)** for the earlier package, APIs and batch benchmarks.

P_AI Assignment 2.2, 2026 S2 — Individual project

Demo video link: [A2.2_Demo](https://drive.google.com/file/d/1yO5ZOsgMCHXo-uaHvlaNNKImGbiffoND/view?usp=sharing)

<!-- Replace VIDEO_LINK_HERE with the shared video URL before submission. -->

A streaming decision-tree framework built with plain Python, NumPy and matplotlib. It processes CSV chunks, updates preprocessing and statistics, retrains a single tree from a buffer, and records metrics over time. No external ML or data libraries are used.

The extension keeps the original NumCompute APIs where possible. Core array computations use NumPy; loops handle arriving chunks, features and tree nodes. Tree split search uses sorted values and cumulative class counts. Numeric NaNs, constant columns and ties have explicit handling.

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
│   ├── stream_demo.ipynb         # Presentation and end-to-end demo
│   ├── stream_test.ipynb         # Development checks and batch comparisons
│   ├── quickstart.ipynb          # Assignment 2.1 demo
│   ├── generate_sample.py
│   ├── data/sample.csv          # Current 3,000-row dataset
│   └── sample.csv               # Older 8-row example
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
print("Retrains:", tree.n_retrains_)                         # 5
```

In this dataset, removing blank targets leaves 192, 198 and 192 rows in the first three chunks. With `buffer_size=500`, the third chunk triggers training on **all 582 buffered rows**, then clears the buffer. The setting is a threshold, not an exact capacity. `fit(X, y)` instead uses all supplied training rows. A buffered retrain replaces the tree using only the current buffer, not all historical rows; any incomplete final buffer stays pending.

## API overview and examples

Unless stated otherwise, X has shape `(n, d)` and y has shape `(n,)`. `fit`, `partial_fit` and `update` return the same object (`self`), not intermediate data. Use `transform` for processed arrays, `predict` for predictions and `result` for metrics. Full validation and complexity details are in the docstrings.

### CSV I/O: reuse `numcompute.io`

| Function              | Parameters                                                                             | Return                                              | Purpose                            |
| --------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------- | ---------------------------------- |
| `read_header()`     | `path`, `delimiter=","`, `encoding="utf-8-sig"`                                  | List of column names                                | Read the CSV header.               |
| `load_csv()`        | `path`, `usecols=None`, `dtype=str`, `return_header=False`; shared CSV options | 2-D array; optionally `(data, names)`             | Load selected columns in one call. |
| `iter_csv_chunks()` | `path`, positive `chunk_size`; shared CSV options                                  | Iterator yielding arrays of at most chunk_size rows | Read new rows a chunk at a time.   |

Shared options include `delimiter=","`, `has_header=True`, `usecols=None`, `dtype=str`, `on_missing="fill"` (`fill`, `skip`, `raise`), `fill_value=None`, `missing_values`, `comments=None` and `encoding="utf-8-sig"`. Missing float cells default to NaN; text cells default to an empty string. See the [original I/O documentation](numcompute/README.md#numcomputeio-csv-input) for all dtype and parsing rules.

### `preprocessing.py`

All four classes provide `fit(X)`, `partial_fit(X)`, `transform(X)` and `fit_transform(X)`. Fit starts again, while partial_fit adds the new chunk to learned state. Transform preserves row order and does not update parameters.

| Class              | Parameters                                                       | Return                                           | Purpose                                                                                   |
| ------------------ | ---------------------------------------------------------------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| `StandardScaler` | No constructor options                                           | Transformed float array `(n, d)`               | Use running column means and population standard deviations; zero scales become one.      |
| `MinMaxScaler`   | `feature_range=(0, 1)`, finite low < high                      | Transformed float array `(n, d)`               | Update observed minima/maxima and scale to the chosen range; constant columns map to low. |
| `SimpleImputer`  | `strategy=None`: mean, median or constant; `fill_value=None` | Filled numeric/text array `(n, d)`             | Update missing-value estimates, or fill with a fixed number/string.                       |
| `OneHotEncoder`  | `handle_unknown="error"`: error or ignore                      | Dense float indicators `(n, total_categories)` | Add new categories across chunks and encode categorical inputs.                           |

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

| Function or method                     | Parameters                                                                | Return                                                    | Purpose                                                                       |
| -------------------------------------- | ------------------------------------------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `RunningStats()`                     | `skip_nan=True`, `store_data=True`                                    | Empty accumulator                                         | Choose NaN handling and whether to retain rows for exact queries.             |
| `update()` / `partial_fit()`       | New numeric chunk                                                         | self                                                      | Merge running moments, counts and extrema.                                    |
| `update_stats()`                     | `X_chunk`, `stats=None`                                               | Updated RunningStats                                      | Update the supplied accumulator, or create one if omitted.                    |
| `mean`, `min`, `max`, `count_` | Properties; no parentheses                                                | Arrays `(d,)`                                           | Read per-column summaries and valid-value counts; n counts all received rows. |
| `var()` / `std()`                  | `ddof=0`                                                                | Arrays `(d,)`                                           | Read population variance/std; ddof=1 gives sample estimates.                  |
| `quantile()`                         | `q` in [0,1], `axis=0`, `method="linear"`                           | Per-column quantiles, or flattened results with axis=None | Query exact quantiles of saved data.                                          |
| `median()`                           | `axis=0`                                                                | Per-column or flattened median                            | Query the 0.5 quantile.                                                       |
| `histogram()`                        | `bins=10` or edges, `range=None`, `density=False`, `feature=None` | `(counts_or_density, bin_edges)`                        | Bin all saved values or one feature; NaNs are omitted.                        |
| `reset()`                            | None                                                                      | self                                                      | Clear the current stream and keep constructor options.                        |

A 1-D stats input means many rows of one feature. Moments return `(d,)`. With axis=0, scalar q returns `(d,)`, and a q vector returns `(len(q), d)`. Axis=None combines all values. Quantile methods are linear, lower, higher, midpoint and nearest. skip_nan=False propagates NaN in affected feature summaries; histograms always omit NaN. Moments-only storage is O(d); exact quantiles, median and histograms require retained history, O(N*d). Histograms are cumulative; the optional rolling histogram is not implemented.

```python
from numcompute_stream.stats import RunningStats, update_stats

summary = RunningStats()
for chunk in ([[1.0], [3.0]], [[5.0]]):
    update_stats(chunk, summary)  # Keep passing the same object.
print(summary.mean, summary.var())  # [3.], [2.666...]
print(summary.quantile([0.25, 0.5, 0.75]))
```

### `tree.py`

| Class or method              | Parameters                                                                                                                     | Return          | Purpose                                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------- | ------------------------------------------------------------------------------------------ |
| `DecisionTreeClassifier()` | `max_depth=5`, `min_samples_split=2`, `max_features=None`, `buffer_size=500`, `criterion="gini"`, `random_state=0` | Classifier      | Configure tree growth, feature selection and buffered retraining.                          |
| `fit()`                    | `X`, `y`                                                                                                                   | self            | Train from scratch on all supplied rows; clear pending data and reset the retrain counter. |
| `partial_fit()`            | `X_chunk`, `y_chunk`                                                                                                       | self            | Append rows; rebuild once when the threshold is reached, then clear the buffer.            |
| `predict()`                | `X`                                                                                                                          | Labels `(n,)` | Use the most recently trained tree.                                                        |

Criterion can be gini or entropy. max_depth=None allows unrestricted depth; max_features is None or a positive integer no greater than d. Numeric feature NaNs are supported; infinities and missing targets are rejected. Majority ties use the first sorted class; near-equal gains prefer the lower feature index, then the smaller threshold, then NaNs on the right. Predict requires fit or the first buffered retrain. Useful attributes are `classes_`, `n_buffered_samples_`, `n_retrains_` and `n_training_samples_`.

### `metrics.py`

Every metric class supports `update(y_true, y_pred) -> self`, `result()` and `reset() -> self`. ROC/AUC take y_score instead of y_pred. `window_size=None` accumulates all samples; a positive integer keeps the latest W samples, not chunks.

| Class                             | Parameters                                                                                        | Return from result()                 | Purpose                                                                  |
| --------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------ |
| `Accuracy`                      | `labels=None`, `window_size=None`                                                             | Correct-prediction fraction, float   | Evaluate sample-weighted accuracy.                                       |
| `Precision`, `Recall`, `F1` | `average="binary"`, `pos_label=1`, `zero_division=0`, `labels=None`, `window_size=None` | Float, or `(k,)` with average=None | Evaluate per-class performance and combine it using the chosen average.  |
| `ConfusionMatrix`               | `labels=None`, `window_size=None`                                                             | Integer counts `(k, k)`            | Accumulate true-row/predicted-column counts; labels_ gives their order.  |
| `MSE`                           | `window_size=None`                                                                              | Mean squared error, float            | Accumulate numeric errors using sample counts.                           |
| `ROCCurve`                      | `pos_label=1`, `window_size=None`                                                             | `(fpr, tpr, thresholds)` arrays    | Rank binary scores; tied scores cross a threshold together.              |
| `AUC`                           | `pos_label=1`, `window_size=None`                                                             | ROC area, float                      | Compute exact area from all active labels/scores, not average chunk AUC. |

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

| Class or method                            | Parameters                                                             | Return                                        | Purpose                                                                           |
| ------------------------------------------ | ---------------------------------------------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------- |
| `Transformer` / `Estimator`            | Protocols declaring fit/partial_fit plus transform/predict             | Interface declarations                        | Describe required methods without forcing component inheritance.                  |
| `Pipeline()`                             | Named `(name, component)` steps                                      | Pipeline                                      | Chain transformers and an optional final estimator, storing component references. |
| `pipe.fit()` / `partial_fit()`         | `X`, `y=None`                                                      | self                                          | Fit afresh or update steps in order; only the final model receives y.             |
| `pipe.transform()` / `fit_transform()` | `X`; fit_transform also accepts `y=None`                           | Processed array                               | Run the final transformer; fit_transform first starts a new fit.                  |
| `pipe.predict()`                         | `X`                                                                  | Model predictions                             | Apply current preprocessing and call the final estimator without fitting.         |
| `StreamTrainer()`                        | `model` or Pipeline, `metrics=None` mapping of named label metrics | Trainer                                       | Coordinate model updates, evaluation and logs using existing components.          |
| `trainer.fit_chunk()`                    | `X`, `y`                                                           | self                                          | Train/update and log time, memory and buffer state; do not score training rows.   |
| `trainer.score_chunk()`                  | `X`, `y`                                                           | Score-log dictionary; None for an empty chunk | Predict and update chunk/cumulative metrics without training.                     |

Pipeline keeps the original batch API; partial_fit explicitly requires an incremental method on every step. It does not rebuild an estimator's older transformed buffers. Use stable preprocessing, such as constant imputation, with buffered trees. If a fixed scaler is needed, fit it separately and transform chunks before a model-only pipeline. Expanding encoder columns also requires a downstream model that supports that schema change.

Trainer `logs_` includes event, row counts, accuracy, cumulative_accuracy, extra metrics, seconds, memory_bytes, buffered_samples and n_retrains. Fit events have no new chunk score. Memory is an estimate of retained model/metric objects and NumPy buffers, excluding input chunks/log history; it is not process RAM or peak memory. Timings exclude initial checks and memory/log copying. ROC/AUC require scores and are updated separately from the trainer's label-based metrics.

### `visualise.py`

| Function                               | Parameters                                                                                                       | Return        | Purpose                                                          |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------- | ---------------------------------------------------------------- |
| `plot_metric_over_time()`            | `metric_values`, `title="Metric over time"`, `ylabel="Value"`                                              | `(fig, ax)` | Plot a metric against one-based chunk positions.                 |
| `compare_models()`                   | `metric1`, `metric2`, `labels=("Model 1", "Model 2")`, `title="Model comparison"`, `ylabel="Accuracy"` | `(fig, ax)` | Compare equal-length histories aligned to the same input chunks. |
| `plot_predictions_vs_ground_truth()` | `y_true`, `y_pred`, `title="Predictions vs ground truth"`                                                  | `(fig, ax)` | Show true/predicted classes using a shared category axis.        |

All three functions accept `save_path=None` and `show=True`. Use `%matplotlib inline` in a notebook to display plots there; provide save_path to export an image, with parent directories created automatically. show=False skips the helper's display call. Close the returned figure to also prevent Jupyter's automatic end-of-cell display. NaN values leave gaps for unscored chunks. Label plots use category positions, not distances between cities.

```python
from numcompute_stream.visualise import plot_metric_over_time
import matplotlib.pyplot as plt

# Display the curve collected in the quick start.
fig, ax = plot_metric_over_time(accuracy_curve, title="Incoming-chunk accuracy", ylabel="Accuracy")
plt.close(fig)

# Export the same curve without displaying it.
fig, ax = plot_metric_over_time(
    accuracy_curve, title="Incoming-chunk accuracy", ylabel="Accuracy",
    save_path="plots/accuracy.png", show=False,
)
plt.close(fig)
```

### `benchmarking.py`

| Function                                    | Parameters                                                                                                            | Return                                              | Purpose                                                            |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------ |
| `time_function()`                         | `func`, `repeat=7`, `number=1`, `warmup=1`                                                                    | Seconds per call `(repeat,)`                      | Repeat a callable and record its runtime.                          |
| `compare()`                               | `case`, `loop_func`, `vec_func`, repeats/count and tolerances                                                   | Dictionary with loop/NumPy median times and speedup | Check matching answers before timing.                              |
| `environment_info()` / `format_table()` | None / result rows                                                                                                    | Environment dictionary / Markdown                   | Describe the machine and present function timings.                 |
| `benchmark_buffers()`                     | `X`, `y`, `buffer_sizes=(100,500,1000)`, `chunk_size=200`, `repeat=3`, `max_depth=5`, `random_state=42` | One result dictionary per buffer size               | Compare fresh models on the same test-before-train scoring rows.   |
| `format_buffer_table()`                   | Buffer result rows                                                                                                    | Markdown                                            | Present accuracy, scored counts, retrains, pending rows and times. |

The first four helpers are reused from the original package. The full runner passes five buffer sizes explicitly; the helper accepts other positive, distinct sizes too.

## Demo and sample data

[stream_demo.ipynb](demo/stream_demo.ipynb) reads the CSV once and reuses its chunks. It demonstrates each module, then connects them with Pipeline and StreamTrainer. Scores come from real tree predictions. The separate [stream_test.ipynb](demo/stream_test.ipynb) retains development checks and comparisons with the original batch package.

| Sections | Demonstration                                                                                                                     |
| -------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 1-1.1    | Read 200-row chunks, handle missing cells, remove blank targets and split training/test chunks.                                   |
| 2-3      | Update both scalers, mean/median imputers, encoder categories and running statistics; query quantiles/histograms and reset state. |
| 4-5      | Fit a tree, retrain from buffers, predict held-out rows and update cumulative/rolling metrics.                                    |
| 6-7      | Chain numeric/text preprocessing and a model, then score before training with per-call logs.                                      |
| 8-8.3    | Plot accuracy, compare buffers, inspect predictions and show optional image-export code.                                          |
| 9        | Preview one benchmark, then run all 13 function comparisons and five buffer settings.                                             |

The charts are drawn from recorded per-chunk results after the loop; they do not refresh after every chunk. Section 8.3 is an optional save example, so Run All does not export an extra image.

### Sample dataset

`demo/data/sample.csv` contains 3,000 fictional records from six city profiles, with 500 generated per city before hiding 80 labels. Age and income are input features; city is the target. The legacy city_predict column is only a rule-based simulation for Assignment 2.1 and is not used as predictions in the streaming demo.

- Missing values: 180 ages, 240 incomes and 80 city labels. Numeric missing tokens include blank, NA, NaN, NULL and N/A.
- Repeated incomes: 14 Sydney rows at 115,000 and 21 Melbourne rows at 90,000.
- Extreme incomes: three rows per city at six times that city's fictional income base.
- Reproduction: `python demo/generate_sample.py` recreates the CSV with seed 42, overwriting that generated file. More blank targets leave fewer rows for training and evaluation.

## Tests and edge cases

The current suite was verified with **386 passing cases: 212 original and 174 streaming**, exceeding the 30-case streaming requirement. Related checks are grouped in test functions. Run from the root:

```bash
python -m pytest tests -q
python -m pytest tests -q -k test_stream
python -m pytest tests/test_stream_tree.py -q
```

Coverage includes empty chunks, invalid types/shapes, NaNs and all-missing columns, constant columns, duplicate categories, class/split ties, large numeric offsets, overflow rejection and non-contiguous arrays. Streaming checks cover uneven chunk sizes, threshold crossing, oversized chunks, buffer clearing, pending final data, prediction warm-up, reset, rolling sample windows, dynamic classes, pipeline order and log snapshots. Plot tests run headlessly and check file output. Benchmark tests check equal answers and shared scoring rows; they do not require one implementation to be faster. See [tests/README.md](tests/README.md) for coverage by file.

## Benchmarking and performance table

```bash
python -m benchmark.run_stream_benchmarks
python -m benchmark.run_stream_benchmarks --quick
python -m benchmark.run_stream_benchmarks --output benchmark/my_results.md
```

The tables below reproduce the latest full benchmark output saved in [stream_demo.ipynb, Section 9](demo/stream_demo.ipynb). Environment: Python 3.12.8, NumPy 2.2.4, Windows 10, Intel64 Family 6 Model 151, 20 logical CPUs; seed 42 and five repeats. The separate [CLI report](benchmark/stream_results.md) records an earlier run, so its timings may differ.

Times are medians; speed-up is loop time / NumPy time. Update cases start fresh, queries use prepared history and prediction uses the same fitted tree. Input lists are prepared before timing, apart from conversion inside loop split search. Normal validation is included.

| Operation                                        | NumCompute result |    Baseline/result | Notes                                                                                                                               |
| ------------------------------------------------ | ----------------: | -----------------: | ----------------------------------------------------------------------------------------------------------------------------------- |
| stats.RunningStats n=50,000, d=6, chunk=500      |         10.982 ms |   42.169 ms (3.8x) | Per-cell Welford vs vectorised chunk merging; NaNs ignored per feature                                                              |
| preprocessing.StandardScaler n=50,000, d=6       |         14.943 ms |  112.747 ms (7.5x) | Fresh partial_fit sequence, then transform using final parameters                                                                   |
| preprocessing.MinMaxScaler n=50,000, d=6         |          6.805 ms | 114.791 ms (16.9x) | Learn cumulative minima/maxima, then scale; same constant-column and NaN rules                                                      |
| preprocessing.SimpleImputer(mean) n=50,000, d=6  |          6.424 ms |  90.961 ms (14.2x) | Observed cells only; missing estimates updated across chunks                                                                        |
| preprocessing.SimpleImputer(median) n=5,000, d=6 |          3.391 ms |   17.565 ms (5.2x) | Exact medians updated after every chunk on both sides; then fill using final medians                                                |
| preprocessing.OneHotEncoder n=20,000, d=3        |         24.504 ms |   22.180 ms (0.9x) | Learn categories across chunks, then create the same sorted dense encoding; blanks/nan/None are missing                             |
| stats.histogram n=5,000, bins=20                 |          0.060 ms |   0.816 ms (13.6x) | Query already saved chunks with identical fixed edges; NaNs skipped and last edge included                                          |
| stats.quantile(linear) n=5,000                   |          0.113 ms |    0.489 ms (4.3x) | Query already saved chunks; same linear quartiles; Python sorts while NumPy may use selection                                       |
| metrics.MSE n=50,000                             |          1.228 ms |    5.330 ms (4.3x) | Squared errors accumulated across chunks                                                                                            |
| metrics.AUC n=1,000                              |          0.132 ms |   4.755 ms (36.1x) | Different algorithms: compare all positive-negative pairs vs sort scores once; same exact AUC                                       |
| tree split search(gini) n=400, d=4               |          1.641 ms |  83.215 ms (50.7x) | Different algorithms: scan rows for each threshold vs cumulative counts; same depth-one fit and NaN rules; list conversion included |
| tree split search(entropy) n=400, d=4            |          1.026 ms |  86.330 ms (84.2x) | Same entropy criterion and zero-count rules; repeated row scans vs cumulative class counts                                          |
| tree.predict n=50,000                            |          2.554 ms |   19.245 ms (7.5x) | Same fitted tree; per-row traversal vs grouped NumPy traversal                                                                      |

Accuracy/ConfusionMatrix speed comparisons are omitted because the current APIs perform extra confusion/schema work compared with simple counting loops. AUC, split search and quantile notes identify algorithm changes as well as vectorisation. Values below 1 are kept: OneHotEncoder is slightly slower in this recorded run.

The buffer comparison uses 6,000 rows, four features, 200-row chunks, max_depth=5 and seed 43 for input generation. All five sizes score the same final 2,000 rows after the largest buffer first trains. Total update time excludes prediction; mean retrain-call time includes validation and joining buffered rows. Pending final buffers are not flushed.

| Buffer size | Accuracy | Scored rows | Retrains | Pending | Total update time | Mean retrain call |
| ----------: | -------: | ----------: | -------: | ------: | ----------------: | ----------------: |
|         100 |   0.7890 |        2000 |       30 |       0 |        212.591 ms |          7.086 ms |
|         500 |   0.8305 |        2000 |       10 |       0 |        105.748 ms |         10.555 ms |
|        1000 |   0.8485 |        2000 |        6 |       0 |         81.707 ms |         13.586 ms |
|        2000 |   0.8540 |        2000 |        3 |       0 |         59.522 ms |         19.785 ms |
|        4000 |   0.8600 |        2000 |        1 |    2000 |         29.384 ms |         29.219 ms |

The 4,000-row configuration ends with 2,000 pending rows: its last model is still the tree trained earlier. Larger buffers need longer warm-up and cost more per retrain, but retrain less often.
