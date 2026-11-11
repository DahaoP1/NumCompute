# NumCompute

P_AI Assignment 2.1, 2026 S2

Group name: 404 Group Not Found

Demo video link: [A2_Demo_v3](https://drive.google.com/file/d/1OQFyF7AuDpzgdBwg2_W-SpvhwnJK2zzi/view?usp=sharing)

A scientific computing toolkit built with Python and NumPy. It includes CSV input, preprocessing, sorting/searching, ranking, statistics, metrics, numerical derivatives, pipelines and reproducible benchmarks.

## Required structure

```text
NumCompute/
├── .gitignore
├── README.md
├── pyproject.toml
├── numcompute/
│   ├── init.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── utils.py
│   └── benchmarking.py
├── tests/
│   └── README.md
├── benchmark/
│   ├── loops.py              # plain-Python versions (the "loop" side)
│   ├── run_benchmarks.py     # seeded runner, writes results.md
│   └── results.md            # latest full run
└── demo/
    ├── quickstart.ipynb
    ├── generate_sample.py
    ├── data/
    │   └── sample.csv
    └── sample.csv            # older 8-row example
```

## Task allocation

| Module                                         | Owner                                                                |
| ---------------------------------------------- | -------------------------------------------------------------------- |
| `io.py`, `metrics.py`, `benchmarking.py` | Jose                                                                 |
| `preprocessing.py`, `stats.py`             | Dat                                                                  |
| `sort_search.py`, `rank.py`                | Faris                                                                |
| `optim.py`, `pipeline.py`                  | Shuhao                                                               |
| `utils.py`, tests, packaging                 | Shared; each member owns their functions and 3--4 tests per function |

## Installation

Requires Python 3.10+ and NumPy (no other runtime dependencies).

```bash
git clone https://github.com/DahaoALG/NumCompute.git
cd NumCompute
python -m venv .venv
.venv\Scripts\activate          # Windows; on macOS/Linux: source .venv/bin/activate
pip install -e ".[test]"        # editable install of numcompute, plus pytest
python -m pytest                # run the test suite
```

To run the demo, install its display/kernel tools:

```bash
python -m pip install ipython ipykernel
```

Then open `demo/quickstart.ipynb` and select the `.venv` interpreter as the kernel in your notebook editor.

## API overview and examples

The tables list the main assignment APIs and their key parameters. Full signatures, validation rules and complexity notes are in the docstrings. `n` means samples, `d` input features, `c` encoded features and `k` classes unless otherwise stated. Run examples from the project root after installation. Boolean options accept True or False; the tables show their default values.

### `numcompute.io`: CSV input

| Function              | Parameters                                                                                       | Return                                                         | Purpose                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `read_header()`     | `path`; `delimiter=","`; `encoding="utf-8-sig"` (delimiter options below)                  | Column names from the first line, `list[str]`                | Read the first CSV line as column names so columns can be identified before loading data.           |
| `load_csv()`        | `path`; shared CSV options below; `return_header=False` (load_csv only)                      | The whole file as a 2-D array, shape `(n_rows, n_cols)`      | Load selected columns with a chosen delimiter and dtype, then fill, skip or reject missing cells.   |
| `iter_csv_chunks()` | `path`; positive integer `chunk_size`; shared CSV options below (no return_header parameter) | The file streamed as 2-D arrays of at most `chunk_size` rows | Process a CSV in bounded-size chunks when loading the entire dataset would require too much memory. |

Shared optional parameters for `load_csv` and `iter_csv_chunks`:

- `delimiter=","`: a custom field separator such as `","` or `"\t"`; `None` splits on whitespace. This option also applies to `read_header`.
- `has_header=True`: treat the first line as column names; set False for a file without a header.
- `usecols=None`: keep all columns, or supply a column name/index or a sequence of names/indices.
- `dtype=str`: text by default; also accepts float, signed integer and unsigned integer types, including corresponding NumPy dtype specifications.
- `missing_values=DEFAULT_MISSING`: default tokens are `""`, `"NA"`, `"N/A"`, `"NaN"`, `"nan"`, `"null"`, `"None"`, `"?"` and `"NULL"`; pass another sequence to customize them.
- `fill_value=None`: use NaN for floating-point output or `""` for text by default; supply a number/string to override. Missing integer cells require an explicit compatible fill value when filling.
- `on_missing="fill"`: fill missing cells; `"skip"` drops incomplete rows; `"raise"` reports the first missing cell.
- `comments=None`: disable comments by default, or supply a prefix such as `"#"`.
- `encoding="utf-8-sig"`: text encoding, handling a UTF-8 byte order mark; another valid encoding can be supplied. This option also applies to `read_header`.

Only `load_csv` accepts `return_header=False`; set True to return `(data, column_names)`. `iter_csv_chunks` always returns an iterator of arrays and has no `return_header` argument. Quoted fields containing the delimiter are not supported.

```python
from numcompute.io import load_csv, read_header, iter_csv_chunks

path = "demo/data/sample.csv"
names = read_header(path)                         # age, income, city, city_predict
X = load_csv(path, usecols=["age", "income"], dtype=float)
print(X.shape)                                    # (3000, 2); missing cells -> NaN
chunks = iter_csv_chunks(path, chunk_size=500, usecols="income", dtype=float)
print(next(chunks).shape)                         # (500, 1)
```

### `numcompute.preprocessing`: preprocessing

| Function                       | Parameters                                                                         | Return                                                                                                                                            | Purpose                                                                                                                                                                     |
| ------------------------------ | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `StandardScaler()`           | No constructor arguments; column mean and population std learned in `fit`        | A new StandardScaler instance with no learned statistics; fit it before calling transform.                                                        | Standardize each column as `(x - mean) / std`, using training statistics. Training constant columns become zero and NaNs remain NaN.                                      |
| `MinMaxScaler()`             | `feature_range=(0.0, 1.0)`                                                       | A new MinMaxScaler configured with feature_range; column minima and ranges are learned during fit.                                                | Map training feature values to a chosen interval using column minima and ranges. Constant columns map to the lower endpoint; new values are not clipped.                    |
| `OneHotEncoder()`            | `handle_unknown="error"` or `"ignore"`                                         | A new OneHotEncoder instance; fit learns categories_ and the number of output columns.                                                            | Represent each categorical value as a block of zero/one indicators. Missing values produce zero blocks; unseen categories raise or are ignored according to handle_unknown. |
| `SimpleImputer()`            | `strategy=None`: `"mean"`, `"median"` or `"constant"`; `fill_value=None` | A new SimpleImputer instance with the selected strategy; fit records one replacement value per input column.                                      | Replace numeric NaNs with a column mean, median or constant. A string fill value enables categorical filling, such as replacing blank cities with Unknown.                  |
| `processor.fit(X)`           | Nonempty 2-D `X` `(n, d)`                                                      | The same processor instance (`self`), now containing fitted statistics or categories and the input feature count.                               | Learn reusable preprocessing parameters from training data, such as means, ranges, replacement values or category lists; do not return transformed data.                    |
| `processor.transform(X)`     | `X` with the fitted feature count                                                | A processed NumPy array `(n, d)` for scaling/imputation, or `(n, c)` for one-hot encoding, where c is the total learned category count.       | Apply previously learned parameters to training, test or new data without fitting again. The input must have the fitted number of features.                                 |
| `processor.fit_transform(X)` | Nonempty 2-D `X` `(n, d)`                                                      | The processed training array: `(n, d)` for scaling/imputation or `(n, c)` for one-hot encoding; fitted state remains available for later use. | Combine fit and transform on the same dataset so training data is processed and learned parameters can later be reused on new rows.                                         |

Scalers ignore NaN during fitting and preserve it during transformation. Training constant columns map to zero in StandardScaler and the lower endpoint in MinMaxScaler. OneHotEncoder converts values to strings; `""`, `"nan"` and `"None"` produce zero blocks. `SimpleImputer(fill_value="Unknown")` fills those categorical missing values. With no strategy, an explicit fill value selects constant filling; otherwise the default is mean filling.

```python
import numpy as np
from numcompute.preprocessing import (
    SimpleImputer, StandardScaler, MinMaxScaler, OneHotEncoder,
)

X = np.array([[1.0], [np.nan], [3.0]])
filled = SimpleImputer(strategy="median").fit_transform(X)  # [[1], [2], [3]]
scaled = StandardScaler().fit_transform(filled)             # about [[-1.225], [0], [1.225]]
bounded = MinMaxScaler().fit_transform(filled)                # [[0], [0.5], [1]]
encoded = OneHotEncoder().fit_transform([["Sydney"], [""], ["Melbourne"]])
print(encoded)                                             # [[0, 1], [0, 0], [1, 0]]
```

### `numcompute.sort_search`: sorting, top-k and searching

| Function            | Parameters                                                    | Return                                                                                                                                                                                       | Purpose                                                                                                                                |
| ------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `stable_sort()`   | Real numeric `values` `(n,)`                              | A NumPy array `(n,)` containing an ascending sorted copy with the input dtype; equal values retain their input order and NaNs come last.                                                   | Return a stable ascending sort using NumPy.                                                                                            |
| `quicksort()`     | Real numeric `values` `(n,)`                              | An ascending sorted NumPy copy `(n,)` with the input dtype and NaNs last; equal values need not retain their original order.                                                               | Additional hand-written, unstable sorting implementation.                                                                              |
| `lexsort()`       | `keys`: equal-length arrays; last key is primary            | An integer index array `(n,)`; applying these indices to the original rows gives the multi-key sorted order.                                                                               | Return indices that sort by multiple keys; the last key is the primary key.                                                            |
| `topk()`          | `values`, `k`; `largest=True`; `return_indices=True`  | By default, `(selected_values, original_indices)`, each a NumPy array `(k,)` in matching order. With return_indices=False, only selected_values is returned.                             | Use NumPy's `argpartition` to return the `k` largest values descending or smallest ascending, with their input indices by default. |
| `top_k()`         | `values`, `k`; `largest=True`; `return_indices=False` | By default, selected values `(k,)` only; with return_indices=True, `(selected_values, original_indices)`, each `(k,)`. Values are descending for largest=True and ascending otherwise. | Backward-compatible spelling that returns only values by default.                                                                      |
| `quickselect()`   | `values`; zero-based `kth`                                | One scalar from the input dtype: the value that would occupy zero-based position kth after ascending sorting; the input is not modified.                                                     | Return the value at zero-based sorted position `kth` without full sorting, using the hand-written selection algorithm.               |
| `binary_search()` | `sorted_array`; query `x`                                 | A pair `(insertion_index, found)`: the leftmost insertion position as an integer and a boolean indicating whether the query exists.                                                        | Return `(insertion_index, found)` using binary search. Input must already be sorted ascending, with NaNs at the end.                 |

Inputs are not modified. NaNs are ordered last. `topk` accepts `0 <= k <= n`; `quickselect` requires `0 <= kth < n`. Binary search assumes sorted input.

```python
from numcompute.sort_search import stable_sort, quicksort, lexsort, topk, top_k, quickselect, binary_search

values = [8, 1, 4, 2, 7, 3]
ordered = stable_sort(values)                   # [1, 2, 3, 4, 7, 8]
manual_sort = quicksort(values)                 # [1, 2, 3, 4, 7, 8]
largest, indices = topk(values, 2)               # [8, 7], [0, 4]
values_only = top_k(values, 2)                   # [8, 7]; indices disabled by default
selected = quickselect(values, 3)               # 4
position, found = binary_search(ordered, 4)     # 3, True
order = lexsort(([20, 10, 30], [1, 0, 1]))      # [1, 0, 2]; second key is primary
```

### `numcompute.rank`: ranks and percentiles

| Function              | Parameters                                                                                                                                       | Return                                                                                                                                               | Purpose                                                                                                                               |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `rank()`            | `values` `(n,)`; `method`: `"average"` (default), `"min"`, `"max"`, `"dense"`, `"ordinal"`; `ascending=True`                   | A floating-point NumPy array `(n,)` containing one-based ranks aligned with the original rows; ties follow method and NaN inputs retain NaN ranks. | Return one-based ranks in the original input order.                                                                                   |
| `rankdata()`        | Same parameters as rank(): `values`; `method`: `"average"` (default), `"min"`, `"max"`, `"dense"`, `"ordinal"`; `ascending=True` | The same floating-point ranks `(n,)` as rank(), in input order; ties follow method and NaNs retain NaN ranks.                                      | `rankdata` remains available as a compatibility alias. It is the same callable as `rank`, with identical parameters and behavior. |
| `percentile()`      | `values`; scalar or 1-D `q` in `[0, 100]`; `interpolation="linear"` (all 13 choices below)                                               | A float for scalar q, or an array matching q for multiple percentiles; NaNs are omitted and at least one non-NaN sample value is required.           | Return percentile value(s) for `q` in the range 0–100.                                                                             |
| `percentile_rank()` | `values`; scalar or 1-D `scores`; `kind`: `"mean"` (default), `"rank"`, `"weak"`, `"strict"`                                       | A float for a scalar score, otherwise an array matching scores; values are percentile positions from 0 to 100, with NaN for NaN scores.              | Return the percentile rank (0–100) for one or more scores relative to a sample.                                                      |

Percentile functions omit NaNs from the sample. Options for `rank` also apply to its `rankdata` alias:

- `method="average"`: ties share their average position; `"min"`/`"max"` use the smallest/largest tied position; `"dense"` gives consecutive ranks to distinct values; `"ordinal"` breaks ties in input order.
- `ascending=True`: smaller values receive smaller ranks; False assigns smaller ranks to larger values.
- `percentile_rank(kind="mean")`: average the percentages strictly below and at-or-below the score. `"strict"` counts values below; `"weak"` counts values at or below; `"rank"` uses average matching ranks as a percentage of the sample size, falling back to weak when no match exists.
- `percentile(interpolation="linear")`: supported choices are `"inverted_cdf"`, `"averaged_inverted_cdf"`, `"closest_observation"`, `"interpolated_inverted_cdf"`, `"hazen"`, `"weibull"`, `"linear"`, `"median_unbiased"`, `"normal_unbiased"`, `"lower"`, `"higher"`, `"midpoint"`, `"nearest"`. These are passed to NumPy's percentile method; the default is `"linear"`.

```python
from numcompute.rank import percentile, percentile_rank, rank, rankdata

scores = [30, 10, 20, 20]
ranks = rank(scores)                               # [4.0, 1.0, 2.5, 2.5]
alias_ranks = rankdata(scores)                      # same result as rank(scores)
dense = rank(scores, method="dense")                # [3.0, 1.0, 2.0, 2.0]
rank_of_20 = percentile_rank(scores, 20)             # 50.0
quartiles = percentile([0, 10, 20, 30], [25, 50, 75])  # [7.5, 15.0, 22.5]
```

### `numcompute.stats`: descriptive and streaming statistics

| Function                                       | Parameters                                                                                                                                                  | Return                                                                                                                                                              | Purpose                                                                                                                                                                 |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `mean()`, `median()`, `min()`, `max()` | Numeric `x`; `axis=None`; `skip_nan=False`                                                                                                            | A Python float when axis=None; otherwise a reduced array with the selected axis removed. For `(n, d)` input, axis=0 returns `(d,)` and axis=1 returns `(n,)`. | Summarize the centre or bounds of the data using the mean, median, minimum or maximum, either across all values or separately along an axis.                            |
| `std()`                                      | `x`; `axis=None`; `ddof=0`; `skip_nan=False`                                                                                                        | A Python float when axis=None, otherwise the reduced array; axis=0 on `(n, d)` returns one standard deviation per feature `(d,)`.                               | Measure spread around the mean using divisor `count - ddof`. Use ddof=0 for population spread or ddof=1 for a sample estimate, optionally ignoring NaNs.              |
| `quantile()`                                 | `x`; `q` in `[0, 1]`; `axis=None`; `method`: `"linear"` (default), `"lower"`, `"higher"`, `"midpoint"`, `"nearest"`; `skip_nan=False` | A float for scalar q with axis=None; otherwise an array of quantiles. Multiple q values add a leading quantile axis, e.g. `(len(q), d)` for axis=0.               | Find data values at requested cumulative probabilities, such as quartiles `[0.25, 0.5, 0.75]`, with a chosen interpolation and NaN policy.                            |
| `histogram()`                                | `x`; `bins=10` (positive integer, bin-edge sequence or NumPy bin-rule string); `range=None` or `(low, high)`; `density=False`                     | A pair of NumPy arrays `(counts, edges)`: counts or densities `(b,)` and bin boundaries `(b+1,)`. The final bin includes its right boundary.                  | Summarize the distribution of flattened, non-NaN values using bin frequencies or densities, making ranges, concentration and long tails visible.                        |
| `RunningStats()`                             | `skip_nan=False`                                                                                                                                          | An empty RunningStats accumulator with n=0; per-feature mean, m2, min and max arrays are created when a retained batch is first added.                              | Maintain mean, variance, minimum and maximum across successive batches using Chan/Welford moment merging, keeping persistent storage proportional to the feature count. |
| `running.update(x)`                          | Scalar, `(b,)` single-feature samples, or `(b, d)` batch                                                                                                | The same accumulator (`self`), with n and feature statistics updated. Empty batches, including batches fully removed by skip_nan, leave its state unchanged.      | Merge a new batch into previously stored moments. A 1-D vector represents many samples of one feature; skip_nan=True drops any row containing NaN.                      |
| `running.var()`, `running.std()`           | `ddof=0`                                                                                                                                                  | A NumPy array `(d,)` containing variance or standard deviation for each feature over all retained rows, using denominator `n - ddof`.                           | Read the accumulated feature spread without reloading previous batches. Data must already have been added and the retained row count must exceed ddof.                  |

For `(n, d)` data, `axis=0` summarizes each feature and `axis=1` each row; `axis=None` reduces all values. `skip_nan=False` propagates NaNs and True ignores them in batch reductions. In contrast, `RunningStats(skip_nan=True)` drops entire rows containing NaN; `running.n` counts retained rows and `running.mean` contains per-feature means.

- `ddof=0`: population variance/standard deviation; `ddof=1` gives the sample estimate. The included value/row count must exceed ddof.
- `quantile(method="linear")`: interpolate between adjacent sample values. `"lower"`/`"higher"` choose the lower/upper value, `"midpoint"` averages them and `"nearest"` selects the nearest.
- `histogram(bins=10)`: an integer chooses that many equal-width bins; a sequence gives explicit increasing edges; a string selects a NumPy bin-width rule, such as `"auto"` or `"fd"`.
- `histogram(range=None)`: infer the range from the data or provide `(low, high)` for automatic edges; values outside are ignored. Explicit bin edges determine their own range.
- `histogram(density=False)`: return counts; True returns densities whose integral over the bins is one. Histogram always discards NaNs.

```python
import numpy as np
from numcompute import stats

X = np.array([[1.0, 10.0], [3.0, np.nan], [5.0, 30.0]])
means = stats.mean(X, axis=0, skip_nan=True)           # [3, 20]
medians = stats.median(X, axis=0, skip_nan=True)       # [3, 20]
stds = stats.std(X, axis=0, skip_nan=True)             # about [1.633, 10]
low = stats.min(X, axis=0, skip_nan=True)              # [1, 10]
high = stats.max(X, axis=0, skip_nan=True)             # [5, 30]
quartiles = stats.quantile(X[:, 0], [0.25, 0.5, 0.75])  # [2, 3, 4]
counts, edges = stats.histogram(X[:, 0], bins=2)       # [1, 2], [1, 3, 5]
running = stats.RunningStats(skip_nan=True)
running.update(X[:2]).update(X[2:])
print(running.n, running.mean, running.var())         # 2, [3, 20], [4, 100]
```

### `numcompute.metrics`: evaluation

| Function               | Parameters                                                                                           | Return                                                                      | Purpose                                                                                                                                                                    |
| ---------------------- | ---------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `accuracy()`         | `y_true`, `y_pred`, each `(n,)`                                                                | Fraction of matching labels, `float`                                      | Measure overall classification correctness by dividing the number of matching true/predicted labels by the total number of samples.                                        |
| `precision()`        | `y_true`, `y_pred`; `average="binary"` (choices below); `pos_label=1`; `zero_division=0.0` | `float`, or shape `(k,)` when `average=None`                          | Calculate `TP / (TP + FP)` for a class: among the samples predicted as that class, determine how many predictions are correct; optionally average over classes.          |
| `recall()`           | `y_true`, `y_pred`; `average="binary"` (choices below); `pos_label=1`; `zero_division=0.0` | `float`, or shape `(k,)` when `average=None`                          | Calculate `TP / (TP + FN)` for a class: among its true samples, determine how many are correctly detected; optionally average over classes.                              |
| `f1()`               | `y_true`, `y_pred`; `average="binary"` (choices below); `pos_label=1`; `zero_division=0.0` | `float`, or shape `(k,)` when `average=None`                          | Combine precision and recall through their harmonic mean, calculated per class before applying the requested averaging rule; use zero_division for undefined ratios.       |
| `confusion_matrix()` | `y_true`, `y_pred`; `labels=None` selects sorted classes                                       | Counts, shape `(k, k)`: rows = true label, columns = predicted label      | Count all true/predicted class combinations to reveal correct predictions and which classes are confused; optional labels controls the row and column order.               |
| `mse()`              | Numeric `y_true`, `y_pred`, each `(n,)`                                                        | Mean squared error, `float`                                               | Measure regression error as the mean of squared differences between true and predicted values, giving larger errors more weight than smaller errors.                       |
| `roc_curve()`        | `y_true`, `y_score`; `pos_label=1`                                                             | `fpr, tpr, thresholds`, each shape `(m + 1,)` for `m` distinct scores | Evaluate binary classification across score thresholds by calculating false-positive and true-positive rates. Samples with tied scores cross a threshold together (bonus). |
| `auc()`              | Equal-length `x`, `y`; `x` monotonic                                                           | Area under any monotonic curve (trapezoid rule), `float`                  | Integrate supplied curve points with the trapezoid rule. With ROC false-positive/true-positive rates as inputs, summarize binary discrimination with ROC AUC (bonus).      |

Options shared by `precision`, `recall` and `f1`:

- `average="binary"`: score only pos_label (at most two observed classes); `"macro"`: unweighted mean of class scores; `"micro"`: pool class counts before scoring; `"weighted"`: weight class scores by their true sample counts; None: return scores for each class in sorted label order.
- `pos_label=1`: positive class for binary averaging; it can be a numeric or string label compatible with the input. `roc_curve` uses the same parameter to identify its positive class.
- `zero_division=0.0`: replacement when a score denominator is zero; supply a non-NaN numeric value (not a boolean) to customize it.

For `confusion_matrix`, `labels=None` selects sorted observed classes; supply a unique label sequence to choose the row/column order. It must include every observed class and may include extra classes. Use `average="macro"` for the six-city example. Inputs must be nonempty 1-D arrays without NaN and classification labels must have compatible types. Exclude rows with blank true city labels before evaluating predictions.

```python
from numcompute import metrics

y_true, y_pred = [0, 1, 1, 0], [0, 1, 0, 0]
accuracy = metrics.accuracy(y_true, y_pred)       # 0.75
precision = metrics.precision(y_true, y_pred)    # 1.0
recall = metrics.recall(y_true, y_pred)          # 0.5
f1 = metrics.f1(y_true, y_pred)                  # about 0.667
matrix = metrics.confusion_matrix(y_true, y_pred)  # [[2, 0], [1, 1]]
error = metrics.mse([1, 2, 3], [1, 2, 5])       # 1.333...
fpr, tpr, thresholds = metrics.roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
area = metrics.auc(fpr, tpr)                     # 0.75
```

### `numcompute.optim`: finite differences

| Function       | Parameters                                                                                             | Return                                                                                                                                             | Purpose                                                                                                                                                              |
| -------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `grad()`     | Scalar-output callable `f`; finite `x` `(n,)`; `h=1e-5`; `method="central"` or `"forward"` | A floating-point NumPy array `(n,)`; element i approximates the partial derivative of scalar output f with respect to input coordinate i at x.   | Estimate a scalar function's local sensitivity to each input using central or forward finite differences, without requiring an analytical derivative or modifying x. |
| `jacobian()` | Vector-output callable `F` returning `(m,)`; `x`, `h`, `method` as above                     | A floating-point NumPy matrix `(m, n)`; entry `[j, i]` approximates the derivative of output `F[j]` with respect to input coordinate i at x. | Estimate how every component of a vector-valued function responds to each input coordinate, using repeated finite-difference evaluations.                            |

The step must be positive and finite. Inputs are not modified. These functions estimate derivatives; they do not train a model.

```python
import numpy as np
from numcompute.optim import grad, jacobian

x = np.array([3.0, 4.0])
g = grad(lambda x: np.sum(x**2), x)                 # about [6, 8]
J = jacobian(lambda x: np.array([x[0]**2 + x[1], x[0] * x[1]]), x)
print(J)                                          # about [[6, 1], [4, 3]]
```

### `numcompute.pipeline`: chained processing

| Function                 | Parameters                                                                  | Return                                                                                                                                            | Purpose                                                                                                                                                                  |
| ------------------------ | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Pipeline()`           | `steps`: nonempty list of `(unique_name, component)` pairs              | A new unfitted Pipeline holding the supplied named component objects in order; components are stored by reference and fitted in place.            | Connect reusable preprocessing steps and an optional final estimator so each component receives the previous step's output through a consistent interface.               |
| `pipe.fit()`           | `X` `(n, d)`; `y=None` (forwarded to a final estimator when supplied) | The same pipeline instance (`self`) after successful fitting; intermediate preprocessors and the final component retain their learned state.    | Fit and transform each intermediate step, then fit the final component. Targets y are passed only to a final estimator providing predict when y is supplied.             |
| `pipe.transform()`     | New `X` with compatible features                                          | The final transformer's output, usually a NumPy array `(n, c)`; the number of columns depends on the transformations, such as one-hot encoding. | Pass new rows through every fitted transformer in sequence without relearning statistics. The final component must also provide transform.                               |
| `pipe.fit_transform()` | `X`; `y=None`; final step must provide `transform`                    | The transformed training data from the final transformer, usually `(n, c)`, with all learned preprocessing state retained in the pipeline.      | Fit the chain and then transform the same training input in one call. Intended for transformer-ending pipelines; estimator-only final steps instead use fit and predict. |
| `pipe.predict()`       | New `X`; final step must provide `predict`                              | The final estimator's predictions, usually one label or numeric value per row `(n,)`; the estimator determines the exact output type and shape. | Transform new features using the fitted intermediate steps, then call the final estimator's predict method. Pipeline itself does not implement a prediction algorithm.   |

The minimal `Transformer` protocol requires `fit(X)` and `transform(X)`; the `Estimator` protocol requires `fit(X, y)` and `predict(X)`. Inheritance is not required. Pipeline fits/transforms intermediate steps, then fits the last step. Prediction reuses fitted transformers before calling the final estimator. No ML model implementation is required; a preprocessing-only pipeline uses `transform`, and the notebook includes a small estimator-interface demo.

```python
import numpy as np
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import SimpleImputer, StandardScaler

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
X_train = np.array([[1.0], [np.nan], [3.0]])
X_ready = pipe.fit_transform(X_train)             # about [[-1.225], [0], [1.225]]
new_ready = pipe.transform([[2.0]])               # [[0.0]]; no refitting
```

### `numcompute.utils`: batching and numerically stable helpers

| Function                | Parameters                                  | Return                                                                                                                                                  | Purpose                                                                                                                                              |
| ----------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `iter_batches()`      | `X` `(n, ...)`; positive `batch_size` | An iterator yielding row-slice arrays `(b, ...)` with b <= batch_size; the final batch may be shorter. Slices are views of the converted input array. | Process an array a few rows at a time without allocating separate copies of each batch, for example when feeding a streaming statistics accumulator. |
| `iter_line_batches()` | `items`; positive integer `batch_size`  | An iterator yielding lists of at most batch_size items; the last list may be shorter and an empty iterable yields no batches.                           | Consume files, generators or other iterables lazily in bounded-size groups, without loading the entire input into memory first.                      |
| `logsumexp()` | `x`; `axis=None` | A float for `axis=None`, otherwise the input shape with the axis removed. | Compute `log(sum(exp(x)))` with the max shift, so large or very negative values neither overflow nor underflow. |
| `softmax()` | `x`; `axis=-1` | Probabilities with the same shape as `x`, summing to 1 along `axis`. | Max-shifted softmax: subtract each slice's maximum before exponentiating, so large scores do not overflow. |
| `sigmoid()` | `x` | A float for scalar input, otherwise an array with the same shape. | Logistic function written with `exp(-abs(x))`, so extreme inputs give 0 or 1 without overflow. |
| `pairwise_distances()` | `X` `(n, d)`; `Y=None` or `(m, d)` | Euclidean distances `(n, m)`; with `Y=None`, `(n, n)` with a zero diagonal. | All row-to-row distances through one matrix product, after centring the data to limit cancellation. |

```python
import numpy as np
from numcompute.utils import iter_batches, iter_line_batches, logsumexp, pairwise_distances, sigmoid, softmax

X = np.arange(10).reshape(5, 2)
print([batch.shape for batch in iter_batches(X, 2)])  # [(2, 2), (2, 2), (1, 2)]
print(list(iter_line_batches(["a", "b", "c"], 2)))    # [['a', 'b'], ['c']]
print(logsumexp([1000.0, 1000.0]))                     # 1000.693..., no overflow
print(softmax([[1.0, 2.0, 3.0]]))                      # [[0.09  0.245 0.665]]
print(sigmoid([-1000.0, 0.0, 1000.0]))                 # [0.  0.5 1. ]
print(pairwise_distances([[0.0, 0.0], [3.0, 4.0]]))    # [[0. 5.] [5. 0.]]
```

### `numcompute.benchmarking`: loop vs NumPy timing

| Function               | Parameters                                                                                      | Return                                                                                                                                              | Purpose                                                                                                                                                 |
| ---------------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `time_function()`    | No-argument `func`; `repeat=7`; `number=1`; `warmup=1`                                  | Seconds per call for each repeat, shape `(repeat,)`                                                                                               | Warm up a no-argument callable and measure repeated executions. number controls calls per measurement, with results reported as seconds per call.       |
| `compare()`          | `case`, `loop_func`, `vec_func`; `repeat=7`; `number=1`; `rtol=1e-7`; `atol=1e-9` | `{"case", "loop_s", "vec_s", "speedup"}`: median seconds per call; `speedup = loop_s / vec_s`                                                   | Check that loop and NumPy results match before timing, then compare median execution times and calculate speedup as loop time divided by NumPy time.    |
| `environment_info()` | None                                                                                            | Python, NumPy, OS, CPU and CPU count, `dict[str, str]`                                                                                            | Record Python, NumPy, operating-system and CPU information alongside results so measurements can be interpreted and reproduced on another environment.  |
| `format_table()`     | `rows`: list of comparison dicts                                                              | `compare` results as a Markdown table (the columns of the performance table below); an optional `"notes"` entry in a row fills the Notes column | Turn comparison dictionaries into a readable Markdown performance table with timing, speedup and optional explanatory notes for the README or notebook. |

Loop baselines live separately in `benchmark/loops.py`. Data generation and array-to-list conversion happen before timing. `compare` first checks matching results, then reports median runtimes and `speedup = loop_s / vec_s`. Rank, top-k and AUC cases compare algorithm choices as well as vectorization.

The following 10 Python-loop baselines are available in `benchmark/loops.py` through `from benchmark import loops`:

| Loop function                     | Compared with                                | Loop implementation                                                                                           |
| --------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `loops.load_csv_loop()`         | `io.load_csv(dtype=float)`                 | Read CSV rows and convert individual cells to numbers, replacing missing tokens with NaN.                     |
| `loops.accuracy_loop()`         | `metrics.accuracy()`                       | Count matching true/predicted labels and divide by the sample count.                                          |
| `loops.confusion_matrix_loop()` | `metrics.confusion_matrix()`               | Look up label indices and increment a count for each true/predicted pair.                                     |
| `loops.mse_loop()`              | `metrics.mse()`                            | Sum squared prediction errors and divide by the sample count.                                                 |
| `loops.auc_loop()`              | `metrics.auc(*metrics.roc_curve(...)[:2])` | Compare every positive-negative score pair, counting tied scores as half a win; compare the final AUC only.   |
| `loops.standard_scale_loop()`   | `StandardScaler().fit_transform()`         | Calculate each column's mean and population standard deviation, then standardize individual values.           |
| `loops.histogram_loop()`        | `stats.histogram(...)[0]`                  | Assign values to equal-width bins and increment their counts; compare counts only.                            |
| `loops.stable_sort_loop()`      | `sort_search.stable_sort()`                | Sort using a handwritten stable merge sort.                                                                   |
| `loops.topk_loop()`             | `sort_search.topk()`                       | Find the largest unselected value in each of k full scans, returning values and original indices.             |
| `loops.average_rank_loop()`     | `rank.rank(method="average")`              | Compare each value with every other value to count smaller and equal values, then calculate its average rank. |

```python
import numpy as np
from numcompute import metrics
from numcompute.benchmarking import compare, format_table
from benchmark import loops

a, b = np.random.default_rng(42).normal(size=(2, 100_000))
a_list, b_list = a.tolist(), b.tolist()            # conversion is not timed
row = compare("mse", lambda: loops.mse_loop(a_list, b_list),
              lambda: metrics.mse(a, b))
print(format_table([row]))                         # loop/NumPy times and speedup
```

## Performance table

Python loops (`benchmark/loops.py`) against NumCompute on the same inputs; both versions are checked to give the same result before they are timed. Reproduce the full benchmark from the project root with:

```bash
python -m benchmark.run_benchmarks
```

Inputs come from `np.random.default_rng(42)` at fixed sizes. Times are the median of 7 measurements per version. Baseline/result gives the Python-loop time and, in parentheses, the speedup (`loop time / NumCompute time`).

The table below is from the full report in `benchmark/results.md` (generated by notebook Section 8.2 with `run(quick=False)`, which uses the same runner as the command above). Running either again produces a new run, and the command-line runner overwrites `results.md`.

Environment: Python 3.14.7, NumPy 2.5.4, OS `Windows-11-10.0.26200-SP0`, CPU `Intel64 Family 6 Model 154 Stepping 4, GenuineIntel`, 12 logical CPUs.

Timings and speedups can differ between team members because hardware, software and system load vary, while the overall performance trend can remain consistent. Rank, top-k and AUC speedups include algorithm differences as well as vectorization.

| Operation                                | NumCompute result |       Baseline/result | Notes                                                                        |
| ---------------------------------------- | ----------------: | --------------------: | ---------------------------------------------------------------------------- |
| io.load_csv dtype=float 20,000 x 5       |         21.278 ms |      35.942 ms (1.7x) | Both parse text line by line; parsing dominates                              |
| metrics.accuracy n=100,000               |          0.204 ms |      4.474 ms (22.0x) | One comparison per element, done in C                                        |
| metrics.confusion_matrix n=100,000       |          4.452 ms |       9.899 ms (2.2x) | Sorts the labels (np.union1d) to accept any label type; the loop uses a dict |
| metrics.mse n=100,000                    |          0.138 ms |      9.087 ms (65.9x) | Element-wise O(n) in C instead of the interpreter                            |
| metrics.roc_curve + auc n=2,000          |          0.168 ms |    25.979 ms (154.5x) | The loop compares every positive-negative pair, O(P*N); NumPy sorts once     |
| preprocessing.StandardScaler 20,000 x 10 |          3.567 ms |     45.506 ms (12.8x) | Column means and stds over axis 0, one pass each                             |
| stats.histogram n=100,000 bins=20        |          2.008 ms |      10.627 ms (5.3x) | Bins all values at once                                                      |
| sort_search.stable_sort n=100,000        |          6.610 ms |    187.249 ms (28.3x) | Merge sort in Python vs np.sort(kind='stable')                               |
| sort_search.topk n=100,000 k=10          |          1.048 ms |     40.484 ms (38.6x) | k passes over the data vs np.argpartition                                    |
| rank.rank method='average' n=2,000       |          0.087 ms | 120.788 ms (1,386.8x) | The loop compares every pair, O(n^2); NumPy sorts once                       |

## Tests

Run all tests from the project root after installing the test dependencies with `python -m pip install -e ".[test]"`:

```bash
python -m pytest tests -q
```

Latest local verification (25 October 2026): **212 tests passed** across 10 test files, using Python 3.14.7 and NumPy 2.5.4.

| Test file                 | Coverage                                                                                                                                                        |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_io.py`            | Headers/BOM, custom delimiters, dtype conversion, missing tokens, fill/skip/raise policies, column selection, empty files and chunked reading.                  |
| `test_preprocessing.py` | Scaling results, constant columns, NaN handling, imputation strategies, categorical encoding, unseen categories and input validation.                           |
| `test_sort_search.py`   | Sorting and stability, multi-key order, top-k values/indices and boundary k, quickselect, duplicate search results and insertion positions.                     |
| `test_rank.py`          | Average/min/max/dense/ordinal ties, descending ranks, compatibility alias, percentile-rank methods, interpolation and NaN handling.                             |
| `test_stats.py`         | Axis-wise summaries, quantile shapes, histogram counts/density, all-equal values and streaming moments with large-offset data.                                  |
| `test_metrics.py`       | Classification and regression results, averaging modes, zero denominators, confusion-matrix label order, tied ROC scores and AUC.                               |
| `test_optim.py`         | Forward/central differences against analytical derivatives, rectangular Jacobian orientation, invalid step sizes, changing output shapes and nonfinite results. |
| `test_pipeline.py`      | Chained preprocessing, reuse of fitted parameters, estimator fit/predict dispatch, fitted-state errors, failed refits and feature-count checks.                 |
| `test_utils.py`         | Array validation, row-batch views, lazy iterable batching, invalid batch sizes, overflow-free logsumexp/softmax/sigmoid, infinite/NaN inputs and distances far from the origin. |
| `test_benchmarking.py`  | Timing output/call counts, result-equivalence checks, validation, environment/table formatting and all 10 quick benchmark cases through the runner.             |

Edge cases include empty arrays, all-equal values, duplicates/ties, NaNs, non-contiguous array views and invalid types/shapes. Benchmark tests verify correctness and report generation rather than requiring a fixed runtime or speedup, since timings depend on the machine.

To run one test file:

```bash
python -m pytest tests/test_pipeline.py -q
```

For a fast benchmark check:

```bash
python -m benchmark.run_benchmarks --quick
```

## Additional Information

### Sample dataset

The dataset and generator described here follow the repository's `main` branch. `demo/data/sample.csv` contains 3,000 synthetic records from six fictional city profiles, with 500 records generated per city before hiding some labels. `age` and `income` are input features, `city` is the true class, and `city_predict` is a rule-based simulated prediction rather than output from a trained ML model.

- Missing values: 180 ages, 240 incomes and 80 true city labels. Numeric missing tokens include `""`, `"NA"`, `"NaN"`, `"NULL"` and `"N/A"`; `city_predict` is complete.
- Repeated incomes: 14 Sydney rows at 115,000 and 21 Melbourne rows at 90,000 demonstrate tied ranking methods.
- Extreme incomes: three rows per city have income equal to six times that city's income base, illustrating scaling effects and distribution tails.
- Evaluation: exclude the 80 blank true city labels from classification metrics. Too many missing labels reduce the number of samples available for evaluation.

### Demo files and directory structure

The directory tree above includes `demo/generate_sample.py`, which generates `demo/data/sample.csv` reproducibly with seed 42, and `demo/quickstart.ipynb` for package demonstrations. The separate `demo/sample.csv` is an older 8-row example with an additional height column for older version testing. To reproduce the documented data, use the generator from `main`.
