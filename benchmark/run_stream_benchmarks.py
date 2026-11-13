"""Streaming benchmarks in the original run_benchmarks.py style.

Run python -m benchmark.run_stream_benchmarks [--quick] [--output FILE].
Update timings create fresh state; queries/prediction reuse prepared state.
Result checks precede timings.
"""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
from benchmark import stream_loops as loops
from numcompute_stream import metrics, preprocessing, stats
from numcompute_stream.tree import DecisionTreeClassifier
from numcompute_stream.benchmarking import compare, environment_info, format_table, benchmark_buffers, format_buffer_table

SEED = 42
DEFAULT_OUTPUT = Path(__file__).with_name("stream_results.md")
FULL = {"n": 50_000, "d": 6, "chunk_size": 500, "n_split": 400, "n_auc": 1000,
        "n_dist": 5000, "n_cat": 20_000, "n_stream": 6000, "stream_chunk": 200,
        "buffers": (100, 500, 1000, 2000, 4000), "repeat": 5}
QUICK = {"n": 1000, "d": 3, "chunk_size": 200, "n_split": 60, "n_auc": 100,
         "n_dist": 600, "n_cat": 1000, "n_stream": 1000, "stream_chunk": 100,
         "buffers": (100, 200, 400, 600, 800), "repeat": 2}


def _chunks(X, size):
    return [X[start:start+size] for start in range(0, len(X), size)]


def _stats(chunks):
    state = stats.RunningStats(store_data=False)
    for chunk in chunks:
        state.update(chunk)
    return np.array([state.count_, state.mean, state.var()])


def _transform(chunks, constructor):
    state = constructor()
    for chunk in chunks:
        state.partial_fit(chunk)
    return np.vstack([state.transform(chunk) for chunk in chunks])


def _metric(chunks, constructor):
    state = constructor()
    for truth, prediction in chunks:
        state.update(truth, prediction)
    return state.result()


def build_cases(cfg, rng):
    """Return (name, loop_callable, vector_callable, notes), using seeded inputs.

    Input/list conversion and fitting the shared prediction tree happen outside
    timing. Each timed update sequence creates its own fresh state. Query
    cases reuse retained history without modifying it. Missing
    numeric values, a constant feature and tied scores
    exercise matching semantics; timings include normal validation overhead.
    """
    n, d, size = cfg["n"], cfg["d"], cfg["chunk_size"]
    X = rng.normal(5, 2, (n, d))
    X[:, -1] = 3
    X[rng.random(X.shape) < 0.03] = np.nan
    chunks = _chunks(X, size)
    lists = [chunk.tolist() for chunk in chunks]
    a, b = rng.normal(size=n), rng.normal(size=n)
    regression = list(zip(_chunks(a, size), _chunks(b, size)))
    regression_lists = [(t.tolist(), p.tolist()) for t, p in regression]
    binary = np.arange(cfg["n_auc"]) % 2
    scores = np.round(rng.random(len(binary))+0.3*binary, 2)
    auc_chunks = list(zip(_chunks(binary, size), _chunks(scores, size)))
    auc_lists = [(t.tolist(), p.tolist()) for t, p in auc_chunks]
    split_X = rng.normal(size=(cfg["n_split"], 4))
    split_y = (split_X[:, 0]+split_X[:, 1] > 0).astype(int)
    split_X[rng.random(split_X.shape) < 0.03] = np.nan
    trained = DecisionTreeClassifier(max_depth=5).fit(split_X, split_y)
    prediction_rows = rng.normal(size=(n, 4))
    prediction_rows[rng.random(prediction_rows.shape) < 0.03] = np.nan
    prediction_lists = prediction_rows.tolist()

    # Distribution queries start with the same already retained scalar chunks.
    distribution_chunks = _chunks(X[:cfg["n_dist"]], size)
    distribution_lists = [chunk.tolist() for chunk in distribution_chunks]
    scalar_chunks = [chunk[:, 0] for chunk in distribution_chunks]
    scalar_lists = [chunk.tolist() for chunk in scalar_chunks]
    distribution = stats.RunningStats()
    for chunk in scalar_chunks:
        distribution.update(chunk)
    edges = np.linspace(-5, 15, 21)
    edge_list, probabilities = edges.tolist(), [0.25, 0.5, 0.75]

    categories = np.char.add("c", rng.integers(0, 8, (cfg["n_cat"], 3)).astype(str))
    categories[rng.random(categories.shape) < 0.03] = ""
    # Widen the string array before adding the longer missing markers.
    categories = categories.astype("U8")
    categories[0, :2] = ["nan", "None"]
    category_chunks = _chunks(categories, size)
    category_lists = [chunk.tolist() for chunk in category_chunks]

    def stump(cls, criterion="gini"):
        tree = cls(max_depth=1, criterion=criterion).fit(split_X, split_y)
        root = tree.root_
        return (root.feature, root.threshold, root.nan_left)
    return [
        (f"stats.RunningStats n={n:,}, d={d}, chunk={size}",
         lambda: loops.running_stats_loop(lists), lambda: _stats(chunks), "Per-cell Welford vs vectorised chunk merging; NaNs ignored per feature"),
        (f"preprocessing.StandardScaler n={n:,}, d={d}",
         lambda: loops.standard_scaler_loop(lists), lambda: _transform(chunks, preprocessing.StandardScaler), "Fresh partial_fit sequence, then transform using final parameters"),
        (f"preprocessing.MinMaxScaler n={n:,}, d={d}",
         lambda: loops.minmax_scaler_loop(lists), lambda: _transform(chunks, preprocessing.MinMaxScaler), "Learn cumulative minima/maxima, then scale; same constant-column and NaN rules"),
        (f"preprocessing.SimpleImputer(mean) n={n:,}, d={d}",
         lambda: loops.imputer_loop(lists), lambda: _transform(chunks, preprocessing.SimpleImputer), "Observed cells only; missing estimates updated across chunks"),
        (f"preprocessing.SimpleImputer(median) n={cfg['n_dist']:,}, d={d}",
         lambda: loops.median_imputer_loop(distribution_lists),
         lambda: _transform(distribution_chunks, lambda: preprocessing.SimpleImputer(strategy="median")),
         "Exact medians updated after every chunk on both sides; then fill using final medians"),
        (f"preprocessing.OneHotEncoder n={cfg['n_cat']:,}, d=3",
         lambda: loops.onehot_encoder_loop(category_lists),
         lambda: _transform(category_chunks, preprocessing.OneHotEncoder),
         "Learn categories across chunks, then create the same sorted dense encoding; blanks/nan/None are missing"),
        (f"stats.histogram n={cfg['n_dist']:,}, bins=20",
         lambda: loops.histogram_loop(scalar_lists, edge_list), lambda: distribution.histogram(bins=edges)[0],
         "Query already saved chunks with identical fixed edges; NaNs skipped and last edge included"),
        (f"stats.quantile(linear) n={cfg['n_dist']:,}",
         lambda: loops.quantile_loop(scalar_lists, probabilities),
         lambda: distribution.quantile(probabilities, axis=None),
         "Query already saved chunks; same linear quartiles; Python sorts while NumPy may use selection"),
        (f"metrics.MSE n={n:,}", lambda: loops.mse_loop(regression_lists),
         lambda: _metric(regression, metrics.MSE), "Squared errors accumulated across chunks"),
        (f"metrics.AUC n={len(binary):,}", lambda: loops.auc_loop(auc_lists),
         lambda: _metric(auc_chunks, metrics.AUC), "Different algorithms: compare all positive-negative pairs vs sort scores once; same exact AUC"),
        (f"tree split search(gini) n={cfg['n_split']}, d=4", lambda: stump(loops.LoopDecisionTreeClassifier),
         lambda: stump(DecisionTreeClassifier), "Different algorithms: scan rows for each threshold vs cumulative counts; same depth-one fit and NaN rules; list conversion included"),
        (f"tree split search(entropy) n={cfg['n_split']}, d=4",
         lambda: stump(loops.LoopDecisionTreeClassifier, "entropy"), lambda: stump(DecisionTreeClassifier, "entropy"),
         "Same entropy criterion and zero-count rules; repeated row scans vs cumulative class counts"),
        (f"tree.predict n={n:,}", lambda: loops.predict_loop(trained, prediction_lists),
         lambda: trained.predict(prediction_rows), "Same fitted tree; per-row traversal vs grouped NumPy traversal"),
    ]


def stream_data(cfg, rng):
    """Fixed-order binary stream with noisy labels and missing features."""
    X = rng.normal(size=(cfg["n_stream"], 4))
    y = (X[:, 0]+0.5*X[:, 1] > 0).astype(int)
    y = np.where(rng.random(len(y)) < 0.1, 1-y, y)
    X[rng.random(X.shape) < 0.02] = np.nan
    return X, y


def run(quick=False):
    """Return the full Markdown report, without writing files (original style)."""
    cfg = QUICK if quick else FULL
    start = perf_counter()
    rows = []
    for name, loop_func, vec_func, notes in build_cases(cfg, np.random.default_rng(SEED)):
        print(f"  {name} ...", flush=True)
        row = compare(name, loop_func, vec_func, repeat=cfg["repeat"], rtol=1e-6)
        row["notes"] = notes
        rows.append(row)
    X, y = stream_data(cfg, np.random.default_rng(SEED+1))
    buffers = benchmark_buffers(X, y, cfg["buffers"], chunk_size=cfg["stream_chunk"], repeat=cfg["repeat"])
    command = "python -m benchmark.run_stream_benchmarks" + (" --quick" if quick else "")
    lines = ["# Streaming benchmark results", "", f"Generated by `{command}`; seed={SEED}, repeats={cfg['repeat']}, elapsed={perf_counter()-start:.1f} s.", "",
             "## Environment", ""]
    lines += [f"- {key}: {value}" for key, value in environment_info().items()]
    lines += ["", "## Performance table", "",
              "All function comparisons are in one table. Answers are checked before timing. Update cases start fresh; histogram/quantile queries reuse saved history, and prediction uses the same fitted tree. Input lists are prepared beforehand, apart from internal node-to-list conversion in the split loop. Normal validation stays included. Times are medians; speed-up = loop/NumPy time, including values below 1.", "",
              "Accuracy is not included: the current class builds confusion counts, while a simple accuracy loop only counts matches. ConfusionMatrix is also left out of the speed-up examples: dictionary counting and per-chunk label/schema management add different work. Their correctness is still checked by the unit tests and demo.", "", format_table(rows), "",
              "Notes explain algorithm differences too: AUC uses sorting instead of pair checks; split search uses cumulative counts instead of repeated scans; quantiles may use selection instead of sorting. Those speed-ups are not only due to vectorisation. Hardware and background load affect timings.", "",
              "## Buffer-size comparison", "", f"Stream: {cfg['n_stream']} rows, 4 features, chunks of {cfg['stream_chunk']}, max_depth=5, tree seed=42. Independent seed={SEED+1} generates this stream.", "",
              "All configurations score the same chunks after the largest buffer first trains, before training on those rows. Final pending buffers are not flushed. Total update time excludes prediction; mean retrain call includes validation and buffer concatenation. Timings are medians across fresh runs; accuracy need not increase with buffer size.", "", format_buffer_table(buffers)]
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Streaming loop/vectorised and buffer-size benchmarks")
    parser.add_argument("--quick", action="store_true", help="small smoke-test data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = run(args.quick)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
