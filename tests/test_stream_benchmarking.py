"""Streaming benchmark correctness, state isolation and fair scoring checks."""
from contextlib import contextmanager
import numpy as np
import pytest
from benchmark import run_stream_benchmarks as runner, stream_loops
from numcompute import benchmarking as old
from numcompute_stream import benchmarking as bench
from numcompute_stream.tree import DecisionTreeClassifier


def test_buffer_order_repeats_short_scoring_tail_and_input_ownership():
    X = np.arange(22.).reshape(11, 2)[::-1, ::-1]
    y = np.arange(11) % 2
    saved_X, saved_y = X.copy(), y.copy()
    first = bench.benchmark_buffers(X, y, (2, 7), chunk_size=3, repeat=1, max_depth=2)
    reverse = bench.benchmark_buffers(X, y, (7, 2), chunk_size=3, repeat=2, max_depth=2)
    for left, right in zip(first, reverse[::-1]):
        for key in ("buffer_size", "accuracy", "n_scored", "n_retrains", "pending_samples"):
            assert left[key] == right[key]
        assert left["n_scored"] == 2  # Only the last short chunk is scored.
    np.testing.assert_array_equal(X, saved_X)
    np.testing.assert_array_equal(y, saved_y)
    with pytest.raises(ValueError, match="scored chunk"):
        bench.benchmark_buffers(X[:9], y[:9], (7,), chunk_size=3, repeat=1)


@contextmanager
def case(name, **options):
    try:
        yield
    except (Exception, pytest.fail.Exception) as error:
        raise AssertionError(f"{name}, {options}: {error}") from error


def test_original_harness_is_reused_and_every_quick_case_checks_matching_outputs():
    for name in ("compare", "time_function", "environment_info", "format_table"):
        assert getattr(bench, name) is getattr(old, name)
    cases = runner.build_cases(runner.QUICK, np.random.default_rng(runner.SEED))
    assert len(cases) == 13
    for required in ("MinMaxScaler", "OneHotEncoder", "SimpleImputer(median)", "stats.histogram", "stats.quantile", "search(entropy)"):
        assert any(required in name for name, *_ in cases)
    assert not any(name.startswith(("metrics.Accuracy", "metrics.ConfusionMatrix")) for name, *_ in cases)
    for name, loop, vector, notes in cases:
        with case("benchmark output", operation=name):
            row = bench.compare(name, loop, vector, repeat=1, rtol=1e-6)
            assert row["loop_s"] >= 0 and row["vec_s"] >= 0 and notes
            # Repeating the callable must start fresh rather than retain updates.
            np.testing.assert_allclose(vector(), vector(), equal_nan=True)


def test_generated_inputs_are_reproducible_and_stream_labels_depend_on_features():
    a, b = runner.stream_data(runner.QUICK, np.random.default_rng(43)), runner.stream_data(runner.QUICK, np.random.default_rng(43))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert np.isnan(a[0]).any() and len(np.unique(a[1])) == 2


def test_loop_tree_matches_vector_tree_for_ties_missing_values_and_both_criteria():
    datasets = [([[0], [0], [1], [1]], [0, 0, 1, 1]),
                ([[np.nan], [np.nan], [1], [1]], [0, 0, 1, 1]),
                ([[1], [1], [1]], [1, 0, 1])]
    rng = np.random.default_rng(6)
    X = rng.integers(0, 5, (30, 3)).astype(float)
    X[::5, 1] = np.nan
    datasets.append((X, rng.integers(0, 3, len(X))))
    for criterion in ("gini", "entropy"):
        for X, y in datasets:
            with case("tree loop", criterion=criterion):
                vector = DecisionTreeClassifier(max_depth=3, criterion=criterion).fit(X, y)
                loop = stream_loops.LoopDecisionTreeClassifier(max_depth=3, criterion=criterion).fit(X, y)
                np.testing.assert_array_equal(vector.predict(X), loop.predict(X))
                np.testing.assert_array_equal(stream_loops.predict_loop(vector, np.asarray(X).tolist()), vector.predict(X))


def test_buffer_sizes_share_scoring_rows_and_match_manual_predictions():
    X = np.arange(26.0).reshape(13, 2)
    y = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1])
    rows = bench.benchmark_buffers(X, y, (2, 5), chunk_size=3, repeat=2, max_depth=2)
    assert [row["n_scored"] for row in rows] == [7, 7]  # Both begin after six warm-up rows.
    assert [row["n_retrains"] for row in rows] == [4, 2]
    assert [row["pending_samples"] for row in rows] == [1, 1]
    for row in rows:
        tree = DecisionTreeClassifier(max_depth=2, buffer_size=row["buffer_size"], random_state=42)
        correct = 0
        for start in range(0, len(X), 3):
            if start >= 6:
                correct += np.count_nonzero(tree.predict(X[start:start+3]) == y[start:start+3])
            tree.partial_fit(X[start:start+3], y[start:start+3])
        assert row["accuracy"] == correct/7
        assert row["train_s"] >= row["retrain_s"] >= 0


def test_oversized_chunks_retrain_once_and_invalid_settings_are_rejected():
    X, y = np.arange(40.0).reshape(20, 2), np.arange(20) % 2
    row = bench.benchmark_buffers(X, y, (2,), chunk_size=5, repeat=1)[0]
    assert row["n_retrains"] == 4 and row["n_scored"] == 15
    invalid = ({"buffer_sizes": ()}, {"buffer_sizes": (2, 2)}, {"buffer_sizes": (True,)},
               {"buffer_sizes": (0,)}, {"buffer_sizes": (100,)}, {"chunk_size": 0},
               {"chunk_size": 1.5}, {"repeat": False}, {"repeat": 0})
    for options in invalid:
        with case("invalid buffer benchmark", options=options), pytest.raises((TypeError, ValueError)):
            bench.benchmark_buffers(X, y, **options)


def test_buffer_table_reports_accuracy_counts_and_milliseconds():
    row = {"buffer_size": 5, "accuracy": 0.75, "n_scored": 20, "n_retrains": 4,
           "pending_samples": 1, "train_s": 0.02, "retrain_s": 0.005}
    text = bench.format_buffer_table([row])
    assert "0.7500" in text and "20.000 ms" in text and "5.000 ms" in text
    assert len(bench.format_buffer_table([]).splitlines()) == 2


def test_quick_report_contains_all_operations_and_fair_comparison_notes():
    report = runner.run(quick=True)
    for name in ("Environment", "RunningStats", "StandardScaler", "SimpleImputer",
                 "MinMaxScaler", "OneHotEncoder", "SimpleImputer(median)", "stats.histogram", "stats.quantile",
                 "MSE", "AUC", "split search", "tree.predict", "Buffer-size comparison", "Performance table"):
        assert name in report
    assert "## Algorithm comparisons" not in report
    assert "same chunks" in report and "not flushed" in report
    assert "| metrics.Accuracy" not in report and "| metrics.ConfusionMatrix" not in report


def test_cli_writes_the_requested_file(tmp_path):
    destination = tmp_path / "reports" / "stream.md"
    runner.main(["--quick", "--output", str(destination)])
    assert destination.is_file()
    assert "Streaming benchmark results" in destination.read_text(encoding="utf-8")


def test_added_baselines_match_missing_constant_tied_and_boundary_cases():
    from numcompute_stream import preprocessing, stats
    X = np.array([[np.nan, 2, np.nan], [1, 2, np.nan], [3, 2, np.nan], [np.nan, 2, np.nan]])
    chunks = [X[:2], X[2:]]
    lists = [chunk.tolist() for chunk in chunks]
    for loop, constructor in ((stream_loops.minmax_scaler_loop, preprocessing.MinMaxScaler),
                              (stream_loops.median_imputer_loop, lambda: preprocessing.SimpleImputer(strategy="median"))):
        with case("new numeric baseline", function=loop.__name__):
            np.testing.assert_allclose(loop(lists), runner._transform(chunks, constructor), equal_nan=True)
    for rows in ([["Sydney", ""], ["Adelaide", "nan"], ["Sydney", "None"]],
                 [["", "nan"], ["None", ""]]):
        with case("new encoder baseline", rows=rows):
            arrays = [np.asarray(rows[:1]), np.asarray(rows[1:])]
            np.testing.assert_array_equal(stream_loops.onehot_encoder_loop([a.tolist() for a in arrays]),
                                          runner._transform(arrays, preprocessing.OneHotEncoder))
    values = [np.array([np.nan, 0, 0.5]), np.array([1, 2, 3, 3.1])]
    state = stats.RunningStats()
    for chunk in values:
        state.update(chunk)
    edges, q = [0, 1, 3], [0, 0.25, 0.5, 0.75, 1]
    np.testing.assert_array_equal(stream_loops.histogram_loop([a.tolist() for a in values], edges), state.histogram(bins=edges)[0])
    np.testing.assert_allclose(stream_loops.quantile_loop([a.tolist() for a in values], q), state.quantile(q, axis=None))


def test_five_buffer_settings_use_the_same_scoring_rows():
    assert runner.FULL["buffers"] == (100, 500, 1000, 2000, 4000)
    assert len(runner.QUICK["buffers"]) == 5
    X, y = runner.stream_data(runner.QUICK, np.random.default_rng(43))
    rows = bench.benchmark_buffers(X, y, runner.QUICK["buffers"], chunk_size=100, repeat=1)
    assert len(rows) == 5 and {row["n_scored"] for row in rows} == {200}
    assert rows[-1]["pending_samples"] == 200
