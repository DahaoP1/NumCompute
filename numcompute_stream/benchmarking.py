"""Reuse the original timing harness; add a fair buffered-tree comparison."""

from numbers import Integral
from time import perf_counter
import numpy as np
from numcompute.benchmarking import time_function, compare, environment_info, format_table
from .tree import DecisionTreeClassifier

__all__ = ["time_function", "compare", "environment_info", "format_table",
           "benchmark_buffers", "format_buffer_table"]


def benchmark_buffers(X, y, buffer_sizes=(100, 500, 1000), chunk_size=200,
                      repeat=3, max_depth=5, random_state=42):
    """Compare buffer sizes on identical test-before-train observations.

    Parameters
    ----------
    X : numeric array_like, shape (n, d)
        Stream features in arrival order. NaNs are supported by the tree.
    y : array_like, shape (n,)
        Aligned finite numeric or nonblank string labels.
    buffer_sizes : iterable of distinct positive int, default (100, 500, 1000)
        Retraining thresholds; each triggering call uses the entire buffer once.
    chunk_size, repeat : positive int, defaults 200, 3
        Rows per chunk and independent fresh-model timing runs.
    max_depth, random_state : int or None, defaults 5, 42
        Depth and seed; see DecisionTreeClassifier.

    Returns
    -------
    list of dict
        buffer_size, accuracy, n_scored, n_retrains, pending_samples, train_s
        (median summed partial_fit time), retrain_s (median per-run mean time
        of buffer-triggered partial_fit calls). Prediction timing is excluded.

    Raises
    ------
    TypeError, ValueError
        If settings/data are invalid, sizes repeat, or no chunk remains after
        the largest buffer first trains.

    Notes
    -----
    Fresh models score exactly the chunks after the largest buffer's first
    training, before learning those rows. Warm-up is therefore common to every
    size. Pending final buffers are not flushed. Retrain-call time includes
    buffer concatenation and validation, not just split search. Total work is
    repeat times all training/prediction runs; memory follows the largest
    tree/buffer, plus O(n*d) inputs and chunk views.
    """
    for name, value in (("chunk_size", chunk_size), ("repeat", repeat)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be a positive integer")
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    sizes = list(buffer_sizes)
    if not sizes:
        raise ValueError("buffer_sizes must not be empty")
    for size in sizes:
        DecisionTreeClassifier(buffer_size=size, max_depth=max_depth, random_state=random_state)
    if len(set(sizes)) != len(sizes):
        raise ValueError("buffer_sizes must be distinct")
    validator = DecisionTreeClassifier()
    X = validator._check_X(X)
    y, _ = validator._check_y(y, len(X))
    chunks = [(X[start:start+chunk_size], y[start:start+chunk_size]) for start in range(0, len(X), chunk_size)]
    received = np.cumsum([len(target) for _, target in chunks])
    common_start = int(np.searchsorted(received, max(sizes))) + 1
    if common_start >= len(chunks):
        raise ValueError("need a scored chunk after the largest buffer first trains")
    rows = []
    for size in sizes:
        train_times, retrain_times = [], []
        for _ in range(repeat):
            tree = DecisionTreeClassifier(max_depth=max_depth, buffer_size=size, random_state=random_state)
            total_time, triggers = 0.0, []
            correct = scored = 0
            for position, (features, target) in enumerate(chunks):
                if position >= common_start:
                    correct += int(np.count_nonzero(tree.predict(features) == target))
                    scored += len(target)
                previous = tree.n_retrains_
                start = perf_counter()
                tree.partial_fit(features, target)
                elapsed = perf_counter() - start
                total_time += elapsed
                if tree.n_retrains_ > previous:
                    triggers.append(elapsed)
            train_times.append(total_time)
            retrain_times.append(float(np.mean(triggers)))
        rows.append({"buffer_size": int(size), "accuracy": correct/scored, "n_scored": scored,
                     "n_retrains": tree.n_retrains_, "pending_samples": tree.n_buffered_samples_,
                     "train_s": float(np.median(train_times)), "retrain_s": float(np.median(retrain_times))})
    return rows


def format_buffer_table(rows):
    """Format buffer comparison rows as a Markdown performance table.

    Parameters
    ----------
    rows : sequence of dict
        benchmark_buffers records; an empty sequence is accepted.

    Returns
    -------
    str
        Accuracy, common scored count, retrain/pending counts and timings.

    Raises
    ------
    KeyError, TypeError
        If records lack expected fields or contain incompatible values.

    Notes
    -----
    O(b) time/output space for b configurations. Times are milliseconds.
    """
    lines = ["| Buffer size | Accuracy | Scored rows | Retrains | Pending | Total update time | Mean retrain call |",
             "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['buffer_size']} | {row['accuracy']:.4f} | {row['n_scored']} | "
                     f"{row['n_retrains']} | {row['pending_samples']} | {row['train_s']*1000:.3f} ms | "
                     f"{row['retrain_s']*1000:.3f} ms |")
    return "\n".join(lines)
