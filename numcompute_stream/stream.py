"""Coordinate existing incremental models, label metrics and per-chunk logs.

Training and evaluation are deliberately separate. For an unseen stream,
score a chunk before training on it once the first model is available.
"""

from collections.abc import Mapping
from copy import deepcopy
from sys import getsizeof
from time import perf_counter
from types import ModuleType

import numpy as np
from numcompute.utils import _check_label_precision

from .metrics import Accuracy, ROCCurve, accuracy, _check_labels, _check_stream_pair, _check_same_family
from .pipeline import Pipeline

__all__ = ["StreamTrainer"]


def _memory_bytes(value, seen=None):
    """Estimate unique reachable Python/NumPy state; O(objects + entries) work.

    Includes owned ndarray buffers and follows array bases for views. Shared
    references are counted once. Object-array element payloads and external
    native allocations are excluded; this is not process RSS or peak memory.
    Loops walk object containers, not the numerical training computations.
    """
    if seen is None:
        seen = set()
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = getsizeof(value)
    if isinstance(value, np.ndarray):
        if value.base is not None:
            size += _memory_bytes(value.base, seen)
    elif isinstance(value, dict):
        size += sum(_memory_bytes(key, seen) + _memory_bytes(item, seen) for key, item in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(_memory_bytes(item, seen) for item in value)
    elif not isinstance(value, (type, ModuleType)) and not callable(value) and hasattr(value, "__dict__"):
        size += _memory_bytes(vars(value), seen)
    return size


class StreamTrainer:
    """Train and evaluate a classification model/pipeline one chunk at a time.

    Parameters
    ----------
    model : object with partial_fit(X, y) and predict(X)
        An incremental classifier or Pipeline, stored by reference. May already
        be fitted. Trainer counters start at zero; existing model state is kept.
    metrics : mapping of str to metric objects or None, default None
        Optional extra label-based metrics providing update(y_true, y_pred)
        and result(). For example F1(average='macro'), ConfusionMatrix(), or
        Accuracy(window_size=500). Objects are updated in place; use fresh
        instances for a fresh evaluation. Cumulative accuracy is always tracked
        separately. ROC/AUC need scores and must be updated separately.

    Attributes
    ----------
    model, metrics
        Supplied model and a dictionary of supplied metric references.
    logs_ : list of dict
        One record per successful nonempty fit_chunk/score_chunk. Each record
        contains event, fit_chunks, score_chunks, n_samples, trained_samples,
        scored_samples, accuracy, cumulative_accuracy, metrics, seconds,
        memory_bytes, buffered_samples and n_retrains. Fit records have
        accuracy=None and metrics={}; cumulative_accuracy is None until scored.
    n_fit_chunks_, n_score_chunks_ : int
        Successful nonempty training and evaluation calls, separately counted.
    n_trained_samples_, n_scored_samples_ : int
        Rows submitted for training and rows evaluated since construction.

    Raises
    ------
    TypeError, ValueError
        If model methods, metric mapping/names/methods or duplicate metric
        instances are invalid; ROC/AUC are not label-based scoring metrics.

    Notes
    -----
    Reuses model.partial_fit/predict and the existing metrics implementations.
    No training samples or predictions are retained by the trainer. State
    memory is model/metric state plus O(L) log entries, for L operations.
    Metrics such as window confusion matrices can add their own storage.
    Array/dict metric results are copied into logs as historical snapshots.

    memory_bytes estimates reachable model/pipeline and metric object state,
    including NumPy buffers. It excludes input chunks and log history, and is
    neither process RSS nor peak memory. seconds measures the training call or
    prediction/metric computation, excluding initial input checks and memory/log copying.
    Tree buffering/retraining fields are None for models without those attrs.

    fit_chunk does not score its training data. score_chunk does not train or
    update preprocessing. Before a buffered tree first trains, score_chunk
    propagates its not-ready error; the caller decides which chunks to score.
    Component errors propagate without rolling back completed component updates.
    """

    def __init__(self, model, metrics=None):
        """Store the model, optional label metrics and empty log/counter state.

        Parameters
        ----------
        model : incremental classifier or Pipeline
            Must provide callable partial_fit and predict methods.
        metrics : mapping of str to metric objects or None, default None
            Extra metrics described in the class docstring; no metric is reset.

        Returns
        -------
        None
            Create a trainer without training, predicting or clearing its model.

        Raises
        ------
        TypeError, ValueError
            If model/metric methods, mapping/names, duplicate metric instances,
            or use of a score-based ROC/AUC metric are invalid.

        Notes
        -----
        Time/space O(m) for m supplied metrics, plus empty Accuracy state.
        Model/metric objects are stored by reference, not cloned.
        """
        if not callable(getattr(model, "partial_fit", None)) or not callable(getattr(model, "predict", None)):
            raise TypeError("model must provide partial_fit(X, y) and predict(X)")
        if metrics is not None and not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping of names to label-based metric objects")
        self.model = model
        self.metrics = {} if metrics is None else dict(metrics)
        seen = set()
        for name, metric in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("metric names must be nonempty strings")
            if not callable(getattr(metric, "update", None)) or not callable(getattr(metric, "result", None)):
                raise TypeError(f"metric '{name}' must provide update and result")
            if isinstance(metric, ROCCurve):
                raise ValueError("ROC/AUC need positive-class scores; update those metrics separately")
            if id(metric) in seen:
                raise ValueError("each metric name must refer to a separate metric instance")
            seen.add(id(metric))
        self._accuracy = Accuracy()
        self.logs_ = []
        self.n_fit_chunks_ = self.n_score_chunks_ = 0
        self.n_trained_samples_ = self.n_scored_samples_ = 0

    def _check_chunk(self, X, y):
        """Validate aligned 2-D features/1-D labels using existing checks; O(b*d)."""
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be a 2-D array (samples, features) with at least one feature")
        y = _check_labels(y, "y", allow_empty=True, strict=True)
        if len(y) != len(X):
            raise ValueError("y must contain one target per sample")
        first = self.model.steps[0][1] if isinstance(self.model, Pipeline) else self.model
        expected = getattr(first, "n_features_in_", None)
        if expected is not None and X.shape[1] != expected:
            raise ValueError(f"expected {expected} features, got {X.shape[1]}")
        return X, y

    def _log(self, event, n_samples, seconds, chunk_accuracy=None, results=None):
        """Append model/metric snapshots; cost follows the traversed state graph."""
        estimator = self.model
        while isinstance(estimator, Pipeline):
            estimator = estimator.steps[-1][1]
        record = {
            "event": event,
            "fit_chunks": self.n_fit_chunks_,
            "score_chunks": self.n_score_chunks_,
            "n_samples": n_samples,
            "trained_samples": self.n_trained_samples_,
            "scored_samples": self.n_scored_samples_,
            "accuracy": chunk_accuracy,
            "cumulative_accuracy": self._accuracy.result() if self.n_scored_samples_ else None,
            "metrics": {} if results is None else deepcopy(results),
            "seconds": seconds,
            "memory_bytes": _memory_bytes((self.model, self._accuracy, self.metrics)),
            "buffered_samples": getattr(estimator, "n_buffered_samples_", None),
            "n_retrains": getattr(estimator, "n_retrains_", None),
        }
        self.logs_.append(record)
        return record

    def fit_chunk(self, X, y):
        """Send one training chunk to the existing incremental model/pipeline.

        Parameters
        ----------
        X : array_like, shape (b, d)
            Features accepted by the model. Numeric NaNs can be handled by
            the tree or preprocessing; feature validation is model-specific.
        y : array_like, shape (b,)
            Finite numeric/boolean or nonblank string class labels. Empty,
            shape-compatible feature/target chunks are a no-op.

        Returns
        -------
        StreamTrainer
            self. A successful nonempty update appends a fit log, even when
            the model is still buffering and has not trained its first tree.

        Raises
        ------
        TypeError, ValueError
            If shapes, labels, feature counts or model-specific data are invalid.
        Exception
            Model errors propagate. Failed/empty calls do not add trainer logs
            or counters; component state changes are not rolled back.

        Notes
        -----
        Training cost follows model.partial_fit, plus O(b*d) validation and
        traversal of retained model/metric state for the memory estimate.
        No scoring metric is updated and no predict method is called.
        """
        X, y = self._check_chunk(X, y)
        if len(X) == 0:
            return self
        start = perf_counter()
        self.model.partial_fit(X, y)
        seconds = perf_counter() - start
        self.n_fit_chunks_ += 1
        self.n_trained_samples_ += len(X)
        self._log("fit", len(X), seconds)
        return self

    def score_chunk(self, X, y):
        """Predict without training; log chunk metrics and cumulative accuracy.

        Parameters
        ----------
        X : array_like, shape (b, d)
            Features accepted by model.predict, with its fitted input schema.
        y : array_like, shape (b,)
            Matching finite/nonblank true class labels. These must be supplied
            separately from model predictions. Blank targets are not imputed.

        Returns
        -------
        dict or None
            A private copy of the score log, including current chunk accuracy,
            sample-weighted cumulative accuracy and extra metric results.
            None for an empty chunk; no state/log is updated in that case.

        Raises
        ------
        TypeError, ValueError
            If features/targets/predictions are invalid or incompatible, or
            an extra metric rejects the labels/configuration.
        RuntimeError
            If the model cannot yet predict, such as a tree before its first
            training. An unfitted Pipeline may instead raise ValueError.
        Exception
            Component errors propagate. Failed calls add no trainer log or
            counters; completed extra-metric updates are not rolled back.

        Notes
        -----
        Time follows prediction/metric costs, O(b*d) validation and the retained
        state traversal. Extra scratch follows component outputs and copied
        log results. Cumulative accuracy weights samples, not chunk scores.
        Repeated successful calls score again; no sample deduplication is done.
        For unseen-stream evaluation, call this before fit_chunk on those rows.
        """
        X, y = self._check_chunk(X, y)
        if len(X) == 0:
            return None
        start = perf_counter()
        prediction = self.model.predict(X)
        y, prediction = _check_stream_pair(y, prediction)
        if self.n_scored_samples_:
            _check_same_family(y, self._accuracy.labels_, ("y", "previous scored labels"))
            _check_label_precision(y, prediction, self._accuracy.labels_)
        chunk_accuracy = accuracy(y, prediction)
        results = {}
        for name, metric in self.metrics.items():
            metric.update(y, prediction)
            results[name] = metric.result()
        self._accuracy.update(y, prediction)
        seconds = perf_counter() - start
        self.n_score_chunks_ += 1
        self.n_scored_samples_ += len(y)
        record = self._log("score", len(y), seconds, chunk_accuracy, results)
        return deepcopy(record)
