"""Small matplotlib helpers for stream metrics and classification predictions.

Functions plot supplied results, without training or recomputing metrics.
They return (figure, axes). Set show=False for scripts/tests, and optionally
provide save_path. No global plotting style or backend is changed here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from .metrics import _check_labels, _check_same_length, _check_same_family

__all__ = ["plot_metric_over_time", "compare_models", "plot_predictions_vs_ground_truth"]


def _metric_values(values, name):
    """Validate a nonempty numeric vector; preserve NaN gaps. O(n) time/space."""
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a nonempty 1-D array")
    if values.dtype.kind not in "biuf":
        raise TypeError(f"{name} must contain real numeric values; use np.nan for missing scores")
    with np.errstate(over="ignore"):
        values = values.astype(float, copy=False)
    if np.isinf(values).any():
        raise ValueError(f"{name} must not contain infinity; NaN gaps are allowed")
    return values


def _axes(title, ylabel, show, xlabel="Chunk"):
    """Create independent axes after validating options; O(1) plot setup."""
    if not isinstance(title, str) or not isinstance(ylabel, str):
        raise TypeError("title and ylabel must be strings")
    if not isinstance(show, (bool, np.bool_)):
        raise TypeError("show must be True or False")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
    ax.grid(alpha=0.25)
    return fig, ax


def _finish(fig, save_path, show):
    """Lay out, optionally save/show; close this figure on failure."""
    try:
        fig.tight_layout()
        if save_path is not None:
            if isinstance(save_path, str) and not save_path:
                raise ValueError("save_path must name an output file")
            path = Path(save_path)
            if path.is_dir():
                raise ValueError("save_path must name a file, not a directory")
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
    except Exception:
        plt.close(fig)
        raise


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Value", *,
                          save_path=None, show=True):
    """Plot one metric against one-based chunk positions.

    Parameters
    ----------
    metric_values : real numeric array_like, shape (n,)
        Nonempty metric sequence. NaNs create gaps for unscored chunks,
        including warm-up; infinity is rejected. All-NaN sequences are allowed.
    title, ylabel : str, defaults 'Metric over time', 'Value'
        Figure title and vertical-axis label. The horizontal label is Chunk.
    save_path : str, pathlib.Path or None, default None
        Optional output file, e.g. 'plots/accuracy.png'. Parent directories are
        created when needed; matplotlib infers the format from the extension.
    show : bool, default True
        Call matplotlib.show(), including inline display in notebooks. False
        leaves the figure available for saving, customising or headless use.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        New figure/axes containing one line. Input and metric state are unchanged.

    Raises
    ------
    TypeError
        If values/options have unsupported types.
    ValueError
        If values are empty/not 1-D/infinite or save_path names a directory,
        is empty or uses an unsupported matplotlib format.
    OSError
        If the output directory or file cannot be created/written.

    Notes
    -----
    Data preparation/plotting takes O(n) time and memory, excluding backend
    rendering and file encoding. Positions run from 1 to n; NaN gaps keep their
    original positions. Re-call with updated log values to draw a newer view.
    The function does not select a backend or close successful figures; close
    returned figures with plt.close(fig) when finished in repeated scripts.
    """
    values = _metric_values(metric_values, "metric_values")
    fig, ax = _axes(title, ylabel, show)
    ax.plot(np.arange(1, len(values) + 1), values, marker="o", linewidth=1.8)
    ax.set_xlim(0.5, len(values) + 0.5)
    _finish(fig, save_path, show)
    return fig, ax


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"), *,
                   title="Model comparison", ylabel="Accuracy", save_path=None, show=True):
    """Plot two aligned metric histories on the same chunk axis.

    Parameters
    ----------
    metric1, metric2 : real numeric array_like, each shape (n,)
        Nonempty, equal-length histories. Entry i must refer to the same input
        chunk in both histories. Use NaN for a model's unscored warm-up chunks;
        no gap filling, truncation or alignment is performed automatically.
    labels : sequence of two nonblank str, default ('Model 1', 'Model 2')
        Legend names, e.g. ('Buffer 500', 'Buffer 1000').
    title, ylabel : str, defaults 'Model comparison', 'Accuracy'
        Plot title and vertical-axis label; the horizontal label is Chunk.
    save_path : str, pathlib.Path or None, default None
        Optional file. Parent directories are created; format follows extension.
    show : bool, default True
        Display with matplotlib.show(); False supports headless use.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        New figure/axes with two labelled lines using different markers/styles.

    Raises
    ------
    TypeError
        If histories/options have unsupported types.
    ValueError
        If histories are empty/not 1-D/infinite/different lengths, names are
        invalid, or the output file/format is invalid.
    OSError
        If the output directory or file cannot be created/written.

    Notes
    -----
    O(n) time/memory for preparing both histories, excluding rendering/encoding.
    NaNs break lines without moving chunk positions. Compare overlapping scored
    chunks when warm-up periods differ. No inputs/model/metric states are changed.
    """
    first = _metric_values(metric1, "metric1")
    second = _metric_values(metric2, "metric2")
    if len(first) != len(second):
        raise ValueError("metric histories must have the same length; use NaN for unscored chunks")
    if isinstance(labels, (str, bytes)):
        raise ValueError("labels must contain two nonblank strings")
    labels = list(labels)
    if len(labels) != 2 or any(not isinstance(name, str) or not name.strip() for name in labels):
        raise ValueError("labels must contain two nonblank strings")
    fig, ax = _axes(title, ylabel, show)
    chunks = np.arange(1, len(first) + 1)
    ax.plot(chunks, first, marker="o", linewidth=1.8, label=labels[0])
    ax.plot(chunks, second, marker="s", linestyle="--", linewidth=1.8, label=labels[1])
    ax.set_xlim(0.5, len(first) + 0.5)
    ax.legend()
    _finish(fig, save_path, show)
    return fig, ax


def plot_predictions_vs_ground_truth(y_true, y_pred, *, title="Predictions vs ground truth",
                                     save_path=None, show=True):
    """Show true and predicted class labels at each sample position.

    Parameters
    ----------
    y_true, y_pred : numeric/boolean or string array_like, shape (n,)
        Nonempty equal-length vectors from compatible label families. Numeric
        labels must be finite, strings nonblank. Predicted-only classes are
        included in the shared category axis.
    title : str, default 'Predictions vs ground truth'
        Plot title. Axes are labelled Sample and Class.
    save_path : str, pathlib.Path or None, default None
        Optional file. Parent directories are created; format follows extension.
    show : bool, default True
        Display with matplotlib.show(); False supports headless use.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.axes.Axes)
        New figure/axes with hollow circles for truth and crosses for predictions.
        Both use the same sorted union of classes as labelled vertical ticks.

    Raises
    ------
    TypeError
        If labels are unsupported/incompatible or options have invalid types.
    ValueError
        If labels are empty/not 1-D/different lengths/missing/nonfinite or the
        output file/format is invalid.
    OSError
        If the output directory or file cannot be created/written.

    Notes
    -----
    Time O(n*log(n)), space O(n+k) for n samples and k classes, excluding backend
    rendering/encoding. Numeric labels are categories, not continuous targets;
    their vertical positions are categorical indices. Coincident markers show
    correct predictions; separated markers show errors. Reuses metric validators
    and never fits a model or changes the supplied arrays.
    """
    y_true = _check_labels(y_true, "y_true", strict=True)
    y_pred = _check_labels(y_pred, "y_pred", strict=True)
    _check_same_length(y_true, y_pred, ("y_true", "y_pred"))
    _check_same_family(y_true, y_pred, ("y_true", "y_pred"))
    classes = np.union1d(y_true, y_pred)
    samples = np.arange(1, len(y_true) + 1)
    fig, ax = _axes(title, "Class", show, xlabel="Sample")
    ax.scatter(samples, np.searchsorted(classes, y_true), marker="o", s=85,
               facecolors="none", edgecolors="tab:blue", linewidths=1.5, label="Ground truth")
    ax.scatter(samples, np.searchsorted(classes, y_pred), marker="x", s=45,
               color="tab:orange", label="Prediction")
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes.astype(str))
    ax.set_xlim(0.5, len(y_true) + 0.5)
    ax.legend()
    _finish(fig, save_path, show)
    return fig, ax
