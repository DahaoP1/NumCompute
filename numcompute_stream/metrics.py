"""Accuracy, precision, recall, F1, MSE, confusion matrix, ROC, and AUC.

Classification metrics take 1-D label arrays (int, float, bool or str labels).
Precision, recall and F1 are all computed from one confusion matrix, so they
share the same counts. Scalar results are returned as Python floats.

Extended from numcompute.metrics: the original eight functions and input
checking helpers are kept below. Stateful classes add update/reset/result
and optional sample windows, reusing those functions and count formulas.
Streaming updates reject missing/nonfinite values; empty pairs do nothing.
"""

from collections.abc import Sequence
from numbers import Integral

import numpy as np

# All that can be imported from this module
__all__ = [
    "accuracy",
    "confusion_matrix",
    "precision",
    "recall",
    "f1",
    "mse",
    "roc_curve",
    "auc",
    "Accuracy", "Precision", "Recall", "F1", "ConfusionMatrix",
    "MSE", "ROCCurve", "AUC",
]

# Possible values for average
_AVERAGES = ("binary", "macro", "micro", "weighted", None)

# Label dtype kinds that can be compared with each other: NumPy compares any two
# numeric kinds by value (True == 1 == 1.0), but numbers never equal text ("0" != 0)
_LABEL_FAMILIES = {
    "b": "number",
    "i": "number",
    "u": "number",
    "f": "number",
    "U": "text",
    "S": "bytes",
}

# Dtype kinds accepted as numeric values (bool counts as 0/1)
_NUMERIC_KINDS = "biuf"


def accuracy(
    y_true: np.ndarray | Sequence[int | str | float],
    y_pred: np.ndarray | Sequence[int | str | float],
) -> float:
    """Fraction of predictions that match the true labels.

    Parameters
    ----------
    y_true : array_like, shape (n,)
        True labels (int, float, bool or str).
    y_pred : array_like, shape (n,)
        Predicted labels, same kind as ``y_true``.

    Returns
    -------
    float
        Value in [0, 1].

    Raises
    ------
    TypeError
        If ``y_true`` and ``y_pred`` hold labels of incompatible or unsupported types.
    ValueError
        If the inputs are empty, not 1-D, of different lengths, or contain NaN.

    Notes
    -----
    Time Complexity: O(n: number of samples).
    Space Complexity: O(n) for the boolean match array.

    Examples
    --------
    >>> accuracy([0, 1, 1, 0], [0, 1, 0, 0])
    0.75
    """
    y_true = _check_labels(y_true, "y_true")
    y_pred = _check_labels(y_pred, "y_pred")
    _check_same_length(y_true, y_pred, ("y_true", "y_pred"))
    _check_same_family(y_true, y_pred, ("y_true", "y_pred"))

    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray | Sequence[int | str | float],
    y_pred: np.ndarray | Sequence[int | str | float],
    labels: np.ndarray | Sequence[int | str | float] | None = None,
) -> np.ndarray:
    """Count of samples for every (true label, predicted label) pair.

    Parameters
    ----------
    y_true : array_like, shape (n,)
        True labels (int, float, bool or str).
    y_pred : array_like, shape (n,)
        Predicted labels, same kind as ``y_true``.
    labels : array_like, shape (k,), optional
        Labels in the order wanted for the rows and columns. May include labels
        that never appear. Default: the sorted labels found in either input.

    Returns
    -------
    numpy.ndarray of int, shape (k, k)
        ``C[i, j]`` is the number of samples whose true label is ``labels[i]``
        and whose predicted label is ``labels[j]``. Rows = true, columns =
        predicted, so the diagonal holds the correct predictions.

    Raises
    ------
    TypeError
        If the inputs (or ``labels``) hold labels of incompatible or unsupported types.
    ValueError
        If the inputs are empty, not 1-D, of different lengths, or contain NaN;
        or ``labels`` is empty, has duplicates or misses a label in the data.

    Notes
    -----
    Time Complexity: O(n log k + k^2), n samples, k labels.
    Space Complexity: O(n + k^2).
    Every (true, predicted) pair is turned into one index ``t * k + p`` and all
    pairs are counted at once with ``np.bincount``, without a Python loop.

    Examples
    --------
    >>> confusion_matrix([0, 1, 1, 0], [0, 1, 0, 0])
    array([[2, 0],
           [1, 1]])
    """
    y_true = _check_labels(y_true, "y_true")
    y_pred = _check_labels(y_pred, "y_pred")
    _check_same_length(y_true, y_pred, ("y_true", "y_pred"))
    _check_same_family(y_true, y_pred, ("y_true", "y_pred"))

    classes = _resolve_labels(y_true, y_pred, labels)

    return _confusion(y_true, y_pred, classes)


def precision(
    y_true: np.ndarray | Sequence[int | str | float],
    y_pred: np.ndarray | Sequence[int | str | float],
    average: str | None = "binary",
    pos_label: int | float | str = 1,
    zero_division: float = 0.0,
) -> float | np.ndarray:
    """Precision: of the samples predicted as a class, the fraction that truly are.

    ``precision = tp / (tp + fp)`` per class.

    Parameters
    ----------
    y_true : array_like, shape (n,)
        True labels (int, float, bool or str).
    y_pred : array_like, shape (n,)
        Predicted labels, same kind as ``y_true``.
    average : {"binary", "macro", "micro", "weighted", None}, default "binary"
        ``"binary"``: only the class ``pos_label`` (at most 2 classes).
        ``"macro"``: unweighted mean over classes.
        ``"micro"``: from the counts summed over all classes.
        ``"weighted"``: mean over classes weighted by their support (true count).
        ``None``: one value per class, in sorted label order.
    pos_label : integer, float, bool or string, default 1
        The positive class when ``average="binary"``.
    zero_division : float, default 0.0
        Value returned where the denominator is 0 (no sample predicted as that
        class), instead of NaN.

    Returns
    -------
    float, or numpy.ndarray of shape (k,) when ``average=None``
        Values in [0, 1] (or ``zero_division``).

    Raises
    ------
    TypeError
        If the inputs or ``pos_label`` hold labels of incompatible or unsupported types,
        ``pos_label`` is not a single label, or ``zero_division`` is not a number.
    ValueError
        If the inputs are empty, not 1-D, of different lengths, or contain NaN;
        ``zero_division`` is NaN; ``average`` is unknown; or ``average="binary"``
        and there are more than 2 classes or ``pos_label`` is not one of the 2 labels.

    Notes
    -----
    Time Complexity: O(n log k + k^2), n samples, k labels.
    Space Complexity: O(n + k^2), dominated by the confusion matrix.

    Examples
    --------
    >>> precision([0, 1, 1, 0], [0, 1, 0, 0])
    1.0
    >>> precision([0, 1, 1, 0], [0, 1, 0, 0], average=None)
    array([0.66666667, 1.        ])
    """
    return _precision_recall_f1(y_true, y_pred, average, pos_label, zero_division)[0]


def recall(
    y_true: np.ndarray | Sequence[int | str | float],
    y_pred: np.ndarray | Sequence[int | str | float],
    average: str | None = "binary",
    pos_label: int | float | str = 1,
    zero_division: float = 0.0,
) -> float | np.ndarray:
    """Recall: of the samples that truly are a class, the fraction predicted as it.

    ``recall = tp / (tp + fn)`` per class.

    Parameters
    ----------
    y_true : array_like, shape (n,)
        True labels (int, float, bool or str).
    y_pred : array_like, shape (n,)
        Predicted labels, same kind as ``y_true``.
    average : {"binary", "macro", "micro", "weighted", None}, default "binary"
        How per-class values are combined, see :func:`precision`.
    pos_label : integer, float, bool or string, default 1
        The positive class when ``average="binary"``.
    zero_division : float, default 0.0
        Value returned where the denominator is 0 (no sample truly of that
        class), instead of NaN.

    Returns
    -------
    float, or numpy.ndarray of shape (k,) when ``average=None``
        Values in [0, 1] (or ``zero_division``).

    Raises
    ------
    TypeError
        If the inputs or ``pos_label`` hold labels of incompatible or unsupported types,
        ``pos_label`` is not a single label, or ``zero_division`` is not a number.
    ValueError
        If the inputs are empty, not 1-D, of different lengths, or contain NaN;
        ``zero_division`` is NaN; ``average`` is unknown; or ``average="binary"``
        and there are more than 2 classes or ``pos_label`` is not one of the 2 labels.

    Notes
    -----
    Time Complexity: O(n log k + k^2), n samples, k labels.
    Space Complexity: O(n + k^2), dominated by the confusion matrix.
    ``recall(..., average="weighted")`` always equals :func:`accuracy`.

    Examples
    --------
    >>> recall([0, 1, 1, 0], [0, 1, 0, 0])
    0.5
    """
    return _precision_recall_f1(y_true, y_pred, average, pos_label, zero_division)[1]


def f1(
    y_true: np.ndarray | Sequence[int | str | float],
    y_pred: np.ndarray | Sequence[int | str | float],
    average: str | None = "binary",
    pos_label: int | float | str = 1,
    zero_division: float = 0.0,
) -> float | np.ndarray:
    """F1 score: harmonic mean of precision and recall.

    Computed per class as ``2 * tp / (2 * tp + fp + fn)``, which equals
    ``2 * p * r / (p + r)`` and stays defined when only one of them is.

    Parameters
    ----------
    y_true : array_like, shape (n,)
        True labels (int, float, bool or str).
    y_pred : array_like, shape (n,)
        Predicted labels, same kind as ``y_true``.
    average : {"binary", "macro", "micro", "weighted", None}, default "binary"
        How per-class values are combined, see :func:`precision`.
    pos_label : integer, float, bool or string, default 1
        The positive class when ``average="binary"``.
    zero_division : float, default 0.0
        Value returned where the denominator is 0 (the class never appears in
        ``y_true`` nor ``y_pred``), instead of NaN.

    Returns
    -------
    float, or numpy.ndarray of shape (k,) when ``average=None``
        Values in [0, 1] (or ``zero_division``).

    Raises
    ------
    TypeError
        If the inputs or ``pos_label`` hold labels of incompatible or unsupported types,
        ``pos_label`` is not a single label, or ``zero_division`` is not a number.
    ValueError
        If the inputs are empty, not 1-D, of different lengths, or contain NaN;
        ``zero_division`` is NaN; ``average`` is unknown; or ``average="binary"``
        and there are more than 2 classes or ``pos_label`` is not one of the 2 labels.

    Notes
    -----
    Time Complexity: O(n log k + k^2), n samples, k labels.
    Space Complexity: O(n + k^2), dominated by the confusion matrix.
    ``f1(..., average="micro")`` always equals :func:`accuracy`.

    Examples
    --------
    >>> f1([0, 1, 1, 0], [0, 1, 0, 0])
    0.6666666666666666
    """
    return _precision_recall_f1(y_true, y_pred, average, pos_label, zero_division)[2]


def mse(
    y_true: np.ndarray | Sequence[int | float],
    y_pred: np.ndarray | Sequence[int | float],
) -> float:
    """Mean squared error between true and predicted values.

    Parameters
    ----------
    y_true : array_like of numbers, shape (n,)
        True values.
    y_pred : array_like of numbers, shape (n,)
        Predicted values.

    Returns
    -------
    float
        ``mean((y_true - y_pred) ** 2)`` over all samples, >= 0.

    Raises
    ------
    TypeError
        If an input is not numeric.
    ValueError
        If an input is not 1-D, is empty or contains NaN, or the lengths differ.

    Notes
    -----
    Time Complexity: O(n: number of samples).
    Space Complexity: O(n) for the difference array.
    Divide squared errors before their sum to avoid unnecessary overflow.

    Examples
    --------
    >>> mse([1, 2, 3], [1, 2, 5])
    1.3333333333333333
    """
    y_true = _check_values(y_true, "y_true")
    y_pred = _check_values(y_pred, "y_pred")
    _check_same_length(y_true, y_pred, ("y_true", "y_pred"))

    squared = np.square(y_true - y_pred)
    # Divide before summing so representable mean errors do not overflow.
    return float(np.sum(squared / squared.size))


def roc_curve(
    y_true: np.ndarray | Sequence[int | str | float],
    y_score: np.ndarray | Sequence[float],
    pos_label: int | float | str = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Receiver operating characteristic curve for a binary classifier.

    Each threshold ``t`` predicts positive for every sample with score >= ``t``;
    the curve gives the false and true positive rates for every distinct score.

    Parameters
    ----------
    y_true : array_like, shape (n,)
        True labels, at most 2 distinct values (int, float, bool or str).
    y_score : array_like of numbers, shape (n,)
        Score for the positive class (higher = more likely positive), e.g. a
        probability.
    pos_label : integer, float, bool or string, default 1
        The label of the positive class in ``y_true``.

    Returns
    -------
    fpr : numpy.ndarray, shape (m + 1,)
        False positive rate ``fp / n_negatives`` at each threshold, from 0 up to 1.
    tpr : numpy.ndarray, shape (m + 1,)
        True positive rate ``tp / n_positives`` at each threshold, from 0 up to 1.
    thresholds : numpy.ndarray, shape (m + 1,)
        The ``m`` distinct scores in decreasing order, after a first ``inf``
        (nothing predicted positive).

    Raises
    ------
    TypeError
        If ``y_true`` has an unsupported label dtype, ``y_score`` is not numeric,
        ``pos_label`` is not the same type of label as ``y_true``, or ``pos_label``
        is not a single label.
    ValueError
        If the inputs are empty, not 1-D or of different lengths; ``y_true`` or
        ``y_score`` contains NaN; ``y_true`` has more than 2 classes; or ``y_true``
        has no positive or no negative sample (a rate would divide by zero).

    Notes
    -----
    Time Complexity: O(n log n), for sorting the scores.
    Space Complexity: O(n).
    Samples with the same score cross a threshold together, so tied scores give
    one point on the curve.

    Examples
    --------
    >>> fpr, tpr, thresholds = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    >>> fpr
    array([0. , 0. , 0.5, 0.5, 1. ])
    >>> tpr
    array([0. , 0.5, 0.5, 1. , 1. ])
    """
    y_true = _check_labels(y_true, "y_true")
    y_score = _check_values(y_score, "y_score")
    _check_same_length(y_true, y_score, ("y_true", "y_score"))

    pos = _check_pos_label(pos_label, y_true)

    classes = np.unique(y_true)

    if classes.size > 2:
        raise ValueError(
            f"roc_curve is for binary labels, got {classes.size} classes: {classes.tolist()}"
        )

    positive = y_true == pos[0]
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f"y_true needs positive and negative samples (pos_label={pos_label!r}), "
            f"got {n_pos} positive and {n_neg} negative"
        )

    # Highest score first; stable so equal scores keep their order
    order = np.argsort(-y_score, kind="stable")
    scores = y_score[order]

    tps = np.cumsum(positive[order])  # positives with score >= scores[i]
    fps = np.arange(1, scores.size + 1) - tps  # negatives with score >= scores[i]

    # Last position of each run of equal scores: tied samples cross the threshold together
    last = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])

    fpr = np.r_[0.0, fps[last] / n_neg]
    tpr = np.r_[0.0, tps[last] / n_pos]
    thresholds = np.r_[np.inf, scores[last]]

    return fpr, tpr, thresholds


def auc(
    x: np.ndarray | Sequence[int | float],
    y: np.ndarray | Sequence[int | float],
) -> float:
    """Area under a curve given by points (x, y), with the trapezoid rule.

    Parameters
    ----------
    x : array_like of numbers, shape (m,)
        x coordinates, all increasing or all decreasing (repeats allowed),
        e.g. ``fpr`` from :func:`roc_curve`.
    y : array_like of numbers, shape (m,)
        y coordinates, e.g. ``tpr`` from :func:`roc_curve`.

    Returns
    -------
    float
        The area. For decreasing ``x`` the sign is flipped, so the same curve
        gives the same area in either direction.

    Raises
    ------
    TypeError
        If an input is not numeric.
    ValueError
        If an input is not 1-D, contains NaN or has fewer than 2 points; the
        lengths differ; or ``x`` is not monotonic.

    Notes
    -----
    Time Complexity: O(m: number of points).
    Space Complexity: O(m).
    Written out instead of ``np.trapz``, which NumPy 2 renamed to ``np.trapezoid``.

    Examples
    --------
    >>> auc([0, 1], [0, 1])
    0.5
    """
    x = _check_values(x, "x")
    y = _check_values(y, "y")
    _check_same_length(x, y, ("x", "y"))

    if x.size < 2:
        raise ValueError(f"auc needs at least 2 points, got {x.size}")

    dx = np.diff(x)

    if (dx < 0).any() and (dx > 0).any():
        raise ValueError("x must be monotonic (all increasing or all decreasing)")

    direction = -1.0 if (dx < 0).any() else 1.0
    area = np.sum(dx * (y[1:] + y[:-1]) / 2)

    return float(direction * area)


def _check_labels(
    y: np.ndarray | Sequence[int | str | float],
    name: str,
    *, allow_empty: bool = False, strict: bool = False,
) -> np.ndarray:
    """Validate a nonempty vector of numeric or string labels.

    Parameters
    ----------
    y : array_like, shape (n,)
        Labels of a supported numeric, boolean or text dtype.
    name : str
        Argument name included in error messages.
    allow_empty : bool, default False
        Permit an empty vector for a streaming no-op update.
    strict : bool, default False
        Reject infinity and, for labels, blank text during streaming.

    Returns
    -------
    ndarray, shape (n,)
        Validated labels with their original dtype.

    Raises
    ------
    TypeError
        If label dtype is unsupported.
    ValueError
        If input is empty, not 1-D, ragged or contains floating-point NaN.

    Notes
    -----
    Time/space: O(n) worst case for conversion and NaN validation.
    """
    # Converting to a NumPy array if a list was received, keeping its dtype
    y = np.asarray(y)

    # Check if the array is 1-D: one label per sample
    if y.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D, got shape {y.shape}; use .ravel() for a column vector"
        )

    # Check if the dtype is a supported label kind: numbers, text or bytes (no objects, no dates)
    if y.dtype.kind not in _LABEL_FAMILIES:
        raise TypeError(
            f"{name} has unsupported label dtype {y.dtype}; use numbers or strings"
        )

    # Check if the array is not empty
    if y.size == 0 and not allow_empty:
        raise ValueError(f"{name} is empty")

    # Check for NaN, only on float labels: no other kind can hold it, and NaN never equals itself
    if y.dtype.kind == "f" and np.isnan(y).any():
        raise ValueError(f"{name} contains NaN")

    # Streaming adds finite/nonblank checks; batch defaults stay unchanged.
    if strict and y.dtype.kind in "biuf" and not np.isfinite(y).all():
        raise ValueError(f"{name} must not contain infinity")
    if strict and y.dtype.kind in "US" and np.any(np.char.str_len(np.char.strip(y)) == 0):
        raise ValueError(f"{name} must not contain blank labels")

    return y


def _check_values(
    y: np.ndarray | Sequence[int | float],
    name: str,
    *, allow_empty: bool = False, strict: bool = False,
) -> np.ndarray:
    """Validate a nonempty numeric vector and convert it to float.

    Parameters
    ----------
    y : numeric array_like, shape (n,)
        Input values; numeric-looking strings are not accepted.
    name : str
        Argument name used in validation messages.
    allow_empty : bool, default False
        Permit an empty vector for a streaming no-op update.
    strict : bool, default False
        Reject infinity and, for labels, blank text during streaming.

    Returns
    -------
    ndarray of float, shape (n,)
        Values without NaN; infinity is not rejected by this helper.

    Raises
    ------
    TypeError
        If dtype is not a supported numeric type.
    ValueError
        If input is empty, not 1-D, ragged or contains NaN.

    Notes
    -----
    Time and worst-case extra space: O(n); float64 input may be reused.
    """
    # Converting to a NumPy array if a list was received, keeping its dtype
    y = np.asarray(y)

    # Check if the array is 1-D: one value per sample
    if y.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D, got shape {y.shape}; use .ravel() for a column vector"
        )

    # Check if dtype is numeric before converting: astype(float) would silently turn "1" into 1.0
    if y.dtype.kind not in _NUMERIC_KINDS:
        raise TypeError(f"{name} must be numeric, got dtype {y.dtype}")

    # Check if the array is not empty
    if y.size == 0 and not allow_empty:
        raise ValueError(f"{name} is empty")

    # Converting to float for the arithmetic; no copy when the input is already float64
    y = y.astype(float, copy=False)

    # Check for NaN, after converting so it always runs on float data
    if np.isnan(y).any():
        raise ValueError(f"{name} contains NaN")

    if strict and not np.isfinite(y).all():
        raise ValueError(f"{name} must not contain infinity")

    return y


def _check_same_length(
    a: np.ndarray,
    b: np.ndarray,
    names: tuple[str, str],
) -> None:
    """Check that two validated vectors have equal lengths.

    Parameters
    ----------
    a, b : ndarray, shapes (n,), (m,)
        Vectors whose sizes are compared.
    names : pair of str
        Argument names included in the error message.

    Returns
    -------
    None
        Only validation is performed.

    Raises
    ------
    ValueError
        If the sizes differ.

    Notes
    -----
    Time and space: O(1).
    """
    # Check if both arrays have one entry per sample: no broadcasting
    if a.size != b.size:
        raise ValueError(
            f"{names[0]} and {names[1]} must have the same length, got {a.size} and {b.size}"
        )


def _check_same_family(
    a: np.ndarray,
    b: np.ndarray,
    names: tuple[str, str],
) -> None:
    """Check that two validated label vectors use compatible label families.

    Parameters
    ----------
    a, b : ndarray, shapes (n,), (m,)
        Label vectors with supported dtypes.
    names : pair of str
        Names to include in the error message.

    Returns
    -------
    None
        Only validation is performed.

    Raises
    ------
    TypeError
        If one vector holds numeric labels and the other holds text labels.

    Notes
    -----
    Time and space: O(1).
    """
    # Check if both arrays hold the same family of labels: NumPy would compare 1 with "1" as unequal
    # (or silently promote 1 to "1"), giving wrong scores instead of an error
    if _LABEL_FAMILIES[a.dtype.kind] != _LABEL_FAMILIES[b.dtype.kind]:
        raise TypeError(
            f"{names[0]} ({a.dtype}) and {names[1]} ({b.dtype}) hold labels of incompatible types"
        )


def _check_pos_label(
    pos_label: int | float | str,
    y_true: np.ndarray,
    *, strict: bool = False,
) -> np.ndarray:
    """Validate a scalar positive label against the true-label family.

    Parameters
    ----------
    pos_label : scalar
        Positive class identifier.
    y_true : ndarray, shape (n,)
        Previously validated true labels.

    strict : bool, default False
        Require a finite/nonblank positive label for streaming.

    Returns
    -------
    ndarray, shape (1,)
        Positive label with a compatible label family.

    Raises
    ------
    TypeError
        If pos_label is nonscalar, unsupported or incompatible with y_true.
    ValueError
        If pos_label contains NaN.

    Notes
    -----
    Time and space: O(1).
    """
    # Check if pos_label is a single label
    if np.ndim(pos_label) != 0:
        raise TypeError(f"pos_label must be a single label, got {pos_label!r}")

    pos = _check_labels([pos_label], "pos_label", strict=strict)
    _check_same_family(y_true, pos, ("y_true", "pos_label"))

    return pos


def _resolve_labels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray | Sequence[int | str | float] | None = None,
) -> np.ndarray:
    """Resolve the confusion-matrix class order and validate optional labels.

    Parameters
    ----------
    y_true, y_pred : ndarray, shape (n,)
        Previously validated, compatible label vectors.
    labels : array_like, shape (k,), or None
        Desired class order. None selects the sorted union of observed labels.

    Returns
    -------
    ndarray, shape (k,)
        Unique class labels covering every observed label.

    Raises
    ------
    TypeError
        If supplied labels have unsupported or incompatible dtype.
    ValueError
        If labels are empty, not 1-D, duplicated, contain NaN or omit a class.

    Notes
    -----
    Time: O(n * log(n) + k * log(k) + n * log(k)); space O(n + k).
    """
    found = np.unique(np.concatenate([y_true, y_pred]))

    if labels is None:
        return found

    labels = _check_labels(labels, "labels")
    _check_same_family(y_true, labels, ("y_true", "labels"))

    if np.unique(labels).size != labels.size:
        raise ValueError(f"labels has duplicates: {labels.tolist()}")

    unknown = found[_label_index(found, labels) < 0]

    if unknown.size > 0:
        raise ValueError(
            f"the data has labels that are not in labels: {unknown.tolist()}"
        )

    return labels


def _label_index(
    y: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Map labels to positions in a possibly unsorted class vector.

    Parameters
    ----------
    y : ndarray, shape (n,)
        Labels to look up.
    classes : ndarray, shape (k,)
        Nonempty unique classes with a compatible dtype.

    Returns
    -------
    ndarray of integers, shape (n,)
        Index in the original classes array, or -1 for an absent label.

    Raises
    ------
    TypeError
        If labels cannot be ordered or compared.
    IndexError
        If a nonempty y is searched in an empty classes array.

    Notes
    -----
    Time: O(k * log(k) + n * log(k)); space O(n + k).
    """
    order = np.argsort(classes, kind="stable")
    sorted_classes = classes[order]

    pos = np.searchsorted(sorted_classes, y)  # where each label is, or would be inserted
    pos = np.minimum(pos, classes.size - 1)  # keep the index valid

    found = sorted_classes[pos] == y

    return np.where(found, order[pos], -1)


def _confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Count checked true/predicted label pairs in the given class order.

    Parameters
    ----------
    y_true, y_pred : ndarray, shape (n,)
        Compatible labels, all present in classes.
    classes : ndarray, shape (k,)
        Nonempty unique classes in the desired output order.

    Returns
    -------
    ndarray of integers, shape (k, k)
        Rows are true classes and columns are predicted classes.

    Raises
    ------
    ValueError
        NumPy may reject invalid indices if unchecked labels violate the
        class-membership precondition.

    Notes
    -----
    Time: O(k * log(k) + n * log(k) + k**2); space O(n + k**2).
    """
    k = classes.size
    t = _label_index(y_true, classes)
    p = _label_index(y_pred, classes)

    # One index per (true, predicted) pair, all pairs counted at once
    counts = np.bincount(t * k + p, minlength=k * k)

    return counts.reshape(k, k)


def _safe_divide(
    num: np.ndarray | Sequence[int | float],
    den: np.ndarray | Sequence[int | float],
    zero_division: float,
) -> np.ndarray:
    """Divide with a configured replacement for zero denominators.

    Parameters
    ----------
    num, den : numeric array_like
        Numerator and denominator with broadcast-compatible shapes.
    zero_division : float
        Value to return wherever den equals zero.

    Returns
    -------
    ndarray of float, broadcast shape of num and den
        Quotients with zero denominators replaced.

    Raises
    ------
    TypeError
        If arguments cannot be converted to floats.
    ValueError
        If conversion fails or shapes cannot be broadcast.

    Notes
    -----
    Time and space: O(B), B is the number of broadcast output elements.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast_shapes(num.shape, den.shape), zero_division, dtype=float)

    np.divide(num, den, out=out, where=den != 0)

    return out


def _precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str | None,
    pos_label: int | float | str,
    zero_division: float,
) -> tuple:
    """Compute shared per-class or averaged precision, recall and F1.

    Parameters
    ----------
    y_true, y_pred : array_like, shape (n,)
        True and predicted labels from compatible families.
    average : {'binary', 'macro', 'micro', 'weighted'} or None
        Reduction rule; None returns per-class scores.
    pos_label : scalar
        Positive class for binary averaging.
    zero_division : number
        Replacement for undefined ratios; must not be NaN or boolean.

    Returns
    -------
    tuple of three floats or three ndarrays of shape (k,)
        Precision, recall and F1; per-class arrays use sorted class order.

    Raises
    ------
    TypeError
        If label families, pos_label or zero_division have invalid types.
    ValueError
        If labels, lengths, averaging mode or binary-class configuration
        are invalid, or zero_division is NaN.

    Notes
    -----
    Time: O(n * log(n) + n * log(k) + k**2); space O(n + k**2).
    """
    if average not in _AVERAGES:
        raise ValueError(f"average must be one of {_AVERAGES}, got {average!r}")

    if isinstance(zero_division, bool) or not isinstance(
        zero_division, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"zero_division must be a number, got {zero_division!r}")

    zero_division = float(zero_division)

    if np.isnan(zero_division):
        raise ValueError("zero_division must not be NaN")

    y_true = _check_labels(y_true, "y_true")
    y_pred = _check_labels(y_pred, "y_pred")
    _check_same_length(y_true, y_pred, ("y_true", "y_pred"))
    _check_same_family(y_true, y_pred, ("y_true", "y_pred"))

    classes = _resolve_labels(y_true, y_pred, None)

    if average == "binary":
        if classes.size > 2:
            raise ValueError(
                f"average='binary' needs at most 2 classes, got {classes.size}: "
                f"{classes.tolist()}; use 'macro', 'micro' or 'weighted'"
            )

        found = classes

        pos = _check_pos_label(pos_label, y_true)

        # pos_label may be absent from the data (e.g. every label is 0): add it so its counts are 0
        classes = np.union1d(found, pos)

        if classes.size > 2:
            raise ValueError(
                f"pos_label {pos_label!r} is not one of the labels {found.tolist()}"
            )

    cm = _confusion(y_true, y_pred, classes)

    return _scores_from_confusion(cm, classes, average, pos_label, zero_division)



def _scores_from_confusion(cm, classes, average, pos_label, zero_division):
    """Reuse the original precision/recall/F1 formulas on batch or stream counts.

    Parameters
    ----------
    cm : ndarray of int, shape (k, k)
        Already validated confusion counts, rows=true and columns=predicted.
    classes : ndarray, shape (k,)
        Label order corresponding to the matrix.
    average, pos_label, zero_division
        Validated options described in precision/recall/f1.

    Returns
    -------
    tuple of three floats or ndarrays, shape (k,)
        Precision, recall and F1, reduced according to average.

    Raises
    ------
    TypeError
        If the binary positive label has an incompatible family.

    Notes
    -----
    Time O(k*k), scratch space O(k). The block below is extracted from the
    original _precision_recall_f1, including its safe divisions and averages.
    """
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp  # predicted as the class, truly another
    fn = cm.sum(axis=1) - tp  # truly the class, predicted as another
    support = cm.sum(axis=1)

    if average == "binary":
        pos = _check_pos_label(pos_label, classes)
        idx = _label_index(pos, classes)[0]
        # A positive class may not have arrived yet during stream warm-up.
        tp, fp, fn = (0, 0, 0) if idx < 0 else (tp[idx], fp[idx], fn[idx])

    elif average == "micro":
        tp, fp, fn = tp.sum(), fp.sum(), fn.sum()

    scores = (
        _safe_divide(tp, tp + fp, zero_division),
        _safe_divide(tp, tp + fn, zero_division),
        _safe_divide(2 * tp, 2 * tp + fp + fn, zero_division),
    )

    if average is None:
        return scores

    if average == "macro":
        return tuple(float(s.mean()) for s in scores)

    if average == "weighted":
        return tuple(float(np.average(s, weights=support)) for s in scores)

    return tuple(float(s) for s in scores)  # binary / micro: already one value


def _window_size(value):
    """Validate a positive sample-window size or None; O(1) time/space."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("window_size must be an integer or None")
    if value < 1:
        raise ValueError("window_size must be at least 1")
    return int(value)


def _check_stream_pair(y_true, other, numeric=False, scores=False):
    """Apply original validators to possibly empty stream pairs; O(b) space/time."""
    check_true = _check_values if numeric else _check_labels
    check_other = _check_values if numeric or scores else _check_labels
    names = ("y_true", "y_score" if scores else "y_pred")
    y_true = check_true(y_true, names[0], allow_empty=True, strict=True)
    other = check_other(other, names[1], allow_empty=True, strict=True)
    _check_same_length(y_true, other, names)
    if y_true.size and not numeric and not scores:
        _check_same_family(y_true, other, names)
    return y_true, other


def _check_binary_classes(classes, pos_label):
    """Apply the original binary-class rules to the active vocabulary."""
    pos = _check_pos_label(pos_label, classes, strict=True)
    if np.union1d(classes, pos).size > 2:
        raise ValueError("binary metrics require at most two classes including pos_label")


class ConfusionMatrix:
    """Extend the original confusion_matrix function with cumulative/window state.

    update uses the original _check_labels, _resolve_labels and _confusion
    helpers. labels_ gives the current true-row/predicted-column order.
    See __init__/update/reset/result for options, shapes and complexity.
    """

    def __init__(self, labels=None, window_size=None):
        """Configure the active classes and optional sample window.

        Parameters
        ----------
        labels : array_like, shape (k,), or None, default None
            Fixed class order covering all incoming labels. None discovers
            sorted classes from active observations (including predictions).
        window_size : positive int or None, default None
            Keep the latest W samples, or accumulate all samples with None.

        Returns
        -------
        None
            Initialise an empty metric; result() requires observations.

        Raises
        ------
        TypeError, ValueError
            If window_size or labels have invalid types/values or labels repeat.

        Notes
        -----
        Constructor time O(k log k); stored state O(k*k), plus O(W) window
        labels when enabled. Labels must be finite numbers or nonblank strings.
        """
        self.window_size = _window_size(window_size)
        self._labels = None if labels is None else _check_labels(labels, "labels", strict=True).copy()
        if self._labels is not None:
            if len(self._labels) == 0 or len(np.unique(self._labels)) != len(self._labels):
                raise ValueError("labels must be nonempty and contain no duplicates")
        self.reset()

    def reset(self):
        """Clear observations while keeping configuration.

        Returns
        -------
        self
            Empty metric with zero counts; result() now raises RuntimeError.

        Notes
        -----
        No parameters. Time/space O(k*k), where k is the fixed label count
        (zero when labels=None). Releases any stored sample window.
        """
        self.labels_ = np.array([]) if self._labels is None else self._labels.copy()
        self._matrix = np.zeros((len(self.labels_), len(self.labels_)), dtype=np.int64)
        self._true = self._pred = None
        self.n_samples_seen_ = self.n_samples_ = 0
        return self

    def _validate_classes(self, classes):
        """Subclass hook for binary constraints; O(1) unless overridden."""

    def update(self, y_true, y_pred):
        """Accumulate new label pairs, optionally expiring old samples.

        Parameters
        ----------
        y_true, y_pred : array_like, shape (b,)
            Compatible numeric or string labels. Missing/nonfinite labels
            are rejected; empty pairs are a no-op. Features/columns are not
            accepted: flatten a single target column explicitly if needed.

        Returns
        -------
        self
            Updated metric. labels_ gives result array/matrix class order;
            n_samples_ is the active count, n_samples_seen_ the total received.

        Raises
        ------
        TypeError
            If labels have unsupported/incompatible types, including a change
            of label family between updates without reset().
        ValueError
            If shape/length/labels are invalid, fixed labels omit a class,
            or a binary metric would contain more than two classes.

        Notes
        -----
        Cumulative mode stores O(k*k) counts, not old rows. Updates take a
        conservative O((b+k)*log(b+k+1)+k*k) time and O(b+k*k) scratch space.
        Window mode keeps copied pairs for the latest W samples and recomputes
        counts: O((b+W+k)*log(b+W+k+1)+k*k) time, O(b+W+k*k) scratch, O(W+k*k)
        state. It never averages chunk scores. Failed validation leaves state.
        """
        y_true, y_pred = _check_stream_pair(y_true, y_pred)
        if len(y_true) == 0:
            return self
        if len(self.labels_):
            _check_same_family(y_true, self.labels_, ("y_true", "previous labels"))
        chunk_classes = _resolve_labels(y_true, y_pred, self._labels)

        if self.window_size is not None:
            active_true = y_true if self._true is None else np.concatenate((self._true, y_true))
            active_pred = y_pred if self._pred is None else np.concatenate((self._pred, y_pred))
            active_true = active_true[-self.window_size:].copy()
            active_pred = active_pred[-self.window_size:].copy()
            classes = _resolve_labels(active_true, active_pred, self._labels)
        else:
            active_true, active_pred = y_true, y_pred
            classes = self._labels
            if classes is None:
                classes = chunk_classes
                if len(self.labels_):
                    classes = np.union1d(self.labels_, classes)
        self._validate_classes(classes)
        matrix = _confusion(active_true, active_pred, classes)
        if self.window_size is None and self.n_samples_:
            # Dynamically discovered classes are sorted; align existing counts
            # when an alphabetically/numerically earlier class arrives.
            indices = _label_index(self.labels_, classes)
            matrix[np.ix_(indices, indices)] += self._matrix
        self.labels_, self._matrix = classes.copy(), matrix
        self.n_samples_seen_ += len(y_true)
        self.n_samples_ = int(matrix.sum())
        if self.window_size is not None:
            self._true, self._pred = active_true, active_pred
        return self

    def _require_samples(self):
        """Reject undefined empty results; O(1) time/space."""
        if self.n_samples_ == 0:
            raise RuntimeError("no observations; call update before result")

    def result(self):
        """Return a private copy of the active confusion counts.

        Returns
        -------
        ndarray of int, shape (k, k)
            Rows are true labels, columns predicted labels, ordered by labels_.

        Raises
        ------
        RuntimeError
            If no nonempty update has completed since construction/reset.

        Notes
        -----
        No parameters. Time/space: O(k*k). Reading results does not reset state.
        """
        self._require_samples()
        return self._matrix.copy()


class Accuracy(ConfusionMatrix):
    """Track the fraction of correct predictions across samples.

    Parameters are labels=None and window_size=None, documented on the shared
    constructor. Uses O(k*k) counts, plus O(W) in window mode. Each sample has
    equal weight even when chunk sizes differ. See update/reset/result for
    shapes, errors and complexity.
    """

    def result(self):
        """Return active accuracy without changing state.

        Returns
        -------
        float
            Correct predictions divided by active samples, in [0, 1].

        Raises
        ------
        RuntimeError
            If no observations are available after construction/reset.

        Notes
        -----
        No parameters. Time/extra space: O(k)/O(1), for k classes.
        """
        self._require_samples()
        return float(np.trace(self._matrix) / self.n_samples_)


class Precision(ConfusionMatrix):
    """Add streamed confusion counts to the original precision calculation.

    Uses the same TP/FP/FN, safe division and averaging block as the original
    precision/recall/f1 functions. See __init__ for the unchanged averaging
    options plus labels and window_size. update/reset are inherited from
    ConfusionMatrix; result computes precision from its active counts.
    """

    _score_index = 0

    def __init__(self, average="binary", pos_label=1, zero_division=0.0,
                 labels=None, window_size=None):
        """Configure precision, recall or F1 averaging.

        Parameters
        ----------
        average : {'binary', 'macro', 'micro', 'weighted'} or None, default 'binary'
            Positive class only; equal class mean; global counts; true-support
            weighted class mean; or one score per class with None.
        pos_label : finite number or nonblank str, default 1
            Positive class for binary averaging. May be absent during warm-up.
        zero_division : finite real number, default 0.0
            Result for undefined denominators. Common choices are zero or one.
        labels : array_like, shape (k,), or None, default None
            Fixed class order (including unobserved classes), or sorted active
            classes with None. Per-class results follow this order.
        window_size : positive int or None, default None
            Evaluate the latest W samples, or all received samples with None.

        Returns
        -------
        None
            Initialise an empty metric using shared confusion counts.

        Raises
        ------
        TypeError, ValueError
            If averaging, zero_division, labels, window or binary pos_label
            settings are invalid. Binary class compatibility is also checked
            before each update commits state.

        Notes
        -----
        Time O(k log k); state O(k*k), plus O(W) window pairs when enabled.
        """
        if average not in _AVERAGES:
            raise ValueError(f"average must be one of {_AVERAGES}, got {average!r}")

        if isinstance(zero_division, bool) or not isinstance(
            zero_division, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"zero_division must be a number, got {zero_division!r}")

        zero_division = float(zero_division)

        if np.isnan(zero_division):
            raise ValueError("zero_division must not be NaN")
        if not np.isfinite(zero_division):
            raise ValueError("zero_division must be finite")
        self.average, self.pos_label, self.zero_division = average, pos_label, float(zero_division)
        if average == "binary":
            _check_pos_label(pos_label, np.asarray([pos_label]), strict=True)
        super().__init__(labels=labels, window_size=window_size)
        if len(self.labels_):
            self._validate_classes(self.labels_)

    def _validate_classes(self, classes):
        """Check binary label compatibility before committing an update."""
        if self.average == "binary":
            _check_binary_classes(classes, self.pos_label)

    def result(self):
        """Compute the selected score from active confusion counts.

        Returns
        -------
        float or ndarray, shape (k,)
            A scalar for averaged scores; one value per labels_ entry when
            average=None. Undefined ratios use zero_division.

        Raises
        ------
        RuntimeError
            If no observations are available after construction/reset.

        Notes
        -----
        No parameters. Time O(k*k) for count reductions; temporary space O(k).
        Uses cumulative/window counts, never the average of chunk F1 scores.
        """
        self._require_samples()
        scores = _scores_from_confusion(self._matrix, self.labels_, self.average, self.pos_label, self.zero_division)
        return scores[self._score_index]


class Recall(Precision):
    """Track TP/(TP+FN): how many true members of each class are found.

    Supports average, pos_label, zero_division, labels and window_size; see
    the shared constructor and update/reset/result docstrings. Stored state
    is O(k*k), plus O(W) for an optional sample window.
    """

    _score_index = 1


class F1(Precision):
    """Track 2*TP/(2*TP+FP+FN), balancing precision and recall.

    Supports average, pos_label, zero_division, labels and window_size; see
    the shared constructor and update/reset/result docstrings. Stored state
    is O(k*k), plus O(W) for an optional sample window.
    """

    _score_index = 2


class MSE:
    """Track mean squared error using count-weighted chunk means.

    Parameters
    ----------
    window_size : positive int or None, default None
        Evaluate the latest W sample errors, or all samples with None.

    Attributes
    ----------
    n_samples_, n_samples_seen_ : int
        Number of active samples and total samples received since reset.

    Notes
    -----
    Cumulative state takes O(1) memory. Window state keeps O(W) squared errors.
    Finite 1-D numeric targets/predictions are required; missing rows must be
    filtered beforehand. Queries do not modify state.
    """

    def __init__(self, window_size=None):
        """Set the optional sample window without receiving observations.

        Parameters
        ----------
        window_size : positive int or None, default None
            Window length in samples; None accumulates the entire stream.

        Returns
        -------
        None
            Create empty running error state.

        Raises
        ------
        TypeError, ValueError
            If window_size is not None or a positive nonboolean integer.

        Notes
        -----
        Time/space: O(1).
        """
        self.window_size = _window_size(window_size)
        self.reset()

    def reset(self):
        """Clear errors and counts while keeping window_size.

        Returns
        -------
        MSE
            self, with empty state. result() requires a new nonempty update.

        Notes
        -----
        No parameters or raised exceptions. Time/space O(1); releases stored
        window errors without changing configuration.
        """
        self.n_samples_seen_ = self.n_samples_ = 0
        self._mean = 0.0
        self._errors = np.empty(0)
        return self

    def update(self, y_true, y_pred):
        """Add numeric sample errors, weighted by their actual sample counts.

        Parameters
        ----------
        y_true, y_pred : numeric array_like, shape (b,)
            Finite true/predicted values of equal length. Empty pairs do nothing.

        Returns
        -------
        MSE
            self, with updated counts and mean error.

        Raises
        ------
        TypeError, ValueError
            If types, dimensions, lengths or finite-value checks fail.
        FloatingPointError
            If squared errors overflow float64; rescale targets. State is kept.

        Notes
        -----
        Cumulative update time/temporary space O(b), state O(1). Window mode
        takes O(b+W) time/temporary space and O(W) state. Dividing before sums
        avoids overflow from summing otherwise representable squared errors.
        """
        y_true, y_pred = _check_stream_pair(y_true, y_pred, numeric=True)
        if len(y_true) == 0:
            return self
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            if self.window_size is None:
                count = self.n_samples_ + len(y_true)
                batch_mean = mse(y_true, y_pred)  # Original MSE function, now with a stable sum.
                mean = self._mean * (self.n_samples_ / count) + batch_mean * (len(y_true) / count)
            else:
                squared = (y_true - y_pred) ** 2
                errors = np.concatenate((self._errors, squared))[-self.window_size:].copy()
                count = len(errors)
                mean = np.sum(errors / count)
        self._mean, self.n_samples_ = float(mean), count
        self.n_samples_seen_ += len(y_true)
        if self.window_size is not None:
            self._errors = errors
        return self

    def result(self):
        """Return active mean squared error.

        Returns
        -------
        float
            Nonnegative mean of active squared errors.

        Raises
        ------
        RuntimeError
            If no observations are available after construction/reset.

        Notes
        -----
        No parameters. Time/space O(1); does not clear state.
        """
        if self.n_samples_ == 0:
            raise RuntimeError("no observations; call update before result")
        return self._mean


class ROCCurve:
    """Retain binary labels/scores to query the exact cumulative or window ROC.

    Parameters
    ----------
    pos_label : finite number or nonblank str, default 1
        Positive class. Larger scores mean stronger evidence for this class.
    window_size : positive int or None, default None
        Keep the latest W pairs, or every pair with None.

    Attributes
    ----------
    n_samples_, n_samples_seen_ : int
        Active pair count and total pair count received since reset.

    Notes
    -----
    Exact ROC/AUC require global score ordering, so cumulative mode retains
    O(N) copied labels/scores, not constant-size counters. A window bounds this
    state to O(W). Tied scores cross their threshold together. Single-class
    chunks can update state, but result() needs both positive/negative labels.
    """

    def __init__(self, pos_label=1, window_size=None):
        """Configure the positive class and optional sample window.

        Parameters
        ----------
        pos_label : finite scalar label, default 1
            Positive class, compatible with incoming labels.
        window_size : positive int or None, default None
            Number of active samples, or None to retain all samples.

        Returns
        -------
        None
            Create empty label/score history.

        Raises
        ------
        TypeError, ValueError
            If pos_label or window_size has an invalid type/value.

        Notes
        -----
        Time/space O(1).
        """
        self.pos_label = pos_label
        _check_pos_label(pos_label, np.asarray([pos_label]), strict=True)
        self.window_size = _window_size(window_size)
        self.reset()

    def reset(self):
        """Clear score history and counts while retaining configuration.

        Returns
        -------
        self
            Empty ROC/AUC accumulator. A new result needs observations from
            both the positive and negative classes.

        Notes
        -----
        No parameters or raised exceptions. Time/space O(1); releases old
        label/score arrays and retains pos_label and window_size.
        """
        self._true = self._score = None
        self.n_samples_seen_ = self.n_samples_ = 0
        return self

    def update(self, y_true, y_score):
        """Add a chunk of binary labels and positive-class scores.

        Parameters
        ----------
        y_true : array_like, shape (b,)
            Finite numeric or nonblank string binary labels.
        y_score : numeric array_like, shape (b,)
            Finite scores, not predicted class labels. Empty pairs do nothing.

        Returns
        -------
        self
            Updated copied history, even if only one class has arrived so far.

        Raises
        ------
        TypeError, ValueError
            If shapes, lengths, finite values, types, label families or binary
            class compatibility are invalid. Rejected updates keep old state.

        Notes
        -----
        For M previously retained pairs, time O((M+b)*log(M+b+1)) for finding
        unique labels, temporary space O(M+b). State O(N) cumulatively or O(W)
        with a window. window_size counts samples, including partial chunks.
        """
        y_true, y_score = _check_stream_pair(y_true, y_score, scores=True)
        if len(y_true) == 0:
            return self
        if self._true is not None:
            _check_same_family(y_true, self._true, ("y_true", "previous labels"))
        active_true = y_true if self._true is None else np.concatenate((self._true, y_true))
        active_score = y_score if self._score is None else np.concatenate((self._score, y_score))
        if self.window_size is not None:
            active_true = active_true[-self.window_size:]
            active_score = active_score[-self.window_size:]
        _check_binary_classes(np.unique(active_true), self.pos_label)
        self._true, self._score = active_true.copy(), active_score.copy()
        self.n_samples_seen_ += len(y_true)
        self.n_samples_ = len(active_true)
        return self

    def result(self):
        """Compute ROC points using all active scores, including cross-chunk ties.

        Returns
        -------
        fpr, tpr, thresholds : tuple of ndarray, each shape (m+1,)
            False/true positive rates and decreasing distinct score thresholds;
            the first threshold is infinity and the first rates are zero.

        Raises
        ------
        RuntimeError
            If no observations are available after construction/reset.
        ValueError
            If the active samples lack positives or negatives.

        Notes
        -----
        No parameters. Time O(M log M), temporary space O(M), for M active
        scores. Reuses numcompute.metrics.roc_curve; state is unchanged.
        """
        if self.n_samples_ == 0:
            raise RuntimeError("no observations; call update before result")
        return roc_curve(self._true, self._score, pos_label=self.pos_label)


class AUC(ROCCurve):
    """Query exact ROC area from accumulated binary labels and scores.

    Parameters are pos_label=1 and window_size=None; see ROCCurve.__init__.
    Scores from all active chunks are ranked together. The result is not an
    average of chunk AUC values. Memory is O(N), or O(W) with a sample window.
    See inherited update/reset for accepted shapes and validation errors.
    """

    def result(self):
        """Return the trapezoidal area under the active ROC curve.

        Returns
        -------
        float
            ROC area in [0, 1]; random rankings/tied scores yield about 0.5.

        Raises
        ------
        RuntimeError
            If no observations are available after construction/reset.
        ValueError
            If active samples lack either the positive or negative class.

        Notes
        -----
        No parameters. Time O(M log M), temporary space O(M). Uses the original
        numcompute ROC and trapezoid helpers without changing metric state.
        """
        fpr, tpr, _ = super().result()
        return auc(fpr, tpr)
