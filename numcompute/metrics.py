"""Accuracy, precision, recall, F1, MSE, confusion matrix, ROC, and AUC.

Classification metrics take 1-D label arrays (int, float, bool or str labels).
Precision, recall and F1 are all computed from one confusion matrix, so they
share the same counts. Scalar results are returned as Python floats.
Numeric label dtype changes must be lossless; otherwise TypeError is raised.
"""

from collections.abc import Sequence

import numpy as np
from .utils import _check_label_precision, _mean_square, _trapezoid_area

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
        If an input is not 1-D, is empty or nonfinite, or the lengths differ.
    FloatingPointError
        If the mean square is outside float64 range.

    Notes
    -----
    Time Complexity: O(n: number of samples).
    Space Complexity: O(n) for the difference array.

    Examples
    --------
    >>> mse([1, 2, 3], [1, 2, 5])
    1.3333333333333333
    """
    y_true = _check_values(y_true, "y_true")
    y_pred = _check_values(y_pred, "y_pred")
    _check_same_length(y_true, y_pred, ("y_true", "y_pred"))

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("mse inputs must be finite")
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        return _mean_square(y_true - y_pred)


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
        lengths differ; an input is nonfinite; or ``x`` is not monotonic.
    FloatingPointError
        If the final area is outside float64 range.

    Notes
    -----
    Time Complexity: O(m: number of points).
    Space Complexity: O(m).
    Power-of-two scaling avoids overflow in widths and endpoint sums.

    Examples
    --------
    >>> auc([0, 1], [0, 1])
    0.5
    """
    x = _check_values(x, "x")
    y = _check_values(y, "y")
    _check_same_length(x, y, ("x", "y"))

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("auc coordinates must be finite")

    if x.size < 2:
        raise ValueError(f"auc needs at least 2 points, got {x.size}")

    decreases = x[1:] < x[:-1]
    increases = x[1:] > x[:-1]
    if decreases.any() and increases.any():
        raise ValueError("x must be monotonic (all increasing or all decreasing)")
    direction = -1.0 if decreases.any() else 1.0
    return float(direction * _trapezoid_area(x, y))


def _check_labels(
    y: np.ndarray | Sequence[int | str | float],
    name: str,
) -> np.ndarray:
    """Validate a nonempty vector of numeric or string labels.

    Parameters
    ----------
    y : array_like, shape (n,)
        Labels of a supported numeric, boolean or text dtype.
    name : str
        Argument name included in error messages.

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
    if y.size == 0:
        raise ValueError(f"{name} is empty")

    # Check for NaN, only on float labels: no other kind can hold it, and NaN never equals itself
    if y.dtype.kind == "f" and np.isnan(y).any():
        raise ValueError(f"{name} contains NaN")

    return y


def _check_values(
    y: np.ndarray | Sequence[int | float],
    name: str,
) -> np.ndarray:
    """Validate a nonempty numeric vector and convert it to float.

    Parameters
    ----------
    y : numeric array_like, shape (n,)
        Input values; numeric-looking strings are not accepted.
    name : str
        Argument name used in validation messages.

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
    if y.size == 0:
        raise ValueError(f"{name} is empty")

    # Converting to float for the arithmetic; no copy when the input is already float64
    y = y.astype(float, copy=False)

    # Check for NaN, after converting so it always runs on float data
    if np.isnan(y).any():
        raise ValueError(f"{name} contains NaN")

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
        If label families differ or numeric dtype promotion would change a label.

    Notes
    -----
    Time and space: O(n + m) for numeric precision checks.
    """
    # Check if both arrays hold the same family of labels: NumPy would compare 1 with "1" as unequal
    # (or silently promote 1 to "1"), giving wrong scores instead of an error
    if _LABEL_FAMILIES[a.dtype.kind] != _LABEL_FAMILIES[b.dtype.kind]:
        raise TypeError(
            f"{names[0]} ({a.dtype}) and {names[1]} ({b.dtype}) hold labels of incompatible types"
        )
    _check_label_precision(a, b)


def _check_pos_label(
    pos_label: int | float | str,
    y_true: np.ndarray
) -> np.ndarray:
    """Validate a scalar positive label against the true-label family.

    Parameters
    ----------
    pos_label : scalar
        Positive class identifier.
    y_true : ndarray, shape (n,)
        Previously validated true labels.

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

    pos = _check_labels([pos_label], "pos_label")
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
    _check_label_precision(y, classes)
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

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp  # predicted as the class, truly another
    fn = cm.sum(axis=1) - tp  # truly the class, predicted as another
    support = cm.sum(axis=1)

    if average == "binary":
        idx = _label_index(pos, classes)[0]
        tp, fp, fn = tp[idx], fp[idx], fn[idx]

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
