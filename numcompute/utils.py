"""Shared helpers. Each member owns the functions in their own block."""
from collections.abc import Iterable, Iterator
from itertools import islice
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "iter_line_batches",
    "check_array",
    "iter_batches",
    "logsumexp",
    "softmax",
    "sigmoid",
    "pairwise_distances",
]


def _as_numeric_array(values: ArrayLike) -> np.ndarray:
    """Validate a one-dimensional real numeric sequence without changing its dtype.

    Parameters
    ----------
    values : array_like, shape (n,)
        Real numeric values; empty input and NaNs are allowed.

    Returns
    -------
    ndarray, shape (n,)
        Input array or an array converted from the sequence.

    Raises
    ------
    TypeError
        If values are nonnumeric or complex.
    ValueError
        If values are not one-dimensional or cannot form a regular array.

    Notes
    -----
    Time/space: O(n) when conversion allocates, otherwise O(1).
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("values must be a one-dimensional sequence")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError("values must contain real numeric values")
    return array

# ---- Jose: iterating line batches ----
def iter_line_batches(items: Iterable, batch_size: int) -> Iterator[list]:
    """Yield consecutive batches of at most ``batch_size`` items.

    Works on any iterable, including ones that can only be read once
    (open files, generators), without loading the whole input into memory.

    Parameters
    ----------
    items : iterable
        Source of items, consumed lazily.
    batch_size : int
        Maximum number of items per batch. Must be a positive integer.

    Returns
    -------
    Iterator[list]
        Lists of length ``batch_size``; the last one may be shorter.
        Nothing is yielded for an empty input.

    Raises
    ------
    TypeError
        If ``batch_size`` is not an integer.
    ValueError
        If ``batch_size < 1``.

    Notes
    -----
    Time Complexity: O(n: number of items), over all batches.
    Space Complexity: O(b: batch_size), only one batch is held at a time.

    Examples
    --------
    >>> list(iter_line_batches(range(5), 2))
    [[0, 1], [2, 3], [4]]
    """
    if not isinstance(batch_size, int):
        raise TypeError(f"batch_size must be an int, got {type(batch_size).__name__}")

    if batch_size < 1:
        raise ValueError(f"batch_size must be greater or equal than 1, got {batch_size}")

    it = iter(items)                          # one iterator: each islice continues from the last
    batch = list(islice(it, batch_size))
    while batch:
        yield batch
        batch = list(islice(it, batch_size))


# ---- Dat: validation and batching ----

def check_array(X, ndim: int = None) -> np.ndarray:
    """Convert input to a nonempty floating-point array and validate its dimensions.

    Parameters
    ----------
    X : array_like
        Values convertible to float; NaN and infinity are not rejected.
    ndim : int or None, default None
        Required number of dimensions; None allows any nonempty shape.

    Returns
    -------
    ndarray of float64, same shape as X
        May share storage with an existing float64 array.

    Raises
    ------
    TypeError
        If X cannot be converted to floating-point data.
    ValueError
        If X is empty or has a different number of dimensions.

    Notes
    -----
    Time/space: O(N) if conversion allocates, otherwise O(1), N=X.size.
    """
    if np.iscomplexobj(X):
        raise TypeError("X must contain real numeric values")
    try:
        X = np.asarray(X, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("X must be numeric")

    if X.size == 0:
        raise ValueError("X is empty")
    if ndim is not None and X.ndim != ndim:
        raise ValueError(f"X must be {ndim}-D, got shape {X.shape}; use X.reshape(-1, 1) for one feature")
    return X


def iter_batches(X: np.ndarray, batch_size: int):
    """Yield consecutive row slices, including a shorter final batch.

    Parameters
    ----------
    X : array_like, shape (n, ...)
        Input with a row axis; converted with np.asarray.
    batch_size : int
        Positive maximum row count per batch.

    Returns
    -------
    iterator of ndarray, shape (b, ...)
        Views of the converted array, with b <= batch_size. Empty input
        with a row axis yields no batches.

    Raises
    ------
    ValueError
        If batch_size is not positive.
    TypeError
        If batch_size is not usable as a range step.
    IndexError
        If X is scalar and has no row axis.

    Notes
    -----
    Time: O(ceil(n / batch_size)) iterations after conversion. Extra space
    is O(1) for array input; converting a sequence may allocate its full size.
    Validation occurs when the iterator is consumed.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    X = np.asarray(X)
    for start in range(0, X.shape[0], batch_size):
        yield X[start:start + batch_size]


# ---- Jose: numerically stable activations and distances ----

def _shifted_exp(x, axis):
    """Return exponentials and maxima, masking nonfinite slices before exp.

    x is a nonempty float array; axis follows NumPy reduction semantics.
    Returns (exponentials of x's shape, maxima with reduced axes retained).
    Invalid axes raise ValueError. Time and space: O(x.size).
    """
    maximum = np.max(x, axis=axis, keepdims=True)
    finite = np.isfinite(maximum)
    shifted = np.full_like(x, -np.inf)
    # A very negative difference may become -inf, whose exponential is zero.
    with np.errstate(over="ignore", under="ignore"):
        np.subtract(x, np.where(finite, maximum, 0.), out=shifted,
                    where=np.broadcast_to(finite, x.shape))
        exponentials = np.exp(shifted)
    return exponentials, maximum


def logsumexp(x: ArrayLike, axis: int = None):
    """Compute log(sum(exp(x))) without overflow or underflow.

    Uses the max-shift identity log(sum(exp(x))) = m + log(sum(exp(x - m))),
    with m = max(x), so the largest term is exp(0) = 1.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Values converted to float, e.g. log-probabilities or logits.
    axis : int or None, default None
        Axis to reduce; None reduces all values.

    Returns
    -------
    float or ndarray
        A Python float when axis=None; otherwise the input shape with the
        axis removed. A slice of only -inf gives -inf, +inf gives +inf and
        NaN propagates.

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty or axis is invalid.

    Notes
    -----
    Time Complexity: O(N), N=x.size.
    Space Complexity: O(N) for the shifted temporary array.

    Examples
    --------
    >>> logsumexp([1000.0, 1000.0])          # the naive formula overflows to inf
    1000.6931471805599
    """
    x = check_array(x)
    exponentials, maximum = _shifted_exp(x, axis)
    finite = np.isfinite(maximum)
    logged = np.zeros_like(maximum)
    np.log(np.sum(exponentials, axis=axis, keepdims=True), out=logged, where=finite)
    out = np.where(finite, logged + np.where(finite, maximum, 0.), maximum)

    if axis is None:
        return float(out.reshape(()))
    return np.squeeze(out, axis=axis)


def softmax(x: ArrayLike, axis: int = -1) -> np.ndarray:
    """Convert scores to probabilities that sum to 1 along an axis.

    Subtracts the maximum of each slice before exponentiating (max-shifted
    softmax), so large scores do not overflow. The result does not change
    when a constant is added to every score in a slice.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Scores (logits) converted to float; for (n, k) input each row is one
        sample's k class scores.
    axis : int, default -1
        Axis along which the probabilities sum to 1 (the last axis by default).

    Returns
    -------
    ndarray of float, same shape as x
        Values in [0, 1]. A slice containing NaN or +inf, or made only of
        -inf, gives NaN.

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty or axis is invalid.

    Notes
    -----
    Time Complexity: O(N), N=x.size.
    Space Complexity: O(N) for the exponentials and the output.

    Examples
    --------
    >>> softmax([1000.0, 1000.0])            # naive exp(1000) overflows
    array([0.5, 0.5])
    """
    x = check_array(x)
    exp, _ = _shifted_exp(x, axis)
    total = np.sum(exp, axis=axis, keepdims=True)
    with np.errstate(under="ignore"):
        return np.divide(exp, total, out=np.full_like(x, np.nan), where=total > 0)


def sigmoid(x: ArrayLike):
    """Compute the logistic function 1 / (1 + exp(-x)) without overflow.

    Uses z = exp(-|x|), which never exceeds 1: x >= 0 gives 1 / (1 + z) and
    x < 0 gives z / (1 + z), the same value written in a safe form.

    Parameters
    ----------
    x : numeric array_like, any nonempty shape
        Values converted to float.

    Returns
    -------
    float or ndarray of float
        A Python float for scalar input; otherwise the same shape as x.
        Values are in [0, 1]; NaN propagates.

    Raises
    ------
    TypeError
        If x cannot be converted to numeric data.
    ValueError
        If x is empty.

    Notes
    -----
    Time Complexity: O(N), N=x.size.
    Space Complexity: O(N) for temporaries and the output.

    Examples
    --------
    >>> sigmoid(0.0)
    0.5
    >>> float(sigmoid([-1000.0])[0])          # naive exp(1000) overflows
    0.0
    """
    x = check_array(x)
    with np.errstate(under="ignore"):
        z = np.exp(-np.abs(x))
    out = np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))

    if out.ndim == 0:
        return float(out)
    return out


def pairwise_distances(X: ArrayLike, Y: ArrayLike = None) -> np.ndarray:
    """Euclidean distance between every row of X and every row of Y.

    Accumulates coordinate differences with hypot, vectorised over all pairs.
    A feature loop avoids an (n, m, d) temporary. Direct differences preserve
    small distances on mixed-scale data; hypot avoids squaring overflow.

    Parameters
    ----------
    X : numeric array_like, shape (n, d)
        First set of points, one per row.
    Y : numeric array_like, shape (m, d), or None, default None
        Second set of points; None compares X with itself.

    Returns
    -------
    ndarray of float, shape (n, m)
        Entry [i, j] is the distance between X[i] and Y[j]. With Y=None the
        result is (n, n), symmetric, with an exact zero diagonal.

    Raises
    ------
    TypeError
        If X or Y cannot be converted to numeric data.
    ValueError
        If X or Y is empty or not 2-D, or they have different feature counts.

    Notes
    -----
    Time Complexity: O(n * m * d).
    Space Complexity: O(n * m + (n + m) * d).

    Examples
    --------
    >>> pairwise_distances([[0.0, 0.0], [3.0, 4.0]])
    array([[0., 5.],
           [5., 0.]])
    """
    X = check_array(X, ndim=2)
    same = Y is None
    Y = X if same else check_array(Y, ndim=2)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"X and Y need the same number of features, got {X.shape[1]} and {Y.shape[1]}")

    distances = np.zeros((len(X), len(Y)))
    with np.errstate(under="ignore"):
        for left, right in zip(X.T, Y.T):
            np.hypot(distances, left[:, None] - right[None, :], out=distances)
    if same:
        np.fill_diagonal(distances, 0.0)
    return distances


def _check_label_precision(*arrays):
    """Reject numeric label promotion that would change integer values.

    Arrays have real numeric or string dtypes. Returns None, or raises
    TypeError for lossy promotion. Time/space: O(total label count).
    """
    dtype = np.result_type(*(array.dtype for array in arrays))
    if dtype.kind != "f":
        return
    for array in arrays:
        if array.dtype.kind in "iu":
            with np.errstate(all="ignore"):
                restored = array.astype(dtype).astype(array.dtype)
            if not np.array_equal(array, restored):
                raise TypeError("numeric labels need a consistent dtype; promotion must be lossless")


def _mean_square(errors):
    """Mean square of a nonempty finite residual array, scaling before squaring.

    Returns a float. Unrepresentable results raise FloatingPointError.
    Time/space: O(errors.size). Tiny squared errors may round to zero.
    """
    with np.errstate(over="ignore", under="ignore"):
        squared = np.square(errors)
    if np.isfinite(squared).all():
        with np.errstate(under="ignore"):
            return float(np.sum(squared / errors.size))
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        scaled = errors / np.sqrt(errors.size)
        return float(np.sum(np.square(scaled)))


def _trapezoid_area(x, y):
    """Signed trapezoid sum for finite 1-D arrays of equal length >= 2.

    Power-of-two scaling avoids overflow in widths and endpoint sums.
    Returns a float; unrepresentable results raise FloatingPointError.
    Time/space: O(len(x)). Tiny final areas may round to zero.
    """
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        x_mantissa, x_exponent = np.frexp(x)
        x_exponent = np.where(x_mantissa == 0, -1074, x_exponent)
        x_scale = np.maximum(x_exponent[1:], x_exponent[:-1])
        width = (np.ldexp(x_mantissa[1:], x_exponent[1:] - x_scale)
                 - np.ldexp(x_mantissa[:-1], x_exponent[:-1] - x_scale))
        width, width_exponent = np.frexp(width)
        y_mantissa, y_exponent = np.frexp(y)
        y_exponent = np.where(y_mantissa == 0, -1074, y_exponent)
        y_scale = np.maximum(y_exponent[1:], y_exponent[:-1])
        midpoint = (np.ldexp(y_mantissa[1:], y_exponent[1:] - y_scale)
                    + np.ldexp(y_mantissa[:-1], y_exponent[:-1] - y_scale)) / 2.
        midpoint, midpoint_exponent = np.frexp(midpoint)
        terms = width * midpoint
        exponents = x_scale + width_exponent + y_scale + midpoint_exponent
        scale = np.max(np.where(terms != 0, exponents, 0))
        total = np.sum(np.ldexp(terms, exponents - scale))
        return float(np.ldexp(total, scale))


if __name__ == "__main__":
    assert check_array([[1, 2]]).dtype == float

    sizes = [batch.shape[0] for batch in iter_batches(np.zeros((10, 1)), 4)]
    assert sizes == [4, 4, 2]

    try:
        check_array([1.0], ndim=2)
        assert False, "1-D input should be rejected"
    except ValueError:
        pass

    print("utils ok")
