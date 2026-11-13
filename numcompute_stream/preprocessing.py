"""Incremental scalers and imputation for (samples, features) arrays.

Reuse the original NumCompute transform and fit_transform implementations.
fit starts afresh; partial_fit learns from all chunks seen so far. NaNs are
ignored separately in each column. One-hot encoding will be added separately.
"""

import numpy as np

from numcompute.preprocessing import (
    MISSING,
    MinMaxScaler as _BatchMinMaxScaler,
    SimpleImputer as _BatchSimpleImputer,
    StandardScaler as _BatchStandardScaler,
)
from .stats import RunningStats

__all__ = ["StandardScaler", "MinMaxScaler", "SimpleImputer"]


def share_array(X, transformer, reset=False):
    """Validate chunks using the transformer's numeric/string mode.

    Parameters
    ----------
    X : array_like, shape (b, d)
        Input data. Numeric NaNs are allowed; infinity/complex values are not.
    transformer : object
        Supplies the optional fill_value and learned n_features_in_.
    reset : bool, default False
        Require nonempty input and permit a new feature count for fit.

    Returns
    -------
    ndarray, shape (b, d)
        Float data, or strings when a string fill_value is configured.

    Raises
    ------
    TypeError
        If numeric input cannot be converted to real numbers.
    ValueError
        If shape/features are invalid, infinity is present or fit is empty.

    Notes
    -----
    Time/temporary space: O(b*d), assuming bounded string lengths.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("X must be a 2-D array (samples, features) with at least one feature")
    strings = isinstance(getattr(transformer, "fill_value", None), str)
    if strings:
        X = X.astype(str)
    else:
        if X.dtype.kind == "c":
            raise TypeError("X must contain real numeric values, not complex numbers")
        try:
            X = X.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypeError("X must contain numeric values") from exc
        if np.isinf(X).any():
            raise ValueError("X must not contain infinity; NaN is allowed")
    if reset and len(X) == 0:
        raise ValueError("fit requires at least one sample")
    if not reset and hasattr(transformer, "n_features_in_"):
        if X.shape[1] != transformer.n_features_in_:
            raise ValueError(f"expected {transformer.n_features_in_} features, got {X.shape[1]}")
    return X


class _StreamingFit:
    """Share the streaming API while inheriting batch conversion formulas."""

    def fit(self, X):
        """Replace previous state with parameters learned from X.

        Parameters
        ----------
        X : array_like, shape (b, d)
            Nonempty training data. Numeric NaNs are ignored per column;
            string constant imputation treats '', 'nan' and 'None' as missing.

        Returns
        -------
        self
            Fitted transformer. Subsequent partial_fit calls extend this fit.

        Raises
        ------
        TypeError
            If numeric input cannot be converted to real numbers.
        ValueError
            If X is empty, not 2-D, has no features or contains infinity.
        FloatingPointError
            If numeric parameters exceed float64 range; rescale the data.

        Notes
        -----
        Time/temporary space: O(b*d), except median imputation, whose time
        is conservatively O(d*b*log(b)). Stored state is O(d), or O(b*d) for
        exact median imputation. A new fit may change the feature count.
        """
        return self._learn(X, reset=True)

    def partial_fit(self, X):
        """Update parameters from a new chunk without forgetting earlier rows.

        Parameters
        ----------
        X : array_like, shape (b, d)
            New observations with the fitted feature count. NaNs are ignored
            per column. Empty chunks are a no-op and do not initialize state.

        Returns
        -------
        self
            Updated transformer. This method does not transform the chunk.

        Raises
        ------
        TypeError
            If numeric input cannot be converted to real numbers.
        ValueError
            If dimensions/features are invalid or infinity is present.
        FloatingPointError
            If numeric parameters exceed float64 range; rescale the data.

        Notes
        -----
        Time/temporary space: O(b*d); stored state: O(d). Exact median uses
        O(N*d) stored/temporary space and conservatively O(d*N*log(N)) time
        per update, for N total rows. Filled values are never observations.
        """
        return self._learn(X, reset=False)

    def transform(self, X):
        """Apply current parameters without updating them.

        Parameters
        ----------
        X : array_like, shape (b, d)
            Data with the fitted feature count; empty batches are allowed.

        Returns
        -------
        ndarray, shape (b, d)
            New float array, or strings for categorical constant imputation.
            Scalers preserve NaNs; imputers replace them with statistics_.

        Raises
        ------
        RuntimeError
            If no nonempty fit or partial_fit has completed.
        TypeError
            If numeric input cannot be converted to real numbers.
        ValueError
            If dimensions/features are invalid or infinity is present.

        Notes
        -----
        Time/space: O(b*d), assuming bounded string lengths. Uses the old
        NumCompute formulas. Input and learned parameters are not modified.
        """
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("call fit or partial_fit before transform")
        X = share_array(X, self)
        if len(X) == 0:
            return X.copy()
        return super().transform(X)


class StandardScaler(_StreamingFit, _BatchStandardScaler):
    """Standardize columns using cumulative mean and population variance.

    Attributes
    ----------
    mean_, var_, scale_ : ndarray, shape (d,)
        Running mean, population variance (ddof=0), and standard deviation.
        Zero scales become one. Unobserved columns have NaN parameters.
    n_samples_seen_ : ndarray of int, shape (d,)
        Nonmissing observation counts, separately for each feature.
    n_features_in_ : int
        Fitted feature count.

    Notes
    -----
    Uses RunningStats (Chan/Welford merging), not an average of chunk means.
    fit resets state; partial_fit accumulates; fit_transform retains its old
    interface. Constant columns map to zero; NaNs pass through unchanged.
    All-NaN columns recover when observed later, without reduction warnings.
    Update time/temporary space is O(b*d), persistent state O(d).
    Exceptions and input/output shapes are documented on the shared methods.
    """

    def _learn(self, X, reset):
        """Learn/reset moments and publish scaler parameters; O(b*d)."""
        X = share_array(X, self, reset)
        if len(X) == 0:
            return self
        stats = RunningStats(store_data=False) if reset or not hasattr(self, "_stats") else self._stats
        stats.update(X)
        self.mean_ = stats.mean
        self.var_ = stats.var()
        self.scale_ = np.where(self.var_ == 0, 1.0, np.sqrt(self.var_))
        self.n_samples_seen_ = stats.count_
        self.n_features_in_ = X.shape[1]
        self._stats = stats
        return self


class MinMaxScaler(_StreamingFit, _BatchMinMaxScaler):
    """Scale columns with cumulative minima/maxima, ignoring NaNs.

    Parameters
    ----------
    feature_range : pair of finite numbers, default (0.0, 1.0)
        Output endpoints (low, high), with a finite positive width.

    Attributes
    ----------
    data_min_, data_max_, data_range_ : ndarray, shape (d,)
        Observed extrema and their differences. As in the original scaler,
        constant-column ranges become infinity so they map to low. Columns
        never observed have NaN parameters and can recover in later chunks.
    n_samples_seen_ : ndarray of int, shape (d,)
        Nonmissing observation counts for each feature.
    n_features_in_ : int
        Fitted feature count.

    Raises
    ------
    ValueError
        If feature_range is not two finite endpoints with finite low < high.

    Notes
    -----
    Updates take O(b*d) time/temporary space; persistent state is O(d).
    Transform does not clip: new values can fall outside feature_range.
    See shared methods for input shapes and fitting/transform exceptions.
    """

    def __init__(self, feature_range=(0.0, 1.0)):
        """Configure the desired scaling interval.

        Parameters
        ----------
        feature_range : pair of finite numbers, default (0.0, 1.0)
            Lower and upper endpoints with a finite positive difference.

        Returns
        -------
        None
            Store configuration without fitting data.

        Raises
        ------
        ValueError
            If two valid endpoints cannot be obtained.

        Notes
        -----
        Time/space: O(1) for a pair of endpoints.
        """
        try:
            bounds = np.asarray(feature_range, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("feature_range must contain two numbers") from exc
        if bounds.shape != (2,):
            raise ValueError("feature_range must contain exactly two numbers")
        low, high = bounds
        if not np.isfinite(bounds).all() or low >= high:
            raise ValueError("feature_range must have two finite endpoints with low < high")
        with np.errstate(over="ignore"):
            width = high - low
        if not np.isfinite(width):
            raise ValueError("feature_range must have two finite endpoints with finite low < high")
        super().__init__(tuple(bounds))

    def _learn(self, X, reset):
        """Merge extrema and counts without retaining rows; O(b*d)."""
        X = share_array(X, self, reset)
        if len(X) == 0:
            return self
        valid = ~np.isnan(X)
        count = valid.sum(axis=0)
        minimum = np.min(np.where(valid, X, np.inf), axis=0)
        maximum = np.max(np.where(valid, X, -np.inf), axis=0)
        if not reset and hasattr(self, "n_features_in_"):
            count = count + self.n_samples_seen_
            minimum = np.fmin(minimum, self.data_min_)
            maximum = np.fmax(maximum, self.data_max_)
        minimum = np.where(count > 0, minimum, np.nan)
        maximum = np.where(count > 0, maximum, np.nan)
        try:
            with np.errstate(over="raise", invalid="raise"):
                data_range = maximum - minimum
        except FloatingPointError as exc:
            raise FloatingPointError("feature ranges exceed float64; rescale the data") from exc
        self.data_min_, self.data_max_ = minimum, maximum
        self.data_range_ = np.where(data_range == 0, np.inf, data_range)
        self.n_samples_seen_, self.n_features_in_ = count, X.shape[1]
        return self


class SimpleImputer(_StreamingFit, _BatchSimpleImputer):
    """Update numeric fill estimates, or use a fixed numeric/string constant.

    Parameters
    ----------
    strategy : {'mean', 'median', 'constant'} or None, default None
        None selects mean, or constant if fill_value is supplied.
    fill_value : finite number, str or None, default None
        Constant replacement and fallback for columns with no observations.
        None becomes 0.0. Strings require strategy='constant'.

    Attributes
    ----------
    statistics_ : ndarray, shape (d,)
        Current fill values. Unobserved columns use fill_value until valid
        observations arrive. Only original, nonmissing cells are learned.
    n_samples_seen_ : ndarray of int, shape (d,)
        Nonmissing observation counts for each feature.
    n_features_in_ : int
        Fitted feature count.

    Raises
    ------
    TypeError, ValueError
        If strategy or fill_value is invalid. See shared methods for data errors.

    Notes
    -----
    Numeric mode fills NaN. String mode fills '', 'nan' and 'None', matching
    the original imputer. Mean/constant updates take O(b*d) time/temporary
    space and O(d) stored state. Exact median retains private chunk copies,
    taking O(N*d) space and conservatively O(d*N*log(N)) time per update.
    No Python loops are used in the numerical calculations.
    """

    def __init__(self, strategy=None, fill_value=None):
        """Configure constant, mean or exact median filling.

        Parameters
        ----------
        strategy : {'mean', 'median', 'constant'} or None, default None
            None selects mean, or constant when fill_value is supplied.
        fill_value : finite number, str or None, default None
            Replacement/fallback. None becomes zero; strings require constant.

        Returns
        -------
        None
            Store configuration without fitting data.

        Raises
        ------
        TypeError, ValueError
            If the strategy or fill value is invalid or numeric fill is nonfinite.

        Notes
        -----
        Time/space: O(1), excluding string parameter storage.
        """
        super().__init__(strategy=strategy, fill_value=fill_value)
        if not isinstance(self.fill_value, str) and not np.isfinite(self.fill_value):
            raise ValueError("numeric fill_value must be finite")

    def _learn(self, X, reset):
        """Update counts/estimates; exact median retains and revisits history."""
        X = share_array(X, self, reset)
        if len(X) == 0:
            return self
        missing = np.isin(X, MISSING) if X.dtype.kind == "U" else np.isnan(X)
        batch_count = (~missing).sum(axis=0)
        previous = not reset and hasattr(self, "n_features_in_")
        old_count = self.n_samples_seen_ if previous else np.zeros(X.shape[1], dtype=np.int64)
        count = old_count + batch_count
        if self.strategy == "constant":
            statistics = np.full(X.shape[1], self.fill_value)
        elif self.strategy == "mean":
            # Divide before summing to avoid overflow from summing large values.
            # Count weights ensure unequal chunks and missing cells are handled.
            old_mean = self._mean if previous else np.zeros(X.shape[1])
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                batch_mean = np.divide(np.where(missing, 0.0, X), batch_count,
                                       out=np.zeros_like(X), where=batch_count > 0).sum(axis=0)
                old_weight = np.divide(old_count, count, out=np.zeros(X.shape[1]), where=count > 0)
                new_weight = np.divide(batch_count, count, out=np.zeros(X.shape[1]), where=count > 0)
                mean = old_mean * old_weight + batch_mean * new_weight
            statistics = np.where(count > 0, mean, self.fill_value)
        else:
            chunks = self._chunks + [X.copy()] if previous else [X.copy()]
            history = np.concatenate(chunks, axis=0)
            # Avoid warnings for all-NaN columns; they receive the fallback below.
            safe = np.where(count == 0, 0.0, history)
            with np.errstate(over="raise", invalid="raise"):
                statistics = np.nanmedian(safe, axis=0)
            statistics = np.where(count > 0, statistics, self.fill_value)
        self.statistics_ = statistics
        self.n_samples_seen_, self.n_features_in_ = count, X.shape[1]
        if self.strategy == "mean":
            self._mean = mean
        elif self.strategy == "median":
            self._chunks = chunks
        return self
