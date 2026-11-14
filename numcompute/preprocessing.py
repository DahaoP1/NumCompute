"""Scalers, imputer and one-hot encoder with the fit / transform / fit_transform API.

X is always (n_samples, n_features). Column statistics ignore NaN; NaN passes
through transform unchanged, so impute first if that is not wanted.
"""

import numpy as np

from .utils import check_array


MISSING = ("", "nan", "None")      # string cells treated as missing by SimpleImputer and OneHotEncoder


def _column_mean(X):
    """NaN-skipping means of a 2-D float array; empty columns return NaN.

    Shift by an observed value to preserve tiny constants. If the shift
    overflows, average scaled values instead. Time/space: O(X.size).
    """
    valid = ~np.isnan(X)
    count = valid.sum(axis=0)
    reference = np.where(count > 0, X[np.argmax(valid, axis=0), np.arange(X.shape[1])], 0.)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        centred = np.where(valid, X, reference) - reference
        mean = reference + np.divide(centred.sum(axis=0), count,
                                    out=np.zeros(X.shape[1]), where=count > 0)
        fallback = ~np.isfinite(mean) & (count > 0)
        if fallback.any():
            values = np.where(valid[:, fallback], X[:, fallback], 0.)
            mean[fallback] = np.sum(values / count[fallback], axis=0)
    return np.where(count > 0, mean, np.nan)


def _column_median(X):
    """NaN-skipping medians of a 2-D float array, without midpoint overflow.

    Empty columns return NaN. Sorting copies input. Time: O(d*n*log(n));
    space: O(n*d), for X shaped (n, d).
    """
    values = np.sort(X, axis=0)
    count = (~np.isnan(values)).sum(axis=0)
    columns = np.arange(X.shape[1])
    lower = values[np.maximum((count - 1) // 2, 0), columns]
    upper = values[np.maximum(count // 2, 0), columns]
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        total = lower + upper
        middle = np.where(np.isfinite(total), total / 2., lower / 2. + upper / 2.)
    return np.where(count > 0, middle, np.nan)


def _scale_minmax(X, minimum, data_range, low, high):
    """Apply the MinMax affine map, repairing overflowing intermediate values.

    X is (n, d); fitted arrays are (d,). Returns a float array of X's shape.
    True output overflow raises FloatingPointError. Time/space: O(n*d).
    """
    width = high - low
    with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
        unit = (X - minimum) / data_range
        result = low + unit * width
    result = np.where(np.isinf(data_range) & np.isfinite(X), low, result)
    tiny_unit = (np.abs(unit) < np.finfo(float).tiny) & (X != minimum)
    repair = np.isfinite(X) & np.isfinite(minimum) & np.isfinite(data_range) & (~np.isfinite(result) | tiny_unit)
    if repair.any():
        # Work with power-of-two mantissas only for the exceptional cells.
        values = X[repair]
        minima = np.broadcast_to(minimum, X.shape)[repair]
        ranges = np.broadcast_to(data_range, X.shape)[repair]
        left, left_exp = np.frexp(values)
        right, right_exp = np.frexp(minima)
        exponent = np.maximum(left_exp, right_exp)
        with np.errstate(under="ignore", over="raise", invalid="raise", divide="raise"):
            delta = np.ldexp(left, left_exp - exponent) - np.ldexp(right, right_exp - exponent)
            delta, delta_exp = np.frexp(delta)
            range_mantissa, range_exp = np.frexp(ranges)
            width_mantissa, width_exp = np.frexp(width)
            term, term_exp = np.frexp(delta * width_mantissa / range_mantissa)
            term_exp = term_exp + delta_exp + exponent + width_exp - range_exp
            low_mantissa, low_exp = np.frexp(low)
            exponent = np.maximum(term_exp, low_exp)
            combined = np.ldexp(term, term_exp - exponent) + np.ldexp(low_mantissa, low_exp - exponent)
            result[repair] = np.ldexp(combined, exponent)
    return result


def _check_fitted(transformer, X: np.ndarray) -> None:
    """Check fitted state and the number of input features.

    Parameters
    ----------
    transformer : object
        Object expected to have n_features_in_.
    X : ndarray, shape (n_samples, n_features)
        Input whose shape is checked.

    Returns
    -------
    None
        Only validation is performed.

    Raises
    ------
    RuntimeError
        If fitted feature metadata is absent.
    ValueError
        If X is not 2-D or its feature count differs.

    Notes
    -----
    Time and extra space: O(1).
    """
    if not hasattr(transformer, "n_features_in_"):
        raise RuntimeError(f"{type(transformer).__name__} is not fitted; call fit(X) first")
    if X.ndim != 2 or X.shape[1] != transformer.n_features_in_:
        raise ValueError(f"expected {transformer.n_features_in_} features, got shape {X.shape}")


class StandardScaler:
    """Standardize each feature using its fitted mean and population standard deviation.

    Attributes
    ----------
    mean_ : ndarray, shape (n_features,)
        Column means, ignoring NaN.
    scale_ : ndarray, shape (n_features,)
        Population standard deviations, with zero scales replaced by one.
    n_features_in_ : int
        Number of input columns recorded during fit.

    Raises
    ------
    TypeError
        If data cannot be converted to numbers.
    ValueError
        If input shape is invalid or the feature count changes.
    RuntimeError
        If transform is called before fit.

    Notes
    -----
    Training constant columns become zero; NaNs pass through transform.
    All-NaN columns retain NaN statistics and may emit NumPy warnings.
    Fit and transform take O(n*d) time for n samples and d features. Stored
    state is O(d); conversion, reduction temporaries and output can use O(n*d).
    """

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Learn column means and population standard deviations, ignoring NaN.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Nonempty numeric data; NaNs are allowed.

        Returns
        -------
        StandardScaler
            The fitted instance (self).

        Raises
        ------
        TypeError
            If X cannot be converted to numbers.
        ValueError
            If X is empty or not two-dimensional.

        Notes
        -----
        Time: O(n_samples * n_features). Extra space: O(n_samples * n_features)
        including dtype conversion and NumPy reduction temporaries; stored state O(n_features).
        All-NaN columns retain NaN statistics and may emit NumPy warnings.
        """
        X = check_array(X, ndim=2)

        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_[self.scale_ == 0] = 1.0       # constant column: do not divide by zero
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize columns using the fitted means and scales.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Nonempty numeric data; NaNs are allowed.

        Returns
        -------
        ndarray, shape (n_samples, n_features)
            Floating-point transformed data; NaNs remain NaN.

        Raises
        ------
        TypeError
            If X cannot be converted to numbers.
        ValueError
            If X is empty or not two-dimensional.
            Also raised if the feature count differs from fit.
        RuntimeError
            If fit has not been called.

        Notes
        -----
        Time and space: O(n_samples * n_features), including the output.
        Does not refit. Training constant columns map to zero.
        """
        X = check_array(X, ndim=2)
        _check_fitted(self, X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return the transformed training data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Nonempty numeric data; NaNs are allowed.

        Returns
        -------
        ndarray, shape (n_samples, n_features)
            Floating-point transformed data; NaNs are preserved.

        Raises
        ------
        TypeError
            If X cannot be converted to numbers.
        ValueError
            If X is empty or not two-dimensional.

        Notes
        -----
        Time and space: O(n_samples * n_features), including the output.
        Equivalent to self.fit(X).transform(X).
        """
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Scale each feature using its fitted minimum and range.

    Parameters
    ----------
    feature_range : pair of numbers, default (0.0, 1.0)
        Desired lower and upper endpoints, with low < high.

    Attributes
    ----------
    feature_range : tuple
        Configured output interval.
    data_min_ : ndarray, shape (n_features,)
        Column minima, ignoring NaN.
    data_range_ : ndarray, shape (n_features,)
        Column ranges; constant-column ranges are replaced by infinity.
    n_features_in_ : int
        Number of fitted columns.

    Raises
    ------
    TypeError
        If endpoints cannot be compared or data cannot be converted to numbers.
    ValueError
        If the range or input shape is invalid, or the feature count changes.
    RuntimeError
        If transform is called before fit.

    Notes
    -----
    Constant columns map to the lower endpoint. NaNs remain NaN; new values
    may lie outside feature_range. Fit/transform take O(n*d) time. Stored
    state is O(d), with O(n*d) worst-case conversion and output space.
    """

    def __init__(self, feature_range: tuple = (0.0, 1.0)):
        """Set the requested scaling interval.

        Parameters
        ----------
        feature_range : pair of numbers, default (0.0, 1.0)
            Two real finite endpoints with low < high and a finite interval width.

        Returns
        -------
        None
            Initialize the scaler without fitting it.

        Raises
        ------
        ValueError
            If endpoints are complex, nonfinite, malformed, unordered or too far apart.

        Notes
        -----
        Time and space: O(1).
        """
        if np.iscomplexobj(feature_range):
            raise ValueError("feature_range must contain real endpoints")
        try:
            bounds = np.asarray(feature_range, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("feature_range must contain two numbers") from exc
        if bounds.shape != (2,) or not np.isfinite(bounds).all():
            raise ValueError("feature_range must contain two finite endpoints")
        low, high = bounds
        with np.errstate(over="ignore"):
            width = high - low
        if low >= high or not np.isfinite(width):
            raise ValueError("feature_range must have finite width and low < high")
        self.feature_range = (low, high)

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Learn column minima and ranges, ignoring NaN.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Nonempty numeric data; NaNs are allowed.

        Returns
        -------
        MinMaxScaler
            The fitted instance (self).

        Raises
        ------
        TypeError
            If X cannot be converted to numbers.
        ValueError
            If X is empty, not two-dimensional or contains infinity.
        FloatingPointError
            If a fitted range exceeds float64. Existing fitted state is kept.

        Notes
        -----
        Time: O(n_samples * n_features). Extra space: O(n_samples * n_features)
        including dtype conversion and NumPy reduction temporaries; stored state O(n_features).
        All-NaN columns retain NaN statistics and may emit NumPy warnings.
        """
        X = check_array(X, ndim=2)

        if np.isinf(X).any():
            raise ValueError("X must not contain infinity; NaN is allowed")
        minimum = np.nanmin(X, axis=0)
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            data_range = np.nanmax(X, axis=0) - minimum
        self.data_min_ = minimum
        self.data_range_ = np.where(data_range == 0, np.inf, data_range)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale columns using the fitted minima and ranges.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Nonempty numeric data; NaNs are allowed.

        Returns
        -------
        ndarray, shape (n_samples, n_features)
            Floating-point transformed data; NaNs remain NaN.

        Raises
        ------
        TypeError
            If X cannot be converted to numbers.
        ValueError
            If X is empty or not two-dimensional.
            Also raised if the feature count differs from fit.
        RuntimeError
            If fit has not been called.
        FloatingPointError
            If the final transformed value exceeds float64.

        Notes
        -----
        Time and space: O(n_samples * n_features), including the output.
        Does not refit. New values may fall outside feature_range. Power-of-two
        scaling repairs overflowing or underflowing intermediate ratios.
        """
        X = check_array(X, ndim=2)
        _check_fitted(self, X)

        low, high = self.feature_range
        return _scale_minmax(X, self.data_min_, self.data_range_, low, high)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on X and return the transformed training data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Nonempty numeric data; NaNs are allowed.

        Returns
        -------
        ndarray, shape (n_samples, n_features)
            Floating-point transformed data; NaNs are preserved.

        Raises
        ------
        TypeError
            If X cannot be converted to numbers.
        ValueError
            If X is empty or not two-dimensional.

        Notes
        -----
        Time and space: O(n_samples * n_features), including the output.
        Equivalent to self.fit(X).transform(X).
        """
        return self.fit(X).transform(X)


class SimpleImputer:
    """Fill missing values using column statistics or a constant.

    Parameters
    ----------
    strategy : {'mean', 'median', 'constant'} or None, default None
        None selects constant when fill_value is supplied, otherwise mean.
    fill_value : number, str or None, default None
        Constant replacement and all-NaN numeric column fallback. None
        becomes 0.0. A string requires constant strategy and selects
        categorical handling.

    Attributes
    ----------
    strategy : str
        Resolved replacement strategy.
    fill_value : float or str
        Resolved constant replacement.
    statistics_ : ndarray, shape (n_features,)
        Replacement value for each fitted column.
    n_features_in_ : int
        Number of fitted columns.

    Raises
    ------
    TypeError
        If numeric values cannot be converted.
    ValueError
        If strategy, fill_value, input shape or feature count is invalid.
    RuntimeError
        If transform is called before fit.

    Notes
    -----
    Numeric mode replaces NaN. Categorical mode converts to strings and
    replaces '', 'nan' and 'None'. Mean/constant fitting and transformation
    take O(n*d) time; median has a conservative O(d*n*log(n)) bound. Working
    space is O(n*d), with O(d) stored statistics, for bounded string lengths.
    """

    def __init__(self, strategy: str = None, fill_value=None):
        """Choose how missing values will be filled.

        Parameters
        ----------
        strategy : {'mean', 'median', 'constant'} or None, default None
            None selects constant when fill_value is supplied, otherwise mean.
        fill_value : number, str or None, default None
            Constant replacement and all-NaN column fallback; None becomes 0.0.
            A string selects categorical handling and requires constant strategy.

        Returns
        -------
        None
            Store configuration; no data is fitted.

        Raises
        ------
        ValueError
            If strategy is unsupported, a string is used with a nonconstant
            strategy, or a nonstring fill_value cannot be parsed as a float.
        TypeError
            If a nonstring fill_value cannot be converted to float.

        Notes
        -----
        Time and space: O(1), excluding conversion of a string-valued parameter.
        """
        if strategy is None:
            strategy = "mean" if fill_value is None else "constant"
        if strategy not in ("mean", "median", "constant"):
            raise ValueError(f"unknown strategy {strategy!r}")
        if isinstance(fill_value, str) and strategy != "constant":
            raise ValueError("a string fill_value requires strategy='constant'")
        self.strategy = strategy
        self.fill_value = 0.0 if fill_value is None else fill_value
        if not isinstance(self.fill_value, str):
            self.fill_value = float(self.fill_value)

    def _as_array(self, X) -> np.ndarray:
        """Convert imputer input to the configured numeric or string representation.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Nonempty data. Numeric mode treats NaN as missing; categorical mode
            converts values to strings and treats '', 'nan' and 'None' as missing.

        Returns
        -------
        ndarray, shape (n_samples, n_features)
            Float data for numeric fill values, otherwise Unicode strings.

        Raises
        ------
        TypeError
            If X cannot be converted to numbers.
        ValueError
            If X is empty or not two-dimensional.

        Notes
        -----
        Time and space: O(n_samples * n_features), assuming bounded string lengths.
        """
        if not isinstance(self.fill_value, str):
            return check_array(X, ndim=2)
        X = np.asarray(X).astype(str)
        if X.ndim != 2 or X.size == 0:
            raise ValueError(f"X must be a non-empty 2-D array, got shape {X.shape}")
        return X

    def fit(self, X: np.ndarray) -> "SimpleImputer":
        """Learn one replacement value per feature.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Nonempty data. Numeric mode treats NaN as missing; categorical mode
            converts values to strings and treats '', 'nan' and 'None' as missing.

        Returns
        -------
        SimpleImputer
            The fitted instance (self), with statistics_ of shape (n_features,).

        Raises
        ------
        TypeError
            If X cannot be converted to numbers.
        ValueError
            If X is empty or not two-dimensional.

        Notes
        -----
        Mean/constant time: O(n_samples * n_features); median has a conservative
        O(n_features * n_samples * log(n_samples)) bound. Space: O(n_samples * n_features).
        All-NaN numeric columns use fill_value instead of a mean or median.
        """
        X = self._as_array(X)
        n_features = X.shape[1]

        if self.strategy == "constant":
            self.statistics_ = np.full(n_features, self.fill_value)
        else:
            all_nan = np.isnan(X).all(axis=0)
            X_safe = np.where(all_nan, 0.0, X)      # nanmean warns on an all-NaN column
            if self.strategy == "mean":
                self.statistics_ = _column_mean(X_safe)
            else:
                self.statistics_ = _column_median(X_safe)
            self.statistics_[all_nan] = self.fill_value

        self.n_features_in_ = n_features
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Replace missing cells using fitted column statistics.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Nonempty data. Numeric mode treats NaN as missing; categorical mode
            converts values to strings and treats '', 'nan' and 'None' as missing.

        Returns
        -------
        ndarray, shape (n_samples, n_features)
            Filled data; numeric mode returns floats and categorical mode strings.

        Raises
        ------
        TypeError
            If X cannot be converted to numbers.
        ValueError
            If X is empty or not two-dimensional.
            Also raised if the feature count differs from fit.
        RuntimeError
            If fit has not been called.

        Notes
        -----
        Time and space: O(n_samples * n_features), assuming bounded string lengths.
        The input is not modified and statistics are not refitted.
        """
        X = self._as_array(X)
        _check_fitted(self, X)
        missing = np.isin(X, MISSING) if X.dtype.kind == "U" else np.isnan(X)
        return np.where(missing, self.statistics_, X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Learn replacement values and fill missing cells in X.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Nonempty data. Numeric mode treats NaN as missing; categorical mode
            converts values to strings and treats '', 'nan' and 'None' as missing.

        Returns
        -------
        ndarray, shape (n_samples, n_features)
            Filled training data.

        Raises
        ------
        TypeError
            If X cannot be converted to numbers.
        ValueError
            If X is empty or not two-dimensional.

        Notes
        -----
        Costs are those of fit plus transform; space O(n_samples * n_features).
        """
        return self.fit(X).transform(X)


class OneHotEncoder:
    """Encode each categorical feature as a dense block of zero/one indicators.

    Parameters
    ----------
    handle_unknown : {'error', 'ignore'}, default 'error'
        Raise for unseen nonmissing categories or encode their block as zeros.

    Attributes
    ----------
    categories_ : list of ndarray
        Sorted nonmissing string categories, one array per input feature.
    n_features_in_ : int
        Number of fitted input columns.
    n_features_out_ : int
        Total number of learned categories and output indicator columns.

    Raises
    ------
    ValueError
        If configuration/input shape is invalid, feature count changes, or
        an unseen category occurs with handle_unknown='error'.
    RuntimeError
        If transform is called before fit.

    Notes
    -----
    Values are converted to strings. '', 'nan' and 'None' are missing and
    always encode to all-zero blocks. Output shape is (n, c), where c is the
    total category count. Fit time is O(d*n*log(n)); transform time and space
    are O(n*d + n*c), including conversion/output, for bounded string lengths.
    """

    MISSING = MISSING

    def __init__(self, handle_unknown: str = "error"):
        """Choose how unseen nonmissing categories are handled.

        Parameters
        ----------
        handle_unknown : {'error', 'ignore'}, default 'error'
            Raise for unseen values, or encode their feature block as zeros.

        Returns
        -------
        None
            Initialize an unfitted encoder.

        Raises
        ------
        ValueError
            If handle_unknown is unsupported.

        Notes
        -----
        Time and space: O(1).
        """
        if handle_unknown not in ("error", "ignore"):
            raise ValueError(f"unknown handle_unknown {handle_unknown!r}")
        self.handle_unknown = handle_unknown

    def fit(self, X: np.ndarray) -> "OneHotEncoder":
        """Learn sorted nonmissing categories separately for each input column.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Categories converted to strings. '', 'nan' and 'None' are missing.

        Returns
        -------
        OneHotEncoder
            The fitted instance (self). categories_ contains one array per feature;
            n_features_out_ is the total number of learned categories.

        Raises
        ------
        ValueError
            If X is empty or not two-dimensional.
        TypeError
            If X cannot be converted to a string array.

        Notes
        -----
        Time: O(d * n * log(n)); space O(n * d + c), where n is sample count,
        d feature count and c total categories, assuming bounded string lengths.
        """
        X = np.asarray(X).astype(str)
        if X.ndim != 2 or X.size == 0:
            raise ValueError(f"X must be a non-empty 2-D array, got shape {X.shape}")

        self.categories_ = []
        for column in X.T:
            present = ~np.isin(column, self.MISSING)
            self.categories_.append(np.unique(column[present]))

        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = sum(len(categories) for categories in self.categories_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode data using fitted category blocks.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Categories converted to strings. '', 'nan' and 'None' are missing.

        Returns
        -------
        ndarray of float, shape (n_samples, n_features_out_)
            Dense zero/one indicators. Missing cells produce an all-zero block;
            unseen values do so only when handle_unknown='ignore'.

        Raises
        ------
        RuntimeError
            If fit has not been called.
        ValueError
            If X is not 2-D, has a different feature count, or an unseen category
            occurs with handle_unknown='error'.
        TypeError
            If X cannot be converted to strings.

        Notes
        -----
        Time and space: O(n * d + n * c), including conversion and output, where
        c is the total category count; assume bounded string lengths.
        """
        X = np.asarray(X).astype(str)
        _check_fitted(self, X)

        out = np.zeros((X.shape[0], self.n_features_out_))
        start = 0
        for j, categories in enumerate(self.categories_):
            column = X[:, j]
            block = column[:, None] == categories          # (n, c) by broadcasting
            present = ~np.isin(column, self.MISSING)
            unknown = present & ~block.any(axis=1)
            if unknown.any() and self.handle_unknown == "error":
                raise ValueError(f"unknown categories in column {j}: {np.unique(column[unknown]).tolist()}")

            out[:, start:start + len(categories)] = block
            start += len(categories)
        return out

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Learn categories and encode the same data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Categories converted to strings. '', 'nan' and 'None' are missing.

        Returns
        -------
        ndarray of float, shape (n_samples, n_features_out_)
            Dense training indicators; missing cells produce all-zero blocks.

        Raises
        ------
        ValueError
            If X is empty or not two-dimensional.
        TypeError
            If X cannot be converted to strings.

        Notes
        -----
        Time: O(d * n * log(n) + n * c); space O(n * d + n * c + c),
        where c is the total category count; assume bounded string lengths.
        """
        return self.fit(X).transform(X)


if __name__ == "__main__":
    X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])

    Z = StandardScaler().fit_transform(X)
    assert np.allclose(np.nanmean(Z, axis=0), 0.0)
    assert np.isnan(Z[1, 1])

    Z = MinMaxScaler((-1, 1)).fit_transform(X)
    assert np.allclose(Z[:, 0], [-1, 0, 1])

    Z = SimpleImputer("median").fit_transform(X)
    assert not np.isnan(Z).any()

    assert np.allclose(SimpleImputer(fill_value=0).fit(X).statistics_, 0.0)
    cities = SimpleImputer(fill_value="Unknown").fit_transform([["Perth"], ["nan"], [""]])
    assert cities[:, 0].tolist() == ["Perth", "Unknown", "Unknown"]

    colours = np.array([["red", "S"], ["blue", "M"], ["red", "M"]])
    Z = OneHotEncoder().fit_transform(colours)
    assert Z.shape == (3, 4) and Z.sum() == 6

    print("preprocessing ok")
