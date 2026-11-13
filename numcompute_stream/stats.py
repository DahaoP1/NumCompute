"""Batch helpers plus per-feature statistics updated from streamed chunks.

The batch functions are reused from numcompute.stats. RunningStats adds state:
update/partial_fit/update_stats merge chunks, and queries read accumulated data.
Exact quantiles and histograms retain history; moments-only mode uses O(d) state.
"""

from numbers import Integral

import numpy as np

from numcompute.stats import mean, median, std, min, max, quantile, histogram

__all__ = [
    "mean", "median", "std", "min", "max", "quantile", "histogram",
    "RunningStats", "update_stats",
]


class RunningStats:
    """Maintain streaming moments and exact distribution summaries.

    Parameters
    ----------
    skip_nan : bool, default True
        Ignore NaN separately in each feature. False propagates NaN into that
        feature's mean, variance, minimum, maximum and quantiles. Histograms
        always omit NaN. Neither mode drops entire rows.
    store_data : bool, default True
        Keep copies of incoming chunks for exact quantiles/median/histograms.
        False retains only moments; distribution queries then raise RuntimeError.

    Attributes
    ----------
    n : int
        Total number of rows received, including rows containing NaN.
    n_features_in_ : int or None
        Feature count learned from the first nonempty chunk; None before that.
    count_ : ndarray of int, shape (d,)
        Number of non-NaN observations per feature, returned as a copy.
    mean, m2, min, max : ndarray of float, shape (d,)
        Per-feature mean, sum of squared deviations, minimum and maximum.
        Properties return copies. Unobserved features return NaN.

    Raises
    ------
    TypeError
        If options or data have unsupported types.
    ValueError
        If shapes, feature counts or query options are invalid, or data is infinite.
    RuntimeError
        If queried before any rows arrive or a distribution query needs disabled
        history storage.
    FloatingPointError
        If moments overflow float64; rescale the data. Failed updates keep state.

    Notes
    -----
    A scalar is one row/feature; (b,) means b rows of one feature; (b,d) means
    b rows and d features. Moments always reduce axis 0, returning (d,).
    Empty chunks do nothing; an empty (0,d) chunk still checks a learned schema.
    A new stream requires reset(). Updates use the Chan/Welford merge formula,
    vectorised over rows and features, without Python element-wise loops.

    Updating b rows takes O(b*d) time and scratch space. Stored moments take
    O(d) space. With store_data=True, history adds O(N*d) for N rows so far.
    Quantiles and histograms are recomputed from that history when queried;
    exact distribution summaries are not constant-memory approximations.
    """

    def __init__(self, skip_nan=True, store_data=True):
        """Configure missing-value handling and optional history storage.

        Parameters
        ----------
        skip_nan, store_data : bool, default True
            Options described in the class docstring.

        Returns
        -------
        None
            Initialise an empty accumulator.

        Raises
        ------
        TypeError
            If either option is not boolean.

        Notes
        -----
        Time and space: O(1).
        """
        if not isinstance(skip_nan, (bool, np.bool_)) or not isinstance(store_data, (bool, np.bool_)):
            raise TypeError("skip_nan and store_data must be boolean")
        self.skip_nan = bool(skip_nan)
        self.store_data = bool(store_data)
        self.reset()

    def reset(self):
        """Clear accumulated data while preserving constructor options.

        Parameters
        ----------
        None

        Returns
        -------
        RunningStats
            self, ready to accept a new feature count.

        Notes
        -----
        Releasing stored arrays takes O(k) references for k chunks; new state
        uses O(1) space. Querying before the next nonempty update raises RuntimeError.
        """
        self.n = 0
        self.n_features_in_ = None
        self._count = self._mean = self._m2 = None
        self._minimum = self._maximum = self._has_nan = None
        self._chunks = []
        return self

    def _check_chunk(self, X):
        """Convert scalar/(b,)/(b,d) real data to float (b,d); O(b*d) time/space."""
        X = np.asarray(X)
        if X.dtype.kind not in "biuf":
            raise TypeError("X_chunk must contain real numeric values")
        if X.ndim == 0:
            X = X.reshape(1, 1)
        elif X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X_chunk must be a scalar, (samples,), or (samples, features) with at least one feature")
        with np.errstate(over="ignore"):
            X = X.astype(float, copy=False)
        if np.isinf(X).any():
            raise ValueError("X_chunk must not contain infinity; NaN is allowed")
        if self.n_features_in_ is not None and X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        return X

    def update(self, X_chunk):
        """Merge a chunk into the accumulated per-feature statistics.

        Parameters
        ----------
        X_chunk : real numeric scalar or array_like, shape (b,) or (b,d)
            New rows. NaNs follow skip_nan; infinities are rejected. A 1-D
            array means one feature, not one row with several features.

        Returns
        -------
        RunningStats
            self. Inputs are not modified; saved history uses a private copy.

        Raises
        ------
        TypeError
            If data is nonnumeric or complex.
        ValueError
            If dimensionality/feature count is invalid or infinity is present.
        FloatingPointError
            If float64 moments overflow. State and history remain unchanged.

        Notes
        -----
        Time and temporary space: O(b*d). Persistent state: O(d), plus O(N*d)
        history if store_data=True. Empty chunks leave state unchanged. A column
        with no observations can acquire valid statistics in later chunks.
        """
        X = self._check_chunk(X_chunk)
        if len(X) == 0:
            return self
        valid = ~np.isnan(X)
        batch_count = valid.sum(axis=0)
        d = X.shape[1]
        old_count = np.zeros(d, dtype=np.int64) if self.n == 0 else self._count
        old_mean = np.zeros(d) if self.n == 0 else self._mean
        old_m2 = np.zeros(d) if self.n == 0 else self._m2
        old_min = np.full(d, np.inf) if self.n == 0 else self._minimum
        old_max = np.full(d, -np.inf) if self.n == 0 else self._maximum
        old_nan = np.zeros(d, dtype=bool) if self.n == 0 else self._has_nan

        # Work around a finite reference to preserve small differences near a
        # large offset. All-NaN columns use zero internally until data arrives.
        reference = np.where(batch_count > 0, X[np.argmax(valid, axis=0), np.arange(d)], 0.0)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                centred = np.where(valid, X, reference) - reference
                batch_mean = reference + np.divide(
                    centred.sum(axis=0), batch_count, out=np.zeros(d), where=batch_count > 0,
                )
                centred = np.where(valid, X, batch_mean) - batch_mean
                batch_m2 = np.sum(centred ** 2, axis=0)
                count = old_count + batch_count
                ratio = np.divide(batch_count, count, out=np.zeros(d), where=count > 0)
                both = (old_count > 0) & (batch_count > 0)
                delta = np.subtract(batch_mean, old_mean, out=np.zeros(d), where=both)
                new_mean = np.where(old_count == 0, batch_mean, old_mean + delta * ratio)
                correction = (delta * np.sqrt(old_count * ratio)) ** 2
                new_m2 = old_m2 + batch_m2 + correction
        except FloatingPointError as exc:
            raise FloatingPointError("streaming moments exceed float64 range; rescale the data") from exc

        minimum = np.minimum(old_min, np.min(np.where(valid, X, np.inf), axis=0))
        maximum = np.maximum(old_max, np.max(np.where(valid, X, -np.inf), axis=0))
        saved = X.copy() if self.store_data else None
        self._count, self._mean, self._m2 = count, new_mean, new_m2
        self._minimum, self._maximum = minimum, maximum
        self._has_nan = old_nan | np.any(~valid, axis=0)
        self.n += len(X)
        self.n_features_in_ = d
        if self.store_data:
            self._chunks.append(saved)
        return self

    # The same state update is available under the assignment's API names.
    partial_fit = update
    update_stats = update

    def _require_data(self):
        """Raise RuntimeError before the first nonempty chunk; O(1)."""
        if self.n == 0:
            raise RuntimeError("no data yet; call update() first")

    def _visible(self, values):
        """Copy (d,) values and apply missing-value policy; O(d) time/space."""
        self._require_data()
        missing = self._count == 0
        if not self.skip_nan:
            missing = missing | self._has_nan
        return np.where(missing, np.nan, values)

    @property
    def count_(self):
        """Non-NaN counts (d,), copied in O(d); raise RuntimeError before data."""
        self._require_data()
        return self._count.copy()

    @property
    def mean(self):
        """Running means (d,), copied in O(d); raise RuntimeError before data."""
        self._require_data()
        return self._visible(self._mean)

    @property
    def m2(self):
        """Squared-deviation sums (d,), copied in O(d); apply the NaN policy."""
        self._require_data()
        return self._visible(self._m2)

    @property
    def min(self):
        """Running minima (d,), copied in O(d); raise RuntimeError before data."""
        self._require_data()
        return self._visible(self._minimum)

    @property
    def max(self):
        """Running maxima (d,), copied in O(d); raise RuntimeError before data."""
        self._require_data()
        return self._visible(self._maximum)

    def var(self, ddof=0):
        """Return population or sample variance for each feature.

        Parameters
        ----------
        ddof : nonnegative int, default 0
            Divide squared deviations by count_ - ddof. Use 1 for sample variance.

        Returns
        -------
        ndarray, shape (d,)
            Variances. Insufficient observations (count_ <= ddof) give NaN
            without warnings. Other NaNs follow skip_nan.

        Raises
        ------
        RuntimeError
            If no rows have arrived.
        TypeError
            If ddof is not an integer or is boolean.
        ValueError
            If ddof is negative.

        Notes
        -----
        Time and output space: O(d). Stored state is not modified.
        """
        self._require_data()
        if isinstance(ddof, (bool, np.bool_)) or not isinstance(ddof, Integral):
            raise TypeError("ddof must be a nonnegative integer")
        if ddof < 0:
            raise ValueError("ddof must be nonnegative")
        denominator = self._count.astype(float) - ddof
        result = np.divide(self._m2, denominator, out=np.full(self.n_features_in_, np.nan), where=denominator > 0)
        return self._visible(result)

    def std(self, ddof=0):
        """Return the square root of running variances.

        Parameters
        ----------
        ddof : nonnegative int, default 0
            Population (0) or sample (1) correction, as in var.

        Returns
        -------
        ndarray, shape (d,)
            Standard deviations; insufficient counts or propagated NaNs give NaN.

        Raises
        ------
        RuntimeError
            If no rows have arrived.
        TypeError, ValueError
            If ddof is invalid, as in var.

        Notes
        -----
        Time and output space: O(d).
        """
        return np.sqrt(self.var(ddof))

    def _data(self):
        """Copy retained history (N,d); O(N*d) time/space; require stored data."""
        self._require_data()
        if not self.store_data:
            raise RuntimeError("exact quantiles and histograms require store_data=True")
        return np.concatenate(self._chunks, axis=0)

    def quantile(self, q, axis=0, method="linear"):
        """Return exact quantiles over all received chunks.

        Parameters
        ----------
        q : float or 1-D array_like
            Probability/probabilities in [0,1].
        axis : {0, None}, default 0
            0 reduces rows for each feature; None flattens the full history.
        method : {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
            NumPy interpolation method, default 'linear'.

        Returns
        -------
        ndarray or float
            axis=0 gives (d,) for scalar q or (len(q),d) for vector q.
            axis=None gives a float for scalar q or (len(q),) otherwise.
            All-NaN groups give NaN without warnings. NaNs follow skip_nan.

        Raises
        ------
        RuntimeError
            If no rows have arrived or store_data=False.
        TypeError, ValueError
            If q is invalid, axis is not 0/None, or method is unsupported.
        FloatingPointError
            If NumPy interpolation exceeds float64 range; rescale the data.

        Notes
        -----
        Conservative time O(N*d*log(N*d) + output_size); extra space O(N*d +
        output_size). Exact results require the O(N*d) retained history.
        """
        if np.iscomplexobj(q):
            raise TypeError("q must contain real probabilities")
        q = np.asarray(q, dtype=float)
        if q.ndim > 1 or q.size == 0 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
            raise ValueError("q must be a scalar or nonempty 1-D array in [0, 1]")
        if axis is not None and (isinstance(axis, (bool, np.bool_)) or not isinstance(axis, Integral) or axis != 0):
            raise ValueError("axis must be 0 or None")
        if method not in ("linear", "lower", "higher", "midpoint", "nearest"):
            raise ValueError("unknown quantile method")
        data = self._data()
        function = np.nanquantile if self.skip_nan else np.quantile
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            if axis is None:
                values = data.ravel()
                if self.skip_nan:
                    values = values[~np.isnan(values)]
                result = function(values, q, method=method) if values.size else np.full(q.shape, np.nan)
                return float(result) if q.ndim == 0 else result
            result = np.full(q.shape + (self.n_features_in_,), np.nan)
            observed = self._count > 0
            if observed.any():
                result[..., observed] = function(data[:, observed], q, axis=0, method=method)
        return result

    def median(self, axis=0):
        """Return exact medians over retained chunks via quantile(0.5).

        Parameters
        ----------
        axis : {0, None}, default 0
            Per-feature (0) or flattened (None), as in quantile.

        Returns
        -------
        ndarray, shape (d,), or float
            Current medians; NaNs follow skip_nan.

        Raises
        ------
        RuntimeError
            If no rows have arrived or history storage is disabled.
        ValueError
            If axis is unsupported.
        FloatingPointError
            If interpolation exceeds float64 range.

        Notes
        -----
        Conservative time O(N*d*log(N*d)); extra space O(N*d).
        """
        return self.quantile(0.5, axis=axis)

    def histogram(self, bins=10, range=None, density=False, feature=None):
        """Bin all retained observations or one selected feature.

        Parameters
        ----------
        bins : positive int or 1-D array_like, default 10
            Number of equal-width bins or finite, strictly increasing edges.
            Integer-bin edges are recomputed from the full history at each query.
        range : pair of finite floats or None, default None
            Automatic-edge limits with low < high; ignored for explicit edges.
            Values outside the bin edges are not counted.
        density : bool, default False
            Return densities instead of counts; zero included values give zeros.
        feature : int or None, default None
            Column index in [0,d). None flattens all feature values.

        Returns
        -------
        tuple of (ndarray, ndarray)
            Counts/densities (b,) and edges (b+1,). NaNs are always omitted.
            The last bin includes its right endpoint. With only NaNs, integer
            bins and no range use [0,1] edges and zero counts.

        Raises
        ------
        RuntimeError
            If no rows have arrived or store_data=False.
        TypeError, ValueError
            If feature, bins, range or density is invalid, or automatic edges
            cannot be represented as distinct finite float64 values.
        FloatingPointError
            If histogram arithmetic exceeds float64 range.

        Notes
        -----
        Conservative time O(N*d + M*log(b) + b) and space O(N*d + b), for M
        selected finite values and b bins. Queries recompute from exact history;
        counts from independently chosen per-chunk edges are never added together.
        """
        data = self._data()
        if not isinstance(density, (bool, np.bool_)):
            raise TypeError("density must be boolean")
        if feature is not None:
            if isinstance(feature, (bool, np.bool_)) or not isinstance(feature, Integral):
                raise TypeError("feature must be an integer or None")
            if feature < 0 or feature >= self.n_features_in_:
                raise ValueError("feature is outside the learned feature count")
            data = data[:, feature]
        values = data.ravel()
        values = values[~np.isnan(values)]
        if isinstance(bins, (bool, np.bool_)):
            raise TypeError("bins must be a positive integer or bin edges")
        if isinstance(bins, Integral):
            if bins < 1:
                raise ValueError("bins must be positive")
        else:
            if np.iscomplexobj(bins):
                raise TypeError("bin edges must be real")
            bins = np.asarray(bins, dtype=float)
            if bins.ndim != 1 or bins.size < 2 or not np.isfinite(bins).all() or not np.all(bins[1:] > bins[:-1]):
                raise ValueError("bin edges must be finite and strictly increasing")
        if range is not None:
            if np.iscomplexobj(range):
                raise TypeError("range must contain real limits")
            limits = np.asarray(range, dtype=float)
            if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError("range must contain two finite limits: low < high")
            range = tuple(limits)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            edges = np.histogram_bin_edges(values, bins=bins, range=range)
            if not np.isfinite(edges).all() or not np.all(edges[1:] > edges[:-1]):
                raise ValueError("histogram needs distinct finite edges; supply explicit edges or rescale the data")
            counts, edges = np.histogram(values, bins=edges)
            if density:
                total = counts.sum()
                counts = (counts / total) / np.diff(edges) if total else np.zeros(counts.shape, dtype=float)
        return counts, edges


def update_stats(X_chunk, stats=None):
    """Create or update a caller-owned streaming statistics object.

    Parameters
    ----------
    X_chunk : real numeric scalar or array_like, shape (b,) or (b,d)
        New rows, interpreted as in RunningStats.update.
    stats : RunningStats or None, default None
        Existing accumulator; None creates RunningStats() with default options.

    Returns
    -------
    RunningStats
        Updated accumulator. Keep and pass it to later calls; no module-global
        state is used. Example: state = update_stats(chunk2, state).

    Raises
    ------
    TypeError
        If stats is not a RunningStats instance or data is nonnumeric.
    ValueError, FloatingPointError
        As raised by RunningStats.update for invalid or overflowing data.

    Notes
    -----
    Time and added space: O(b*d), with accumulated-state costs described in
    RunningStats. Calling repeatedly without stats creates independent streams.
    """
    if stats is None:
        stats = RunningStats()
    if not isinstance(stats, RunningStats):
        raise TypeError("stats must be a RunningStats instance or None")
    return stats.update(X_chunk)
