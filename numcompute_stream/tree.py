"""A NumPy classification tree with chunk buffering and periodic retraining.

Split scores are calculated for all thresholds of a feature at once. Loops
visit features and tree nodes, not individual thresholds or prediction rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

__all__ = ["DecisionTreeClassifier"]


@dataclass
class _Node:
    """A leaf prediction, optionally extended with a split and two children.

    prediction is a zero-based index into the fitted classes_. A leaf has
    feature=None; an internal node sends values <= threshold to the left.
    nan_left records the direction learned for missing feature values.
    Each node stores O(1) state, excluding its child nodes.
    """

    prediction: int
    feature: int | None = None
    threshold: float = 0.0
    nan_left: bool = False
    left: _Node | None = None
    right: _Node | None = None


class DecisionTreeClassifier:
    """Train a single decision tree, rebuilding it when its buffer is full.

    Parameters
    ----------
    max_depth : int or None, default 5
        Maximum number of splits along a path. Zero gives a majority-class
        leaf; None allows growth until another stopping rule applies.
    min_samples_split : int, default 2
        Minimum number of samples required to attempt a split; at least 2.
    max_features : int or None, default None
        Number of randomly selected features considered at each node. None
        considers all features. An integer must be between 1 and X.shape[1].
    buffer_size : int, default 500
        Retrain when at least this many samples are buffered. The entire
        buffer is used once, even if it exceeds this size, then cleared.
    criterion : {'gini', 'entropy'}, default 'gini'
        Impurity measure used to choose splits.
    random_state : nonnegative int or None, default 0
        Seed for feature selection. A fixed seed makes each training run
        reproducible; None requests a fresh random seed.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Sorted labels from the most recent training batch, available after
        training. Labels absent from that batch are not retained as classes.
    root_ : _Node
        Most recently trained tree, available after training.
    n_features_in_ : int
        Feature count fixed by fit or the first nonempty partial_fit chunk.
    n_buffered_samples_ : int
        Number of samples waiting for the next buffered retraining.
    n_retrains_ : int
        Number of buffer-triggered retrainings since construction or fit.
    n_training_samples_ : int
        Samples used to build the most recent tree, available after training.

    Raises
    ------
    TypeError
        If parameter types, feature values or label types are unsupported.
    ValueError
        If parameter values, input shapes, labels or feature counts are invalid.
    RuntimeError
        If predict is called before fit or the first buffered retraining.

    Notes
    -----
    Features are real numeric values converted to float64. NaNs are allowed;
    infinity is rejected. Labels are finite numbers, booleans or nonblank
    strings. Numeric and string label families cannot be mixed across chunks.
    NaNs are sent together to the child giving the best split gain. Ties
    within 1e-12 prefer the lowest feature index, then lowest threshold, then
    NaNs on the right. Majority ties use the first sorted class. Gains <=1e-12
    do not split. Prediction follows the learned missing-value direction.

    At a node with m samples, f candidate features and c classes, split search
    takes O(f*(m*log(m) + m*c)) time and O(m*c + m) scratch space. Total fitting
    time is the sum over nodes. With n samples, d features, depth D and T nodes,
    recursive fitting uses O((D+1)*n*d + n*c + T) worst-case working space.
    The fitted tree stores O(T+c) state; buffered data additionally uses O(B*d)
    for B waiting samples. Large class counts increase split-search memory.
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_features=None,
                 buffer_size=500, criterion="gini", random_state=0):
        """Validate settings and initialise an empty buffer.

        Parameters
        ----------
        max_depth, min_samples_split, max_features, buffer_size, criterion, random_state
            Settings described in the class docstring; no data is fitted here.

        Returns
        -------
        None
            Create the classifier with empty buffers and zero retrainings.

        Raises
        ------
        TypeError
            If an integer setting is not an integer or is boolean.
        ValueError
            If a setting is out of range or criterion is unsupported.

        Notes
        -----
        Time and space: O(1). max_features is checked against the feature
        count when data is supplied.
        """
        self.max_depth = self._integer(max_depth, "max_depth", 0, allow_none=True)
        self.min_samples_split = self._integer(min_samples_split, "min_samples_split", 2)
        self.max_features = self._integer(max_features, "max_features", 1, allow_none=True)
        self.buffer_size = self._integer(buffer_size, "buffer_size", 1)
        self.random_state = self._integer(random_state, "random_state", 0, allow_none=True)
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        self._X_buffer = []
        self._y_buffer = []
        self.n_buffered_samples_ = 0
        self.n_retrains_ = 0

    @staticmethod
    def _integer(value, name, minimum, allow_none=False):
        """Check one integer setting in O(1) time/space; raise on bad values."""
        if allow_none and value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise TypeError(f"{name} must be an integer" + (" or None" if allow_none else ""))
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        return int(value)

    def _check_X(self, X, expected_features=None):
        """Validate numeric (n,d) X, allowing n=0 and NaN; O(n*d) time/space."""
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("X must be a 2-D array (samples, features) with at least one feature")
        if X.dtype.kind not in "biuf":
            raise TypeError("X must contain real numeric values; encode categorical features first")
        X = X.astype(float, copy=False)
        if np.isinf(X).any():
            raise ValueError("X must not contain infinity; NaN is allowed")
        if expected_features is not None and X.shape[1] != expected_features:
            raise ValueError(f"expected {expected_features} features, got {X.shape[1]}")
        if self.max_features is not None and self.max_features > X.shape[1]:
            raise ValueError("max_features must not exceed the number of input features")
        return X

    @staticmethod
    def _check_y(y, n_samples):
        """Validate (n,) numeric/string labels; O(n) time/space, no missing labels."""
        y = np.asarray(y)
        if y.ndim != 1 or y.shape[0] != n_samples:
            raise ValueError("y must be one-dimensional with the same number of samples as X")
        if y.dtype.kind not in "biufUS":
            raise TypeError("y must contain numeric, boolean or string labels")
        if y.dtype.kind == "S":
            y = y.astype(str)
        family = "text" if y.dtype.kind in "US" else "numeric"
        if family == "numeric" and not np.isfinite(y).all():
            raise ValueError("y must not contain NaN or infinity")
        if family == "text" and np.any(np.char.strip(y) == ""):
            raise ValueError("y must not contain blank labels")
        return y, family

    def _impurity(self, counts):
        """Compute Gini/entropy along the last count axis; O(counts.size) time/space."""
        totals = counts.sum(axis=-1, keepdims=True)
        probabilities = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0)
        if self.criterion == "gini":
            values = 1.0 - np.sum(probabilities ** 2, axis=-1)
            return np.where(totals[..., 0] == 0, 0.0, np.maximum(values, 0.0))
        logs = np.zeros_like(probabilities)
        np.log2(probabilities, out=logs, where=probabilities > 0)
        return -np.sum(probabilities * logs, axis=-1)

    def _best_split(self, X, y, n_classes, parent_impurity, rng):
        """Find (feature, threshold, nan_left), or None when no gain exceeds 1e-12.

        X is (m,d), y contains encoded class indices (m,). For f considered
        features and c classes: O(f*(m*log(m)+m*c)) time, O(m*c+m) scratch space.
        Threshold scores and both NaN directions are vectorised; only features
        are visited by a Python loop.
        """
        features = np.arange(X.shape[1])
        if self.max_features is not None and self.max_features < features.size:
            features = np.sort(rng.choice(features, self.max_features, replace=False))
        best = None
        best_gain = 0.0
        for feature in features:
            values = X[:, feature]
            known = ~np.isnan(values)
            if not known.any():
                continue
            order = np.argsort(values[known], kind="stable")
            ordered_values = values[known][order]
            ordered_y = y[known][order]
            positions = np.flatnonzero(ordered_values[:-1] < ordered_values[1:])
            missing_counts = np.bincount(y[~known], minlength=n_classes)
            if missing_counts.sum():
                # Also allow all known values on one side and NaNs on the other.
                positions = np.r_[positions, ordered_values.size - 1]
            if positions.size == 0:
                continue

            indicators = np.zeros((len(ordered_y), n_classes), dtype=np.int64)
            indicators[np.arange(len(ordered_y)), ordered_y] = 1
            cumulative = np.cumsum(indicators, axis=0)
            left_known = cumulative[positions]
            right_known = cumulative[-1] - left_known
            # Axis 1 chooses NaNs right (0) or left (1).
            left = np.stack((left_known, left_known + missing_counts), axis=1)
            right = np.stack((right_known + missing_counts, right_known), axis=1)
            n_left, n_right = left.sum(axis=-1), right.sum(axis=-1)
            gains = parent_impurity - (n_left * self._impurity(left) + n_right * self._impurity(right)) / len(y)
            gains = np.where((n_left > 0) & (n_right > 0), gains, -np.inf)
            gain = float(gains.max())
            if gain <= best_gain + 1e-12:
                continue
            # First near-equal maximum: lower threshold, then NaNs on the right.
            candidate = np.flatnonzero(gains >= gain - 1e-12)[0]
            cut, nan_direction = divmod(int(candidate), 2)
            position = positions[cut]
            low = ordered_values[position]
            threshold = low
            if position + 1 < ordered_values.size:
                high = ordered_values[position + 1]
                with np.errstate(under="ignore"):
                    midpoint = low / 2.0 + high / 2.0
                # Halving avoids overflow; the fallback handles adjacent floats.
                if low <= midpoint < high:
                    threshold = midpoint
            best = (int(feature), float(threshold), bool(nan_direction))
            best_gain = gain
        return best

    def _build(self, X, y, depth, n_classes, rng):
        """Recursively build a node; costs sum split searches and child-array copies."""
        counts = np.bincount(y, minlength=n_classes)
        node = _Node(prediction=int(np.argmax(counts)))
        if ((self.max_depth is not None and depth >= self.max_depth)
                or len(y) < self.min_samples_split or np.count_nonzero(counts) == 1):
            return node
        split = self._best_split(X, y, n_classes, float(self._impurity(counts)), rng)
        if split is None:
            return node
        node.feature, node.threshold, node.nan_left = split
        values = X[:, node.feature]
        mask = np.where(np.isnan(values), node.nan_left, values <= node.threshold)
        node.left = self._build(X[mask], y[mask], depth + 1, n_classes, rng)
        node.right = self._build(X[~mask], y[~mask], depth + 1, n_classes, rng)
        return node

    def _train(self, X, y, family):
        """Build locally, then replace fitted state; complexity is that of fit."""
        classes, encoded = np.unique(y, return_inverse=True)
        root = self._build(X, encoded, 0, classes.size, np.random.default_rng(self.random_state))
        self.classes_ = classes
        self.root_ = root
        self.n_features_in_ = X.shape[1]
        self._label_family = family
        self.n_training_samples_ = len(y)

    def fit(self, X, y):
        """Build a fresh tree and discard any previously buffered samples.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Nonempty real numeric features. NaNs are allowed, infinity is not.
        y : array_like, shape (n_samples,)
            Finite numeric/boolean or nonblank string labels; no missing labels.

        Returns
        -------
        DecisionTreeClassifier
            The same instance (self), with a new tree, empty buffer and
            n_retrains_=0. A later fit may change feature count and class labels.

        Raises
        ------
        TypeError
            If features or labels have unsupported types.
        ValueError
            If inputs are empty, have incompatible shapes, contain forbidden
            values or supply fewer features than max_features.

        Notes
        -----
        Inputs are not modified. Fitting time is the sum of node split-search
        costs; working space is described in the class docstring. A failed
        validation/build preserves the previous fitted tree and buffer.
        """
        X = self._check_X(X)
        y, family = self._check_y(y, len(X))
        if len(y) == 0:
            raise ValueError("fit requires at least one sample")
        self._train(X, y, family)
        self._X_buffer.clear()
        self._y_buffer.clear()
        self.n_buffered_samples_ = 0
        self.n_retrains_ = 0
        return self

    def partial_fit(self, X_chunk, y_chunk):
        """Append a chunk; retrain once on the entire buffer when it is full.

        Parameters
        ----------
        X_chunk : array_like, shape (n_samples, n_features)
            New real numeric rows, with NaN allowed. Feature count must remain
            unchanged across chunks. A correctly shaped empty chunk is a no-op.
        y_chunk : array_like, shape (n_samples,)
            Matching finite numeric/boolean or nonblank string labels. New
            classes are allowed within the existing numeric or text family.

        Returns
        -------
        DecisionTreeClassifier
            The same instance (self). Each triggered retraining increments
            n_retrains_ and sets n_buffered_samples_ to zero.

        Raises
        ------
        TypeError
            If features/labels have unsupported types or label families change.
        ValueError
            If shapes, feature count, max_features or values are invalid.

        Notes
        -----
        Below the threshold, time and added storage are O(b*d) for b new rows;
        copied chunks protect the buffer from later changes to caller arrays.
        Retraining adds O(B*d) concatenation work and fit costs for all B buffered
        rows. An oversized chunk triggers one retraining on all buffered rows.
        Earlier training rows are not retained. A final incomplete buffer stays
        pending until later chunks arrive; it does not silently trigger training.
        Before the first retraining, predict raises RuntimeError. Invalid chunks
        do not change the current tree or buffer.
        """
        expected = getattr(self, "n_features_in_", None)
        X = self._check_X(X_chunk, expected)
        y, family = self._check_y(y_chunk, len(X))
        if len(y) == 0:
            return self
        if hasattr(self, "_label_family") and family != self._label_family:
            raise TypeError("y_chunk must keep the same numeric or string label family")
        X, y = X.copy(), y.copy()
        n_buffered = self.n_buffered_samples_ + len(y)
        if n_buffered >= self.buffer_size:
            buffered_X = np.concatenate(self._X_buffer + [X], axis=0)
            buffered_y = np.concatenate(self._y_buffer + [y], axis=0)
            self._train(buffered_X, buffered_y, family)
            self._X_buffer.clear()
            self._y_buffer.clear()
            self.n_buffered_samples_ = 0
            self.n_retrains_ += 1
        else:
            self._X_buffer.append(X)
            self._y_buffer.append(y)
            self.n_buffered_samples_ = n_buffered
            self.n_features_in_ = X.shape[1]
            self._label_family = family
        return self

    def _predict(self, node, X, rows, output):
        """Route row indices in batches and fill output; O(n*D) total traversal."""
        if rows.size == 0:
            return
        if node.feature is None:
            output[rows] = node.prediction
            return
        values = X[rows, node.feature]
        mask = np.where(np.isnan(values), node.nan_left, values <= node.threshold)
        self._predict(node.left, X, rows[mask], output)
        self._predict(node.right, X, rows[~mask], output)

    def predict(self, X):
        """Predict labels using the most recently trained tree.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Real numeric features with the fitted feature count. NaNs follow
            the learned branch direction. Empty input of shape (0,d) is allowed.

        Returns
        -------
        ndarray, shape (n_samples,)
            Predicted labels, using the dtype of classes_. Prediction never
            updates the tree, preprocessing state or pending buffer.

        Raises
        ------
        RuntimeError
            If no tree has been trained yet, including a partly filled buffer.
        TypeError
            If X contains nonnumeric or complex feature values.
        ValueError
            If X has an invalid shape/feature count or contains infinity.

        Notes
        -----
        Time: O(n*d + n*D), including input validation, for n samples, d features
        and depth D. Extra space: O(n*d + n*D) worst case for numeric conversion,
        output and recursive row indices; the tree itself is reused.
        """
        if not hasattr(self, "root_"):
            raise RuntimeError("Call fit or supply enough partial_fit samples to reach buffer_size before predict")
        X = self._check_X(X, self.n_features_in_)
        output = np.empty(len(X), dtype=np.intp)
        self._predict(self.root_, X, np.arange(len(X)), output)
        return self.classes_[output]
