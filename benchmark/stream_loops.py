"""Plain-Python streaming baselines, following benchmark/loops.py.

Rows arrive as prebuilt lists. NumPy is used for output arrays and the shared
tree scaffolding; scalar computations below use Python loops.
"""
import math
from bisect import bisect_right
import numpy as np
from benchmark.loops import auc_loop as _batch_auc_loop
from numcompute_stream.tree import DecisionTreeClassifier


def running_stats_loop(chunks):
    """Per-cell Welford; return count/mean/population variance, shape (3,d)."""
    d = len(chunks[0][0])
    count, mean, m2 = [0]*d, [0.0]*d, [0.0]*d
    for chunk in chunks:
        for row in chunk:
            for j, value in enumerate(row):
                if math.isnan(value):
                    continue
                count[j] += 1
                delta = value-mean[j]
                mean[j] += delta/count[j]
                m2[j] += delta*(value-mean[j])
    mean = [value if count[j] else math.nan for j, value in enumerate(mean)]
    var = [m2[j]/count[j] if count[j] else math.nan for j in range(d)]
    return np.array([count, mean, var])


def standard_scaler_loop(chunks):
    """Fit across chunks, then transform using final statistics; O(n*d)."""
    moments = running_stats_loop(chunks).tolist()
    mean, var = moments[1], moments[2]
    scale = [math.sqrt(value) if value != 0 else 1.0 for value in var]
    return np.array([[(value-mean[j])/scale[j] for j, value in enumerate(row)]
                     for chunk in chunks for row in chunk])


def imputer_loop(chunks):
    """Online mean fills, then transform; NaNs ignored, unseen columns use 0."""
    d = len(chunks[0][0])
    count, mean = [0]*d, [0.0]*d
    for chunk in chunks:
        for row in chunk:
            for j, value in enumerate(row):
                if not math.isnan(value):
                    count[j] += 1
                    mean[j] += (value-mean[j])/count[j]
    return np.array([[mean[j] if math.isnan(value) else value for j, value in enumerate(row)]
                     for chunk in chunks for row in chunk])


def minmax_scaler_loop(chunks):
    """Learn column extrema across chunks, then scale to [0,1]; O(n*d)."""
    d = len(chunks[0][0])
    low, high = [math.inf]*d, [-math.inf]*d
    for chunk in chunks:
        for row in chunk:
            for j, value in enumerate(row):
                if not math.isnan(value):
                    low[j] = min(low[j], value)
                    high[j] = max(high[j], value)
    for j in range(d):
        if low[j] == math.inf:
            low[j] = high[j] = math.nan
    width = [high[j]-low[j] if high[j] != low[j] else math.inf for j in range(d)]
    return np.array([[(value-low[j])/width[j] for j, value in enumerate(row)]
                     for chunk in chunks for row in chunk])


def onehot_encoder_loop(chunks):
    """Learn sorted categories per chunk, then return dense indicators.

    Missing text follows OneHotEncoder: '', 'nan', 'None'. Both sides create
    the full dense output, including zero columns for the other categories.
    """
    missing = {"", "nan", "None"}
    d = len(chunks[0][0])
    seen = [set() for _ in range(d)]
    for chunk in chunks:
        for row in chunk:
            for j, value in enumerate(row):
                if value not in missing:
                    seen[j].add(value)
        categories = [sorted(values) for values in seen]
    lookup, width = [], 0
    for values in categories:
        lookup.append({value: width+i for i, value in enumerate(values)})
        width += len(values)
    encoded = []
    for chunk in chunks:
        for row in chunk:
            output = [0.0]*width
            for j, value in enumerate(row):
                if value not in missing:
                    output[lookup[j][value]] = 1.0
            encoded.append(output)
    return np.array(encoded, dtype=float)


def _linear_quantile(values, q):
    """Python sort and linear interpolation; empty observations give NaN."""
    if not values:
        return math.nan
    values = sorted(values)
    position = (len(values)-1)*q
    lower, upper = math.floor(position), math.ceil(position)
    fraction = position-lower
    return values[lower]*(1-fraction)+values[upper]*fraction


def median_imputer_loop(chunks):
    """Recompute exact column medians after EACH chunk, then fill all rows.

    Like SimpleImputer(median), retain observations and update intermediate
    fill estimates. Sorting only once at the end would time less work.
    """
    d = len(chunks[0][0])
    observed = [[] for _ in range(d)]
    for chunk in chunks:
        for row in chunk:
            for j, value in enumerate(row):
                if not math.isnan(value):
                    observed[j].append(value)
        medians = [_linear_quantile(values, 0.5) if values else 0.0 for values in observed]
    return np.array([[medians[j] if math.isnan(value) else value for j, value in enumerate(row)]
                     for chunk in chunks for row in chunk])


def histogram_loop(chunks, edges):
    """Query saved scalar chunks with fixed edges; the last edge is inclusive."""
    values = []
    for chunk in chunks:
        values.extend(chunk)
    counts = [0]*(len(edges)-1)
    for value in values:
        if math.isnan(value) or value < edges[0] or value > edges[-1]:
            continue
        position = min(bisect_right(edges, value)-1, len(counts)-1)
        counts[position] += 1
    return np.asarray(counts)


def quantile_loop(chunks, probabilities):
    """Query saved scalar chunks, sort once, then use linear interpolation."""
    values = []
    for chunk in chunks:
        for value in chunk:
            if not math.isnan(value):
                values.append(value)
    values.sort()
    result = []
    for q in probabilities:
        if not values:
            result.append(math.nan)
            continue
        position = (len(values)-1)*q
        low, high = math.floor(position), math.ceil(position)
        fraction = position-low
        result.append(values[low]*(1-fraction)+values[high]*fraction)
    return np.array(result)






def mse_loop(chunks):
    """Accumulate squared errors and total sample count; O(n)."""
    total, count = 0.0, 0
    for truth, pred in chunks:
        for target, value in zip(truth, pred):
            total += (target-value)**2
            count += 1
    return total/count


def auc_loop(chunks):
    """Retain scores, then reuse the original positive-negative pair loop."""
    truth, scores = [], []
    for target, score in chunks:
        truth.extend(target)
        scores.extend(score)
    return _batch_auc_loop(truth, scores)


def predict_loop(tree, rows):
    """Traverse the fitted tree one sample at a time; O(n*depth)."""
    predictions = []
    for row in rows:
        node = tree.root_
        while node.feature is not None:
            value = row[node.feature]
            left = node.nan_left if math.isnan(value) else value <= node.threshold
            node = node.left if left else node.right
        predictions.append(tree.classes_[node.prediction])
    return np.asarray(predictions)


def _impurity_loop(counts, criterion):
    """Scalar impurity with zero-count handling."""
    total = sum(counts)
    if not total:
        return 0.0
    if criterion == "gini":
        return 1-sum((value/total)**2 for value in counts)
    return -sum((value/total)*math.log2(value/total) for value in counts if value)


class LoopDecisionTreeClassifier(DecisionTreeClassifier):
    """Replace only split search with scalar loops; keep original tree logic.

    Same NaN/tie rules, O(d*n*n) split work. A depth-one benchmark isolates
    this change from the unchanged recursive tree construction.
    """

    def _best_split(self, X, y, n_classes, parent_impurity, rng):
        features = list(range(X.shape[1]))
        if self.max_features is not None and self.max_features < len(features):
            features = sorted(rng.choice(features, self.max_features, replace=False).tolist())
        rows, labels = X.tolist(), y.tolist()
        best, best_gain = None, 0.0
        for feature in features:
            known = sorted(set(row[feature] for row in rows if not math.isnan(row[feature])))
            if not known:
                continue
            thresholds = []
            for low, high in zip(known[:-1], known[1:]):
                midpoint = low/2+high/2
                thresholds.append(midpoint if low <= midpoint < high else low)
            if any(math.isnan(row[feature]) for row in rows):
                thresholds.append(known[-1])
            candidates = []
            for threshold in thresholds:
                for nan_left in (False, True):
                    left, right = [0]*n_classes, [0]*n_classes
                    for row, label in zip(rows, labels):
                        value = row[feature]
                        goes_left = nan_left if math.isnan(value) else value <= threshold
                        (left if goes_left else right)[label] += 1
                    nl, nr = sum(left), sum(right)
                    gain = parent_impurity-(nl*_impurity_loop(left, self.criterion)+nr*_impurity_loop(right, self.criterion))/len(rows)
                    candidates.append((gain if nl and nr else -math.inf, threshold, nan_left))
            if not candidates:
                continue
            gain = max(value[0] for value in candidates)
            if gain <= best_gain+1e-12:
                continue
            _, threshold, nan_left = next(value for value in candidates if value[0] >= gain-1e-12)
            best, best_gain = (feature, threshold, nan_left), gain
        return best
