"""Unit tests for decision-tree learning and buffered streaming retraining."""

import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier


def test_buffered_label_promotion_cannot_silently_merge_integer_classes():
    labels = np.array([2**63, 2**63 + 1], dtype=np.uint64)
    tree = DecisionTreeClassifier(buffer_size=3).partial_fit([[0], [1]], labels)
    with pytest.raises(TypeError, match="lossless|dtype"):
        tree.partial_fit([[2]], np.array([-1], dtype=np.int64))
    assert tree.n_buffered_samples_ == 2 and tree.n_retrains_ == 0
    np.testing.assert_array_equal(tree._y_buffer[0], labels)
    tree.partial_fit([[2]], labels[:1])
    np.testing.assert_array_equal(tree.classes_, labels)
    np.testing.assert_array_equal(tree.predict([[0], [1]]), labels)
    # A replacement tree need not share the previous tree's label dtype.
    tree.partial_fit([[0], [1], [2]], [-1, -1, -1])
    np.testing.assert_array_equal(tree.classes_, [-1])
    assert tree.n_retrains_ == 2


def test_full_tree_and_unequal_buffers_match_loop_search_for_multiple_streams():
    from benchmark.stream_loops import LoopDecisionTreeClassifier

    for seed in range(8):
        rng = np.random.default_rng(seed)
        X = rng.integers(-2, 3, size=(18, 3)).astype(float)
        X[rng.random(X.shape) < .2] = np.nan
        y = rng.integers(0, 3, size=len(X))
        query = rng.normal(size=(9, 3))
        query[rng.random(query.shape) < .2] = np.nan
        for criterion in ("gini", "entropy"):
            for threshold in (1, 7, 100):
                tree = DecisionTreeClassifier(max_depth=3, criterion=criterion, buffer_size=threshold)
                pending_X, pending_y, reference, retrains = [], [], None, 0
                start = 0
                for size in (2, 1, 8, 4, 3):
                    stop = start + size
                    pending_X.append(X[start:stop])
                    pending_y.append(y[start:stop])
                    tree.partial_fit(X[start:stop], y[start:stop])
                    count = sum(len(chunk) for chunk in pending_y)
                    if count >= threshold:
                        reference = LoopDecisionTreeClassifier(max_depth=3, criterion=criterion).fit(
                            np.vstack(pending_X), np.concatenate(pending_y))
                        pending_X, pending_y = [], []
                        retrains += 1
                    assert tree.n_buffered_samples_ == sum(len(chunk) for chunk in pending_y)
                    assert tree.n_retrains_ == retrains
                    if reference is None:
                        with pytest.raises(RuntimeError):
                            tree.predict(query)
                    else:
                        np.testing.assert_array_equal(tree.predict(query), reference.predict(query))
                    start = stop


@pytest.fixture
def simple_data():
    return np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1])


# Basic learning and impurity calculations.

@pytest.mark.parametrize("criterion, expected", [("gini", 0.5), ("entropy", 1.0)])
def test_impurity_matches_hand_calculation_and_handles_zero_counts(criterion, expected):
    model = DecisionTreeClassifier(criterion=criterion)
    with np.errstate(divide="raise", invalid="raise"):
        result = model._impurity(np.array([[2, 2], [4, 0], [0, 0]]))
    np.testing.assert_allclose(result, [expected, 0, 0])


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_fit_returns_self_and_learns_numeric_threshold(simple_data, criterion):
    X, y = simple_data
    model = DecisionTreeClassifier(max_depth=1, criterion=criterion)
    assert model.fit(X, y) is model
    assert model.root_.feature == 0
    assert model.root_.threshold == pytest.approx(1.5)
    np.testing.assert_array_equal(model.predict(X), y)
    np.testing.assert_array_equal(model.predict([[0.5], [1.6]]), [0, 1])
    assert model.n_training_samples_ == 4


def test_multiclass_string_labels_and_prediction_dtype():
    X = np.arange(6.0).reshape(-1, 1)
    y = np.array(["Adelaide", "Adelaide", "Perth", "Perth", "Sydney", "Sydney"])
    model = DecisionTreeClassifier().fit(X, y)
    np.testing.assert_array_equal(model.classes_, ["Adelaide", "Perth", "Sydney"])
    result = model.predict(X)
    np.testing.assert_array_equal(result, y)
    assert result.shape == (6,)
    assert result.dtype == model.classes_.dtype


def test_negative_and_nonconsecutive_numeric_labels():
    X = np.arange(4).reshape(-1, 1)
    y = np.array([-7, -7, 42, 42])
    np.testing.assert_array_equal(DecisionTreeClassifier().fit(X, y).predict(X), y)


def test_boolean_features_and_labels():
    X = np.array([[False], [True], [False], [True]])
    y = np.array([False, True, False, True])
    np.testing.assert_array_equal(DecisionTreeClassifier().fit(X, y).predict(X), y)


def test_byte_string_labels_are_converted_and_blank_labels_rejected():
    model = DecisionTreeClassifier().fit([[0], [1]], np.array([b"a", b"b"]))
    np.testing.assert_array_equal(model.predict([[0], [1]]), ["a", "b"])
    with pytest.raises(ValueError, match="blank"):
        model.fit([[0]], np.array([b""]))


def test_pure_class_creates_leaf():
    model = DecisionTreeClassifier().fit([[1], [2], [3]], [8, 8, 8])
    assert model.root_.feature is None
    np.testing.assert_array_equal(model.predict([[100], [np.nan]]), [8, 8])


def test_zero_depth_uses_majority_and_sorted_tie_rule():
    model = DecisionTreeClassifier(max_depth=0).fit([[0], [1]], ["b", "a"])
    assert model.root_.feature is None
    np.testing.assert_array_equal(model.predict([[0], [1]]), ["a", "a"])


def test_min_samples_split_stops_small_nodes(simple_data):
    X, y = simple_data
    model = DecisionTreeClassifier(min_samples_split=5).fit(X, y)
    assert model.root_.feature is None
    np.testing.assert_array_equal(model.predict(X), [0, 0, 0, 0])


def test_constant_features_do_not_make_empty_children():
    model = DecisionTreeClassifier().fit(np.ones((4, 2)), [2, 2, 2, 1])
    assert model.root_.feature is None
    np.testing.assert_array_equal(model.predict([[1, 1]]), [2])


def test_duplicate_feature_values_are_not_split_apart():
    X = np.array([[0], [0], [1], [1]], dtype=float)
    model = DecisionTreeClassifier().fit(X, [0, 1, 1, 1])
    result = model.predict(X)
    assert result[0] == result[1]
    assert model.root_.threshold == pytest.approx(0.5)


def test_equal_split_gains_choose_lower_feature_then_threshold():
    x = np.arange(4.0)
    X = np.column_stack((x, x))
    model = DecisionTreeClassifier(max_depth=1).fit(X, [0, 1, 0, 1])
    assert model.root_.feature == 0
    assert model.root_.threshold == pytest.approx(0.5)
    assert not model.root_.nan_left


def test_depth_limit_and_unlimited_growth():
    X = np.arange(8.0).reshape(-1, 1)
    y = np.arange(8) % 2

    def depth(node):
        return 0 if node.feature is None else 1 + max(depth(node.left), depth(node.right))

    shallow = DecisionTreeClassifier(max_depth=2).fit(X, y)
    assert depth(shallow.root_) <= 2
    full = DecisionTreeClassifier(max_depth=None).fit(X, y)
    np.testing.assert_array_equal(full.predict(X), y)


def test_feature_subsets_are_reproducible_and_respect_limit():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(50, 4))
    y = (X[:, 0] + X[:, 2] > 0).astype(int)
    a = DecisionTreeClassifier(max_features=1, random_state=7).fit(X, y)
    b = DecisionTreeClassifier(max_features=1, random_state=7).fit(X, y)
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    assert a.root_.feature == b.root_.feature
    expected_feature = int(np.random.default_rng(7).choice(np.arange(4), 1)[0])
    assert a.root_.feature in (None, expected_feature)


# Missing features and floating-point edge cases.

@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("missing_label, nan_left", [(0, True), (1, False)])
def test_missing_values_learn_the_better_branch(criterion, missing_label, nan_left):
    X = np.array([[0.0], [1.0], [2.0], [np.nan]])
    y = np.array([0, 0, 1, missing_label])
    model = DecisionTreeClassifier(max_depth=1, criterion=criterion).fit(X, y)
    assert model.root_.nan_left == nan_left
    np.testing.assert_array_equal(model.predict(X), y)


def test_missingness_can_split_an_otherwise_constant_feature():
    X = np.array([[3.0], [3.0], [np.nan], [np.nan]])
    y = np.array([0, 0, 1, 1])
    model = DecisionTreeClassifier().fit(X, y)
    assert model.root_.threshold == 3.0
    np.testing.assert_array_equal(model.predict(X), y)


def test_all_nan_features_make_majority_leaf():
    model = DecisionTreeClassifier().fit(np.full((3, 2), np.nan), [1, 0, 1])
    assert model.root_.feature is None
    np.testing.assert_array_equal(model.predict([[np.nan, np.nan]]), [1])


def test_unseen_nan_uses_fixed_right_branch_on_gain_tie(simple_data):
    X, y = simple_data
    model = DecisionTreeClassifier().fit(X, y)
    assert not model.root_.nan_left
    np.testing.assert_array_equal(model.predict([[np.nan]]), [1])


@pytest.mark.parametrize("values", [
    [-1e308, 1e308],
    [1e308, 1.5e308],
    [1.0, np.nextafter(1.0, 2.0)],
    [0.0, np.nextafter(0.0, 1.0)],
])
def test_threshold_handles_extreme_and_adjacent_floats(values):
    X = np.asarray(values).reshape(-1, 1)
    with np.errstate(over="raise", divide="raise", invalid="raise", under="raise"):
        model = DecisionTreeClassifier().fit(X, [0, 1])
        assert np.isfinite(model.root_.threshold)
        np.testing.assert_array_equal(model.predict(X), [0, 1])


def test_non_contiguous_inputs_work_without_mutating_callers():
    base = np.arange(40.0).reshape(10, 4)
    X = base[::2, ::2]
    y = np.array([0, 9, 0, 9, 1, 9, 1, 9, 1, 9])[::2]
    before_X, before_y = X.copy(), y.copy()
    assert not X.flags.c_contiguous
    model = DecisionTreeClassifier().fit(X, y)
    np.testing.assert_array_equal(model.predict(X), model.predict(X.copy()))
    np.testing.assert_array_equal(X, before_X)
    np.testing.assert_array_equal(y, before_y)


def test_predict_empty_rows_returns_empty_label_array(simple_data):
    X, y = simple_data
    model = DecisionTreeClassifier().fit(X, y)
    result = model.predict(np.empty((0, 1)))
    assert result.shape == (0,)
    assert result.dtype == model.classes_.dtype


# An independent loop-based reference checks the vectorised split search.

def _reference_impurity(y, criterion):
    if not len(y):
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - sum(p * p) if criterion == "gini" else -sum(p * np.log2(p))


def _reference_split(X, y, criterion):
    before = _reference_impurity(y, criterion)
    best_gain, best = 0.0, None
    for feature in range(X.shape[1]):
        values = X[:, feature]
        known_values = np.unique(values[~np.isnan(values)])
        if not len(known_values):
            continue
        thresholds = list((known_values[:-1] + known_values[1:]) / 2)
        if np.isnan(values).any():
            thresholds.append(known_values[-1])
        for threshold in thresholds:
            for nan_left in (False, True):
                mask = np.where(np.isnan(values), nan_left, values <= threshold)
                if mask.all() or not mask.any():
                    continue
                after = (mask.sum() * _reference_impurity(y[mask], criterion)
                         + (~mask).sum() * _reference_impurity(y[~mask], criterion)) / len(y)
                gain = before - after
                if gain > best_gain + 1e-12:
                    best_gain, best = gain, (feature, threshold, nan_left)
    return best


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
@pytest.mark.parametrize("seed", [0, 3, 7, 19])
def test_vectorised_root_matches_exhaustive_loop_reference(criterion, seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 5, size=(25, 3)).astype(float)
    X[rng.random(X.shape) < 0.15] = np.nan
    y = rng.integers(0, 3, size=len(X))
    expected = _reference_split(X, y, criterion)
    assert expected is not None
    model = DecisionTreeClassifier(max_depth=1, criterion=criterion).fit(X, y)
    assert model.root_.feature == expected[0]
    assert model.root_.threshold == pytest.approx(expected[1])
    assert model.root_.nan_left == expected[2]


# Streaming: buffer boundaries, replacement and state preservation.

def test_partial_fit_waits_below_threshold_and_predict_requires_tree():
    model = DecisionTreeClassifier(buffer_size=3)
    with pytest.raises(RuntimeError, match="before predict"):
        model.predict([[0]])
    assert model.partial_fit([[0], [1]], [0, 1]) is model
    assert model.n_buffered_samples_ == 2
    assert model.n_retrains_ == 0
    assert not hasattr(model, "root_")
    with pytest.raises(RuntimeError, match="buffer_size"):
        model.predict([[0]])


def test_exact_threshold_retrains_and_clears_the_entire_buffer(simple_data):
    X, y = simple_data
    model = DecisionTreeClassifier(buffer_size=4)
    model.partial_fit(X[:2], y[:2]).partial_fit(X[2:], y[2:])
    assert model.n_buffered_samples_ == 0
    assert model.n_retrains_ == 1
    assert model.n_training_samples_ == 4
    assert not model._X_buffer and not model._y_buffer
    np.testing.assert_array_equal(model.predict(X), y)


def test_overflow_uses_all_samples_including_unequal_final_chunk():
    model = DecisionTreeClassifier(buffer_size=5, max_depth=0)
    model.partial_fit([[0], [1], [2]], [0, 0, 0])
    model.partial_fit([[3], [4], [5], [6]], [1, 1, 1, 1])
    assert model.n_training_samples_ == 7
    assert model.n_buffered_samples_ == 0
    np.testing.assert_array_equal(model.predict([[0]]), [1])


def test_oversized_single_chunk_trains_once_not_in_fixed_size_slices():
    model = DecisionTreeClassifier(buffer_size=2, max_depth=0)
    model.partial_fit(np.arange(7).reshape(-1, 1), [0, 0, 1, 1, 1, 1, 1])
    assert model.n_training_samples_ == 7
    assert model.n_retrains_ == 1
    assert model.n_buffered_samples_ == 0


def test_buffer_size_one_trains_immediately():
    model = DecisionTreeClassifier(buffer_size=1).partial_fit([[2]], ["Sydney"])
    assert model.n_retrains_ == 1
    np.testing.assert_array_equal(model.predict([[3]]), ["Sydney"])


def test_between_retrains_predict_uses_the_previous_tree():
    model = DecisionTreeClassifier(buffer_size=2).partial_fit([[0], [1]], [0, 0])
    old_root = model.root_
    model.partial_fit([[2]], [1])
    assert model.root_ is old_root
    assert model.n_buffered_samples_ == 1
    np.testing.assert_array_equal(model.predict([[2]]), [0])
    model.partial_fit([[3]], [1])
    assert model.root_ is not old_root
    assert model.n_retrains_ == 2
    np.testing.assert_array_equal(model.classes_, [1])
    np.testing.assert_array_equal(model.predict([[2]]), [1])


def test_partial_fit_copies_pending_data_and_allows_new_text_labels():
    X, y = np.array([[0.0]]), np.array(["Adelaide"])
    model = DecisionTreeClassifier(buffer_size=2).partial_fit(X, y)
    X[0, 0], y[0] = 100, "Perth"
    model.partial_fit([[1]], ["Sydney"])
    np.testing.assert_array_equal(model.classes_, ["Adelaide", "Sydney"])
    np.testing.assert_array_equal(model.predict([[0], [1]]), ["Adelaide", "Sydney"])


def test_incomplete_final_buffer_remains_pending():
    model = DecisionTreeClassifier(buffer_size=5)
    model.partial_fit([[0], [1]], [0, 1])
    model.partial_fit([[2]], [1])
    assert model.n_buffered_samples_ == 3
    assert model.n_retrains_ == 0
    assert not hasattr(model, "root_")


def test_empty_chunks_are_no_ops_before_and_after_training():
    model = DecisionTreeClassifier(buffer_size=2)
    assert model.partial_fit(np.empty((0, 1)), []) is model
    assert not hasattr(model, "n_features_in_")
    model.partial_fit([[0], [1]], ["a", "b"])
    old_root = model.root_
    model.partial_fit(np.empty((0, 1)), np.array([], dtype=str))
    assert model.root_ is old_root
    assert model.n_retrains_ == 1
    assert model.n_buffered_samples_ == 0


def test_fit_resets_pending_buffer_counter_and_feature_schema():
    model = DecisionTreeClassifier(buffer_size=2).partial_fit([[0], [1]], [0, 1])
    model.partial_fit([[2]], [0])
    assert model.n_retrains_ == 1 and model.n_buffered_samples_ == 1
    model.fit([[1, 2], [3, 4]], ["a", "b"])
    assert model.n_retrains_ == 0 and model.n_buffered_samples_ == 0
    assert model.n_features_in_ == 2
    assert not model._X_buffer and not model._y_buffer
    np.testing.assert_array_equal(model.predict([[1, 2], [3, 4]]), ["a", "b"])


def test_streamed_buffer_build_matches_batch_fit_with_same_data(simple_data):
    X, y = simple_data
    streamed = DecisionTreeClassifier(buffer_size=4).partial_fit(X[:1], y[:1])
    streamed.partial_fit(X[1:], y[1:])
    batch = DecisionTreeClassifier().fit(X, y)
    np.testing.assert_array_equal(streamed.predict(X), batch.predict(X))


# Invalid inputs must be informative and leave existing state untouched.

@pytest.mark.parametrize("settings, error", [
    ({"max_depth": -1}, ValueError), ({"max_depth": True}, TypeError),
    ({"max_depth": 1.5}, TypeError), ({"min_samples_split": 1}, ValueError),
    ({"min_samples_split": False}, TypeError), ({"max_features": 0}, ValueError),
    ({"max_features": "sqrt"}, TypeError), ({"buffer_size": 0}, ValueError),
    ({"buffer_size": 1.2}, TypeError), ({"buffer_size": True}, TypeError),
    ({"criterion": "other"}, ValueError), ({"random_state": -1}, ValueError),
    ({"random_state": True}, TypeError),
])
def test_invalid_constructor_settings(settings, error):
    with pytest.raises(error):
        DecisionTreeClassifier(**settings)


@pytest.mark.parametrize("X, y, error", [
    ([], [], ValueError), ([[1], [2]], [0], ValueError),
    ([[1], [2]], [[0], [1]], ValueError), ([[1, 2]], [0, 1], ValueError),
    (np.empty((0, 1)), [], ValueError), (np.empty((2, 0)), [0, 1], ValueError),
    ([[np.inf]], [0], ValueError), ([[1j]], [0], TypeError),
    ([["one"]], [0], TypeError), ([[1]], [np.nan], ValueError),
    ([[1]], [np.inf], ValueError), ([[1]], [""], ValueError),
    ([[1]], ["  "], ValueError), ([[1]], [1j], TypeError),
])
def test_fit_rejects_invalid_data(X, y, error):
    with pytest.raises(error):
        DecisionTreeClassifier().fit(X, y)


def test_max_features_cannot_exceed_feature_count():
    model = DecisionTreeClassifier(max_features=2)
    with pytest.raises(ValueError, match="max_features"):
        model.fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError, match="max_features"):
        model.partial_fit([[0]], [0])


@pytest.mark.parametrize("X, error", [
    ([1, 2], ValueError), ([[1, 2]], ValueError),
    ([[np.inf]], ValueError), ([[1j]], TypeError), ([["one"]], TypeError),
])
def test_predict_rejects_wrong_shape_or_values(simple_data, X, error):
    train_X, y = simple_data
    model = DecisionTreeClassifier().fit(train_X, y)
    with pytest.raises(error):
        model.predict(X)


@pytest.mark.parametrize("X, y, error", [
    ([[2, 3]], [0], ValueError), ([[2]], [np.nan], ValueError),
    ([[2]], ["different family"], TypeError), ([[2], [3]], [0], ValueError),
])
def test_invalid_chunks_preserve_pending_buffer_and_tree(X, y, error):
    model = DecisionTreeClassifier(buffer_size=3).fit([[0], [1]], [0, 1])
    model.partial_fit([[2]], [1])
    old_root = model.root_
    before = model.predict([[0], [1]])
    with pytest.raises(error):
        model.partial_fit(X, y)
    assert model.root_ is old_root
    assert model.n_buffered_samples_ == 1
    assert len(model._X_buffer) == 1 and model.n_retrains_ == 0
    np.testing.assert_array_equal(model.predict([[0], [1]]), before)


def test_invalid_fit_keeps_previous_tree_and_buffer():
    model = DecisionTreeClassifier(buffer_size=3).fit([[0], [1]], [0, 1])
    model.partial_fit([[2]], [1])
    old_root = model.root_
    with pytest.raises(ValueError):
        model.fit([], [])
    assert model.root_ is old_root and model.n_buffered_samples_ == 1


def test_first_pending_chunk_fixes_feature_count():
    model = DecisionTreeClassifier(buffer_size=3).partial_fit([[0, 1]], [0])
    with pytest.raises(ValueError, match="features"):
        model.partial_fit([[2]], [1])
    assert model.n_buffered_samples_ == 1
