"""Grouped checks for the original pipeline API and chunk-wise extensions."""

from contextlib import contextmanager

import numpy as np
import pytest

from numcompute.pipeline import Pipeline as BatchPipeline
from numcompute.preprocessing import SimpleImputer as BatchImputer, StandardScaler as BatchScaler
from numcompute_stream.pipeline import Pipeline, Transformer, Estimator
from numcompute_stream.preprocessing import SimpleImputer, StandardScaler, OneHotEncoder
from numcompute_stream.tree import DecisionTreeClassifier


@contextmanager
def case(name, **options):
    """Identify subcases without expanding the number of pytest functions."""
    try:
        yield
    except (Exception, pytest.fail.Exception) as error:
        raise AssertionError(f"{name}, {options}: {error}") from error


class RecordingTransformer:
    """Duck-typed transformer that shifts arrays and records dispatch order."""

    def __init__(self, name, log, offset=0):
        self.name, self.log, self.offset = name, log, offset

    def fit(self, X):
        self.log.append((self.name, "fit"))
        return self

    def partial_fit(self, X):
        self.log.append((self.name, "partial_fit"))
        self.received_X = np.asarray(X).copy()
        return self

    def transform(self, X):
        self.log.append((self.name, "transform"))
        return np.asarray(X) + self.offset


class RecordingEstimator:
    """Duck-typed model; predict sums features without fitting a real model."""

    def __init__(self, log):
        self.log = log

    def fit(self, X, y):
        self.log.append(("model", "fit"))
        self.received_X, self.received_y = X.copy(), y.copy()
        return self

    def partial_fit(self, X, y):
        self.log.append(("model", "partial_fit"))
        self.received_X, self.received_y = X.copy(), y.copy()
        return self

    def predict(self, X):
        self.log.append(("model", "predict"))
        return X.sum(axis=1)


def test_original_batch_fit_transform_and_predict_are_preserved():
    X = np.array([[1.0, np.nan], [3.0, 2.0], [5.0, 4.0]])
    pipe = Pipeline([("impute", SimpleImputer(fill_value=0)), ("scale", StandardScaler())])
    original = BatchPipeline([("impute", BatchImputer(fill_value=0)), ("scale", BatchScaler())])
    np.testing.assert_allclose(pipe.fit_transform(X), original.fit_transform(X))
    log = []
    model = RecordingEstimator(log)
    pipe = Pipeline([("shift", RecordingTransformer("shift", log, 2)), ("model", model)])
    assert pipe.fit(X[:, :1], np.array([0, 1, 1])) is pipe
    np.testing.assert_array_equal(pipe.predict([[3.0]]), [5.0])
    assert log == [("shift", "fit"), ("shift", "transform"), ("model", "fit"), ("shift", "transform"), ("model", "predict")]


def test_partial_fit_follows_original_order_and_routes_targets_only_to_model():
    X, y = np.array([[1.0, 2.0], [3.0, 4.0]]), np.array(["a", "b"])
    saved_X, saved_y = X.copy(), y.copy()
    log = []
    first, second = RecordingTransformer("first", log, 1), RecordingTransformer("second", log, 10)
    model = RecordingEstimator(log)
    pipe = Pipeline([("first", first), ("second", second), ("model", model)])
    assert pipe.steps[0][1] is first and pipe.steps[-1][1] is model
    assert pipe.partial_fit(X, y) is pipe
    assert log == [("first", "partial_fit"), ("first", "transform"), ("second", "partial_fit"), ("second", "transform"), ("model", "partial_fit")]
    np.testing.assert_array_equal(first.received_X, X)
    np.testing.assert_array_equal(second.received_X, X + 1)
    np.testing.assert_array_equal(model.received_X, X + 11)
    np.testing.assert_array_equal(model.received_y, y)
    log.clear()
    np.testing.assert_array_equal(pipe.predict(X), (X + 11).sum(axis=1))
    assert log == [("first", "transform"), ("second", "transform"), ("model", "predict")]
    np.testing.assert_array_equal(X, saved_X)
    np.testing.assert_array_equal(y, saved_y)
    assert callable(Transformer.partial_fit) and callable(Estimator.partial_fit)


def test_numeric_chunks_match_batch_with_stable_imputation_and_strides():
    X = np.random.default_rng(7).normal(size=(47, 3))
    X[::5, 0] = np.nan
    X[::7, 2] = np.nan
    view = X[::-2, ::-1]
    saved = view.copy()
    expected = BatchPipeline([("impute", BatchImputer(fill_value=0)), ("scale", BatchScaler())]).fit_transform(view)
    for size in (1, 4, 100):
        with case("numeric chunk size", size=size):
            scaler = StandardScaler()
            pipe = Pipeline([("impute", SimpleImputer(fill_value=0)), ("scale", scaler)])
            for start in range(0, len(view), size):
                pipe.partial_fit(view[start:start + size])
            np.testing.assert_allclose(pipe.transform(view), expected, atol=1e-14)
            np.testing.assert_array_equal(scaler.n_samples_seen_, [len(view)] * 3)
    np.testing.assert_array_equal(view, saved)


def test_each_transformer_learns_the_current_upstream_output():
    X = np.array([[1.0, np.nan], [3.0, 2.0], [100.0, 8.0], [np.nan, 20.0]])
    imputer, scaler = SimpleImputer(strategy="mean"), StandardScaler()
    pipe = Pipeline([("impute", imputer), ("scale", scaler)])
    manual_imputer, manual_scaler = SimpleImputer(strategy="mean"), StandardScaler()
    for chunk in (X[:2], X[2:]):
        pipe.partial_fit(chunk)
        manual_imputer.partial_fit(chunk)
        manual_scaler.partial_fit(manual_imputer.transform(chunk))
    np.testing.assert_allclose(imputer.statistics_, manual_imputer.statistics_)
    np.testing.assert_allclose(scaler.mean_, manual_scaler.mean_)
    np.testing.assert_allclose(scaler.var_, manual_scaler.var_)
    np.testing.assert_allclose(pipe.transform(X), manual_scaler.transform(manual_imputer.transform(X)))


def test_categorical_pipeline_expands_and_keeps_missing_value_behavior():
    encoder = OneHotEncoder(handle_unknown="ignore")
    pipe = Pipeline([("impute", SimpleImputer(fill_value="Unknown")), ("encode", encoder)])
    pipe.partial_fit([["Sydney"], [""]])
    assert pipe.transform([["Sydney"], [""]]).shape == (2, 2)
    np.testing.assert_array_equal(pipe.transform([["Adelaide"]]), [[0, 0]])
    pipe.partial_fit([["Adelaide"], [None]])
    np.testing.assert_array_equal(encoder.categories_[0], ["Adelaide", "Sydney", "Unknown"])
    np.testing.assert_array_equal(pipe.transform([["Adelaide"], ["Sydney"], [""]]), np.eye(3))
    # fit starts again rather than merging with categories from earlier chunks.
    pipe.fit([["Perth"]])
    np.testing.assert_array_equal(encoder.categories_[0], ["Perth"])


def test_buffered_tree_matches_manual_updates_and_predicts_last_trained_tree():
    tree = DecisionTreeClassifier(max_depth=2, buffer_size=5)
    pipe = Pipeline([("impute", SimpleImputer(fill_value=0)), ("model", tree)])
    manual = DecisionTreeClassifier(max_depth=2, buffer_size=5)
    chunks = [(np.array([[np.nan], [1.0]]), np.array([0, 0])),
              (np.array([[8.0], [9.0]]), np.array([1, 1])),
              (np.array([[2.0], [10.0]]), np.array([0, 1]))]
    for number, (X, y) in enumerate(chunks, start=1):
        assert pipe.partial_fit(X, y) is pipe
        manual.partial_fit(np.where(np.isnan(X), 0, X), y)
        assert tree.n_buffered_samples_ == manual.n_buffered_samples_
        assert tree.n_retrains_ == manual.n_retrains_
        if number < 3:
            assert pipe.is_fitted_  # Updates succeeded; the tree is still warming up.
            with pytest.raises(RuntimeError):
                pipe.predict([[1.0]])
    assert tree.n_training_samples_ == 6 and tree.n_retrains_ == 1
    probe = np.array([[np.nan], [1.0], [9.0]])
    previous = pipe.predict(probe)
    np.testing.assert_array_equal(previous, manual.predict(np.where(np.isnan(probe), 0, probe)))
    pipe.partial_fit([[3.0]], [0])
    assert tree.n_buffered_samples_ == 1 and tree.n_retrains_ == 1
    np.testing.assert_array_equal(pipe.predict(probe), previous)
    # The original fit path resets the estimator and its pending buffer.
    pipe.fit([[0.0], [10.0]], [0, 1])
    assert tree.n_buffered_samples_ == 0 and tree.n_retrains_ == 0
    np.testing.assert_array_equal(pipe.predict(probe), [0, 0, 1])


def test_single_step_and_final_transformer_ignore_optional_targets():
    scaler = StandardScaler()
    pipe = Pipeline([("scale", scaler)])
    pipe.partial_fit([[1.0], [3.0]], [100, 200])
    np.testing.assert_allclose(pipe.transform([[1.0], [3.0]]), [[-1], [1]])
    with pytest.raises(ValueError, match="predict"):
        pipe.predict([[1.0]])
    model = DecisionTreeClassifier(buffer_size=2)
    pipe = Pipeline([("model", model)]).partial_fit([[0.0], [2.0]], [0, 1])
    np.testing.assert_array_equal(pipe.predict([[0.0], [2.0]]), [0, 1])
    with pytest.raises(ValueError, match="transform"):
        pipe.transform([[0.0]])
    with pytest.raises(ValueError, match="transformer"):
        pipe.fit_transform([[0.0]], [0])


def test_batch_only_components_fail_stream_preflight_without_partial_updates():
    first = StandardScaler()
    last = BatchScaler()
    pipe = Pipeline([("new", first), ("old", last)])
    pipe.fit([[1.0], [3.0]])  # Batch-only components still work through fit.
    before = first.mean_.copy()
    with pytest.raises(ValueError, match="old.*partial_fit"):
        pipe.partial_fit([[100.0]])
    np.testing.assert_array_equal(first.mean_, before)
    assert pipe.is_fitted_
    assert not hasattr(last, "partial_fit")


def test_empty_chunks_and_invalid_shapes_do_not_update_steps():
    scaler = StandardScaler()
    pipe = Pipeline([("scale", scaler)])
    assert pipe.partial_fit(np.empty((0, 2))) is pipe
    assert not pipe.is_fitted_ and not hasattr(scaler, "n_features_in_")
    for method in (pipe.transform, pipe.predict):
        with pytest.raises(ValueError, match="fit"):
            method([[1.0, 2.0]])
    pipe.partial_fit([[1.0, 2.0], [3.0, 4.0]])
    before = scaler.mean_.copy()
    assert pipe.partial_fit(np.empty((0, 2)), np.empty(0)) is pipe
    assert pipe.transform(np.empty((0, 2))).shape == (0, 2)
    for X, y in (([1.0, 2.0], None), (np.zeros((1, 1, 2)), None), (np.empty((2, 0)), None),
                 ([[1.0]], None), (np.empty((0, 1)), None), ([[1.0, 2.0]], [0, 1]),
                 ([[1.0, 2.0]], [[0]]), ([[1.0, 2.0]], 0)):
        with case("invalid stream shape", X=X, y=y), pytest.raises(ValueError):
            pipe.partial_fit(X, y)
        np.testing.assert_array_equal(scaler.mean_, before)
        assert pipe.is_fitted_


def test_component_errors_clear_flag_and_preserve_original_recovery_path():
    class FailingTransformer(RecordingTransformer):
        def partial_fit(self, X):
            raise RuntimeError("test update failure")

    log = []
    first = RecordingTransformer("first", log, 1)
    pipe = Pipeline([("first", first), ("fail", FailingTransformer("fail", log))])
    pipe.fit([[1.0]])
    log.clear()
    with pytest.raises(RuntimeError, match="test update failure"):
        pipe.partial_fit([[2.0]])
    assert not pipe.is_fitted_
    assert log == [("first", "partial_fit"), ("first", "transform")]
    with pytest.raises(ValueError, match="fit"):
        pipe.transform([[2.0]])
    assert pipe.fit([[2.0]]) is pipe  # Original batch fit still recovers.
    assert pipe.is_fitted_
    # Target validation owned by a component still propagates through the chain.
    pipe = Pipeline([("model", DecisionTreeClassifier(buffer_size=2))])
    with pytest.raises(TypeError):
        pipe.partial_fit([[0.0], [1.0]])  # Required y was omitted.
    assert not pipe.is_fitted_
    # A generic pipeline cannot silently resize a downstream model's schema.
    encoder, tree = OneHotEncoder(), DecisionTreeClassifier(buffer_size=2)
    pipe = Pipeline([("encode", encoder), ("model", tree)]).partial_fit([["Sydney"]], [0])
    with pytest.raises(ValueError, match="features"):
        pipe.partial_fit([["Adelaide"]], [1])
    assert not pipe.is_fitted_ and tree.n_buffered_samples_ == 1
    assert encoder.n_features_out_ == 2
    pipe.fit([["Sydney"], ["Adelaide"]], [0, 1])
    np.testing.assert_array_equal(pipe.predict([["Sydney"], ["Adelaide"]]), [0, 1])


def test_original_constructor_validation_is_retained():
    bad_steps = ([], [("", StandardScaler())], [("x", StandardScaler()), ("x", StandardScaler())],
                 [("x", object())], [("model", RecordingEstimator([])), ("scale", StandardScaler())])
    for steps in bad_steps:
        with case("invalid original steps", steps=steps), pytest.raises(ValueError):
            Pipeline(steps)
