"""Grouped tests for incremental scaling, imputation and categorical encoding."""

from contextlib import contextmanager
import warnings

import numpy as np
import pytest

from numcompute import preprocessing as batch
from numcompute_stream.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler


def test_strict_float_extremes_keep_finite_imputation_and_scaling():
    tiny = np.nextafter(0., 1.)
    for strategy in ("mean", "median"):
        for value in (tiny, 1e308):
            with np.errstate(all="raise"):
                imputer = SimpleImputer(strategy=strategy).partial_fit([[value]])
                imputer.partial_fit([[value], [np.nan]])
                np.testing.assert_array_equal(imputer.statistics_, [value])
                np.testing.assert_array_equal(imputer.transform([[np.nan]]), [[value]])
                np.testing.assert_array_equal(imputer.n_samples_seen_, [2])
    with np.errstate(all="raise"):
        opposite = SimpleImputer().partial_fit([[1e308]])
        opposite.partial_fit([[-1e308], [-1e308], [-1e308]])
        np.testing.assert_allclose(opposite.statistics_, [-5e307])
        mixed = SimpleImputer().fit([[1e308], [-1e308], [1e308]])
        np.testing.assert_allclose(mixed.statistics_, [1e308 / 3])
        scaler = MinMaxScaler().partial_fit([[-1e308]]).partial_fit([[0.]])
        np.testing.assert_array_equal(scaler.transform([[1e308]]), [[2.]])
        np.testing.assert_array_equal(scaler.n_samples_seen_, [2])
        large_range = MinMaxScaler((0, 1e308)).fit([[0], [1e308]])
        assert large_range.transform([[1e-308]])[0, 0] == pytest.approx(1e-308, rel=1e-14, abs=0)
        for bounds in ((0, 1e-308), (-1e308, 0)):
            train = [[0.], [1e-308]] if bounds[0] == 0 else [[0.], [1.]]
            test = [[1e308]] if bounds[0] == 0 else [[2.]]
            np.testing.assert_allclose(MinMaxScaler(bounds).fit(train).transform(test), [[1e308]])
    with pytest.raises(ValueError):
        MinMaxScaler(np.array([0 + 1j, 1 + 2j]))
    with pytest.raises(FloatingPointError):
        MinMaxScaler((0, 1e308)).fit([[0], [1]]).transform([[2]])


@contextmanager
def case(name, **parameters):
    """Keep grouped failures identifiable without multiplying pytest cases."""
    try:
        yield
    except (Exception, pytest.fail.Exception) as error:
        raise AssertionError(f"{name}, {parameters}: {error}") from error


@pytest.fixture
def X():
    X = np.random.default_rng(42).normal(size=(37, 3))
    X[::5, 0] = np.nan
    X[::7, 1] = np.nan
    return X


def test_standard_scaler_chunk_sizes_match_full_data(X):
    original = X.copy()
    for size in (1, 4, 15, 100):
        with case("chunk size", size=size):
            scaler = StandardScaler()
            for start in range(0, len(X), size):
                assert scaler.partial_fit(X[start:start + size]) is scaler
            np.testing.assert_allclose(scaler.mean_, np.nanmean(X, axis=0), atol=1e-14)
            np.testing.assert_allclose(scaler.var_, np.nanvar(X, axis=0), atol=1e-14)
            np.testing.assert_array_equal(scaler.n_samples_seen_, (~np.isnan(X)).sum(axis=0))
            expected = batch.StandardScaler().fit_transform(X)
            np.testing.assert_allclose(scaler.transform(X), expected, atol=1e-14)
            before = scaler.mean_.copy()
            scaler.transform(X * 10)
            np.testing.assert_array_equal(scaler.mean_, before)
            assert scaler._stats.store_data is False
            assert scaler._stats._chunks == []
    np.testing.assert_array_equal(X, original)


def test_nan_columns_recover_and_constant_columns_are_safe():
    first = np.array([[np.nan, 7.0], [np.nan, 7.0]])
    later = np.array([[2.0, np.nan], [4.0, 7.0]])
    for cls in (StandardScaler, MinMaxScaler):
        with case("NaNs and constants", cls=cls.__name__), warnings.catch_warnings():
            warnings.simplefilter("error")
            scaler = cls().partial_fit(first)
            assert np.isnan(scaler.transform([[5.0, 7.0]])[0, 0])
            assert scaler.transform([[np.nan, 7.0]])[0, 1] == 0
            scaler.partial_fit(later)
            np.testing.assert_array_equal(scaler.n_samples_seen_, [2, 3])
            np.testing.assert_allclose(scaler.transform([[2.0, 7.0]]), [[-1.0, 0.0]] if cls is StandardScaler else [[0.0, 0.0]])
            # An entirely missing later chunk must not erase earlier estimates.
            previous = scaler.transform([[3.0, 7.0]])
            scaler.partial_fit(np.full((2, 2), np.nan))
            np.testing.assert_array_equal(scaler.transform([[3.0, 7.0]]), previous)
    for strategy in ("mean", "median"):
        with case("imputer recovery", strategy=strategy), warnings.catch_warnings():
            warnings.simplefilter("error")
            imputer = SimpleImputer(strategy=strategy, fill_value=-9).partial_fit(first)
            np.testing.assert_array_equal(imputer.statistics_, [-9, 7])
            imputer.partial_fit(later)
            np.testing.assert_allclose(imputer.statistics_, [3, 7])
            imputer.partial_fit(np.full((1, 2), np.nan))
            np.testing.assert_allclose(imputer.statistics_, [3, 7])


def test_standard_scaler_large_offsets_and_population_variance():
    X = (1e12 + np.arange(40.0)).reshape(20, 2)
    scaler = StandardScaler().partial_fit(X[:3]).partial_fit(X[3:])
    np.testing.assert_allclose(scaler.mean_, X.mean(axis=0), rtol=0, atol=1e-4)
    np.testing.assert_allclose(scaler.var_, X.var(axis=0), rtol=1e-6)
    np.testing.assert_allclose(scaler.transform(X).std(axis=0), 1, rtol=1e-6)
    # This is population variance (ddof=0), not sample variance.
    assert StandardScaler().fit([[1.0], [3.0]]).var_[0] == 1.0


def test_minmax_scaler_updates_extrema_and_custom_ranges(X):
    for bounds in ((0, 1), (-2, 3)):
        with case("feature range", bounds=bounds):
            scaler = MinMaxScaler(bounds).partial_fit(X[:2]).partial_fit(X[2:])
            np.testing.assert_allclose(scaler.data_min_, np.nanmin(X, axis=0))
            np.testing.assert_allclose(scaler.data_max_, np.nanmax(X, axis=0))
            np.testing.assert_array_equal(scaler.n_samples_seen_, (~np.isnan(X)).sum(axis=0))
            np.testing.assert_allclose(scaler.transform(X), batch.MinMaxScaler(bounds).fit_transform(X))
    scaler = MinMaxScaler((-2, 3)).partial_fit([[0.0], [10.0]])
    np.testing.assert_allclose(scaler.transform([[-10.0], [20.0]]), [[-7.0], [8.0]])
    scaler.partial_fit([[-10.0], [20.0]])
    np.testing.assert_allclose(scaler.transform([[-10.0], [20.0]]), [[-2.0], [3.0]])
    # Preserve the original constant-column rule, including later inputs.
    np.testing.assert_array_equal(MinMaxScaler((-2, 3)).fit([[5.0]]).transform([[99.0]]), [[-2]])


def test_mean_imputer_uses_observation_counts_not_chunk_averages():
    imputer = SimpleImputer(strategy="mean")
    assert imputer.partial_fit([[1.0, np.nan]]) is imputer
    np.testing.assert_array_equal(imputer.statistics_, [1, 0])
    imputer.transform([[np.nan, np.nan]])  # Filled output is not learned.
    imputer.partial_fit([[3.0, 12.0], [100.0, 20.0]])
    np.testing.assert_allclose(imputer.statistics_, [104 / 3, 16])
    np.testing.assert_array_equal(imputer.n_samples_seen_, [3, 2])
    np.testing.assert_allclose(imputer.transform([[np.nan, 4.0]]), [[104 / 3, 4]])
    # No variance is needed for a mean: these finite observations remain usable.
    extreme = SimpleImputer().partial_fit([[1e200], [-1e200]])
    np.testing.assert_array_equal(extreme.statistics_, [0.0])
    assert not hasattr(imputer, "_chunks")


def test_median_imputer_uses_complete_history_and_copies_inputs():
    first = np.array([[0.0, np.nan]])
    second = np.array([[100.0, 2.0], [101.0, 4.0], [102.0, np.nan], [103.0, 8.0]])
    imputer = SimpleImputer(strategy="median").partial_fit(first)
    first[:] = -999
    imputer.partial_fit(second)
    np.testing.assert_array_equal(imputer.statistics_, [101, 4])
    second[:] = -999
    imputer.partial_fit([[104.0, 10.0]])
    np.testing.assert_array_equal(imputer.statistics_, [101.5, 6])
    np.testing.assert_array_equal(imputer.transform([[np.nan, np.nan]]), [[101.5, 6]])
    np.testing.assert_array_equal(imputer.n_samples_seen_, [6, 4])


def test_constant_imputation_numeric_and_categorical():
    imputer = SimpleImputer(fill_value=-1).partial_fit([[np.nan, 3.0]])
    assert imputer.strategy == "constant"
    imputer.partial_fit([[2.0, np.nan]])
    np.testing.assert_array_equal(imputer.statistics_, [-1, -1])
    np.testing.assert_array_equal(imputer.transform([[np.nan, 6.0]]), [[-1, 6]])
    cities = np.array([["", "Sydney"], ["nan", "None"], ["Melbourne", ""]])
    saved = cities.copy()
    imputer = SimpleImputer(fill_value="Unknown").partial_fit(cities[:1])
    imputer.partial_fit(cities[1:])
    np.testing.assert_array_equal(imputer.n_samples_seen_, [1, 1])
    np.testing.assert_array_equal(imputer.transform(cities), [["Unknown", "Sydney"], ["Unknown", "Unknown"], ["Melbourne", "Unknown"]])
    np.testing.assert_array_equal(cities, saved)
    assert SimpleImputer().strategy == "mean"


def test_fit_resets_history_and_fit_transform_matches_batch(X):
    constructors = (StandardScaler, MinMaxScaler, SimpleImputer,
                    lambda: SimpleImputer(strategy="median"),
                    lambda: SimpleImputer(fill_value=-5))
    batch_constructors = (batch.StandardScaler, batch.MinMaxScaler, batch.SimpleImputer,
                          lambda: batch.SimpleImputer(strategy="median"),
                          lambda: batch.SimpleImputer(fill_value=-5))
    for cls, old_cls in zip(constructors, batch_constructors):
        with case("reset and batch compatibility", cls=cls):
            transformer = cls().partial_fit(X)
            replacement = np.array([[10.0], [20.0], [np.nan]])
            assert transformer.fit(replacement) is transformer
            np.testing.assert_array_equal(transformer.n_samples_seen_, [2])
            assert transformer.n_features_in_ == 1
            transformer.partial_fit([[30.0]])
            np.testing.assert_allclose(transformer.transform([[np.nan], [20.0]]),
                                       cls().fit([[10.0], [20.0], [np.nan], [30.0]]).transform([[np.nan], [20.0]]))
            np.testing.assert_allclose(cls().fit_transform(X), old_cls().fit_transform(X))


def test_empty_chunks_and_noncontiguous_inputs(X):
    for cls in (StandardScaler, MinMaxScaler, SimpleImputer,
                lambda: SimpleImputer(strategy="median"),
                lambda: SimpleImputer(fill_value="Unknown")):
        with case("empty chunks", cls=cls):
            transformer = cls()
            assert transformer.partial_fit(np.empty((0, 3))) is transformer
            assert not hasattr(transformer, "n_features_in_")
            with pytest.raises(RuntimeError):
                transformer.transform(np.empty((0, 3)))
            with pytest.raises(ValueError):
                transformer.fit(np.empty((0, 3)))
            transformer.partial_fit(X)
            before = transformer.n_samples_seen_.copy()
            transformer.partial_fit(np.empty((0, 3)))
            np.testing.assert_array_equal(transformer.n_samples_seen_, before)
            assert transformer.transform(np.empty((0, 3))).shape == (0, 3)
            with pytest.raises(ValueError):
                transformer.partial_fit(np.empty((0, 2)))
    view = X[::-2, ::-1]
    assert not view.flags.c_contiguous
    saved = view.copy()
    for cls in (StandardScaler, MinMaxScaler, SimpleImputer):
        with case("strides", cls=cls.__name__):
            np.testing.assert_allclose(cls().partial_fit(view).transform(view), cls().fit_transform(view.copy()))
    np.testing.assert_array_equal(view, saved)


def test_input_errors_preserve_learned_state():
    invalid = ([[1.0]], [1.0, 2.0], np.empty((2, 0)), np.zeros((1, 1, 2)),
               [[np.inf, 1.0]], [[1j, 2.0]], [["bad", "data"]])
    for cls in (StandardScaler, MinMaxScaler, SimpleImputer,
                lambda: SimpleImputer(strategy="median")):
        with case("input validation", cls=cls):
            with pytest.raises(RuntimeError):
                cls().transform([[1.0, 2.0]])
            transformer = cls().fit([[1.0, 2.0], [3.0, 4.0]])
            expected = transformer.transform([[2.0, np.nan]])
            for value in invalid:
                with case("bad data", value=value):
                    for method in (transformer.partial_fit, transformer.transform):
                        with pytest.raises((ValueError, TypeError)):
                            method(value)
                    np.testing.assert_array_equal(transformer.n_samples_seen_, [2, 2])
                    np.testing.assert_array_equal(transformer.transform([[2.0, np.nan]]), expected)
            for value in ([], [[np.inf, 1.0]], [["bad", "data"]]):
                with pytest.raises((ValueError, TypeError)):
                    transformer.fit(value)
                np.testing.assert_array_equal(transformer.transform([[2.0, np.nan]]), expected)


def test_configuration_errors_and_finite_range_limits():
    for bounds in ((1, 1), (2, 1), (0,), (0, 1, 2), (np.nan, 1),
                   (0, np.inf), (-1e308, 1e308), "bad", None):
        with case("bad range", bounds=bounds), pytest.raises((ValueError, TypeError)):
            MinMaxScaler(bounds)
    for options in ({"strategy": "mode"}, {"strategy": "mean", "fill_value": "Unknown"},
                    {"fill_value": np.nan}, {"fill_value": np.inf}):
        with case("bad imputer configuration", options=options), pytest.raises((ValueError, TypeError)):
            SimpleImputer(**options)
    scaler = MinMaxScaler().fit([[1e308], [1e308]])
    np.testing.assert_array_equal(scaler.transform([[1e308]]), [[0.0]])


def test_numeric_overflow_is_reported_without_corrupting_state():
    for cls in (StandardScaler, MinMaxScaler):
        with case("overflow", cls=cls.__name__):
            scaler = cls().fit([[1.0], [3.0]])
            expected = scaler.transform([[2.0]])
            for method in (scaler.partial_fit, scaler.fit):
                with pytest.raises(FloatingPointError):
                    method([[-1e308], [1e308]])
                np.testing.assert_array_equal(scaler.n_samples_seen_, [2])
                np.testing.assert_array_equal(scaler.transform([[2.0]]), expected)


def test_onehot_expands_sorted_categories_and_matches_batch():
    X = np.array([["Sydney", "red"], ["Sydney", "blue"], ["Adelaide", "red"],
                  ["Perth", "green"], ["", "blue"], ["Sydney", ""]])
    expected = batch.OneHotEncoder().fit_transform(X)
    for size in (1, 2, 4, 100):
        with case("onehot chunk size", size=size):
            encoder = OneHotEncoder()
            for start in range(0, len(X), size):
                assert encoder.partial_fit(X[start:start + size]) is encoder
            assert encoder.n_features_in_ == 2 and encoder.n_features_out_ == 6
            np.testing.assert_array_equal(encoder.categories_[0], ["Adelaide", "Perth", "Sydney"])
            np.testing.assert_array_equal(encoder.categories_[1], ["blue", "green", "red"])
            np.testing.assert_array_equal(encoder.transform(X), expected)
    # An alphabetically earlier category moves the existing category's column.
    encoder = OneHotEncoder().partial_fit([["Sydney"]])
    earlier_output = encoder.transform([["Sydney"]])
    encoder.partial_fit([["Adelaide"], ["Sydney"]])
    np.testing.assert_array_equal(earlier_output, [[1.0]])
    np.testing.assert_array_equal(encoder.transform([["Sydney"]]), [[0.0, 1.0]])
    encoder.partial_fit([["Sydney"]])
    assert encoder.n_features_out_ == 2  # Repeated categories do not grow state.


def test_onehot_unknown_policy_does_not_prevent_learning_new_categories():
    for policy in ("error", "ignore"):
        with case("unknown policy", policy=policy):
            encoder = OneHotEncoder(handle_unknown=policy).partial_fit([["Sydney"]])
            if policy == "error":
                with pytest.raises(ValueError, match="unknown"):
                    encoder.transform([["Adelaide"]])
            else:
                np.testing.assert_array_equal(encoder.transform([["Adelaide"]]), [[0.0]])
            np.testing.assert_array_equal(encoder.categories_[0], ["Sydney"])
            encoder.partial_fit([["Adelaide"]])
            np.testing.assert_array_equal(encoder.transform([["Adelaide"]]), [[1.0, 0.0]])
            # transform never adds an unknown category to learned state.
            assert encoder.n_features_out_ == 2
    with pytest.raises(ValueError):
        OneHotEncoder(handle_unknown="append")


def test_onehot_missing_columns_and_no_categories_recover():
    missing = np.array([["", "red"], [np.nan, ""], [None, "None"]], dtype=object)
    for policy in ("error", "ignore"):
        with case("missing categorical values", policy=policy):
            encoder = OneHotEncoder(handle_unknown=policy).partial_fit(missing)
            assert len(encoder.categories_[0]) == 0 and encoder.n_features_out_ == 1
            np.testing.assert_array_equal(encoder.transform(missing), [[1.0], [0.0], [0.0]])
            encoder.partial_fit([["Sydney", "blue"]])
            np.testing.assert_array_equal(encoder.transform(missing), [[0, 0, 1], [0, 0, 0], [0, 0, 0]])
            np.testing.assert_array_equal(encoder.transform([["Sydney", "blue"]]), [[1, 1, 0]])
    encoder = OneHotEncoder().partial_fit([[""], [None], [np.nan]])
    assert encoder.n_features_out_ == 0
    assert encoder.transform([[""], [None]]).shape == (2, 0)
    with pytest.raises(ValueError):
        encoder.transform([["new"]])
    encoder.partial_fit([["new"]])
    np.testing.assert_array_equal(encoder.transform([["new"], [""]]), [[1.0], [0.0]])


def test_onehot_fit_reset_and_string_conversion_match_original():
    encoder = OneHotEncoder().partial_fit([["Sydney", "red"]])
    encoder.partial_fit([["Perth", "blue"]])
    assert encoder.fit([[2], [10]]) is encoder  # fit can establish a new schema.
    assert encoder.n_features_in_ == 1 and encoder.n_features_out_ == 2
    np.testing.assert_array_equal(encoder.categories_[0], ["10", "2"])
    encoder.partial_fit([[3]])
    np.testing.assert_array_equal(encoder.transform([[2], [3]]), [[0, 1, 0], [0, 0, 1]])
    for X in ([[1, 2], [2, 1]], [["", "a"], ["x", "b"]]):
        with case("fit_transform compatibility", X=X):
            np.testing.assert_array_equal(encoder.fit_transform(X), batch.OneHotEncoder().fit_transform(X))


def test_onehot_empty_and_invalid_chunks_preserve_state():
    encoder = OneHotEncoder()
    assert encoder.partial_fit(np.empty((0, 2))) is encoder
    assert not hasattr(encoder, "n_features_in_")
    with pytest.raises(RuntimeError):
        encoder.transform(np.empty((0, 2)))
    with pytest.raises(ValueError):
        encoder.fit(np.empty((0, 2)))
    encoder.partial_fit([["Sydney", "red"]])
    assert encoder.partial_fit(np.empty((0, 2))) is encoder
    assert encoder.transform(np.empty((0, 2))).shape == (0, 2)
    for value in (["Sydney", "red"], np.zeros((1, 1, 2)), np.empty((2, 0)),
                  [["Sydney"]], np.empty((0, 1)), [["ragged"], ["a", "b"]]):
        with case("invalid onehot input", value=value):
            for method in (encoder.partial_fit, encoder.transform):
                with pytest.raises((ValueError, TypeError)):
                    method(value)
            assert encoder.n_features_in_ == 2 and encoder.n_features_out_ == 2
            np.testing.assert_array_equal(encoder.transform([["Sydney", "red"]]), [[1, 1]])
    for value in ([], ["Sydney"], np.empty((1, 0))):
        with case("invalid onehot fit", value=value), pytest.raises(ValueError):
            encoder.fit(value)
    assert encoder.n_features_out_ == 2


def test_onehot_multiple_features_strides_and_input_ownership():
    X = np.array([["same", "b"], ["other", "same"], ["same", "same"], ["", "b"]])
    view = X[::-1, ::-1]
    saved = view.copy()
    encoder = OneHotEncoder().partial_fit(view[:2]).partial_fit(view[2:])
    np.testing.assert_array_equal(encoder.transform(view), batch.OneHotEncoder().fit_transform(view))
    np.testing.assert_array_equal(view, saved)
    view[:] = "changed"
    # Categories own their data, and matching labels in two features stay separate.
    np.testing.assert_array_equal(encoder.categories_[0], ["b", "same"])
    np.testing.assert_array_equal(encoder.categories_[1], ["other", "same"])
    np.testing.assert_array_equal(encoder.transform([["same", "same"]]), [[0, 1, 0, 1]])
