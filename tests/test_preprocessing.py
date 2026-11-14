import numpy as np
import pytest
from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler


def test_minmax_finite_bounds_and_extreme_finite_scaling_regressions():
    from numcompute.preprocessing import MinMaxScaler, SimpleImputer

    tiny = np.nextafter(0., 1.)
    for bounds in ((np.nan, 1), (0, np.nan), (-np.inf, 1), (0, np.inf),
                   (-1e308, 1e308), np.array([0 + 1j, 1 + 2j])):
        with np.errstate(all="raise"), pytest.raises(ValueError):
            MinMaxScaler(bounds)
    with np.errstate(all="raise"):
        assert MinMaxScaler().fit([[-1e308], [0.]]).transform([[1e308]])[0, 0] == 2.
        narrow = MinMaxScaler((0, 1e-308)).fit([[0.], [1e-308]])
        np.testing.assert_allclose(narrow.transform([[1e308]]), [[1e308]])
        wide = MinMaxScaler((-1e308, 0)).fit([[0.], [1.]])
        np.testing.assert_allclose(wide.transform([[2.]]), [[1e308]])
        large_range = MinMaxScaler((0, 1e308)).fit([[0.], [1e308]])
        assert large_range.transform([[1e-308]])[0, 0] == pytest.approx(1e-308, rel=1e-14, abs=0)
        for value in (tiny, 1e308):
            filled = SimpleImputer(strategy="median").fit_transform([[value], [value], [np.nan]])
            np.testing.assert_array_equal(filled, np.full((3, 1), value))
    scaler = MinMaxScaler().fit([[0], [1]])
    for X in ([[-1e308], [1e308]], [[np.inf]]):
        with pytest.raises((FloatingPointError, ValueError)):
            scaler.fit(X)
        np.testing.assert_array_equal(scaler.data_min_, [0])
        np.testing.assert_array_equal(scaler.data_range_, [1])


@pytest.fixture
def X():
    return np.random.default_rng(42).normal(5.0, 3.0, size=(50, 4))


# StandardScaler

def test_standard_scaler_zero_mean_unit_std(X):
    Z = StandardScaler().fit_transform(X)
    assert Z.shape == X.shape
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.std(axis=0), 1.0, atol=1e-12)


def test_standard_scaler_constant_column_gives_zero_not_nan():
    Z = StandardScaler().fit_transform([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    assert np.isfinite(Z).all()
    np.testing.assert_array_equal(Z[:, 1], 0.0)


def test_standard_scaler_ignores_nan_when_fitting_and_propagates_it():
    X = np.array([[1.0, 10.0], [np.nan, 20.0], [3.0, 30.0]])
    sc = StandardScaler().fit(X)
    np.testing.assert_allclose(sc.mean_, [2.0, 20.0])
    Z = sc.transform(X)
    assert np.isnan(Z[1, 0]) and np.isfinite(Z[:, 1]).all()


def test_standard_scaler_errors(X):
    with pytest.raises(ValueError, match="reshape"):
        StandardScaler().fit(X[:, 0])
    with pytest.raises(ValueError, match="features"):
        StandardScaler().fit(X).transform(X[:, :2])
    with pytest.raises(RuntimeError, match="not fitted"):
        StandardScaler().transform(X)


def test_standard_scaler_non_contiguous_input(X):
    view = X[::2]
    np.testing.assert_allclose(StandardScaler().fit_transform(view),
                               StandardScaler().fit_transform(np.ascontiguousarray(view)))


# MinMaxScaler

def test_minmax_default_and_custom_range(X):
    Z = MinMaxScaler().fit_transform(X)
    np.testing.assert_allclose(Z.min(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z.max(axis=0), 1.0, atol=1e-12)
    Z = MinMaxScaler((-1, 1)).fit_transform(X)
    np.testing.assert_allclose(Z.min(axis=0), -1.0, atol=1e-12)
    np.testing.assert_allclose(Z.max(axis=0), 1.0, atol=1e-12)


def test_minmax_constant_column_maps_to_range_min():
    Z = MinMaxScaler((2, 5)).fit_transform([[4.0, 1.0], [4.0, 2.0], [4.0, 3.0]])
    np.testing.assert_array_equal(Z[:, 0], 2.0)


def test_minmax_invalid_range_rejected():
    with pytest.raises(ValueError):
        MinMaxScaler(feature_range=(1, 1))


def test_minmax_unseen_values_extrapolate():
    sc = MinMaxScaler().fit([[0.0], [10.0]])
    np.testing.assert_allclose(sc.transform([[20.0], [-5.0]]), [[2.0], [-0.5]])


# SimpleImputer

def test_imputer_strategies():
    X = np.array([[1.0, 100.0], [np.nan, 1.0], [3.0, np.nan], [5.0, 2.0]])
    np.testing.assert_allclose(SimpleImputer("mean").fit(X).statistics_, [3.0, 103 / 3])
    np.testing.assert_allclose(SimpleImputer("median").fit(X).statistics_, [3.0, 2.0])
    np.testing.assert_allclose(SimpleImputer("constant", -1).fit(X).statistics_, [-1, -1])
    assert not np.isnan(SimpleImputer().fit_transform(X)).any()


def test_imputer_all_nan_column_uses_fill_value():
    imp = SimpleImputer("mean", fill_value=9.0).fit([[np.nan, 1.0], [np.nan, 2.0]])
    np.testing.assert_allclose(imp.statistics_, [9.0, 1.5])


def test_imputer_does_not_modify_input_and_rejects_bad_strategy():
    X = np.array([[np.nan, 1.0], [2.0, 3.0]])
    SimpleImputer().fit_transform(X)
    assert np.isnan(X[0, 0])
    with pytest.raises(ValueError):
        SimpleImputer(strategy="mode")


def test_imputer_fill_value_alone_implies_constant():
    X = np.array([[1.0, np.nan], [np.nan, 4.0]])
    imp = SimpleImputer(fill_value=0).fit(X)
    assert imp.strategy == "constant"
    np.testing.assert_array_equal(imp.fit_transform(X), [[1.0, 0.0], [0.0, 4.0]])
    assert SimpleImputer().strategy == "mean"


def test_imputer_string_fill_value_for_categorical_column():
    X = np.array([["Perth"], ["nan"], [""], ["Adelaide"]])
    out = SimpleImputer(fill_value="Unknown").fit_transform(X)
    assert out[:, 0].tolist() == ["Perth", "Unknown", "Unknown", "Adelaide"]
    assert OneHotEncoder().fit_transform(out).shape == (4, 3)
    with pytest.raises(ValueError):
        SimpleImputer("mean", fill_value="Unknown")


# OneHotEncoder

def test_onehot_string_categories_sorted_and_encoded():
    X = np.array([["red", "S"], ["blue", "M"], ["red", "M"]])
    enc = OneHotEncoder().fit(X)
    assert [c.tolist() for c in enc.categories_] == [["blue", "red"], ["M", "S"]]
    np.testing.assert_array_equal(enc.transform(X), [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]])


def test_onehot_numeric_categories_one_hot_per_column():
    Z = OneHotEncoder().fit_transform(np.array([[1, 10], [2, 10], [3, 20], [1, 20]]))
    assert Z.shape == (4, 5)
    np.testing.assert_array_equal(Z[:, :3].sum(axis=1), 1)
    np.testing.assert_array_equal(Z[:, 3:].sum(axis=1), 1)


def test_onehot_unknown_category_error_or_ignore():
    train, test = np.array([["a"], ["b"]]), np.array([["a"], ["zzz"]])
    with pytest.raises(ValueError, match="unknown"):
        OneHotEncoder().fit(train).transform(test)
    np.testing.assert_array_equal(OneHotEncoder("ignore").fit(train).transform(test), [[1, 0], [0, 0]])


def test_onehot_nan_and_blank_encode_to_zeros_and_1d_rejected():
    X = np.array([[1.0], [np.nan], [2.0]])
    np.testing.assert_array_equal(OneHotEncoder().fit_transform(X), [[1, 0], [0, 0], [0, 1]])
    np.testing.assert_array_equal(OneHotEncoder().fit_transform(np.array([["a"], [""]])), [[1], [0]])
    with pytest.raises(ValueError):
        OneHotEncoder().fit(np.array(["a", "b"]))
