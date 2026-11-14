from itertools import count
import numpy as np
import pytest
from numcompute.utils import (
    check_array,
    iter_batches,
    iter_line_batches,
    logsumexp,
    pairwise_distances,
    sigmoid,
    softmax,
)


def test_distance_mixed_scales_tiny_values_and_large_finite_norms():
    from numcompute.utils import pairwise_distances

    with np.errstate(all="raise"):
        for magnitude in (1e9, 1e150, 1e300):
            X = np.array([[0., 0.], [magnitude, 0.], [magnitude, 1.]])
            assert pairwise_distances(X)[1, 2] == 1.0
            assert pairwise_distances(X[:2], X[2:])[1, 0] == 1.0
            np.testing.assert_array_equal(pairwise_distances(X[1:, ::-1]), [[0, 1], [1, 0]])
        assert pairwise_distances([[0.]], [[1e-200]])[0, 0] == 1e-200
        np.testing.assert_allclose(pairwise_distances([[0., 0.]], [[1e200, 1e200]]),
                                   [[np.sqrt(2.) * 1e200]])
    with pytest.raises(TypeError, match="real"):
        pairwise_distances(np.array([[1 + 2j]]))


def test_logsumexp_and_softmax_mixed_infinite_slices_under_strict_floats():
    from numcompute.utils import logsumexp, sigmoid, softmax

    X = np.array([[np.inf, 1000.], [-np.inf, -np.inf], [np.nan, 1000.],
                  [1000., -1000.], [1e308, -1e308]])
    with np.errstate(all="raise"):
        expected = [np.inf, -np.inf, np.nan, 1000., 1e308]
        np.testing.assert_array_equal(logsumexp(X, axis=1), expected)
        np.testing.assert_array_equal(logsumexp(X.T, axis=0), expected)
        assert logsumexp([np.inf, 1000.]) == np.inf
        assert logsumexp([-np.inf, -np.inf]) == -np.inf
        assert np.isnan(logsumexp([np.nan, 1000.]))
        probabilities = softmax(X, axis=1)
        assert np.isnan(probabilities[:3]).all()
        np.testing.assert_array_equal(probabilities[3:], [[1, 0], [1, 0]])
        np.testing.assert_array_equal(softmax(X.T, axis=0), probabilities.T)
        np.testing.assert_array_equal(sigmoid([-np.inf, -1000., 1000., np.inf]), [0, 0, 1, 1])


def test_check_array_converts_to_float64_without_copying_float64():
    assert check_array([[1, 2], [3, 4]]).dtype == np.float64
    X = np.ones((3, 2))
    assert check_array(X) is X


def test_check_array_errors():
    with pytest.raises(ValueError, match="2-D"):
        check_array([1.0, 2.0], ndim=2)
    with pytest.raises(ValueError, match="empty"):
        check_array(np.zeros((0, 3)))
    with pytest.raises(TypeError):
        check_array([["a"]])


def test_iter_batches_covers_all_rows_as_views():
    X = np.arange(10).reshape(10, 1)
    batches = list(iter_batches(X, 4))
    assert [b.shape[0] for b in batches] == [4, 4, 2]
    assert np.shares_memory(batches[0], X)
    np.testing.assert_array_equal(np.vstack(batches), X)
    with pytest.raises(ValueError):
        list(iter_batches(X, 0))


# ---- Jose: iterating line batches ----

def test_iter_line_batches_sizes_and_empty_input():
    """Full batches first, a shorter last one, nothing for an empty input."""
    assert list(iter_line_batches(range(5), 2)) == [[0, 1], [2, 3], [4]]
    assert list(iter_line_batches([], 3)) == []


def test_iter_line_batches_is_lazy_on_infinite_iterators():
    """Only one batch is read at a time, so an endless source still works."""
    batches = iter_line_batches(count(), 3)

    assert next(batches) == [0, 1, 2]
    assert next(batches) == [3, 4, 5]  # continues where the last batch stopped


@pytest.mark.parametrize("batch_size, error", [(0, ValueError), (-2, ValueError), (2.5, TypeError), ("3", TypeError)])
def test_iter_line_batches_rejects_bad_batch_size(batch_size, error):
    """batch_size must be an integer >= 1."""
    with pytest.raises(error, match="batch_size"):
        list(iter_line_batches(range(5), batch_size))


# ---- Jose: numerically stable activations and distances ----

def test_logsumexp_matches_naive_formula_on_small_values():
    x = np.array([[0.5, -1.0, 2.0], [3.0, 0.0, -2.5]])
    assert logsumexp(x) == pytest.approx(np.log(np.exp(x).sum()))
    np.testing.assert_allclose(logsumexp(x, axis=1), np.log(np.exp(x).sum(axis=1)))
    assert logsumexp(x, axis=0).shape == (3,)


def test_logsumexp_no_overflow_or_underflow():
    """The naive formula gives inf for large values and -inf for very negative ones."""
    with np.errstate(all="raise"):
        assert logsumexp([1000.0, 1000.0]) == pytest.approx(1000.0 + np.log(2.0))
        assert logsumexp([-1000.0, -1000.0]) == pytest.approx(-1000.0 + np.log(2.0))


def test_logsumexp_infinite_and_nan_slices():
    assert logsumexp([-np.inf, -np.inf]) == -np.inf
    assert logsumexp([1.0, np.inf]) == np.inf
    np.testing.assert_array_equal(logsumexp([[-np.inf, -np.inf], [0.0, 0.0]], axis=1), [-np.inf, np.log(2.0)])
    assert np.isnan(logsumexp([1.0, np.nan]))


def test_logsumexp_and_softmax_reject_empty_and_text():
    with pytest.raises(ValueError, match="empty"):
        logsumexp([])
    with pytest.raises(ValueError, match="empty"):
        softmax(np.zeros((0, 3)))
    with pytest.raises(TypeError):
        softmax(["a", "b"])


def test_softmax_rows_sum_to_one_and_match_logsumexp():
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = softmax(x)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_allclose(p, np.exp(x - logsumexp(x, axis=1)[:, None]))
    np.testing.assert_allclose(p[1], 1 / 3)               # equal scores, equal probabilities


def test_softmax_is_shift_invariant_and_does_not_overflow():
    x = np.array([1.0, 2.0, 3.0])
    with np.errstate(over="raise", invalid="raise"):     # underflow to 0 is fine
        np.testing.assert_allclose(softmax(x + 1000.0), softmax(x))
        np.testing.assert_allclose(softmax([1000.0, 0.0]), [1.0, 0.0], atol=1e-300)


def test_softmax_axis_zero_and_non_contiguous_view():
    x = np.arange(12.0).reshape(3, 4)
    np.testing.assert_allclose(softmax(x, axis=0).sum(axis=0), 1.0)
    view = x[:, ::2]
    assert not view.flags.c_contiguous
    np.testing.assert_allclose(softmax(view), softmax(view.copy()))


def test_sigmoid_values_and_symmetry():
    x = np.linspace(-20, 20, 41)
    np.testing.assert_allclose(sigmoid(x), 1 / (1 + np.exp(-x)))
    np.testing.assert_allclose(sigmoid(-x), 1 - sigmoid(x))
    assert sigmoid(0) == 0.5
    assert isinstance(sigmoid(1.0), float)


def test_sigmoid_extreme_inputs_without_overflow_and_nan_propagates():
    with np.errstate(over="raise", invalid="raise"):     # underflow to 0 is fine
        np.testing.assert_array_equal(sigmoid([-1000.0, 1000.0]), [0.0, 1.0])
    out = sigmoid([np.nan, 0.0])
    assert np.isnan(out[0]) and out[1] == 0.5


def test_pairwise_distances_matches_direct_differences():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(5, 3)), rng.normal(size=(4, 3))
    direct = np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))
    D = pairwise_distances(X, Y)
    assert D.shape == (5, 4)
    np.testing.assert_allclose(D, direct)


def test_pairwise_distances_self_is_symmetric_with_zero_diagonal():
    X = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])      # duplicate rows
    D = pairwise_distances(X)
    np.testing.assert_array_equal(np.diag(D), 0.0)
    np.testing.assert_allclose(D, D.T)
    assert D[0, 1] == pytest.approx(5.0) and D[1, 2] == 0.0


def test_pairwise_distances_far_from_origin():
    """Centring keeps unit distances accurate for points near 1e8."""
    X = np.array([[1e8, 1e8], [1e8 + 1.0, 1e8]])
    np.testing.assert_allclose(pairwise_distances(X), [[0.0, 1.0], [1.0, 0.0]], atol=1e-6)


def test_pairwise_distances_errors():
    with pytest.raises(ValueError, match="same number of features"):
        pairwise_distances(np.zeros((2, 3)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="2-D"):
        pairwise_distances([1.0, 2.0])
    with pytest.raises(ValueError, match="empty"):
        pairwise_distances(np.zeros((0, 2)))
