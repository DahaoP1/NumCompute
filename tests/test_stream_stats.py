"""Key streaming-statistics tests, grouped by behaviour.

Related input variants run inside each test; all original scenarios are kept.
The _case helper names the scenario and parameters if a grouped check fails.
"""

from contextlib import contextmanager
import warnings

import numpy as np
import pytest

from numcompute import stats as batch_stats
from numcompute_stream import stats
from numcompute_stream.stats import RunningStats, update_stats


def test_all_queries_match_independent_prefix_values_and_reused_batch_functions():
    # Exact expectations, rather than comparing two copies of the same formula.
    X = np.array([[1., np.nan], [3., 10.], [5., 20.], [7., 30.]])
    running = update_stats(X[:1])
    running.partial_fit(X[1:3]).update_stats(X[3:])
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(running.count_, [4, 3])
        np.testing.assert_array_equal(running.mean, [4, 20])
        np.testing.assert_allclose(running.m2, [20, 200], rtol=1e-14)
        np.testing.assert_allclose(running.var(), [5, 200 / 3])
        np.testing.assert_allclose(running.var(ddof=1), [20 / 3, 100], rtol=1e-14)
        np.testing.assert_allclose(running.std(), np.sqrt([5, 200 / 3]))
        np.testing.assert_array_equal(running.min, [1, 10])
        np.testing.assert_array_equal(running.max, [7, 30])
        np.testing.assert_array_equal(running.median(), [4, 20])
        np.testing.assert_array_equal(running.quantile([0, .5, 1]), [[1, 10], [4, 20], [7, 30]])
        np.testing.assert_array_equal(running.histogram(bins=[0, 10, 20, 30])[0], [4, 1, 2])
    for name in ("mean", "median", "std", "min", "max", "quantile", "histogram"):
        assert getattr(stats, name) is getattr(batch_stats, name)
    # Every reused function is executed, including both explicit axis choices.
    for axis in (0, None):
        for name, numpy_name in (("mean", "nanmean"), ("median", "nanmedian"),
                                 ("std", "nanstd"), ("min", "nanmin"), ("max", "nanmax")):
            np.testing.assert_allclose(getattr(stats, name)(X, axis=axis, skip_nan=True),
                                       getattr(np, numpy_name)(X, axis=axis))
        np.testing.assert_allclose(stats.quantile(X, [.25, .75], axis=axis, skip_nan=True),
                                   np.nanquantile(X, [.25, .75], axis=axis))
    tiny = np.nextafter(0., 1.)
    with np.errstate(all="raise"):
        for value in (tiny, 1e308):
            constant = RunningStats(store_data=False).update([value]).update([value])
            np.testing.assert_array_equal(constant.mean, [value])
            np.testing.assert_array_equal(constant.var(), [0])


@contextmanager
def _case(name, **parameters):
    """Keep the original exception traceback and add the failing scenario."""
    try:
        yield
    except (Exception, pytest.fail.Exception) as error:
        raise AssertionError(f"{name}, {parameters}: {error}") from error


@pytest.fixture
def X():
    return np.random.default_rng(7).normal(size=(103, 3))


@pytest.fixture
def X_nan(X):
    result = X.copy()
    result[[0, 9, 51], [0, 1, 2]] = np.nan
    return result


def test_batch_reuse_and_streaming_update_interfaces():
    with _case("batch_helpers_are_reused_from_original_package"):
        for name in ("mean", "median", "std", "min", "max", "quantile", "histogram"):
            assert getattr(stats, name) is getattr(batch_stats, name)

    with _case("scalar_and_1d_chunks_both_represent_one_feature"):
        running = RunningStats().update(2).update([4, 6])
        assert running.n == 3 and running.n_features_in_ == 1
        np.testing.assert_array_equal(running.count_, [3])
        np.testing.assert_allclose(running.mean, [4])
        assert running.quantile(0.5).shape == (1,)

    with _case("partial_fit_and_update_stats_methods_share_the_same_state"):
        running = RunningStats()
        assert running.partial_fit([[1, 2]]) is running
        assert running.update_stats([[3, 4]]) is running
        np.testing.assert_allclose(running.mean, [2, 3])

    with _case("module_update_stats_keeps_explicit_caller_owned_state"):
        running = update_stats([1, 2])
        assert update_stats([3, 4], running) is running
        np.testing.assert_allclose(running.mean, [2.5])
        independent = update_stats([100])
        np.testing.assert_allclose(independent.mean, [100])
        assert running.n == 4
        with pytest.raises(TypeError, match="RunningStats"):
            update_stats([1], stats=object())


def test_chunked_moments_and_weighted_updates(X):
    for chunk_size in [1, 7, 32, 200]:
        with _case("chunked_moments_match_numpy_with_unequal_last_chunk", chunk_size=chunk_size):
            running = RunningStats()
            for start in range(0, len(X), chunk_size):
                assert running.update(X[start:start + chunk_size]) is running
            assert running.n == len(X)
            assert running.n_features_in_ == 3
            np.testing.assert_array_equal(running.count_, [103, 103, 103])
            np.testing.assert_allclose(running.mean, X.mean(axis=0), atol=1e-15)
            np.testing.assert_allclose(running.var(), X.var(axis=0))
            np.testing.assert_allclose(running.var(ddof=1), X.var(axis=0, ddof=1))
            np.testing.assert_allclose(running.std(ddof=1), X.std(axis=0, ddof=1))
            np.testing.assert_allclose(running.m2, ((X - X.mean(axis=0)) ** 2).sum(axis=0))
            np.testing.assert_array_equal(running.min, X.min(axis=0))
            np.testing.assert_array_equal(running.max, X.max(axis=0))
            assert running.mean.shape == running.var().shape == (3,)

    with _case("unequal_chunk_means_are_weighted_by_observation_count"):
        running = RunningStats().update([0, 0, 0]).update([10])
        np.testing.assert_allclose(running.mean, [2.5])
        np.testing.assert_allclose(running.var(), [18.75])


def test_per_feature_nan_omission_and_counts(X_nan):
    with _case("nan_omission_keeps_other_features_and_uses_per_column_counts"):
        running = RunningStats().update([[1, np.nan], [np.nan, 4]])
        running.update([[3, 6], [np.nan, 8]])
        assert running.n == 4
        np.testing.assert_array_equal(running.count_, [2, 3])
        np.testing.assert_allclose(running.mean, [2, 6])
        np.testing.assert_allclose(running.var(), [1, 8 / 3])
        np.testing.assert_allclose(running.min, [1, 4])
        np.testing.assert_allclose(running.max, [3, 8])

    with _case("nan_omission_matches_numpy_over_multiple_chunks"):
        running = RunningStats().update(X_nan[:20]).update(X_nan[20:])
        np.testing.assert_array_equal(running.count_, np.sum(~np.isnan(X_nan), axis=0))
        np.testing.assert_allclose(running.mean, np.nanmean(X_nan, axis=0))
        np.testing.assert_allclose(running.var(ddof=1), np.nanvar(X_nan, axis=0, ddof=1))


def test_nan_propagation_and_unaffected_features(X_nan):
    with _case("nan_propagation_persists_but_does_not_poison_other_features"):
        running = RunningStats(skip_nan=False).update([[1, 2], [np.nan, 4]])
        running.update([[3, 6]])
        for result in (running.mean, running.var(), running.std(), running.m2, running.min, running.max):
            assert np.isnan(result[0]) and np.isfinite(result[1])
        np.testing.assert_allclose(running.mean[1], 4)
        assert np.isnan(running.quantile(0.5)[0])
        assert np.isnan(running.median(axis=None))
        # Histograms omit NaNs regardless of the moments policy.
        counts, _ = running.histogram(bins=[0, 2, 4], feature=0)
        np.testing.assert_array_equal(counts, [1, 1])

    with _case("quantile_nan_propagation_matches_numpy"):
        running = RunningStats(skip_nan=False).update(X_nan[:50]).update(X_nan[50:])
        np.testing.assert_allclose(running.quantile([0, 0.5, 1]),
                                   np.quantile(X_nan, [0, 0.5, 1], axis=0), equal_nan=True)


def test_all_nan_chunks_and_recovery():
    with _case("all_nan_column_can_learn_from_later_valid_values_without_warnings"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            running = RunningStats().update([[np.nan, 1], [np.nan, 3]])
            np.testing.assert_allclose(running.mean, [np.nan, 2], equal_nan=True)
            np.testing.assert_allclose(running.var(), [np.nan, 1], equal_nan=True)
            np.testing.assert_allclose(running.quantile([0, 0.5, 1])[:, 0], np.nan)
            running.update([[5, np.nan], [7, np.nan]])
        np.testing.assert_allclose(running.mean, [6, 2])
        np.testing.assert_allclose(running.var(), [1, 1])
        np.testing.assert_array_equal(running.count_, [2, 2])

    for skip_nan in [True, False]:
        with _case("entirely_nan_chunks_have_defined_query_results", skip_nan=skip_nan):
            running = RunningStats(skip_nan=skip_nan).update(np.full((4, 2), np.nan))
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                for result in (running.mean, running.var(), running.min, running.max, running.median()):
                    assert np.isnan(result).all()
                assert np.isnan(running.quantile(0.5, axis=None))
                counts, edges = running.histogram(bins=4)
                density, _ = running.histogram(bins=4, density=True)
            np.testing.assert_array_equal(counts, np.zeros(4, dtype=int))
            np.testing.assert_array_equal(density, np.zeros(4))
            np.testing.assert_allclose(edges, np.linspace(0, 1, 5))
            assert running.n == 4

    with _case("missing_only_later_chunk_does_not_change_observed_moments"):
        running = RunningStats().update([[1, 2], [3, 4]])
        running.update(np.full((10, 2), np.nan))
        np.testing.assert_allclose(running.mean, [2, 3])
        np.testing.assert_allclose(running.var(), [1, 1])
        np.testing.assert_array_equal(running.count_, [2, 2])
        assert running.n == 12


def test_constant_columns_and_degrees_of_freedom():
    with _case("all_equal_chunks_have_zero_variance"):
        running = RunningStats().update(np.full((5, 3), 7.0)).update(np.full((8, 3), 7.0))
        np.testing.assert_array_equal(running.mean, [7, 7, 7])
        np.testing.assert_array_equal(running.var(), [0, 0, 0])
        np.testing.assert_array_equal(running.std(ddof=1), [0, 0, 0])

    with _case("insufficient_per_feature_sample_counts_return_nan"):
        running = RunningStats().update([[1, np.nan], [3, 5]])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            np.testing.assert_allclose(running.var(ddof=1), [2, np.nan], equal_nan=True)
            assert np.isnan(running.std(ddof=2)).all()

    for ddof, error in [(-1, ValueError), (0.5, TypeError), (True, TypeError)]:
        with _case("invalid_ddof_rejected", ddof=ddof, error=error):
            with pytest.raises(error, match="ddof"):
                RunningStats().update([1, 2]).var(ddof)


def test_numerical_stability_and_atomic_overflow():
    with _case("large_offset_small_spread_remains_accurate"):
        X = 1e12 + np.random.default_rng(19).normal(size=(2000, 2))
        running = RunningStats(store_data=False)
        for chunk in np.array_split(X, 31):
            running.update(chunk)
        np.testing.assert_allclose(running.mean, X.mean(axis=0), rtol=0, atol=0.002)
        np.testing.assert_allclose(running.var(), X.var(axis=0), rtol=3e-5)

    with _case("huge_constant_does_not_overflow_its_mean"):
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            running = RunningStats(store_data=False).update([1e308, 1e308]).update([1e308])
        np.testing.assert_allclose(running.mean, [1e308])
        np.testing.assert_array_equal(running.var(), [0])

    with _case("welford_correction_does_not_square_large_delta_too_early"):
        running = RunningStats().update([0]).update([1.5e154])
        np.testing.assert_allclose(running.mean, [7.5e153])
        np.testing.assert_allclose(running.var(), [5.625e307])

    with _case("overflow_rejects_update_without_changing_state_or_history"):
        running = RunningStats().update([[1, 2], [3, 4]])
        before = running.mean.copy()
        with pytest.raises(FloatingPointError, match="rescale"):
            running.update([[-1e308, 0], [1e308, 0]])
        assert running.n == 2 and len(running._chunks) == 1
        np.testing.assert_array_equal(running.mean, before)
        np.testing.assert_allclose(running.median(), [2, 3])


def test_exact_quantiles_methods_and_shapes(X_nan):
    with _case("exact_quantiles_use_full_history_not_mean_of_chunk_quantiles"):
        running = RunningStats().update([0, 1, 2, 3, 4]).update([100])
        np.testing.assert_allclose(running.quantile([0, 0.5, 1]), [[0], [2.5], [100]])
        np.testing.assert_allclose(running.median(), [2.5])

    for method in ['linear', 'lower', 'higher', 'midpoint', 'nearest']:
        for axis in [0, None]:
            with _case("quantile_methods_and_shapes_match_numpy", method=method, axis=axis):
                running = RunningStats().update(X_nan[:50]).update(X_nan[50:])
                q = [0.0, 0.25, 0.5, 0.9, 1.0]
                np.testing.assert_allclose(running.quantile(q, axis=axis, method=method),
                                           np.nanquantile(X_nan, q, axis=axis, method=method))
                scalar = running.quantile(0.5, axis=axis, method=method)
                assert isinstance(scalar, float) if axis is None else scalar.shape == (3,)
                assert running.quantile(q, axis=axis).shape == ((5,) if axis is None else (5, 3))


def test_histogram_counts_edges_and_features(X_nan):
    with _case("histogram_with_fixed_edges_accumulates_counts_and_includes_last_edge"):
        running = RunningStats().update([-1, 0, 1, np.nan]).update([2, 3, 4, 5])
        counts, edges = running.histogram(bins=[0, 2, 4])
        np.testing.assert_array_equal(counts, [2, 3])
        np.testing.assert_array_equal(edges, [0, 2, 4])

    with _case("automatic_histogram_uses_whole_history_when_range_expands"):
        running = RunningStats().update([0, 1]).update([10, 20])
        counts, edges = running.histogram(bins=2)
        expected_counts, expected_edges = np.histogram([0, 1, 10, 20], bins=2)
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_array_equal(edges, expected_edges)

    for feature in [0, 1, 2, None]:
        with _case("histogram_feature_selection_matches_numpy", feature=feature):
            running = RunningStats().update(X_nan[:30]).update(X_nan[30:])
            values = X_nan.ravel() if feature is None else X_nan[:, feature]
            values = values[~np.isnan(values)]
            actual = running.histogram(bins=7, feature=feature)
            expected = np.histogram(values, bins=7)
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_allclose(actual[1], expected[1])

    with _case("histogram_constant_values_are_counted"):
        counts, edges = RunningStats().update([5, 5]).update([5]).histogram(bins=3)
        assert counts.sum() == 3
        assert edges[0] < 5 < edges[-1]


def test_histogram_density_and_extreme_values():
    with _case("histogram_density_normalises_only_in_range_observations"):
        running = RunningStats().update([-100, 0, 1, 2, 3, 4, 100])
        density, edges = running.histogram(bins=2, range=(0, 4), density=True)
        expected, _ = np.histogram([0, 1, 2, 3, 4], bins=2, range=(0, 4), density=True)
        np.testing.assert_allclose(density, expected)
        assert np.sum(density * np.diff(edges)) == pytest.approx(1)

    with _case("histogram_zero_in_range_counts_produce_zero_density"):
        density, edges = RunningStats().update([10, 20]).histogram(bins=[0, 1, 2], density=True)
        np.testing.assert_array_equal(density, [0, 0])
        np.testing.assert_array_equal(edges, [0, 1, 2])

    with _case("histogram_requires_representable_distinct_edges_for_huge_constant"):
        running = RunningStats().update([1e308, 1e308])
        with pytest.raises(ValueError, match="bins|edges"):
            running.histogram(bins=3)
        low, high = np.nextafter(1e308, 0), np.nextafter(1e308, np.inf)
        counts, _ = running.histogram(bins=[low, 1e308, high])
        assert counts.sum() == 2


def test_history_storage_and_input_ownership(X):
    with _case("moments_only_mode_keeps_no_history_and_blocks_distribution_queries"):
        running = RunningStats(store_data=False).update(X[:50]).update(X[50:])
        np.testing.assert_allclose(running.mean, X.mean(axis=0))
        assert running._chunks == []
        for query in (lambda: running.quantile(0.5), running.median, running.histogram):
            with pytest.raises(RuntimeError, match="store_data"):
                query()

    with _case("non_contiguous_chunks_work_and_stored_history_copies_callers"):
        base = np.arange(40.0).reshape(10, 4)
        view = base[::2, ::2]
        assert not view.flags.c_contiguous
        expected = view.copy()
        running = RunningStats().update(view)
        np.testing.assert_array_equal(view, expected)
        base[:] = -99
        np.testing.assert_allclose(running.mean, expected.mean(axis=0))
        np.testing.assert_allclose(running.median(), np.median(expected, axis=0))

    with _case("returned_properties_and_query_arrays_do_not_expose_state"):
        running = RunningStats().update([[1, 2], [3, 4]])
        for values in (running.count_, running.mean, running.m2, running.min, running.max,
                       running.var(), running.quantile(0.5), running.histogram()[0]):
            values[:] = 999
        np.testing.assert_array_equal(running.count_, [2, 2])
        np.testing.assert_allclose(running.mean, [2, 3])
        np.testing.assert_allclose(running.median(), [2, 3])


def test_empty_chunks_reset_and_queries_before_data():
    with _case("empty_chunks_are_no_ops_but_schema_is_checked"):
        running = RunningStats()
        assert running.update([]) is running
        assert running.n == 0 and running.n_features_in_ is None
        running.update([[1, 2]])
        assert running.update(np.empty((0, 2))) is running
        assert running.n == 1
        with pytest.raises(ValueError, match="features"):
            running.update(np.empty((0, 3)))

    with _case("reset_clears_state_and_allows_new_feature_count"):
        running = RunningStats(skip_nan=False, store_data=False).update([[1, 2]])
        assert running.reset() is running
        assert running.n == 0 and running.n_features_in_ is None and running._chunks == []
        assert not running.skip_nan and not running.store_data
        with pytest.raises(RuntimeError, match="no data"):
            running.var()
        running.update([10, 20])
        np.testing.assert_allclose(running.mean, [15])

    for query in [
        lambda s: s.mean,
        lambda s: s.count_,
        lambda s: s.m2,
        lambda s: s.min,
        lambda s: s.max,
        lambda s: s.var(),
        lambda s: s.std(),
        lambda s: s.median(),
        lambda s: s.quantile(0.5),
        lambda s: s.histogram(),
    ]:
        with _case("queries_before_first_rows_raise", query=query):
            with pytest.raises(RuntimeError, match="no data"):
                query(RunningStats())


def test_constructor_and_update_validation_preserve_state():
    for options in [{'skip_nan': 1}, {'store_data': 'yes'}]:
        with _case("constructor_rejects_non_boolean_options", options=options):
            with pytest.raises(TypeError, match="boolean"):
                RunningStats(**options)

    for X, error in [
        ([[1, 2, 3]], ValueError),
        (np.empty((2, 0)), ValueError),
        (np.zeros((1, 2, 1)), ValueError),
        ([[np.inf, 0]], ValueError),
        ([['1', '2']], TypeError),
        ([[1j, 0]], TypeError),
    ]:
        with _case("invalid_updates_preserve_state_and_history", X=X, error=error):
            running = RunningStats().update([[1, 2], [3, 4]])
            with pytest.raises(error):
                running.update(X)
            assert running.n == 2 and len(running._chunks) == 1
            np.testing.assert_array_equal(running.count_, [2, 2])
            np.testing.assert_allclose(running.mean, [2, 3])


def test_quantile_query_validation():
    for q in [[], [[0.5]], -0.1, 1.1, np.nan, np.inf]:
        with _case("invalid_quantile_probabilities_rejected", q=q):
            with pytest.raises(ValueError, match="q"):
                RunningStats().update([1, 2]).quantile(q)

    with _case("complex_quantile_probability_rejected"):
        with pytest.raises(TypeError, match="real"):
            RunningStats().update([1, 2]).quantile(0.5 + 1j)

    for axis in [1, -1, True, 'rows']:
        with _case("stream_quantiles_only_allow_explicit_supported_axes", axis=axis):
            with pytest.raises(ValueError, match="axis"):
                RunningStats().update([[1, 2]]).quantile(0.5, axis=axis)

    with _case("unknown_quantile_method_rejected"):
        with pytest.raises(ValueError, match="method"):
            RunningStats().update([1, 2]).quantile(0.5, method="cubic")


def test_histogram_query_validation():
    for options, error in [
        ({'bins': 0}, ValueError),
        ({'bins': -1}, ValueError),
        ({'bins': True}, TypeError),
        ({'bins': 2.5}, ValueError),
        ({'bins': [0, 1, 1]}, ValueError),
        ({'bins': [2, 1]}, ValueError),
        ({'bins': [0, np.inf]}, ValueError),
        ({'bins': [0, np.nan]}, ValueError),
        ({'bins': [0, 1j]}, TypeError),
        ({'range': (2, 1)}, ValueError),
        ({'range': (1, 1)}, ValueError),
        ({'range': (0, np.inf)}, ValueError),
        ({'range': (0, 1, 2)}, ValueError),
        ({'range': (0, 1j)}, TypeError),
        ({'feature': 2}, ValueError),
        ({'feature': -1}, ValueError),
        ({'feature': 0.5}, TypeError),
        ({'feature': True}, TypeError),
        ({'density': 'yes'}, TypeError),
    ]:
        with _case("invalid_histogram_options_rejected", options=options, error=error):
            with pytest.raises(error):
                RunningStats().update([[1, 2], [3, 4]]).histogram(**options)

