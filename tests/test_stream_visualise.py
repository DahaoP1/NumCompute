"""Grouped headless tests for plots, NaN gaps, class axes and file output."""

from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from numcompute_stream import visualise


@pytest.fixture(autouse=True)
def close_figures():
    """Keep independent tests from retaining matplotlib figures."""
    yield
    plt.close("all")


@contextmanager
def case(name, **options):
    """Identify variants inside grouped tests."""
    try:
        yield
    except (Exception, pytest.fail.Exception) as error:
        raise AssertionError(f"{name}, {options}: {error}") from error


def test_metric_curve_preserves_positions_nan_gaps_and_noncontiguous_input():
    values = np.array([np.nan, 0.4, 0.6, 0.8, 1.0])[::-2]
    saved = values.copy()
    fig, ax = visualise.plot_metric_over_time(values, "Streaming accuracy", "Accuracy", show=False)
    assert ax.figure is fig and ax.get_title() == "Streaming accuracy"
    assert ax.get_xlabel() == "Chunk" and ax.get_ylabel() == "Accuracy"
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2, 3])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), saved)
    assert ax.get_xlim()[0] < 1 and ax.get_xlim()[1] > len(values)
    np.testing.assert_array_equal(values, saved)
    fig.canvas.draw()
    for data in ([np.nan, np.nan], [0.75], [True, False]):
        with case("valid metric history", data=data):
            fig, ax = visualise.plot_metric_over_time(data, show=False)
            assert len(ax.lines[0].get_xdata()) == len(data)
            fig.canvas.draw()


def test_comparison_has_aligned_series_legend_and_custom_labels():
    first, second = [np.nan, 0.7, 0.8], [np.nan, np.nan, 0.9]
    fig, ax = visualise.compare_models(first, second, ["Buffer 500", "Buffer 1000"],
                                      title="Buffered trees", ylabel="Chunk accuracy", show=False)
    assert len(ax.lines) == 2
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["Buffer 500", "Buffer 1000"]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2, 3])
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [1, 2, 3])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), first)
    np.testing.assert_array_equal(ax.lines[1].get_ydata(), second)
    assert ax.get_title() == "Buffered trees" and ax.get_ylabel() == "Chunk accuracy"
    assert ax.lines[0].get_linestyle() != ax.lines[1].get_linestyle()
    assert ax.get_xlim()[0] < 1  # Leading warm-up NaNs do not crop the chunk axis.
    fig.canvas.draw()


def test_prediction_plot_shares_sorted_classes_and_shows_actual_mismatches():
    datasets = [(["Sydney", "Adelaide", "Sydney"], ["Perth", "Adelaide", "Sydney"]),
                ([10, 90, 10], [90, 90, 10]), ([True, False], [True, True]),
                ([b"a", b"b"], [b"b", b"b"])]
    for truth, pred in datasets:
        with case("label type", truth=truth):
            fig, ax = visualise.plot_predictions_vs_ground_truth(truth, pred, show=False)
            classes = np.union1d(truth, pred)
            assert [label.get_text() for label in ax.get_yticklabels()] == classes.astype(str).tolist()
            assert ax.get_xlabel() == "Sample" and ax.get_ylabel() == "Class"
            true_points, pred_points = (collection.get_offsets() for collection in ax.collections)
            np.testing.assert_array_equal(true_points[:, 0], np.arange(1, len(truth) + 1))
            np.testing.assert_array_equal(true_points[:, 1], np.searchsorted(classes, truth))
            np.testing.assert_array_equal(pred_points[:, 1], np.searchsorted(classes, pred))
            assert [text.get_text() for text in ax.get_legend().get_texts()] == ["Ground truth", "Prediction"]
            fig.canvas.draw()
    view = np.array([0, 1, 0, 1])[::2]
    saved = view.copy()
    visualise.plot_predictions_vs_ground_truth(view, view, show=False)
    np.testing.assert_array_equal(view, saved)


def test_figures_save_to_nested_paths_with_supported_formats(tmp_path):
    cases = [(visualise.plot_metric_over_time, ([0.2, 0.5],), "metric.png"),
             (visualise.compare_models, ([0.3, 0.6], [0.4, 0.7]), "comparison.svg"),
             (visualise.plot_predictions_vs_ground_truth, (["a", "b"], ["a", "a"]), "predictions.png")]
    for function, args, filename in cases:
        with case("saved plot", function=function.__name__):
            target = tmp_path / "new" / "plots" / filename
            fig, ax = function(*args, save_path=target, show=False)
            assert target.is_file() and target.stat().st_size > 100
            assert ax.figure is fig
            if target.suffix == ".png":
                assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            else:
                assert "<svg" in target.read_text(encoding="utf-8")


def test_show_can_be_enabled_or_disabled_for_every_function(monkeypatch):
    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append("show"))
    functions = [(visualise.plot_metric_over_time, ([0.5],)),
                 (visualise.compare_models, ([0.5], [0.6])),
                 (visualise.plot_predictions_vs_ground_truth, ([0], [0]))]
    for function, args in functions:
        with case("show option", function=function.__name__):
            before = len(calls)
            function(*args, show=False)
            assert len(calls) == before
            function(*args)  # Default shows the plot.
            assert len(calls) == before + 1


def test_metric_validation_rejects_bad_histories_without_leaking_figures():
    for values in ([], [[0.5]], [np.inf], [-np.inf], [1j], ["0.5"], [None]):
        for function, args in ((visualise.plot_metric_over_time, (values,)),
                               (visualise.compare_models, (values, [0.5]))):
            with case("invalid history", values=values, function=function.__name__):
                before = plt.get_fignums()
                with pytest.raises((TypeError, ValueError)):
                    function(*args, show=False)
                assert plt.get_fignums() == before
    for args, options in ((([0.1], [0.1, 0.2]), {}), (([0.1], [0.2]), {"labels": "AB"}),
                          (([0.1], [0.2]), {"labels": ["one"]}), (([0.1], [0.2]), {"labels": ["one", " "]}),
                          (([0.1], [0.2]), {"labels": [1, 2]}), (([0.1], [0.2]), {"labels": None})):
        with case("invalid comparison", options=options), pytest.raises((TypeError, ValueError)):
            visualise.compare_models(*args, show=False, **options)


def test_predictions_and_options_validate_and_failed_saves_close_new_figures(tmp_path):
    for truth, pred in (([], []), ([[0]], [[0]]), ([0], [0, 1]), ([0], ["0"]),
                        ([np.nan], [0]), ([0], [np.inf]), ([""], ["a"]), ([None], [0]), ([1j], [0])):
        with case("invalid prediction labels", truth=truth, pred=pred), pytest.raises((TypeError, ValueError)):
            visualise.plot_predictions_vs_ground_truth(truth, pred, show=False)
    for function, args in ((visualise.plot_metric_over_time, ([0.5],)),
                           (visualise.compare_models, ([0.5], [0.6])),
                           (visualise.plot_predictions_vs_ground_truth, ([0], [0]))):
        for options in ({"show": 1}, {"show": "no"}, {"title": None}):
            with case("invalid options", options=options), pytest.raises(TypeError):
                function(*args, **options)
        for path in ("", tmp_path, 7, tmp_path / "plot.unsupported_format"):
            before = plt.get_fignums()
            with case("invalid save path", path=path), pytest.raises((ValueError, TypeError)):
                function(*args, save_path=path, show=False)
            assert plt.get_fignums() == before


def test_repeated_calls_return_independent_customisable_figures():
    first, ax1 = visualise.plot_metric_over_time([0.2, 0.3], "First", show=False)
    second, ax2 = visualise.plot_metric_over_time([0.7, 0.8], "Second", show=False)
    assert first is not second and ax1 is not ax2
    ax1.set_title("Customised")
    assert ax2.get_title() == "Second"
    np.testing.assert_array_equal(ax2.lines[0].get_ydata(), [0.7, 0.8])
