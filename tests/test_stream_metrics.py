"""Grouped correctness and edge-case checks for streamed evaluation metrics."""

from contextlib import contextmanager

import numpy as np
import pytest

from numcompute import metrics as batch
from numcompute_stream import metrics as sm


@contextmanager
def case(name, **options):
    """Identify variants within grouped tests without inflating test counts."""
    try:
        yield
    except (Exception, pytest.fail.Exception) as error:
        raise AssertionError(f"{name}, {options}: {error}") from error


@pytest.fixture
def labels():
    rng = np.random.default_rng(7)
    return rng.integers(0, 4, 47), rng.integers(0, 4, 47)


def test_original_batch_apis_and_weighted_streaming_accuracy():
    # The new module extends the original source rather than importing aliases.
    truth, pred = [0, 0, 1, 1], [0, 1, 1, 1]
    for name in ("accuracy", "precision", "recall", "f1", "confusion_matrix", "mse"):
        with case("original batch API", function=name):
            np.testing.assert_allclose(getattr(sm, name)(truth, pred), getattr(batch, name)(truth, pred))
    for actual, reference in zip(sm.roc_curve(truth, [0.1, 0.4, 0.4, 0.8]),
                                 batch.roc_curve(truth, [0.1, 0.4, 0.4, 0.8])):
        np.testing.assert_array_equal(actual, reference)
    assert sm.auc([0, 0.5, 1], [0, 0.8, 1]) == batch.auc([0, 0.5, 1], [0, 0.8, 1])
    metric = sm.Accuracy()
    assert metric.update([1], [1]) is metric
    metric.update([1, 1, 1], [0, 0, 0])
    assert metric.result() == 0.25  # Not the average of chunk accuracies (0.5).
    assert metric.n_samples_ == metric.n_samples_seen_ == 4
    assert metric._true is None and metric._pred is None


def test_cumulative_classification_matches_batch_at_each_prefix(labels):
    truth, pred = labels
    for size in (1, 7, 100):
        for average in (None, "macro", "micro", "weighted"):
            with case("multiclass prefixes", size=size, average=average):
                metrics = [sm.Accuracy(), sm.Precision(average=average), sm.Recall(average=average),
                           sm.F1(average=average), sm.ConfusionMatrix()]
                for start in range(0, len(truth), size):
                    end = start + size
                    for metric in metrics:
                        metric.update(truth[start:end], pred[start:end])
                    expected = [batch.accuracy(truth[:end], pred[:end]),
                                batch.precision(truth[:end], pred[:end], average=average),
                                batch.recall(truth[:end], pred[:end], average=average),
                                batch.f1(truth[:end], pred[:end], average=average),
                                batch.confusion_matrix(truth[:end], pred[:end])]
                    for metric, value in zip(metrics, expected):
                        np.testing.assert_allclose(metric.result(), value)


def test_confusion_expands_classes_and_preserves_fixed_order():
    metric = sm.ConfusionMatrix().update(["Sydney"], ["Sydney"])
    metric.update(["Adelaide", "Sydney"], ["Perth", "Adelaide"])
    np.testing.assert_array_equal(metric.labels_, ["Adelaide", "Perth", "Sydney"])
    np.testing.assert_array_equal(metric.result(), [[0, 1, 0], [0, 0, 0], [1, 0, 1]])
    returned = metric.result()
    returned[:] = 0
    assert metric.result().sum() == 3
    order = np.array([2, 0, 1, 3])
    fixed = sm.ConfusionMatrix(labels=order).update([0, 1], [1, 1])
    order[:] = -1
    fixed.update([2], [0])
    np.testing.assert_array_equal(fixed.labels_, [2, 0, 1, 3])
    np.testing.assert_array_equal(fixed.result(), batch.confusion_matrix([0, 1, 2], [1, 1, 0], labels=[2, 0, 1, 3]))


def test_binary_scores_positive_labels_and_undefined_ratios():
    pairs = [([0, 1, 1, 0], [0, 1, 0, 1], 1),
             ([0, 1, 1, 0], [0, 1, 0, 1], 0),
             (["ham", "spam", "spam"], ["spam", "spam", "ham"], "spam"),
             ([True, False, True], [False, False, True], True),
             ([b"a", b"b", b"a"], [b"a", b"b", b"b"], b"a")]
    for cls, function in ((sm.Precision, batch.precision), (sm.Recall, batch.recall), (sm.F1, batch.f1)):
        for truth, pred, positive in pairs:
            with case("binary labels", cls=cls.__name__, positive=positive):
                metric = cls(pos_label=positive).update(truth[:1], pred[:1]).update(truth[1:], pred[1:])
                assert metric.result() == pytest.approx(function(truth, pred, pos_label=positive))
        for fallback in (0.0, 1.0):
            with case("undefined positive", cls=cls.__name__, fallback=fallback):
                metric = cls(zero_division=fallback).update([0, 0], [0, 0])
                assert metric.result() == fallback
                metric.update([1], [0])
                assert metric.result() == pytest.approx(function([0, 0, 1], [0, 0, 0], zero_division=fallback))
    # Fixed labels can keep an absent class in per-class/macro calculations.
    np.testing.assert_array_equal(sm.Precision(average=None, labels=[2, 1, 0], zero_division=1).update([0], [0]).result(), [1, 1, 1])
    assert sm.F1(average="weighted", labels=[0, 1, 2]).update([0], [0]).result() == 1


def test_rolling_classification_matches_last_samples_across_chunk_boundaries(labels):
    truth, pred = labels
    for window in (1, 5, 20, 100):
        for average in (None, "macro", "micro", "weighted"):
            with case("rolling classification", window=window, average=average):
                metrics = [sm.Accuracy(window_size=window), sm.Precision(average=average, window_size=window),
                           sm.Recall(average=average, window_size=window), sm.F1(average=average, window_size=window),
                           sm.ConfusionMatrix(window_size=window)]
                for start in (0, 3, 18, 43):
                    end = {0: 3, 3: 18, 18: 43, 43: len(truth)}[start]
                    recent_true, recent_pred = truth[max(0, end-window):end], pred[max(0, end-window):end]
                    for metric in metrics:
                        metric.update(truth[start:end], pred[start:end])
                        assert metric.n_samples_ == min(window, end)
                        assert metric.n_samples_seen_ == end
                    expected = [batch.accuracy(recent_true, recent_pred),
                                batch.precision(recent_true, recent_pred, average=average),
                                batch.recall(recent_true, recent_pred, average=average),
                                batch.f1(recent_true, recent_pred, average=average),
                                batch.confusion_matrix(recent_true, recent_pred)]
                    for metric, value in zip(metrics, expected):
                        np.testing.assert_allclose(metric.result(), value)
                        assert len(metric._true) <= window


def test_window_labels_expire_and_fixed_classes_keep_zero_counts():
    dynamic = sm.ConfusionMatrix(window_size=2).update(["a", "b"], ["b", "b"])
    dynamic.update(["c", "c", "c"], ["c", "c", "c"])
    np.testing.assert_array_equal(dynamic.labels_, ["c"])
    np.testing.assert_array_equal(dynamic.result(), [[2]])
    assert dynamic.n_samples_seen_ == 5
    fixed = sm.ConfusionMatrix(labels=[2, 1, 0], window_size=2).update([0], [0])
    fixed.update([1, 1, 1], [1, 1, 1])
    np.testing.assert_array_equal(fixed.labels_, [2, 1, 0])
    np.testing.assert_array_equal(fixed.result(), [[0, 0, 0], [0, 2, 0], [0, 0, 0]])
    # Invalid fixed labels are checked even when an oversized chunk would evict them.
    with pytest.raises(ValueError):
        fixed.update([99, 1, 1], [99, 1, 1])
    assert fixed.n_samples_seen_ == 4


def test_mse_streaming_windows_and_noncontiguous_inputs():
    truth = np.arange(35.0)[::-2]
    pred = truth + np.resize([0.0, 2.0, -3.0], truth.size)
    saved = truth.copy()
    for window in (None, 1, 7, 100):
        with case("MSE window", window=window):
            metric = sm.MSE(window_size=window)
            for start in range(0, len(truth), 5):
                end = min(start + 5, len(truth))
                assert metric.update(truth[start:end], pred[start:end]) is metric
                begin = 0 if window is None else max(0, end-window)
                assert metric.result() == pytest.approx(batch.mse(truth[begin:end], pred[begin:end]))
                assert metric.n_samples_seen_ == end
            assert len(metric._errors) <= (0 if window is None else window)
    weighted = sm.MSE().update([0], [0]).update([0, 0, 0], [2, 2, 2])
    assert weighted.result() == 3  # Per-sample weighting, not mean of 0 and 4.
    np.testing.assert_array_equal(truth, saved)


def test_mse_large_representable_errors_and_overflow_keep_state():
    for window in (None, 4):
        with case("MSE stability", window=window):
            metric = sm.MSE(window_size=window).update(np.zeros(4), np.full(4, 1e154))
            assert np.isfinite(metric.result())
            assert metric.result() == pytest.approx(1e308)
            before = metric.result()
            with pytest.raises(FloatingPointError):
                metric.update([0.0], [1e200])
            assert metric.result() == before and metric.n_samples_seen_ == 4
    assert sm.MSE().update([1e12, 1e12 + 1], [1e12 + 1, 1e12]).result() == 1


def test_roc_and_auc_accumulate_single_class_chunks_and_cross_chunk_ties():
    truth = np.array([0, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.4, 0.4, 0.8, 0.8, 0.4])
    roc, area = sm.ROCCurve(), sm.AUC()
    assert roc.update(truth[:2], scores[:2]) is roc
    area.update(truth[:2], scores[:2])
    for metric in (roc, area):
        with pytest.raises(ValueError):
            metric.result()  # Warm-up is accepted but the rates are not defined.
    for start in (2, 4):
        roc.update(truth[start:start+2], scores[start:start+2])
        area.update(truth[start:start+2], scores[start:start+2])
        expected = batch.roc_curve(truth[:start+2], scores[:start+2])
        for actual, reference in zip(roc.result(), expected):
            np.testing.assert_array_equal(actual, reference)
        assert area.result() == pytest.approx(batch.auc(expected[0], expected[1]))
    assert len(roc.result()[2]) == len(np.unique(scores)) + 1
    assert sm.AUC().update([0, 1], [0.5, 0.5]).result() == 0.5
    assert sm.AUC().update([0, 1], [1.0, 0.0]).result() == 0.0
    assert sm.AUC().update([0, 1], [0.0, 1.0]).result() == 1.0
    assert sm.AUC(pos_label="yes").update(["yes", "no"], [0.9, 0.1]).result() == 1


def test_rolling_roc_auc_match_sample_window_and_own_copied_inputs():
    truth = np.array([0, 1, 0, 1, 0, 1, 1, 0])
    scores = np.array([0.2, 0.8, 0.5, 0.5, 0.1, 0.9, 0.4, 0.7])
    for window in (1, 3, 5, 100):
        with case("ROC window", window=window):
            roc, area = sm.ROCCurve(window_size=window), sm.AUC(window_size=window)
            for start, end in ((0, 2), (2, 7), (7, 8)):
                roc.update(truth[start:end], scores[start:end])
                area.update(truth[start:end], scores[start:end])
                begin = max(0, end-window)
                assert area.n_samples_ == min(window, end)
                assert area.n_samples_seen_ == end
                if len(np.unique(truth[begin:end])) < 2:
                    with pytest.raises(ValueError):
                        area.result()
                else:
                    expected = batch.roc_curve(truth[begin:end], scores[begin:end])
                    for actual, reference in zip(roc.result(), expected):
                        np.testing.assert_array_equal(actual, reference)
                    assert area.result() == pytest.approx(batch.auc(expected[0], expected[1]))
    metric = sm.AUC().update(truth, scores)
    expected = metric.result()
    truth[:] = 0
    scores[:] = 0
    assert metric.result() == expected
    metric.update([0, 1], [0.3, 0.7])
    assert metric.n_samples_ == 10


def test_empty_updates_reset_and_result_lifecycle():
    constructors = (sm.Accuracy, sm.Precision, sm.Recall, sm.F1, sm.ConfusionMatrix, sm.MSE, sm.ROCCurve, sm.AUC)
    for cls in constructors:
        with case("metric lifecycle", cls=cls.__name__):
            metric = cls(window_size=3)
            assert metric.update([], []) is metric
            with pytest.raises(RuntimeError):
                metric.result()
            metric.update([0, 1], [0, 1])
            before = metric.result()
            metric.update([], [])
            if isinstance(before, tuple):
                for actual, expected in zip(metric.result(), before):
                    np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_array_equal(metric.result(), before)
            assert metric.reset() is metric
            assert metric.window_size == 3
            assert metric.n_samples_ == metric.n_samples_seen_ == 0
            with pytest.raises(RuntimeError):
                metric.result()
            metric.update([0, 1], [1, 0])
            assert metric.n_samples_ == 2
    fixed = sm.ConfusionMatrix(labels=[1, 0]).update([1], [0]).reset()
    np.testing.assert_array_equal(fixed.labels_, [1, 0])
    # Auto-discovered label families can change after reset.
    assert sm.Accuracy().update([0], [0]).reset().update(["city"], ["city"]).result() == 1


def test_bad_input_vectors_leave_metrics_unchanged():
    invalid = (([0], [0, 1]), ([[0], [1]], [[0], [1]]), ([np.nan], [0]),
               ([np.inf], [0]), ([0], [np.inf]), ([1j], [0]), ([None], [0]))
    for cls in (sm.Accuracy, sm.Precision, sm.Recall, sm.F1, sm.ConfusionMatrix, sm.MSE, sm.ROCCurve, sm.AUC):
        for values in invalid:
            with case("invalid metric vectors", cls=cls.__name__, values=values):
                metric = cls().update([0, 1], [0, 1])
                before = metric.result()
                with pytest.raises((TypeError, ValueError)):
                    metric.update(*values)
                assert metric.n_samples_seen_ == 2
                if isinstance(before, tuple):
                    for actual, expected in zip(metric.result(), before):
                        np.testing.assert_array_equal(actual, expected)
                else:
                    np.testing.assert_array_equal(metric.result(), before)
    for cls in (sm.Accuracy, sm.ConfusionMatrix, sm.Precision):
        for truth, pred in (([""], ["a"]), ([" "], ["a"]), ([0], ["0"])):
            with case("missing/incompatible labels", cls=cls.__name__), pytest.raises((TypeError, ValueError)):
                cls().update(truth, pred)
    for cls in (sm.MSE, sm.ROCCurve, sm.AUC):
        with case("nonnumeric predictions/scores", cls=cls.__name__), pytest.raises(TypeError):
            cls().update([0], ["0.5"])


def test_label_family_and_binary_rejections_preserve_previous_counts():
    for cls in (sm.Accuracy, sm.ConfusionMatrix, sm.Precision, sm.Recall, sm.F1):
        with case("family change", cls=cls.__name__):
            metric = cls().update([0, 1], [0, 1])
            with pytest.raises(TypeError):
                metric.update(["0"], ["0"])
            assert metric.n_samples_seen_ == 2
    for cls in (sm.Precision, sm.Recall, sm.F1, sm.ROCCurve, sm.AUC):
        with case("third class", cls=cls.__name__):
            metric = cls().update([0, 1], [0, 1])
            with pytest.raises(ValueError):
                metric.update([2], [0])
            assert metric.n_samples_seen_ == 2
    for cls in (sm.Precision, sm.ROCCurve, sm.AUC):
        with case("incompatible positive", cls=cls.__name__), pytest.raises(TypeError):
            cls().update(["no", "yes"], ["no", "yes"] if cls is sm.Precision else [0.1, 0.9])
        with case("absent positive among two labels", cls=cls.__name__), pytest.raises(ValueError):
            cls(pos_label=3).update([0, 1], [0, 1])


def test_invalid_configuration_and_fixed_label_validation():
    for cls in (sm.Accuracy, sm.Precision, sm.Recall, sm.F1, sm.ConfusionMatrix, sm.MSE, sm.ROCCurve, sm.AUC):
        for window in (0, -1, True, np.bool_(False), 1.5, "3"):
            with case("invalid window", cls=cls.__name__, window=window), pytest.raises((TypeError, ValueError)):
                cls(window_size=window)
    for options in ({"average": "mean"}, {"zero_division": np.nan}, {"zero_division": np.inf},
                    {"zero_division": True}, {"zero_division": "0"}, {"pos_label": [1]},
                    {"labels": [0, 1, 2]}, {"labels": [0, 1], "pos_label": 3}):
        with case("invalid averaging settings", options=options), pytest.raises((TypeError, ValueError)):
            sm.Precision(**options)
    for labels in ([], [0, 0], [0, np.nan], [0, np.inf], [[0, 1]], [""], [None]):
        with case("invalid fixed labels", labels=labels), pytest.raises((TypeError, ValueError)):
            sm.ConfusionMatrix(labels=labels)
    with pytest.raises(ValueError):
        sm.ConfusionMatrix(labels=[0, 1]).update([2], [0])
    for cls in (sm.ROCCurve, sm.AUC):
        for positive in ([1], None, "", np.nan, np.inf):
            with case("invalid ROC positive", cls=cls.__name__, positive=positive), pytest.raises((TypeError, ValueError)):
                cls(pos_label=positive)


def test_label_dtypes_strides_and_input_ownership():
    for truth, pred in (([True, False], [True, True]), ([0.0, 1.0], [0, 1]),
                        ([b"a", b"b"], [b"a", b"a"])):
        with case("supported label types", truth=truth):
            assert sm.Accuracy().update(truth, pred).result() == batch.accuracy(truth, pred)
    truth = np.array(["a", "b", "a", "b", "c", "c"])[::-2]
    pred = np.array(["b", "b", "a", "a", "b", "c"])[::-2]
    expected = batch.confusion_matrix(truth, pred)
    metric = sm.ConfusionMatrix(window_size=10).update(truth, pred)
    truth[:] = "changed"
    pred[:] = "changed"
    np.testing.assert_array_equal(metric.result(), expected)
    metric.update(["d"], ["d"])
    assert metric.result().sum() == 4
