"""Grouped tests for chunk orchestration, evaluation logs and state memory."""

from contextlib import contextmanager
from sys import getsizeof

import numpy as np
import pytest

from numcompute_stream import metrics
from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import SimpleImputer
from numcompute_stream.stream import StreamTrainer, _memory_bytes
from numcompute_stream.tree import DecisionTreeClassifier


@contextmanager
def case(name, **options):
    """Identify variants within grouped test functions."""
    try:
        yield
    except (Exception, pytest.fail.Exception) as error:
        raise AssertionError(f"{name}, {options}: {error}") from error


class RecordingModel:
    """Test double: return the first feature as a label; record all dispatch."""

    def __init__(self):
        self.calls = []

    def partial_fit(self, X, y):
        self.calls.append("fit")
        return self

    def predict(self, X):
        self.calls.append("predict")
        return X[:, 0].copy()


def test_training_and_scoring_are_separate_and_logs_describe_each_operation():
    model = RecordingModel()
    trainer = StreamTrainer(model)
    assert trainer.model is model
    assert trainer.fit_chunk([[0], [1]], [0, 1]) is trainer
    assert model.calls == ["fit"]
    fit = trainer.logs_[-1]
    assert fit["event"] == "fit" and fit["accuracy"] is None
    assert fit["cumulative_accuracy"] is None and fit["metrics"] == {}
    assert trainer.n_scored_samples_ == 0
    score = trainer.score_chunk([[0], [1]], [0, 0])
    assert model.calls == ["fit", "predict"]
    assert score["accuracy"] == score["cumulative_accuracy"] == 0.5
    assert score["fit_chunks"] == score["score_chunks"] == 1
    assert score["trained_samples"] == score["scored_samples"] == 2
    for row in trainer.logs_:
        assert isinstance(row["memory_bytes"], int) and row["memory_bytes"] > 0
        assert row["seconds"] >= 0
        assert row["buffered_samples"] is None and row["n_retrains"] is None
    trainer.fit_chunk([[1]], [1])
    assert trainer.n_trained_samples_ == 3 and trainer.n_scored_samples_ == 2
    assert trainer.logs_[-1]["cumulative_accuracy"] == 0.5


def test_cumulative_accuracy_weights_samples_and_extra_metrics_keep_their_windows():
    recent = metrics.Accuracy(window_size=3)
    f1 = metrics.F1(average="macro")
    trainer = StreamTrainer(RecordingModel(), metrics={"recent": recent, "f1": f1})
    assert trainer.metrics["recent"] is recent
    first = trainer.score_chunk([[1]], [1])
    second = trainer.score_chunk([[0], [0], [0]], [1, 1, 1])
    assert first["accuracy"] == first["cumulative_accuracy"] == 1
    assert second["accuracy"] == 0
    assert second["cumulative_accuracy"] == 0.25  # Not mean([1, 0]).
    assert second["metrics"]["recent"] == 0
    assert second["metrics"]["f1"] == pytest.approx(metrics.f1([1, 1, 1, 1], [1, 0, 0, 0], average="macro"))
    assert recent.n_samples_ == 3 and recent.n_samples_seen_ == 4
    assert trainer.n_fit_chunks_ == 0 and trainer.n_score_chunks_ == 2


def test_buffer_warmup_retraining_and_pending_chunks_are_reported():
    tree = DecisionTreeClassifier(max_depth=2, buffer_size=5)
    trainer = StreamTrainer(tree)
    for X, y in (([[0], [1]], [0, 0]), ([[8], [9]], [1, 1])):
        trainer.fit_chunk(X, y)
        count = len(trainer.logs_)
        with pytest.raises(RuntimeError):
            trainer.score_chunk([[0], [9]], [0, 1])
        assert len(trainer.logs_) == count and trainer.n_score_chunks_ == 0
    assert trainer.logs_[-1]["buffered_samples"] == 4
    assert trainer.logs_[-1]["n_retrains"] == 0
    trainer.fit_chunk([[2], [10]], [0, 1])
    assert trainer.logs_[-1]["buffered_samples"] == 0
    assert trainer.logs_[-1]["n_retrains"] == 1
    assert tree.n_training_samples_ == 6
    row = trainer.score_chunk([[0], [9]], [0, 1])
    assert row["accuracy"] == 1
    trainer.fit_chunk([[3]], [0])
    assert trainer.logs_[-1]["buffered_samples"] == 1
    assert trainer.logs_[-1]["n_retrains"] == 1
    assert trainer.score_chunk([[0], [9]], [0, 1])["accuracy"] == 1
    assert trainer.n_trained_samples_ == 7 and trainer.n_scored_samples_ == 4


def test_pipeline_dispatch_uses_existing_preprocessing_and_prediction_only_for_scoring():
    imputer = SimpleImputer(fill_value=0)
    tree = DecisionTreeClassifier(buffer_size=4)
    pipe = Pipeline([("impute", imputer), ("model", tree)])
    trainer = StreamTrainer(pipe, metrics={"confusion": metrics.ConfusionMatrix(), "f1": metrics.F1(average="macro")})
    trainer.fit_chunk([[np.nan], [10]], [0, 1])
    trainer.fit_chunk([[1], [9]], [0, 1])
    count_before = imputer.n_samples_seen_.copy()
    row = trainer.score_chunk([[np.nan], [10]], [0, 1])
    assert row["accuracy"] == row["metrics"]["f1"] == 1
    np.testing.assert_array_equal(row["metrics"]["confusion"], [[1, 0], [0, 1]])
    np.testing.assert_array_equal(imputer.n_samples_seen_, count_before)
    assert tree.n_retrains_ == 1 and tree.n_buffered_samples_ == 0
    assert row["n_retrains"] == 1
    # A fresh trainer may evaluate an already trained model without retraining it.
    fresh = StreamTrainer(pipe)
    assert fresh.score_chunk([[1]], [0])["accuracy"] == 1
    assert fresh.n_trained_samples_ == 0 and tree.n_retrains_ == 1


def test_score_results_and_previous_logs_are_independent_snapshots():
    confusion = metrics.ConfusionMatrix()
    trainer = StreamTrainer(RecordingModel(), metrics={"confusion": confusion})
    returned = trainer.score_chunk([[0], [1]], [0, 1])
    returned["accuracy"] = -1
    returned["metrics"]["confusion"][:] = 0
    assert trainer.logs_[0]["accuracy"] == 1
    np.testing.assert_array_equal(trainer.logs_[0]["metrics"]["confusion"], [[1, 0], [0, 1]])
    trainer.score_chunk([[0], [1]], [0, 1])
    assert trainer.logs_[0]["metrics"]["confusion"].sum() == 2
    assert trainer.logs_[1]["metrics"]["confusion"].sum() == 4


def test_empty_chunks_shapes_and_missing_targets_do_not_change_trainer_counts():
    trainer = StreamTrainer(DecisionTreeClassifier(buffer_size=2))
    assert trainer.fit_chunk(np.empty((0, 1)), []) is trainer
    assert trainer.score_chunk(np.empty((0, 1)), []) is None
    assert trainer.logs_ == []
    trainer.fit_chunk([[0], [1]], [0, 1])
    before = (len(trainer.logs_), trainer.n_trained_samples_, trainer.n_scored_samples_)
    invalid = (([0, 1], [0, 1]), (np.zeros((1, 1, 1)), [0]), (np.empty((1, 0)), [0]),
               ([[0], [1]], [0]), ([[0]], [[0]]), ([[0]], [np.nan]), ([[0]], [np.inf]),
               ([[0]], [""]), ([[0]], [None]), (np.empty((0, 2)), []))
    for X, y in invalid:
        for method in (trainer.fit_chunk, trainer.score_chunk):
            with case("invalid trainer chunk", X=X, y=y), pytest.raises((ValueError, TypeError)):
                method(X, y)
            assert (len(trainer.logs_), trainer.n_trained_samples_, trainer.n_scored_samples_) == before


def test_invalid_predictions_and_failed_model_updates_propagate():
    class BadPrediction(RecordingModel):
        def __init__(self, output):
            super().__init__()
            self.output = output

        def predict(self, X):
            return self.output

    for output in ([[0], [1]], [0], [np.nan, 1], [np.inf, 1], [None, 1], ["", "a"], ["a", "b"]):
        with case("invalid model prediction", output=output):
            extra = metrics.F1(average="macro")
            trainer = StreamTrainer(BadPrediction(output), metrics={"f1": extra})
            with pytest.raises((ValueError, TypeError)):
                trainer.score_chunk([[0], [1]], [0, 1])
            assert trainer.logs_ == [] and trainer.n_scored_samples_ == 0
            assert extra.n_samples_seen_ == 0

    class FailingModel(RecordingModel):
        def partial_fit(self, X, y):
            raise RuntimeError("test training failure")

    trainer = StreamTrainer(FailingModel())
    with pytest.raises(RuntimeError, match="test training failure"):
        trainer.fit_chunk([[0]], [0])
    assert trainer.logs_ == [] and trainer.n_fit_chunks_ == trainer.n_trained_samples_ == 0
    untrained_pipe = StreamTrainer(Pipeline([("model", DecisionTreeClassifier())]))
    with pytest.raises(ValueError, match="fit"):
        untrained_pipe.score_chunk([[0]], [0])


def test_metric_configuration_and_model_interfaces_are_validated():
    for model in (None, object(), metrics.Accuracy()):
        with case("invalid model", model=model), pytest.raises(TypeError):
            StreamTrainer(model)
    for options in ([], [("f1", metrics.F1())], metrics.Accuracy()):
        with case("invalid metric mapping", options=options), pytest.raises(TypeError):
            StreamTrainer(RecordingModel(), metrics=options)
    for options in ({"": metrics.Accuracy()}, {0: metrics.Accuracy()}, {"x": object()},
                    {"auc": metrics.AUC()}, {"roc": metrics.ROCCurve()}):
        with case("invalid metric", options=options), pytest.raises((ValueError, TypeError)):
            StreamTrainer(RecordingModel(), metrics=options)
    duplicate = metrics.Accuracy()
    with pytest.raises(ValueError, match="separate"):
        StreamTrainer(RecordingModel(), metrics={"a": duplicate, "b": duplicate})
    # Constructor neither clears existing metrics nor evaluates the model.
    existing = metrics.Accuracy().update([1], [1])
    model = RecordingModel()
    StreamTrainer(model, metrics={"existing": existing})
    assert existing.result() == 1 and model.calls == []


def test_memory_estimate_counts_shared_numpy_buffers_once_and_handles_cycles():
    array = np.arange(2000.0)
    assert _memory_bytes(array) >= array.nbytes
    shared = [array, array]
    assert _memory_bytes(shared) == getsizeof(shared) + _memory_bytes(array)
    view = array[::2]
    assert _memory_bytes(view) >= array.nbytes
    pair = (array, view)
    assert _memory_bytes(pair) == getsizeof(pair) + _memory_bytes(array) + getsizeof(view)
    cyclic = []
    cyclic.append(cyclic)
    assert _memory_bytes(cyclic) == getsizeof(cyclic)


def test_noncontiguous_inputs_and_label_family_errors_are_handled():
    X = np.arange(20).reshape(10, 2)[::-2]
    y = X[:, 0].copy()
    saved_X, saved_y = X.copy(), y.copy()
    recent = metrics.Accuracy(window_size=3)
    trainer = StreamTrainer(RecordingModel(), metrics={"recent": recent})
    trainer.fit_chunk(X, y)
    assert trainer.score_chunk(X, y)["accuracy"] == 1
    np.testing.assert_array_equal(X, saved_X)
    np.testing.assert_array_equal(y, saved_y)
    before = recent.n_samples_seen_
    with pytest.raises(TypeError):
        trainer.score_chunk([["a"]], ["a"])
    assert recent.n_samples_seen_ == before and trainer.n_score_chunks_ == 1
    # A new trainer can score string labels from a corresponding string model.
    assert StreamTrainer(RecordingModel()).score_chunk([["city"]], ["city"])["accuracy"] == 1
