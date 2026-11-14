# Tests

This folder covers both assignments. The suite currently has **386 collected cases**: 212 for the original NumCompute package and 174 for streaming extensions. Related edge cases are grouped in test functions. The old tests remain in place to catch regressions.

From the project root:

```bash
python -m pytest tests -q
python -m pytest tests -q -k test_stream
python -m pytest tests/test_stream_tree.py -q
```

## Streaming coverage

| File | Main checks |
| --- | --- |
| `test_stream_tree.py` | Splits, NaNs, ties, reference loop results, buffer thresholds/clearing, prediction warm-up and pending rows. |
| `test_stream_preprocessing.py` | Chunk/batch agreement, valid counts, missing/constant columns, category growth, reset and sliced inputs. |
| `test_stream_stats.py` | Running moments, exact distribution queries, memory mode, overflow rejection and NaN recovery. |
| `test_stream_metrics.py` | Cumulative and rolling results, class order, zero denominators, binary ROC/AUC and reset. |
| `test_stream_pipeline.py` | Original batch API, incremental step order, label routing, empty chunks and error propagation. |
| `test_stream_trainer.py` | Training/scoring separation, weighted accuracy, model readiness, logs and memory estimates. |
| `test_stream_visualise.py` | Headless figures, NaN gaps, category axes, saving and invalid input. |
| `test_stream_benchmarking.py` | Matching loop/NumPy results, reproducible inputs, fair buffer scoring and report output. |

Tests do not assert that NumPy must be faster. Timing comparisons belong to the reproducible benchmark runner, not correctness tests. See the [project README](../README.md) for usage, APIs and performance tables.
