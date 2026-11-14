# Tests

This folder covers both assignments. The suite currently has **401 passing cases**: 215 for the original NumCompute package and 186 for streaming extensions. Related edge cases are grouped in test functions. The old tests remain in place to catch regressions.

From the project root:

```bash
python -m pytest tests -q
python -m pytest tests -q -k test_stream
python -m pytest tests/test_stream_tree.py -q
```

## Streaming coverage

| File | Main checks |
| --- | --- |
| `test_stream_tree.py` | Splits, NaNs, ties, full-tree loop comparisons over multiple streams, buffer thresholds/clearing, prediction warm-up, pending rows and lossless label types. |
| `test_stream_preprocessing.py` | Chunk/batch agreement, valid counts, missing/constant columns, category growth, reset, sliced inputs and extreme finite scaling/imputation under strict floating-point checks. |
| `test_stream_stats.py` | Running moments, exact distribution queries, independent numerical expectations, memory mode, overflow rejection, NaN recovery and huge/tiny constants. |
| `test_stream_metrics.py` | Cumulative and rolling results, independent count/pairwise-AUC oracles, class order/types, zero denominators, extreme finite MSE/AUC, expired rows and reset. |
| `test_stream_pipeline.py` | Original batch API, incremental step order, nested pipelines, label routing, empty chunks and error propagation. |
| `test_stream_trainer.py` | CSV-to-pipeline integration, score-before-train order, weighted accuracy, model readiness, log snapshots, label rejection and memory estimates. |
| `test_stream_visualise.py` | Headless figures, NaN gaps, exact category axes, inline-display dispatch, PNG/SVG saving and invalid input. |
| `test_stream_benchmarking.py` | Matching loop/NumPy results, reproducible inputs, fair buffer scoring, buffer-order/repeat independence, short final chunks and report output. |

## Numerical regressions and API coverage

`test_utils.py` checks mixed-scale distances with an exact expected distance of 1, tiny/large finite norms, and mixed infinite/NaN reductions under `np.errstate(all="raise")`. `test_preprocessing.py` rejects NaN/infinite/complex scaler endpoints and checks finite scaling and median results at extreme magnitudes. The streaming tests exercise the same shared paths and verify that invalid updates preserve metric/model state.

The suite executes all public A2.2 functions, concrete methods and query properties, including reused batch interfaces. Protocol declarations describe interfaces and have no implementation to execute. This is API coverage, not a claim that every branch or possible input has been exhausted. Independent oracles avoid relying only on comparisons with the previous package, where the same numerical bug could otherwise appear on both sides.

Tests do not assert that NumPy must be faster. Timing comparisons belong to the reproducible benchmark runner, not correctness tests. See the [project README](../README.md) for usage, APIs and performance tables.
