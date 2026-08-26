# Baseline vs. optimized heuristic comparison

- Seeds: 10125-10129
- Games per agent/depth: 5
- Depths: 1-5
- Search: exact, probability cutoff `0`
- Status: partial (8/10 runs found)
- Generated: 2026-08-18T20:07:12+00:00

## Score and runtime summary

| Depth | Baseline mean | Optimized mean | Mean difference | Improvement | Baseline median | Optimized median | Baseline max tile | Optimized max tile | Baseline ms/move | Optimized ms/move |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13,222.4 | 16,753.6 | 3,531.2 | +26.7% | 7,432 | 14,444 | 2048 | 2048 | 0.0011 | 0.0012 |
| 2 | 23,412.8 | 23,067.2 | -345.6 | -1.5% | 16,064 | 16,588 | 4096 | 2048 | 0.0145 | 0.0156 |
| 3 | 30,624.0 | 41,151.2 | 10,527.2 | +34.4% | 25,412 | 38,308 | 4096 | 4096 | 0.2544 | 0.2467 |
| 4 | 37,800.0 | 67,934.4 | 30,134.4 | +79.7% | 36,252 | 70,940 | 4096 | 4096 | 4.0647 | 3.8198 |
| 5 | pending | pending | — | — | — | — | — | — | — | — |

## Paired-seed outcomes

| Depth | Optimized wins | Baseline wins | Ties | Mean paired score difference |
|---:|---:|---:|---:|---:|
| 1 | 3 | 2 | 0 | 3,531.2 |
| 2 | 2 | 3 | 0 | -345.6 |
| 3 | 4 | 1 | 0 | 10,527.2 |
| 4 | 4 | 1 | 0 | 30,134.4 |
| 5 | pending | pending | pending | pending |

## Source result files

- Depth 1, `baseline`: [baseline_depth1_seeds10125-10129_20260818T200442859Z.json](baseline_depth1_seeds10125-10129_20260818T200442859Z.json)
- Depth 1, `baseline-optimized`: [baseline_optimized_depth1_seeds10125-10129_20260818T200442868Z.json](baseline_optimized_depth1_seeds10125-10129_20260818T200442868Z.json)
- Depth 2, `baseline`: [baseline_depth2_seeds10125-10129_20260818T200442959Z.json](baseline_depth2_seeds10125-10129_20260818T200442959Z.json)
- Depth 2, `baseline-optimized`: [baseline_optimized_depth2_seeds10125-10129_20260818T200443061Z.json](baseline_optimized_depth2_seeds10125-10129_20260818T200443061Z.json)
- Depth 3, `baseline`: [baseline_depth3_seeds10125-10129_20260818T200445068Z.json](baseline_depth3_seeds10125-10129_20260818T200445068Z.json)
- Depth 3, `baseline-optimized`: [baseline_optimized_depth3_seeds10125-10129_20260818T200447611Z.json](baseline_optimized_depth3_seeds10125-10129_20260818T200447611Z.json)
- Depth 4, `baseline`: [baseline_depth4_seeds10125-10129_20260818T200525859Z.json](baseline_depth4_seeds10125-10129_20260818T200525859Z.json)
- Depth 4, `baseline-optimized`: [baseline_optimized_depth4_seeds10125-10129_20260818T200625203Z.json](baseline_optimized_depth4_seeds10125-10129_20260818T200625203Z.json)
- Depth 5, `baseline`: pending
- Depth 5, `baseline-optimized`: pending

## Interpretation note

Five games per condition are a pilot comparison. Paired seeds reduce variance, but larger validation samples are required before treating small score differences as reliable.
