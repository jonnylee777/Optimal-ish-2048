# V1 optimization comparison

- Experiment plan: depth 1: 10200-10299 (100 games), depth 2: 10200-10299 (100 games), depth 3: 10200-10299 (100 games), depth 4: 10200-10229 (30 games), depth 5: 10200-10204 (5 games)
- Depths: 1-5
- Search: exact, probability cutoff `0`
- Status: complete
- Generated: 2026-08-18T20:49:03+00:00

## Score and runtime summary

| Depth | Baseline mean | Optimized mean | Mean difference | Improvement | Baseline median | Optimized median | Baseline max tile | Optimized max tile | Baseline ms/move | Optimized ms/move |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11,473.0 | 11,989.9 | 516.9 | +4.5% | 10,168 | 12,024 | 2048 | 2048 | 0.0008 | 0.0008 |
| 2 | 17,856.2 | 29,232.0 | 11,375.8 | +63.7% | 16,068 | 28,426 | 4096 | 4096 | 0.0151 | 0.0139 |
| 3 | 26,865.6 | 39,013.5 | 12,147.9 | +45.2% | 27,280 | 35,558 | 4096 | 4096 | 0.2537 | 0.2405 |
| 4 | 39,997.6 | 51,333.1 | 11,335.5 | +28.3% | 36,192 | 47,502 | 4096 | 8192 | 4.0904 | 3.9737 |
| 5 | 42,283.2 | 83,253.6 | 40,970.4 | +96.9% | 36,568 | 78,824 | 4096 | 8192 | 50.6218 | 43.5451 |

## Tile achievement rates

| Depth | Baseline 1024 | Optimized 1024 | Baseline 2048 | Optimized 2048 | Baseline 4096 | Optimized 4096 | Baseline 8192 | Optimized 8192 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50.0% | 59.0% | 11.0% | 8.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| 2 | 77.0% | 95.0% | 32.0% | 73.0% | 2.0% | 10.0% | 0.0% | 0.0% |
| 3 | 99.0% | 98.0% | 64.0% | 84.0% | 6.0% | 32.0% | 0.0% | 0.0% |
| 4 | 100.0% | 100.0% | 80.0% | 93.3% | 36.7% | 50.0% | 0.0% | 3.3% |
| 5 | 100.0% | 100.0% | 80.0% | 100.0% | 40.0% | 100.0% | 0.0% | 20.0% |

## Paired-seed outcomes

| Depth | Optimized wins | Baseline wins | Ties | Mean paired score difference |
|---:|---:|---:|---:|---:|
| 1 | 56 | 44 | 0 | 516.9 |
| 2 | 78 | 22 | 0 | 11,375.8 |
| 3 | 71 | 29 | 0 | 12,147.9 |
| 4 | 19 | 11 | 0 | 11,335.5 |
| 5 | 5 | 0 | 0 | 40,970.4 |

## Source result files

- Depth 1, `baseline`: [baseline_depth1_seeds10200-10299_20260818T201630469Z.json](baseline_depth1_seeds10200-10299_20260818T201630469Z.json)
- Depth 1, `baseline-optimized`: [baseline_optimized_depth1_seeds10200-10299_20260818T201630530Z.json](baseline_optimized_depth1_seeds10200-10299_20260818T201630530Z.json)
- Depth 2, `baseline`: [baseline_depth2_seeds10200-10299_20260818T201632083Z.json](baseline_depth2_seeds10200-10299_20260818T201632083Z.json)
- Depth 2, `baseline-optimized`: [baseline_optimized_depth2_seeds10200-10299_20260818T201634196Z.json](baseline_optimized_depth2_seeds10200-10299_20260818T201634196Z.json)
- Depth 3, `baseline`: [baseline_depth3_seeds10200-10299_20260818T201710094Z.json](baseline_depth3_seeds10200-10299_20260818T201710094Z.json)
- Depth 3, `baseline-optimized`: [baseline_optimized_depth3_seeds10200-10299_20260818T201756390Z.json](baseline_optimized_depth3_seeds10200-10299_20260818T201756390Z.json)
- Depth 4, `baseline`: [baseline_depth4_seeds10200-10229_20260818T202158742Z.json](baseline_depth4_seeds10200-10229_20260818T202158742Z.json)
- Depth 4, `baseline-optimized`: [baseline_optimized_depth4_seeds10200-10229_20260818T202650997Z.json](baseline_optimized_depth4_seeds10200-10229_20260818T202650997Z.json)
- Depth 5, `baseline`: [baseline_depth5_seeds10200-10204_20260818T203535102Z.json](baseline_depth5_seeds10200-10204_20260818T203535102Z.json)
- Depth 5, `baseline-optimized`: [baseline_optimized_depth5_seeds10200-10204_20260818T204903216Z.json](baseline_optimized_depth5_seeds10200-10204_20260818T204903216Z.json)

## Interpretation note

Each comparison uses matched seeds. Depths with larger samples support more reliable score estimates; depth 5 remains a five-game pilot and should be interpreted cautiously.
