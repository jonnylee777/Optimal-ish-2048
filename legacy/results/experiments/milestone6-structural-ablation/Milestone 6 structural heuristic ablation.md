# Milestone 6 structural heuristic ablation

- Experiment plan: depth 3: 10300-10399 (100 games)
- Depths: 3-3
- Search: exact, probability cutoff `0`
- Agents: `baseline-optimized`, `structural-mainline`, `structural-movement`, `structural-full`
- Reference agent: `baseline-optimized`
- Status: complete
- Generated: 2026-08-19T03:26:37+00:00

## Score and runtime summary

| Depth | Agent | Games | Mean score | Difference vs. reference | Improvement | Median | Mean max tile | Highest tile | ms/move |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | `baseline-optimized` | 100 | 41,557.3 | 0.0 | +0.0% | 36,334 | 2,647.0 | 8192 | 0.3857 |
| 3 | `structural-mainline` | 100 | 41,803.0 | 245.6 | +0.6% | 36,534 | 2,647.0 | 8192 | 1.0624 |
| 3 | `structural-movement` | 100 | 42,744.3 | 1,187.0 | +2.9% | 36,416 | 2,606.1 | 4096 | 1.4401 |
| 3 | `structural-full` | 100 | 43,313.2 | 1,755.9 | +4.2% | 36,802 | 2,703.4 | 4096 | 3.9008 |

## Tile achievement rates

| Depth | Agent | 1024 | 2048 | 4096 | 8192 |
|---:|---|---:|---:|---:|---:|
| 3 | `baseline-optimized` | 99.0% | 89.0% | 33.0% | 1.0% |
| 3 | `structural-mainline` | 97.0% | 88.0% | 34.0% | 1.0% |
| 3 | `structural-movement` | 99.0% | 89.0% | 33.0% | 0.0% |
| 3 | `structural-full` | 100.0% | 90.0% | 37.0% | 0.0% |

## Paired-seed outcomes

| Depth | Comparison | Candidate wins | Reference wins | Ties | Mean paired difference |
|---:|---|---:|---:|---:|---:|
| 3 | `structural-mainline` vs. `baseline-optimized` | 49 | 51 | 0 | 245.6 |
| 3 | `structural-movement` vs. `baseline-optimized` | 52 | 48 | 0 | 1,187.0 |
| 3 | `structural-full` vs. `baseline-optimized` | 54 | 45 | 1 | 1,755.9 |

## Source result files

- Depth 3, `baseline-optimized`: [baseline_optimized_depth3_seeds10300-10399_20260819T030240624Z.json](baseline_optimized_depth3_seeds10300-10399_20260819T030240624Z.json)
- Depth 3, `structural-mainline`: [structural_mainline_depth3_seeds10300-10399_20260819T030619544Z.json](structural_mainline_depth3_seeds10300-10399_20260819T030619544Z.json)
- Depth 3, `structural-movement`: [structural_movement_depth3_seeds10300-10399_20260819T031123898Z.json](structural_movement_depth3_seeds10300-10399_20260819T031123898Z.json)
- Depth 3, `structural-full`: [structural_full_depth3_seeds10300-10399_20260819T032515434Z.json](structural_full_depth3_seeds10300-10399_20260819T032515434Z.json)

## Interpretation note

Each comparison uses matched seeds to reduce variance. Interpret results together with sample size and score spread, then validate the selected policy on unused seeds.
