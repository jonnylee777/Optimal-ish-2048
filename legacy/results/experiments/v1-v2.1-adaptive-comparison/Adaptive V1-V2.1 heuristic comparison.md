# Adaptive V1-V2.1 heuristic comparison

- Experiment plan: depth 8: 10400-10402 (3 games)
- Depth: Adaptive 4/6/8
- Search: time-bounded iterative deepening, probability cutoff `0`
- Per-move time limit: 250 ms
- Agents: `v1`, `v1.1`, `v2`, `v2.1`
- Reference agent: `v1`
- Status: complete
- Generated: 2026-08-20T15:41:59+00:00

## Score and runtime summary

| Depth | Agent | Games | Average score | Difference vs. reference | Improvement | Maximum tile | Mode maximum tile | Runtime | ms/move |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Adaptive 4/6/8 | `v1` | 3 | 82,222.7 | 0.0 | +0.0% | 8192 | 8192 | 45.0 min | 243.5542 |
| Adaptive 4/6/8 | `v1.1` | 3 | 123,641.3 | 41,418.7 | +50.4% | 8192 | 8192 | 1.23 h | 284.2666 |
| Adaptive 4/6/8 | `v2` | 3 | 110,177.3 | 27,954.7 | +34.0% | 8192 | 8192 | 58.7 min | 248.7562 |
| Adaptive 4/6/8 | `v2.1` | 3 | 121,308.0 | 39,085.3 | +47.5% | 8192 | 8192 | 1.07 h | 248.2310 |

## Adaptive depth tracking

Requested depth is the adaptive ceiling selected from the board's empty-cell count. Completed depth is the last iterative-deepening level finished within the per-move time limit.

### Requested maximum depth

| Agent | Total moves | Depth 4 requested | Depth 6 requested | Depth 8 requested |
|---|---:|---:|---:|---:|
| `v1` | 11,076 | 164 (1.5%) | 4,504 (40.7%) | 6,408 (57.9%) |
| `v1.1` | 15,548 | 160 (1.0%) | 5,722 (36.8%) | 9,666 (62.2%) |
| `v2` | 14,158 | 104 (0.7%) | 4,726 (33.4%) | 9,328 (65.9%) |
| `v2.1` | 15,567 | 145 (0.9%) | 4,833 (31.0%) | 10,589 (68.0%) |

### Actually completed depth

| Agent | Average | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 | Depth 6 | Depth 7 | Depth 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v1` | 5.27 | 0 | 2 | 11 | 350 | 8,206 | 1,909 | 386 | 212 |
| `v1.1` | 5.23 | 0 | 5 | 31 | 695 | 11,473 | 2,597 | 480 | 267 |
| `v2` | 4.41 | 0 | 0 | 10 | 9,372 | 4,023 | 562 | 136 | 55 |
| `v2.1` | 4.48 | 0 | 0 | 0 | 9,502 | 5,032 | 740 | 214 | 79 |

## Tile achievement rates

| Depth | Agent | 1024 | 2048 | 4096 | 8192 |
|---:|---|---:|---:|---:|---:|
| 8 | `v1` | 100.0% | 100.0% | 66.7% | 33.3% |
| 8 | `v1.1` | 100.0% | 100.0% | 100.0% | 100.0% |
| 8 | `v2` | 100.0% | 100.0% | 100.0% | 66.7% |
| 8 | `v2.1` | 100.0% | 100.0% | 100.0% | 66.7% |

## Paired-seed outcomes

| Depth | Comparison | Candidate wins | Reference wins | Ties | Mean paired difference |
|---:|---|---:|---:|---:|---:|
| 8 | `v1.1` vs. `v1` | 2 | 1 | 0 | 41,418.7 |
| 8 | `v2` vs. `v1` | 2 | 1 | 0 | 27,954.7 |
| 8 | `v2.1` vs. `v1` | 1 | 2 | 0 | 39,085.3 |

## Source result files

- Depth 8, `v1`: [v1_depth8_seeds10400-10402_20260820T043103039Z.json](v1_depth8_seeds10400-10402_20260820T043103039Z.json)
- Depth 8, `v1.1`: [v1_1_depth8_seeds10400-10402_20260820T054443987Z.json](v1_1_depth8_seeds10400-10402_20260820T054443987Z.json)
- Depth 8, `v2`: [v2_depth8_seeds10400-10402_20260820T064326743Z.json](v2_depth8_seeds10400-10402_20260820T064326743Z.json)
- Depth 8, `v2.1`: [v2_1_depth8_seeds10400-10402_20260820T074751885Z.json](v2_1_depth8_seeds10400-10402_20260820T074751885Z.json)

## Interpretation note

Each comparison uses matched seeds to reduce variance. Interpret results together with sample size and score spread, then validate the selected policy on unused seeds.

This small game sample is a pilot comparison, not a robust estimate of expected playing strength. Matched seeds improve fairness but do not eliminate sampling uncertainty.
