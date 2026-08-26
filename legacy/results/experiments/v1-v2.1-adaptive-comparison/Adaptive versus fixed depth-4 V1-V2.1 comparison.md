# Adaptive versus fixed depth-4 V1-V2.1 comparison

- Experiment plan: depth 4: 10400-10402 (3 games), depth 8: 10400-10402 (3 games)
- Depth modes: Fixed depth 4, Adaptive 4/6/8
- Search: mixed fixed-depth and time-bounded iterative deepening, probability cutoff `0`
- Per-move time limits: Fixed depth 4=none, Adaptive 4/6/8=250 ms
- Agents: `v1`, `v1.1`, `v2`, `v2.1`
- Reference agent: `v1`
- Status: complete
- Generated: 2026-08-20T17:43:48+00:00

## Score and runtime summary

| Depth | Agent | Games | Average score | Difference vs. reference | Improvement | Maximum tile | Mode maximum tile | Runtime | ms/move |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed depth 4 | `v1` | 3 | 61,397.3 | 0.0 | +0.0% | 8192 | 2048 | 52.72 s | 6.2775 |
| Fixed depth 4 | `v1.1` | 3 | 64,228.0 | 2,830.7 | +4.6% | 4096 | 4096 | 55.15 s | 6.0426 |
| Fixed depth 4 | `v2` | 3 | 39,958.7 | -21,438.7 | -34.9% | 4096 | 4096 | 5.2 min | 52.6171 |
| Fixed depth 4 | `v2.1` | 3 | 61,998.7 | 601.3 | +1.0% | 4096 | 4096 | 7.7 min | 52.3256 |
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
| Fixed depth 4 | `v1` | 100.0% | 100.0% | 33.3% | 33.3% |
| Fixed depth 4 | `v1.1` | 100.0% | 100.0% | 66.7% | 0.0% |
| Fixed depth 4 | `v2` | 100.0% | 66.7% | 33.3% | 0.0% |
| Fixed depth 4 | `v2.1` | 100.0% | 100.0% | 66.7% | 0.0% |
| Adaptive 4/6/8 | `v1` | 100.0% | 100.0% | 66.7% | 33.3% |
| Adaptive 4/6/8 | `v1.1` | 100.0% | 100.0% | 100.0% | 100.0% |
| Adaptive 4/6/8 | `v2` | 100.0% | 100.0% | 100.0% | 66.7% |
| Adaptive 4/6/8 | `v2.1` | 100.0% | 100.0% | 100.0% | 66.7% |

## Paired-seed outcomes

| Depth | Comparison | Candidate wins | Reference wins | Ties | Mean paired difference |
|---:|---|---:|---:|---:|---:|
| Fixed depth 4 | `v1.1` vs. `v1` | 2 | 1 | 0 | 2,830.7 |
| Fixed depth 4 | `v2` vs. `v1` | 1 | 2 | 0 | -21,438.7 |
| Fixed depth 4 | `v2.1` vs. `v1` | 1 | 1 | 1 | 601.3 |
| Adaptive 4/6/8 | `v1.1` vs. `v1` | 2 | 1 | 0 | 41,418.7 |
| Adaptive 4/6/8 | `v2` vs. `v1` | 2 | 1 | 0 | 27,954.7 |
| Adaptive 4/6/8 | `v2.1` vs. `v1` | 1 | 2 | 0 | 39,085.3 |

## Fixed versus adaptive search mode

Adaptive outcomes are compared with Fixed depth 4 for the same policy and matched seed.

| Agent | Adaptive wins | Fixed wins | Ties | Mean adaptive score difference | Fixed ms/move | Adaptive ms/move |
|---|---:|---:|---:|---:|---:|---:|
| `v1` | 2 | 1 | 0 | 20,825.3 | 6.2775 | 243.5542 |
| `v1.1` | 3 | 0 | 0 | 59,413.3 | 6.0426 | 284.2666 |
| `v2` | 3 | 0 | 0 | 70,218.7 | 52.6171 | 248.7562 |
| `v2.1` | 2 | 1 | 0 | 59,309.3 | 52.3256 | 248.2310 |

## Source result files

- Fixed depth 4, `v1`: [v1_depth4_seeds10400-10402_20260820T172959775Z.json](v1_depth4_seeds10400-10402_20260820T172959775Z.json)
- Fixed depth 4, `v1.1`: [v1_1_depth4_seeds10400-10402_20260820T173055979Z.json](v1_1_depth4_seeds10400-10402_20260820T173055979Z.json)
- Fixed depth 4, `v2`: [v2_depth4_seeds10400-10402_20260820T173608641Z.json](v2_depth4_seeds10400-10402_20260820T173608641Z.json)
- Fixed depth 4, `v2.1`: [v2_1_depth4_seeds10400-10402_20260820T174348784Z.json](v2_1_depth4_seeds10400-10402_20260820T174348784Z.json)
- Adaptive 4/6/8, `v1`: [v1_depth8_seeds10400-10402_20260820T043103039Z.json](v1_depth8_seeds10400-10402_20260820T043103039Z.json)
- Adaptive 4/6/8, `v1.1`: [v1_1_depth8_seeds10400-10402_20260820T054443987Z.json](v1_1_depth8_seeds10400-10402_20260820T054443987Z.json)
- Adaptive 4/6/8, `v2`: [v2_depth8_seeds10400-10402_20260820T064326743Z.json](v2_depth8_seeds10400-10402_20260820T064326743Z.json)
- Adaptive 4/6/8, `v2.1`: [v2_1_depth8_seeds10400-10402_20260820T074751885Z.json](v2_1_depth8_seeds10400-10402_20260820T074751885Z.json)

## Interpretation note

Each comparison uses matched seeds to reduce variance. Interpret results together with sample size and score spread, then validate the selected policy on unused seeds.

This small game sample is a pilot comparison, not a robust estimate of expected playing strength. Matched seeds improve fairness but do not eliminate sampling uncertainty.
