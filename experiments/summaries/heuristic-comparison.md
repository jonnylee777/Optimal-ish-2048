# Heuristic comparison (H0-H5)

- Generated: 2026-08-24T05:35:09+00:00
- Result files scanned: 10

## Score and runtime summary

| Config | Heuristic | Games | Mean score | Median | Min/Max | Highest tile | ms/move | Deadline hit rate |
|---|---|---:|---:|---:|---|---:|---:|---:|
| adaptive, 250 ms/move, seeds 20500-20502 | `H0` | 3 | 47,694.7 | 36,164.0 | 27,012/79,908 | 4,096 | 243.6860 | n/a |
| adaptive, 250 ms/move, seeds 20500-20502 | `H1` | 3 | 64,090.7 | 58,984.0 | 56,688/76,600 | 4,096 | 443.1336 | n/a |
| adaptive, 250 ms/move, seeds 20500-20502 | `H2` | 3 | 84,716.0 | 72,552.0 | 67,648/113,948 | 8,192 | 247.0407 | n/a |
| adaptive, 250 ms/move, seeds 20500-20502 | `H3` | 3 | 74,788.0 | 56,224.0 | 36,760/131,380 | 8,192 | 248.9728 | n/a |
| adaptive, 250 ms/move, seeds 20500-20502 | `H4` | 3 | 46,224.0 | 50,588.0 | 34,328/53,756 | 4,096 | 249.6673 | n/a |
| depth 4, exact, seeds 20000-20009 | `H0` | 10 | 26,769.2 | 27,328.0 | 16,080/36,244 | 2,048 | 6.3430 | 0.0% |
| depth 4, exact, seeds 20000-20009 | `H1` | 10 | 36,141.6 | 36,378.0 | 15,116/60,828 | 4,096 | 7.2246 | 0.0% |
| depth 4, exact, seeds 20000-20009 | `H2` | 10 | 34,910.4 | 34,612.0 | 16,400/57,140 | 4,096 | 7.5775 | 0.0% |
| depth 4, exact, seeds 20000-20009 | `H3` | 10 | 49,982.4 | 54,124.0 | 32,392/71,588 | 4,096 | 46.6574 | 0.0% |
| depth 4, exact, seeds 20000-20009 | `H4` | 10 | 45,493.2 | 46,156.0 | 15,016/76,940 | 4,096 | 4.7096 | 0.0% |

## Paired-seed outcomes (within matching configs)

| Config | Comparison | Wins | Losses | Ties | Mean paired difference |
|---|---|---:|---:|---:|---:|
| adaptive, 250 ms/move, seeds 20500-20502 | `H1` vs. `H0` | 2 | 1 | 0 | 16,396.0 |
| adaptive, 250 ms/move, seeds 20500-20502 | `H2` vs. `H0` | 2 | 1 | 0 | 37,021.3 |
| adaptive, 250 ms/move, seeds 20500-20502 | `H3` vs. `H0` | 2 | 1 | 0 | 27,093.3 |
| adaptive, 250 ms/move, seeds 20500-20502 | `H4` vs. `H0` | 2 | 1 | 0 | -1,470.7 |
| adaptive, 250 ms/move, seeds 20500-20502 | `H2` vs. `H1` | 2 | 1 | 0 | 20,625.3 |
| adaptive, 250 ms/move, seeds 20500-20502 | `H3` vs. `H1` | 1 | 2 | 0 | 10,697.3 |
| adaptive, 250 ms/move, seeds 20500-20502 | `H4` vs. `H1` | 0 | 3 | 0 | -17,866.7 |
| adaptive, 250 ms/move, seeds 20500-20502 | `H3` vs. `H2` | 1 | 2 | 0 | -9,928.0 |
| adaptive, 250 ms/move, seeds 20500-20502 | `H4` vs. `H2` | 0 | 3 | 0 | -38,492.0 |
| adaptive, 250 ms/move, seeds 20500-20502 | `H4` vs. `H3` | 1 | 2 | 0 | -28,564.0 |
| depth 4, exact, seeds 20000-20009 | `H1` vs. `H0` | 8 | 2 | 0 | 9,372.4 |
| depth 4, exact, seeds 20000-20009 | `H2` vs. `H0` | 9 | 1 | 0 | 8,141.2 |
| depth 4, exact, seeds 20000-20009 | `H3` vs. `H0` | 10 | 0 | 0 | 23,213.2 |
| depth 4, exact, seeds 20000-20009 | `H4` vs. `H0` | 8 | 2 | 0 | 18,724.0 |
| depth 4, exact, seeds 20000-20009 | `H2` vs. `H1` | 6 | 4 | 0 | -1,231.2 |
| depth 4, exact, seeds 20000-20009 | `H3` vs. `H1` | 9 | 1 | 0 | 13,840.8 |
| depth 4, exact, seeds 20000-20009 | `H4` vs. `H1` | 7 | 3 | 0 | 9,351.6 |
| depth 4, exact, seeds 20000-20009 | `H3` vs. `H2` | 8 | 2 | 0 | 15,072.0 |
| depth 4, exact, seeds 20000-20009 | `H4` vs. `H2` | 7 | 3 | 0 | 10,582.8 |
| depth 4, exact, seeds 20000-20009 | `H4` vs. `H3` | 5 | 5 | 0 | -4,489.2 |

## Source result files

- adaptive, 250 ms/move, seeds 20500-20502, `H0`: [H0_timed_depth8_seeds20500-20502_20260823T190313758Z.json](../results/phase1-heuristics/H0_timed_depth8_seeds20500-20502_20260823T190313758Z.json)
- adaptive, 250 ms/move, seeds 20500-20502, `H1`: [H1_timed_depth8_seeds20500-20502_20260823T070852857Z.json](../results/phase1-heuristics/H1_timed_depth8_seeds20500-20502_20260823T070852857Z.json)
- adaptive, 250 ms/move, seeds 20500-20502, `H2`: [H2_timed_depth8_seeds20500-20502_20260823T194906740Z.json](../results/phase1-heuristics/H2_timed_depth8_seeds20500-20502_20260823T194906740Z.json)
- adaptive, 250 ms/move, seeds 20500-20502, `H3`: [H3_timed_depth8_seeds20500-20502_20260823T203041683Z.json](../results/phase1-heuristics/H3_timed_depth8_seeds20500-20502_20260823T203041683Z.json)
- adaptive, 250 ms/move, seeds 20500-20502, `H4`: [H4_timed_depth8_seeds20500-20502_20260823T205752327Z.json](../results/phase1-heuristics/H4_timed_depth8_seeds20500-20502_20260823T205752327Z.json)
- depth 4, exact, seeds 20000-20009, `H0`: [H0_fixed_depth4_seeds20000-20009_20260823T190202584Z.json](../results/phase1-heuristics/H0_fixed_depth4_seeds20000-20009_20260823T190202584Z.json)
- depth 4, exact, seeds 20000-20009, `H1`: [H1_fixed_depth4_seeds20000-20009_20260823T053550134Z.json](../results/phase1-heuristics/H1_fixed_depth4_seeds20000-20009_20260823T053550134Z.json)
- depth 4, exact, seeds 20000-20009, `H2`: [H2_fixed_depth4_seeds20000-20009_20260823T055116344Z.json](../results/phase1-heuristics/H2_fixed_depth4_seeds20000-20009_20260823T055116344Z.json)
- depth 4, exact, seeds 20000-20009, `H3`: [H3_fixed_depth4_seeds20000-20009_20260823T061807190Z.json](../results/phase1-heuristics/H3_fixed_depth4_seeds20000-20009_20260823T061807190Z.json)
- depth 4, exact, seeds 20000-20009, `H4`: [H4_fixed_depth4_seeds20000-20009_20260823T184955535Z.json](../results/phase1-heuristics/H4_fixed_depth4_seeds20000-20009_20260823T184955535Z.json)
