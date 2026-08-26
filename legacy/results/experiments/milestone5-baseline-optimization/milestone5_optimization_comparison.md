# Milestone 5 Baseline Weight Optimization

## Methodology

- Optimizer: deterministic multi-stage random search.
- Objective: mean complete-game score.
- Search depth during optimization: 3 player layers.
- Candidate count: 16, including the original human weights as candidate 0.
- Stage 1: 3 matched training games per candidate; retain 6.
- Stage 2: 8 matched training games per survivor; retain 3.
- Stage 3: 20 matched training games per finalist; retain 1.
- Training seeds: nested subsets beginning at 1000.
- Finalist selection seeds: validation seeds 10000-10024.
- Cross-depth measurement seeds: disjoint validation seeds beginning at 10025.
- Final-test seeds used: no.

Optimization required 262.85 seconds. Candidate 9 won the final training stage
with a 41,800 mean score versus 30,639 for the original weights.

## Frozen optimized weights

| Feature | Original | Optimized |
|---|---:|---:|
| Empty cells | 270.0000 | 85.2216 |
| Monotonicity | 47.0000 | 44.8306 |
| Smoothness | 15.0000 | 43.6767 |
| Corner preference | 100.0000 | 28.7518 |

The optimized policy is preserved separately as `baseline-optimized`; it does
not replace the original `baseline` policy.

## Finalist selection at depth 3

Validation seeds 10000-10024:

| Policy | Games | Mean score | 95% CI | 2048 rate | 4096 rate |
|---|---:|---:|---|---:|---:|
| Original weights | 25 | 27,948.00 | 24,742.97-31,153.03 | 80% | 0% |
| Candidate 4 | 25 | 32,755.04 | 26,935.68-38,574.40 | 72% | 16% |
| Candidate 9 | 25 | 42,276.32 | 35,822.37-48,730.27 | 96% | 28% |

Candidate 9 was frozen before the disjoint cross-depth measurements below.

## Optimized policy across depths

| Depth | Games | Seeds | Mean score | Mean full-game runtime | Search ms/move |
|---:|---:|---|---:|---:|---:|
| 1 | 100 | 10025-10124 | 13,643.32 | 0.0070 sec | 0.0033 |
| 2 | 100 | 10025-10124 | 25,961.20 | 0.1005 sec | 0.0649 |
| 3 | 25 | 10025-10049 | 39,092.48 | 2.0346 sec | 0.9671 |
| 4 | 1 | 10025 | 35,716.00 | 27.4632 sec | 13.8702 |
| 5 | 1 | 10025 | 128,300.00 | 754.6337 sec | 128.9341 |

Mean full-game runtime includes initialization, game logic, and search. Search
milliseconds per move measures Expectimax only. Depths 4 and 5 contain one game
each, so their “mean” runtime is one observation and their scores are not stable
strength estimates.

## Paired original-versus-optimized results

- Depth 1, 75 overlapping unseen seeds: +160.64 mean score, 95% CI
  -2,384.00 to +2,705.28; optimized won 42 games. No reliable gain detected.
- Depth 2, 75 overlapping unseen seeds: +6,256.43 mean score, 95% CI
  +2,044.89 to +10,467.96; optimized won 51 games.
- Depth 3, 25 unseen matched seeds: +10,320.00 mean score, 95% CI
  +2,967.56 to +17,672.44; optimized won 16 games.

The weights optimized at depth 3 transfer strongly to depths 2 and 3, but not
reliably to depth 1. This is evidence of search-depth specialization.

## Methodology assessment

This methodology is correct for producing an empirically optimized depth-3
policy under the tested random-search budget because it uses common training
seeds, held-out selection seeds, a disjoint measurement subset, and preserves
the final-test set. It is not sufficient to claim globally optimal weights.

Additional robustness work should include repeated optimizer seeds, a stronger
black-box optimizer such as CMA-ES, and multi-game depth-4/5 evaluation only if
the compute budget justifies it. Depth 5 currently costs about 12.6 minutes per
game and should not be used during candidate search.
