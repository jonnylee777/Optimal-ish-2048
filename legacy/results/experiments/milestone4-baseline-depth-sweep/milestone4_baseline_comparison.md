# Milestone 4 Baseline Comparison

Build: Release  
Evaluator: baseline-v1  
Weights: empty cells 270, monotonicity 47, smoothness 15, corner preference 100  
Seed partition: validation  
Final-test seeds used: no

## Recorded experiments

| Depth | Games | Seeds | Mean score | Median | 95% CI | Best | 2048 rate | 4096 rate | Search ms/move |
|---:|---:|---|---:|---:|---|---:|---:|---:|---:|
| 1 | 100 | 10000-10099 | 12,931.52 | 11,554 | 11,321.98-14,541.06 | 52,504 | 16% | 1% | 0.0033 |
| 2 | 100 | 10000-10099 | 18,926.44 | 15,784 | 16,687.21-21,165.67 | 56,152 | 33% | 5% | 0.0640 |
| 3 | 25 | 10000-10024 | 27,948.00 | 29,040 | 24,742.97-31,153.03 | 37,184 | 80% | 0% | 0.9755 |
| 4 | 1 | 10000 | 36,168.00 | 36,168 | not estimated | 36,168 | 100% | 0% | 11.3154 |
| 5 | 1 | 10000 | 60,428.00 | 60,428 | not estimated | 60,428 | 100% | 100% | 149.7571 |

Depth 3 used fewer games because its measured search cost is roughly 15 times
depth 2 and 296 times depth 1. Its maximum-tile percentages should not be
compared directly with the 100-game rows without accounting for sample size.
Depths 4 and 5 are performance measurements on one game only. Their scores and
tile rates are not statistically stable estimates of playing strength.

## Depth 4 and 5 performance

| Depth | Full-game runtime | Moves | Search nodes | Nodes/sec | Cache hit rate |
|---:|---:|---:|---:|---:|---:|
| 4 | 22.80 seconds | 1,865 | 402,288,288 | 19.06 million | 53.84% |
| 5 | 462.32 seconds | 2,802 | 6,779,853,129 | 16.16 million | 55.75% |

Depth 5 took 7.71 minutes for one game. It was 20.3 times slower per complete
game and 13.2 times slower per searched move than depth 4. A 25-game depth-5
sample would take approximately 3.2 hours at this observed rate; 100 games
would take approximately 12.8 hours.

For seed 10000 specifically, the scores by depth were 14,496, 31,796, 25,804,
36,168, and 60,428 for depths 1 through 5. A deeper search does not have to win
every individual seed, as shown by depth 3 versus depth 2, so multi-seed samples
remain necessary for strength claims.

## Matched first 25 seeds

| Depth | Mean score | Median score |
|---:|---:|---:|
| 1 | 11,166.88 | 10,368 |
| 2 | 20,188.80 | 16,056 |
| 3 | 27,948.00 | 29,040 |

Paired score differences:

- Depth 2 minus depth 1 over 100 seeds: +5,994.92 mean, 95% CI
  +3,384.62 to +8,605.22; depth 2 won 73 of 100 games.
- Depth 3 minus depth 2 over 25 seeds: +7,759.20 mean, 95% CI
  +3,070.71 to +12,447.69; depth 3 won 18 of 25 games.
- Depth 3 minus depth 1 over 25 seeds: +16,781.12 mean, 95% CI
  +13,146.88 to +20,415.36; depth 3 won all 25 games.

These results show a clear strength increase with depth under matched seeds,
alongside a steep compute increase.

## Milestone 5 compute recommendation

Use a multi-fidelity optimization budget:

1. Screen many weight candidates at depth 1 on a small matched subset of
   training seeds.
2. Re-evaluate promising candidates at depth 2 on a larger training subset.
3. Select finalists using validation seeds at depth 2.
4. Use depth 3 only for a small final comparison of the strongest candidates.

Do not use the reserved final-test seeds during weight optimization or model
selection.
