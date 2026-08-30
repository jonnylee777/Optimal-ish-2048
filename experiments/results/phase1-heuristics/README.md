# Phase 1 — Hand-written heuristics (H0–H5)

Human-designed board scores driven by expectimax search. No learning.

Both benchmark regimes live here, deliberately: comparing fixed-depth against
timed play *for the same agent* is the point, so splitting them by regime would
separate the two halves of every comparison.

| Agent | Features |
|---|---|
| H0 | empty cells, edge/corner max bonus |
| H1 | + monotonicity, smoothness, corner preference |
| H2 | + "corner chain" around the max tile |
| H3 | + snake / main-line structure |
| H4 | port of nneonneo's evaluator |
| H5 | port of the endgame-tablebase project's evaluator |

**H5 is the strongest and among the cheapest** — 109,213 at fixed depth 4 (n=40)
at 5.0 ms/move, against H3's 57,318 at 71.8 ms/move.

**H0–H4 are n=10 with overlapping intervals**, so their relative ordering is not
established. Only H5-vs-H3 was rerun at n=40 (+90.5%, p=1e-10). Any table
quoting the others must carry that caveat.

Scores: [../../../experiment_results.md](../../../experiment_results.md).
