# Heuristic version history

This is the changelog of record for evaluator versions. Each entry names the
feature set, the weights, where the weights came from, and which
`results/experiments/` folder measured its performance. See
[../../README.md](../../README.md) for how to run these policies and
[../results/README.md](../results/README.md) for the full experiment index.

## v1 — baseline (human-selected weights)

- Code: `BaselineHeuristic` in
  [`src/evaluation/baseline_heuristic.hpp`](../../src/evaluation/baseline_heuristic.hpp).
- Four features, each computed over compacted rows/columns:
  - **Empty cells** — count of available cells.
  - **Monotonicity** — negative ordering violations along each row/column
    (increasing or decreasing order both accepted).
  - **Smoothness** — negative exponent difference between each tile and the
    next occupied tile to its right/below.
  - **Corner preference** — the maximum exponent, if a max tile occupies any
    corner, else zero.
- Weights are hand-picked, not optimized: empty cells 270, monotonicity 47,
  smoothness 15, corner preference 100.
- Measured in
  [`results/experiments/milestone4-baseline-depth-sweep/`](../results/experiments/milestone4-baseline-depth-sweep/).

## v1.1 — optimized baseline

- Same four v1 features; only the weights change
  (`kDepth3OptimizedBaselineWeights`). Feature definitions stay fixed so
  optimization can't silently change what's being measured.
- Weights: empty cells 85.2216, monotonicity 44.8306, smoothness 43.6767,
  corner preference 28.7518.
- Chosen by a deterministic multi-stage random search at depth 3 (16
  candidates, screened at 3 games, retained 6 at 8 games, retained 1 winner
  at 20 games — "candidate 9"), then cross-checked on validation seeds.
  Search log and finalist evaluations:
  [`results/experiments/milestone5-baseline-optimization/`](../results/experiments/milestone5-baseline-optimization/).
  Further cross-depth validation:
  [`results/experiments/milestone5b-baseline-vs-optimized-pilot/`](../results/experiments/milestone5b-baseline-vs-optimized-pilot/),
  [`results/experiments/v1-optimization-heuristic-sweep/`](../results/experiments/v1-optimization-heuristic-sweep/).
- Compiled as the separate `baseline-optimized` policy; `baseline` (v1)
  remains unchanged so the two can always be compared directly.

## v2 — structural heuristic

- Code: `StructuralHeuristic` in
  [`src/evaluation/structural_heuristic.hpp`](../../src/evaluation/structural_heuristic.hpp).
  Keeps the v1.1 baseline weights frozen underneath and adds three features
  anchored at a preferred board corner (bottom-right by default, both
  snake orientations supported):
  - **Main line** — rewards a continuous "snake" ordering of decreasing
    exponents leading away from the corner (adjacency bonus when a tile is
    exactly double its snake-neighbor, penalty otherwise).
  - **Structural stability** — main-line quality evaluated at search leaves.
  - **Adverse-stuck penalty** — penalizes being forced into a move that
    displaces the anchored high-value prefix when no safer move exists.
  - **Structural displacement** — penalizes moves that actually move the
    anchored high-value tiles, evaluated on the transition between boards.
- Initial weights are hand-picked starting points for ablation, not
  optimized: main line 4.0, structural stability 1.0, adverse stuck 2.0,
  structural displacement 2.0.
- Ablated in three stages, each vs. `baseline-optimized` (v1.1), depth 3,
  matched validation seeds:
  `structural-mainline` (main line only) →
  `structural-movement` (+ displacement penalty) →
  `structural-full` (+ stability and adverse-stuck).
  Results:
  [`results/experiments/milestone6-structural-ablation/`](../results/experiments/milestone6-structural-ablation/).
- Design rationale for these features — the "main line" / "target tile" /
  "stuck state" concepts — is in [Design notes](#design-notes-and-open-ideas)
  below.

## v2.1 — jointly optimized structural

- Same eight-weight structure as v2 (`kDepth4OptimizedStructuralWeights`),
  but every weight — baseline and structural — was optimized together
  instead of freezing v1.1 underneath. Joint candidates were normalized to
  the initial total weight, since a common scale factor doesn't change
  Expectimax decisions.
- Weights: baseline {empty 115.8297, monotonicity 37.3024, smoothness
  24.7123, corner 27.4687}; main line 1.3726, structural stability 1.1744,
  adverse stuck 3.0597, structural displacement 0.5609.
- Chosen as "candidate 6" from an exact depth-4 joint optimization on
  training seeds 1200-1219. The code comment on
  `kDepth4OptimizedStructuralWeights` notes it "must be treated as a frozen
  candidate until validation" — the validation referenced there is the
  comparison below.
  Optimizer run:
  [`results/optimization-runs/structural-depth4-joint-optimization/`](../results/optimization-runs/structural-depth4-joint-optimization/).
- Head-to-head against v1/v1.1/v2, both as adaptive (4/6/8 empty-cell
  schedule) and fixed-depth-4 search:
  [`results/experiments/v1-v2.1-adaptive-comparison/`](../results/experiments/v1-v2.1-adaptive-comparison/).

## Search methodology, orthogonal to heuristic version

`v1`/`v1.1`/`v2`/`v2.1` are the canonical policy names used for cross-version
comparisons; they all share the same adaptive player-layer schedule (10-16
empty cells → depth 4, 6-9 → depth 6, 0-5 → depth 8). Earlier ablation runs
used `baseline`/`baseline-optimized`/`structural-mainline`/`structural-movement`/`structural-full`
at a single fixed depth to isolate heuristic quality from search-depth
effects. Both axes matter and are recorded separately in each experiment's
report — don't compare a fixed-depth run against an adaptive one without
checking which mode each side used.

## Design notes and open ideas

Carried over from earlier working notes; these are proposals for a future
v3+, not implemented history:

- **Main line / target tile**: the main-line feature above formalizes part of
  this, but the original proposal also wanted an explicit "target tile" — the
  first increasable tile walking backward from the max corner tile along the
  snake path — and to bias move selection toward directions that increase it.
- **Stuck state / adverse stuck state**: `StructuralHeuristic` implements a
  stuck-state classifier and an adverse-stuck penalty, but the broader idea of
  actively avoiding entering an adverse stuck state (not just penalizing it at
  the leaf) isn't fully built out.
- **Minimizing large-tile displacement**: partially covered by the structural
  displacement penalty; the original framing was more general (prefer
  orientations where large tiles never have to move at all).
- Additional heuristics suggested but not yet implemented: number of
  immediately mergeable pairs, weighted board position, high-tile clustering,
  trapped large-tile penalties, row/column ordering, mobility, board entropy /
  disorder measures.
