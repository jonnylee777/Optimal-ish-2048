# Experiment taxonomy

This replaces the ambiguous `v1`/`v1.1`/`v2`/`v2.1` milestone naming for all
new work. It separates three independent axes — **heuristic**, **search**,
**weights** — so changing one never silently changes another. See
[`../ROADMAP.md`](../ROADMAP.md) for what isn't built yet, and
[`../legacy/docs/heuristics.md`](../legacy/docs/heuristics.md) for how the
old names map onto history.

## Heuristics (evaluation functions)

| Name | Class | Features | Status |
|---|---|---|---|
| H0 | `H0Heuristic` (`src/evaluation/h0_heuristic.hpp`) | empty cells, edge/corner max-value bonus | implemented, row-table accelerated (42% node/sec gain, verified bit-identical scores — see `tests/h0_heuristic_tests.cpp`) |
| H1 | `BaselineHeuristic` (`src/evaluation/baseline_heuristic.hpp`) | empty cells, monotonicity, smoothness, corner preference | implemented (unchanged from legacy `v1`) |
| H2 | `H2Heuristic` (`src/evaluation/h2_heuristic.hpp`) | H1's four features, plus "corner chain": rewards the max tile's two orthogonal neighbors for holding exactly the next-largest value | implemented |
| H3 | `H3Heuristic` (`src/evaluation/h3_heuristic.hpp`) | H2's five features, plus the snake/main-line structure: corner-anchored main line, structural stability, adverse-stuck penalty, and a move-transition displacement penalty (reuses the tested functions in `structural_heuristic.hpp`) | implemented — best fixed-depth score, but ~10x the per-move cost of H4 |
| H4 | `H4Heuristic` (`src/evaluation/h4_heuristic.hpp`) | empty cells, merges (adjacent-equal runs), monotonicity cost `pow(rank,4)`, tile-sum cost `pow(rank,3.5)` — all computed per row over 4 rows + 4 columns | implemented — **standalone port**, see note below |
| H5 | `H5Heuristic` (`src/evaluation/h5_heuristic.hpp`) | saturating tile weights, per-line max(DPDF, T-formation), per-axis orientation choice | implemented — port of the endgame-tablebase project's *search* evaluator |

### Learned evaluators (N-series)

Everything above is hand-crafted: our own features, or a transcription of
someone else's tuned constants. The N-series is different in kind — the
weights are *learned from self-play with no human 2048 knowledge encoded*.

| Name | Class | What it is | Status |
|---|---|---|---|
| N1 | `N1Evaluator` (`src/evaluation/n1_evaluator.hpp`) | n-tuple network trained by afterstate TD learning; 4 tuples x 8 symmetries, 33.7M float32 weights (~128 MB) | implemented |

N1 is selected with `--heuristic N1 --weights <path>`; train a weight file
with the `train_ntuple` binary. Because the weights are learned rather than
declared, `--weight name=value` is rejected for N1, and provenance is recorded
as the weight-file path plus a content fingerprint instead of a parameter list
(33.7M weights cannot fit in `ResultMetadata::evaluator_parameters`).

See [`ntuple-learning.md`](ntuple-learning.md) for the algorithm, the network
shape, and why this approach fits our hardware where the endgame tablebase
did not.

**H4 is not "H3 + one feature."** Unlike H1→H2→H3, which each add a feature
to the previous, H4 is a faithful port of the reference
[nneonneo/2048-ai](https://github.com/nneonneo/2048-ai) heuristic (MIT
licensed; described in [`../reference.md`](../reference.md)), with its own
independent feature set and its published CMA-ES-optimized weights
(empty 270, merges 700, monotonicity 47, sum 11). It takes the H4 slot
simply as the next number. Worth knowing when reading comparisons: H4's
weights are already tuned, whereas H2's and H3's are hand-picked, so an
H4-vs-H3 result partly measures "tuned vs. untuned," not purely feature
quality. The reference implementation reports reaching 32768 in ~36% of
games at depth 8.

A heuristic name identifies **feature definitions only**, never weights.
Weight presets are named separately (e.g. "H1, default weights" vs. "H1,
`kDepth3OptimizedBaselineWeights` preset") so optimizing weights never
changes what a heuristic name means.

See [`../experiments/summaries/heuristic-comparison.md`](../experiments/summaries/heuristic-comparison.md)
for measured results. H2's hand-picked `corner_chain` weight (60.0) is still
untuned, and it scores slightly *below* H1 at fixed depth 4 (-1,231 mean, 6/4
paired). **Under the 250 ms timed benchmark it beats H1 by 32% and is the
first heuristic here to reach the 8192 tile** — while searching shallower and
staying inside budget. That reversal is the clearest demonstration so far of
why the two benchmark regimes are kept separate: a fixed-depth ranking would
have written H2 off.

H0 is deliberately the simplest heuristic in the codebase, one rung below
H1: it's modeled on the *original* nneonneo heuristic described in
[`../reference.md`](../reference.md) — "bonuses for open squares and for
having large values on the edge" — before monotonicity/smoothness were
added. It exists to give every benchmark matrix a genuine floor.

## Search

There is currently one configurable search engine, `Expectimax`
(`src/search/expectimax.hpp`), not a family of named algorithms yet. Its
capabilities, and which are exposed as ablation switches in `run_experiment`:

| Capability | Exposed? | Default | Notes |
|---|---|---|---|
| Transposition table | yes — `--transposition-table on\|off` | on | previously always-on with no way to isolate its contribution |
| Iterative deepening / time budget | yes — `--search timed --time-limit-ms N` | off (`--search fixed`) | |
| Adaptive depth ceiling | yes — `--adaptive-schedule hi:H,mid:M,lo:L` | `10:4,6:6,0:8` | only meaningful with `--search timed` |
| Probability cutoff (approximate expectimax) | yes — `--probability-cutoff X` | 0 (exact) | |
| Symmetry (dihedral) canonicalization | yes — `--symmetry on\|off` | off | refused unless the heuristic reports `is_rotation_invariant() == true` (see below) — `StructuralHeuristic` is corner-anchored and unsafe with this |
| Move ordering | no | — | not implemented anywhere; reserved as a future S-series item |
| Search-tree reuse across moves | no | — | not implemented; reserved |

**Depth semantics are unchanged from the legacy engine and are exact**: depth
counts player-decision layers only. This is enforced in
`Expectimax::chance_value` (`src/search/expectimax.cpp`) and documented on
`ExpectimaxOptions` (`src/search/expectimax.hpp`) — "depth x" means the same
thing for every heuristic and every experiment in this framework.

**What a leaf is depends on the evaluator, and that is deliberate.** A 2048
transition is `state --(move)--> afterstate --(spawn)--> next state`, and the
two are different distributions, so a value function trained on one is not
meaningful on the other. `Evaluator::semantics()` declares which kind it
wants:

| `EvaluationSemantics` | Depth 1 means | Used by |
|---|---|---|
| `post_spawn_state` (default) | move → all random spawns → evaluator | H0-H5 |
| `afterstate` | move → evaluator, no spawn | N-series |

Both consume exactly one player decision layer per depth unit, so depth
remains comparable across evaluators — only the leaf changes. This is a
capability query rather than a name check, so a new learned evaluator gets the
right treatment without touching the search.

For an afterstate evaluator, `--search fixed --depth 1` is therefore *exactly*
1-ply greedy `max(reward + V(afterstate))`, which
`tests/evaluation_semantics_tests.cpp` pins as an identity. Getting this wrong
cost N1 a factor of 7 in playing strength before it was found; see
`ULTIMATE_AGENT_PROGRESS.md`.

Future distinct search algorithms (move ordering, tree reuse, further
pruning) get their own S-names (S1, S2, ...) once implemented, each
independently toggleable the same way the table above already treats TT,
iterative deepening, and cutoff as separable switches rather than bundling
them into one opaque mode.

## Row-table acceleration (reusable for future heuristics)

The board is four packed 16-bit rows (4 cells × 4 exponent bits). A 16-bit
value has only 65,536 states, so **any feature computable from one row in
isolation** can be precomputed for every possible row once at startup, then
read back with a bit-shift plus an array index. Columns come free by running
the same table over `transpose(board)`. Cost per evaluation drops from
"decode 16 cells and loop" to 8 table lookups.

Used by H0, H1, and H4. Measured effect when H0 was converted from its
original `decode()`-and-loop implementation: **25.1M → 35.6M nodes/sec
(+42%)** with bit-identical scores.

**A feature qualifies if** it is a sum (or min/max) over rows and columns
independently — empty counts, per-row merges, per-row monotonicity, tile
sums, smoothness within a line.

**A feature does not qualify if** it depends on relationships *across* row
boundaries, or on a board-global value. Two cases seen here:
- H3's snake/main-line score follows a zigzag path where the end of one row
  connects to the start of the next, so a single row's entry cannot know
  its neighbours. This is the main reason H3 runs at 4.6M nodes/sec versus
  H4's 36M. Accelerating it is possible but needs a different scheme
  (per-row interior score plus separately-stitched boundary terms) — see
  `ROADMAP.md`.
- "Is the board's maximum tile on an edge" (H0) and "corner preference"
  (H1) need a board-global maximum, so the table supplies each row's own
  maximum and a short direct check finishes the job.

**Pattern to copy** (see `src/evaluation/h0_heuristic.cpp` or
`h4_heuristic.cpp` for working examples):
1. A `RowFeatures` struct holding the per-row values.
2. A table class whose constructor fills `std::array<RowFeatures, 1<<16>`;
   expose it through a function-local `static` so it builds once, lazily.
3. An ordinary path that loops 4 rows over `packed_exponents` and 4 over
   `transpose(board).packed_exponents`.
4. **An exact fallback for `board.exponent_high_bits != 0`** — tiles above
   exponent 15 don't fit the 4-bit table. Every heuristic here does this;
   skipping it silently corrupts high-tile boards.
5. A differential fuzz test comparing the fast path against a plain
   `decode()` reimplementation over thousands of random boards, as in
   `tests/h0_heuristic_tests.cpp` and `tests/h4_heuristic_tests.cpp`.

Keep the *weights* out of the table. Store unweighted feature values and
apply weights at evaluation time, so weights stay runtime-configurable for
the optimizer work in `ROADMAP.md`. (The reference nneonneo implementation
bakes weights into one fused table for speed; that is why H4 keeps four
separate feature values instead of copying that exactly.)

## Symmetry safety

`Evaluator::is_rotation_invariant()` (`src/evaluation/evaluator.hpp`)
defaults to `false`. `H0Heuristic` and `BaselineHeuristic` (H0/H1) override
it to `true` because their features treat every corner identically.
`StructuralHeuristic` keeps the safe default because it's anchored to one
preferred corner — enabling `--symmetry on` with it would silently corrupt
transposition-table entries across dihedral rotations that don't preserve
which physical corner holds the anchor. `run_experiment` checks this flag
and refuses `--symmetry on` for any heuristic that doesn't report itself
safe.

## Depth sizing guidance (answers the open question from the spec)

Exact-depth cost grows **~13x per move and ~20x per full game for each +1
depth** in the 4-5 range on this hardware — measured, not estimated, in
[`../legacy/results/experiments/milestone4-baseline-depth-sweep/`](../legacy/results/experiments/milestone4-baseline-depth-sweep/):
depth 4 is 11.3 ms/move (22.8s/game); depth 5 is 149.8 ms/move (462s/game).
Extrapolating that same ~13x-per-move factor (a lower bound, since deeper
search also tends to produce longer games): depth 6 ≈ 2s/move, i.e. roughly
1-2 **hours** for a single game, and depth 8 is where the legacy README's own
note applies — "a zero time limit restores full-depth completion and can
take days at depth 8." Recommendation, revised after actually running this:

- Depth 4, fixed: affordable at moderate scale (~20s/game) — a 10-20 game
  sample is a reasonable first pilot; the full `quick_benchmark` (100 seeds)
  is doable but takes on the order of 30-40 minutes per heuristic and should
  be run in the background, not interactively.
- Depth 6 and 8, fixed (no time limit): **not practical to run at all** with
  the current engine — not "expensive but doable for 3 games" as originally
  guessed, but hours-to-days for even a single game. Don't schedule fixed
  depth ≥6 runs until real optimization work (`ROADMAP.md`) changes this
  math. This is itself a useful finding: it's the concrete reason the timed/
  adaptive regime exists at all, and a strong argument for prioritizing
  runtime optimization before deep fixed-depth benchmarking becomes useful.
- Timed/adaptive runs: cost is bounded *per move*, not per game — but
  updated after actually running one: a 250ms-budget H0 game was killed
  after 25+ minutes without finishing even the first game of a 3-game pilot.
  Iterative deepening almost always benefits from going one level deeper, so
  it tends to spend close to the full budget on nearly every move, and games
  at this search quality can run into the thousands to tens of thousands of
  moves (matching the reference nneonneo AI's own reported ~28,000-move,
  96-minute games at comparable depth). At 250ms/move that's potentially
  **hours per game**, not "many minutes" as originally guessed here. Two
  practical adjustments:
  - For a quick smoke test that a timed configuration works at all, use a
    much smaller budget (25-50ms) and/or a hard move-count expectation
    check, not 250ms.
  - For a real 250ms-budget comparison, treat it like the legacy `v1`-`v2.1`
    pilots actually did: launch it wrapped for a long unattended run (the
    legacy README uses `caffeinate`), expect single-digit hours for a 3-game
    pilot, and don't block interactive work on it.
- `standard_benchmark` (500 seeds) and `final_benchmark` (2000+ seeds) are
  reserved for configurations already clocked as affordable at that scale —
  currently that's depth ≤4 fixed only. No timed/adaptive configuration has
  been clocked as affordable at any scale yet.

## Old → new mapping

| Legacy name | New equivalent |
|---|---|
| `baseline` / `v1` | H1, default weights |
| `baseline-optimized` / `v1.1` | H1, `kDepth3OptimizedBaselineWeights` preset |
| `structural-mainline` / `structural-movement` / `structural-full` / `v2` | structural features (H3 slot), not reintroduced under this framework yet |
| `v2.1` | jointly-optimized structural weights (`kDepth4OptimizedStructuralWeights`), same status as above |

These stay fully documented in
[`../legacy/docs/heuristics.md`](../legacy/docs/heuristics.md) and their
underlying results in [`../legacy/results/`](../legacy/results/README.md) —
nothing about them changes; new work simply doesn't extend that naming
scheme further.
