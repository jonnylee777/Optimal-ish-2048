# Roadmap

What the standardized experiment framework (`run_experiment`,
`docs/experiment-taxonomy.md`, `experiments/`) doesn't cover yet — sequenced,
not dropped. Confirmed absent by direct codebase investigation before this
list was written, not assumed missing.

1. **Per-move instrumentation.** Search statistics are currently cumulative
   per-experiment only (`SearchAgent::cumulative_statistics()`); there's no
   per-move nodes/timing record. Needed for p95/max search time per move and
   nodes-searched-per-move, and to make `deadline_hit_rate` exact for the
   adaptive empty-cell schedule (right now it's only computed for
   non-adaptive runs — see the comment on `ResultMetadata::deadline_hit_rate`
   in `src/experiments/result_writer.hpp`). Context, revised: an H1 timed run
   averaged 443 ms/move against a nominal 250 ms budget, but H0 and H2 timed
   runs both land at ~245 ms, so this is **not** the systematic engine
   problem it first looked like — the H1 run predates the others and ran
   under heavier machine load. Per-move timing would still settle it
   definitively (which moves overran, and by how much) and is needed for
   p95/max anyway, but it is no longer suspected of masking a correctness
   bug.
2. **Unique-states-evaluated counter.** No visited-state tracking exists
   anywhere; would need a separate set alongside the transposition table,
   with a real design tradeoff between accuracy and overhead.
3. **Transposition-table eviction counter and memory accounting.** The table
   already tracks lookups/hits/hit-rate; eviction counts and live-memory
   size are not tracked.
4. **A generic, heuristic-agnostic weight optimizer.** `src/optimization/baseline_optimizer.*`
   and `structural_optimizer.*` each hardcode their own weight-struct field
   names in ~150 lines of near-duplicate screen/retain logic. Needed before
   H2+ weight optimization can start cleanly — building a third hardcoded
   optimizer for H2 would just be more of the same duplication.
5. **Config-file-driven experiments** (`experiments/configs/*.json`).
   `run_experiment`'s flags plus the JSON output's recorded metadata already
   give one-command reproducibility; a config-file front end would mostly
   help for sweeping many configurations at once. Needs a JSON-parsing
   decision first (vendor a header-only library vs. hand-rolled — there is
   currently no JSON parser in this codebase, only a hand-rolled writer).
6. **Parallel game execution for fixed-depth benchmarks only.** No
   threading/multiprocessing exists anywhere in the codebase today. Timed
   benchmarks should stay single-threaded by default (CPU contention would
   corrupt the per-move time budget) — only fixed-depth, no-time-limit runs
   are safe to parallelize across independent seeds.
7. **Move ordering / search-tree reuse.** `Expectimax` always evaluates
   moves in a fixed `kDirections` order with no PV-first ordering and no
   reuse of a previous move's search tree after the actual move + spawn.
   Future S-series search variants.
7b. **Row-table acceleration for H3's main-line score.** H3 runs at 4.6M
   nodes/sec against H4's 36M, and profiling attributes that to its
   evaluator, not the search. Its snake path crosses row boundaries so the
   plain per-row table trick doesn't apply (see "Row-table acceleration" in
   `docs/experiment-taxonomy.md`), but a two-part scheme should: precompute
   each row's interior path contribution, then stitch the 3 inter-row
   boundary terms directly. If it lands, H3 becomes viable under a time
   budget where it currently isn't, which matters because H3 has the best
   fixed-depth score so far. **Caution:** move-table memory is already
   3584 KB against a 4096 KB L2 cache, so a new 65,536-entry table must have
   its cache effect measured, not assumed — see
   `docs/engine-optimization-notes.md`.
8. **H5+ heuristics.** H0-H4 are done (see `docs/experiment-taxonomy.md` and
   `experiments/summaries/heuristic-comparison.md`). The timed benchmark now
   shows the ranking **inverts** between regimes: fixed-depth-4 gives
   H3 > H1 > H2 > H0, while timed-250ms gives H2 > H3 > H1 > H0. H3 has the
   best evaluation but at 2.9M nodes/sec cannot afford it under a budget
   (reaching depth 4.26 vs H2's 4.89); H2 has the best strength-per-node.
   Only H2 and H3 break the 4096 ceiling. Next steps, in order of expected
   value: (a) a hybrid of H4's near-free row-table features with H3's
   corner-anchored main-line term — H2's timed win shows cheap structural
   features pay, and H4 proves row tables are almost free; (b) item 7b, which
   would make H3's evaluation affordable and likely make it win both regimes;
   (c) optimize H2/H3 weights so the comparison against H4's already-tuned
   weights is fair (blocked on item #4).
9. **A random-rollout agent** (from ronzil/2048-ai-cpp): plays random games
   to completion and picks the move with the best average outcome, using no
   heuristic at all. Reported far weaker than expectimax (~70% to 4096) but
   it is a genuinely independent baseline that shares none of our heuristic
   assumptions. Fits behind the existing `Agent` interface as a new agent
   type, not as an `Evaluator`. See `docs/engine-optimization-notes.md`.
10. **Extend `--symmetry on` safety-checking.** `Evaluator::is_rotation_invariant()`
   already generalizes the check `run_experiment` performs today (H0/H1
   report `true`, anything anchored like `StructuralHeuristic` keeps the
   default `false`) — this item is just "keep overriding it correctly as H2+
   heuristics are added," not new mechanism.

## Endgame tablebase (in progress)

Port of [game-difficulty/2048EndgameTablebase](https://github.com/game-difficulty/2048EndgameTablebase).
See `docs/tablebase.md` and `experiments/summaries/tablebase-variants-solved.md`.

- **Stage A — DONE.** Layered generation + backward DP in `src/tablebase/`,
  validated against an independent brute-force solver on every state. 2x4 and
  3x3 variants exactly solved.
- **Stage B — DONE.** Disk-backed rolling 3-layer window (`layer_store.cpp`,
  `solve_formation_to_disk`), validated bit-for-bit against the in-memory
  solver. Enabled 3x4 at target 128: **77.06M states** in ~200 MB RAM and
  1.18 GB disk, a table that could not be completed in memory at all.
  Remaining: 3x4 at target 256+ (~1B states, ~15 GB, multi-hour) if wanted.
- **Stage C.** A real 4x4 formation table (`L3_512` or `t_512`, ~2-2.5 GB).
  First use of non-trivial formation masks and success cells.
- **Stage D.** `TablebaseAgent` behind the existing `Agent` interface (the
  table returns per-successor probabilities, so it picks moves directly and
  cannot be expressed as an `Evaluator`), with expectimax fallback.
- **Stage E — DONE, and it is the surprise of the project.** Their search
  evaluator, ported as H5 (saturating tile weights, DPDF + T formation scores,
  per-evaluation orientation choice), scores **109,213 at fixed depth 4**
  (n=40, 95% CI [95,292, 123,135]) against H3's 49,982 — while costing
  **5.0 ms/move to H3's 111.3**. Both the strongest *and* among the cheapest
  hand-crafted evaluators.

  The tablebase itself was abandoned as unaffordable (250 GB and 1.1 TB for its
  two workhorse tables). **Its evaluator is worth more than the rest of the
  H-series combined.** It is statistically indistinguishable from N1 at 1-ply
  (102,861) — the intervals overlap and the seed sets differ — so the older
  claim that a learned network beat the best hand-crafted agent by 24% does not
  survive: that comparison used H2 (84,716), not H5.

Out of scope on this hardware (8 GB RAM, 31 GB free disk, ARM/no AVX-512):
`LL_4096` (1.1 TB), `free9`/`free10`/`free12` (460 GB - 2 TB), their AD/EX/BC
algorithms and compression stack, AVX-512 kernels.

## N-series (learned evaluators)

- **N1 — DONE (1-ply).** N-tuple network, afterstate TD learning, 1M
  self-play games. **102,861 mean at 1-ply, 95% CI [98,206, 107,517], n=300**,
  beating the reference paper's own 99,916 for this network shape and beating
  every hand-crafted evaluator here (best: H2 at 84,716 *with* a 250 ms search
  budget). Earlier figures of 105,472/108,946 were smaller samples of this same
  policy. See `docs/ntuple-learning.md`.
- **N1 + search — UNBLOCKED.** The afterstate/post-spawn leaf mismatch is
  fixed via `EvaluationSemantics` on the `Evaluator` interface (a capability
  query, not the `prefers_afterstate()` predicate first sketched here) plus a
  short-circuit in `chance_value`. Depth 1 went 14,262 → 102,861 on unchanged
  weights, and depth 1 is now provably identical to 1-ply greedy.
  **But depth still does not pay** for this network: 96,485 at depth 2,
  65,477 at depth 3. At a 100k-game training budget depth *does* pay (+33-38%),
  so this is a training-budget interaction, not a search bug. Open experiment —
  see `docs/ULTIMATE_AGENT_PROGRESS.md`.
- **TC learning — IMPLEMENTED, evaluation pending.** Per-weight adaptive step
  sizes (`--temporal-coherence`). Measured cost at the current network size is
  **390 MB resident**, not the ~960 MB estimated here for the larger `42-33`
  shape. Mechanically verified: beta decays 1.00 → 0.76 → 0.66 → 0.57 with
  training, and 8 unit tests pin its two limiting cases.
- **Optimistic initialisation — IMPLEMENTED, evaluation pending**
  (`--optimistic-init X`). Exploration without an epsilon parameter, which
  matters because the papers found explicit exploration harmful in 2048.
- **Optimistic initialisation — KEEP.** `--optimistic-init 20000`.
  **+27.7%** at 100k games (n=300, p~0, wins 218/300 seeds), zero runtime cost.
  Broad optimum: 20,000 and 50,000 are within 1% of each other.
- **Backward episode replay — KEEP.** `--backward-updates`. **+13.1%** at 100k.
  Built to fix the late-game calibration error and measurably did NOT; the gain
  is real, the mechanism is unexplained.
- **Multi-stage — IMPLEMENTED, unmeasured.** `--stages N`. **This entry
  previously said 15 GB and out of reach; that was the full 2^4-stage x 3-network
  configuration.** A reduced 4-stage network measures **514 MB** and fits
  easily. Strength effect not yet tested.
- **Larger network — AVAILABLE, untried.** `--tuples large` (5x 6-tuples,
  320 MB). Weight files are now self-describing, so any shape trains and plays
  without a recompile.
- **Still out of reach:** the full 2^4-stage x 3-network configuration (~15 GB)
  and the carousel shaping that depends on it.

**Largest known defect:** the value function overvalues the last fifth of a
game by roughly **4x** (predicts ~43,000 remaining, earns ~11,000), measured
across every network regardless of training budget. See E8 in
`docs/ULTIMATE_AGENT_PROGRESS.md`.

> **Read every score in this file with its sample size.** The H-series
> numbers below are n=3 (timed) or n=10 (fixed depth); at this game's variance
> those cannot separate configurations differing by less than ~2x. See E4 in
> `docs/ULTIMATE_AGENT_PROGRESS.md`.
