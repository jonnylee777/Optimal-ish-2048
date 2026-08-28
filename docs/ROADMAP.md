# Roadmap

Current state, what is settled, what is open. **Confirmed** items are backed by
measurement in this repository; **speculative** items are not.

## Current state

**Best agent: 356,178** — `n5_large_1M.bin` at depth 4. Reaches 16384 in 97% of
games, 32768 in 3%.

The project is at a plateau. **Ten separate attempts have failed to beat that
number**, and the mechanism is understood: at depth 4, search already
compensates for the value function's weaknesses, so improvements appear at
depth 1 and vanish at depth 4.

## The binding constraint (confirmed)

Score is almost entirely a function of the highest tile reached:

| Highest tile | Share of games | Mean score of those games |
|---:|---:|---:|
| 8,192 | 3% | 162,320 |
| 16,384 | 93% | 355,226 |
| 32,768 | 3% | 576,688 |

An autopsy of 40 games found **38 died having never assembled a second 16384**.
None died jammed; none died one merge short. The deciding skill is holding a
16384 and rebuilding beneath it — roughly a 100-move task, beyond any search
horizon, so it must come from the value function.

## Completed

- **Phase 1 — hand-written heuristics.** H0-H5. Best is H5 at 109,213 (depth 4,
  n=40), which is also among the cheapest at 5.0 ms/move.
- **Phase 2 — exact endgame tablebase.** Solver built and validated against an
  independent brute force; board-size variants solved exactly. Abandoned twice
  (storage, then 1% coverage). Its evaluator survives as H5.
- **Phase 3 — TD n-tuple learning.** The winning approach. 102,861 -> 356,178
  via temporal coherence, a larger tuple shape, more training, and two bug fixes.
- **Infrastructure.** Benchmark harness with full provenance; paired
  significance testing (`tools/compare_runs.py`); resumable training
  (`--tc-state`); parallel search; self-describing weight files.

## Current work

**Neural value network** (`src/learning/neural_network.cpp`) — *in progress*.
The last idea not bound by this machine's 8 GB. A lookup table over fixed cell
groups cannot express "the big tiles are in descending order along an edge";
a network can. ~66k weights against the table's 83.9M, so it trades
memorization for generalization.

Status at time of writing: learns (3,450 untrained -> ~42,000), but was
**declining between checkpoints** in the runs then in flight, which is unresolved
— see open questions.

## Known problems

- **The 32768 wall** (confirmed). The agent has maxed the 16384 regime. Nothing
  tried has moved 32768 above 3%.
- **Search masks evaluator improvements** (confirmed, three independent
  measurements). Makes most value-function work unmeasurable at the depth we
  actually play.
- **Neural net stability** (open). Learning rates at or above 1e-5 diverge to
  NaN. 1e-6 is stable but slow; whether it plateaus below the table is unknown.
- **Memory ceiling** (confirmed). 8 GB total. Published stronger configurations
  need ~15 GB. A 512 MB model measured *worse* than 320 MB at every budget.
- **Timed-regime benchmarks are missing for every learned agent** (confirmed
  gap). No N-series agent has ever been measured at 250 ms/move, because
  contention corrupts timed runs and the machine has been busy. This is the one
  deliverable from the original brief never completed.

## Rejected — do not repeat without new information

Each was measured, not assumed. Details in `EXPERIMENTS.md`.

| Idea | Result |
|---|---|
| More training (1M -> 2M games) | tie at depth 4 |
| Larger network (512 MB) | 21% worse |
| Multi-stage networks | 19% worse |
| Whole-board feature | tie at depth 4 |
| Tile downgrading (scale-relative indexing) | 2.7x worse |
| Endgame lookup tables | 1% coverage |
| Endgame-seeded training | tie |
| Split table at 16384 with promotion | worse |
| Structural (snake-order) features | worse |
| Distillation from depth-4 search | worse |
| TD(lambda = 0.5) | much worse |
| Training under search | no gain at 34x cost |
| Symmetry reduction in search | 60% slower |
| Optimistic initialisation *combined with* temporal coherence | 15% worse |

## Proposed next steps

**Confirmed plans:**

1. **Finish and judge the neural network.** It is built, tested, and running.
   Resolve whether the decline between checkpoints is instability, an
   overfitting artefact, or a learning-rate problem.
2. **Run the timed-regime benchmark** for the best agent and H5. This closes a
   known gap and needs only a quiet machine.

**Speculative ideas, in rough order of expected value:**

3. **Hybrid table + network.** Sum an n-tuple value and a small network's
   output. The table supplies memorization, the network supplies relational
   structure neither hand-picked features nor the table can express. Untested;
   the closest attempt (512 hand-picked structural buckets) failed, but a
   learned network is far more expressive than fixed buckets.
4. **Redundant encoding.** Appears in the published ladder worth roughly +16%.
   Never attempted here; not fully understood from the papers available.
5. **Much longer training at the best configuration.** Days-scale rather than
   hours. Cheap to start, low expected value given 1M -> 2M gained nothing at
   depth 4.

**Deliberately not planned:** anything that only sharpens endgame *judgment*
(search already covers it), and anything needing more than ~3 GB of RAM.

## Open technical questions

- **Why does search mask evaluator improvements so completely?** Every
  improvement measured so far has been small (~4% at depth 1). It is unknown
  whether a *large* value-function gain would survive to depth 4, because none
  has been produced to test. This matters: if large gains do transfer, the
  plateau is about improvement size rather than a structural ceiling.
- **Is 356,178 near the true ceiling for 320 MB?** Published work reaches
  324,710 at 1-ply (we are at 226,324) with a comparable search multiplier, so
  better value functions demonstrably exist — but the configurations achieving
  them need ~15 GB.
- **Why did the neural network decline between checkpoints?** Not yet
  diagnosed.
- **Why did distillation from depth-4 search make the network worse?** It was
  fit to strictly better targets than its own bootstrap estimates, so this is
  counterintuitive and unexplained.
- **Does depth beyond 4 ever pay?** Depth 5 ties depth 4 at matched pruning.
  Untested whether a substantially better evaluator would change that.

## Infrastructure gaps (carried forward, none blocking)

Verified absent by direct inspection:

1. **Per-move instrumentation.** Search statistics are cumulative per experiment
   only; no per-move nodes/timing record. Needed for p95/max search time and to
   make `deadline_hit_rate` exact under the adaptive schedule.
2. **Unique-states-evaluated counter.** No visited-state tracking exists.
3. **Transposition-table eviction counts and live-memory accounting.**
4. **A generic, heuristic-agnostic weight optimizer.** `baseline_optimizer` and
   `structural_optimizer` each hardcode their own weight-struct field names in
   ~150 lines of near-duplicate logic.
