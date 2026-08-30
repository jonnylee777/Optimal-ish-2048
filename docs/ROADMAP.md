# Roadmap

Current state, what is settled, what is open. **Confirmed** items are backed by
measurement in this repository; **speculative** items are not.

## Current state

**The 32768 ceiling has its own consolidated write-up:**
[`docs/32768-investigation.md`](32768-investigation.md) — every hypothesis raised
against it, how each was tested, and which four were refuted.

**Best agent: 345,380** — `n12_plain_2M.bin` at depth 4, n=200. Reaches 16384 in
93% of games, 32768 in 4.5%. The previously published 356,178 was n=60 and is
3.3% optimistic; the same weights score 344,399 over 200 games.

The project spent months believing the mechanism of its plateau was understood:
*"at depth 4, search already compensates for the value function's weaknesses, so
improvements appear at depth 1 and vanish at depth 4."* **That is not
established.** Every one of those experiments was n=60, where this benchmark
cannot resolve anything below ~9%, and the effects being tested were 2-7%. They
were not measured equal — they were not measured.

## The binding constraint (confirmed)

Score is almost entirely a function of the highest tile reached:

| Highest tile | Share of games | Mean score of those games |
|---:|---:|---:|
| 8,192 | 3% | 162,320 |
| 16,384 | 93% | 355,226 |
| 32,768 | 3% | 576,688 |

**The diagnosis of *why* has been corrected.** This section used to say the
deciding skill is "holding a 16384 and rebuilding beneath it — roughly a 100-move
task, beyond any search horizon, so it must come from the value function," on the
strength of an autopsy that classified 0/40 games as dying one merge short.

A 160-game autopsy at depth 4 says otherwise. Of games reaching 16384, the
largest second tile ever held beside it:

| Largest second tile | Share |
|---|---:|
| 4,096 | 13.8% |
| **8,192** | **83.6%** |
| 16,384 (converted) | 2.6% |

**The agent completes the rebuild in 83.6% of games and then fails to convert.**
Building an 8192 with one cell locked (15 free) succeeds 95.5%; with two cells
locked (14 free) it succeeds ~3% — identical ladder, one fewer cell, ~32x worse.
That is precision on a nearly-full board, which is what **search depth** buys,
not a planning horizon the value function has to carry.

Consequence: deeper search was deprioritised, and the endgame tablebase dropped
a second time, on a premise that does not hold.

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

- **The 32768 wall** (confirmed, but re-diagnosed). 83.6% of games reach
  16384 + 8192 and fail to convert. It is an endgame-precision problem, not a
  planning-horizon problem.
- **Search masks evaluator improvements** — ~~confirmed, three independent
  measurements~~ **NOT ESTABLISHED**. All three measurements were n=60 against
  effects of 2-7%, and this benchmark resolves ~9% at n=60. One properly powered
  data point exists so far: `n12_plain_2M` is +5.2% at depth 1 (n=10,000) and
  +0.3% at depth 4 (n=200, p=0.89) — consistent with masking, but n=200 cannot
  resolve below 5.5% so it does not establish it either.
- **Neural net stability** (open). Learning rates at or above 1e-5 diverge to
  NaN. 1e-6 is stable but slow; whether it plateaus below the table is unknown.
- **Memory ceiling** (confirmed). 8 GB total. Published stronger configurations
  need ~15 GB. A 512 MB model measured *worse* than 320 MB at every budget.
- **Timed-regime benchmarks are missing for every learned agent** (confirmed
  gap). No N-series agent has ever been measured at 250 ms/move, because
  contention corrupts timed runs and the machine has been busy. This is the one
  deliverable from the original brief never completed.

## Rejected — do not repeat without new information

Each was measured, not assumed. Details in `../experiment_results.md`.

| Idea | Result |
|---|---|
| ~~More training (1M -> 2M games)~~ | **REINSTATED: +5.2% at depth 1, n=10,000.** The "tie" was an n=60 depth-4 run |
| Larger network (512 MB) | **-34.0%** at n=10,000 (confirmed, was -21% at n=60) |
| Multi-stage networks | 19% worse |
| Whole-board (global) feature | **-9.2%** at depth 1, n=10,000. The recorded "+3.8%" compared an n=60 run against an n=200 run; the sign reverses |
| Tile downgrading (scale-relative indexing) | 2.7x worse |
| Endgame lookup tables | 1% coverage |
| Endgame-seeded training | **-1.8% vs matched control** (p=0.001, n=10,000) — a real regression, not a tie. Never tested on its own stated metric (32768 rate at depth 4) |
| Split table at 16384 with promotion | worse |
| Structural (snake-order) features | +0.5% vs matched control — a measured tie, not "worse" |
| Distillation from depth-4 search | -1.7% vs matched control |
| TD(lambda = 0.5) | much worse |
| Training under search | no gain at 34x cost — **but run at depth 2, which itself reaches 32768 in 0% of games**, so it generated no new-regime data and could not have tested its own hypothesis |
| Symmetry reduction in search | 60% slower |
| Optimistic initialisation *combined with* temporal coherence | 15% worse |

## Proposed next steps

Reordered by the corrected diagnosis: the failure is endgame precision on a
nearly-full board, and depth is what buys that.

1. **Deeper search on tight boards — RUNNING.** The `AdaptiveDepthSchedule`
   (depth 4/6/8 by empty-cell count) has existed since Phase 1 and had **never
   been run with a learned evaluator**, because the CLI only accepted it
   alongside `--search timed`, which forces a wall-clock deadline and makes a
   run neither reproducible nor parallelisable. Fixed-depth + schedule is now
   accepted. Measured cost: **48.6 ms/move against a 250 ms budget** — and 88%
   of moves fall in the <=5-empty bucket, so this is nearly a uniform depth-8
   agent. Primary endpoint is the 32768 rate at n=200.
2. **Then sweep the probability cutoff.** It has been fixed at 0.0015 since it
   was picked ad hoc; it appears nowhere in the source and has never been swept.
   It is the binding constraint on tree size, not depth: depth 5 costs only 3.5x
   depth 4 at a matched cutoff.
3. **Much longer training.** Now the one intervention with a properly powered
   positive result (+5.2% per doubling at depth 1). Training is Hogwild-parallel
   as of this session, so a doubling costs hours rather than a day.

**Speculative, unchanged in rank:**

4. **Hybrid table + network**, and **redundant encoding** (never attempted).

**Deliberately not planned:** anything needing more than ~3 GB of RAM.

**No longer deliberately excluded:** endgame judgment. It was excluded because
"search already covers it"; the autopsy says it does not.

## Open technical questions

- **Does search mask evaluator improvements at all?** Open. The three
  measurements that "confirmed" it were all underpowered. One properly powered
  point (+5.2% at depth 1 -> +0.3% at depth 4) is consistent with masking but
  does not establish it.
- **Is ~345,000 near the true ceiling for 320 MB?** Published work reaches
  324,710 at 1-ply (we are at 240,366) with a comparable search multiplier, so
  better value functions demonstrably exist — but the configurations achieving
  them need ~15 GB.
- **Why does one fewer free cell cost 32x?** Building an 8192 with 15 free cells
  succeeds 95.5%; with 14 it succeeds ~3%. The ladder to build is identical.
  Nothing yet explains a factor that large, and understanding it is probably
  worth more than any single intervention.
- **Why did the neural network decline between checkpoints?** Not yet
  diagnosed.
- **Why did distillation from depth-4 search make the network worse?** It was
  fit to strictly better targets than its own bootstrap estimates, so this is
  counterintuitive and unexplained.
- **Does depth beyond 4 ever pay?** The only evidence was depth 5 at n=30 with a
  *different* cutoff — depth and pruning varied together, the exact confound that
  invalidated E3. Being measured now at matched cutoff.

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
