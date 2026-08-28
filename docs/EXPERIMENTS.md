# Experiments

An index of experiments recoverable from this repository, with methodology and
conclusions. Per-version scores live in `RESULTS.md`; the full chronological
narrative including retractions lives in `experiment-log.md`. This file is the
middle layer: what was tried, how, and what it showed.

Every number here comes from a JSON result file under `experiments/results/`, a
training log, or `experiment-log.md`. **Items that cannot be verified from the
repository are marked as such.**

## Methodology and conventions

**Two benchmark regimes.** *Fixed depth* isolates evaluation quality (every
agent searches equally deep). *Timed* (250 ms/move) isolates usable strength
(a cheap evaluator can search deeper). Both are recorded in every result file.

**Seed sets.** Defined in `src/experiments/seed_sets.hpp`.

| Range | Purpose |
|---|---|
| 20000-... | H-series benchmarks |
| 30000-30299 | N-series matched comparison |
| 0xE0000000+ | training-time evaluation (disjoint from training seeds) |

Never compare across sets.

**Statistics.** Per-game scores span ~3,000 to ~580,000. `tools/compare_runs.py`
runs a paired test on matched seeds and reports the difference, a 95% interval,
and a p-value. It flags n<30 as overstating significance.

**Primary metrics.** Mean score, 95% interval, median, and per-tile achievement
rates. For endgame work, `achievement_rate_32768` is the leading indicator —
mean score only registers a change after the tile ceiling moves.

## Phase 1 — hand-written heuristics

**Method.** Each heuristic scores a board by a hand-written formula; expectimax
picks the move leading to the best-scoring board. No learning. H0-H3 add
features cumulatively; H4 and H5 are ports of other projects' tuned evaluators.

**Results** (fixed depth 4):

| Agent | Score | n | ms/move |
|---|---:|---:|---:|
| H0 | 26,769 | 10 | 9.0 |
| H1 | 36,141 | 10 | 13.2 |
| H2 | 34,910 | 10 | 13.5 |
| H3 | 57,318 | 40 | 71.8 |
| H4 | 45,493 | 10 | 10.2 |
| **H5** | **109,213** | 40 | **5.0** |

**Conclusion.** H5 is both the strongest and among the cheapest — +90.5% over H3
(p=1e-10, paired, n=40). H0-H4 are n=10 with overlapping intervals, so **their
relative ordering is not established**; only H5-vs-H3 was rerun at adequate n.

**Correction on record.** This repository previously claimed the learned network
beat "the best hand-crafted evaluator" by 24%. That comparison used H2 (84,716),
not H5. Against H5 there is no significant gap at 1-ply.

## Phase 2 — exact endgame tablebase

**Method.** Solve endgame positions exactly by layered dynamic programming over
tile sum, then look up the answer instead of estimating it. Ported in concept
from `2048EndgameTablebase`.

**Verification.** Two chained correctness gates: the layered solver was checked
against an independent brute-force solver (agreement to 1e-12 on every state),
and the disk-backed solver was checked bit-for-bit against the in-memory one.
Board-size variants (2x4, 3x3, 3x4) were solved exactly.

**Two bugs found by those gates**, both recorded in `phase2-endgame-tablebase.md`:
a merge-direction error (gathering tiles in forward scan order merged the pair
furthest from the movement direction) and a symmetry group that was not closed,
which broke canonicalization idempotence.

**Abandoned twice.**

1. *Storage.* The two tables carrying 81% of the reference project's moves need
   250 GB and 1.1 TB; ~21 GB was available. The affordable 10-free-cell tables
   (2-2.5 GB) covered endgames the agent could not then reach.
2. *Coverage.* Once the agent could reach them, this was re-examined and
   measured: only **1% of late-game moves** land in a position a formation
   covers, against a 20% threshold set in advance.

**Conclusion.** Not viable here, for two independent reasons. The lasting
benefit is H5, the evaluator that came with the port.

## Phase 3 — TD n-tuple learning

**Method.** A large lookup table over tile patterns, trained by afterstate
TD(0) self-play. Each pattern's 8 dihedral orderings share one table entry, so
every update generalizes across symmetries.

### Changes that helped

| Change | Effect | Evidence |
|---|---:|---|
| Afterstate leaf semantics (bug fix) | 14,262 -> 102,861 | n=300 |
| Temporal coherence at alpha=1.0 | **+71%** | n=300 |
| Backward episode replay | +17.9% | n=300 |
| `large` tuple shape (128 -> 320 MB) | +20.8% | p=6e-09, n=300 |
| Terminal-position fix (bug) | **+48% at depth 4** | same weights |
| Training 100k -> 1M games | +68% | n=200 |
| Training 1M -> 2M games | **+5.2%** | n=10,000 — previously filed as "no gain" from n=60 |

### Changes that did not

| Change | Result | Evidence |
|---|---:|---|
| Reward-index correction | −13.3% | p=8e-06 — *theoretically correct, empirically worse* |
| Optimistic init alone | +27.7% | n=300 |
| Optimistic init **with** temporal coherence | **−14.9%** | substitutes, not complements |
| Multi-stage networks | −19.4% | p=8e-09 |
| `xlarge` shape (512 MB) | **−34.0%** | n=10,000 (was −21% at n=60) |
| Symmetry reduction in search | 60% slower | timed A/B |
| Whole-board (global) feature | **−9.2%** | n=10,000 — recorded as +3.8% from an n=60 vs n=200 comparison; sign reverses |
| Endgame-seeded training | **−1.8%** | p=0.001, n=10,000, vs its matched control |
| Distillation from depth-4 search | −1.7% | n=10,000, vs matched control |
| Structural (snake-order) features | +0.5% | n=10,000 — a measured tie, not "worse" |

Every entry below the rule is measured against `n9_ext_1M1`, the control that
got the same 100k extra games without the feature. Against `n5_large_1M`
instead, three of them look positive; the difference is the extra training, not
the feature.

### Learning-rate finding

`alpha=1.0` collapses plain TD to 15,300 (vs 55,640 at alpha=0.1). Cause:
different dihedral orderings of a tuple can collide on the same table entry,
which on sparse boards amplifies a step by ~2.75x. Temporal coherence damps
exactly those weights, which is what makes alpha=1.0 not merely survivable but
optimal (95,371). The same collision effect makes plain distillation diverge at
alpha=1.0 — pinned as a regression test.

## Two bugs that invalidated prior results

**Afterstate/post-spawn mismatch.** The search evaluated learned networks on
post-spawn boards when they were trained on afterstates. Depth 1 went 14,262 ->
102,861 on unchanged weights. Invalidated runs: `results/invalid-afterstate-mismatch/`.

**Terminal positions valued by the evaluator.** A dead board returned
`evaluate()` rather than 0. The network overestimates endgames ~6x and a dead
board is full of large tiles, so terminal positions scored ~137,000 — search was
rewarded for dying.

| Depth | Buggy | Fixed |
|---:|---:|---:|
| 2 | 184,096 | 306,417 |
| 3 | 219,168 | 334,030 |
| 4 | 42,735 | 356,178 |

Depth 1 was unaffected (afterstate semantics expand no spawns, so no terminal
board is reached), which is why **every training conclusion survived**. 22
invalidated runs: `results/invalid-terminal-bug/`.

**Conclusions retracted because of the second bug:** "search benefit shrinks as
the evaluator improves", the depth-vs-budget curve, and the dismissal of
whole-board features. All were measuring the bug.

## Phase 4 — attempts on the 32768 ceiling

**Diagnosis — SUPERSEDED.** An autopsy of 40 depth-4 games
(`scratchpad/endgame_autopsy.cpp`, not committed) classified each death:

| Cause | Games |
|---|---:|
| Never assembled a second 16384 | 38 |
| Large-tile chain broken | 2 |
| Jammed / one merge short | 0 |

That "0 one merge short" drove the roadmap for months — it is why deeper search
was deprioritised and the tablebase dropped a second time. A 160-game autopsy
(`scratchpad/autopsy.cpp`) measuring *how far* the rebuild gets contradicts it:

| Largest second tile held beside a 16384 | Games | Share |
|---|---:|---:|
| 4096 | 21 | 13.8% |
| **8192** | **127** | **83.6%** |
| 16384 (converted) | 4 | 2.6% |

**83.6% reach one merge short and fail to convert.** Building an 8192 with one
cell locked succeeds 95.5%; with two cells locked, ~3%. At that point the board
must hold a tile sum near 32,768, which is essentially a full clean ladder across
14 of 16 cells — almost no slack. It is a precision-on-a-full-board problem,
which is what search depth buys.

**Attempts** (all at depth 4, n=60, against 356,178):

| Attempt | Score |
|---|---:|
| Distillation from 748k depth-4 search values | 341,911 |
| Structural (snake-order) features | 341,790 |
| Split table at 16384 with weight promotion | 333,535 |
| Endgame-seeded training | 350,925 |
| Whole-board feature | 331,994 |
| 2x training (2M games) | 345,858 |
| Tile downgrading | 49,638 (1-ply eval) |
| TD(lambda=0.5) | 222,296 |

**Conclusion — WITHDRAWN.** "None beat the baseline" was read as evidence that
search compensates for value-function weaknesses. It is not. Every row above is
n=60, where this benchmark resolves ~9%, and the effects were 2-7%. They were
unmeasured, not measured equal. Re-run at n=10,000 (depth 1) against matched
controls, the ordering is different and one intervention — more training — is
clearly positive. Whether depth-1 gains survive to depth 4 remains open: the one
properly powered pair is +5.2% at depth 1 and +0.3% at depth 4 (n=200, p=0.89),
which n=200 cannot resolve either way.

## Neural value network — in progress

**Method.** Replace the lookup table with a small network (16 one-hot cells ->
16 summed embedding rows -> ReLU -> linear), trained by the same afterstate TD
recipe so the model class is the only variable.

**Verified.** Gradient moves values toward their target; speed is 844 ns/eval at
hidden=64 against the table's 658 ns; learning rates >= 1e-5 diverge to NaN.

**Status.** Learns (3,450 untrained -> ~42,000) but was declining between
checkpoints in the runs then in flight. **Final outcome not determined at the
time of writing.**

**A measurement bug worth recording.** Three of four learning-rate probes
printed nothing, which looked like "did not reach the reporting threshold". They
had diverged to NaN — and a diverged network makes every move look illegal, so
every episode came back empty and hit a `continue` that skipped the reporting
line. *The failure mode erased its own evidence.* The probe now checks for
non-finite values and reports divergence explicitly.

## Not determinable from the repository

- Wall-clock cost of most training runs (logs were written to `/tmp` and are not
  committed).
- Exact tuple cell sets used by the published reference networks; the `large`
  and `xlarge` shapes here are **this project's own choices**, documented as such
  in `ntuple_network.cpp`, because the papers describe shape rather than explicit
  cells.
- Per-move timing distributions (p95/max) — only cumulative statistics are
  recorded. See the infrastructure gaps in `ROADMAP.md`.
- H5 in the timed regime, and **any** learned agent in the timed regime — never
  run.
