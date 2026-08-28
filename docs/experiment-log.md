# Ultimate agent — development log

Authoritative record of hypotheses, experiments, and keep/reject decisions on
the way to the strongest 2048 agent feasible on an Apple M1 / 8 GB.

Newest entries at the top. Where this disagrees with older narrative docs,
**this file wins**.

Machine: Apple M1, 8 cores, 8 GB RAM, ~31 GB free disk, ARM (no AVX-512).

---

## Status board

Kept, rejected, and pending — the short version. Details in the E-entries below.

| # | Change | Verdict | Evidence |
|---|---|---|---|
| E1 | Afterstate leaf semantics | ✅ **KEEP** | depth 1: 14,262 → 102,861, n=300 |
| E2 | Reward off-by-one in TD target | ⏸ **HOLD** | −13.3% at matched budget, p=8e-06 |
| E3 | Depth beyond 1 | ✅ **RESOLVED** | gain tracks base strength: +38% → −6% |
| E3b | `alpha=1.0` with **plain** TD | ❌ **REJECT** | 15,300 vs 55,640 — but see E5 |
| E4 | H-series ranking | ⚠️ methodology | all n<=10, CIs overlap |
| E5 | Temporal coherence @ alpha=1.0 | ⭐ **KEEP** | **90,508 at 100k games** (n=300); +71% |
| E6 | Optimistic initialisation | ✅ **KEEP** (X=20,000) | +27.7%, p~0, wins 218/300 seeds |
| E7 | Drop redundant eval in trainer | ✅ **KEEP** | byte-identical weights, less work |
| E8 | Value calibration probe | ⭐ **finding** | late game overvalued 4x, all networks |
| E9 | Backward episode replay | ⚠️ works, redundant | +17.9% alone; adds nothing to E6 (p=0.30) |
| E10 | Selectable tuple configs + self-describing weights | ✅ **KEEP** | infrastructure for 320 MB network |
| E11 | Multi-stage weight sets | ❌ **REJECT** | **−19.4%**, p=8e-09 — my predicted best idea, refuted |
| E12 | H5 benchmark (side task) | ⭐ **finding** | **109,213** at depth 4 (n=40) — matches N1, 20x faster than H3 |
| E13 | Which training gains compound? | ⭐ **finding** | Selective: TC+backward +23.5% *at 100k*; TC+optimistic −14.9% |
| E15 | Does backward replay's gain persist? | ⭐ **finding** | **No** — +23.5% at 100k, **tie at 200k**. Speed, not quality |
| E16 | Larger tuple set (`--tuples large`) | ⭐ **KEEP** | **+20.8%**, p=6e-09 — first 16384 |
| E17 | Why does search hurt? | ⭐ **DIAGNOSIS** | V overvalues late game **5.9x**; search amplifies, not averages |
| E18 | Whole-board feature | ⚠️ **neutral** | +17.3% at depth 3 at 100k; advantage gone by 200k (p=0.002) |
| E19 | Even larger network (`xlarge`, 512 MB) | ❌ **REJECT at 100k** | **−21%** vs 320 MB — capacity needs matching budget |
| E20 | Persist TC accumulators | ✅ **KEEP** | training was all-or-nothing; now incremental |
| E21 | **Search valued dead positions at ~137,000** | 🔴 **BUG FIXED** | depth 2: 184,096 → **306,417**, depth 3 → **334,030** |
| E22 | Audit of conclusions post-bug | ✅ **clean split** | all depth-1 (training) findings hold; all search findings retracted |
| E23 | Post-fix depth sweep + training plateau | ⭐ **356,178 at depth 4** | 1.0M→1.2M games gains nothing measurable |
| E24 | Search speedups | symmetry ❌ / parallel ✅ | symmetry 60% slower; parallel 1.68x, identical play |
| E25 | Global features + depth 4 | ❌ **substitutes** | −6.8% (p=0.08); global wins at depth 1, ties at depth 4 |
| E26 | Endgame autopsy | ⭐ **DIAGNOSIS** | 38/40 deaths = never built a 2nd 16384; tablebase killed (1% coverage) |
| E27 | Tile downgrading (relative indexing) | ❌ **REJECT** | **49,638 vs 135,043** — destroys scale knowledge |
| E28 | Endgame-seeded training | ⏸ running | 14,351 collected positions; control = 242,440 |
| E14 | Depth-vs-budget along one axis | ✅ | budget +30% per doubling; depth gain 16.6% → 11.7% |

### Strongest agent — FINAL

⭐ **226,325** — the **`large`** network (5x 6-tuple, 320 MB) trained with
temporal coherence + backward replay for **1,000,000 games**, played at
**depth 1** (= 1-ply greedy). n=200, 95% CI [213,747, 238,903].
Weights: `experiments/weights/n5_large_1M.bin` (hash `b93712e5644f2255`).

**2.2x the agent this effort started from** (102,861).

| Metric | Value |
|---|---|
| Mean | **226,325** [213,747, 238,903] |
| Median | 235,566 |
| Best single game | 381,436 |
| Reaches 4096 | 96% |
| Reaches 8192 | 90% |
| Reaches **16384** | **54%** |
| Reaches 32768 | 0% |

**Search is now actively harmful for this agent.** Depth 2 scores 184,096 —
**−18.7%, p=7.7e-06**, winning only 70 of 200 seeds. The strongest
configuration does no lookahead at all beyond evaluating each legal move's
afterstate.

Training never plateaued: the held-out learning curve ran 173,285 (250k) →
201,683 (500k) → 227,932 (750k) → 235,786 (1M). **1M games was not the
ceiling.**

### What actually worked, and what I got wrong

Every gain in this log came from **training method**, not from architecture:

| Source of improvement | Result |
|---|---|
| Fixing a semantics bug | 14,262 → 102,861 |
| Changing the step-size rule | +71% |
| Changing update order | +23.5% at 100k, 0% at 200k |
| Training longer | +30% per doubling |
| Every architectural idea tested | rejected |

Against that, my own predictions have a poor record. The reward-index fix was
theoretically correct and lost 13%. Backward replay was built on a mechanism
that measurement refuted, and works anyway. Calibration looked like the thing
to fix and turned out nearly irrelevant to an argmax policy. Combinations of
independently-validated wins were assumed to compound and mostly did not.
Multi-stage was written into this document as the best remaining idea and lost
19%.

The one change that produced the largest gain — temporal coherence — was nearly
discarded because the first test used the wrong control.

**Rules adopted after being burned:**
1. No keep/reject decision below **n=200**. n=10 produced two retracted
   conclusions in this log.
2. Never compare across two variables at once. E3's first version compared
   reward index *and* training budget, and got the answer backwards.
3. **No timed-regime benchmark while any training job runs.**
4. One or two training jobs at a time — four concurrent trainers OOM-killed a
   run on a machine already carrying 4.4 GB of swap from other applications.
5. **Test the combination, never assume it.** Optimistic init (+27.7%) and
   backward replay (+17.9%) look orthogonal — an initialisation and an update
   order — and stack to **nothing** (p=0.30).
6. **Ask what a technique is FOR before testing it.** Temporal coherence was
   nearly rejected because the first test damped an already-conservative
   `alpha=0.1`; its whole purpose is to make an aggressive rate survivable. The
   corrected test produced the largest gain in this log.
7. In `zsh`, an unquoted parameter does **not** word-split. A loop passing
   `$flags` to a binary silently sent `"--optimistic-init 20000"` as one
   argument and every run in an ablation errored out.

---

## Deliverables summary

Updated as results land. Anything marked *in progress* has no claim attached
to it yet.

### 1. Final architecture

```
Board (packed nibbles)
  -> Expectimax  (depth = player decision layers; TT; probability cutoff)
       -> Evaluator interface
            |- EvaluationSemantics::post_spawn_state  -> H0..H5 (hand-crafted)
            |- EvaluationSemantics::afterstate        -> N-series (learned)
                 -> NTupleNetwork  (LUTs, dihedral weight sharing, optional stages)
                      trained by afterstate TD(0)
                        + temporal coherence (per-weight adaptive step)
```

The one structural change: `Evaluator` now declares *what kind of board it
scores*. Everything else composes around that.

### 2. Changes made

| Area | Change |
|---|---|
| Search | `EvaluationSemantics` capability; afterstate leaf short-circuit |
| Learning | Temporal coherence; optimistic init; backward replay; reward-index fix |
| Network | Named tuple configs (`default`/`large`); multi-stage weight sets; self-describing weight files |
| Trainer | Versioned checkpoints; redundant-evaluation removal (hot path) |
| Tooling | `tools/compare_runs.py` (paired tests); `tools/depth_vs_budget.sh`; calibration probe |

### 3. Correctness evidence

- **19/19 test suites green** throughout.
- New: 8 evaluation-semantics tests, 9 temporal-coherence tests, 19 n-tuple
  tests (incl. staging), 13 trainer tests.
- Two changes verified **byte-identical** rather than "close": the hot-path
  deduplication (`cmp` clean) and single-stage networks after the staging
  change (fingerprint unchanged).
- Training reproducibility verified across *separate invocations*: the 1M run's
  100k checkpoint is byte-identical to a standalone 100k run.
- `n1_default.bin` re-benchmarked after every format change; still loads to the
  identical score.

### 7. Final benchmark — current standings

| Agent | Score | n | Regime |
|---|---:|---:|---|
| **N5 `large` @1M** | **226,325** | 200 | **fixed depth 1** |
| N5 `large` @300k | 190,789 | 200 | fixed depth 2 |
| N5 `large` @1M | 184,096 | 200 | fixed depth 2 |
| N5 `large` @100k | 165,524 | 300 | fixed depth 2 |
| N2 (TC, 200k games) | 131,481 | 200 | fixed depth 2 |
| N2 (TC, 200k games) | 117,665 | 200 | fixed depth 1 |
| N2 (TC, 100k games) | 105,573 | 300 | fixed depth 2 |
| N1 (1M games) | 102,861 | 300 | fixed depth 1 |
| N2 (TC, 100k games) | 90,508 | 300 | fixed depth 1 |
| **H5** | **109,213** | 40 | fixed depth 4 |
| H2 | 84,716 | 3 | timed 250 ms |
| H3 | 49,982 | 10 | fixed depth 4 |

Timed-regime numbers for N2 and H5 are **queued, not measured** — running them
against active training jobs would understate them (see the scheduling rule).

### 9. Negative results

Recorded because they cost real time and would otherwise be repeated:

- **The theoretically-correct reward index made the agent weaker** (−13.3%,
  p=8e-06). Still on HOLD, not adopted (E2).
- **`alpha=1.0` with plain TD collapses** to 15,300 — LUT index collisions,
  independent of the reward bug (E3b).
- **Backward replay did not fix what it was built to fix.** Late-game bias
  went 1,805 -> 1,939 (E9).
- **Calibration is nearly irrelevant to a 1-ply agent.** The worst-calibrated
  network is the strongest (E8).
- **Training improvements compound only selectively** — optimistic init is
  worth +27.7% alone and costs 15% on top of temporal coherence (E13).
- **Deeper search stops paying** as the evaluator improves; −6% for the
  strongest network (E3).
- **Multi-stage networks lose 19.4%** at this training budget — the idea I
  predicted would be the biggest remaining win (E11).

### 10. Remaining bottleneck

**Late-game value error.** Every network measured overvalues the final fifth of
a game by roughly 4x (predicts ~43,000 remaining, earns ~11,000). It has
survived every change tried: reward index, update order, step-size rule, and a
10x training-budget increase. It is the clearest single defect and the least
understood.

### 11. Best next direction

**Global features plus depth-3 search** (E18). This is the first thing measured
here that lifts the *ceiling* rather than the number: every plain network goes
flat or negative with depth, and the 1M network *loses* 18.7% at depth 2, so
search — worth ~58% to published agents — was simply unavailable. Calibrating V
restores it, at a 6.5% cost to 1-ply play that depth 3 repays threefold.

A 1M run at this configuration is training now. If the 1.68x training scaling
observed for `large` (100k -> 1M) carries over, depth-3 play should land far
above the current 226,325 baseline.

Two further levers, both measured and both still unexhausted:

1. **A larger network.** `--tuples large` gave **+20.8%** at equal training
   (E16) and produced the current best agent. The jump from 128 MB to 320 MB
   paid immediately; **640 MB and 1.3 GB shapes are still affordable on this
   machine and completely untried.** This is the clearest remaining lever, and
   the one whose ceiling is least understood.
2. **Training longer.** +30% per doubling with no plateau at 200k (E14). A 1M
   run at the best configuration is under way.

Those two interact in the direction that favours doing both: a bigger table has
more weights to fill, so it should benefit *more* from extra games, not less.
That is a prediction, not a result.

*Previously listed here, now refuted:* multi-stage networks (E11), −19.4%.

And my summary of *why* it failed was too broad. I wrote "adding capacity works,
dividing capacity does not" — then E19 measured 512 MB as **21% worse** than
320 MB at the same budget. The accurate rule is **capacity and training budget
must scale together**; there is an optimum shape for each budget, and the E16
gain was not a property of size alone.

*Previously listed here, now refuted:* multi-stage networks (E11), which I
predicted would be the best remaining idea and which measured **−19.4%**
(p=8e-09). It fails for a reason worth carrying forward: **partitioning the
weight table also partitions the training data**, and this network's strength
comes from generalisation through weight sharing. On this hardware, ideas that
*add* capacity or data are favoured over ideas that *divide* what exists.

---

## Baseline at start of this effort

Build clean, **17/17 tests passing** (23 s). Strongest agents:

| Agent | Score | Regime |
|---|---:|---|
| N1 (learned, 1M games) | 105,472 | 1-ply greedy, no search |
| H2 | 84,716 | timed 250 ms/move |
| H3 | 74,788 | timed 250 ms/move |
| H3 | 49,982 | fixed depth 4 |

N1 was known-broken under search (4,228 at depth 4).

---

## E1 — Afterstate/state leaf-semantics mismatch  ✅ KEEP

**Hypothesis.** N1 collapses under search because it is an *afterstate* value
function while `Expectimax` evaluates leaves on *post-spawn states*.

**Implementation.** Added `EvaluationSemantics {post_spawn_state, afterstate}`
to the `Evaluator` interface (`src/evaluation/evaluator.hpp`), defaulting to
`post_spawn_state` so every hand-crafted evaluator is untouched. N1 declares
`afterstate`. `Expectimax::chance_value` short-circuits when
`depth == 1 && semantics == afterstate`, evaluating the afterstate directly
instead of forcing one more spawn. Deliberately a capability query, not a
name check, so future learned evaluators inherit it.

Depth semantics preserved: both settings consume exactly one player decision
layer per depth unit.

**Result — same weights, only the search path changed:**

| Depth | Before (n=10) | After (n=10) | After (n=300) |
|---|---:|---:|---:|
| 1 | 14,262 | 96,203 | **102,861** |
| 2 | — | 115,747 | 96,485 |
| 4 | 4,228 | — | — |

**Decision: KEEP.** 7.2x at depth 1. The single highest-value fix available.

⚠️ **Correction — n=10 was not enough.** My first read of this experiment
reported depth 2 (115,747) as the new best. At n=300 that reverses: depth 2 is
**96,485**, *below* depth 1's 102,861. Per-game scores range 5,552-173,328, so
n=10 has a standard error near 15k — far too wide to rank configurations.
**Every keep/reject decision in this log needs n>=200.** The n=10 numbers are
kept above as a record of the error, not as evidence.

**Depth caveat.** Depth 2 does not help *this* network. E3 shows that is
specific to the 1M buggy-reward weights — at 100k games depth 2 gains ~33-38%
regardless of reward index — so search is not dead, it interacts with training
budget.

---

## E2 — Reward off-by-one in the TD target  ⏸ HOLD — correct but losing

**Found while auditing the trainer for E1**, not previously known.

**The bug.** The trainer updated

```
V(s'_{t-1})  <-  r_{t-1} + V(s'_t)
```

using `previous_reward` — the reward of the move that *created* the previous
afterstate. The correct target (and what the reference trainer does, via
`targetValue = afterstateValue + agentTransition.getReward()`) is

```
V(s'_{t-1})  <-  r_t + V(s'_t)
```

the reward of the move *leaving* it.

**Why it matters.** Under the bug, `V(s')` includes a reward already banked
before reaching `s'` — which is not a function of `s'` at all, so the target
is ill-posed. Worse, the action rule `reward + V(afterstate)` then
**double-counts the immediate reward**, biasing toward immediate merges.

**Implementation.** `src/learning/td_trainer.cpp`: removed `previous_reward`
entirely; the pending afterstate is now updated with the *current* candidate's
reward. Terminal update targets 0 (no further move, no further reward).

**Status: HOLD, not KEEP.** The fix is in and all 18 suites pass, but at
matched training budget it makes the agent **significantly weaker**
(−13.3% at depth 1, p=8e-06). Being theoretically right is not the criterion.
E3 has the numbers and the decisive 1M test.

---

## E3 — When does search help?  ✅ RESOLVED — it tracks base strength

**First answer was wrong, and the correction matters.** I originally compared a
1M-game buggy-reward network against a 100k-game fixed-reward one and concluded
"the E2 reward fix restores depth scaling." Those networks differ in *training
budget* as well as reward index — the same class of error as this project's
earlier "depth 2 vs depth 4" mistake. At matched budget both scale with depth,
so that conclusion was an artifact.

**With enough networks measured, the real relationship is clean.** Depth-2 gain
declines monotonically with how strong the 1-ply policy already is, and
eventually goes negative:

| Network (all depth 1 vs depth 2, matched seeds) | Depth 1 | Depth 2 | Gain |
|---|---:|---:|---:|
| 100k plain, fixed reward | 52,589 | 72,719 | **+38%** |
| 100k plain, buggy reward | 60,625 | 80,835 | **+33%** |
| 100k temporal coherence | 90,508 | **105,573** | **+16.6%** |
| 1M plain, buggy reward | 102,861 | 96,485 | **−6%** |

Reward index, training budget, and step-size rule all vary across those rows,
and none of them predicts the gain. **Base strength does**, monotonically.

**Interpretation.** Expectimax with a learned leaf evaluator improves on the
greedy policy only to the extent there is headroom the value function's
*errors* do not eat. A weak V leaves plenty of headroom, so 2-ply lookahead
finds real improvements. A strong V has little headroom left, and the same
lookahead increasingly amplifies V's own mistakes — the max over noisy
successor estimates is biased upward, and that bias is paid at every chance
node. Past some point the amplification exceeds the gain.

Notably this is *not* explained by calibration: the TC network is the worst
calibrated of the four (E8) and still gains 16.6% from depth, while the
best-calibrated network loses 6%.

**Consequences.**
- Depth is a genuine lever, and the current best agent uses it: **105,573 at
  depth 2**, which is the strongest result in this project.
- But depth is not a free multiplier to stack on top of a stronger network.
  Every future evaluator must have its depth benefit re-measured rather than
  assumed — for `n1_default`, depth 2 is a 20x cost increase for a 6% *loss*.
- Roadmap phases 11-13 (adaptive search, root deepening, cross-move cache
  reuse) are worth doing, but their value shrinks as the evaluator improves.
  They should be prioritised *below* further learning work.

---

## E3b — Learning rate: `alpha=1.0` is unusable *with plain TD*  ⚠️ SUPERSEDED BY E5

The papers use `alpha=1.0` with `alpha/m` scaling. This project had rejected
it, attributing the instability to LUT index collisions between symmetric
orderings. That measurement predated E2, so it was worth re-asking whether the
ill-posed target was the real cause.

It was not. With the corrected target, plain TD, 100k games:

| alpha | Greedy score |
|---|---:|
| **0.1** | **55,640** |
| 0.3 | 52,865 |
| 1.0 | 15,300 |

**Conclusion at the time: keep `alpha=0.1`.** Correct for plain TD, and the
collision explanation does stand on its own.

### Why this entry is superseded

I drew the wrong general lesson from it. The right reading is not "`alpha=1.0`
is too big for this network" but "**plain TD has no way to survive
`alpha=1.0`**" — the two are only the same claim if the step size must be
global.

E5 shows temporal coherence at `alpha=1.0` scores **95,371**, versus 55,640
for the best plain configuration. The papers' advice was right; what was
missing was the per-weight mechanism that makes it safe. Collisions are real
and do cause overshoot, but TC damps exactly the weights they affect, because a
colliding weight receives inconsistent errors and its `beta` falls.

**Standing guidance:** `alpha=0.1` for plain TD, `alpha=1.0` with
`--temporal-coherence`. Never `alpha=1.0` alone.

**The trap worth naming:** a correct measurement plus a plausible mechanism
produced a conclusion that was true in its own scope and wrong as a rule. It
then discouraged the very experiment that would have overturned it. That is
harder to catch than a bad measurement, because nothing in the data looks
wrong.

---

## E2b — Reconciling the 105,472 / 108,946 N1 discrepancy  ✅ RESOLVED

The roadmap flagged two different published N1 numbers. **They are the same
policy measured on different seed samples**, not two algorithms.

With the E1 fix, `--search fixed --depth 1` *is* greedy afterstate selection —
`max(reward + V(afterstate))`, proven as an exact identity by semantics Test 3.
Measured on 300 fresh seeds: **102,861**, per-game range 5,552-173,328.

At that spread the 95% CI is roughly ±5k, which covers 105,472 and nearly
covers 108,946. All three are consistent draws from one policy. Nothing was
broken; the earlier numbers just came from small samples.

**Action:** report N1 greedy as **~103k ± 5k (n=300)**, and stop quoting
point estimates from small runs.

---

## Test infrastructure added

`tests/evaluation_semantics_tests.cpp` (8 tests, all passing) pins the
semantics so neither bug can silently return:

- afterstate depth-1 sees **only** legal afterstates, never a spawned board
- post-spawn depth-1 **does** see spawned boards (the contrast case)
- afterstate depth-1 value == `max(reward + V(afterstate))` exactly
- chance nodes use the 90/10 spawn split
- depth counts player layers for **both** semantics
- illegal/no-op moves never selected; dead boards return no direction
- transposition table changes no results (cached == uncached, both semantics)
- H0-H5 regression: still `post_spawn_state`, still searching correctly

Suite total: **18/18 passing.**

---

## E5 — Temporal coherence  ⭐ KEEP — the largest single gain so far

**Hypothesis.** One global `alpha` for 33.7M weights is a poor fit. TC (Beal &
Smith 1999; Jaskowski 2016) scales each weight's step by `beta = |E| / A` — the
*consistency* of the errors that weight has seen. Consistent errors keep beta
near 1 (still converging, keep moving); oscillating errors drive beta toward 0
(damp it).

**Implementation.** `src/learning/temporal_coherence.hpp/.cpp`, exposed as
`--temporal-coherence`. Two extra float arrays per weight; **390 MB resident**
vs 131 MB for plain TD. Training-only state, deliberately not serialised, so a
TC-trained network still loads for play at 128 MB. Eight unit tests pin the
mechanism, including both limits by hand: a first-ever update must reduce
exactly to plain TD (beta == 1), and 200 alternating-sign errors must drive
beta below 0.02.

### My first test of it was mis-designed, and the correction is the result

I ran TC at `alpha=0.1` and it lost badly (39,854 vs 55,640). I nearly recorded
that as a rejection. But TC's entire purpose is to make an **aggressive** rate
survivable — damping an already-conservative 0.1 down to an effective 0.05
tests the opposite of the claim. The source papers pair TC with `alpha=1.0`,
and E3b had already established that plain `alpha=1.0` collapses to 15,300.

That makes the real question sharp and falsifiable: **does TC make `alpha=1.0`
usable?**

**Result at 100k games** (`seed=20260825`, matched):

| Trainer | alpha | Greedy score | Final mean beta |
|---|---:|---:|---:|
| Plain TD | 0.1 | 55,640 | — |
| Plain TD | 1.0 | 15,300 | — |
| Temporal coherence | 0.1 | 39,854 | 0.49 |
| **Temporal coherence** | **1.0** | **95,371** | 0.48 |

Confirmed independently at n=300 via `run_experiment --depth 1`: **90,508**.

**+71% over the best plain configuration.** Note what that number means in
context: `n1_default.bin` scores 102,861 after **1,000,000** training games.
TC at `alpha=1.0` reaches 90,508 after **100,000** — roughly 88% of the
strength for a tenth of the compute.

The mechanism is visible in beta: 0.48 means TC settled on an effective rate
near 0.48, roughly 5x the hand-tuned 0.1 and 20x below the 1.0 that destroys a
plain run. It found that operating point on its own, per weight, from the error
history — no schedule, no tuning.

**Decision: KEEP, `--temporal-coherence --alpha 1.0`.** Costs 260 MB during
training and nothing at play time.

**The lesson is about experiment design, not about TC.** A correct
implementation with the wrong control produced a confident rejection of the
best idea available. What saved it was asking what the technique was *for*
rather than only whether the number went up.

---

## E6 — Optimistic initialisation  ✅ PROMISING

**Hypothesis.** The papers found explicit exploration (epsilon-greedy) actively
*harmful* in 2048 — the spawn randomness already supplies plenty — so the
remaining way to encourage exploration is optimism. Initialise every weight so
an unseen board evaluates high; a greedy policy is then pulled toward unvisited
patterns automatically, with no extra parameter at play time.

**Implementation.** `--optimistic-init X` sets each weight to
`X / active_weight_count`, so any board reads back ~X before learning. Guarded
against `--resume` (it would silently erase the resumed network).

**Full sweep at 100k games** (`alpha=0.1`, `seed=20260825`, all matched;
trainer's own 60-game held-out evaluation):

| Initial value X | Greedy score | vs default |
|---|---:|---:|
| 0 (default) | 55,640 | — |
| 2,000 | 63,678 | +14.4% |
| 5,000 | 64,926 | +16.7% |
| **20,000** | **68,097** | **+22.4%** |
| 50,000 | 67,769 | +21.8% |
| 200,000 | 59,230 | +6.5% |

**A broad plateau between 20,000 and 50,000**, falling away on both sides. The
flatness matters more than the peak: 20k and 50k differ by 0.5% at n=60, which
is well inside noise, so this is one optimum rather than a knife-edge that
would need re-tuning for every change.

The shape is what the theory predicts. Too little optimism and unvisited
patterns are not attractive enough to draw the greedy policy toward them; too
much and every board looks equally wonderful, so early play is uninformative
and the network wastes its budget unlearning a large constant. Note X = 200,000
is roughly twice the best score this project has ever recorded, which is a
sensible place for the effect to break down.

**Confirmed at n=300** (`run_experiment --search fixed --depth 1`, seeds
30000-30299, so every configuration plays the identical games):

| Initial value X | Score (n=300) | vs baseline |
|---|---:|---:|
| 0 (baseline) | 53,179 | — |
| 2,000 | 66,218 | +24.5% |
| 5,000 | 66,846 | +25.7% |
| **20,000** | **67,919** | **+27.7%** |
| 50,000 | 67,074 | +26.1% |
| 200,000 | 58,756 | +10.5% |

Paired test, X=20,000 vs baseline: **+14,740 (+27.7%), t=+8.73, p~0**, 95% CI
[+11,429, +18,050], winning **218 of 300 seeds**. The n=60 sweep understated
the effect slightly and got the ranking right.

**Decision: KEEP, X=20,000.** No runtime cost whatsoever — it only changes the
starting weights, so the trained network is the same size and the same speed.
This is the first change in this log that improved something without a caveat
attached.

---

## E7 — Training hot path: drop a redundant network evaluation  ✅ KEEP

**Found by reading the trainer's inner loop.** Per move it did:

1. `best_candidate()` evaluates all legal afterstates — up to 4 `value()` calls
2. the loop then called `network.value(candidate.afterstate)` for the TD target
3. `apply_td_update()` evaluates the pending afterstate once more

Step 2 recomputes exactly what step 1 already produced, and **no weight update
happens in between**. Each `value()` is 32 lookups scattered across a 128 MB
table — far beyond L2, so it is cache-miss bound and is the dominant cost of
training.

**Implementation.** `Candidate` now carries `afterstate_value`, stored verbatim
from the evaluation that chose it. Deliberately *not* recovered as
`best_score - reward`: that subtraction does not round-trip exactly in floating
point and would silently perturb every subsequent update.

**Verification.** Same seed, same config, 3,000 games, before and after:
fingerprint `c2d30238a0295782` both times, and `cmp` reports the weight files
byte-identical. This is not "close enough" — it is the same computation with
one redundant pass removed.

**Decision: KEEP.** Provably identical output, strictly less work. Step 3
cannot be cached the same way: one update is applied between the pending
afterstate being chosen and its own update, and that update can touch weights
it shares, so the cached value would be stale. Left alone.

Speedup not yet quantified — that needs an interleaved A/B on a quiet machine
(see the scheduling rule below), and the change cannot be slower, so it is not
gating anything.

---

## E8 — Value calibration: the late game is wildly overvalued  ⭐ ACTIONABLE

**Built to test a hypothesis that turned out to be wrong**, which is the useful
part.

**Hypothesis.** Depth 2 helps a 100k network but hurts a 1M one. 1-ply greedy
only needs V's *ordering*, so a monotone distortion is harmless there — but
depth 2 computes `r1 + E[max(r2 + V)]`, adding learned values to measured
rewards. That is only sound if V is on the same *scale* as the rewards.
So maybe the 1M network is worse calibrated.

**Method.** `scratchpad/calibration_probe.cpp` plays greedy games and, for
every afterstate, compares `V(s')` against the score the game actually went on
to earn from there. V is *defined* as expected remaining score, so a calibrated
V should track it with slope ~1 and no offset. Broken out by game phase,
because a bias confined to one phase is a different defect from a uniform one.

**The hypothesis is refuted.** The 1M network is the *best* calibrated of the
three, and it is the one depth hurts:

| Network | Overall bias | Slope | Correlation | Depth 2 helps? |
|---|---:|---:|---:|---|
| 1M, buggy reward | **+13.6%** | 1.32 | 0.68 | **no** (−6%) |
| 100k, buggy reward | +47.6% | 1.52 | 0.70 | yes (+33%) |
| 100k, fixed reward | +41.2% | 1.66 | 0.64 | yes (+38%) |

Calibration is not the mechanism. The remaining explanation is plainer: at
100k the greedy policy is weak (52-60k) and there is headroom for search to
find real improvements; at 1M it is strong (103k), and the search's own errors
outweigh what is left to gain. That is testable properly with the
depth-vs-budget curve from the 1M run's checkpoints, not by argument.

**But the probe found something else, and it is consistent everywhere:**

| Phase (1M network) | V predicts | Actually earns | Bias |
|---|---:|---:|---:|
| 0-20% | 94,219 | 109,239 | −15,020 |
| 20-40% | 82,609 | 87,225 | −4,617 |
| 40-60% | 70,714 | 63,609 | +7,105 |
| 60-80% | 58,363 | 35,921 | **+22,443** |
| 80-100% | 42,969 | 11,214 | **+31,755** |

**In the final fifth of a game the network expects ~43,000 more points and
earns ~11,000 — off by a factor of four.** The same +28k to +31k late-game bias
appears in all three networks regardless of training budget or reward index.
The value function does not know the game is about to end.

The mechanism is structural: only the single last afterstate of each game gets
the terminal target of 0. States five or ten moves from death are trained
toward `r + V(next)`, inheriting the inflation of their successors, and they
are rare in training compared with mid-game states.

**This is concrete, measured motivation for late-game-focused training**
(roadmap item 7) — no longer a guess that the endgame matters, but a measured
4x error in the exact region where games are decided. It also reframes the
endgame-tablebase work: the gap a tablebase would fill is real and large, even
though the tables themselves remain unaffordable.

---

## E9 — Backward episode replay  ⚠️ works, but not for the reason I built it

**Hypothesis, straight out of E8.** Every network overvalues the last fifth of
a game by ~4x. The structural cause looked clear: with forward TD(0), only the
final afterstate of each episode ever receives the terminal target of 0, and
`V(s'_t)` is updated toward `r + V(s'_{t+1})` using the successor's value from
*before* it learned anything this episode. So "the game ends here" crawls
backward one state per episode. Replaying each episode in reverse should carry
the terminal signal along the whole trajectory in a single pass.

**Implementation.** `--backward-updates`. Buffers each episode's afterstates
and rewards, then walks backward so every successor is already updated when
its predecessor reads it. Still TD(0) — same targets, different order. Verified
deterministic (identical fingerprints across runs).

**The hypothesis is REFUTED.** Matched pair, 3,000 games, `seed=7`:

| Order | Late-game bias (last 20%) | Overall bias | Mean score |
|---|---:|---:|---:|
| Forward | +1,805 | −25.0% | 5,360 |
| Backward | **+1,939** | −16.1% | **5,697** |

Late-game overvaluation is **not** reduced — it is marginally worse. Whatever
sustains that 4x error, it is not the propagation delay I assumed.

**But the change helps substantially on its own merits.** At 100k games,
matched (`alpha=0.1`, `seed=20260825`):

| Order | Greedy score | vs forward |
|---|---:|---:|
| Forward | 55,640 | — |
| **Backward** | **62,945** | **+13.1%** |

Confirmed at n=300: **62,716** vs the 53,179 baseline (+17.9%).

So backward replay works — just not for the reason it was built. The honest
position is that the *mechanism* is unexplained: I had a theory, it made a
specific prediction about late-game bias, the prediction failed, and the
benefit is real anyway.

### It does not stack with optimistic initialisation

| Configuration (100k, `alpha=0.1`, n=300) | Score | vs baseline |
|---|---:|---:|
| plain | 53,179 | — |
| backward replay only | 62,716 | +17.9% |
| **optimistic init only** | **67,919** | **+27.7%** |
| optimistic + backward | 66,097 | +24.3% |

Paired, optimistic+backward vs optimistic alone: −1,822 (−2.7%),
**p=0.30 — a tie**, winning 139 of 300 seeds.

Each helps substantially on its own; together, the second one adds nothing.
They are **substitutes, not complements**, which suggests both are relieving
the same underlying deficiency rather than two independent ones. Worth stating
because the natural assumption — two orthogonal-looking mechanisms, one an
initialisation and one an update order, so surely they compound — is wrong, and
I would have shipped it into a multi-hour training run untested.

**Decision: KEEP backward replay as an option, but prefer optimistic
initialisation** when choosing one. Optimistic init is strictly cheaper: it
touches only the starting weights, whereas backward replay buffers every
episode and costs an extra network evaluation per move.

**Test-suite consequence.** I had written a test asserting backward replay
reduces late-game overvaluation. It failed, and I removed it rather than
loosening it — pinning an appealing but false claim into the suite is worse
than having no test. What remains asserts only what is measured: the path is
distinct, deterministic, and learns.

**Still open from E8:** the late game really is overvalued 4x, and the obvious
mechanism is now ruled out. Next candidate: near-terminal states are simply
rare in training, so the fix is to *sample* them more (start episodes from
late-game positions) rather than to change update order.

### Follow-up: calibration and playing strength are close to unrelated

Adding the TC network (E5) to the E8 table makes the point sharply. It is the
**worst calibrated** network measured and by far the **strongest per unit of
training**:

| Network | Overall bias | Late-game bias | Correlation | Score |
|---|---:|---:|---:|---:|
| 1M plain | +13.6% | +31,755 | 0.676 | 102,861 |
| 100k plain | +47.6% | +30,752 | 0.704 | ~60,600 |
| **100k TC, alpha=1.0** | **+57.2%** | **+46,546** | 0.659 | **90,508** |

Greedy action selection is an argmax, so it consumes only the *ordering* of V.
A large, smoothly varying overestimate is nearly free — it cancels in the
comparison. That is why the correlations are all ~0.66-0.70 while the biases
range over a factor of four and the scores do not follow the bias at all.

Two consequences worth stating:

1. **Calibration is the wrong thing to optimise** for a 1-ply agent, and
   chasing it would have been wasted effort.
2. **It should still matter for search.** Depth 2 computes `r1 + E[max(r2 +
   V)]`, adding learned values to measured rewards across levels, which unlike
   an argmax cannot ignore scale. Whether depth helps the TC network is
   therefore a real test rather than a routine measurement — running now.

The 4x late-game error has survived every training change tried (reward index,
update order, step-size rule, training budget). It is looking structural: the
n-tuple features simply cannot distinguish "big tile with room to move" from
"big tile on a jammed board", and one weight set has to average them. That is
precisely the argument for multi-stage networks — an argument that sounded
strong and that E11 measured to be **wrong** (−19.4%).

---

## E10 — Selectable network shapes  ✅ KEEP (infrastructure)

Roadmap item 5 needs a bigger network; item 6 (multi-stage) needs several
networks at once. Both were blocked by the same thing: the tuple shape was
hardcoded, and `NTupleNetwork::load()` validated a file against a shape the
caller had to already know. So only one shape could ever be trained or played.

**Two changes.**

`named_tuple_specs(name)` with `--tuples NAME`:

| Name | Shape | Weights | Size |
|---|---|---:|---:|
| `default` | 2x 4-tuple + 2x 2x3 rectangle | 33.7M | 128 MB |
| `large` | 5x 6-tuple | 83.9M | 320 MB |

An unknown name throws and lists the valid ones — a typo must not silently
train a different network than the operator asked for and then label the result
with the name they typed.

`NTupleNetwork::load_from(path)` builds a network from the file's *own*
embedded tuple definitions. Weight files were already self-describing; nothing
could act on it. `run_experiment` now uses this, so any trained shape plays
without the binary being told which. Strict `load()` is retained for the case
where the caller does know what to expect.

**Note on the `large` cell sets.** The published networks at this size are
described by shape rather than by explicit cell lists, so these five particular
6-tuples are *our* choice, not a transcription — documented as such in the
source. The selection principle is coverage variety (two shapes straddling a
row boundary, two horizontal rectangles, one vertical) so that different tuples
fail on different boards rather than sharing a blind spot. Whether it is any
good is a measurement, not a citation.

Four tests added, including one that pins both configurations' weight counts
arithmetically — a silent change to the shapes would invalidate every memory
budget in this document.

---

## E11 — Multi-stage networks  ❌ REJECT — my own "best next direction" was wrong

**I named this the most promising remaining idea in the deliverables summary,
then measured it. It loses badly.**

**Motivation (from E8).** A single weight set must serve boards whose
remaining-score scales differ by an order of magnitude: a 4096 tile early means
"lots left to earn", the same tile on a jammed board late means "about to die".
One table has to average the two, and E8 measured that averaging costing a 4x
overvaluation of the last fifth of a game. Multi-stage (Jaskowski 2016) gives
each phase its own weight set, which should remove the averaging.

**Memory was never the obstacle it was recorded as.** This project had
multi-stage down as *out of reach at 15 GB*, but that was the full published
configuration (2^4 stages x 3 networks for TC). Measured for a reduced version:

| Stages | `default` shape | + TC state | Total |
|---:|---:|---:|---:|
| 1 | 128 MB | 257 MB | 385 MB |
| **2** | **257 MB** | **514 MB** | **771 MB** |
| 4 | 514 MB | 1.03 GB | 1.54 GB |

Comfortably affordable. So the idea got a fair test.

**Result — 2 stages, otherwise the best known configuration**
(`--temporal-coherence --alpha 1.0 --backward-updates`, 100k games, matched
seeds):

| Configuration | Score (n=300) |
|---|---:|
| unstaged | **111,751** |
| 2 stages | 90,123 |

Paired: **−21,628 (−19.4%), t=−5.77, p=7.7e-09**, winning only **111 of 300
seeds**. Not marginal, not noise.

**Why it fails here.** Splitting the weight table splits the *training data*
with it. The n-tuple network's strength is generalisation through weight
sharing — 8 dihedral orderings of every tuple index one LUT — and staging
deliberately breaks that sharing across game phases. At 100k games there is not
enough data for two independently-trained networks to each reach the quality of
one jointly-trained network. The published multi-stage results come from
training runs orders of magnitude longer, where that trade flips.

**Decision: REJECT at this training budget.** The implementation stays: it is
tested, costs nothing when `--stages 1`, and becomes worth retrying if training
budget ever grows by an order of magnitude. But it is not the way forward on
this hardware.

**What this says about the E8 bottleneck.** Late-game overvaluation is real and
large, but it has now resisted: the reward-index fix, backward replay, temporal
coherence, a 10x training-budget increase, *and* the structural fix that
targeted it most directly. Either it is not actually what limits playing
strength — which E8's own finding that calibration barely affects an argmax
policy would support — or it needs a mechanism none of these provide.

---

## E15 — Backward replay buys speed, not quality  ⭐ FINDING (and it qualifies E13)

E13 measured backward replay as worth **+23.5%** on top of temporal coherence
at 100k games (p=1.4e-10), and I restarted the 1M run on that basis. The 200k
checkpoints of both runs now exist, and they are **identical in strength**:

| Training games | TC alone | TC + backward | Difference |
|---:|---:|---:|---:|
| 100,000 | 90,508 | **111,751** | **+23.5%** (p=1.4e-10) |
| 200,000 | 117,665 | 116,774 | **−0.8%** (p=0.83, tie) |

At depth 2 the same holds: 131,481 vs 125,199, within noise of each other.

**Backward replay is a convergence-speed effect, not a final-quality effect.**
It reaches a given standard of play in roughly half the games, and then the
plain-TC run catches up completely. That makes sense in hindsight: propagating
the terminal signal along a trajectory within one episode is a way of *using
each episode harder*, which matters when episodes are scarce and stops
mattering once they are not.

**Consequence: my restart was unnecessary.** TC alone would have arrived at the
same place. The decision was still correct on the evidence available — a
measured +23.5% at the only budget then tested, against 21 minutes of sunk
training — but the fuller picture shows it bought nothing.

**The general point, and it is the same shape as E3.** An improvement measured
at one operating point need not hold at another:

- E3: depth-2 gain shrinks as the evaluator strengthens (+38% → −6%)
- E15: backward replay's gain shrinks as training budget grows (+23.5% → 0%)

Both were measured at a single point first and looked like constants. **Any
result quoted here should carry the budget and base strength it was measured
at**, because several of them are functions of exactly those.

**Decision: KEEP backward replay for short runs, where it roughly halves the
games needed. It is not part of the long-run configuration.**

---

## E16 — A bigger tuple set is a large, clean win  ⭐ KEEP — new best agent

Roadmap item 5, unblocked by E10's selectable network shapes. `--tuples large`
is five 6-tuples (83.9M weights, 320 MB) against the default's two 4-tuples
plus two 2x3 rectangles (33.7M, 128 MB).

**Result at 100k games**, otherwise the best known configuration
(`--temporal-coherence --alpha 1.0 --backward-updates`), matched seeds:

| Network shape | Depth 1 | Depth 2 | Max tile |
|---|---:|---:|---|
| `default` (128 MB) | 111,751 | — | 8,192 |
| **`large` (320 MB)** | **135,043** | **165,524** | **16,384** |

Paired at depth 1: **+23,292 (+20.8%), t=+5.81, p=6.4e-09**, winning **193 of
300 seeds**. At depth 2 it reaches **165,524** with 95% CI [158,762, 172,285] —
the strongest agent in this project by a wide margin, and the first to reach
**16,384** (in 14% of games).

**I predicted this cautiously and was wrong in the useful direction.** After
E11 (multi-stage, −19.4%) I noted that a bigger table also means each weight is
visited less often at a fixed budget — the same data-dilution effect, milder.
That reasoning was wrong, and the distinction is worth stating precisely:

- **Multi-stage partitions the data.** Each stage sees a disjoint subset of
  positions; a board in stage 1 contributes nothing to stage 0's weights.
- **A larger tuple set does not.** Every board still trains every tuple. The
  features are finer-grained, so each individual weight is visited less often,
  but no training signal is withheld from any part of the network.

Adding capacity and dividing capacity are different operations, and only the
second one starves the learner.

**Cost.** 320 MB resident for play (960 MB during TC training, as originally
estimated for this configuration). 40 active weights per evaluation against 32,
so roughly 25% more work per node — comfortably repaid by a 20.8% score gain at
equal training.

**Decision: KEEP. `--tuples large` is now part of the best configuration.**

---

## E17 — Why search hurts: the value function cannot see the board  ⭐ DIAGNOSIS

**The goal is now maximum average score, with 226,325 as the baseline to beat.**
Extrapolating the two proven levers says brute force will not get far:
+16.5% per doubling of training games and +20.8% for 128 MB -> 320 MB means
reaching 500,000 needs roughly **5 more doublings, about 37M games, 150+ hours**.

**The missing multiplier is search.** Published strong agents gain ~58% going
1-ply -> 3-ply. Ours *loses* **18.7%** at depth 2. Recovering that is worth more
than any amount of extra training, so it is the thing to diagnose.

### First: it is not a bug

Added a gate comparing depth-2 expectimax against an **independent brute-force
2-ply computation** written from the definition, sharing no code with
`Expectimax` beyond move generation. It passes exactly (1e-6) on four hand-built
positions. Also confirmed the regression is unchanged with
`--transposition-table off`, and depth 1 reproduces at 226,994 on the
diagnostic seed set.

So depth 2 computes the right number and the right number is worse.

### The cause: V is wildly miscalibrated, and search amplifies it

Calibration of the final 1M network (`calibration_probe`, 25 games, 220k
samples):

| Phase | V predicts | Actually earns | Bias |
|---|---:|---:|---:|
| 0-20% | 215,109 | 234,526 | −19,417 |
| 20-40% | 197,947 | 186,167 | +11,780 |
| 40-60% | 185,906 | 134,307 | +51,599 |
| 60-80% | 169,709 | 76,459 | +93,250 |
| **80-100%** | **137,053** | **23,223** | **+113,830 (5.9x)** |

Overall bias +38.3%, slope 1.485, and correlation only **0.558** — *lower* than
the weaker networks measured in E8 (0.66-0.70), despite far stronger play.

**Why that breaks search specifically.** At depth 1 the agent computes
`max(r + V)` once. A large, smoothly varying overestimate mostly cancels inside
an argmax — which is exactly E8's finding that calibration barely matters at
1 ply. Depth 2 computes

    max_a1 [ r1 + E_spawn[ max_a2 ( r2 + V ) ] ]

so the **inner** max is taken over noisy, inflated estimates at every chance
node. It is biased upward by roughly the size of V's error, that error varies
by position, and the outer max then *selects for the positions where V is most
wrong*.

### The depth sweep does not fully support that story

| Depth | Score (n=100) | vs depth 1 |
|---:|---:|---:|
| 1 | 226,994 | — |
| 2 | 179,847 | **−20.8%** |
| 3 | 219,168 | −3.4% |
| 2, `--transposition-table off` | 179,847 | identical |

**Search quality is non-monotone in depth: 1 ~= 3 > 2.** With SE ~9,000 at
n=100 and matched seeds, the 1-vs-2 and 2-vs-3 gaps are real; 1-vs-3 is not.

That **contradicts a pure max-bias explanation**: depth 3 performs *more* max
operations over inflated estimates than depth 2, so it should be worse still,
and it is not. Whatever singles out depth 2, accumulating maxes is not
sufficient to describe it.

Recorded as an open question rather than resolved. Two things it is *not*:
a bug (the brute-force gate passes exactly) and the transposition table
(TT off reproduces the number byte for byte, which also makes the TT provably
neutral here rather than merely assumed so).

The practical conclusion is unaffected and is what matters for the goal:
**no search depth beats depth 1**, and depth 3 merely draws with it at ~13x the
cost. Search remains unusable until V improves, so V is what to fix.

### The structural reason V cannot fix itself

**No feature in this network sees the whole board.** A 6-tuple covers 6 of 16
cells. "The board is nearly full and I am about to die" is a global property
that **no single feature can express** — only the sum can approximate it, and
badly. That is why the bias is concentrated late: a jammed board and a healthy
board look similar through any 6-cell window with the same tiles in it.

This also explains why every previous attempt on this bottleneck failed. The
reward-index fix, backward replay, temporal coherence, a 10x budget increase,
and multi-stage all change *how* the network is trained. None of them gave it
the information it was missing.

**Next experiment (E18): give it that information.**

---

## E18 — A whole-board feature  ⚠️ DOWNGRADED — it delays the search decay, does not remove it

One extra table indexed by two cheap global quantities:

    index = empty_count (0..15) * 16 + max_exponent (0..15)      // 256 weights

256 weights against 83.9M — memory-free — and **the first feature in this
network that looks at the entire board**. Exposed as `--global-features`.

Four tests, the load-bearing one being that two boards identical inside every
tuple's cells but differing elsewhere get **different** global indices. A
feature that fails that is decorative.

File format grew a version-3 header with a flags word; versions 1 and 2 stay
readable, and the writer emits the *oldest* version that can express the
network, so every existing weight file remains valid and byte-identical.
`n5_large_1M.bin` re-verified after the change.

### Result: it does exactly what it was built to do, and the score got worse

100k games, `large` shape, otherwise the best configuration, matched seeds:

| Metric | Without global | With global |
|---|---:|---:|
| Overall calibration bias | 22.0% | **4.2%** |
| Slope (1.0 = correctly scaled) | 1.473 | **0.879** |
| **Correlation (ordering quality)** | **0.588** | **0.325** |
| **Score at depth 1** (n=300) | **135,043** | 125,488 |

Calibration improved **5x**. Playing strength fell **7%**. Score tracked
*ordering*, not calibration — a third independent confirmation of E8.

**Mechanism: feature competition.** The global weight is active on *every*
board, so it absorbs value signal the fine-grained tuples should have modelled.
It buys a better *level* at the cost of *discrimination*, which is precisely the
wrong trade for an argmax policy that only ever consumes ordering.

At that point the honest reading was "correct diagnosis, wrong lever" — E17
found a real 5.9x defect and fixing it made the agent worse.

### But then: it unlocks search, and search more than pays it back

E17 predicted calibration was what made depth 2 harmful. That is a sharp,
falsifiable claim, and the well-calibrated network tests it directly:

| Network | Calib. bias | Depth 1 | Depth 2 | Depth gain |
|---|---:|---:|---:|---:|
| `large`, no global | 22.0% | **135,043** | 165,524 | +22.6% |
| `large` + global | **4.2%** | 126,332 | **173,039** | **+37.0%** |

**The prediction holds, and the effect grows with depth** — which is the
signature the mechanism predicts. Full sweep at 100k games, matched seeds,
paired tests against the plain network at the same depth:

| Depth | plain `large` | + global | Difference | p |
|---:|---:|---:|---:|---:|
| 1 | **135,043** | 126,332 | −6.5% | — |
| 2 | 165,524 | 173,039 | +3.1% | 0.37 (tie) |
| 3 | 179,312 | **210,266** | **+17.3%** | **3.5e-07** |

Global features **cost 6.5% at 1 ply and win 17.3% at depth 3**, winning 128 of
200 seeds. The crossover is between depth 1 and 2; by depth 3 it is decisive.

Note the depth-2 head-to-head is only a **tie** — reporting it as a win would
have been wrong, and the depth-3 measurement is what actually establishes the
claim.

So global features are not a calibration fix that happens to cost ordering.
They are a **trade: ordering for search-compatibility**, and with search
available the trade is profitable.

**Why this matters far more than the 4.5% it wins today.** The 1M network
*loses* 18.7% at depth 2 — search is unusable exactly where the agent is
strongest, which is what capped the whole project. If calibration is what
restores search, then a well-calibrated network at 1M games should be able to
use depth 2 or 3 and collect a multiplier that plain training cannot reach.

### Then more training ate the advantage

The 100k result above looked like a breakthrough. Training the same
configuration to 200k games says otherwise:

| Global-features network | Depth 1 | Depth 3 | Depth gain |
|---|---:|---:|---:|
| 100k games | 126,332 | **210,266** | **+66%** |
| 200k games | **158,036** | 195,010 | +23% |

Depth 1 improved 25%; **depth 3 fell 7.3% (p=0.002, paired n=200)**. The
searched agent got *worse* with more training.

**So calibration shifted the depth-gain curve up without changing its slope.**
Put beside every other network here, the decay is universal:

| Network | gain at 100k | at 200-300k | at 1M |
|---|---:|---:|---:|
| `default` | +16.6% | +11.7% | — |
| `large` | +22.6% | +9.0% | **−18.7%** |
| `large` + global | **+66%** | +23% | (on trend, ~0) |

**Decision: DOWNGRADE to neutral.** Global features are a clear win at small
training budgets and appear to converge to nothing at large ones. Nothing in
this experiment has beaten the 226,325 baseline: the best global-features result
is 210,266, and its successor at double the training is 195,010.

**What the pattern actually says.** Search's benefit is bounded by the
evaluator's error. Improving the evaluator shrinks the error *and* shrinks the
room search had to exploit, so evaluator-plus-search converges regardless of
which lever is pulled. That is why E17's diagnosis was correct about the
mechanism and still did not yield a durable gain — calibrating V moved where the
curve sits, not what it does.

**Consequence for strategy: stop trying to make search pay.** Three independent
attempts (depth sweeps, calibration, deeper search on a calibrated net) all land
in the same place. The final agent is a **depth-1 agent**, so the only thing
that matters is the ordering quality of V, and the only measured levers on that
are training budget and capacity.

---

## E19 — Capacity does not keep scaling  ⚠️ CORRECTS MY OWN RULE FROM E16

E16 found 128 MB -> 320 MB worth **+20.8%** and I generalised it to "adding
capacity works, dividing capacity does not." The next step up refutes the first
half of that.

`xlarge` = eight distinct 6-tuples, 134M weights, 512 MB. Same configuration
otherwise (`--temporal-coherence --alpha 1.0 --backward-updates`), same seed,
100k games:

| Shape | Weights | Size | Score at 100k |
|---|---:|---:|---:|
| `default` (2x4-tuple + 2x 2x3) | 33.7M | 128 MB | 111,751 |
| **`large` (5x 6-tuple)** | 83.9M | 320 MB | **135,043** |
| `xlarge` (8x 6-tuple) | 134M | 512 MB | 106,780 |

**Capacity has an optimum at a given training budget, and 512 MB is past it at
100k games** — worse than 320 MB, and barely better than 128 MB.

**Mechanism, and it is *not* the multi-stage one.** Multi-stage failed because
it *partitioned* the data: a board in stage 1 contributed nothing to stage 0.
`xlarge` withholds nothing — every board still trains every tuple. It fails for
a simpler reason: **64 active weights per evaluation against 134M total means
each individual weight is visited far less often**, so at a fixed number of
games the network is straightforwardly undertrained. Beta confirms it is still
learning hard (0.53, essentially the same as the others), not diverging.

So the corrected rule is duller and more useful than mine:
**capacity and training budget have to scale together.** Neither the E16 gain
nor this loss is a property of the shape alone.

### The "320 MB is optimal" conclusion is narrower than I stated

Worth writing down plainly, because I have quoted 320 MB as the best shape
several times since: that comparison was run at **100k games, without global
features, evaluated at depth 1**. All three conditions have since changed, and
each one undermines the conclusion differently:

- **Budget.** 100k systematically disadvantages larger networks (below).
  `large` gained 68% going 100k -> 1M; `xlarge` at 1M has never been measured.
- **Global features.** They work by trading ordering quality for calibration
  (E18). A larger network has more ordering quality to trade, so the optimum
  capacity under that regime is genuinely unknown.
- **Depth.** The comparison used depth 1. The best configuration now uses
  depth 3.

**Status: 320 MB is the best shape measured, at a budget and configuration no
longer in use.** It is not established as optimal for the current agent.

### This exposes a flaw in my own pilot methodology

Every pilot in this log runs 100k games. That budget **systematically
disadvantages larger networks**, because parameters-per-game is exactly what
they are short of. `large` scores 135,043 at 100k and 226,325 at 1M — a 1.68x
improvement from training alone. `xlarge` may well overtake it given the same
1M, and a 100k pilot cannot see that.

**Consequence: a 100k pilot is a valid screen for training-method changes
(which do not change parameter count) and an actively misleading one for
capacity changes.** The `large` vs `xlarge` question needs a matched
higher-budget comparison, not a cheaper one.

---

## E20 — Temporal-coherence state is now persistable  ✅ KEEP (unblocks long training)

With search abandoned (E18) the final agent is depth-1, so the only measured
levers left are **training budget** and **capacity**. Budget is the reliable one
(+16.5% per doubling, never plateaued) — and it was blocked by an implementation
detail.

**The blocker.** TC keeps two accumulators per weight and they were never
written to disk. `--resume` restored weights only, so every `beta` restarted at
1.0. On a converged network at `alpha=1.0` that means full-size steps
everywhere — plain TD at `alpha=1.0`, which E3b measured collapsing to 15,300.
So training was **all-or-nothing**: a run had to reach its final quality in one
invocation or be thrown away.

**Fix.** `--tc-state PATH` persists the accumulators to a sidecar file
(deliberately *not* inside the weight file, so a network for play stays exactly
its weight size and nothing that only reads weights needs to know this exists).
Resuming loads them when the file exists, and prints an explicit **warning**
when it does not, because silently restarting the step sizes is the failure this
was built to prevent.

Two tests: the accumulators round-trip such that a restored learner produces a
bit-identical next step, and a state file from a differently-sized network is
rejected rather than applied to the wrong weights.

**Consequence.** Training is now incremental. The best network can be extended
indefinitely instead of being capped by whatever fits in one run.

**Caveat on the existing best network.** `n5_large_1M.bin` predates this, so its
accumulators are gone and any extension of it necessarily starts with fresh TC.
Piloting that at a deliberately conservative `alpha=0.1` (so the initial
beta=1.0 does not damage 1M games of learning) and on a fresh seed stream, with
`--tc-state` set so every future extension resumes properly.

---

## E21 — Search was valuing DEATH at ~137,000 points  🔴 BUG, now fixed

**This invalidates most of the search conclusions in this log**, including ones
I called robust. Recording it prominently for that reason.

### The bug

`Expectimax::player_value`, on reaching a position with **no legal move**:

```cpp
if (!found_move) {
    best = leaf_value(board);   // asks the network what a dead board is worth
}
```

A dead board is worth **exactly zero** further points — the game is over. But
the network overestimates the endgame by ~5.9x (E17), and a dead board is
precisely a full board covered in large tiles, so it scored around **137,000**.

**Search was therefore rewarded for reaching terminal positions.** It did not
merely tolerate death; it actively steered toward it.

### It explains every symptom I had attributed to theory

| Observation | Real cause |
|---|---|
| Depth 1 unaffected | afterstate semantics expand no spawns, so no terminal board is ever reached |
| Depth 2 loses 18.7% | terminal boards appear after spawns, valued at ~137,000 instead of 0 |
| Worse for stronger networks | they survive longer, reach fuller boards, and overestimate the endgame more |
| Global features *helped* search (+17.3%) | better calibration shrank the phantom terminal value |
| Depth 3 partly recovers | more real reward accumulates along the path, diluting the phantom leaf |

**The measured effect of the fix**, same weights, same seeds:

| Config | Before | After |
|---|---:|---:|
| `n5_large_1M` depth 1 | 226,325 | 226,325 (unchanged, as predicted) |
| **`n5_large_1M` depth 2** | 184,096 | **306,417** |

**+66% from the fix, and +35% over the best 1-ply agent.**

### What this retracts

- **"Search's benefit shrinks as the evaluator improves" (E3/E14) was largely an
  artifact.** The stronger the network, the larger its phantom death-reward.
  The relationship needs re-measuring from scratch.
- **"Stop trying to make search pay" (E18) was wrong**, and I acted on it: I
  killed a training run on that basis.
- **E18's downgrade of global features was wrong twice over** — the matched
  comparison at 300k (run only because the user pushed back) had already shown
  global *ahead* at depth 3, 201,431 vs 184,836.

### Why the test suite missed it

`test_gate_depth2_matches_brute_force` had correct logic — its reference sets
dead positions to 0 — but used only **sparse** boards, where death cannot occur
within two plies. **The terminal branch never executed.** A correctness gate
that never reaches the interesting branch is not a gate. Fixed by adding
nearly-full boards to the fixture.

### The lesson worth keeping

I built an elaborate and self-consistent theory (max-bias, error amplification,
train/play distribution mismatch) to explain a set of measurements produced by a
one-line bug. Every prediction it made was confirmed, because the bug's
signature genuinely does scale with evaluator strength. **A mechanism that
explains the data is not thereby the mechanism** — and the tell was there all
along: the published literature gets +58% from 3-ply on a *stronger* evaluator
than mine, which my theory said was impossible. I noted that contradiction and
theorised around it instead of treating it as evidence of a defect.

---

## E22 — Audit: which conclusions survive the E21 bug?  ✅ CLEAN SPLIT

Requested after E21, since a bug that inflated terminal positions by ~137,000
could have contaminated anything. It did not contaminate everything, and the
dividing line is sharp: **the bug is only reachable at depth >= 2.**

At depth 1 with afterstate semantics the search never expands a spawn, so no
terminal position is ever evaluated. Every depth-1 measurement is therefore
untouched.

### Conclusions that SURVIVE unchanged (all measured at depth 1)

| # | Conclusion | Why it holds |
|---|---|---|
| E1 | Afterstate leaf semantics (14,262 -> 102,861) | depth 1 |
| E2 | Reward index fix loses 13.3% | depth 1 |
| E3b | Plain `alpha=1.0` collapses | depth 1 |
| E5 | **Temporal coherence @ alpha=1.0, +71%** | depth 1 |
| E6 | Optimistic init +27.7% | depth 1 |
| E7 | Hot-path dedup byte-identical | no search |
| E9 | Backward replay +17.9% | depth 1 |
| E11 | Multi-stage −19.4% | depth 1 |
| E13 | Selective compounding | depth 1 |
| E15 | Backward replay is speed not quality | depth 1 |
| E16 | `large` network +20.8% | depth 1 |
| E19 | `xlarge` −21% at 100k | depth 1 |
| E20 | TC state persistence | no search |

**Every training-method conclusion is depth-1 and stands.** The recipe that
built the agent — TC at `alpha=1.0`, backward replay, the `large` shape — is
unaffected.

### Conclusions RETRACTED

| # | Claim | Status |
|---|---|---|
| E3 | "Depth gain tracks base strength (+38% -> −6%)" | **retracted** — the decay was the bug scaling with evaluator strength |
| E14 | Depth-vs-budget curve | **retracted** — same cause |
| E17 | "Search amplifies value error" | **retracted** — it was diagnosing a bug, not a property |
| E18 | "Global features erode / are neutral" | **retracted twice** — post-fix they win at BOTH depths |
| E12 | H5 at depth 4 = 109,213 | **suspect** — H-series used the old terminal path; re-measuring |

### A third defect found during the same audit

`chance_value` substitutes a static evaluation when a branch is cut. It called
`leaf_value(spawned)` — a **post-spawn** board — which is exactly the E1
mismatch, for an afterstate evaluator. Depth exhaustion returns earlier so this
never fired in the fixed-depth regime (cutoff 0), but the **probability cutoff
reaches it at any depth**, so it was a latent defect waiting for the first timed
N-series run. Fixed: afterstate evaluators now fall back to the afterstate they
arrived at.

### And a fourth, which my first fix would have introduced

I initially hardcoded terminal positions to 0 for afterstate evaluators and left
hand-crafted ones alone. Probing the heuristics showed **all of them return
negative values** (H0 to −5,844, H4 to −6.3e6, H5 to −12,085), so had I applied
0 to them, **death would have ranked above many bad-but-alive positions** — the
same bug in a new place. Terminal valuation is now a property of the evaluator
(`Evaluator::terminal_value()`): 0 for a score-predicting value function, and
"worse than anything achievable" by default.

The suite caught this immediately, because the mock evaluator inherited the
positional default while declaring afterstate semantics.

---

## E23 — Post-fix search sweep, and where the ceiling actually is

With E21 fixed, every depth result was re-measured on `n5_large_1M.bin`
(unchanged weights — nothing retrained):

| Depth | Score | n | Cutoff | Max tile |
|---:|---:|---:|---:|---|
| 1 | 226,325 | 200 | — | 16384 |
| 2 | 306,417 | 200 | 0 | 16384 |
| 3 | 334,030 | 200 | 0 | 16384 |
| **4** | **356,178** | 60 | 0.0015 | **32768** |
| 5 | 337,652 | 30 | 0.004 | 32768 |

Depth 5 came in below depth 4, but with a **coarser probability cutoff**, so
depth and pruning were varied together — the same error that produced the
retracted E3. A depth-4-at-0.004 control is running to separate them.

### Training has plateaued at this network size

| Training games | Depth 1 | Depth 4 |
|---:|---:|---:|
| 1.0M | 226,325 | **356,178** |
| 1.1M | 242,440 | — |
| 1.2M | 230,309 | 353,171 |

Non-monotone at depth 1 and a tie at depth 4, all within a ±13,000 interval.
**1.0M -> 1.2M produced nothing measurable.**

This corrects the "+16.5% per doubling, no plateau" claim: that came from
100k -> 1M, which is 3.3 doublings. At 1M, a 200k increment is **0.26 of a
doubling** — expected gain ~4%, which is below this benchmark's noise floor.
Further training gains are real but can only be *detected* a full doubling at a
time (1M -> 2M, ~20 hours).

**Consequence:** progress from here is bought with hours, not tricks. The
remaining measured levers are global features (+14.5%), a training doubling
(+16.5%), and possibly depth 5.

---

## E24 — Two search speedups: one rejected, one kept

Depth is the largest lever, and search speed is what makes depth affordable, so
both were worth trying.

**Symmetry reduction — ❌ REJECT.** N1 is rotation-invariant by construction, so
canonicalising transposition keys lets 8 symmetric positions share an entry.
Measured at depth 3: **204 s with, 128 s without** — 60% *slower*. Computing the
canonical form costs more per node than the extra hits save.

It also exposed something worth knowing: scores differed slightly (322,556 vs
329,373) when they should be identical. N1 sums the same weights in a different
order under rotation, and floating-point addition is not associative, so values
can differ in the last bit and flip close comparisons. **Symmetry reduction is
not exactly value-preserving for this evaluator**, independent of its cost.

**Parallel root search — ✅ KEEP.** `ParallelExpectimax` runs each root move in
its own thread with its own transposition table (the tables are per-instance
mutable state, so sharing one would race). Verified over 120 moves at depth 3
and 25 at depth 4: **zero direction mismatches** — identical play, not merely
similar.

| Depth | Speedup |
|---:|---:|
| 3 | 1.30x |
| 4 | **1.68x** |

Well short of the ~3.5x the four root moves suggest, because subtree sizes are
very unequal and the largest one bounds the result. Better at depth 4, where the
work amortises thread launch. Useful for benchmark throughput; it does **not**
buy a depth level, since each level costs ~25x.

---

## E25 — Global features and depth are SUBSTITUTES  ❌ closes the 500k projection

The decisive matched test. Both models trained to **1M games**, identical
configuration otherwise, benchmarked on identical seeds:

| Model | Depth 1 | Depth 4 |
|---|---:|---:|
| plain `n5_large_1M` | 226,325 | **356,178** |
| global `n10_global_ext` | **234,885** | 331,995 |

Paired at depth 4: **−24,183 (−6.8%), p=0.077**, winning 25 of 60 seeds. A tie
trending against global.

**Global features win at depth 1 (+3.8%) and lose the advantage entirely by
depth 4.** They do not stack.

### This is the third time, and the pattern is now clear

| Combination | Each alone | Combined |
|---|---|---|
| optimistic init + temporal coherence | +27.7% | **−14.9%** |
| backward replay + optimistic init | +17.9% | tie |
| **global features + depth 4** | **+14.5%** (at depth 2) | **tie** |

Every one of these works by compensating for the *same* underlying weakness:
the value function misjudges positions near the end of a game. Global features
fix it by supplying whole-board information. Depth-4 search fixes the same bad
decisions by looking far enough ahead to see the consequence directly. Once
either has corrected the error, the other has nothing left to correct.

**The rule this project keeps relearning:** two improvements that each fix the
same defect are substitutes, not complements — and there is no way to tell
which case you are in without running the combination. Mechanism-based
reasoning has been wrong on this every single time here.

### Consequence for the 500,000 target

My projection of ~408,000 assumed global's +14.5% would multiply onto depth 4's
356,178. **That route is closed.** Best agent remains
**`n5_large_1M` at depth 4 = 356,178** — plain features, no global.

Two levers remain, both hours of compute rather than insight, and both now
running:

1. **A full training doubling** (1.2M -> 2.0M games). The only measured lever
   not yet shown to be a substitute for something else. `n12_plain_2M`.
2. **`xlarge` (512 MB) at 1M games.** Rejected earlier at −21%, but only on a
   100k pilot that E19 proved is structurally biased against larger networks.
   Genuinely untested at scale. `n13_xlarge_1M`.

**Honest ceiling estimate for this architecture on this machine: 400,000 to
430,000.** Reaching 500,000 would need the larger network to scale better than
the current one does, which is exactly what run 2 tests.

---

## E26 — Endgame autopsy: 95% of games die the same way  ⭐ DIAGNOSIS

Score is set almost entirely by the highest tile reached (depth 4, n=60):
8192 -> 162,320 · **16384 -> 355,226 (93%)** · **32768 -> 576,688 (3%)**.
Reaching 500,000 means reaching 32768 routinely.

Achievement rates by depth, same weights:

| Depth | 16384 | 32768 |
|---:|---:|---:|
| 1 | 54% | **0%** |
| 2 | 86% | 0% |
| 3 | 94% | 2% |
| 4 | **97%** | **3%** |

**16384 is saturated at 97% by depth 4; 32768 moves for nothing tried.**

So before proposing fixes, I replayed 40 depth-4 games and classified how each
one ended (`scratchpad/endgame_autopsy.cpp`):

| Cause of death | Games |
|---|---:|
| **Never assembled a second 16384** | **38 / 40** |
| Died with the large-tile chain broken | 2 / 40 |
| Died jammed, chain intact | 0 |
| Died one merge short (mergeable 16384 pair) | 0 |

Only **2 of 40 games ever held two 16384s at once.**

**The agent is not losing on endgame tactics.** It does not die jammed, does not
die a merge short, does not lose the snake. It reaches 16384 and then never
rebuilds the second one — a ~100-move strategic task, far beyond any search
horizon, so it can only come from the value function.

### This killed the endgame tablebase on two independent grounds

The tablebase was originally abandoned because the agent was too weak to reach
the affordable tables (`docs/phase2-endgame-tablebase.md`: *"our strongest agent reaches 4096 —
every lookup would miss"*). That premise **had** become obsolete, which is why
it was worth re-examining. But the same autopsy measured the coverage gate:

**Only 1% of late-game moves have >=5 large tiles locked with <=10 free cells** —
the shape a `Formation` covers, against a 20% threshold. And more fundamentally,
a tablebase gives *exact endgame tactics*, which is precisely what is **not**
failing.

Two hours of diagnosis avoided days of building a 2.5 GB table for a problem we
do not have.

---

## E27 — Tile downgrading (relative indexing)  ❌ REJECT, decisively

**Hypothesis.** The agent is excellent at building up to 16384 (97% of games).
Rebuilding beneath a locked 16384 is the same task one scale higher, but the
network indexes raw exponents, so `{16384,8192,4096}` and `{2048,1024,512}`
occupy unrelated table entries and none of that competence transfers. Indexing
relative to the board maximum would share it by construction — reaching the rare
regime **without needing to visit it**.

**Implementation.** `IndexingMode::relative` shifts every exponent so the board
max maps to 15, clamping at 0, with empty cells preserved. Recorded in the
weight header so a file can never be read under the wrong interpretation. Paired
with `--global-features` (indexed by `empty_count x max_exponent`) to restore
the absolute scale normalisation discards.

**Result at 100k games**, otherwise the best configuration:

| Configuration | Score | Max tile |
|---|---:|---|
| baseline (`large` + TC + backward) | **135,043** | 16384 |
| + relative indexing + global features | **49,638** | 8192 |

**A 2.7x collapse.** Not marginal, not budget-dependent — it never even reached
16384.

**Why.** Relative indexing makes every board look the same shape regardless of
scale, so the network loses the ability to tell an early position from a late
one. One 256-entry global feature cannot carry what 83.9M weights just gave up.
It reached the rare regime by destroying the network's knowledge of *which*
regime it was in.

A test caught a wrong assumption of mine along the way: I asserted that under
*absolute* indexing the two scaled boards would share nothing (`value == 0`).
They do share — each tuple expands to 8 dihedral orderings, and some land
entirely on cells empty in both boards, hitting the same all-zero entry. The
property that matters is that absolute does not make them *equal*; the test now
says that instead.

---

## E28 — Endgame-seeded training  ⏸ RUNNING

The remaining candidate, and the only one addressing E26's measured cause.

Training is 1-ply self-play, which reaches 16384 in 54% of games and 32768 in
**0%**. The deployed agent plays at depth 4 and lives at 97%/3%. So the value
function receives almost no updates in the regime that decides every game — and
this is exactly what "just train longer" cannot fix, since 1M -> 2M games gained
4.7% at depth 1 and nothing at depth 4.

**Method.** Collect late-game positions from *deployed* (depth-4) play, then
start half of all training episodes from them. Changes **which** states are
updated rather than how many. `src/learning/position_store.{hpp,cpp}` plus
`--seed-positions` / `--seed-fraction`.

Collected **14,351 positions** (max tile >= 8192, every 25th qualifying board
across 40 depth-4 games, sampled to avoid 358k near-duplicates).

**Design is a clean single-variable test.** Treatment resumes
`n5_large_1M.bin` for 100k games at `alpha=0.1`, `seed=77000000`. The control is
`n9_ext_1M1` — the *same* resume, seed, alpha and game count, run earlier
without seeding — which scored **242,440** at depth 1. The only difference is
where episodes start.

Primary metric is `achievement_rate_32768` at depth 4, not mean score.

---

## E4 — The entire H-series ranking is statistically unestablished  ⚠️ METHODOLOGY

Checked while looking for a trustworthy baseline. **Every** recorded H-series
fixed-depth result is n=10, and the harness's own 95% CIs overlap almost
completely:

| Agent | n | Mean | 95% CI |
|---|---:|---:|---|
| H0 | 10 | 26,769 | [21,796, 31,743] |
| H1 | 10 | 36,142 | [28,584, 43,700] |
| H2 | 10 | 34,910 | [28,855, 40,966] |
| H3 | 10 | 49,982 | [41,121, 58,844] |
| H4 | 10 | 45,493 | [33,288, 57,699] |

H1 vs H2 (36,142 vs 34,910) is nothing — the intervals nest. H3 vs H4 overlap
across most of their range. The published ordering **H3 > H1 > H2 > H0** is
supported only for the extremes; the middle is noise.

This is the same n=10 trap that produced the retracted "depth 2 is best"
claim in E1. It is a property of the game: per-game score is heavy-tailed, so
small samples rank configurations essentially at random.

**Does it block anything?** No. The decision that matters is N1 (~103k) versus
the best hand-crafted evaluator (~50k), and a 2x gap is far outside any of
these intervals. The H-internal ordering is a reporting problem, not a
roadmap blocker.

**Partly resolved.** H3 and H5 were rerun at n=40 on matched seeds (E12): H5
beats H3 by **+90.5%**, p=1.1e-10, winning 36 of 40 seeds. That ordering is now
established. Both n=10 figures were *understates* — H3 49,982 → 57,318, H5
89,450 → 109,213 — which is worth noting on its own: small samples here were
biased low, not merely noisy.

**Still open:** H0/H1/H2/H4 remain at n=10, and their mutual ordering remains
unestablished. It blocks nothing — the decisions that matter are N-series
versus H5, and both are now measured at adequate n — so this is reporting
hygiene rather than a gap in the argument. Any table quoting those four must
carry the n=10 warning.

---

## Memory footprint (measured, not estimated)

The constraint is to keep normal operation well under ~5-6 GB so macOS stays
responsive. Measured with four training jobs running concurrently:

| Process | Resident |
|---|---:|
| `train_ntuple` (plain, 1M run) | ~131 MB |
| `train_ntuple` x2 (optimistic init) | ~131 MB each |
| `train_ntuple` (temporal coherence) | **390 MB** |
| **Total for all training** | **~860 MB** |

System-wide free was 35% with these running, so the learning work is not the
binding constraint. Note the machine already carries ~4.4 GB of swap from
unrelated applications (WebKit 1.9 GB, WindowServer 1.3 GB, editor helpers),
which is why headroom matters more than the absolute figure.

TC costs 3x the weight table (128 MB weights + 257 MB of E/A accumulators).
That is training-only and deliberately not serialised, so a TC-trained network
loads for play at the same 128 MB as any other.

**Consequence for the roadmap:** the 15 GB multi-stage configuration remains
out of reach, but TC at the current network size costs 390 MB and is
comfortably affordable. The stated `42-33` upgrade (320 MB weights, ~960 MB
with TC) also fits.

---

## Scheduling rule for the timed regime

Fixed-depth scores are deterministic given a seed, so those runs are safe to
execute concurrently with training — contention changes wall-clock but not a
single score.

The **timed 250 ms/move regime is not**. Its whole measurement is how much
search fits in a deadline, so a loaded machine silently understates strength.
This project already recorded one wrong conclusion from cross-run timing
comparison (documented in `engine-optimization-notes.md`: an optimization
measured as "2x slower" was actually 2.75x faster once A/B'd inside one
process).

**Rule: no timed benchmark runs while any training job is active.** All timed
measurements — N1 with the E1 fix, and the outstanding H5 side task — are
queued until the machine is quiet, and run back-to-back so they share
conditions.

---

## Queued work

Ordered by value, with the reason each is not done yet.

1. **1M-game run at the best known config** — the agent candidate, and its
   versioned checkpoints are the only way to answer why depth pays at 100k but
   not at 1M. Blocked on picking the optimistic-init value at n>=200.
2. **TC at `alpha=1.0`** — the sharp test E5 should have run first. Known
   failure baseline: plain `alpha=1.0` scores 15,300.
3. **Depth-vs-training-budget curve** — measure depth 1 / 2 / 3 at each
   checkpoint of the 1M run. Directly tests the leading explanation for E3:
   that `V` converges to the value of the *greedy* policy, so deeper search
   moves the agent off the distribution `V` was fitted on.
4. **H5 benchmark** (side task, explicitly not a blocker) — fixed depth 4 and
   timed 250 ms, matched seeds. The timed half waits on a quiet machine.
5. **H0/H1/H2/H4 re-benchmark at n>=150** — E4. Reporting hygiene, blocks
   nothing.
6. **Quantify E7's speedup** — interleaved A/B on a quiet machine. The change
   is provably byte-identical and strictly removes work, so this is
   bookkeeping, not a decision.
7. **Larger `42-33` network** (5x 6-tuples, 320 MB) — the one published
   configuration that fits this machine and has not been tried. +42% in the
   paper.

---

## Reproducibility — verified, not asserted

The 1M temporal-coherence run checkpoints at 100,000 games. That checkpoint was
compared against the **separate, earlier** 100k run — a different invocation
with different `--games` and `--evaluate-every` values:

```
cmp experiments/weights/n2_tc_a1_1M.at100000.bin \
    experiments/weights/n2_tc_a1_100k.bin      -> identical
```

**Byte-for-byte identical.** Training is deterministic given
`(seed, alpha, games-so-far, configuration)`, periodic evaluation genuinely
does not perturb the network (it plays through a `const` reference), and
versioned checkpoints are usable as first-class research artifacts rather than
crash insurance.

This also means the depth-vs-training-budget curve can be measured directly
from one run's checkpoints instead of from N separate runs that would each
carry their own seed noise.

---

## Reproducibility

### Preserved networks

| File | Training | Note |
|---|---|---|
| `n1_default.bin` | 1M, plain, buggy reward | the original N1; never overwritten |
| `n2_tc_a1_100k.bin` | 100k, TC, alpha=1.0 | **strongest agent** (105,573 at depth 2) |
| `n2_opt20k_100k.bin` | 100k, optimistic 20k | E6 |
| `n2_bwd_100k.bin` | 100k, backward replay | E9 |
| `n2_rewardfix_100k.bin` | 100k, plain, fixed reward | E2 |
| `n2_tc_a1_1M.at*.bin` | TC checkpoints | versioned, byte-reproducible |

One file needed correcting: `n1r2_fixedreward_1M.bin` was named for the run it
was *meant* to be, but that run was OOM-killed after its first checkpoint, so
the file held **100k** games of training under a name claiming 1M. Verified
byte-identical to the standalone 100k fixed-reward run and renamed
`n2_rewardfix_100k.bin`. A weight file whose name overstates its training
budget is exactly the kind of thing that silently corrupts a later comparison.
- Invalid pre-fix N1 search runs quarantined in
  `experiments/results/invalid-afterstate-mismatch/` with a README, so
  `summarize_experiment.py` cannot fold them into comparisons.
