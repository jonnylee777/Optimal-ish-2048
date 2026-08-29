# The 32768 investigation

One place for everything learned about the 32768 ceiling: what the cliff is, every
hypothesis raised against it, how each was tested, and what survived.

`docs/experiment-log.md` is the chronological record (entries E28b-E34 cover this
work). This file is the consolidated view — read it first, then the log entries
for detail.

**Status: the cliff is not cleared.** Five hypotheses tested, four refuted, one
promising but underpowered. The measurement apparatus is now good enough that
further work is worth doing, which was not true when this started.

---

## 1. What the cliff is

Score is set almost entirely by the largest tile reached:

| Largest tile | Mean score of those games |
|---|---:|
| 8,192 | 162,320 |
| 16,384 | 355,226 |
| 32,768 | 576,688 |

The agent reaches 16384 in ~95% of games and 32768 in ~4%. That gap is the
entire ceiling.

Stated as a ladder of conditional probabilities — P(reach next | reached this) —
for `n5_large_1M` at depth 4, n=200:

| | 4096 | 8192 \| 4096 | 16384 \| 8192 | **32768 \| 16384** |
|---|---:|---:|---:|---:|
| depth 1 (n=10,000) | 0.972 | 0.922 | 0.603 | **~0.0001** |
| depth 4 (n=200) | 1.000 | 1.000 | 0.955 | **0.021** |

Every rung is >=0.6 until the last, which drops **45x**. It is a cliff, not a
slope.

### Current best agent

**`n12_plain_2M.bin` at depth 4, cutoff 0.0015: 345,380 mean (n=200), 32768 in
4.5%.** The long-published 356,178 was an n=60 figure and is 3.3% optimistic;
the same `n5_large_1M` weights score 344,399 over 200 games.

---

## 2. Where the failure actually is

**This overturned the project's central diagnosis.** RESULTS.md and E26 held
that the agent reaches 16384 and then cannot rebuild beneath it — "a roughly
100-move task, beyond any search horizon, so it must come from the value
function." That framing is why deeper search was deprioritised and the endgame
tablebase abandoned a second time.

A 160-game autopsy (`n12_plain_2M`, depth 4) recorded, for every game reaching
16384, the **largest second tile ever held beside it**:

| Largest second tile | Games | Share of games reaching 16384 |
|---|---:|---:|
| 4096 | 21 | 13.8% |
| **8192** | **127** | **83.6%** |
| 16384 (converted) | 4 | 2.6% |

**The agent completes the rebuild.** 83.6% reach `16384 + 8192` — one merge from
a second 16384, two from 32768 — and then fail to convert.

### The rung, restated

| Task | Free cells | Success |
|---|---:|---:|
| Build an 8192 with one cell locked | 15 | 95.5% |
| Build an 8192 with two cells locked | 14 | ~3% |

Both need the identical ladder (4096+2048+...+2, twelve cells). Only the room to
manoeuvre differs.

### Why the last rung is so demanding

At its tightest the board must hold

```
16384 + 8192 + 4096 + 2048 + 1024 + 512 + 256 + 128 + 64 + 32 + 16 + 8 + 4 + 2
= 32,766 across 14 of 16 cells — 2 spare
```

a **perfect** descending ladder with no duplicate and no gap, while spawns keep
landing in the two free cells. That is near the geometric limit of a 4x4 board,
and it is a real part of why the rung is hard — a caveat on any claim that the
45x drop is purely a learning failure.

### Two numbers from the autopsy that must not be quoted

- *"Mean free cells at death = 0.00."* A **tautology** — a 2048 game can only end
  on a full board, since an empty cell guarantees a legal move. Measures nothing.
- *"Max tile cornered at death: 31.2%."* Measured after the agent is already
  thrashing, and pooled over games that never reached 16384. Not evidence.

---

## 3. Hypothesis ledger

Every idea tested against the cliff in this investigation.

### H1 — The agent cannot transfer its skill across tile scales ❌ REFUTED

**Claim.** The tuples index raw exponents in 4 bits, so `(13,12,11,10,9,8)` and
`(14,13,12,11,10,9)` — the same shape one scale apart — share exactly zero
weights. The first is visited nearly every game, the second almost never, so the
rebuild skill cannot transfer upward.

**Test** (`scratchpad/scaleprobe.cpp`, ~10 minutes, no training, no games).
2048's dynamics are scale-invariant for a fixed shape, so a network that
generalises should pick the same move on a board with every exponent shifted up
by one. Rank the legal moves by `reward + V(afterstate)`, shift, re-rank.

| Boards with max tile | Shifted up to | Same best move |
|---|---|---:|
| 1024 | 2048 | 79.2% |
| 2048 | 4096 | 73.7% |
| 4096 | 8192 | 77.0% |
| **8192** | **16384 (untrained)** | **76.3%** |

**Result.** Agreement at the boundary into untrained territory is
indistinguishable from boundaries wholly inside trained territory. ~76% is just
how sensitive a move ranking is to relabelling. **No coverage-specific deficit
at 16384.**

**Why the reasoning failed.** The claim about indices is true; the inference is
not. Shifting the whole board moves *every* lookup, and most of the 40 land on
well-trained entries either way. Only the ~15 tuples containing the largest tile
enter novel territory, which is not enough to break the ranking.

**Cost avoided:** an 8-hour matched-pair training run, cancelled.

### H2 — It needs deeper search ❌ REFUTED

**Claim.** Precision on a nearly-full board is what search depth buys, and the
`AdaptiveDepthSchedule` (depth 4/6/8 by empty-cell count) had never been run
with a learned evaluator.

**Test.** Adaptive 4/6/8 vs fixed depth 4, 100 matched seeds, same weights, same
cutoff. (Required lifting a CLI restriction that forced the schedule to be paired
with a wall-clock deadline, which made it neither reproducible nor
parallelisable.)

| | depth 4 | adaptive 4/6/8 |
|---|---:|---:|
| Mean score | 345,380 | 364,431 (+3.8%, p=0.19) |
| **32768 rate** | **4.5%** | **5.0% (p=0.76)** |
| 16384 rate | 93% | 96% (n.s.) |

**Result.** 24x the compute, no movement on the endpoint.

### H3 — It needs broader search (the pruning threshold) ❌ REFUTED

**Claim.** The depth *setting* is not what bounds the tree — the probability
cutoff is. A 2-spawn on an `e`-empty board carries probability `0.9/e`, so a path
of `k` chance layers survives only while `(0.9/e)^k >= cutoff`:

| Empty cells | Player layers the 0.0015 cutoff permits |
|---:|---:|
| 2 | ~8.1 |
| 3 | ~5.4 |
| 5 | ~3.8 |
| 8 | ~3.0 |
| 12 | ~2.5 |

On a 5-empty board, **asking for depth 8 gets about depth 4**. Confirmed by node
count: adaptive 4/6/8 runs ~364k nodes/move, where an unpruned depth-8
expectimax is orders of magnitude larger. `0.0015` appears nowhere in the source
— it entered through a README example and every run since inherited it.

**Test.** Conversion probe at depth 6 / cutoff 0.0002 = 523k nodes/move against
the baseline's 16.5k — **32x more search** — on 224 matched junctures.

| | converts | rate |
|---|---:|---:|
| depth 4, cutoff 0.0015 | 11/224 | 4.91% |
| depth 6, cutoff 0.0002 | 13/224 | 5.80% |

**Result.** p=0.67; the test could have detected 4.9% -> 10.9%. Thirty-two times
more search bought **two extra conversions out of 224**.

The deeper agent does play better — 10% longer survival from the juncture (3,147
vs 2,862 tail moves), ~4% more score. It loses the same way, more slowly.

**What that implies.** Search can only choose among options the evaluator can
distinguish. Better survival with identical conversion is the signature of an
evaluator that cannot tell the deciding positions apart, so more lookahead
explores more branches it also cannot judge. **The constraint is the value
function, not the search.**

### H4 — The evaluator cannot see the whole ladder ❌ REFUTED

**Claim.** The network sees the board through 5 windows of 6 cells. "Is my
14-tile descending ladder unbroken" is a whole-board property no 6-cell window
can express — so the agent may be blind to the one condition that decides the
conversion.

**Test.** `n17_structural` already carries whole-board features (snake-order run
length, corner occupancy, empty count). Probe it on the conversion.

**Result.** 12/224 = 5.36% vs baseline 11/224 = 4.91%. **p=0.83.** The
information can be supplied and it does not help.

### H5 — The conversion is already lost on arrival ❌ REFUTED

**Claim.** If the agent arrives at `16384 + 8192` in a broken configuration, no
endgame skill recovers it — which would also explain why more search changed
nothing.

**Test.** Record each juncture's condition *before* any tail play, paired with
the outcome.

| Arrival feature | Converted | Failed | Difference |
|---|---:|---:|---:|
| Free cells | 6.55 | 6.14 | +0.41 |
| Unbroken ladder run | 1.18 | 1.04 | +0.14 |
| 16384 cornered | 0.82 | 0.65 | +0.17 |
| Distinct tiles | 5.91 | 5.73 | +0.18 |

**Result.** Boards that convert and boards that fail are indistinguishable on
arrival. Stratified breakdowns are not quoted: with 11 successes across a dozen
bins, apparent patterns are chance.

**Weakness:** the ladder-run feature counts from one fixed corner and reads ~1 on
almost every board, so it did not get a fair test.

### H6 — It needs experience in the high-tile regime ⏸ THE SURVIVOR

**Claim.** Training self-play essentially never reaches 32768 — over 10,000
games at 1 ply, the max-tile distribution stops dead at exponent 14 (0 to 1 game
per 10,000 reaches 15). Weights for patterns containing a 32768 are effectively
untrained.

**Evidence for.**

1. **Halving the training halves the conversion.** `n5_large_1M` (1M games)
   converts 6/224 = 2.68% against `n12_plain_2M` (2M games) at 11/224 = 4.91%
   (p=0.22 — directionally right, not conclusive).
2. **Endgame-seeded training converts 3.3x its matched control.** See below.
3. Doubling training is worth **+5.2%** at depth 1 (n=10,000, p<0.001) — the
   only intervention in the project's history that survives proper measurement.

**Evidence against.** None yet, but neither conversion result is individually
significant at the sample sizes used.

---

## 4. The one positive result

**Endgame-seeded training, measured on the metric it was built for.**

E28 built it, E28/E29 rejected it — both on **mean score at depth 1**, where
P(32768) is ~1/10,000 for every network ever trained here. That metric is
structurally incapable of detecting what the intervention changes. E28's own
text names `achievement_rate_32768` at depth 4 as the primary metric. It was
never measured.

224 matched junctures, depth 4 / cutoff 0.0015, network the only variable:

| Network | Converts | Rate |
|---|---:|---:|
| `n17_structural` | 12/224 | 5.36% |
| `n12_plain_2M` (2M games) | 11/224 | 4.91% |
| **`n15_seeded`** | **10/224** | **4.46%** |
| `n5_large_1M` (1M games) | 6/224 | 2.68% |
| `n19_distill` | 6/224 | 2.68% |
| **`n9_ext_1M1`** — seeded's matched control | **3/224** | **1.34%** |

**Seeded vs its own matched control: 4.46% against 1.34%, a 3.3x lift,
p~=0.048.** Same resume, same seed, same alpha, same 100k extra games; seeding
is the only difference. The comparison was named before the run.

### Why this is not yet a result

`n9_ext_1M1` at 1.34% converts *worse* than the weaker network it was built from
(`n5_large_1M`, 2.68%) despite beating it decisively on ordinary play. That gap
is noise (p=0.31) — which is exactly how much three-versus-six events wobble. If
the control's true rate is nearer 2.7%, seeding's lift falls to ~1.6x and stops
being significant. **The finding rests on 3 events against 10.**

Sample needed to settle it, 80% power:

| If the control's true rate is | to resolve 2x | to resolve 1.5x |
|---|---:|---:|
| 1.34% | n~1,720/arm | n~5,750/arm |
| 2.70% | n~840/arm | n~2,810/arm |

**Do not act on this until it is re-run at n>=1,000 per arm.**

---

## 5. Methodology failures found

The reason so many past conclusions were wrong.

**The benchmark could not see what it was asked to judge.** At n=60 the depth-4
benchmark resolves ~9%; the effects tested were 2-7%. Every Phase-4 "tie" sat
inside the noise band. "Eight attempts all tied" was really "eight attempts
produced effects this benchmark cannot resolve" — and the wrong lesson ("search
masks evaluator improvements") was drawn from it.

**Matched seeds do not reduce variance for score.** `compare_runs.py` claimed
pairing "shrinks the standard error several-fold". Measured per-seed correlation
between runs is **r ~ 0** (-0.10 to +0.27 at depth 4). Two agents decorrelate
within a few moves, so "the same seed" is not the same game. Pairing *does* work
for tile-rate endpoints (r ~ +0.48).

**Depth-1 comparisons were run at n=200-300 out of habit.** A depth-1 game costs
~9 ms, so n=10,000 takes 90 seconds and resolves ~1.1%. Re-run properly:

| Network | Score (n=10,000) | vs `n5_large_1M` | vs matched control | Was recorded as |
|---|---:|---:|---:|---|
| `n12_plain_2M` | 240,366 | **+5.2%** | — | "no gain" |
| `n17_structural` | 235,800 | +3.2% | **+0.5% tie** | "worse" |
| `n9_ext_1M1` | 234,548 | +2.6% | — | 242,440 (n=200) |
| `n19_distill` | 230,505 | +0.9% | **-1.7%** | "worse" |
| `n15_seeded` | 230,305 | +0.8% | **-1.8%** | "tie" |
| `n5_large_1M` | 228,532 | — | — | 226,325 |
| `n10_global_ext` | 207,550 | **-9.2%** | — | **"+3.8%"** — sign reversed |
| `n13_xlarge_1M` | 150,778 | **-34.0%** | — | "-32%" — confirmed |

The "+3.8% for global features" came from comparing an n=60 run against a
separate n=200 run. Against a matched control at n=10,000, **every
feature-based intervention is a tie or a loss.** What moved `n17` and `n19` was
the extra training, not the feature.

**Judge an intervention on the metric it targets.** Endgame seeding was built to
raise the 32768 rate and was rejected on mean score at 1 ply. This is the single
most expensive methodological error in the project's history — it discarded the
only intervention that has since shown a positive signal.

---

## 6. Instruments built

| Tool | What it does | Why it exists |
|---|---|---|
| `--threads N` on `run_experiment` | Games across worker threads; scores **bit-identical** at any worker count (GATE test), 3.43x measured | Sample size was the binding constraint on every decision |
| `--threads N` on `train_ntuple` | Hogwild parallel TD; serial path proven **bit-identical** (same weight hash) | Training is the only proven lever and was single-threaded on 8 cores |
| `tools/convert_probe.cpp` | Measures P(second 16384 \| 16384+8192) directly on matched junctures | A depth-8 game runs ~20,000 moves; only the last ~2,800 test the conditional |
| `compare_runs.py --metric tile:N` | Proportion test on tile-rate endpoints, with power reporting | Mean score buries a tile-rate change; "tie" was being reported for unmeasured |
| `--search fixed --adaptive-schedule` | Depth schedule without a wall-clock deadline | The schedule was unusable and unbenchmarkable while coupled to a stopwatch |
| `--relative-bank` | Rank-relative tuple tables, +10.6 MB | Built for H1, which was then refuted. Kept, unused, no measured defect to repair |

### A cost table worth memorising

Search cost scales with **nodes per move**, not the depth number:

| Config | Nodes/move | Probe time (224 trials) |
|---|---:|---:|
| depth 4, cutoff 0.0015 | 16,546 | ~3 min |
| depth 6, cutoff 0.0015 | 102,994 | — |
| depth 6, cutoff 0.0002 | 522,885 | **~15 h** |

The first run of the probe was sized by eye at that last setting and took 15
hours while printing nothing. It now reports every 25 trials and carries this
table in its header.

---

## 7. Where this leaves the mission

**Refuted:** scale transfer, search depth, search breadth, whole-board vision,
arrival state.

**Surviving:** the value function needs experience in the high-tile regime.
Supported by two underpowered results pointing the same way, and by the one
intervention that survives proper measurement anywhere in this project (more
training, +5.2% per doubling).

**Immediate next step.** Re-run `n15_seeded` vs `n9_ext_1M1` on >=1,000 matched
junctures. Collection is the only cost (~1.4 h per 900 junctures); probing is ~3
minutes per arm. If the 3.3x holds, seeding is a training flag rather than a
research programme and applies to the current best network immediately.

**If it does not hold**, the fallback is training scale — the one lever with
unambiguous evidence — now ~4x cheaper via parallel training.

### Honest ceiling estimate

If conversion scales ~1.8x per doubling of training (one comparison, p=0.22 — a
weak basis):

| Training | Conversion | 32768 rate | Mean score |
|---|---:|---:|---:|
| 2M (today) | 4.9% | ~4% | 345,000 |
| 8M | ~16% | ~13% | ~365,000 |
| 16M | ~29% | ~23% | ~377,000 |

A path to ~375-385k. **Not** a path to 500k, which needs conversion above 60%.
No mechanism currently in hand reaches that.

---

## 8. Rules this investigation earned

- **Probe the mechanism on the artefact before spending the machine on it.** H1
  was persuasive, wrong, and killed by a 10-minute probe that saved 8 hours. The
  log already carried this rule; it was broken again anyway.
- **Know what your n can resolve before you run it.** Write down the detectable
  effect first. `compare_runs.py` now returns UNDERPOWERED rather than "tie".
- **A metric that cannot move cannot judge.** P(32768) at depth 1 is ~1/10,000
  for every network ever trained here, so no depth-1 measurement can evaluate an
  intervention aimed at 32768.
- **Cost scales with nodes, not with the depth number.** Size the run first.
- **Commit the instruments.** The previous endgame autopsy lived in a scratch
  directory, was lost, and its central claim ("0/40 died one merge short") turned
  out to be wrong with no way to audit it. `convert_probe` is in the repo for
  this reason.
