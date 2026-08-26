# N-tuple network + temporal difference learning (the N-series)

The first *learned* evaluator in this project. H0-H5 are all hand-crafted —
our own features, or a transcription of someone else's tuned constants. N1's
weights come from self-play with **no human 2048 knowledge encoded at all**.

Sources: [Szubert & Jaśkowski 2014](http://www.cs.put.poznan.pl/mszubert/pub/szubert2014cig.pdf),
[Jaśkowski 2016 (arXiv:1604.05085)](https://arxiv.org/abs/1604.05085),
runtime [aszczepanski/2048](https://github.com/aszczepanski/2048),
trainer [wjaskowski/mastering-2048](https://github.com/wjaskowski/mastering-2048).

## Why this approach and not the endgame tablebase

Both are stronger than anything hand-crafted. Only one fits this machine
(Apple M1, 8 GB RAM, ~31 GB free disk):

| Approach | Needs | Verdict |
|---|---|---|
| Endgame tablebase, its two workhorse tables (81% of its moves) | 250 GB + 1.1 TB **disk** | hard wall |
| N-tuple at the authors' best config | 32-50 GB **RAM** to train | hard wall |
| **N-tuple, lower rungs of the same ladder** | **0.1-1 GB** | **fits** |

The tablebase wall is absolute — the finished tables simply don't fit. The
n-tuple wall is only at the *top* of a ladder whose lower rungs already beat
us. See [`tablebase.md`](tablebase.md) for the tablebase analysis.

Published progression, memory recomputed for float32 weights:

| Configuration | Weights | RAM | Score (1-ply) | Fits? |
|---|---:|---:|---:|---|
| 2014 small: 17 4-tuples | 0.86M | 3 MB | 51,320 | yes |
| **2014 large — our N1** | **33.7M** | **128 MB** | **99,916** | **yes** |
| 2016 `42-33`: 5x 6-tuple | 83.9M | 320 MB | 141,456 | yes |
| + TC learning (3 networks) | 83.9M x3 | 960 MB | 250,393 | yes |
| + 2^4 stages & weight promotion | 1.34B x3 | 15 GB | 267,544 | no |

## What an n-tuple network is

A **tuple** is a set of board cells plus one lookup table of size `16^n`,
indexed by the tile exponents in those cells. The network's value is a plain
sum of one table read per tuple per symmetry:

```
V(board) = sum over tuples i, over orderings j:  lut_i[ index(board, cells_ij) ]
```

`index` packs `n` nibbles most-significant-first, matching the reference's
`address = address * 16 + value`.

**Symmetry is weight sharing, not board rotation.** Each tuple's 8 dihedral
cell-orderings all index the *same* table, so the board is never transformed.
This gives 8x generalization for free — and makes the whole value invariant
under all 8 board symmetries, so `is_rotation_invariant()` is safely `true`.

### N1's shape

| Tuple | Cells (row-major) | LUT size |
|---|---|---|
| straight 4 | `{0,1,2,3}` | 65,536 |
| straight 4 | `{4,5,6,7}` | 65,536 |
| 2x3 rectangle | `{0,1,2,4,5,6}` | 16,777,216 |
| 2x3 rectangle | `{4,5,6,8,9,10}` | 16,777,216 |

4 tables, 8 orderings each → **32 active weights per evaluation**;
**33,685,504 weights = 128 MB** as float32.

## The learning algorithm

**Afterstate TD(0).** The key idea is the *afterstate* `s'`: the board after
sliding and merging but **before** the random tile spawns. It's deterministic
given (state, action), so a move costs one network pass instead of averaging
over the ~30 possible spawn outcomes.

Policy is greedy with **no exploration** — the authors found exploration
actively hurts here, since 2048's own randomness supplies enough:

```
a = argmax over legal a of [ reward(s, a) + V(afterstate(s, a)) ]
```

The update is applied one move late, because the target needs the *next*
afterstate:

```
delta = reward + V(afterstate_next) - V(afterstate_current)
every active weight of afterstate_current += (alpha / m) * delta
```

At game over there is no successor and no further reward, so `delta = -V(s')`.
Dividing by `m` (the active weight count) is what keeps `alpha` O(1).

### A caveat the papers gloss over

The papers state `alpha = 1.0` is the maximum sensible value because it
"immediately reduces the prediction error to zero." That holds **only when
every ordering lands on a distinct table entry**. Different symmetric
orderings *can* collide — commonly on sparse early-game boards, where several
orderings map entirely into the empty region and all index 0. When they
collide, one update overshoots, because the shared weight is both incremented
and read multiple times.

This is measured and pinned in `tests/ntuple_network_tests.cpp`
("sparse boards collide and overshoot"), and it is why `alpha` is a tunable
flag here rather than hardcoded to 1.0.

### Extension-plane clamping

A tuple index carries 4 bits per cell, but our `Board` supports exponents to
31 via `exponent_high_bits`. Exponents at or above 16 are **clamped to 15**
for indexing, so a 65536 tile indexes identically to a 32768 tile. The
reference implementations cap at 32768 and never face this; we must handle it
explicitly or a high-tile board would index entirely wrong weights.

## Training

```sh
cmake --build build-release --target train_ntuple
./build-release/train_ntuple --out experiments/weights/n1.bin \
    --games 1000000 --alpha 0.1 --seed 20260825 \
    --checkpoint-every 50000 --evaluate-every 50000
```

Measured: **1M games took 5.5 hours** single-threaded on an M1. Throughput
starts at ~50,000 games in 75 s and slows substantially as play improves,
because better agents play far longer games — so do not extrapolate the early
rate (an early estimate of "25 minutes" was wrong by ~13x for exactly this
reason). The reference authors used 24 cores with lock-free parallel updates;
matching their training *scale* is harder for us than matching their memory.
Checkpointing is on by default — training is a long background job and must
survive interruption; `--resume PATH` continues from a checkpoint.

Training is **1-ply greedy self-play, no expectimax**, matching the papers.
That also sidesteps a real hazard: the search's transposition table keys on
the board alone, so it would serve stale values if weights changed mid-search.
Weights are frozen before any search-based evaluation.

Periodic evaluation uses seeds disjoint from the training seeds, so the
learning curve measures generalization rather than memorization.

## RESOLVED: N1 with expectimax (was "must not be used with search")

Two distinct bugs made N1 collapse under search. Both are fixed. Full
narrative and statistics in
[`ULTIMATE_AGENT_PROGRESS.md`](ULTIMATE_AGENT_PROGRESS.md); the summary:

### Bug 1 — leaf-semantics mismatch

N1 is an *afterstate* value function, trained by
`V(afterstate) <- reward + V(next afterstate)`, so it answers "expected future
score from this post-move, **pre-spawn** board." But `Expectimax::chance_value`
expanded every empty cell x {2,4} spawn and evaluated the results, so N1 only
ever saw **post-spawn** boards — a question it was never trained to answer.

**Fix.** `EvaluationSemantics` on the `Evaluator` interface
(`src/evaluation/evaluator.hpp`), defaulting to `post_spawn_state`. N1 returns
`afterstate`, and `chance_value` short-circuits:

```cpp
if (depth == 1 && evaluator_.semantics() == EvaluationSemantics::afterstate) {
    return leaf_value(board);   // `board` here IS the afterstate
}
```

Surgical by design: every H0-H5 result is unchanged, and the mechanism is a
capability query rather than a name check, so future learned evaluators
inherit it for free. (An earlier draft of this document proposed
`prefers_afterstate()`; the enum was chosen instead because it names both
sides of the distinction rather than treating post-spawn as the unmarked
default.)

### Bug 2 — reward off-by-one in the TD target

Found while auditing the trainer for bug 1. The update used the reward of the
move that *created* the previous afterstate rather than the reward of the move
*leaving* it, so `V(s')` included a reward already banked before reaching `s'`.
Fixed in `src/learning/td_trainer.cpp`.

### Measured effect

Same weights, only the search path changed (n=300, paired):

| How N1 is evaluated | Before | After |
|---|---:|---:|
| depth 1 (== 1-ply greedy, exactly) | 14,262 | **102,861** |
| depth 4 | 4,228 | — |

And the reward fix is what makes *depth* pay, tested at matched
hyperparameters:

| Weights | Depth 1 | Depth 2 | p |
|---|---:|---:|---:|
| Buggy reward | 102,861 | 96,485 (−6%) | 0.060 |
| Fixed reward | 52,589 | **72,719 (+38%)** | ~0 |

With the fix, `--search fixed --depth 1` is *provably identical* to 1-ply
greedy `max(reward + V(afterstate))` — pinned by
`tests/evaluation_semantics_tests.cpp`, which also proves depth still counts
player decision layers for both semantics.

> **Sample size.** Per-game scores span 5k-175k. n=10 once produced a
> confident, wrong conclusion here. Use n>=200 and
> `tools/compare_runs.py` for any keep/reject decision.

### Known defect: the late game is overvalued about 4x

Measured with a calibration probe that compares `V(afterstate)` against the
score each game actually went on to earn:

| Phase | V predicts | Actually earns |
|---|---:|---:|
| 0-20% | 94,219 | 109,239 |
| 80-100% | **42,969** | **11,214** |

The same +28k to +31k late-game bias appears in every network measured,
regardless of training budget or reward index. The value function does not know
the game is about to end. This is the largest known defect in the current
agent — details and refuted explanations in
[`ULTIMATE_AGENT_PROGRESS.md`](ULTIMATE_AGENT_PROGRESS.md).

## Training options beyond plain TD

Two additions, both opt-in and independent — one changes how big each step is,
the other changes where the weights start. Neither costs anything at play time.

### `--temporal-coherence`

Per-weight adaptive step sizes. Each weight accumulates its signed error sum
`E` and absolute error sum `A`, and its step is scaled by `beta = |E| / A`:
weights whose errors consistently point one way keep a full step, weights whose
errors cancel get damped. See `src/learning/temporal_coherence.hpp`.

Costs two extra float arrays the size of the weight table — measured at
**390 MB resident** versus 131 MB for plain TD. Training-only; not serialised,
so a TC-trained network still loads for play at 128 MB.

**Pair it with a high `alpha`.** TC's purpose is to make an aggressive rate
survivable, so the source papers use `alpha=1.0`. Running TC at `alpha=0.1`
damps an already-conservative rate and measurably loses.

| Trainer | alpha | Score (100k games) |
|---|---:|---:|
| plain TD | 0.1 | 55,640 |
| plain TD | 1.0 | 15,300 (collapses) |
| temporal coherence | 0.1 | 39,854 |
| **temporal coherence** | **1.0** | **95,371** |

**Best known training configuration:**

```sh
train_ntuple --temporal-coherence --alpha 1.0 --backward-updates
```

Compounding is selective, and the pattern is not guessable — measure before
combining (see E13 in
[`ULTIMATE_AGENT_PROGRESS.md`](ULTIMATE_AGENT_PROGRESS.md)):

| On top of TC at `alpha=1.0` | Effect |
|---|---:|
| `--backward-updates` | **+23.5%** (p=1.4e-10) |
| `--optimistic-init 20000` | **−14.9%** |

Optimistic initialisation is worth +27.7% on its own and *costs* 15% here: it
and TC both act on the effective step size, so they collide. Backward replay
changes which targets exist rather than how big the steps are, so it stacks.

### `--optimistic-init X`

Sets every weight to `X / m`, so an unseen board evaluates to about `X`. A
greedy policy is then drawn toward unvisited patterns — exploration without an
epsilon parameter, which matters because the papers found explicit exploration
actively *harmful* in 2048 (the spawn randomness supplies enough).

Measured at 100k games, `alpha=0.1`:

| X | Greedy score |
|---|---:|
| 0 | 55,640 |
| **20,000** | **68,097** |
| 200,000 | 59,230 |

Cannot be combined with `--resume`, which would silently erase the resumed
weights; the trainer rejects that combination rather than doing it quietly.

### `--backward-updates`

Replays each finished episode in reverse instead of updating one move behind
as the game is played. Both orders are TD(0) with the same targets; they differ
only in whether a successor has already been updated when its predecessor reads
it. Measured **+13.1%** at 100k games (62,945 vs 55,640).

It was built to fix the late-game overvaluation described below, and **it does
not** — that hypothesis was measured and refuted. The gain is real; the
mechanism is currently unexplained.

### `--tuples default|large`

Selects the network shape.

| Name | Shape | Weights | Size |
|---|---|---:|---:|
| `default` | 2x 4-tuple + 2x 2x3 rectangle | 33.7M | 128 MB |
| `large` | 5x 6-tuple | 83.9M | 320 MB |

An unknown name is rejected with the valid list rather than falling back to a
default, so a typo cannot silently train a different network than requested.

Weight files embed their tuple definitions, and `NTupleNetwork::load_from()`
rebuilds a network from the file alone — so `run_experiment` plays any trained
shape without being told which.

The `large` cell sets are **our** choice, not a transcription: the published
networks at this size are described by shape rather than explicit cell lists.
Chosen for coverage variety so different tuples fail on different boards.

### Checkpoints are versioned

`--checkpoint-every N` writes `<out>.at<N>.bin`, not one repeatedly-overwritten
file. An interrupted run therefore never destroys its own last good snapshot,
and the intermediate networks are usable research data — they are the only way
to ask how some property changes with training budget, which cannot be
reconstructed once training has moved on.

## Using a trained network

```sh
./build-release/run_experiment --heuristic N1 --weights experiments/weights/n1.bin \
    --search fixed --depth 4 --seeds quick
```

`--weight name=value` is rejected for N1 (its weights are learned, not
declared). Provenance is recorded as the weight-file path plus a content
fingerprint, since 33.7M weights cannot fit in
`ResultMetadata::evaluator_parameters`.

The weight file format is binary with a magic number, format version, and the
full tuple definitions in the header — so a weight file can never be silently
paired with a different network shape. Mismatches, truncation, and trailing
garbage all throw. Saves go to a `.tmp` and are renamed, so an interrupted
save can't be read back as complete (same discipline as
`tablebase::LayerStore`).

## Verification

Two independent gates, following the house style:

- **Index and value vs an independent transcription** — a separate
  reimplementation written from the reference's own formula, cross-checked
  over 3,000 random boards, exact.
- **Rotation invariance empirically** — all 8 dihedral transforms on 500
  boards (4,000 checks), on a network with structure trained into it (an
  all-zero network would pass trivially).
- **The TD update matches the equation** — hand-verified `delta` and
  post-update value. A sign error or a missing `/m` would otherwise just look
  like "training converges slowly."
- **Training actually improves play** — the failure mode all the other tests
  miss. A network can index correctly, round-trip its weights, and apply
  mathematically correct updates while learning nothing.

## Next steps

The highest-value upgrade is **TC learning** (adaptive per-weight learning
rates): the papers measure **+77%** over plain TD for two extra same-shape
networks (~960 MB total, still comfortably in budget). After that, the
`42-33` 5x6-tuple network at 320 MB. Multi-stage weight promotion, redundant
encoding, and carousel shaping are all out of reach — the `2^g` stage
multiplier is what pushes their config to 15 GB.
