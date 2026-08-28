# Results

Every agent built for this project, how it was made, and what it scored.

## Best agent

| | |
|---|---|
| **Score** | **345,380** average (n=200) |
| Weights | `experiments/weights/n12_plain_2M.bin` (320 MB) |
| How it plays | looks 4 moves ahead |
| How it was made | 2,000,000 self-play games |
| Reaches 16,384 | 93% of games |
| Reaches 32,768 | 4.5% of games |
| Best single game | 580,784 |

**The old headline of 356,178 was an n=60 measurement and is 3.3% optimistic.**
Re-run on 200 games, the same `n5_large_1M` weights score **344,399**.
`n12_plain_2M` is now listed as best on the strength of a properly powered
depth-1 result (+5.2%, n=10,000); at depth 4 the two are indistinguishable
(+0.3%, p=0.89) and n=200 cannot resolve a difference smaller than 5.5%.

Run it with:

```sh
run_experiment --heuristic N1 --weights experiments/weights/n12_plain_2M.bin \
  --search fixed --depth 4 --probability-cutoff 0.0015 --seeds 30000-30199 --threads 8
```

---

## How to read these numbers

**Scores are averages over many games**, listed as `n=...`. A single game can
score anywhere from 3,000 to 580,000, so averages over a handful of games are
close to meaningless. **This project drew and then retracted four conclusions
from samples of 3–30 games.** Treat anything below n=200 as a pilot, and use
`tools/compare_runs.py` to check whether two runs actually differ.

**"Looks N moves ahead" is chosen when you play, not when you train.** The same
saved network can play at any depth, and it matters a lot — the best network
scores 240,366 looking one move ahead and 345,380 looking four ahead. Never
compare two networks at different depths.

**Use enough games.** At depth 4 this benchmark resolves ~5% at n=200 and only
~9% at n=60. Matched seeds do *not* help: measured per-seed correlation between
two runs is ~0 for score. `--threads` is how you buy sample size — scores are
bit-identical at any worker count. A depth-1 game costs ~9 ms, so depth-1
comparisons should use n=10,000, not the n=200-300 used historically.

**Score is mostly decided by the biggest tile you reach:**

| Biggest tile reached | Average score of those games |
|---|---:|
| 8,192 | 162,320 |
| 16,384 | 355,226 |
| 32,768 | 576,688 |

This is why the tile percentages matter as much as the score.

---

## The four approaches tried, in order

| | Approach | Best score | Outcome |
|---|---|---:|---|
| **1** | Hand-written rules | 109,213 | superseded, but better than expected |
| **2** | Perfect endgame lookup tables | — | abandoned twice |
| **3** | Learning from self-play | **345,380** | **the winner** |
| **4** | Structural changes to break the ceiling | no gain yet | 8 attempts, all ties or worse |

These are genuinely different ways of choosing a move, not refinements of each
other.

---

## Approach 1 — Hand-written rules

A human writes a formula scoring how good a board looks; search picks the move
leading to the best-looking board. Nothing is learned.

| Agent | What it looks at | Score (4 moves ahead) |
|---|---|---:|
| H0 | empty squares, big tile near an edge | 26,769 (n=10) |
| H1 | + tiles in order, similar neighbours, corner preference | 36,141 (n=10) |
| H2 | + a "chain" around the biggest tile | 34,910 (n=10) |
| H3 | + full snake structure | 57,318 (n=40) |
| H4 | another project's formula (nneonneo) | 45,493 (n=10) |
| **H5** | the endgame-table project's formula | **109,213 (n=40)** |

**H5 was the surprise.** It nearly doubles H3 while being **14× faster per
move** (5.0 ms vs 71.8 ms). It arrived as a leftover from approach 2, which was
otherwise abandoned.

**Two honest caveats.** H0–H4 are only 10 games each with heavily overlapping
ranges, so their ordering between themselves is not established — only H5 vs H3
was properly rerun. And this repository used to claim the learned network beat
"the best hand-written evaluator" by 24%; that compared against **H2**, not H5.
Against H5 there is no significant gap at one-move-ahead play.

---

## Approach 2 — Perfect endgame lookup tables

Instead of estimating how good an endgame position is, solve it exactly and look
up the answer.

**What got built and verified:** a working solver, checked against an
independent brute-force solver, plus a disk-based version proven to give
bit-identical answers. Smaller board sizes (2×4, 3×3, 3×4) were solved perfectly.

**Abandoned twice, for different reasons:**

1. **Storage.** The two most useful tables need 250 GB and 1.1 TB. We have
   ~21 GB. The tables that *do* fit (2–2.5 GB) cover 16k/32k endgames the agent
   could not reach at the time.
2. **Relevance.** Once the agent *could* reach them, this was re-examined — and
   measurement killed it again. Only **1% of late-game moves** land in a position
   those tables cover. A second reason was given at the time — "the agent does
   not lose on endgame tactics" — and **that one has since been refuted**: 83.6%
   of games reach 16,384 + 8,192 and fail to convert, which is precisely an
   endgame-precision failure (see approach 4). The 1% coverage argument stands
   on its own; the tactics argument does not.

The lasting benefit was **H5**, the evaluation formula that came with it.

---

## Approach 3 — Learning from self-play

The agent keeps a huge lookup table of tile patterns and learns their values by
playing millions of games against itself. Each pattern's 8 rotations and
reflections share one entry, so every game teaches 8 positions at once.

**Three table sizes were tried:**

| Name | Patterns | Memory when playing |
|---|---|---:|
| `default` | 4 patterns | 128 MB |
| **`large`** | 5 patterns of 6 squares | **320 MB** |
| `xlarge` | 8 patterns of 6 squares | 512 MB |

### Version-by-version progression

Each row is the best measurement for that version. All at one-move-ahead play
unless noted, so the progression is comparable.

| Version | What changed | Score | Reaches 16k |
|---|---|---:|---:|
| `n1_default` | first learned agent, 1M games | 102,861 | 0% |
| `n2_tc_bwd_100k` | **new learning method** + replay trick, 100k games | 111,751 | 0% |
| `n5_large_tcbwd_100k` | **bigger table** (128 → 320 MB), 100k games | 135,043 | 5% |
| `n5_large_1M` | **10× more training** (1M games) | 226,324 | 54% |
| `n5_large_1M` @ 2 ahead | *same network, deeper search* | 306,416 | 86% |
| `n5_large_1M` @ 3 ahead | *same network, deeper search* | 334,030 | 94% |
| **`n5_large_1M` @ 4 ahead** | *same network, deeper search* | **356,178 (n=60)** | **97%** |
| `n5_large_1M` @ 4 ahead | *re-run at n=200* | **344,399** | 95% |
| `n12_plain_2M` @ 1 ahead | 2M games, n=10,000 | 240,366 | 59% |
| **`n12_plain_2M` @ 4 ahead** | *same network, deeper search*, n=200 | **345,380** | 93% |

Note the middle rows are **one network**. Nothing was retrained — searching
deeper took it from 226,324 to 344,399. Every figure above n=200 in this table
is a pilot; the two n=60 entries are 3.3% optimistic.

### What actually moved the number

| Change | Effect |
|---|---:|
| Fixing how the search reads the network *(bug)* | 14,262 → 102,861 |
| **A smarter learning rate rule** (temporal coherence) | **+71%** |
| Replaying each game backwards while learning | +17.9% |
| **Bigger pattern table** (128 → 320 MB) | **+20.8%** |
| **Fixing how dead positions were valued** *(bug)* | **+48%** |
| More training beyond 1M games | **+5.2% per doubling** (was recorded as "no gain" from an n=60 run) |

### The single biggest fix

The search asked the network what a **dead board** was worth instead of using
**zero**. A dead board earns no more points, but the network overestimates
endgames badly and a dead board is full of big tiles — so it valued dying at
around **137,000 points**. The agent was being rewarded for losing.

Same network, same games, before → after:

| Looking ahead | Before | After |
|---|---:|---:|
| 2 moves | 184,096 | **306,417** |
| 3 moves | 219,168 | **334,030** |
| 4 moves | 42,735 | **356,178** |

One-move-ahead play was unaffected, which is why none of the training results
had to be thrown out. The 22 runs that *were* invalidated are kept separately in
`experiments/results/invalid-terminal-bug/` so they can't be mistaken for data.

### Things that were tried and did not work

| Change | Result |
|---|---|
| Even bigger table (512 MB) | **21% worse** — too many patterns, not enough practice |
| Splitting the table by game stage | **19% worse** |
| Starting values optimistically | +27.7% alone, but **15% worse** combined with the new learning rule |
| Training while searching deeply | no gain, 34× the cost |
| Reusing rotations to speed up search | 60% slower |

---

## Approach 4 — Trying to break the ceiling

### Why the agent is stuck

The agent reaches 16,384 in ~95% of games — it has essentially mastered that
level. To score much higher it needs **32,768**, which it manages in 4.5%.

**This section previously said the agent "cannot rebuild underneath" a 16,384,
calling it a ~100-move task beyond any search horizon. That was wrong**, and it
set the roadmap for months: it is why deeper search was deprioritised and why
the endgame tablebase was dropped a second time.

A 160-game autopsy (`scratchpad/autopsy.cpp`) recorded, for every game that
reached 16,384, the **largest second tile it ever held beside it**:

| Largest second tile | Games | Share |
|---|---:|---:|
| 4,096 | 21 | 13.8% |
| **8,192** | **127** | **83.6%** |
| 16,384 (done it) | 4 | 2.6% |

**83.6% of games reach 16,384 + 8,192 — one merge short of a second 16,384.**
The agent finishes the rebuild. It then fails to convert, 97% of the time.

The same fact as a like-for-like comparison:

| Task | Free cells | Succeeds |
|---|---:|---:|
| Build an 8,192 with one cell locked | 15 | 95.5% |
| Build an 8,192 with two cells locked | 14 | ~3% |

Identical ladder to build, one fewer cell to do it in, ~32x worse. That is
**precision on a nearly-full board**, which is what search depth buys — not a
planning horizon problem.

### Attempts so far

| Attempt | Score (4 ahead) | Result |
|---|---:|---|
| Double the training (2M games) | 345,858 | tie |
| Add a whole-board feature | 331,994 | tie |
| Bigger table at full training | 152,990 (1 ahead) | worse |
| Perfect endgame tables | — | only 1% coverage, dropped |
| Treat big-tile boards like small-tile ones | 49,638 | **much worse** |
| Practice from real late-game positions | 350,925 | tie |
| Split table at 16,384, both halves pre-trained | *measuring* | — |
| Add snake-order / corner features | *measuring* | — |
| Learn from what deep search concludes | *building* | — |

**Why "treat big-tile boards like small-tile ones" failed so badly:** it made
every board look identical regardless of scale, so the network lost track of
whether it was early or late in the game. It reached the rare situation by
destroying its knowledge of where it was.

### The pattern behind these ties — and why it is not what it looked like

Six attempts landed at ~350,000, and this was read as "search already covers for
the network's mistakes." **That reading is not supported.** Every one of those
runs was n=60, where this benchmark cannot resolve anything below ~9%, and the
effects being tested were 2-7%. They were not measured equal; they were not
measured.

Re-measured at depth 1 with n=10,000 (~1.1% resolution), against each change's
own matched control:

| Change | vs matched control |
|---|---:|
| More training (1M -> 2M) | **+5.2%** (real) |
| Structural features | +0.5% (tie) |
| Distillation | -1.7% |
| Endgame seeding | -1.8% |
| Global features | **-9.2%** |
| `xlarge` 512 MB | **-34.0%** |

**Only more training has ever worked.** Every feature idea is a tie or a loss
once compared against a control that got the same extra games.

Related: **two improvements that fix the same weakness are substitutes, not
additive.** Measured three times — optimistic starting values plus the new
learning rule was 15% *worse* than the learning rule alone.

---

## Honest assessment

**356,178 looks close to the limit of this design on this hardware.** Eight
independent attempts since the last real gain have all tied or regressed.

The one structural idea not yet tested is replacing the lookup table with a
small neural network. A lookup table over fixed groups of squares physically
cannot express "the big tiles are in descending order along an edge" — the idea
that decides these games, and the reason a hand-written formula reaches 109,213
with no learning at all.

For context, published results for this kind of agent are ~234,000 and ~324,000
at one-move-ahead play, the latter needing configurations around 15 GB. At
356,178 in 320 MB, this agent is competitive with published work.

---

## Where things live

| Path | Contents |
|---|---|
| `RESULTS.md` | this file |
| `docs/experiment-log.md` | full chronological log, including retracted conclusions |
| `docs/phase1-heuristics.md` | hand-written rules, search semantics |
| `docs/phase2-endgame-tablebase.md` | solver design, why it was dropped |
| `docs/phase3-td-learning.md` | the learning method and its options |
| `experiments/results/phase1-heuristics/` | hand-written agent runs |
| `experiments/results/phase3-learning/` | learned agent runs |
| `experiments/results/invalid-*/` | runs invalidated by bugs, with explanations |
| `experiments/weights/` | trained networks (excluded from git — large) |
| `tools/compare_runs.py` | tests whether two runs really differ |
