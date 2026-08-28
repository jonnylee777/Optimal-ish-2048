# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project is

A C++20 research project building the strongest 2048-playing agent that fits on
an 8 GB laptop, by measuring genuinely different approaches against each other
under one benchmark harness.

**Current best agent: 345,380 mean score (n=200)** —
`experiments/weights/n12_plain_2M.bin` (a learned 320 MB n-tuple network) played
at fixed search depth 4. The long-published 356,178 was an n=60 figure and is
3.3% optimistic.

```sh
./build-release/run_experiment --heuristic N1 \
  --weights experiments/weights/n12_plain_2M.bin \
  --search fixed --depth 4 --probability-cutoff 0.0015 \
  --seeds 30000-30199 --threads 8
```

## Build and test

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j3
ctest --test-dir build-release            # 19 suites, all must pass
```

Builds are warning-clean (`-Wall -Wextra -Wpedantic -Wconversion`). Keep them
that way. Tests are standalone `int main()` programs with a `CHECK` macro — no
framework. Load-bearing tests are prefixed `GATE:`.

Executables: `run_experiment` (benchmark an agent), `train_ntuple` (train a
network), `adversarial_2048`, `optimize_baseline`, `optimize_structural`.

## Invariants that are easy to break

- **Bit order.** Cell `(r,c)` is nibble `r*4+c`, so **(0,0) is the LOW nibble**.
  Reference projects use the opposite convention; never transcribe their hex
  constants directly.
- **Depth counts player decision layers**, not plies of any other kind. Depth is
  chosen at *play* time and is not a property of a model — the same weights
  score 226,324 at depth 1 and 356,178 at depth 4. Never compare two models at
  different depths.
- **Afterstate vs post-spawn.** `Evaluator::semantics()` declares which kind of
  board an evaluator scores. Learned networks score *afterstates* (post-move,
  pre-spawn); hand-written heuristics score post-spawn states. Getting this
  wrong cost a factor of 7 in playing strength.
- **Terminal positions.** A dead board is worth `Evaluator::terminal_value()`,
  never `evaluate()`. For a score-predicting network that is 0; for positional
  heuristics it is a large negative sentinel, because their outputs go negative
  and 0 would rank death above bad-but-alive positions. Getting this wrong cost
  a factor of 1.7.
- **Training must not use the transposition table.** It keys on board alone and
  weights change every move, so cached values go stale.

## Experiment discipline

This project has retracted five conclusions to sloppy measurement. The rules
below are not ceremony.

- **Know what your n can see, before you run it.** At depth 4 this benchmark
  resolves ~9% at n=60, ~5% at n=200, ~3% at n=600. Most ideas here are worth
  2-7%, so n=60 cannot distinguish any of them from a tie — and eight were filed
  as ties on exactly that basis. `compare_runs.py` now prints the effect a run
  was powered to detect and returns UNDERPOWERED rather than "tie".
- **Depth-1 comparisons should use n=10,000, not 200.** A depth-1 game costs
  ~9 ms, so n=10,000 takes ~90 seconds and resolves ~1.1%. Every historical
  depth-1 conclusion was drawn at n=200-300 out of habit; several reversed when
  re-run properly.
- **`--threads N` buys sample size for free.** Scores are bit-identical at any
  worker count (pinned by a GATE test); only per-move timing is contended, and
  the result file records `worker_threads`/`timing_valid`. Measured 3.4x here.
- **Matched seeds do NOT reduce variance for score.** Measured per-seed
  correlation between two runs is ~0 — two agents decorrelate within a few moves,
  so "the same seed" is not the same game. Pairing *does* help for tile-rate
  endpoints (r ~ 0.5); use `--metric tile:32768` for those.
- **Judge an intervention on the metric it targets.** Endgame seeding was built
  to raise the 32768 rate at depth 4 and was rejected on mean score at depth 1,
  where P(32768) is ~1/10,000 for every network ever trained here.
- **Change one variable at a time.** Varying depth and probability cutoff
  together once produced a retracted conclusion.
- **Match seeds** across compared runs. Seed `30000-30299` is the N-series
  comparison set; H-series uses `20000-...`. Do not compare across sets.
- **Improvements that fix the same weakness are substitutes, not additive.**
  Measured three times — though all three at n=60, so treat it as a prior, not a
  result.
- **Probe the mechanism on the artefact before spending the machine on it.**
  Mechanism-based reasoning has been wrong here every time it has been checked.
  The most recent: "the tuples index raw exponents, so a rebuild under a 16384
  shares no weights with one under an 8192, and the skill cannot transfer." True
  about the indices, false about the behaviour — move-ranking agreement across a
  one-scale shift is ~76% at *every* boundary, trained or not. A ten-minute probe
  killed an eight-hour experiment.
- **No timed-regime benchmark while training runs** — contention silently
  understates it.
- Results land in `experiments/results/phase1-heuristics/` or
  `phase3-learning/` automatically, by methodology.

## Coding conventions

- Comments explain *why*, especially where a choice looks arbitrary or where a
  previous approach failed. Several comments record measured outcomes — keep
  them accurate if you change the code.
- Weight files are self-describing (shape, stages, feature flags in the header)
  and refuse to load into a mismatched network. Preserve that.
- Never overwrite `experiments/weights/n5_large_1M.bin` or `n1_default.bin`.
  Version new models; `.tcstate` files are disposable training scratch.
- Prefer extending `Evaluator` over special-casing an agent by name in search.

## Where to look

| Document | Contents |
|---|---|
| `RESULTS.md` | every agent version, training recipe, and score |
| `docs/DESIGN.md` | architecture, components, control flows, design decisions |
| `docs/ROADMAP.md` | current state, next steps, open questions |
| `docs/EXPERIMENTS.md` | experiment index with methodology and conclusions |
| `docs/experiment-log.md` | full chronological log, including retractions |
| `docs/engine-optimization-notes.md` | benchmarking discipline — read before timing anything |

**Read `docs/ROADMAP.md` before proposing work.** Ten approaches have already
been measured and rejected; the log explains why, and several appealing ideas
are on that list.
