# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this project is

A C++20 research project building the strongest 2048-playing agent that fits on
an 8 GB laptop, by measuring genuinely different approaches against each other
under one benchmark harness.

**Current best agent: 356,178 mean score** — `experiments/weights/n5_large_1M.bin`
(a learned 320 MB n-tuple network) played at fixed search depth 4.

```sh
./build-release/run_experiment --heuristic N1 \
  --weights experiments/weights/n5_large_1M.bin \
  --search fixed --depth 4 --probability-cutoff 0.0015 --seeds 30000-30059
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

- **n >= 200 for any keep/reject decision.** Per-game scores span ~3,000 to
  ~580,000; n=10 ranks configurations *wrongly*.
- **Use `tools/compare_runs.py`** for a paired significance test between two
  runs. Report p-values, not just means.
- **Change one variable at a time.** Varying depth and probability cutoff
  together once produced a retracted conclusion.
- **Match seeds** across compared runs. Seed `30000-30299` is the N-series
  comparison set; H-series uses `20000-...`. Do not compare across sets.
- **Improvements that fix the same weakness are substitutes, not additive.**
  Measured three times. Always test the combination rather than assuming.
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
