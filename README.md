# Optimal 2048 AI Agent Experimentation

A C++20 research project that builds the strongest 2048-playing agent viable for a standard 8 GB laptop, by measuring genuinely different approaches against each other
under one benchmark harness — and by treating every measurement as something that
has to earn its own credibility.

---

## Results
<p align="center">
  <img src="docs/figures/best-game.gif" alt="The best agent playing its highest-scoring game, reaching a 32768 tile for 624,164 points" width="330">
</p>

<p align="center">
  <em>624,164 points · 32,768 tile · 21,960 moves · seed 30080</em><br>
  <sub>Sampled across the full game, slowing down as the 32,768 tile is assembled.
  Reproduce with <code>watch_agent --seed 30080</code>.</sub>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/tile-distribution-dark.png">
  <img src="docs/figures/tile-distribution.png" alt="How often the best agent reaches each tile" width="72%">
</picture>

| | Result over 400 games |
|---|---:|
| **Mean score** | **362,341** |
| **Best single game** | **624,164** |
| Reaches 16,384 | 96.25% |
| **Reaches 32,768** | **7.00%** |
| Time per move | 2.7 ms |
| Time per game | 40 s |


## Problem

2048 is a single-player game on a 4×4 grid. You slide tiles; equal tiles merge
and add their value to your score; after every move the game drops a random tile
into a free square. The game ends when no move is legal.

**Objective: maximise the average score over many games.** Score is almost
entirely determined by the largest tile a game reaches, so this is really the
question *how often can the agent build a very large tile* — 16,384, and then
32,768.

The constraint that shapes every decision here is an 8 GB laptop. Published
agents of this kind reach comparable scores using configurations that need around
15 GB.

---

## Training Data

The agent generates its own training data by playing games by itself.

| | |
|---|---|
| **Training data** | 2.5 million self-play games, roughly 20 billion position updates |
| **Stored model** | 83.9 million learned values, 320 MB |
| **Benchmark games** | Fixed random seeds, so any two versions face identical games |
| **Evaluation seeds** | Held separate from training seeds throughout |
| **Target value** | Expected remaining score from a given board |

**The important limitation of this data is that the agent generates it.** Training
games are played one move ahead, and those games essentially never reach a 32,768
tile — 0 to 1 game in 10,000. The positions that decide the agent's ceiling are
therefore almost absent from its own training data, which turns out to be central
to everything in phase five.

---

## Methodology

**Board representation.** Each board is a 64-bit integer, four bits per square
holding the tile's exponent. Moves are precomputed lookup tables over 16-bit rows.

**Search.** Expectimax: alternating layers of the agent's own move and the game's
random tile placement, weighted by probability. Depth counts only the agent's
decision layers. Depth is chosen at play time and is not a property of the model.

**Baseline.** Hand-written evaluation formulas (six versions) scoring properties
such as empty squares, tile ordering, and corner structure.

**Model.** A learned pattern table. Five overlapping windows of six squares each
are laid over the board; every arrangement visible through a window has a stored
value, and the board's value is the sum of the five lookups. Each window's eight
rotations and reflections share one set of values, so every update teaches eight
equivalent positions.

**Training.** Temporal difference learning from self-play, with three refinements
that mattered: per-value adaptive learning rates (temporal coherence), replaying
finished games in reverse, and scoring the board after the agent's move but before
the random tile appears.

**Validation strategy.** Fixed held-out seed sets, sample sizes chosen in advance
from the measurable effect size, and — for any result that was selected from
several candidates — a confirmation run on completely fresh seeds.

---


### How each approach compared

All measured at search depth four, so the comparison is fair:

| Approach | Mean score | Games |
|---|---:|---:|
| Hand-written formula, first version | 26,769 | 10 |
| Hand-written formula, snake structure | 57,318 | 40 |
| Hand-written formula, best version | 109,213 | 40 |
| Learned table, 1 million games | 344,399 | 200 |
| Learned table, 2 million games | 348,960 | 400 |
| **Learned table, 2.5 million games** | **362,341** | **400** |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/progression-dark.png">
  <img src="docs/figures/progression.png" alt="Mean score of every agent version at search depth four" width="100%">
</picture>

**Full version-by-version record, with parameters and runtimes:
[`experiment_results.md`](experiment_results.md).**

### Watching it play

`watch_agent` renders a live game in the terminal with the agent's score, move
count, largest tile and per-move search time. Any trained network or hand-written
evaluator can be loaded, so two versions can be compared side by side in two
terminals.

```sh
./build-release/watch_agent --list                # what networks are available
./build-release/watch_agent                       # watch the best agent
./build-release/watch_agent --from-tile 16384     # sprint to the interesting part
```

A full game at depth four runs about 15,000 moves, so `--from-tile` plays at full
speed until the board reaches the tile you name and only then slows to a
watchable rate — otherwise the first several thousand moves are a formality you
have to sit through. `--step` advances one move per keypress.

---

## Key findings

1. **Search depth is worth more than every training improvement combined.** One
   fixed set of weights scores 226,325 looking one move ahead and 344,399 looking
   four ahead. Nothing was retrained between those numbers.

2. **Training longer eventually makes the agent worse at the thing that matters.**
   Its ability to finish the critical endgame peaks at 2.5 million games and then
   declines steadily, even while its ordinary play keeps improving. The project
   was about to train to 4 million games and would have shipped a *worse* agent
   while every conventional measurement said it was better.

3. **Two of the biggest gains were defect fixes, not ideas.** The search was
   asking the network what a *lost* position was worth instead of using zero; a
   lost board is full of large tiles, which the network overvalues, so it scored
   dying at roughly 137,000 points. Fixing that was worth 48 percent.

4. **Most measurements in this project's history were too small to support their
   conclusions.** Eight ceiling-breaking attempts were filed as ties from 60-game
   runs that could only resolve differences above 9 percent, against effects of 2
   to 7 percent. Re-measured properly, two of them reverse sign.

5. **The standing explanation for the ceiling was wrong.** The agent was believed
   unable to rebuild its structure under a 16,384 tile. It completes that rebuild
   in 84 percent of games and fails one merge from the finish.

---

## Validation

**Sample sizes are chosen from the effect being tested, not by habit.** At depth
four, 60 games resolves only differences above about 9 percent; 400 games reaches
3.9 percent. The comparison tool reports the smallest difference a run could have
detected and refuses to call an underpowered result a tie.

**Parallel execution does not change results.** Games run across all cores produce
bit-for-bit identical scores at any thread count, pinned by a permanent test. The
same holds for the single-threaded training path after parallel training was
added.

**Selected results get a fresh-seed confirmation.** The best agent was chosen as
the strongest of seven checkpoints, which inflates its score on the games it was
chosen on. On the selection seeds it measured +5.5 percent; on fresh seeds,
+2.2 percent; pooled over 400 games, **+3.8 percent**. The reported figure is the
pooled one.

**Timing figures are single-threaded.** Parallel runs leave scores identical but
make per-move timings contended, so no timing measured under load is published.

**Failures are recorded alongside successes.** Six explanations for the 32,768
barrier were tested and refuted, including one that initially looked like a
3.3-times improvement and disappeared under a larger sample.

---

## How to run

```sh
# Build
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j4

# Verify (19 test suites, all must pass)
ctest --test-dir build-release

# Play the best agent
./build-release/run_experiment \
  --heuristic N1 --weights experiments/weights/n25_best_2M5.bin \
  --search fixed --depth 4 --probability-cutoff 0.0015 \
  --seeds 30000-30199 --threads 8

# Watch an agent play, live in the terminal
./build-release/watch_agent --list                    # available networks
./build-release/watch_agent                           # best agent, default game
./build-release/watch_agent --weights experiments/weights/n12_plain_2M.bin
./build-release/watch_agent --heuristic H5 --delay-ms 40
./build-release/watch_agent --from-tile 16384 --delay-ms 150   # skip to the endgame
./build-release/watch_agent --step                    # one move per keypress

# Compare two runs, with a power report
python3 tools/compare_runs.py BASELINE.csv CANDIDATE.csv
python3 tools/compare_runs.py --metric tile:32768 BASELINE.csv CANDIDATE.csv

# Train a network
./build-release/train_ntuple --out weights.bin --tuples large \
  --games 2500000 --alpha 0.1 --temporal-coherence --backward-updates \
  --threads 8
```

Requirements: a C++20 compiler and CMake 3.20 or newer. Python 3 for the analysis
tools. Trained weight files are excluded from version control because of their
size; `experiment_results.md` records the recipe for every one.

---

## Repository structure

| Path | Contents |
|---|---|
| `src/core/` | Board representation, move tables, tile spawning |
| `src/search/` | Expectimax search, including the parallel variant |
| `src/evaluation/` | Hand-written evaluators and the learned-network adapter |
| `src/learning/` | Pattern-table network, temporal difference trainer, temporal coherence |
| `src/tablebase/` | Exact endgame solver from phase two |
| `src/experiments/` | Benchmark harness, result recording |
| `tools/` | Analysis scripts, the conversion probe, and the live viewer |
| `tests/` | 19 standalone suites; those prefixed `GATE:` are load-bearing |
| `experiments/results/` | Every benchmark run, as CSV and JSON with full provenance |
| `experiments/weights/` | Trained networks (excluded from version control) |
| `docs/` | Design, roadmap, and the full experiment log |

| Document | Read it for |
|---|---|
| [`experiment_results.md`](experiment_results.md) | **Every version, its parameters, and its results** |
| [`docs/32768-investigation.md`](docs/32768-investigation.md) | The 32,768 barrier: every hypothesis and outcome |
| [`docs/experiment-log.md`](docs/experiment-log.md) | The chronological laboratory record |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Architecture and control flow |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Current state and open questions |
| [`CLAUDE.md`](CLAUDE.md) | Conventions and invariants — read before editing |

---

## Limitations and future work

**What this project does not solve.** The agent reaches 32,768 in 7 percent of
games. It is not a reliable 32,768 agent, and 65,536 is out of reach — that tile
would require a near-perfect ladder occupying fifteen of sixteen squares.

**The barrier is not understood.** Six explanations have been tested and refuted:
skill transfer across tile scales, search depth, search breadth, whole-board
vision, position quality on arrival, and training from late-game positions. The
only lever that moves it is training volume, and that relationship reverses past
2.5 million games.

**The strongest open lead.** The peak sits 500,000 games after a learning-rate
reset, not at any particular total. If the *reset* is what produces it rather than
the total, then periodic resets are a repeatable lever rather than a one-off —
cheap to test and not yet tested.

**Known measurement gaps.** No agent has been benchmarked under the intended
250 ms-per-move time budget; every result here is fixed-depth. Per-move timing
distributions are not recorded, only aggregates.

**Memory.** Configurations that reach higher scores in published work need roughly
15 GB. Doubling the table to 512 MB was measured here and is 34 percent *worse* at
this training scale — more capacity needs proportionally more training.
