# Adversarial 2048

A C++20 research project building the strongest 2048 agent that fits on a
laptop, by measuring progressively different approaches against each other under
one benchmark harness.

**Current best: 356,178 mean score**, reaching 16384 in 97% of games and 32768
in 3%. See **[RESULTS.md](RESULTS.md)** for every version, how it was trained,
and what it scored.

## Four methodologies, in the order they were tried

| Phase | Approach | Best | Documentation |
|---|---|---:|---|
| 1 | Hand-written heuristics + expectimax | 109,213 | [phase1-heuristics.md](docs/phase1-heuristics.md) |
| 2 | Exact endgame tablebase | abandoned | [phase2-endgame-tablebase.md](docs/phase2-endgame-tablebase.md) |
| 3 | Temporal-difference n-tuple learning | **356,178** | [phase3-td-learning.md](docs/phase3-td-learning.md) |
| 4 | Structural / architectural changes | in progress | [experiment-log.md](docs/experiment-log.md) |

Each phase is a genuinely different way of deciding a move, not a tuning of the
last. Phase 1 scores boards by hand-written rules; phase 2 tried to solve the
endgame exactly; phase 3 learns a value function from self-play; phase 4 attacks
the ceiling phase 3 hit.

## Documentation map

| Document | Read it for |
|---|---|
| [RESULTS.md](RESULTS.md) | every agent version, its recipe, and its score |
| [CLAUDE.md](CLAUDE.md) | conventions and invariants — start here before editing |
| [docs/DESIGN.md](docs/DESIGN.md) | architecture, components, control flows |
| [docs/ROADMAP.md](docs/ROADMAP.md) | current state, next steps, open questions |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | what was tried, how, and what it showed |
| [docs/experiment-log.md](docs/experiment-log.md) | full chronological log, including retractions |

**Read [docs/ROADMAP.md](docs/ROADMAP.md) before proposing work.** Ten
approaches have already been measured and rejected; several appealing ideas are
on that list.

## Reading the numbers

Per-game scores span roughly 3,000 to 580,000, so small samples rank
configurations *wrongly*; this project retracted four conclusions drawn from
samples of 3-30. Use `tools/compare_runs.py` for a paired significance test
between two runs, and treat anything below n=200 as a pilot.

The implementation covers the board engine, baseline agents, hand-written
heuristics H0-H5, the learned N-series, and instrumented Expectimax search.

## Requirements

- A C++20 compiler
- CMake 3.20 or newer

## Build and verify

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

Run the optimized core-engine benchmark separately:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target engine_benchmark
./build-release/engine_benchmark
```

## Standardized experiments (`run_experiment`)

This is the current, recommended way to run experiments. It replaces the
ambiguous `v1`/`v1.1`/`v2`/`v2.1` milestone naming used below with
independent flags for heuristic, weights, and search configuration — see
[`docs/phase1-heuristics.md`](docs/phase1-heuristics.md) for what H0/H1
mean and which search capabilities are exposed as ablation switches.

```sh
cmake --build build-release --target run_experiment

# Fixed depth, no time limit — isolates heuristic/search quality.
./build-release/run_experiment --heuristic H1 --search fixed --depth 4 --seeds quick

# Timed, adaptive empty-cell depth schedule (10-16 empty=4, 6-9=6, 0-5=8) —
# realistic play under a per-move compute budget. Use a small custom seed
# range for a first pilot: timed-adaptive cost scales with total moves
# played (often thousands per game), not search depth, so even `quick`
# (100 games) can take a long time. See experiments/seeds/README.md.
./build-release/run_experiment --heuristic H1 --search timed --time-limit-ms 250 --seeds 20500-20502

# Ablate the transposition table at fixed depth 6 on the same seeds.
./build-release/run_experiment --heuristic H1 --search fixed --depth 6 --seeds quick --transposition-table off

# Manual weight override (Phase 1 has no config-file/optimizer integration yet).
./build-release/run_experiment --heuristic H0 --weight empty_cells=120 --search fixed --depth 4 --seeds quick
```

Every run writes CSV/JSON into `--output-dir` (defaults to
`experiments/results/phase1-heuristics/` or `experiments/results/phase1-heuristics/` based on
`--search`) via the same `result_writer` used by the legacy CLI below, with
several additive metrics fields (min score, median moves, full max-tile
distribution, achievement rates up to 65536, a resolved `deadline_hit_rate`
where computable, and a real `git_commit`). See
[`experiments/README.md`](experiments/README.md) and
[`ROADMAP.md`](docs/ROADMAP.md) for what's built and what's next.

## Legacy work

Everything from here through "Baseline weight optimization" below predates
the standardized framework above and uses the `v1`/`v1.1`/`v2`/`v2.1`
milestone naming the new framework replaces. It's kept working and
documented so old results remain reproducible, not as guidance for new
experiments. See [`legacy/results/README.md`](legacy/results/README.md) and
[`legacy/docs/heuristics.md`](legacy/docs/heuristics.md) for that history.
"Board representation" further below is current, general engine
documentation, not legacy-specific.

## Run baseline experiments

The command accepts an agent, number of games, and first environment seed:

```sh
./build-release/adversarial_2048 random 100 1000
./build-release/adversarial_2048 greedy 100 1000
./build-release/adversarial_2048 baseline 20 1000 2
./build-release/adversarial_2048 baseline-optimized 20 10000 3
./build-release/adversarial_2048 baseline-optimized 1 10026 5 0.0001
```

Each game uses the requested environment seed. Agent randomness uses a stable
derived seed, keeping the two random streams independent and making repeated
runs exactly reproducible. `greedy` maximizes immediate merge score and uses a
fixed direction order to break ties.

The optional fourth argument for `baseline` is Expectimax depth. Depth counts
player decision layers. Depth 1 is exactly:

```text
player move -> random spawn -> evaluator
```

Depth 2 adds one more player move and random spawn before evaluation. Search
output includes player/chance/leaf node totals, nodes per second, milliseconds
per move, and transposition-cache hit rate.

The optional final argument is a cumulative path-probability cutoff. Its
default is `0`, which keeps Expectimax exact. A positive cutoff such as
`0.0001` evaluates very unlikely spawn branches early and can make depths 4–5
substantially faster, but it is an approximation and can change move choices.
Always record and compare the cutoff when comparing experiments.

Every run prints a summary and writes permanent CSV/JSON files to `results/`.
CSV contains one row per game. JSON contains experiment metadata, summary
statistics, search instrumentation, evaluator weights, and the same per-game
records.

Combine matched baseline and optimized runs into a Markdown report:

```sh
python3 tools/generate_comparison_report.py \
  --first-seed 10125 \
  --last-seed 10129 \
  --min-depth 1 \
  --max-depth 5 \
  --probability-cutoff 0
```

The generated report is written to
`results/baseline_vs_optimized_depth1-5_seeds10125-10129.md`. The command can
be rerun while an experiment is active: unfinished combinations are marked
pending, and the same report is safely replaced as new result files appear.

For experiments with a different game count at each depth, use
`--games-by-depth` instead of `--last-seed`. For example:

```sh
python3 tools/generate_comparison_report.py \
  --first-seed 10200 \
  --games-by-depth 1:100,2:100,3:100,4:30,5:5 \
  --min-depth 1 \
  --max-depth 5 \
  --probability-cutoff 0 \
  --title "V1 optimization comparison" \
  --output "results/V1 optimization comparison.md"
```

Every game's CSV/JSON lands in flat `results/` regardless of which comparison
it belongs to (this legacy CLI writes there unconditionally — prefer
`run_experiment` above for new work, which writes directly into
`experiments/results/`). If you do use this legacy CLI, move its result
files and report into `legacy/results/experiments/<name>/` and re-run
`generate_comparison_report.py` with `--results-dir`/`--output` pointed at
that folder, so the report's relative links resolve alongside its data,
matching the convention documented in
[legacy/results/README.md](legacy/results/README.md).

## Experimental seed sets

Seed partitions are fixed in code so optimization and evaluation cannot
silently reuse the final-test data:

```text
Training:   1000-1999
Validation: 10000-10499
Final test: 50000-50999
```

The final-test range is reserved until the final standard-2048 comparison.
Milestone 4 baseline depth comparisons use validation seeds only.

## Baseline heuristic

The evaluator is intentionally limited to the four initial features from the
project design:

- Empty cells: number of available cells.
- Monotonicity: negative ordering violations along compacted rows and columns;
  either increasing or decreasing order is accepted per line.
- Smoothness: negative exponent differences between each tile and the next
  occupied tile to its right and below.
- Corner preference: the maximum exponent when a maximum tile occupies any
  corner, otherwise zero.

The initial, unoptimized weights are:

```text
empty cells       270
monotonicity       47
smoothness         15
corner preference 100
```

Feature extraction and weights are separate so future optimization changes
parameters without silently changing feature definitions. See
[legacy/docs/heuristics.md](legacy/docs/heuristics.md) for the full version history
(v1 → v1.1 → v2 → v2.1) and which experiment measured each one.

## Structural heuristic milestone

Milestone 6 preserves the existing policies and adds three explicit ablations:

- `structural-mainline`: optimized baseline plus continuous snake quality.
- `structural-movement`: main-line plus penalties for displacing the anchored
  high-value prefix during player moves.
- `structural-full`: movement ablation plus structural stability and
  adverse-stuck risk at search leaves.

The preferred corner is initially bottom-right. Both horizontal-first and
vertical-first snake paths are supported for every corner; evaluation selects
the stronger orientation for the current board. Frontier diagnostics report
the first deficient path position and desired exponent, but frontier progress
does not yet affect gameplay.

The initial experimental structural weights are:

```text
main line               4
structural stability    1
adverse stuck           2
structural displacement 2
```

These are starting values for ablation, not optimized final weights. All three
policies retain the depth-3 optimized baseline weights for empty cells,
monotonicity, smoothness, and corner preference. See
[legacy/docs/heuristics.md](legacy/docs/heuristics.md) for how these features are defined
and how v2/v2.1 relate to v1/v1.1.

Run the initial exact depth-3 ablation on 100 matched validation games per
policy:

```sh
for agent in baseline-optimized structural-mainline structural-movement structural-full; do
  ./build-release/adversarial_2048 "$agent" 100 10300 3 0

  python3 tools/generate_comparison_report.py \
    --first-seed 10300 \
    --last-seed 10399 \
    --min-depth 3 \
    --max-depth 3 \
    --probability-cutoff 0 \
    --agents baseline-optimized,structural-mainline,structural-movement,structural-full \
    --reference-agent baseline-optimized \
    --title "Milestone 6 structural heuristic ablation" \
    --output "results/Milestone 6 structural heuristic ablation.md"
done
```

The report updates after each policy and is stored at
`results/Milestone 6 structural heuristic ablation.md`; the completed run is
filed at
[legacy/results/experiments/milestone6-structural-ablation/](legacy/results/experiments/milestone6-structural-ablation/).

### Structural weight optimization

Build the Release optimizer first:

```sh
cmake --build build-release --target optimize_structural adversarial_2048
```

For a controlled ablation, optimize the four new structural weights while
keeping the depth-3 optimized V1 baseline weights frozen:

```sh
./build-release/optimize_structural --depth 3 --probability-cutoff 0
```

For the recommended final search, jointly optimize all eight evaluator weights:

```sh
./build-release/optimize_structural \
  --optimize-baseline \
  --candidates 64 \
  --depth 3 \
  --first-training-seed 1200 \
  --probability-cutoff 0
```

The frozen search uses 24 candidates by default. The eight-dimensional joint
search uses 64 candidates by default, retains 16 after 3 games, retains 6 after
8 games, and chooses one winner after 20 matched training games. Candidate
generation and game seeds are deterministic. All four structural terms are
varied in both modes; joint mode also varies empty cells, monotonicity,
smoothness, and corner preference. The original controls remain in the pool so
the search can reject structural terms that do not improve the training score.
Joint candidates are normalized to the initial total weight because a common
scale factor does not change Expectimax decisions; this keeps the search focused
on identifiable weight ratios while still allowing every weight to change.

Timestamped `structural_optimization_depth3_*.csv` and `.json` files are
written to `results/`. The JSON file is the simplest summary of the winning
eight weights and optimizer configuration. The CSV file contains every stage,
candidate, score, runtime, search statistic, and weight vector. File
completed search runs under `results/optimization-runs/<name>/` — see
[legacy/results/optimization-runs/structural-depth4-joint-optimization/](legacy/results/optimization-runs/structural-depth4-joint-optimization/)
for the search that produced v2.1's weights.

Validate the winning JSON vector on unused validation seeds without recompiling:

```sh
./build-release/adversarial_2048 structural-custom 100 10300 3 \
  <empty> <monotonicity> <smoothness> <corner> \
  <main-line> <stability> <adverse-stuck> <displacement> 0
```

Do not feed validation or final-test results back into candidate generation.
Frozen-baseline optimization provides the clean feature ablation; joint
optimization is the better final policy search because the old and new features
can compensate for one another. Compare the frozen training winners once on the
same validation seeds, select the policy there, and leave final-test seeds
untouched for the final unbiased report.

### Adaptive V1–V2.1 comparison

The versioned policies provide a stable comparison vocabulary:

- `v1`: original four-part baseline weights.
- `v1.1`: optimized four-part baseline weights.
- `v2`: V1.1 plus the initial structural weights.
- `v2.1`: the jointly optimized eight-weight depth-4 candidate 6.

All four versioned policies use the same adaptive player-layer schedule:

```text
10-16 empty cells: depth 4
 6-9 empty cells: depth 6
 0-5 empty cells: depth 8
```

Run the matched three-game time-bounded pilot and refresh its report after every
completed policy:

```sh
caffeinate -i zsh -c '
set -e
for agent in v1 v1.1 v2 v2.1; do
  ./build-release/adversarial_2048 "$agent" 3 10400 8 0 250
  python3 tools/generate_comparison_report.py \
    --first-seed 10400 \
    --last-seed 10402 \
    --min-depth 8 \
    --max-depth 8 \
    --probability-cutoff 0 \
    --time-limit-ms 250 \
    --agents v1,v1.1,v2,v2.1 \
    --reference-agent v1 \
    --depth-label "Adaptive 4/6/8" \
    --title "Adaptive V1-V2.1 heuristic comparison" \
    --output "results/Adaptive V1-V2.1 heuristic comparison.md"
done
'
```

The completed pilot is filed at
[legacy/results/experiments/v1-v2.1-adaptive-comparison/](legacy/results/experiments/v1-v2.1-adaptive-comparison/).

The final `250` gives iterative deepening a 250-millisecond budget per move; the
adaptive depth is a ceiling, and the last fully completed depth supplies the
move. The report records games, average score, highest maximum tile, modal
maximum tile, total runtime, and milliseconds per move. JSON also records how
many moves requested depths 4, 6, and 8 and the depths actually completed. All
policies use matched validation seeds. Three games is a pilot rather than a
statistically robust strength estimate; use a
larger unused validation sample for a final selection. A zero time limit
restores full-depth completion and can take days at depth 8.

Run the same four policies at exact fixed depth 4 and update a second report
that combines fixed and adaptive results:

```sh
caffeinate -i zsh -c '
set -e
for agent in v1 v1.1 v2 v2.1; do
  ./build-release/adversarial_2048 "$agent" 3 10400 4 0
  python3 tools/generate_comparison_report.py \
    --first-seed 10400 \
    --last-seed 10402 \
    --depths 4,8 \
    --probability-cutoff 0 \
    --time-limits-by-depth 4:0,8:250 \
    --agents v1,v1.1,v2,v2.1 \
    --reference-agent v1 \
    --depth-labels "4:Fixed depth 4,8:Adaptive 4/6/8" \
    --title "Adaptive versus fixed depth-4 V1-V2.1 comparison" \
    --output "results/Adaptive versus fixed depth-4 V1-V2.1 comparison.md"
done
'
```

The combined report compares heuristic versions within each search mode and
compares fixed versus adaptive search for each version on the same seeds.
Both reports and all their source data are filed together at
[legacy/results/experiments/v1-v2.1-adaptive-comparison/](legacy/results/experiments/v1-v2.1-adaptive-comparison/)
since they share the same depth-8 runs.

## Runtime optimizations

Ordinary boards use precomputed feature scores for all 65,536 possible rows.
Rows from the original and transposed board provide the four-part heuristic
without decoding every cell at every leaf. Extended boards still use the exact
fallback, so tiles above 32768 remain supported.

Empty-cell counting uses bit population counting, and random-spawn children are
constructed directly on the packed board. Expectimax uses a bounded,
generation-tagged transposition table that is reused between moves, avoiding
per-node hash-map allocation and keeping memory bounded. These optimizations do
not alter exact search values.

Run the deterministic runtime benchmark in Release mode:

```sh
cmake --build build-release --target runtime_benchmark
./build-release/runtime_benchmark
```

The benchmark reports evaluator throughput and exact/approximate search data
for depths 3–5. The implementation report and controlled measurements are in
[legacy/results/experiments/runtime-optimization/runtime_optimization_report.md](legacy/results/experiments/runtime-optimization/runtime_optimization_report.md).

## Baseline weight optimization

Run one deterministic multi-stage optimization at the default depth 3:

```sh
./build-release/optimize_baseline
```

Run independent optimizations for depths 1 through 5:

```sh
./build-release/optimize_baseline \
  --min-depth 1 \
  --max-depth 5 \
  --candidates 16 \
  --probability-cutoff 0
```

The optimizer writes separate timestamped CSV and JSON files for each depth to
`results/` as soon as that depth finishes. Files are named
`baseline_optimization_depthN_*.csv` and `.json`. JSON is the easiest place to
find `best_weights`; CSV contains every candidate and stage measurement. The
depth-3 run that produced v1.1 is filed at
[legacy/results/experiments/milestone5-baseline-optimization/](legacy/results/experiments/milestone5-baseline-optimization/).

`--probability-cutoff 0` means exact Expectimax. An exact depth-5 campaign can
take many hours, so run it in a terminal that can remain open. Completed
lower-depth files remain available if a later depth is interrupted. For an
explicitly approximate and faster campaign, use a nonzero cutoff such as
`0.0001`; validate those weights separately from exact results.

Show all optimizer options without starting a run:

```sh
./build-release/optimize_baseline --help
```

Each depth evaluates 16 candidates by default using only training seeds. It
screens all candidates on 3 matched games, retains 6 for 8 games, then retains
3 for a 20-game final training comparison. The
original human weights are always candidate 0. A fixed optimizer seed makes
the candidate set reproducible. CSV records every candidate evaluation; JSON
records the winning weights and complete optimizer configuration.

Evaluate any frozen weight vector without recompiling:

```sh
./build-release/adversarial_2048 baseline-custom 25 10000 3 \
  <empty> <monotonicity> <smoothness> <corner>
```

The selected Milestone 5 vector is compiled as the separate
`baseline-optimized` policy. The original `baseline` remains unchanged for
controlled comparisons.

Optimization uses training seeds only. Choose and cross-test frozen weights on
validation seeds, and leave final-test seeds untouched until the final project
comparison. Depth-4 and depth-5 single games are runtime probes rather than
stable estimates of playing strength.

## Board representation

The common board representation keeps four exponent bits per row-major cell in
one `std::uint64_t`: `0` is empty, `1` is tile 2, `2` is tile 4, and so on.
Moves use precomputed results for all 65,536 possible 16-bit rows. Vertical
moves transpose the board and reuse the horizontal tables.

A compact 16-bit extension plane stores a fifth exponent bit per cell. This
allows exact play through exponent 31 while ordinary boards remain on the
original lookup-table fast path. A 32768 + 32768 merge therefore produces the
correct 65536 tile and score. Boards containing tiles of 65536 or larger use a
clear fallback merge routine because they are outside the four-bit row tables.
