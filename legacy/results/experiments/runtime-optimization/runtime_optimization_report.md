# Runtime optimization report

## Scope

This pass optimized the existing four-part baseline heuristic and Expectimax
runtime. It did not adopt the reference document's alternative heuristic.

Implemented exact optimizations:

- Precomputed empty-cell, monotonicity, smoothness, and maximum-tile features
  for all 65,536 packed rows. Columns reuse the same table after transposition.
- Constant-time packed-board empty-cell counting using population count.
- Direct packed-board construction of 2/4 spawn children.
- A 65,536-entry, four-probe, generation-tagged transposition table. It is
  bounded and reused between moves, so searches avoid hash-node allocation and
  full table clearing.

Implemented reference-inspired approximate optimization:

- Optional cumulative path-probability cutoff. The exact default is `0`.
  Branch probability includes both the empty-cell choice and the 0.9/0.1 tile
  probability. Cache keys include path probability whenever cutoff is enabled.

## Controlled exact depth-5 result

Both runs used validation seed 10025, the same optimized four-part weights,
depth 5, and cutoff 0. Both produced score 128,300 in 5,301 moves with an 8192
maximum tile.

| Measurement | Before | After | Improvement |
|---|---:|---:|---:|
| Total runtime | 754.63 s | 418.67 s | 1.80x faster |
| Average runtime | 142.36 ms/move | 78.98 ms/move | 1.80x faster |
| Search throughput | 16.01 M nodes/s | 30.57 M nodes/s | 1.91x higher |

The bounded table recomputed some evicted positions, so total searched nodes
increased from 10.94 billion to 12.80 billion. Higher throughput more than
offset that increase while bounding cache memory to roughly 3 MiB.

Source results:

- Before: `baseline_custom_depth5_seeds10025-10025_20260818T190214211Z.json`
- After: `baseline_optimized_depth5_seeds10025-10025_20260818T192649710Z.json`

## Microbenchmark

The deterministic Release benchmark retained the evaluator checksum
`-12413935672.96` and exact search values. Compared with the initial profile:

- Heuristic throughput rose from about 9.63 M to about 55 M evaluations/s.
- Depth-4 throughput rose from about 17.54 M to roughly 50–55 M nodes/s.
- The final representative depth-5 search processes roughly 55 M nodes/s.

Short microbenchmarks vary between runs; the controlled full-game result above
is the primary runtime measurement.

At cutoff `0.0001`, the representative depth-5 position visits about 0.43
million nodes instead of 6.20 million (about 93% fewer). Its search value also
changes, so cutoff results must not be presented as exact Expectimax results.

## Adaptive-search reference comparison

The reference project's adaptive depth is a maximum, not a promise to finish
that depth. Its default search combines iterative deepening, a 50–100 ms
per-move deadline, and probability cutoff `0.0001`. It returns the last fully
completed iteration. This is the primary reason its game runtime is far lower
than an exact depth-8 projection.

The C++ version now supports the same bounded-search structure while retaining
full-depth search when the time limit is zero:

- iterative deepening with a configurable per-move deadline;
- completed-depth telemetry for every move;
- duplicate legal-move outcome elimination;
- optional eight-way symmetry canonicalization for symmetry-invariant
  evaluators;
- configurable transposition-table capacity.

On the representative benchmark board with a 50 ms budget and cutoff `0.0001`:

| Configuration | Completed depth | Nodes |
|---|---:|---:|
| 64K table, no symmetry | 7 | 1,875,753 |
| 1M table, no symmetry | 6 | 1,582,076 |
| 64K table, symmetry | 6 | 1,471,298 |
| 1M table, symmetry | 6 | 1,247,372 |
| Full structural, 64K | 3 | 224,888 |
| Full structural, 1M | 3 | 213,873 |

The larger cache and symmetry reduction visited fewer nodes but completed less
depth within the deadline because of memory-locality and canonicalization
overhead. They remain available and correctness-tested, but the adaptive
policies use the faster 64K non-symmetry configuration. Symmetry is not valid
for V2/V2.1 because their structural evaluator anchors a preferred corner.

## Where results are stored

Every game writes CSV and JSON files to `results/`. Console output reports the
two exact filenames. JSON includes search depth, probability cutoff, aggregate
node counts, elapsed search time, throughput, evaluator weights, and game
metrics. CSV keeps one row per game and also records the cutoff.

Run exact depth 5:

```sh
./build-release/adversarial_2048 baseline-optimized 1 10025 5
```

Run approximate depth 5 with the reference-style cutoff:

```sh
./build-release/adversarial_2048 baseline-optimized 1 10025 5 0.0001
```

## Verification

- All six Release test suites pass.
- All six AddressSanitizer/UndefinedBehaviorSanitizer test suites pass.
- A randomized equivalence test compares table-based feature extraction with
  the original calculation on 25,000 packed boards and an extended-tile board.
- Cached and uncached cutoff searches return the same direction and value.
- Symmetry-reduced baseline search matches ordinary exact search.
- Bounded iterative deepening returns a legal move from the last completed depth.
