# Engine optimization notes

Measured results for engine/runtime optimizations, including the ones that
didn't work. Kept so nobody re-litigates a settled question or repeats a
measurement mistake.

Sources mined for techniques: [nneonneo/2048-ai](https://github.com/nneonneo/2048-ai)
and [ronzil/2048-ai-cpp](https://github.com/ronzil/2048-ai-cpp) (both MIT).

## Methodology warning: never A/B across separate benchmark runs

While this machine had a background experiment consuming ~1 core, sequential
`engine_benchmark` invocations returned **63.8M, then 29–32M, then 14–40M
moves/sec for functionally identical builds**. Comparing "before" and
"after" numbers from separate runs produced a completely wrong conclusion
(a real 2.75x speedup looked like a 2x slowdown), and nearly got a good
optimization reverted.

**Always A/B interleaved inside one process**: run variant A and variant B in
alternating blocks in the same binary, repeat several rounds, compare best-
and median-of-rounds. Ambient load then hits both variants equally. See
`/tmp/vertical_ab.cpp` in the session that added this file for the pattern —
build both implementations, alternate blocks, assert checksums match.

## Adopted: single-transpose vertical moves (XOR-delta column tables)

**Technique** (from ronzil/nneonneo): a vertical move used to transpose the
board, run the horizontal row tables, then transpose back — two transposes.
Instead, precompute for each of the 65,536 possible lines a *delta* in
packed-board layout: `unpack_column(line) ^ unpack_column(moved_line)`.
A vertical move then transposes once (to read columns as rows) and XORs each
column's delta straight into the original board. The XOR form is what makes
in-place column rewriting possible without masking.

**Measured**: 17.04 → 6.19 ns per vertical move, **2.75x faster**, interleaved
A/B, identical checksums. Implemented in `MoveTables::up`/`down`
(`src/core/move_tables.cpp`) and `move_vertical_fast` (`src/core/board.cpp`).

**End-to-end effect is small** — profiling puts `move` + `transpose` at only
~2% of search time, and only half of moves are vertical, so expect ~1%
overall. It was not separately measurable under contention. Verified as a
non-regression instead: per-seed scores, move counts, and max tiles are
bit-identical before and after (H4, depth 4, seeds 20000-20002).

**Correctness guards** the fast path needs, both present:
- Defers to the general path when `exponent_high_bits != 0`; tiles above
  exponent 15 don't fit four-bit tables.
- Defers when a table entry reports `exponent_overflow` (a 15+15 merge).

Covered by `engine_tests`' 25,000-board differential fuzz against an
independent reference `move()` implementation.

## Cost: table memory now near the L2 limit

Adding the two column tables took total move-table memory from **1536 KB to
3584 KB**, against this machine's **4096 KB L2** (`sysctl hw.l2cachesize`).
`sizeof(ColumnMove) == 16`, `sizeof(RowMove) == 12`.

This was initially blamed for the (illusory) slowdown. It is not a slowdown
here, but it **is** a real budget constraint for future work:

- The default transposition table is another ~3 MB (65,536 entries), so the
  search's hot working set already exceeds L2.
- **Any future plan that adds another 65,536-entry table should measure L2
  effects explicitly, not assume the lookup is free.** This applies directly
  to `ROADMAP.md` item 7b (row-table acceleration for H3's main-line score).
  Consider narrower entry types, or folding new features into the existing
  tables rather than adding parallel ones.

## Not adopted

- **Fused weights-in-table scoring.** The reference implementations bake
  heuristic weights into a single `heur_score_table`, so one lookup yields a
  finished score. Faster, but it makes weights compile-time constants. Our
  heuristics keep unweighted per-row feature values and apply weights at
  evaluation time, because runtime-configurable weights are required for the
  weight-optimizer work (`ROADMAP.md` item 4). Revisit only if a frozen
  final policy needs maximum speed.
- **ronzil's random-rollout algorithm.** Not an optimization but a different
  agent: it plays random games to completion and picks the move with the best
  average outcome, using no heuristic at all. Reported ~70% to 4096 and ~1%
  to 8192 — well below our expectimax agents — but it would be a genuinely
  independent baseline. Belongs behind the `Agent` interface as a new agent
  type, not as an evaluator.
- **`CACHE_DEPTH_LIMIT`** (ronzil caches only nodes within 6 plies of the
  root, never deeper ones, to stop cheap deep nodes evicting valuable shallow
  ones). Plausible for us, but our fixed-depth benchmarks run at depth 4, so
  every node is already within that window and the change would be a no-op
  there. Only worth testing alongside the depth-6+ timed runs.
