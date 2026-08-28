# Profiling baseline

Per the "profile before optimizing" instruction, and to avoid distorting
timed-search measurements with new internal instrumentation (see
`docs/phase1-heuristics.md`), this profiles externally with macOS's
built-in `sample` against the existing `runtime_benchmark` binary — zero
measurement overhead added to the engine itself.

```sh
cmake --build build-release --target runtime_benchmark
./build-release/runtime_benchmark &
sample <pid> 5 -file /tmp/runtime_benchmark_sample.txt
```

## Result (5-second sample, aggregated leaf/self time by function)

| Function | Samples | Share |
|---|---:|---:|
| `Expectimax::chance_value` | 325 | 35% |
| `Expectimax::player_value` | 267 | 29% |
| `StructuralHeuristic::evaluate` | 63 | 7% |
| `extract_baseline_features` | 26 | 3% |
| `BaselineHeuristic::evaluate` | 22 | 2% |
| `Expectimax::cached_value` | 22 | 2% |
| `Expectimax::cache_key` | 17 | 2% |
| `transpose` | 13 | 1% |
| `decode` | 12 | 1% |
| `main_line_score` | 12 | 1% |
| `move` | 12 | 1% |
| allocator (`_szone_free`, `free_large`, `madvise`) | 3 | <1% |

(Percentages are of ~925 total samples in the search/heuristic/board
portion of the run; `runtime_benchmark` also spends time in setup code not
listed here.)

## Reading

- **No surprises, confirms the engine-review finding from `docs/phase1-heuristics.md`
  and `ROADMAP.md`**: compute is dominated by the search recursion itself
  (`chance_value`/`player_value`, ~64% combined) — expected and largely
  unavoidable for expectimax with no move ordering or pruning (see ROADMAP
  #7). Allocator activity is negligible (<1%), confirming there's no hidden
  allocation churn in the hot path worth chasing.
- **Heuristic evaluation cost differs meaningfully by heuristic**, and this
  matters for interpreting future benchmark-matrix results:
  `StructuralHeuristic::evaluate` (7%, plus its own `main_line_score` helper
  at another 1%) costs noticeably more per call than
  `BaselineHeuristic::evaluate` + `extract_baseline_features` combined (5%
  total) — a heuristic that wins at a given fixed depth despite being more
  expensive per node is a stronger result than one that wins partly because
  it's cheaper and therefore reaches slightly deeper in the same wall-clock
  budget. Depth-fixed and time-budgeted comparisons (section 1 of the
  original spec) are exactly the mechanism for telling these apart.
- Transposition-table bookkeeping (`cached_value` + `cache_key`, ~4%) and
  raw board operations (`transpose`/`decode`/`move`, ~3%) are both small
  relative to the search recursion — consistent with the codebase-review
  conclusion that the board representation and move tables are already
  near-optimal (bitboard, zero hot-path allocation, precomputed row tables)
  and not a productive optimization target right now.

## Conclusion

No action item from this pass — the profile confirms rather than
contradicts the "engine is already efficient, don't rewrite it" assessment.
The actual near-term efficiency lever is search-side (move ordering,
better pruning-adjacent techniques — ROADMAP #7), not the board/evaluator
layer.
