# Results index

Each experiment's raw CSV/JSON output lives alongside the Markdown report
generated from it, so every report's relative links resolve without needing
files from another folder. See [../docs/heuristics.md](../docs/heuristics.md)
for what each heuristic version (v1, v1.1, v2, v2.1) actually is; this file
indexes what was measured and where.

## `experiments/` — policy comparisons with a published report

| Folder | Heuristic version(s) | Search mode | Takeaway | Report |
|---|---|---|---|---|
| `milestone4-baseline-depth-sweep` | v1 | Fixed depth 1-5 | Score rises clearly with depth (depth 3 beats depth 2 beats depth 1 on matched seeds); depth 4-5 are single-game runtime probes, not strength estimates. | [milestone4_baseline_comparison.md](experiments/milestone4-baseline-depth-sweep/milestone4_baseline_comparison.md) |
| `milestone5-baseline-optimization` | v1 vs. v1.1 | Fixed depth 1-5 (optimized at depth 3) | Depth-3 random-search optimization (candidate 9) became v1.1; it transfers strongly to depth 2-3 but not reliably to depth 1 — evidence of search-depth specialization. | [milestone5_optimization_comparison.md](experiments/milestone5-baseline-optimization/milestone5_optimization_comparison.md) |
| `milestone5b-baseline-vs-optimized-pilot` | v1 vs. v1.1 | Fixed depth 1-4 (depth 5 pending) | 5-game pilot cross-check of v1 vs. v1.1 on a fresh seed range. | [baseline_vs_optimized_depth1-5_seeds10125-10129.md](experiments/milestone5b-baseline-vs-optimized-pilot/baseline_vs_optimized_depth1-5_seeds10125-10129.md) |
| `v1-optimization-heuristic-sweep` | v1 vs. v1.1 | Fixed depth 1-5 | Larger-sample (up to 100 games/depth) confirmation that v1.1's gains over v1 hold on an independent seed range. | [V1 optimization comparison.md](experiments/v1-optimization-heuristic-sweep/V1%20optimization%20comparison.md) |
| `runtime-optimization` | v1.1 only (engine change, not a heuristic change) | Fixed depth 5 | Transposition-table and packed-board fast-path work made exact depth-5 search 1.8-1.9x faster at identical search values; the "before" data point lives in `milestone5-baseline-optimization/` since it's shared with that report's cross-depth table. | [runtime_optimization_report.md](experiments/runtime-optimization/runtime_optimization_report.md) |
| `milestone6-structural-ablation` | v1.1 vs. three v2 ablation stages | Fixed depth 3 | Each structural feature adds score over v1.1: main-line only +0.6%, +movement penalty +2.9%, +stability/adverse-stuck (full v2) +4.2%. | [Milestone 6 structural heuristic ablation.md](experiments/milestone6-structural-ablation/Milestone%206%20structural%20heuristic%20ablation.md) |
| `v1-v2.1-adaptive-comparison` | v1, v1.1, v2, v2.1 | Fixed depth 4 and adaptive 4/6/8 | 3-game pilot comparing all four canonical versions under both search modes; a pilot, not a robust strength estimate. | [Adaptive V1-V2.1 heuristic comparison.md](experiments/v1-v2.1-adaptive-comparison/Adaptive%20V1-V2.1%20heuristic%20comparison.md), [Adaptive versus fixed depth-4 V1-V2.1 comparison.md](experiments/v1-v2.1-adaptive-comparison/Adaptive%20versus%20fixed%20depth-4%20V1-V2.1%20comparison.md) |

## `optimization-runs/` — raw hyperparameter search artifacts

Not single-policy evaluations; these are the search logs that produced a
frozen weight vector used above.

- `structural-depth4-joint-optimization/` — the exact depth-4 joint
  8-weight search; candidate 6 from this run is `kDepth4OptimizedStructuralWeights`
  (v2.1) in
  [`src/evaluation/structural_heuristic.hpp`](../../src/evaluation/structural_heuristic.hpp).
- `unfiled-probes/` — ad hoc single-game runtime probes and duplicate runs
  (mostly `baseline-optimized` at depth 4/5, seeds 10025-10026) that aren't
  cited by any report above. Kept for the raw data rather than deleted, but
  don't treat them as evidence for anything without checking what they
  actually measure first.

## Filing a new experiment

The CLI (`adversarial_2048`) always writes new CSV/JSON to flat `results/`
(this is a compiled-in default, unchanged by this reorganization). After a
comparison run finishes:

1. Create `results/experiments/<name>/`.
2. Move the run's CSV/JSON files (and the report, if `generate_comparison_report.py`
   already wrote one to flat `results/`) into that folder.
3. Re-run `generate_comparison_report.py` with `--results-dir results/experiments/<name>`
   and `--output "results/experiments/<name>/<Title>.md"` so the report's
   relative JSON links resolve inside the same folder.
4. Add a row to the table above and a version entry in
   [`../docs/heuristics.md`](../docs/heuristics.md) if a new heuristic
   version was introduced.
