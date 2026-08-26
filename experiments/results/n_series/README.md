# N-series results (learned evaluators)

Held separately from `fixed_depth/` and `timed/` because these runs share a
seed set (30000-30299) chosen for **matched-seed comparison between learned
networks**, not the H-series benchmark partitions. Mixing them into the
heuristic comparison tables would compare across different seed populations.

Every run here is `run_experiment --heuristic N1 --weights <file>`, so the
`agent` field says `N1` in all of them. The network that actually played is
identified by the weight path and content hash in
`evaluator.feature_configuration` — read that, not the agent name.

## Results

| Weights | Training | Depth | n | Score |
|---|---|---:|---:|---:|
| `n2_tc_a1_100k.bin` | 100k, TC, alpha=1.0 | **2** | 300 | **105,573** |
| `n1_default.bin` | 1M, plain, alpha=0.1 | 1 | 300 | 102,861 |
| `n1_default.bin` | 1M, plain, alpha=0.1 | 2 | 300 | 96,485 |
| `n2_tc_a1_100k.bin` | 100k, TC, alpha=1.0 | 1 | 300 | 90,508 |
| `n2_opt20k_100k.bin` | 100k, optimistic 20k | 1 | 300 | 67,919 |
| `n1_default.bin` | 1M, plain, alpha=0.1 | 3 | 150 | 65,477 |
| `n2_bwd_100k.bin` | 100k, backward replay | 1 | 300 | 62,716 |
| `n2_rewardfix_100k.bin` | 100k, plain, alpha=0.1 | 1 | 300 | 53,179 |

All 100k rows share `alpha`-appropriate settings, `seed=20260825`, and the
identical evaluation seeds, so they are directly comparable to each other. Use
`tools/compare_runs.py` for paired significance rather than reading the means.

**Canonical weight locations.** Runs recorded before the networks were
promoted out of scratch space carry `/tmp/...` in their
`feature_configuration`. Those paths are stale; the content hash in the same
field is the real identifier, and every network referenced here is preserved in
`experiments/weights/` under the same basename.

Analysis and keep/reject decisions:
[`../../../docs/ULTIMATE_AGENT_PROGRESS.md`](../../../docs/ULTIMATE_AGENT_PROGRESS.md).

## Two things to be careful about

**Sample size.** Per-game scores here span roughly 300 to 178,000. n=10 is
enough to rank configurations *wrongly*; this project did exactly that twice.
Nothing below n=200 should drive a decision.

**Depth is not a free multiplier.** Depth 2 helps the TC network (+16.6%) and
*hurts* `n1_default` (−6%) — the gain shrinks as the evaluator gets stronger.
Re-measure it per network instead of assuming.
