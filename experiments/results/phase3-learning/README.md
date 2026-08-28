# Phase 3/4 — Learned n-tuple networks (N-series)

Every run here is `run_experiment --heuristic N1 --weights <file>`, so the
`agent` field says `N1` in all of them. **The network that actually played is
identified by the weight path and content hash in
`evaluator.feature_configuration`** — read that, not the agent name.

Seed set `30000-30299` is used throughout for matched comparison between
networks. It is deliberately distinct from the H-series benchmark partitions, so
do not compare an N-series score against an H-series score without checking the
seeds.

Some older runs record `/tmp/...` weight paths from before the networks were
promoted out of scratch space; the content hash is the real identifier, and every
referenced network is preserved in `experiments/weights/`.

## Two things to be careful about

**Sample size.** Per-game scores here span roughly 3,000 to 580,000. n=10 is
enough to rank configurations *wrongly*; this project did exactly that twice.
Nothing below n=200 should drive a decision — use `tools/compare_runs.py`.

**Depth is chosen at play time**, not baked into a model. The same weights run at
any depth, and depth changes the answer a great deal (226,324 at depth 1 versus
356,178 at depth 4 for the same file). Always compare at matched depth.

Full results and training recipes: [../../../RESULTS.md](../../../RESULTS.md).
