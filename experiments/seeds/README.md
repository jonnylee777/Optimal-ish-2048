# Benchmark seed sets

Defined in [`../../src/experiments/seed_sets.hpp`](../../src/experiments/seed_sets.hpp).
Distinct purpose from that file's `training`/`validation`/`final_test` sets,
which exist to keep weight optimization from overfitting to its own
selection seeds — these three are for comparing heuristics/search configs
against each other on identical seeds:

| Name | `--seeds` value | Seeds | Count |
|---|---|---|---|
| Quick | `quick` | 20000-20099 | 100 |
| Standard | `standard` | 30000-30499 | 500 |
| Final | `final` | 40000-41999 | 2000 |

All disjoint from each other and from training (1000-1999) / validation
(10000-10499) / final_test (50000-50999).

A custom range is also accepted directly as `FIRST-LAST` (e.g. `20500-20502`
for a 3-game sanity check) — useful for expensive configurations where even
the `quick` set isn't affordable. See "Depth sizing guidance" in
[`../../docs/phase1-heuristics.md`](../../docs/phase1-heuristics.md) for
when to reach for a custom range instead of a named set: it's not just
fixed-depth-8 that's expensive — timed/adaptive runs are expensive in a
different way (cost scales with total moves played across a game, which can
run into the thousands, not with search depth), so a 250ms-budget adaptive
comparison over 100 games can still take a very long time even though no
individual search node is slow. Reach for a small custom seed range (3-10
games) for a first adaptive-mode pilot, the same way the legacy `v1`-`v2.1`
comparisons did, and reserve `standard`/`final` for configurations you've
already clocked as affordable at scale.

Determinism: per-game seeds are `first_seed + game_index`, environment/tile
spawns and agent decisions use two independent `mt19937_64` streams derived
from that one seed (`decision_seed()` in
[`../../src/experiments/game_runner.cpp`](../../src/experiments/game_runner.cpp)) —
unchanged from the legacy engine, so any run here reproduces exactly given
the same seed set and configuration.
