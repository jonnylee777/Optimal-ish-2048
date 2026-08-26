# Exact endgame tablebase: 2x4, 3x3 and 3x4 variants solved

Stages A and B of the endgame-tablebase port. These are **exact** results, not
estimates: every reachable position is enumerated and its win probability
computed by backward induction, so the numbers below are the true optimal-play
probabilities (to the stated horizon).

Engine: `src/tablebase/` — `formation.cpp` (masks, success tests, symmetry),
`variant_mover.cpp` (wall-aware moves), `solver.cpp` (layered generation +
backward DP). Ported from
[game-difficulty/2048EndgameTablebase](https://github.com/game-difficulty/2048EndgameTablebase).

## What "variant" means

The reference project notes a full 4x4 tablebase is infeasible but smaller
board variants are solvable. A variant walls off part of the 4x4 grid with
immovable `0xF` tiles; the wall splits each line into independently-compacting
segments. `2x4` keeps rows 1-2 (8 playable cells); `3x3` keeps the top-left
3x3 (9 playable cells).

## Results: probability of reaching each tile under perfect play

Starting from an empty board, spawn rates 90% / 10%.

| Target tile | 2x4 (8 cells) | 3x3 (9 cells) | 3x4 (12 cells) |
|---|---:|---:|---:|
| 32 | 1.000000 | — | 0.999913 |
| 64 | 0.999875 | 0.999450 | 0.997811 |
| 128 | 0.996093 | 0.995372 | 0.989320 |
| 256 | 0.867382 | 0.975537 | — |
| 512 | 0.024575 | 0.716952 | — |
| 1024 | **0.000000** | 0.011144 | — |

(3x4 beyond 128 is left for later: at ~13x growth per target doubling it would
be ~1 billion states and ~15 GB, which fits the disk budget but is a
multi-hour run. Note 3x4's *lower* probability at small targets is not
weakness — more cells means more spawn positions to mismanage while the target
is still trivial; its advantage appears at large targets, which is exactly
where the other two have already collapsed.)

Two things stand out:

1. **Both variants have a hard cliff, and 2x4's is absolute.** On 2x4 the 1024
   tile is *provably unreachable* — not merely unlikely. Eight cells cannot
   hold a 1024 plus the working space needed to assemble it. The 512 tile is
   reachable but only 2.5% of the time.
2. **One extra cell is worth enormously more than it looks.** Going from 8 to
   9 playable cells moves the 512 tile from 2.5% to 71.7%, a ~29x improvement,
   and makes 1024 possible at all. This is the same steep space-sensitivity the
   reference project reports for the full game ("each additional empty space
   increases the volume by one order of magnitude") showing up as *playing
   strength*, not just table size.

## Table sizes

Single-threaded, Apple M1 (8 GB RAM).

| Variant | Target | States | Peak layer | Time | Mode |
|---|---|---:|---:|---:|---|
| 2x4 | 512 | 1,771,763 | 12,030 | 5.2 s | memory |
| 2x4 | 1024 | 3,406,689 | 12,807 | 9.9 s | memory |
| 3x3 | 512 | 6,712,311 | 43,932 | 30.1 s | memory |
| 3x3 | 1024 | 15,816,307 | 53,381 | 72.9 s | memory |
| 3x4 | 64 | 5,966,760 | 701,510 | 49.8 s | memory |
| 3x4 | 128 | **77,064,998** | 4,079,234 | 603 s | **disk** |

### Why the disk-backed path was needed

3x4 at target 128 **could not be completed in memory at all** — it exceeded
available RAM. The disk-backed solver finished it using **~200 MB of RAM and
1.18 GB of disk**.

The reason it works is that the peak layer is far smaller than the total:
4.08M vs 77.06M states, an 19x ratio. Layer L receives successors only from
L-1 (spawn 2) and L-2 (spawn 4), so once L-1 has been expanded, L is complete
— three rolling buffers suffice for generation, and three resident layers
suffice for the backward pass. Everything else lives on disk.

Held in memory the same table would need roughly 77M x 24 bytes ≈ 1.85 GB for
solved layers alone, before generation buffers, which is what made it fail on
an 8 GB machine with other work running.

## Correctness

Two chained gates, so nothing rests on a single implementation agreeing with
itself.

**Gate 1 — layered DP vs. independent brute force.** The layered solver is
cross-checked against a **separate** top-down memoized recursion over raw,
non-canonicalized boards (`brute_force_probability`), sharing no code with it
beyond the formation predicates. They agree to 1e-12 on **every state** in the
table — 3,597 states for 2x4 and 1,917 for 3x3, the latter exercising 8-way
symmetry folding. See `tests/tablebase_solver_tests.cpp`.

**Gate 2 — disk vs. memory.** The disk-backed solver must reproduce the
in-memory solver **bit-for-bit** across every layer, board and probability, on
all three variants, with the buffer-compaction path forced active. Because
gate 1 anchors the in-memory solver to real ground truth, gate 2 inherits that
guarantee. See `tests/tablebase_disk_tests.cpp`.

Additional invariants under test: symmetric boards get identical
probabilities; every stored board in layer L has free tile sum exactly 2L;
walls never move and never appear/disappear; the wall-aware mover agrees
exactly with the engine's own `move()` on 20,000 wall-free boards; harder
targets are never easier.

Two real bugs were caught this way and are worth recording, since both
produced *plausible* wrong answers rather than obvious failures:

- **Merge direction.** Gathering tiles in forward scan order merges the pair
  furthest from the movement direction. `[2,2,4,4]` moving right gave
  `[0,0,8,4]` instead of `[0,0,4,8]` — same tiles, wrong scores.
- **Horizon semantics.** Success is terminal and worth 1.0 at *any* tile sum,
  but the layered solver initially only credited successes that happened to
  fall inside the generated horizon, while the brute force credited all of
  them (a 1.3e-4 disagreement on 3x3). Fixed by crediting success inline
  during the backward pass instead of looking it up. The same change removed
  an undefined-behaviour binary search over never-sorted overflow layers.

## Reproducing

```sh
cmake --build build-release --target tablebase_solver_tests
ctest --test-dir build-release -R tablebase
```
