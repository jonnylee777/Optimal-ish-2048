# Endgame tablebase

A tablebase answers a different question from a heuristic. A heuristic scores a
board; a tablebase stores, for every reachable position in a constrained
endgame, the **exact probability of reaching a target tile under optimal play**.
Because it returns per-successor probabilities rather than a board score, it
picks moves directly and cannot be expressed as an `Evaluator` — it belongs
behind the `Agent` interface.

Ported from [game-difficulty/2048EndgameTablebase](https://github.com/game-difficulty/2048EndgameTablebase)
(the strongest published 2048 AI: 65536 tile 5.8% of the time). Code lives in
[`../src/tablebase/`](../src/tablebase/); measured results are in
[`../experiments/summaries/tablebase-variants-solved.md`](../experiments/summaries/tablebase-variants-solved.md).

## The algorithm

States are **post-move, pre-spawn** boards, grouped into **layers by total free
tile sum**. This layering is the load-bearing idea: a spawn adds 2 or 4 and
merges preserve the sum, so the layer index only ever increases. The state graph
is therefore a DAG and can be solved exactly by backward induction, with no
iteration to convergence.

```
P(B) = (1/N_empty) · Σ over empty cells c of
         [ 0.9 · max_d P(move(B + 2@c, d))
         + 0.1 · max_d P(move(B + 4@c, d)) ]
```

- **Player nodes** take `max` over the four directions (we choose).
- **Chance nodes** average over empty cells, weighted 0.9 / 0.1 (the game chooses).
- **P = 1** at success (a target-valued tile in a designated cell). Terminal.
- **P = 0** when no legal in-formation move exists.

Two phases:

1. **Forward generation** — from a seed board, expand each layer into the next
   (spawn 2 → layer+1, spawn 4 → layer+2), keeping only boards that changed and
   still satisfy the formation. Canonicalize, sort, dedupe.
2. **Backward solve** — walk layers high→low applying the recurrence. Only three
   layers are ever needed at once (i, i+1, i+2 — the +2 because a spawned 4
   skips a layer), which is what makes huge tables tractable at all.

**Moves that leave the formation are simply unavailable.** That is what confines
play to the endgame shape, and it is why the probabilities are conditional on
*staying* in the formation.

## Formations

A formation locks some cells to immovable large tiles (nibble `0xF`), leaving the
rest free. A board is "in" the formation if it matches **any** of the masks —
alternatives exist so that e.g. the T formation's large-tile domino can slide
between three positions.

`0xF` doubles as a **wall** in the board-size variants, where it is immovable and
splits each line into independently-compacting segments. `0xE` (16384) is the
largest mergeable tile, since merging two would produce `0xF` and be
indistinguishable from a wall.

Implemented so far: the `2x4`, `3x3`, and `3x4` variants
(`variant_2x4` / `variant_3x3` / `variant_3x4` in `formation.hpp`).

## Symmetry

Storing only the lexicographically smallest board in each symmetry orbit shrinks
a table by up to 8x. Which symmetries are legal is a **per-formation** property:
only those mapping the mask set and success cells onto themselves. `SymmetryMode`
covers `identity`, `full` (all 8 dihedral elements), `diagonal`, `horizontal`,
and the variant-specific `min24` / `min33` / `min34`.

Each mode's element set must be a closed group, or `canonicalize` stops being
idempotent and the table silently stores duplicates that disagree with each
other. This is asserted directly in `tests/tablebase_formation_tests.cpp`, and
it caught a real bug: `rotate_180_34` initially did not flip row 3's columns
while `reverse_lr ∘ reverse_ud34` did, so the `min34` set was not closed.

## Note on bit order

Our packed layout puts cell (r,c) at nibble `r*4+c`, so **(0,0) is the LOW
nibble**. The reference project uses the opposite convention — (0,0) is its HIGH
nibble, and its hex masks read row-major left-to-right. Formations here are
therefore built from explicit `(row, column)` lists rather than by transcribing
its hex constants, because a silent nibble-order slip would be very hard to
detect and would produce plausible-looking wrong tables.

## Hardware reality on this machine

Apple M1, 8 GB RAM, ~31 GB free disk, ARM (no AVX-512). The reference asks for
32–128 GB RAM and 1 TB+ SSD for its flagship tables. Published sizes:

| Table | Size | Feasible here |
|---|---|---|
| `t_512`, `L3_512`, `442_512` (10 free cells) | 2–2.5 GB | yes |
| `444_2048` (12 free) | 130 GB | no |
| `LL_4096` (their #1 table, 40% of moves) | 1.1 TB | no |
| `free10_1024` | 2 TB | no |
| `free12_4096` | — (24 days, 128 GB RAM) | no |

Their own scaling law: *each additional free cell is one order of magnitude;
free tables are ~1000x larger than masked ones; volume doubles as the target
doubles.* Compression only reaches ~41%, so `LL_4096` is still ~450 GB.

### The affordable-vs-reachable bind

Table level is `lvl = (locked large tiles) + log2(target tile)`. Fewer free
cells means more locked tiles means a *later* endgame — so small tables are
intrinsically late-game:

- The affordable 10-free-cell 4x4 tables are 16k/32k endgames.
- Our strongest agent (H3) reaches max tile **4096**. It would never query a
  32k-endgame table — every lookup would miss.
- The tables our agents could actually reach (lvl 12) are 250–800 GB.

This is structural, not incidental, and it sets the build order: a 4x4 tablebase
only pays off once the search half is strong enough to reach the endgame. Hence
Stage E (porting their formation-aware evaluator as H5) rather than starting
there. The variants are unaffected by this bind, which is why they came first —
they are complete games in their own right, solvable end to end.
