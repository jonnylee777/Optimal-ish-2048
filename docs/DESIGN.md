# Design

How the codebase is put together, and why. Read this before modifying anything;
several decisions look arbitrary until you know what they cost to get wrong.

## Layering

```
core/          bitboard, move tables, spawn, RNG        (no dependencies)
  |
evaluation/    Evaluator interface + H0-H5, N1, ensemble
learning/      n-tuple network, neural network, TD trainer, position store
  |
search/        Expectimax, ParallelExpectimax
  |
agents/        Agent interface, SearchAgent, greedy/random
  |
experiments/   GameRunner, config parsing, result writing
  |
main / run_experiment / train_ntuple / optimize_*
```

`tablebase/` sits apart: a complete, validated exact endgame solver that nothing
currently depends on. See `phase2-endgame-tablebase.md` for why it is unused.

## Core representation

`Board` is a packed bitboard: **cell `(r,c)` occupies nibble `r*4+c`, so (0,0) is
the LOW nibble.** Reference implementations use the opposite order, which is why
formations here are built from explicit `(row, column)` lists rather than
transcribed hex constants — a nibble-order slip produces plausible-looking wrong
answers rather than an obvious failure.

Each cell stores a tile *exponent*, not a tile value (11 means 2048). Exponents
0-15 live in `packed_exponents`; a separate `exponent_high_bits` word extends to
31 for tiles above 32768. The extension is zero on ordinary boards, so the
lookup-table fast path is unaffected.

Moves are precomputed 65536-entry row tables (`move_tables.cpp`). A move is four
table reads plus, for vertical moves, a transpose.

## Evaluation

`Evaluator` (`evaluation/evaluator.hpp`) is the extension point. Three virtuals
carry real weight:

**`semantics()`** declares *what kind of board* the evaluator scores:

| Value | Depth 1 means | Used by |
|---|---|---|
| `post_spawn_state` | move -> all random spawns -> evaluate | H0-H5 |
| `afterstate` | move -> evaluate, no spawn | learned networks |

A 2048 transition is `state -> (move) -> afterstate -> (spawn) -> next state`.
Those are different distributions, so a value function trained on one is
meaningless on the other. Search consults this to decide what a leaf is. It is a
*capability query*, deliberately not a name check, so a new learned evaluator
inherits correct treatment without touching search. Both settings consume
exactly one player decision layer per depth unit, so depth stays comparable.

**`terminal_value()`** is what a position with no legal move is worth. For a
score-predicting network, 0 — the game is over, no further points exist. For a
positional heuristic, a large negative sentinel: their outputs go negative
(H4 reaches -6.3e6), so returning 0 would rank *death* above many bad-but-alive
positions.

**`is_rotation_invariant()`** gates symmetry reduction in search. Declared but
not worth using — measured 60% slower than it saves.

Implementations: H0-H5 are hand-written formulas of increasing sophistication;
`N1Evaluator` wraps an `NTupleNetwork`; `EnsembleEvaluator` averages several
networks.

## Search

`Expectimax` alternates player nodes (maximize over legal moves) and chance
nodes (expectation over spawns, 90% a 2 / 10% a 4, uniform over empty cells).

**Depth counts player decision layers only.** It is chosen at play time and is
not a property of a model.

Three details that are load-bearing:

- **Afterstate short-circuit.** At `depth == 1` with afterstate semantics,
  `chance_value` evaluates the board directly rather than expanding spawns —
  the board *is* the afterstate the evaluator was trained on. This makes
  `--depth 1` provably identical to 1-ply greedy `max(reward + V)`, pinned by a
  test.
- **Terminal handling.** Both interior and root terminal positions return
  `evaluator_.terminal_value()`. Previously they returned `leaf_value()`, which
  valued dead boards at ~137,000 for a learned network and made search actively
  seek death. Fixing it took depth 4 from 42,735 to 356,178.
- **Cutoff fallback.** When the probability cutoff prunes a branch, an
  afterstate evaluator is given the *afterstate it arrived at*, not the spawned
  board — the same distinction as above, on a path only the timed regime
  reaches.

The transposition table keys on `{board, depth, node type, path probability}`
with generation tagging. It is provably neutral on results (verified by an
on/off comparison producing byte-identical scores).

`ParallelExpectimax` runs each root move in its own thread with its own table
(tables are per-instance mutable state, so sharing would race). Verified to
produce identical play. ~1.7x at depth 4 — bounded by the largest subtree, not
by core count.

## Learning

**`NTupleNetwork`** is the workhorse. `V(board)` is a sum of lookups: for each
tuple (a set of board cells), for each of its 8 dihedral orderings, read one
entry from that tuple's table. Symmetry is exploited by **weight sharing** — all
8 orderings index the *same* table — so the board is never rotated and every
update teaches 8 positions at once.

Optional features, all recorded in the weight-file header so a file can never be
read under the wrong interpretation:

| Feature | Effect |
|---|---|
| `stage_count` + `stage_base_exponent` | separate weight sets per game phase |
| `global_features` | one table indexed by `empty_count x max_exponent` |
| `structural_features` | one table indexed by snake-order / cornered / empties |
| `IndexingMode::relative` | index tiles relative to the board max (**measured: much worse**) |

Weight files are self-describing and `load_from()` reconstructs a network from
the file alone. Strict `load()` refuses any mismatch — silent acceptance would
read plausible numbers from wrong entries.

**`NeuralValueNetwork`** is the alternative model class: 16 one-hot cells ->
sum 16 embedding rows -> ReLU -> linear. The input is sparse (16 of 256), so
layer 1 is row additions, not a matrix multiply. ~66k weights against the
table's 83.9M — it trades memorization for the ability to express relations a
table structurally cannot.

**`TemporalCoherenceLearner`** gives each weight its own step size from the
*consistency* of its errors (`beta = |E| / A`). Consistent errors keep a full
step; oscillating ones get damped. This is what makes `alpha=1.0` survivable —
plain TD at that rate collapses. Costs 2 extra floats per weight, so training
uses 3x the model size. Persistable via `--tc-state`, which is what makes long
training incremental rather than all-or-nothing.

## Control flow: choosing one move

```
SearchAgent::select_move(board)
  -> Expectimax::search(board, depth)
       player node:  for each legal move
                       value = reward + chance_value(afterstate, depth)
       chance node:  depth==1 && afterstate ? evaluate(board)
                                            : expectation over spawns
                                                -> player_value(spawned, depth-1)
       terminal:     evaluator.terminal_value()
  -> best direction
```

## Control flow: one training episode

```
for each game:
  board = empty (or a collected late-game position, if seeding)
  loop:
    candidate = argmax over legal moves of (reward + V(afterstate))
    buffer (afterstate, reward)
    board = afterstate; spawn a tile
  replay the episode BACKWARD:
    target_t = reward_t + (1-lambda) V(afterstate_{t+1}) + lambda G_{t+1}
    update V(afterstate_t) toward target_t
```

Backward replay means a successor is already updated when its predecessor reads
it, so the terminal signal travels the whole trajectory in one episode rather
than seeping one state per episode.

Training uses 1-ply greedy self-play, matching published practice, and never
uses the transposition table — it keys on board alone and weights change every
move, so a cached value would be stale.

## Experiment harness

`run_experiment` parses a config, builds an evaluator and a `SearchAgent`, runs
`GameRunner` over a seed range, and writes a JSON + CSV pair per run. The JSON
records full provenance: evaluator identity and content fingerprint, search
parameters, seed partition, and per-tile achievement rates.

Results file by **methodology**, not search regime — both regimes for one agent
belong together, since comparing them is the point.

## Design decisions worth knowing

**Depth is a play-time parameter.** Consequence: every result must be quoted
with its depth, and models are never compared across depths.

**Weight files are self-describing.** Consequence: adding a feature changes the
weight count, so a trained file can no longer `load()`. `adopt_tuple_weights()`
exists so a new feature can be tested without retraining from scratch and
confounding "does the feature help" with "is this budget enough".

**Memory is the binding constraint, not disk or compute.** The machine has 8 GB.
A 320 MB model needs 960 MB to train (temporal coherence triples it). Published
configurations that score higher need ~15 GB. Bigger is also not automatically
better: a 512 MB model scored *worse* at every budget tested, because more
weights need proportionally more games to fill.

**Search compensates for evaluator weakness.** At depth 4, improvements to the
value function show up at depth 1 and vanish at depth 4. This is the single most
important fact for choosing what to work on, and it is why ten separate
improvements have all tied at ~350,000.
