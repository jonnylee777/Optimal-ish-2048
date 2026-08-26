# Invalid results — search valued terminal positions at ~137,000

These 22 runs were produced before the bug in E21 was found, and **none of them
measure what their filenames claim**. Kept as evidence of the bug's size and
shape; deliberately held outside `n_series/` so no analysis folds them in.

## The bug

`Expectimax::player_value`, on reaching a position with no legal move, returned
the *evaluator's* opinion of that board instead of **0**:

```cpp
if (!found_move) {
    best = leaf_value(board);   // a dead board is worth 0, not ~137,000
}
```

A dead board is full and covered in large tiles, and the learned network
overestimates the endgame by ~5.9x, so terminal positions scored around
**137,000**. Search was therefore rewarded for reaching them — it steered
*toward* death rather than merely tolerating it.

## Why only these files

The bug is unreachable at depth 1: with afterstate semantics the search never
expands a spawn, so it never evaluates a post-spawn board and never reaches a
terminal node. **Every depth-1 result remains valid**, which is why all the
training-method conclusions in `ULTIMATE_AGENT_PROGRESS.md` survive unchanged.
Only depth >= 2 runs are quarantined here.

## The magnitude

Same weights (`n5_large_1M.bin`), same seeds, before and after the one-line fix:

| Depth | Here (invalid) | Correct |
|---:|---:|---:|
| 2 | 184,096 | **306,417** |
| 3 | 219,168 | **334,030** |
| 4 | 42,735 | **356,178** |

Depth 4 was hit hardest, which is the signature you would predict: deeper search
encounters far more terminal nodes, so it accumulated far more phantom reward.

## What these files are good for

Nothing quantitative. They are useful only as a record of how a plausible,
self-consistent set of measurements can be produced entirely by one wrong line —
and of how convincing the resulting theory looked. See E21 and E22.
