# Trained networks

Excluded from git (hundreds of MB each). Everything needed to regenerate them is
committed — see `../../experiment_results.md` for each version's training recipe.

## What to keep

| File | Why |
|---|---|
| `n5_large_1M.bin` | **the best agent** — 356,178 at depth 4 |
| `n1_default.bin` | the first learned agent; the original baseline |
| `endgame_positions.bin` | late-game boards, for endgame-focused training |
| `search_targets.bin` | 748k positions + what depth-4 search concluded |

## `.tcstate` files are training scratch, not models

Training keeps two extra numbers per weight to adapt its learning rate, so a
320 MB model has a 640 MB companion. **They are only needed to RESUME that
specific run.** Delete them once a run is finished and judged — they are the
single largest disk consumer in this project (12 of them once occupied 8 GB).

Without one, resuming still works but restarts every learning rate at its
initial value, which can damage a converged network. The trainer prints a
warning when that happens.

## Intermediate checkpoints

`<name>.at<N>.bin` are snapshots every N games. Useful while a run is live (to
measure progress without waiting) and for the depth-vs-training curves in
`../../experiment_results.md`. Safe to delete once the final weights exist and the run has been
benchmarked.
