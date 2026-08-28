# Experiments

Standardized experiment output, produced by `run_experiment`
([`../src/run_experiment_main.cpp`](../src/run_experiment_main.cpp)). See
[`../docs/phase1-heuristics.md`](../docs/phase1-heuristics.md) for what
H0/H1/etc. and the search-capability flags mean.

This is a fresh tree, separate from [`../legacy/results/`](../legacy/results/README.md)
(the milestone-named work that predates this framework — kept for reference,
not extended further).

```
experiments/
  seeds/README.md       # the three standardized seed sets (quick/standard/final)
  results/
    fixed_depth/         # --search fixed runs: no time limit, isolates heuristic/search quality
    timed/                # --search timed runs: realistic time-budgeted play
  summaries/              # cross-run comparison reports (tools/summarize_experiment.py)
```

## Filing a run

`run_experiment` writes its CSV/JSON directly into `--output-dir`, which
defaults to `results/phase1-heuristics/` or `results/phase1-heuristics/` based on `--search` —
so, unlike the legacy CLI, there's no separate "move it into place"
step for experiments started with a correct `--output-dir` from the start.
Point `--output-dir` at a more specific subfolder (e.g.
`experiments/results/phase1-heuristics/h0-vs-h1/`) when you want a set of runs
grouped together for a specific comparison; `tools/summarize_experiment.py`
matches files by their JSON metadata (heuristic, search config, seed set),
not by directory location.

## Current status

H0-H4 have been run through this framework so far (see
[`summaries/heuristic-comparison.md`](summaries/heuristic-comparison.md),
regenerate anytime with `python3 tools/summarize_experiment.py`).

Fixed depth 4, 10 games, seeds 20000-20009:

| Heuristic | Mean score | ms/move | Note |
|---|---:|---:|---|
| H0 | 26,769 | 9.7 | simplest floor |
| H1 | 36,142 | 7.2 | baseline |
| H2 | 34,910 | 7.6 | untuned corner-chain weight costs slightly vs. H1 |
| **H3** | **49,982** | 46.7 | snake/main-line structure; best score, most expensive |
| H4 | 45,493 | **4.7** | nneonneo reference port; best score-per-cost by far |

Two findings worth carrying forward:

1. **H3 vs. H4 is a genuine tie on score, and a blowout on cost.** H4 loses
   the paired comparison to H3 by mean score (-4,489) but splits the seeds
   5-5 — with only 10 games that is no detectable score difference. Yet H4
   runs **~10x faster per move** (4.7 vs 46.7 ms). H4 is also the fastest
   heuristic measured, beating even H0 and H1, because all four of its
   features come from a single precomputed 65536-entry row table. Under a
   real per-move time budget H4 should therefore search substantially
   deeper than H3 for the same wall clock — exactly the fixed-depth vs.
   timed distinction the two benchmark regimes exist to separate, and the
   strongest argument yet for finishing the timed comparison.
2. **The comparison isn't apples-to-apples on tuning.** H4 ships the
   reference project's CMA-ES-optimized weights; H2's and H3's are
   hand-picked. So "H4 ≈ H3" partly reflects tuned-vs-untuned, not purely
   feature quality — H3 may have real headroom left once weight
   optimization exists (`ROADMAP.md` item #4).

### Timed benchmark (250 ms/move, adaptive 4/6/8, 3 games, seeds 20500-20502)

| Heuristic | Timed score | ms/move | Avg depth reached | Nodes/sec | Max tile | (Fixed-depth-4 score) |
|---|---:|---:|---:|---:|---|---:|
| H0 | 47,695 | 243.7 | 5.04 | 24.1M | 4,096 | 26,769 |
| H1 | 64,091 | 443.1 | 5.24 | 28.9M | 4,096 | 36,142 |
| **H2** | **84,716** | 247.0 | 4.89 | 16.9M | **8,192** | 34,910 |
| H3 | 74,788 | 249.0 | 4.26 | 2.9M | **8,192** | **49,982** |

**The ranking inverts between the two regimes, which is exactly why both
exist:**

| Regime | Ranking |
|---|---|
| Fixed depth 4 | H3 > H1 > H2 > H0 |
| Timed 250 ms | H2 > H3 > H1 > H0 |

The mechanism is visible in the numbers rather than inferred:

- **H3 has the best evaluation but cannot afford it.** It wins fixed depth by
  a wide margin (49,982, +38% over H1) but its evaluator runs at 2.9M
  nodes/sec — 6x slower than H2 and 10x slower than H4. Under a time budget
  that costs it real depth (4.26 vs H2's 4.89), and it drops to second.
- **H2 has the best strength-per-node.** It loses at fixed depth but wins
  timed by 13% over H3, searching shallower than H0/H1 and staying inside
  budget. Its corner-chain feature is cheap and evidently worth its cost.
- **Only H2 and H3 reach 8192.** H0 and H1 cap at 4096 in both regimes, so
  the structural features (corner chain, main line) are what break that
  ceiling — not extra search.

Two caveats, stated because they cut against the clean story:

- **H1's timed run overran its budget** (443 ms/move against a nominal 250)
  and searched deeper than anything else here. H0/H2/H3 all land at ~245-249
  ms, so this is specific to that one older run under heavier machine load,
  not an engine problem. The effect *flatters* H1, so the gaps above it are
  if anything understated — but a clean H1 re-run would settle it.
- **Three games is a pilot.** Per-seed spread is very wide (H3: 36,760 to
  131,380), so treat the ordering as a strong signal, not a measured effect
  size.

H4 timed is running. It is the cheapest evaluator measured (36M nodes/sec, all
four features from one row table), so it should reach the greatest depth of
any heuristic under budget — the direct test of whether depth or evaluation
quality dominates here.

See [`../ROADMAP.md`](../docs/ROADMAP.md) for what's next (H4+, that per-move
instrumentation, a generic weight optimizer, parallelism, etc.).

## N1: the first learned evaluator (headline result)

Trained by afterstate temporal-difference learning over 1M self-play games
(5.5 hours, single-threaded). See [`../docs/phase3-td-learning.md`](../docs/phase3-td-learning.md).

| Evaluator | Score | n | Search used | Max tile |
|---|---:|---:|---|---|
| **N2 (TC, 200k games)** | **131,481** | 200 | **fixed depth 2** | 8,192 |
| N2 (TC, 200k games) | 117,665 | 200 | 1-ply greedy | 8,192 |
| N2 (TC + backward, 100k) | 111,751 | 300 | 1-ply greedy | 8,192 |
| **H5** | **109,213** [95.3k, 123.1k] | 40 | fixed depth 4 | 8,192 |
| N2 (TC, 100k games) | 105,573 | 300 | fixed depth 2 | 8,192 |
| N1 | 102,861 [98.2k, 107.5k] | 300 | 1-ply greedy | 8,192 |
| H2 | 84,716 | 3 | timed, 250 ms/move | 8,192 |
| H3 | 74,788 | 3 | timed, 250 ms/move | 8,192 |
| H3 | 49,982 | 10 | fixed depth 4 | 4,096 |

> **Sample sizes differ by an order of magnitude.** N1's figure is n=300 with
> a real confidence interval. Every H-series figure is n=3 or n=10, which at
> this game's variance cannot separate configurations that differ by less than
> roughly 2x — see the E4 entry in
> [`../docs/experiment-log.md`](../docs/experiment-log.md).
> The N1-vs-H gap is ~2x and survives; the H-internal ordering does not.
>
> The older N1 figures of 105,472 and 108,946 came from smaller samples of
> this same policy and sit inside the interval above. Nothing changed but n.

**N2 is the strongest agent in this project**, and it was trained on
**100,000** games against N1's 1,000,000 — a tenth of the compute. The
difference is temporal-coherence learning at `alpha=1.0`: a per-weight adaptive
step size that makes an aggressive learning rate survivable, where plain TD at
`alpha=1.0` collapses to 15,300.

Training ablation, all at 100k games with matched seeds (n=300, depth 1):

| Trainer configuration | Score |
|---|---:|
| plain TD, `alpha=0.1` | 53,179 |
| + backward episode replay | 62,716 |
| + optimistic initialisation (X=20,000) | 67,919 |
| **temporal coherence, `alpha=1.0`** | **90,508** |

N1 slightly exceeds the reference paper's own published 99,916 for this network
shape, which is independent evidence the port is correct rather than merely
self-consistent.

> **Correction.** This section previously read "N1 beats the best hand-crafted
> evaluator by 24% while doing no search at all." That compared N1 against
> **H2** (84,716), which is not the best hand-crafted evaluator here. **H5
> scores 109,213** at fixed depth 4 (n=40, 95% CI [95,292, 123,135]) — its
> interval overlaps N1's [98,206, 107,517], and the two runs use different
> seed sets, so N1 and H5 are **not separated by the available evidence**.
>
> The learned approach still wins overall — the temporal-coherence network
> reaches **131,481** — but the margin over hand-crafted work is roughly 20%,
> not 55%, and it comes from the training method rather than from learning
> per se.

Worth stating plainly: N1 encodes **no human 2048 knowledge** — no
monotonicity, no snake heuristic, no corner preference. It learned from
self-play alone. That is a different kind of result from H4/H5, which are
transcriptions of other people's hand-tuned constants.

**N1 + search: the bug is fixed.** N1 used to collapse under search (4,228 at
depth 4) because it is an afterstate value function while the search evaluated
post-spawn states. That is resolved by `EvaluationSemantics` on the
`Evaluator` interface: depth 1 went from 14,262 to 102,861 on the same
weights, and `--search fixed --depth 1` is now provably identical to 1-ply
greedy. The pre-fix runs stay quarantined in
`results/invalid-afterstate-mismatch/` as a record of the bug, not as data.

Depth beyond 1 does **not** currently help this network (96,485 at depth 2,
65,477 at depth 3). That turns out to depend on training budget rather than on
the semantics fix, and is the subject of an open experiment — see
[`../docs/experiment-log.md`](../docs/experiment-log.md).
