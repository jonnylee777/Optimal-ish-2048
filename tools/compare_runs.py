#!/usr/bin/env python3
"""Decide whether two experiment runs actually differ.

2048 scores are wildly dispersed -- a single configuration routinely spans
5k to 175k across seeds. That makes eyeballing two mean scores useless at
small n, and it is exactly how this project once concluded "depth 2 is best"
from an n=10 run that a larger sample reversed.

MATCHED SEEDS DO NOT HELP HERE, and this file used to claim they did.
Measured across every pair of committed runs, the per-seed score correlation
between two agents is r ~ 0 (-0.10 to +0.27 at depth 4, +/-0.08 at depth 1,
n=200). The seed fixes the spawn STREAM, but which cell each spawn lands in
depends on the moves played, so two agents decorrelate within a few moves and
"the same seed" is not the same game. The paired test is still applied when
seed sets overlap -- at r=0 it is neither better nor worse than unpaired -- but
it buys no precision, and sample size is the only thing that does.

That matters because it sets what this benchmark can see. At depth 4 the
per-game sd is ~85,000 on a ~356,000 mean, so n=60 cannot resolve anything
smaller than ~9%. Eight ceiling-breaking attempts were filed as "ties" at that
n while sitting between -1% and -7%: unmeasured, not measured equal. Every
non-significant verdict below therefore reports the effect the run was
POWERED to detect, and refuses to call a tie it could not have distinguished
from a real regression.

    tools/compare_runs.py BASELINE.csv CANDIDATE.csv

Exits 0 if the candidate is significantly better, 1 otherwise, so it can gate
a keep/reject decision in a script.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


def read_scores(path: Path) -> dict[int, float]:
    """seed -> score. Duplicate seeds keep the last row, matching a rerun."""
    with path.open(newline="") as handle:
        return {int(row["seed"]): float(row["score"]) for row in csv.DictReader(handle)}


def read_tile_reached(path: Path, exponent: int) -> dict[int, float]:
    """seed -> 1.0 if the game reached 2**exponent, else 0.0.

    The endpoint that actually matters for the 32768 ceiling. Mean score buries
    it: a game reaching 32768 scores ~577k against ~355k for one reaching 16384,
    so a change in the tile rate shows up in the mean only after being diluted by
    every game that did not get there. Testing the proportion directly says what
    happened, and a Bernoulli endpoint needs a different variance estimate than a
    heavy-tailed score does.
    """
    with path.open(newline="") as handle:
        return {int(row["seed"]): float(int(row["max_tile_exponent"]) >= exponent)
                for row in csv.DictReader(handle)}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def correlation(xs: list[float], ys: list[float]) -> float:
    """Per-seed correlation between two runs -- i.e. how much pairing buys."""
    if len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p(t: float) -> float:
    """Normal approximation. Fine at the n>=200 this project requires; it is
    mildly anti-conservative below about n=30, which is flagged in output."""
    return 2.0 * (1.0 - normal_cdf(abs(t)))


def describe(label: str, values: list[float]) -> str:
    mu, sd = mean(values), stdev(values)
    half = 1.96 * sd / math.sqrt(len(values)) if values else 0.0
    return (f"{label:<28} n={len(values):<5} mean={mu:>10,.0f}  "
            f"sd={sd:>9,.0f}  95% CI [{mu - half:>9,.0f}, {mu + half:>9,.0f}]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="significance level (default 0.05)")
    parser.add_argument("--metric", default="score",
                        help="'score' (default) or 'tile:N' for the rate of "
                             "reaching tile N, e.g. --metric tile:32768")
    args = parser.parse_args()

    if args.metric.startswith("tile:"):
        tile = int(args.metric.split(":", 1)[1])
        exponent = tile.bit_length() - 1
        if 1 << exponent != tile:
            print(f"error: {tile} is not a power of two", file=sys.stderr)
            return 2
        base = read_tile_reached(args.baseline, exponent)
        cand = read_tile_reached(args.candidate, exponent)
    else:
        base, cand = read_scores(args.baseline), read_scores(args.candidate)
    if not base or not cand:
        print("error: a run has no rows", file=sys.stderr)
        return 2

    if args.metric.startswith("tile:"):
        for label, values in (("baseline", base), ("candidate", cand)):
            hits = int(sum(values.values()))
            n = len(values)
            rate = hits / n
            half = 1.96 * math.sqrt(rate * (1 - rate) / n)
            print(f"{label:<28} n={n:<5} {hits}/{n} = {rate:.3%}  "
                  f"95% CI [{max(0.0, rate - half):.3%}, {rate + half:.3%}]")
    else:
        print(describe("baseline", list(base.values())))
        print(describe("candidate", list(cand.values())))
    print()

    shared = sorted(set(base) & set(cand))
    if len(shared) >= 2:
        diffs = [cand[s] - base[s] for s in shared]
        mu, sd = mean(diffs), stdev(diffs)
        se = sd / math.sqrt(len(diffs))
        r = correlation([base[s] for s in shared], [cand[s] for s in shared])
        kind = f"PAIRED on {len(shared)} shared seeds"
        wins = sum(1 for d in diffs if d > 0)
        extra = (f"  candidate wins {wins}/{len(diffs)} seeds ({wins / len(diffs):.0%})\n"
                 f"  per-seed correlation r={r:+.3f} "
                 f"({'pairing is buying precision' if abs(r) > 0.3 else 'pairing buys nothing -- expected here'})")
    else:
        b, c = list(base.values()), list(cand.values())
        mu = mean(c) - mean(b)
        se = math.sqrt(stdev(b) ** 2 / len(b) + stdev(c) ** 2 / len(c))
        kind = "UNPAIRED (Welch) -- seed sets do not overlap"
        extra = "  note: matched seeds would give a much tighter test"
        diffs = c

    if se == 0.0:
        print(f"{kind}: runs are bit-identical.")
        return 1

    t = mu / se
    p = two_sided_p(t)
    base_mean = mean(list(base.values()))
    if base_mean == 0.0:
        # A rate endpoint the baseline never hits: relative framing is undefined,
        # so report absolutely rather than dividing by zero.
        base_mean = float("nan")
    print(f"{kind}")
    print(extra)
    if args.metric.startswith("tile:"):
        print(f"  difference   {mu:>+12.2%}  (absolute, vs a {base_mean:.2%} baseline)")
    else:
        print(f"  difference   {mu:>+12,.0f}  ({mu / base_mean:>+.1%} vs baseline)")
    if args.metric.startswith("tile:"):
        print(f"  95% CI       [{mu - 1.96 * se:>+.2%}, {mu + 1.96 * se:>+.2%}]")
    else:
        print(f"  95% CI       [{mu - 1.96 * se:>+,.0f}, {mu + 1.96 * se:>+,.0f}]")
    print(f"  t={t:>+.2f}   p={p:.4g}")
    if len(diffs) < 30:
        print("  WARNING: n<30, the normal approximation overstates significance.")
    print()

    # What this run could have detected: the effect size reachable at 80% power
    # and the given alpha. 2.80 = z(1-alpha/2) + z(0.80) at alpha=0.05.
    mde = 2.80 * se
    n_now = len(diffs)
    if args.metric.startswith("tile:"):
        # A rate endpoint is quoted in PERCENTAGE POINTS, not as a relative
        # change: "2.1% -> 6%" is the question, and framing it as "+186%" is
        # both true and useless.
        print(f"  POWER: at n={n_now} this run can only detect a change of "
              f"{mde:.1%} (absolute) or larger at 80% power.")
        for target in (0.02, 0.04, 0.08):
            needed = math.ceil(n_now * (mde / target) ** 2)
            print(f"         detecting a {target:.0%} absolute change would need n~{needed:,}")
    else:
        print(f"  POWER: at n={n_now} this run can only detect effects of "
              f"{mde:,.0f} or larger ({mde / base_mean:+.1%}) at 80% power.")
        for target in (0.05, 0.03, 0.02):
            needed = math.ceil(n_now * (mde / (target * base_mean)) ** 2)
            print(f"         detecting {target:.0%} would need n~{needed:,}")
    print()

    if p < args.alpha and mu > 0:
        print(f"VERDICT: candidate is significantly BETTER (p<{args.alpha}).")
        return 0
    if p < args.alpha:
        print(f"VERDICT: candidate is significantly WORSE (p<{args.alpha}).")
        return 1
    if abs(mu) < mde:
        # The distinction this project kept losing: "no evidence of a
        # difference" is not "evidence of no difference". Say which one it is.
        floor = (f"{mde:.1%} (absolute)" if args.metric.startswith("tile:")
                 else f"{mde / base_mean:+.1%}")
        print(f"VERDICT: UNDERPOWERED -- no significant difference (p>={args.alpha}), but "
              f"this run could not have\n         detected anything smaller than "
              f"{floor}. Do NOT file this as a tie; either raise n "
              f"(--threads\n         buys it) or record it as unmeasured.")
        return 1
    print(f"VERDICT: no significant difference (p>={args.alpha}), and the run was powered "
          f"to see one.\n         Treat as a tie; prefer the cheaper option.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
