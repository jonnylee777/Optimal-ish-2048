#!/usr/bin/env python3
"""Decide whether two experiment runs actually differ.

2048 scores are wildly dispersed -- a single configuration routinely spans
5k to 175k across seeds. That makes eyeballing two mean scores useless at
small n, and it is exactly how this project once concluded "depth 2 is best"
from an n=10 run that a larger sample reversed.

Two runs on the SAME seeds are paired observations: the seed's luck cancels
out of the difference, which typically shrinks the standard error several-fold
versus treating the runs as independent. This tool uses the paired test
whenever the seed sets overlap, and says so in the output.

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


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


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
    args = parser.parse_args()

    base, cand = read_scores(args.baseline), read_scores(args.candidate)
    if not base or not cand:
        print("error: a run has no rows", file=sys.stderr)
        return 2

    print(describe("baseline", list(base.values())))
    print(describe("candidate", list(cand.values())))
    print()

    shared = sorted(set(base) & set(cand))
    if len(shared) >= 2:
        diffs = [cand[s] - base[s] for s in shared]
        mu, sd = mean(diffs), stdev(diffs)
        se = sd / math.sqrt(len(diffs))
        kind = f"PAIRED on {len(shared)} shared seeds"
        wins = sum(1 for d in diffs if d > 0)
        extra = f"  candidate wins {wins}/{len(diffs)} seeds ({wins / len(diffs):.0%})"
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
    print(f"{kind}")
    print(extra)
    print(f"  difference   {mu:>+12,.0f}  ({mu / base_mean:>+.1%} vs baseline)")
    print(f"  95% CI       [{mu - 1.96 * se:>+,.0f}, {mu + 1.96 * se:>+,.0f}]")
    print(f"  t={t:>+.2f}   p={p:.4g}")
    if len(diffs) < 30:
        print("  WARNING: n<30, the normal approximation overstates significance.")
    print()

    if p < args.alpha and mu > 0:
        print(f"VERDICT: candidate is significantly BETTER (p<{args.alpha}).")
        return 0
    if p < args.alpha:
        print(f"VERDICT: candidate is significantly WORSE (p<{args.alpha}).")
        return 1
    print(f"VERDICT: no significant difference (p>={args.alpha}). "
          "Treat as a tie; prefer the cheaper option.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
