#!/usr/bin/env python3
"""Summarize run_experiment JSON results into a Markdown report.

Adapted from the legacy tools/generate_comparison_report.py — reuses its
matched-seed paired-comparison approach, pointed at the standardized
run_experiment schema (see docs/phase1-heuristics.md) instead of the
milestone-named legacy schema.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory to scan for *.json result files (repeatable). "
        "Defaults to experiments/results/phase1-heuristics and experiments/results/phase1-heuristics.",
    )
    parser.add_argument("--title", default="Experiment summary")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_results(directories: list[Path]) -> list[dict[str, Any]]:
    results = []
    for directory in directories:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            data["_path"] = path
            results.append(data)
    return results


def run_key(data: dict[str, Any]) -> tuple:
    search = data["search"]
    experiment = data["experiment"]
    return (
        search["depth"],
        search.get("adaptive_depth", False),
        search["time_limit_seconds"],
        search["minimum_path_probability"],
        experiment["first_seed"],
        experiment["last_seed"],
    )


def key_label(key: tuple) -> str:
    depth, adaptive, time_limit, cutoff, first_seed, last_seed = key
    mode = "adaptive" if adaptive else f"depth {depth}"
    budget = f", {time_limit * 1000:g} ms/move" if time_limit > 0 else ", exact"
    cutoff_text = f", cutoff {cutoff:g}" if cutoff > 0 else ""
    return f"{mode}{budget}{cutoff_text}, seeds {first_seed}-{last_seed}"


def number(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def paired_summary(a: dict[str, Any], b: dict[str, Any]) -> tuple[int, int, int, float]:
    a_games = {g["seed"]: g["score"] for g in a["games"]}
    b_games = {g["seed"]: g["score"] for g in b["games"]}
    common = sorted(a_games.keys() & b_games.keys())
    b_wins = sum(b_games[s] > a_games[s] for s in common)
    a_wins = sum(b_games[s] < a_games[s] for s in common)
    ties = len(common) - b_wins - a_wins
    mean_delta = (
        sum(b_games[s] - a_games[s] for s in common) / len(common) if common else 0.0
    )
    return b_wins, a_wins, ties, mean_delta


def generate_report(title: str, results: list[dict[str, Any]], output_dir: Path) -> str:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for data in results:
        groups.setdefault(run_key(data), []).append(data)

    lines = [
        f"# {title}",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- Result files scanned: {len(results)}",
        "",
        "## Score and runtime summary",
        "",
        "| Config | Heuristic | Games | Mean score | Median | Min/Max | "
        "Highest tile | ms/move | Deadline hit rate |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for key in sorted(groups, key=key_label):
        for data in sorted(groups[key], key=lambda d: d["evaluator"]["type"]):
            metrics = data["metrics"]
            search = data["search"]
            deadline = search.get("deadline_hit_rate")
            deadline_text = f"{deadline * 100:.1f}%" if deadline is not None else "n/a"
            lines.append(
                f"| {key_label(key)} | `{data['evaluator']['type']}` | "
                f"{data['experiment']['game_count']} | {number(metrics['mean_score'])} | "
                f"{number(metrics['median_score'])} | "
                f"{metrics['worst_score']:,}/{metrics['best_score']:,} | "
                f"{metrics['highest_tile']:,} | "
                f"{number(metrics['mean_milliseconds_per_move'], 4)} | {deadline_text} |"
            )

    lines.extend(["", "## Paired-seed outcomes (within matching configs)", "",
                  "| Config | Comparison | Wins | Losses | Ties | Mean paired difference |",
                  "|---|---|---:|---:|---:|---:|"])
    for key in sorted(groups, key=key_label):
        entries = sorted(groups[key], key=lambda d: d["evaluator"]["type"])
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                a, b = entries[i], entries[j]
                b_wins, a_wins, ties, mean_delta = paired_summary(a, b)
                lines.append(
                    f"| {key_label(key)} | `{b['evaluator']['type']}` vs. "
                    f"`{a['evaluator']['type']}` | {b_wins} | {a_wins} | {ties} | "
                    f"{number(mean_delta)} |"
                )

    lines.extend(["", "## Source result files", ""])
    for key in sorted(groups, key=key_label):
        for data in sorted(groups[key], key=lambda d: d["evaluator"]["type"]):
            path: Path = data["_path"]
            relative = os.path.relpath(path, start=output_dir)
            lines.append(f"- {key_label(key)}, `{data['evaluator']['type']}`: [{path.name}]({relative})")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    directories = args.results_dir or [
        Path("experiments/results/phase1-heuristics"),
        Path("experiments/results/phase1-heuristics"),
    ]
    results = load_results(directories)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = generate_report(args.title, results, args.output.parent)
    args.output.write_text(report, encoding="utf-8")
    print(f"Report written: {args.output} ({len(results)} result files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
