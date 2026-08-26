#!/usr/bin/env python3
"""Combine matched baseline experiment JSON files into a Markdown report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_AGENTS = ("baseline", "baseline-optimized")


def parse_games_by_depth(text: str) -> dict[int, int]:
    plan: dict[int, int] = {}
    try:
        for item in text.split(","):
            depth_text, games_text = item.split(":", maxsplit=1)
            depth = int(depth_text)
            games = int(games_text)
            if depth < 1 or games < 1 or depth in plan:
                raise ValueError
            plan[depth] = games
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected comma-separated DEPTH:GAMES values, for example 1:100,2:100"
        ) from error
    return plan


def parse_agents(text: str) -> tuple[str, ...]:
    agents = tuple(agent.strip() for agent in text.split(",") if agent.strip())
    if not agents or len(set(agents)) != len(agents):
        raise argparse.ArgumentTypeError("agents must be a nonempty, unique comma-separated list")
    return agents


def parse_depths(text: str) -> tuple[int, ...]:
    try:
        depths = tuple(int(item) for item in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("depths must be comma-separated integers") from error
    if not depths or any(depth < 1 for depth in depths) or len(set(depths)) != len(depths):
        raise argparse.ArgumentTypeError("depths must be positive and unique")
    return tuple(sorted(depths))


def parse_depth_labels(text: str) -> dict[int, str]:
    labels: dict[int, str] = {}
    try:
        for item in text.split(","):
            depth_text, label = item.split(":", maxsplit=1)
            depth = int(depth_text)
            if depth < 1 or not label.strip() or depth in labels:
                raise ValueError
            labels[depth] = label.strip()
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected DEPTH:LABEL entries, for example '4:Fixed 4,8:Adaptive 4/6/8'"
        ) from error
    return labels


def parse_time_limits_by_depth(text: str) -> dict[int, float]:
    limits: dict[int, float] = {}
    try:
        for item in text.split(","):
            depth_text, limit_text = item.split(":", maxsplit=1)
            depth = int(depth_text)
            limit = float(limit_text)
            if depth < 1 or limit < 0.0 or depth in limits:
                raise ValueError
            limits[depth] = limit
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected DEPTH:MILLISECONDS entries, for example 4:0,8:250"
        ) from error
    return limits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a baseline-vs-optimized comparison report."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--first-seed", type=int, required=True)
    range_group = parser.add_mutually_exclusive_group(required=True)
    range_group.add_argument("--last-seed", type=int)
    range_group.add_argument("--games-by-depth", type=parse_games_by_depth)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--depths", type=parse_depths)
    parser.add_argument("--probability-cutoff", type=float, default=0.0)
    parser.add_argument("--time-limit-ms", type=float, default=0.0)
    parser.add_argument("--time-limits-by-depth", type=parse_time_limits_by_depth)
    parser.add_argument("--agents", type=parse_agents, default=DEFAULT_AGENTS)
    parser.add_argument("--reference-agent")
    parser.add_argument("--title", default="Baseline vs. optimized heuristic comparison")
    parser.add_argument("--depth-label")
    parser.add_argument("--depth-labels", type=parse_depth_labels)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.last_seed is not None and args.first_seed > args.last_seed:
        parser.error("--first-seed must not exceed --last-seed")
    if args.min_depth < 1 or args.min_depth > args.max_depth:
        parser.error("depth range must be positive and ordered")
    if not 0.0 <= args.probability_cutoff < 1.0:
        parser.error("--probability-cutoff must be in [0, 1)")
    if args.time_limit_ms < 0.0:
        parser.error("--time-limit-ms must be nonnegative")
    args.depths = args.depths or tuple(range(args.min_depth, args.max_depth + 1))
    requested_depths = set(args.depths)
    if args.depth_label and len(args.depths) != 1:
        parser.error("--depth-label requires exactly one requested depth")
    if args.depth_labels is not None and set(args.depth_labels) != requested_depths:
        parser.error("--depth-labels must label every requested depth exactly once")
    if args.time_limits_by_depth is not None:
        if args.time_limit_ms != 0.0:
            parser.error("use either --time-limit-ms or --time-limits-by-depth")
        if set(args.time_limits_by_depth) != requested_depths:
            parser.error("--time-limits-by-depth must specify every requested depth")
    else:
        args.time_limits_by_depth = {
            depth: args.time_limit_ms for depth in args.depths
        }
    if args.games_by_depth is None:
        games = args.last_seed - args.first_seed + 1
        args.games_by_depth = {depth: games for depth in requested_depths}
    elif set(args.games_by_depth) != requested_depths:
        parser.error("--games-by-depth must specify every requested depth exactly once")
    if args.reference_agent is None:
        args.reference_agent = args.agents[0]
    elif args.reference_agent not in args.agents:
        parser.error("--reference-agent must be included in --agents")
    if args.output is None:
        if args.last_seed is not None:
            suffix = f"seeds{args.first_seed}-{args.last_seed}"
        else:
            suffix = f"firstseed{args.first_seed}_variable_games"
        args.output = args.results_dir / (
            f"baseline_vs_optimized_depth{args.min_depth}-{args.max_depth}_{suffix}.md"
        )
    return args


def load_matching_results(args: argparse.Namespace) -> dict[tuple[int, str], dict[str, Any]]:
    selected: dict[tuple[int, str], dict[str, Any]] = {}
    for path in args.results_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            agent = data["agent"]
            search = data["search"]
            experiment = data["experiment"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
        depth = search.get("depth")
        cutoff = search.get("minimum_path_probability", 0.0)
        time_limit_seconds = search.get("time_limit_seconds", 0.0)
        expected_games = args.games_by_depth.get(depth) if isinstance(depth, int) else None
        expected_last_seed = (
            args.first_seed + expected_games - 1 if expected_games is not None else None
        )
        if (
            agent not in args.agents
            or not isinstance(depth, int)
            or depth not in args.depths
            or experiment.get("first_seed") != args.first_seed
            or experiment.get("last_seed") != expected_last_seed
            or experiment.get("game_count") != expected_games
            or abs(float(cutoff) - args.probability_cutoff) > 1e-15
            or abs(
                float(time_limit_seconds) * 1000.0
                - args.time_limits_by_depth[depth]
            ) > 1e-9
        ):
            continue
        key = (depth, agent)
        previous = selected.get(key)
        if previous is None or path.stat().st_mtime > previous["_path"].stat().st_mtime:
            data["_path"] = path
            selected[key] = data
    return selected


def number(value: float, digits: int = 1) -> str:
    return f"{value:,.{digits}f}"


def percent_delta(before: float, after: float) -> str:
    if before == 0:
        return "n/a"
    return f"{(after - before) * 100.0 / before:+.1f}%"


def rate(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def duration(seconds: float) -> str:
    if seconds >= 3600.0:
        return f"{seconds / 3600.0:.2f} h"
    if seconds >= 60.0:
        return f"{seconds / 60.0:.1f} min"
    return f"{seconds:.2f} s"


def mode_max_tile(result: dict[str, Any]) -> int:
    recorded = result["metrics"].get("mode_max_tile")
    if isinstance(recorded, int):
        return recorded
    counts = Counter(int(game["max_tile"]) for game in result["games"])
    highest_frequency = max(counts.values())
    return max(tile for tile, count in counts.items() if count == highest_frequency)


def depth_usage(result: dict[str, Any], field: str, depth: int) -> int:
    usage = result.get("search", {}).get(field, {})
    return int(usage.get(f"depth_{depth}_moves", 0))


def count_and_rate(count: int, total: int) -> str:
    return f"{count:,} ({count * 100.0 / total:.1f}%)" if total else "0 (0.0%)"


def depth_label(args: argparse.Namespace, depth: int) -> str:
    if args.depth_label:
        return args.depth_label
    if args.depth_labels:
        return args.depth_labels[depth]
    return str(depth)


def paired_summary(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> tuple[int, int, int, float]:
    reference_games = {game["seed"]: game["score"] for game in reference["games"]}
    candidate_games = {game["seed"]: game["score"] for game in candidate["games"]}
    common = sorted(reference_games.keys() & candidate_games.keys())
    candidate_wins = sum(candidate_games[seed] > reference_games[seed] for seed in common)
    reference_wins = sum(candidate_games[seed] < reference_games[seed] for seed in common)
    ties = len(common) - candidate_wins - reference_wins
    mean_delta = (
        sum(candidate_games[seed] - reference_games[seed] for seed in common) / len(common)
        if common
        else 0.0
    )
    return candidate_wins, reference_wins, ties, mean_delta


def generate_report(args: argparse.Namespace, results: dict[tuple[int, str], dict[str, Any]]) -> str:
    expected = len(args.depths) * len(args.agents)
    complete = len(results) == expected
    time_limits = set(args.time_limits_by_depth.values())
    if len(time_limits) > 1:
        mode = "mixed fixed-depth and time-bounded iterative deepening"
    elif next(iter(time_limits)) > 0:
        mode = "time-bounded iterative deepening"
    else:
        mode = "exact" if args.probability_cutoff == 0 else "approximate"
    seed_plan = ", ".join(
        f"depth {depth}: {args.first_seed}-{args.first_seed + games - 1} ({games} games)"
        for depth, games in sorted(args.games_by_depth.items())
    )
    lines = [
        f"# {args.title}",
        "",
        f"- Experiment plan: {seed_plan}",
        f"- Depth modes: {', '.join(depth_label(args, depth) for depth in args.depths)}",
        f"- Search: {mode}, probability cutoff `{args.probability_cutoff:g}`",
        "- Per-move time limits: " + ", ".join(
            f"{depth_label(args, depth)}="
            f"{'none' if args.time_limits_by_depth[depth] == 0 else f'{args.time_limits_by_depth[depth]:g} ms'}"
            for depth in args.depths
        ),
        f"- Agents: {', '.join(f'`{agent}`' for agent in args.agents)}",
        f"- Reference agent: `{args.reference_agent}`",
        f"- Status: {'complete' if complete else f'partial ({len(results)}/{expected} runs found)' }",
        f"- Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Score and runtime summary",
        "",
        "| Depth | Agent | Games | Average score | Difference vs. reference | Improvement | "
        "Maximum tile | Mode maximum tile | Runtime | ms/move |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for depth in args.depths:
        reference = results.get((depth, args.reference_agent))
        reference_score = reference["metrics"]["mean_score"] if reference else None
        depth_display = depth_label(args, depth)
        for agent in args.agents:
            result = results.get((depth, agent))
            if result is None:
                lines.append(f"| {depth_display} | `{agent}` | pending | — | — | — | — | — | — | — |")
                continue
            metrics = result["metrics"]
            difference = metrics["mean_score"] - reference_score if reference_score is not None else None
            total_runtime = sum(float(game["runtime_seconds"]) for game in result["games"])
            lines.append(
                f"| {depth_display} | `{agent}` | {result['experiment']['game_count']} | "
                f"{number(metrics['mean_score'])} | "
                f"{number(difference) if difference is not None else 'pending'} | "
                f"{percent_delta(reference_score, metrics['mean_score']) if reference_score is not None else 'pending'} | "
                f"{metrics['highest_tile']} | {mode_max_tile(result)} | "
                f"{duration(total_runtime)} | {number(metrics['mean_milliseconds_per_move'], 4)} |"
            )

    has_depth_tracking = (
        bool(args.depth_label and "adaptive" in args.depth_label.lower())
        or any(
            "adaptive_depth_usage" in result.get("search", {})
            for result in results.values()
        )
    )
    if has_depth_tracking:
        lines.extend([
            "",
            "## Adaptive depth tracking",
            "",
            "Requested depth is the adaptive ceiling selected from the board's empty-cell "
            "count. Completed depth is the last iterative-deepening level finished within "
            "the per-move time limit.",
            "",
            "### Requested maximum depth",
            "",
            "| Agent | Total moves | Depth 4 requested | Depth 6 requested | Depth 8 requested |",
            "|---|---:|---:|---:|---:|",
        ])
        for depth in args.depths:
            if args.time_limits_by_depth[depth] == 0:
                continue
            for agent in args.agents:
                result = results.get((depth, agent))
                if result is None:
                    lines.append(f"| `{agent}` | pending | — | — | — |")
                    continue
                if "adaptive_depth_usage" not in result.get("search", {}):
                    continue
                counts = [depth_usage(result, "adaptive_depth_usage", item) for item in (4, 6, 8)]
                total = sum(counts)
                lines.append(
                    f"| `{agent}` | {total:,} | {count_and_rate(counts[0], total)} | "
                    f"{count_and_rate(counts[1], total)} | {count_and_rate(counts[2], total)} |"
                )

        lines.extend([
            "",
            "### Actually completed depth",
            "",
            "| Agent | Average | Depth 1 | Depth 2 | Depth 3 | Depth 4 | Depth 5 | Depth 6 | Depth 7 | Depth 8 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for depth in args.depths:
            if args.time_limits_by_depth[depth] == 0:
                continue
            for agent in args.agents:
                result = results.get((depth, agent))
                if result is None:
                    lines.append(f"| `{agent}` | pending | — | — | — | — | — | — | — | — |")
                    continue
                if "completed_depth_usage" not in result.get("search", {}):
                    continue
                counts = [
                    depth_usage(result, "completed_depth_usage", item)
                    for item in range(1, 9)
                ]
                total = sum(counts)
                average = (
                    sum(item * counts[item - 1] for item in range(1, 9)) / total
                    if total else 0.0
                )
                lines.append(
                    f"| `{agent}` | {average:.2f} | "
                    + " | ".join(f"{count:,}" for count in counts)
                    + " |"
                )

    lines.extend([
        "",
        "## Tile achievement rates",
        "",
        "| Depth | Agent | 1024 | 2048 | 4096 | 8192 |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for depth in args.depths:
        for agent in args.agents:
            result = results.get((depth, agent))
            if result is None:
                lines.append(f"| {depth_label(args, depth)} | `{agent}` | pending | — | — | — |")
                continue
            metrics = result["metrics"]
            lines.append(
                f"| {depth_label(args, depth)} | `{agent}` | {rate(metrics['achievement_rate_1024'])} | "
                f"{rate(metrics['achievement_rate_2048'])} | "
                f"{rate(metrics['achievement_rate_4096'])} | "
                f"{rate(metrics['achievement_rate_8192'])} |"
            )

    lines.extend([
        "",
        "## Paired-seed outcomes",
        "",
        "| Depth | Comparison | Candidate wins | Reference wins | Ties | Mean paired difference |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for depth in args.depths:
        reference = results.get((depth, args.reference_agent))
        for agent in args.agents:
            if agent == args.reference_agent:
                continue
            candidate = results.get((depth, agent))
            comparison = f"`{agent}` vs. `{args.reference_agent}`"
            if reference is None or candidate is None:
                lines.append(
                    f"| {depth_label(args, depth)} | {comparison} | "
                    "pending | pending | pending | pending |"
                )
                continue
            candidate_wins, reference_wins, ties, mean_delta = paired_summary(
                reference, candidate)
            lines.append(
                f"| {depth_label(args, depth)} | {comparison} | {candidate_wins} | {reference_wins} | "
                f"{ties} | {number(mean_delta)} |"
            )

    if len(args.depths) == 2:
        fixed_depth, adaptive_depth = args.depths
        lines.extend([
            "",
            "## Fixed versus adaptive search mode",
            "",
            f"Adaptive outcomes are compared with {depth_label(args, fixed_depth)} "
            f"for the same policy and matched seed.",
            "",
            "| Agent | Adaptive wins | Fixed wins | Ties | Mean adaptive score difference | Fixed ms/move | Adaptive ms/move |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for agent in args.agents:
            fixed = results.get((fixed_depth, agent))
            adaptive = results.get((adaptive_depth, agent))
            if fixed is None or adaptive is None:
                lines.append(f"| `{agent}` | pending | pending | pending | pending | pending | pending |")
                continue
            adaptive_wins, fixed_wins, ties, mean_delta = paired_summary(fixed, adaptive)
            lines.append(
                f"| `{agent}` | {adaptive_wins} | {fixed_wins} | {ties} | "
                f"{number(mean_delta)} | "
                f"{number(fixed['metrics']['mean_milliseconds_per_move'], 4)} | "
                f"{number(adaptive['metrics']['mean_milliseconds_per_move'], 4)} |"
            )

    lines.extend(["", "## Source result files", ""])
    for depth in args.depths:
        for agent in args.agents:
            result = results.get((depth, agent))
            if result is None:
                lines.append(f"- {depth_label(args, depth)}, `{agent}`: pending")
            else:
                path: Path = result["_path"]
                lines.append(
                    f"- {depth_label(args, depth)}, `{agent}`: "
                    f"[{path.name}]({path.name})"
                )

    lines.extend([
        "",
        "## Interpretation note",
        "",
        "Each comparison uses matched seeds to reduce variance. Interpret results together with "
        "sample size and score spread, then validate the selected policy on unused seeds.",
        "",
    ])
    if max(args.games_by_depth.values()) <= 5:
        lines.extend([
            "This small game sample is a pilot comparison, not a robust estimate of expected "
            "playing strength. Matched seeds improve fairness but do not eliminate sampling "
            "uncertainty.",
            "",
        ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    results = load_matching_results(args)
    report = generate_report(args, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(args.output)
    print(f"Report written: {args.output}")
    print(
        f"Matched runs: {len(results)}/"
        f"{len(args.depths) * len(args.agents)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
