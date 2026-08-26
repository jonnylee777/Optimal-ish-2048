#!/usr/bin/env bash
# Measures how the benefit of search depth changes with training budget.
#
# E3 established that depth-2 gain shrinks as the 1-ply policy gets stronger,
# using four networks that differed in several ways at once (reward index,
# step-size rule, training budget). This measures the same relationship along a
# SINGLE axis: the checkpoints of one training run, which differ only in how
# many games they have seen.
#
# Checkpoints are byte-reproducible (verified), so this curve is a property of
# the training run rather than of seed noise.
#
# Usage:
#   tools/depth_vs_budget.sh experiments/weights/n2_tc_a1_1M [n_games] [max_depth]
#
# Expects checkpoints named <prefix>.at<N>.bin, as written by
# `train_ntuple --checkpoint-every`.
set -u

prefix="${1:?usage: depth_vs_budget.sh <weight-prefix> [games] [max-depth]}"
games="${2:-200}"
max_depth="${3:-2}"
seeds="30000-$((30000 + games - 1))"
outdir="${TMPDIR:-/tmp}/depth_vs_budget"
mkdir -p "$outdir"

binary=./build-release/run_experiment
if [[ ! -x "$binary" ]]; then
    echo "error: $binary not found; build first" >&2
    exit 2
fi

shopt -s nullglob
checkpoints=("$prefix".at*.bin)
if (( ${#checkpoints[@]} == 0 )); then
    echo "error: no checkpoints matching $prefix.at*.bin" >&2
    exit 2
fi

# Sort by the embedded game count, which is numeric and not zero-padded, so a
# lexical sort would put 1000000 before 200000.
mapfile -t checkpoints < <(
    for path in "${checkpoints[@]}"; do
        n="${path##*.at}"
        echo "${n%%.bin} $path"
    done | sort -n | cut -d' ' -f2-
)

printf '%-12s' "games"
for ((d = 1; d <= max_depth; d++)); do printf '%12s' "depth$d"; done
printf '%12s\n' "best gain"

for path in "${checkpoints[@]}"; do
    n="${path##*.at}"
    n="${n%%.bin}"
    printf '%-12s' "$n"

    baseline=""
    best=""
    for ((d = 1; d <= max_depth; d++)); do
        score=$("$binary" --heuristic N1 --weights "$path" --search fixed --depth "$d" \
                    --seeds "$seeds" --output-dir "$outdir" 2>/dev/null |
                grep -o 'mean score: [0-9.]*' | cut -d' ' -f3)
        # A blank here means the run failed; print it rather than silently
        # dropping the row, so a partial table is never mistaken for a complete
        # one.
        printf '%12s' "${score:-FAILED}"
        [[ -z "$score" ]] && continue
        [[ -z "$baseline" ]] && baseline="$score"
        if [[ -z "$best" ]] || awk "BEGIN{exit !($score > $best)}"; then best="$score"; fi
    done

    if [[ -n "$baseline" && -n "$best" ]]; then
        printf '%11s%%\n' "$(awk "BEGIN{printf \"%+.1f\", 100*($best-$baseline)/$baseline}")"
    else
        printf '%12s\n' "-"
    fi
done

echo
echo "Raw results in $outdir; use tools/compare_runs.py for paired significance."
