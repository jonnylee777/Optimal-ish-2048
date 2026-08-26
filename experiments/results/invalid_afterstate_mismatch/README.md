# Invalid results — afterstate/state leaf mismatch

These N1 runs are **not valid measurements of N1's playing strength**. They
were produced before the bug in `docs/ntuple-learning.md` ("KNOWN ISSUE") was
understood: N1 is an *afterstate* value function, but our expectimax evaluates
leaves on *post-spawn states*, so the evaluator is asked a question it was
never trained to answer.

Kept as evidence of the bug's magnitude (4,228 at depth 4 versus 108,946 for
the same weights at 1-ply greedy), and deliberately held outside
`fixed_depth/` and `timed/` so `tools/summarize_experiment.py` does not fold
them into the heuristic comparison as if they were real.

## Resolved — do not move these files back

Both underlying defects are now fixed (see
[`../../../docs/ULTIMATE_AGENT_PROGRESS.md`](../../../docs/ULTIMATE_AGENT_PROGRESS.md)):

- **E1** — the leaf mismatch, fixed by `EvaluationSemantics` on the
  `Evaluator` interface. The capability is a query, not the
  `prefers_afterstate()` predicate this README originally anticipated, so
  future learned evaluators inherit it without touching the search.
- **E2** — a separate reward off-by-one in the TD target, found while
  auditing for E1.

These files stay quarantined permanently. They are a record of the bug, not
data: re-running the same commands today produces completely different
numbers (depth 1 went 14,262 → 102,861), so folding them into any comparison
would be wrong. Replacement measurements live in `fixed_depth/` and `timed/`.
