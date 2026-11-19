# ADR-0004: A run's state is rebuilt from its own audit chain

Status: accepted. Date: 2026-11-19.

## Context

Budgets, loop detection and spend caps need a run's history. Keeping counters in a second place (memory, a table)
creates a second source of truth that can disagree with the log and be tampered with separately.

## Decision

The server and the Claude Code hook rebuild steps, loop history, spend, tokens and elapsed time by folding the run's
rows on every request, after verifying the chain. A chain that does not verify, or rows that do not tell a coherent
story, stop the run. Once a run is halted, further calls are answered without writing more rows so a flood cannot
grow the log.

## Consequences

- Good: one source of truth; tampering with history stops the run instead of producing a decision from bad state.
- Bad: the cost grows with the length of the run. Measured on the server: about 26 ms per decision after 10 calls and about 78 ms after 1000. The run is bounded by `max_steps`, but a long session pays for it. A cache keyed by the chain head would fix this and is not built.

## Where

`src/portcullis/runstate.py`, `src/portcullis/server.py`, `src/portcullis/hook.py`, `tests/test_runstate.py`
