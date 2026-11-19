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
- Bad, and what was done about it: rebuilding and re-verifying the whole history on every request made the server slower as a run grew (about 78 ms per decision after 1000 calls, measured). The server now keeps what it verified, and reuses it only while the database confirms, with a digest of every stored field of every row, that the history is unchanged to the byte; any difference means a full read and verification as before. After 1000 calls a decision costs about 42 ms (measured). A cache hit still detects an edit anywhere in history. Like plain verification, it cannot detect a rewrite of the whole chain made consistent.
- The hook runs as a new process each time, so it cannot keep a cache; it reads and verifies the session once per call.

## Where

`src/portcullis/runstate.py`, `src/portcullis/server.py`, `src/portcullis/hook.py`, `tests/test_runstate.py`
