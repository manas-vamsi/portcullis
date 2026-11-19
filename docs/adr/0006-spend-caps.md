# ADR-0006: Spend is counted when a call is allowed, under two caps that approval cannot exceed

Status: accepted. Date: 2026-11-19.

## Context

A per-call approval threshold is defeated by splitting one large action into many small ones, and by starting
new runs.

## Decision

Spend is added up when a call is allowed or approved, not when the caller reports it ran, so it can only over-count.
`max_spend` caps one run; `max_spend_per_day` caps a whole tenant over a rolling 24 hours across all its runs (kept in
a ledger table, read under a lock so concurrent runs cannot both pass it). Both are checked before the approval
rule, so a human cannot approve past either, and an approval is checked again when it arrives.

## Consequences

- Good: the attack suite's split-order gap is closed and tested, including a race of eight runs at once.
- Bad: a call that is allowed but never runs still counts. Only the server tracks the daily cap; the library agent loop, the Claude Code hook and the MCP proxy treat it as zero (the proxy warns at startup).

## Where

`src/portcullis/policy.py`, `src/portcullis/server.py`, `tests/test_attacks.py`, `tests/test_server.py`
