# ADR-0012: Nothing is claimed that is not built, tested or measured, and weaknesses are pinned by tests

Status: accepted. Date: 2026-11-19.

## Context

A governance project that overstates what it does is worse than useless, and a portfolio project that publishes
invented numbers is not credible.

## Decision

The README marks a feature as built only when tests back it. Numbers appear only when measured, with the machine and the
method. Known weaknesses are written down and pinned by tests named `test_known_gap__...`, so they cannot change or be forgotten
silently. Most security-sensitive pieces (the hook, the server, the model adapter and evals, the gap fixes, the MCP proxy) were reviewed independently and the findings fixed with tests; the dashboard was checked by tests and a DOM run but not independently reviewed. Tests are checked by
breaking the code and watching them fail. Real-model results are not published until a real model has been run.

## Consequences

- Good: every claim has something behind it, and the gaps are visible.
- Bad: the README is long and states many limits. That is the point.

## Where

`README.md`, `tests/test_attacks.py`, `evals/`, `benchmarks/`
