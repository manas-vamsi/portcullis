# ADR-0001: Decisions are made by a deterministic function, never by a model

Status: accepted. Date: 2026-11-19.

## Context

An agent's tool calls need a gate. The obvious gate is another LLM that judges each call. But a judge reads the
same untrusted text the agent read, so the text that hijacks the agent can hijack the judge; its answers are not
reproducible; and it costs tokens and latency on every call.

## Decision

Every allow, deny or ask is the result of `decide(policy, call, state)`, a pure function of declarative data
(allowlist, risk levels, argument patterns, amount thresholds, budgets). No model is consulted. It never raises: any
failure while deciding is a deny (`evaluation_error`). Deny always beats ask.

## Consequences

- Good: the same input always gives the same answer, it can be tested exhaustively (100% line coverage, and each rule is checked by breaking it and watching a test fail), and a decision takes about 10 microseconds (measured, see `benchmarks/`).
- Bad: it is only as good as the rules someone writes. Patterns are regular expressions and can be evaded in ways a person would notice. It cannot judge intent. The README lists these limits.

## Where

`src/portcullis/policy.py`, `tests/test_policy.py`, `tests/test_attacks.py`
