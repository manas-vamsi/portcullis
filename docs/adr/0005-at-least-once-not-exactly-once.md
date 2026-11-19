# ADR-0005: Side effects are at-least-once, deduplicated by a key the model cannot choose

Status: accepted. Date: 2026-11-19.

## Context

If a process dies after a tool ran but before its result was recorded, a resume cannot know whether the effect
happened. Exactly-once across systems that do not share a transaction is not achievable.

## Decision

Each call gets an idempotency key derived from the run and the call's position in the history, never from the model's
own call id (models reuse ids). A Postgres ledger stores the first result for a key; a retry after a crash reuses the
key and returns the stored result. The outside service must deduplicate by that key. One that ignores keys can see
one duplicate, and a test pins that case.

## Consequences

- Good: honest about what is guaranteed. With a service that deduplicates by the key, a retry has no extra effect; the delivery itself is still at-least-once.
- Bad: delivery is at-least-once. The README says so in several places; it must never be described as exactly-once.

## Where

`src/portcullis/idempotency.py`, `tests/test_idempotency.py`, `tests/test_kill_and_resume.py`
