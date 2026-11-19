# ADR-0003: One hash-chained audit log per run, in Postgres

Status: accepted. Date: 2026-11-19.

## Context

Reviewers and regulators need evidence of what an agent was allowed and what a human approved. A plain table
can be edited without trace.

## Decision

Each run has its own chain. Every row stores the hash of the previous row and a hash over all its own fields, so an
edit, a removal from the middle or a reorder is detected by `verify()`. Appends take a per-run advisory lock. Event ids
are deterministic so a repeated append (LangGraph re-runs an interrupted node) is recorded once; reusing an id for
different content raises an error. Rows hold a hash of each request, not its arguments, because arguments may be
sensitive.

## Consequences

- Good: tamper-evident, cheap to verify (about 17 microseconds per row, measured), and independent per run so writers do not serialize across runs.
- Bad: tamper-evident is not tamper-proof. Deleting the last rows or a whole run is not detected unless you keep the head hash somewhere else; someone with database write access can rewrite a whole chain; an empty log and a missing log look the same. Appends cost about 12 ms each (measured). A global chain or periodic external anchors would close the gaps and are not built.

## Where

`src/portcullis/audit.py`, `tests/test_audit.py`, `agentctl verify`
