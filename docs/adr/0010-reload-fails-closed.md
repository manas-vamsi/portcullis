# ADR-0010: Policy and key files reload on change, and a bad file refuses everything

Status: accepted. Date: 2026-11-19.

## Context

Operators need to tighten a policy or revoke a key without a restart. Carrying on with the last good version after
a bad edit is dangerous: it might be the looser one.

## Decision

The server reloads each file when its modification time, size or inode changes. If the new version cannot be read or is
invalid, every request fails with 503 until a good version appears, and a failed read is retried on each request. The old
version is deliberately not kept. A bad file at startup is an error.

## Consequences

- Good: a revoked key stops working on the next request; a typo cannot leave an older, looser policy in force unnoticed.
- Bad: a typo takes the service down until it is fixed. That is the intended trade (see the fail-closed decision).

## Where

`src/portcullis/reloadable.py`, `src/portcullis/server.py`, `tests/test_reloadable.py`
