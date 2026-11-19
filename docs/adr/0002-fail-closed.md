# ADR-0002: Every failure refuses the call

Status: accepted. Date: 2026-11-19.

## Context

Governance that fails open is worse than none: an outage becomes an open door. Some of the places portcullis
runs fail open by default. In Claude Code a hook that exits 1, crashes on import or times out lets the tool call
through; only exit 2 blocks.

## Decision

Every error path refuses. The hook catches everything and exits 2, a launcher with a watchdog fires before the
hook timeout, and `hooks.json` sets `onFailure: block`. The server answers any failure with a non-2xx status and
tells callers to treat anything other than `allowed` or `approved` as a refusal. The agent loop refuses malformed
calls, does not run a tool if its audit write fails, and a configuration file that becomes invalid refuses every
request until it is fixed.

## Consequences

- Good: a bug, a missing file, a dead database or a bad edit can stop work but cannot silently permit it.
- Bad: availability is traded for safety. With a database configured, a database outage blocks every Claude Code tool call. A typo in the policy file takes the server down until it is fixed.

## Where

`src/portcullis/hook.py`, `plugin/scripts/policy_hook.py`, `plugin/hooks/hooks.json`, `src/portcullis/server.py`, `tests/test_hook_policy.py`
