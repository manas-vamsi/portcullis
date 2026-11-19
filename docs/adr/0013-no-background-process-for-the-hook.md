# ADR-0013: The Claude Code hook does not use a background process to go faster

Status: accepted. Date: 2026-11-19.

## Context

The hook runs as a fresh process before every tool call and costs roughly 0.4 to 1.0 seconds each time, mostly starting
Python and importing libraries (measured; the figure varies a lot between runs). A long-running process that the hook
talks to would cut that sharply.

## Decision

Not built. A background process needs a place where the hook finds it (a port and a secret in a file). The agent that
the hook is meant to restrain runs shell commands as the same user, so it could read that file, stop the real process,
start a fake one that answers "allow" and point the hook at it. That would turn a guard into a bypass, to save time.
The hook stays a plain process per call. It still fails closed, and its cost is stated in the README instead of hidden.

## Consequences

- Good: nothing the agent can edit decides whether the agent is allowed to do something, beyond the policy file (which the README already says must be made read-only or owned by another user).
- Bad: every tool call is slower. If the cost matters, a faster-starting implementation of the same logic is the honest fix, and it is not built.

## Where

`src/portcullis/hook.py`, `plugin/scripts/policy_hook.py`, `benchmarks/run.py`
