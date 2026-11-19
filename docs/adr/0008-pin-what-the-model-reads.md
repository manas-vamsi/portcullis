# ADR-0008: Tool definitions are pinned by a hash of what the model reads, and re-read on every call

Status: accepted. Date: 2026-11-19.

## Context

A tool server can change a tool's description after you reviewed it (a rug pull) to inject instructions.

## Decision

A policy can pin the SHA-256 of a tool's name, description and input schema. The proxy fetches the definition fresh
for each call and refuses a mismatch; it lists a pinned tool only while it matches, and lists each tool with only those
fields (titles, annotations, icons and metadata are dropped, because they could change unnoticed). `pin` prints the
text to review next to its hash.

## Consequences

- Good: a change in the middle of a session is caught on the next call; also while a call waits for approval.
- Bad: a pin proves "this is what was reviewed", not "this is safe": a description that was already poisoned when pinned passes (a known-gap test pins this). It cannot tie the forwarded call to the definition that was hashed; a swap between the check and the call is only caught on the next call.

## Where

`src/portcullis/mcp_proxy.py`, `src/portcullis/policy.py`, `tests/test_mcp_proxy.py`, `tests/test_attacks.py`
