# ADR-0007: A human approves through one of three channels, each with a stated strength

Status: accepted. Date: 2026-11-19.

## Context

Calls that need a person (irreversible actions, large amounts, sensitive arguments) arrive through different
integrations that offer different channels back to the person.

## Decision

The library agent loop pauses with a LangGraph interrupt and counts only a named, explicit approval. The server lists
pending calls to an `approver` key (a role an agent key cannot hold), takes the approver's name from the key, and shows a
bounded preview of the arguments. The MCP proxy asks the client with an MCP input-required round trip, bound to the exact
call by a signed, expiring, single-use token, and refuses to ask when the arguments are too long to show in full.
Whatever the channel, the rules are checked again when the answer arrives.

## Consequences

- Good: nobody approves blind, and an approval cannot be reused for another call.
- Bad: the MCP channel is only as strong as the client. The answer comes from the client, so a compromised client can fabricate it. For an answer a client cannot fake, use the server's approvals.

## Where

`src/portcullis/agent.py`, `src/portcullis/server.py`, `src/portcullis/mcp_proxy.py`, `tests/test_attacks.py`
