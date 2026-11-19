# ADR-0009: Who is acting comes from the credential, never from the request

Status: accepted. Date: 2026-11-19.

## Context

A model that can name a tenant in its request can borrow another tenant's authority (a confused deputy).

## Decision

In the server the tenant comes from the API key; bodies with extra fields are refused; run names are namespaced by
tenant. Keys have one role, `agent` or `approver`, and the two never overlap. Keys are random, shown once and stored only
as hashes. In the agent loop the tenant is set by the trusted caller. The attack suite tests a model naming another
tenant and gaining nothing.

## Consequences

- Good: tenants cannot see or answer each other's runs; a hijacked agent key cannot approve.
- Bad: the policy checks the text of an id, not who owns the thing it names. The tool's own backend must check ownership too (a known-gap test pins this).

## Where

`src/portcullis/keys.py`, `src/portcullis/server.py`, `tests/test_keys.py`, `tests/test_server.py`
