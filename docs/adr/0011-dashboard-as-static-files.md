# ADR-0011: The dashboard is plain HTML and JavaScript served by the server

Status: accepted. Date: 2026-11-19.

## Context

A viewer for approvals, the audit log and eval results was wanted. Everything it shows from the log is text a model
chose, so showing it safely matters more than a rich UI.

## Decision

Four static files with no build step and no third-party code, served by the server. A strict Content-Security-Policy (this
server only, no inline script or style, no framing), only `textContent` is used to put text on the page, the API key is held in
memory only, and only the four named files are ever served. Tests ban the DOM calls that would make log text markup, and the
real page was run in a DOM implementation against the real server with a hostile payload.

## Consequences

- Good: nothing to build or patch, a small review surface, no supply chain.
- Bad: it is a thin viewer: no sessions, no live updates beyond a 10-second refresh, and it has not been run in a real browser.

## Where

`src/portcullis/dashboard.py`, `src/portcullis/web/`, `tests/test_dashboard.py`, `tests/js/`
