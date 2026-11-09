# Portcullis

**A control layer between AI agents and the tools they use.**

Portcullis sits in front of an agent's tools and decides, with plain deterministic rules, what the agent is allowed to do. It can allow a call, block it, or pause it until a human approves. Every decision is written to a tamper-evident audit log.

> **Status: early development.** The deterministic policy engine (allow, deny or ask) is built and tested. The server, audit log, approval flow, dashboard and plugin policy are not. A feature is marked as built only when it has tests behind it; everything else is design.

---

## The problem

Agents are no longer only chatting. They create tickets, send messages, move money and change records. Most teams protect these actions with prompt instructions such as "never do X without asking". A prompt is a request to the model, not a guarantee. The model can ignore it, misread it, or be tricked by text inside a tool result.

When an agent can take real actions, teams need answers to questions a prompt cannot give:

- Is this tenant or user even allowed to call this tool?
- Did a human approve this irreversible action, and who?
- If the agent crashed halfway, did the retry send the same email twice?
- Did the agent get stuck in a loop and burn the budget?
- Did a tool quietly change its description after we approved it?
- If something went wrong, can we show exactly what happened?

## What Portcullis does

Portcullis is a normal server plus a library. It is not an agent, and no LLM makes its safety decisions.

| Capability | What it means | Status |
|---|---|---|
| Tool allowlists and risk levels | Per-tenant rules; each tool is `read`, `write` or `irreversible` | Built (policy engine) |
| Human approval | Irreversible actions and spend above a threshold wait for an explicit approve or deny | Engine returns `ask`; approval flow planned |
| Tamper-evident audit log | Hash-chained records of who approved what and when | Planned |
| Budgets and loop detection | Limits on steps, tokens and time, enforced by the runtime | Decision logic built; runtime counters planned |
| Idempotency keys | A crash and resume does not repeat a side effect | Planned (a test marked expected-fail pins the gap) |
| Tool definition pinning | Pinned hashes of MCP tool definitions; refuse tools that changed (rug-pull defense) | Hash check built; reading MCP definitions planned |
| Crash recovery | Resume a run from a Postgres checkpoint (LangGraph) | Resume after a kill is tested; a step killed mid-way still repeats its side effect until idempotency lands |
| Tracing | Every run can be inspected end to end (OpenTelemetry) | Planned |
| Eval gate | Block a release if the agent got measurably worse | Planned |

## How it works

```
 Agent (LangGraph, Claude Code, any MCP client)
        |
        |  "call tool X with these arguments"
        v
 +------------------------------------------+
 |  Portcullis policy engine (deterministic) |
 |   1. Is the tool allowed for this tenant? |
 |   2. What is its risk level?              |
 |   3. Is it inside budget? Is it a loop?   |
 |   4. Has the tool definition changed?     |
 +------------------------------------------+
        |
        +--> allow ------> run the tool
        +--> deny  ------> return a clear refusal
        +--> ask   ------> wait for a human, then allow or deny
        |
        v
 Hash-chained audit log (Postgres)
```

The same policy engine is reached through several entry points, so the rules are written once and apply everywhere.

## Ways to use it (planned)

1. **Python library**: wrap a LangGraph agent's tool calls.
2. **Server**: FastAPI policy engine with a Postgres audit log.
3. **CLI** (`agentctl`): inspect runs, approve actions, verify the audit chain.
4. **Skill**: a portable `SKILL.md` folder for coding agents that support the open skill format.
5. **Claude Code plugin**: a `PreToolUse` hook sends each tool call to the server and returns allow, deny or ask. The hook is designed to fail closed, so an error blocks the call instead of letting it through.
6. **Dashboard**: a small Next.js app with three pages: approvals, audit log, eval report.

## Demo domain

A property-management maintenance triage agent, built on **synthetic data only**.

- `check_vendor_availability` (read)
- `create_work_order` (write, can be undone with `cancel_work_order`)
- `notify_tenant` (idempotent)
- Spend above a threshold needs human approval.
- Emergency words such as "gas leak" skip the model and escalate to a human deterministically.

## Research question

MCP servers, CLI plus skills, and code execution are three common ways to give an agent tools. The plan is to expose the same tools all three ways under one policy layer and measure, on the demo agent: pass^k, tokens, tool calls, latency, cost and attacks blocked. **No results exist yet.** Numbers will be published only after they are measured.

## Design principles

- **Deterministic safety.** Policy decisions never depend on an LLM.
- **Fail closed.** When Portcullis cannot decide, the answer is no.
- **At-least-once plus idempotency.** This is not true exactly-once delivery, and the docs will not claim it is.
- **Tested without a model.** Tests use a scripted fake LLM. LLM responses can be recorded and replayed, so reruns are free and repeatable.
- **Honest scope.** Commercial MCP gateways and observability platforms already exist. This project is a clear, tested reference for how these controls are designed and built, not a claim to replace them.

## Who it is for

Teams that run agents with real side effects and need control and evidence: multi-tenant products that embed agents for each customer, regulated businesses that must show an audit trail, and organizations rolling out coding agents with MCP tools. It is less useful for a single developer running an agent on their own machine.

## Known limits of the policy engine (today)

- The spend threshold is checked per call, on one named argument. There is no cumulative spend cap yet, so many small calls can add up.
- Loop detection compares the tool name and arguments. A model can vary the arguments to avoid it. The step budget is the hard stop.
- The engine trusts its caller for the tenant and for the hash of the live tool definition. These must come from the runtime (authenticated identity, the real MCP definition), never from model output.

## Stack

Python 3.12, uv, LangGraph, Postgres (Docker), FastAPI, Pydantic v2, SQLAlchemy 2, Alembic, OpenTelemetry, pytest, Hypothesis, GitHub Actions. Dashboard in Next.js. Everything runs locally and needs no paid service.

## Roadmap

1. Policy engine, fake LLM, tests (done; server around the engine still to do)
2. LangGraph integration, checkpoint and resume, approvals, budgets
3. Hash-chained audit log, dashboard
4. MCP surface, CLI and skill, code sandbox
5. Eval gate, recorded LLM responses, three-surface comparison
6. Claude Code plugin and demo
7. Executable attack suite (poisoned tool description, rug pull, confused deputy, prompt injection in a tool result)
8. Benchmarks, architecture decision records, docs

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and Docker (Docker is needed once the Postgres service lands).

```bash
git clone https://github.com/manas-vamsi/portcullis.git
cd portcullis
uv sync
uv run pytest        # unit tests, no database needed
```

The Postgres tests (checkpoint and kill-and-resume) need the database container:

```bash
cp .env.example .env     # then put the same real password in both lines
docker compose up -d --wait
uv run --env-file .env pytest
```

The container listens on `127.0.0.1:5433` only. Without `.env`, the Postgres tests are skipped. The test setup refuses to run against any database whose name does not start with `portcullis`.

The test suite is the only thing to run so far. This section will grow with the code.

## Data and licensing notes

All demo data is synthetic. No employer code or data is used. A license has not been chosen yet.
