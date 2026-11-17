# Portcullis

[![CI](https://github.com/manas-vamsi/portcullis/actions/workflows/ci.yml/badge.svg)](https://github.com/manas-vamsi/portcullis/actions/workflows/ci.yml)

**A control layer between AI agents and the tools they use.**

Portcullis sits in front of an agent's tools and decides, with plain deterministic rules, what the agent is allowed to do. It can allow a call, block it, or pause it until a human approves. Every decision is written to a tamper-evident audit log.

> **Status: early development.** The deterministic policy engine (allow, deny or ask), the guarded LangGraph agent loop with human approval, the idempotency ledger and the hash-chained audit log are built and tested. A first version of the Claude Code plugin policy hook is built too. The attack suite runs scripted-model attacks against the policy. A first `agentctl` CLI is built (verify and read an audit chain, dry-run a policy). The server and dashboard are not. A feature is marked as built only when it has tests behind it; everything else is design.

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
| Human approval | Irreversible actions and spend above a threshold wait for an explicit approve or deny | Built in the LangGraph agent loop (pauses with `interrupt`, only a named explicit approval counts, the approver is recorded in state and never shown to the model); approval UI planned |
| Tamper-evident audit log | Hash-chained records of every decision, who approved or denied what, and what ran | Built (one chain per run in Postgres, wired into the agent loop); read and verify with `agentctl`; no dashboard yet |
| Budgets and loop detection | Limits on steps, tokens and time, enforced by the runtime, plus an optional cap on the total amount spent in a run | Built in the LangGraph agent loop; a breach of steps, tokens, time or loops halts the run, and a call that would pass the spend cap is refused while the run carries on |
| Idempotency keys | A crash and resume does not repeat a side effect | Built (Postgres ledger), wired into the LangGraph agent loop |
| Tool definition pinning | Pinned hashes of MCP tool definitions; refuse tools that changed (rug-pull defense) | Built in the agent loop: you supply a function that returns each tool's current definition, and a pinned tool whose hash changed (or that cannot be read) is refused; reading definitions from a real MCP server is planned |
| Crash recovery | Resume a run from a Postgres checkpoint (LangGraph) | Tested with a hard-killed worker, including the repeat-effect cases below |
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

1. **Python library**: a guarded LangGraph agent loop where every tool call goes through the policy, approval and ledger (first version built; see `src/portcullis/agent.py`).
2. **Server**: FastAPI policy engine with a Postgres audit log.
3. **CLI** (`agentctl`, first version built, see below): verify and read an audit chain, dry-run a policy. Approving actions from the CLI is not built.
4. **Skill**: a portable `SKILL.md` folder for coding agents that support the open skill format.
5. **Claude Code plugin** (first version built, see below): a `PreToolUse` hook checks each tool call against the policy and answers deny or ask. It fails closed, so an error blocks the call instead of letting it through.
6. **Dashboard**: a small Next.js app with three pages: approvals, audit log, eval report.

## Demo domain

A property-management maintenance triage agent, built on **synthetic data only** (`src/portcullis/demo.py`). The model is scripted, so the demo needs no LLM and no quota, but the policy engine, approvals, idempotency ledger and audit log are the real ones.

- `check_vendor_availability` (read)
- `create_work_order` (write, can be undone with `cancel_work_order`)
- `notify_tenant` (write, deduplicated by idempotency key)
- Work orders above $500 need human approval.
- Emergency words such as "gas leak" or "on fire" skip the model and escalate to a human through plain keyword rules. The list is short and not exhaustive.

Five scenarios: `routine`, `expensive` (approved), `expensive_denied`, `emergency`, and `attack` (a forbidden tool, then a spend sent as a string to dodge the limit). Each prints its transcript and verifies its audit chain. See "Getting started" to run it.

## Claude Code plugin

The plugin in `plugin/` runs a `PreToolUse` hook before every tool call (`Bash`, `Edit`, `Read`, MCP tools and so on). The policy can look inside arguments with `deny_if` and `ask_if` regex rules.

- **Allowed:** the hook prints nothing, so Claude Code's own permission prompts still apply. Portcullis only narrows what is permitted.
- **Denied:** the call is blocked with exit code 2 and Claude is told the rule that fired, for example `Portcullis [forbidden_argument]: argument 'file_path' matches a forbidden pattern`. Exit 2 is used because a deny sent as JSON would turn into "proceed" if the output were ever truncated or polluted.
- **Ask:** Claude Code asks the user.
- **Any error blocks the call.** Claude Code treats exit code 1 and a hook timeout as non-blocking, and a Python import failure exits 1. So a thin standard-library launcher turns every failure into exit 2, catches errors of any kind, and runs a watchdog that blocks a hung hook before Claude Code's own timeout would let the call through. The hook is also registered with `onFailure: "block"` (Claude Code 2.1.295 or newer), so even a hook that cannot start blocks the call; I checked that with a hook command that does not exist. Tools that are not in the policy are denied, and a rule can mark an argument `required` so a misspelled argument name denies the call instead of silently switching the rule off.
- **Optional audit:** with `PORTCULLIS_DATABASE_URL` set, every decision is written to the hash-chained log before the answer is returned, and a failed write blocks the call. The log holds a hash of the request, not the command or path.

Setup, from a checkout with the package installed (`uv sync`):

```bash
mkdir -p .portcullis && cp plugin/policy.example.json .portcullis/policy.json
export PORTCULLIS_PYTHON="$(pwd)/.venv/Scripts/python.exe"   # the Python that has portcullis (.venv/bin/python on Linux and macOS)
claude --plugin-dir ./plugin
```

`plugin/policy.example.json` blocks reading or writing `.env` files and keys, writing the policy, the hook and Claude Code's settings, `rm -r` on `/` or `~`, piping a download into a shell and force pushes. It asks before a plain `git push`, an edit to `.git` or workflow files, or a shell command that names the policy or settings. It is a starting point, not a complete defence: write your own, and add rules for the MCP tools you use, because anything not listed is denied.

Protect the policy itself. The example denies file tools from writing `.portcullis/` and the hook, but the `Bash` tool can edit any file, so the policy cannot fully protect itself. Make the policy file read-only or owned by another user, which is the real control. For the same reason `Grep` and `Glob` can still surface secrets by searching a directory; keep secrets out of the project folder. Other settings: `PORTCULLIS_POLICY` (policy path) and `PORTCULLIS_TENANT` (default `claude-code`).

Checked with a real Claude Code session on Windows: a read of a normal file went through, a read of a file named `.env` was refused with the reason above, and both decisions appeared in the audit chain.

## Attack suite

`tests/test_attacks.py` scripts a model that has been fully hijacked and checks what still gets through. The dangerous tools are implemented and counted, so "never ran" is measured. **No real LLM is called**, so this shows what the deterministic layer does when a model is fooled, not whether a real model resists these attacks.

| Attack | What the tests show | Gap |
|---|---|---|
| Prompt injection in a tool result | The injected text reaches the model; calls to tools off the allowlist are refused and never run. A call that claims `approved: true` still waits for a human, and only a named explicit answer approves. A repeat loop is halted. A secret sent through an allowed tool waits for approval. Spend disguised as a string, a bool, a negative or NaN is refused. | Five orders of 499 (each under the $500 approval line) are stopped at the fifth by a `max_spend` cap of 2000. **The cap is per run:** a second run starts at zero, so two runs can spend 3992. Without a cap, all five go through. |
| Poisoned tool description | A model that obeys a poisoned description still cannot call unlisted tools. A description changed after review is refused by its pin. | **A description that was already poisoned when you pinned it passes the pin.** Review the text before pinning. |
| Rug pull (definition changes later) | A definition that changes mid-run is refused from the next call. One that changes while a call waits for approval stops the call. | Definitions come from a function you supply; reading them from a real MCP server is not built. |
| Confused deputy | The acting tenant comes from the caller, so a model naming another tenant gains nothing. A work order for another tenant's vendor is refused, including with a newline, a leading space, a list or a missing id. | **The policy checks the text of an id, not who owns it.** The tool's own backend must check ownership too. |

Tests named `test_known_gap__...` pin the weaknesses above so they cannot change or be forgotten silently. Run it with `uv run --env-file .env pytest tests/test_attacks.py -v` (it needs the Postgres container). Claude Code hook attacks (editing the policy, `.env` evasions, hook failures) are covered in `tests/test_hook_policy.py`, and a cross-tenant replay of the idempotency ledger in `tests/test_agent.py`.

## agentctl

A small read-only command line for the audit log and the policy (`uv run agentctl --help`).

```bash
uv run --env-file .env agentctl verify RUN_ID --rows 12 --head <hash>   # is the chain intact?
uv run --env-file .env agentctl show RUN_ID                             # one JSON object per row, then the chain status
uv run agentctl check plugin/policy.example.json --tenant claude-code --tool Read --args '{"file_path": ".env"}'
```

- **verify** exits 0 if the chain is intact and 1 if a row was changed, removed from the middle or reordered. A chain that lost its last rows is still self-consistent, so pass `--rows` and `--head` from a copy you kept elsewhere. A run with no rows fails unless you pass `--rows 0`.
- **show** prints every row, then the chain status. Control characters in logged values are escaped, so a logged value cannot inject terminal escapes.
- **check** runs one call through a policy file and prints `allow`, `deny` or `ask` with the rule. Exit codes are 0, 1 and 3. It needs no database and starts every call with a fresh state, so budgets and loop limits are not exercised. Put arguments in a file with `--args @args.json` if shell quoting gets in the way.
- Exit 2 means the command itself failed (bad input, no database), not that the log is bad. The database comes only from `PORTCULLIS_DATABASE_URL`, over a read-only connection, so the tool cannot change the log it inspects. Errors never print the URL.

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

- The approval threshold is checked per call, on one named argument. The optional `max_spend` cap adds up the amounts of calls that ran in one run, and not even a human approval can push a run past it. It is per run, so a new run starts at zero; it counts only tools that declare an `approve_above` amount argument; and the Claude Code hook is stateless, so it has no cap.
- Loop detection compares the tool name and arguments. A model can vary the arguments to avoid it. The step budget is the hard stop.
- Delivery is at-least-once, not exactly-once. If the process dies after a tool call but before the ledger records it, the call is retried with the same idempotency key. The outside service must deduplicate by that key; one that ignores keys can see one duplicate. A test pins this.
- The audit log is tamper-evident, not tamper-proof. Editing or deleting a row inside a run's chain is detected. Deleting the last rows or a whole run is not, unless you keep the head hash somewhere else. Someone with database write access can rewrite a whole chain. It stores a hash of each request, not the arguments.
- The Claude Code hook is stateless: each call is a separate process, so budgets and loop detection do not apply to it yet. An `ask` is answered inside Claude Code, which does not tell the hook what the user chose, so approvals are not recorded there.
- The hook adds delay to every tool call. Measured on one Windows laptop (i7-1355U), a call takes about 0.5 to 0.8 seconds, most of it importing Pydantic; importing the database driver is skipped unless auditing is on. A faster implementation (the plan mentions a Rust CLI) is not built, and these numbers are from one machine.
- Argument rules are regexes written by the policy author and run on text the model controls. Python's `re` has no timeout, so avoid nested quantifiers. Arguments over 10,000 characters are denied instead of inspected. A pattern list is only as good as its author.
- An empty audit chain and a missing one look the same to `verify()`, so check the row count or head hash against what you expect. Reusing an event id for different content raises an error instead of being deduplicated.
- A call that keeps failing is retried on every resume. There is no per-call retry cap yet.
- The time budget is checked when each tool step starts, so one slow step can overshoot it, and tool time spent before an approval pause is not counted after the pause.
- LangGraph's checkpoint serializer rejects integers beyond 64 bits and very deep nesting, so the agent loop accepts only plain JSON tool arguments (string keys, 64-bit integers, at most 32 levels deep) and treats anything else as an invalid call.
- All malformed tool calls share one loop fingerprint, so a run with `max_identical_calls` of them is halted as a loop.
- The ledger call id is the message and call position in the checkpointed history, not the model's tool-call id, so a model that reuses ids cannot collide with the ledger.
- A pin proves a tool's definition is the one you reviewed, not that it is safe or that the server behind it does what the definition says. If a definition changes while a call waits for approval, the call is stopped with an `AuditConflict` error rather than run. Definitions are not read from a real MCP server yet.
- The engine trusts its caller for the tenant and for the hash of the live tool definition. These must come from the runtime (authenticated identity, the real MCP definition), never from model output.

## Stack

Python 3.12, uv, LangGraph, Postgres (Docker), FastAPI, Pydantic v2, SQLAlchemy 2, Alembic, OpenTelemetry, pytest, Hypothesis, GitHub Actions. Dashboard in Next.js. Everything runs locally and needs no paid service.

## Roadmap

1. Policy engine, fake LLM, tests (done; server around the engine still to do)
2. LangGraph integration, checkpoint and resume, approvals, budgets
3. Hash-chained audit log (done), dashboard
4. MCP surface, CLI and skill, code sandbox
5. Eval gate, recorded LLM responses, three-surface comparison
6. Claude Code plugin and demo
7. Executable attack suite (done: see "Attack suite"; it also lists the gaps it found)
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
uv run --env-file .env python -m portcullis.demo                      # all five demo scenarios
uv run --env-file .env python -m portcullis.demo --scenario expensive  # one of them
```

The container listens on `127.0.0.1:5433` only. Without `.env`, the Postgres tests are skipped. The test setup refuses to run against any database whose name does not start with `portcullis`.

Run `uv run agentctl --help` for the CLI. The test suite and the demo are the main things to run so far. This section will grow with the code.

## Data and licensing notes

All demo data is synthetic. No employer code or data is used. Licensed under the MIT License (see `LICENSE`).
