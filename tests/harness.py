"""Shared helpers for tests of the guarded agent: a scripted model and tools that count how often they run."""

import json
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any

import psycopg
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from portcullis.agent import build_agent, initial_state
from portcullis.llm import FakeLLM
from portcullis.policy import Budget, Policy, Risk, TenantPolicy, Threshold, ToolRule, definition_hash

IMPLEMENTED_TOOLS = ("check_vendor_availability", "create_work_order", "drop_database")


def make_policy(**budget) -> Policy:
    limits = {"max_steps": 10, "max_tokens": 10_000, "max_seconds": 60, "max_identical_calls": 3} | budget
    tools = {
        "check_vendor_availability": ToolRule(risk=Risk.READ),
        "create_work_order": ToolRule(risk=Risk.WRITE, approve_above=Threshold(arg="cost", limit=500)),
        "notify_tenant": ToolRule(risk=Risk.READ),  # allowed by policy but not implemented below
    }
    return Policy(tenants={name: TenantPolicy(budget=Budget(**limits), tools=tools) for name in ("acme", "globex")})


def tool_call(call_id: str, name: str, **args) -> dict:
    return {"id": call_id, "name": name, "args": args}


def reply(*calls, content: str = "", tokens: int = 0) -> dict:
    return {"content": content, "tool_calls": list(calls), "tokens": tokens}


class Harness:
    """A guarded agent over a scripted model and tools that count how often they really run."""

    def __init__(
        self,
        conn: psycopg.Connection,
        run_id: str,
        script: list[dict],
        policy: Policy | None = None,
        fail_once: tuple[str, ...] = (),
        tool_names: tuple[str, ...] = IMPLEMENTED_TOOLS,
        definitions: Callable[[str], Mapping[str, Any] | None] | None = None,
        results: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.llm = FakeLLM(script)
        self.runs: Counter = Counter()
        self.keys: dict[str, list[str]] = {}
        self._fail_once = set(fail_once)
        self._results = dict(results or {})  # what a tool returns instead of {'tool': name}
        tools = {name: self._tool(name) for name in tool_names}
        self.agent = build_agent(self.llm, policy or make_policy(), tools, conn, InMemorySaver(), definitions)
        self.config = {"configurable": {"thread_id": run_id}}

    def _tool(self, name: str):
        def run(args: dict, key: str) -> dict:
            self.keys.setdefault(name, []).append(key)
            if name in self._fail_once:
                self._fail_once.discard(name)
                raise RuntimeError("provider down")
            self.runs[name] += 1
            return self._results.get(name, {"tool": name})

        return run

    def start(self, tenant: str = "acme", **overrides) -> dict:
        state = initial_state(tenant, self.run_id, "leak in unit 4") | overrides
        return self.agent.invoke(state, self.config)

    def resume(self, answer: object) -> dict:
        return self.agent.invoke(Command(resume=answer), self.config)


def audit_events(conn, run_id: str) -> list[tuple[str, dict]]:
    rows = conn.execute("SELECT event, payload FROM audit_log WHERE run_id = %s ORDER BY idx", (run_id,)).fetchall()
    return [(event, json.loads(payload)) for event, payload in rows]


def tool_results(state: dict) -> list[dict]:
    return [json.loads(m["content"]) for m in state["messages"] if m["role"] == "tool"]


VENDOR_TOOL = {"name": "check_vendor_availability", "description": "Look up vendor availability by trade.", "schema": {"trade": "string"}}


def pinned_policy(definition: dict[str, Any] = VENDOR_TOOL, **budget) -> Policy:
    """make_policy(), but check_vendor_availability is pinned to this reviewed definition."""
    base = make_policy(**budget)
    tenants = {}
    for name, tenant in base.tenants.items():
        rule = tenant.tools["check_vendor_availability"].model_copy(update={"pinned_sha256": definition_hash(definition)})
        tenants[name] = tenant.model_copy(update={"tools": {**tenant.tools, "check_vendor_availability": rule}})
    return Policy(tenants=tenants)
