"""A LangGraph agent loop whose every tool call passes through portcullis.

agent (model) -> tools (policy, approval, ledger) -> agent ... until the model stops asking
for tools or the run is halted.

Trust: `tenant_id` and `run_id` in the state must be set by the trusted caller, never taken
from model output. The model controls each tool call's id, name and arguments and the message
text. Token counts come from the llm adapter's usage data, not from model text. Run ids should
be unique per run; the ledger is also scoped by tenant, so two tenants never see each other's
stored results.

Each call is decided by policy.decide, including malformed calls and calls to tools that are
not allowed, so budgets and loop detection cover every attempt. A breach halts the run. ASK
pauses the graph with interrupt() until a human answers {"approved": True, "by": "<name>"};
anything else counts as a denial. Approvals are recorded in state["approvals"] and never sent
to the model, which sees only a rule code for a refusal. Allowed and approved calls run through
the idempotency ledger, and each one that ran adds its amount to state["spend"], which the policy's
optional max_spend cap is checked against. Its call id is the message and call position in the checkpointed history
(m3-c0), never the model's own tool-call id, because models can reuse ids.

Every call is written to the hash-chained audit log (portcullis.audit): the decision before
anything runs, so if the audit write fails the tool does not run, then any human answer, then
the execution with its idempotency key. Event ids come from the call position, so the repeat
when LangGraph re-runs an interrupted node is recorded once.

Only validated tool calls are stored in the checkpointed state, and their arguments must be
plain JSON data: string keys, integers within 64 bits, at most 32 levels deep. Anything else
(which could crash the checkpoint write or the loop check) makes the call invalid instead.

When the human answer arrives, LangGraph re-runs the interrupted node from the top. Decisions
are computed from the state at the start of the node, so the re-run decides identically, and
the ledger stops calls that already ran from running again.

Known limits: the time budget is checked when each tool step starts, so one slow step can
overshoot it, and tool time spent before an approval pause is not counted after the pause.
Pinned tool definitions: pass `definitions`, a function from tool name to the definition as it is
advertised right now (or None). It is asked again for every call, so a definition that changes
mid-run (a rug pull) is caught on its next call. A pinned tool is refused when the hash differs,
when no `definitions` function is given, or when it fails or returns None. If a definition changes
while a call waits for approval, the resume decides the call again, logs a different definition
hash for the same call, and the audit log refuses it with AuditConflict, so the tool does not run.
That happens for any tool with a definition, pinned or not. The pin proves the definition is the
one you reviewed, not that the server behind it does what the definition says.
"""

import json
import time
from collections.abc import Callable, Mapping
from typing import Any, TypedDict

import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from portcullis.audit import AuditTrail
from portcullis.idempotency import idempotency_key, run_once
from portcullis.jsondata import MAX_INT, check_json
from portcullis.llm import LLM
from portcullis.policy import (
    HALTING_RULES,
    Decision,
    Policy,
    RunState,
    ToolCall,
    Verdict,
    decide,
    definition_hash,
    fingerprint,
    spend_amount,
)

Tool = Callable[[dict[str, Any], str], Any]  # (args, idempotency key) -> JSON-compatible result
Definitions = Callable[[str], Mapping[str, Any] | None]  # tool name -> its definition as advertised now
# Stands in for a malformed call so it still counts against the budget. All malformed calls share
# one fingerprint, so max_identical_calls of them in a run is treated as a loop.
INVALID_TOOL = "<invalid-tool-call>"
class AgentState(TypedDict):
    tenant_id: str
    run_id: str
    messages: list[dict[str, Any]]
    steps: int
    tokens: int
    spend: float  # total amount of the calls that ran; a float so the checkpoint never meets a huge int
    seconds: float  # model time plus tool time of a node's final pass; waiting for a human is not counted
    history: list[str]  # fingerprints of the calls made so far
    approvals: list[dict[str, Any]]  # who approved which call; never sent to the model
    halted: str | None  # the rule that stopped the run, if any


Node = Callable[[AgentState], dict[str, Any]]


class ModelToolCall(BaseModel):
    """What the model asked for. Strict, because the model is untrusted."""

    model_config = ConfigDict(frozen=True, strict=True)

    id: str = Field(min_length=1)
    name: str = Field(max_length=128)  # the name is written to the audit log, so it must be bounded
    args: dict[str, Any]

    @field_validator("args")
    @classmethod
    def _args_must_be_safe_to_store(cls, args: dict[str, Any]) -> dict[str, Any]:
        check_json(args)  # a ValueError here becomes a ValidationError, so the call is invalid
        return args


def initial_state(tenant_id: str, run_id: str, user_message: str) -> AgentState:
    return {
        "tenant_id": tenant_id,
        "run_id": run_id,
        "messages": [{"role": "user", "content": user_message}],
        "steps": 0,
        "tokens": 0,
        "spend": 0.0,
        "seconds": 0.0,
        "history": [],
        "approvals": [],
        "halted": None,
    }


def _parse(raw: object) -> ModelToolCall | None:
    try:
        return ModelToolCall.model_validate(raw)
    except ValidationError:
        return None


def _raw_id(raw: object, index: int) -> str:
    if isinstance(raw, dict) and isinstance(raw.get("id"), str) and raw["id"]:
        return raw["id"]
    return f"invalid-{index}"


def _stored_call(raw: object, index: int) -> dict[str, Any]:
    """What goes into the checkpointed message: the validated call, or a stub that keeps only the id."""
    call = _parse(raw)
    return call.model_dump() if call else {"id": _raw_id(raw, index), "invalid": True}


def _tool_message(call_id: str, content: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(content, default=str)}


def _refusal(call_id: str, rule: str) -> dict[str, Any]:
    """The model is told the rule code only: no limits, tenant names or other details."""
    return _tool_message(call_id, {"error": rule})


def _approver(state: AgentState, call: ModelToolCall, position_id: str, decision: Decision) -> tuple[bool, str | None]:
    """Pause until a human answers. Returns (approved, name).

    Approved only for an explicit True from a named person. The name is returned for a denial too,
    so the audit log shows who said no.
    """
    answer = interrupt(
        {
            "tenant_id": state["tenant_id"],
            "run_id": state["run_id"],
            "ledger_call_id": position_id,
            "call_id": call.id,
            "tool": call.name,
            "args": call.args,
            "rule": decision.rule,
            "reason": decision.reason,
        }
    )
    name = answer.get("by") if isinstance(answer, dict) else None
    name = name.strip() if isinstance(name, str) and name.strip() else None
    approved = isinstance(answer, dict) and answer.get("approved") is True and name is not None
    return approved, name


def _ledger_call_id(state: AgentState, position_id: str) -> str:
    return json.dumps([state["tenant_id"], position_id])  # scoped by tenant


def _execute(
    conn: psycopg.Connection, tool: Tool, state: AgentState, position_id: str, call: ModelToolCall
) -> dict[str, Any]:
    ledger_call_id = _ledger_call_id(state, position_id)
    result = run_once(conn, state["run_id"], ledger_call_id, call.name, call.args, lambda key: tool(call.args, key))
    return _tool_message(call.id, {"result": result})


def _live_hash(definitions: Definitions | None, name: str) -> str | None:
    """Hash of the definition advertised right now. Any failure counts as no definition, so a pinned tool is refused."""
    if definitions is None:
        return None
    try:
        definition = definitions(name)
        return None if definition is None else definition_hash(definition)
    except Exception:
        return None


def _log_decision(
    trail: AuditTrail,
    position_id: str,
    name: str,
    args: dict[str, Any],
    decision: Decision,
    definition_sha256: str | None = None,
) -> None:
    extra = {} if definition_sha256 is None else {"definition_sha256": definition_sha256}
    trail.record(
        f"{position_id}:decision",
        "decision",
        {
            "call_id": position_id,
            "tool": name,
            "verdict": decision.verdict.value,
            "rule": decision.rule,
            "reason": decision.reason,
            "request_sha256": fingerprint(name, args),  # a hash, not the arguments, which may be sensitive
            **extra,
        },
    )


def _answer(
    state: AgentState,
    conn: psycopg.Connection,
    tools: Mapping[str, Tool],
    call: ModelToolCall | None,
    decision: Decision,
    position_id: str,
    message_id: str,
    trail: AuditTrail,
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    """The tool message for one call, the approval record if a human approved it, and whether the tool ran."""
    if decision.rule in HALTING_RULES:
        return _refusal(message_id, decision.rule), None, False
    if call is None:
        return _refusal(message_id, "invalid_tool_call"), None, False
    if decision.verdict is Verdict.DENY:
        return _refusal(message_id, decision.rule), None, False
    tool = tools.get(call.name)
    if tool is None:
        trail.record(
            f"{position_id}:refused",
            "refused",
            {"call_id": position_id, "tool": call.name, "rule": "tool_unavailable"},
        )
        return _refusal(message_id, "tool_unavailable"), None, False
    approval = None
    if decision.verdict is Verdict.ASK:
        approved, by = _approver(state, call, position_id, decision)
        trail.record(
            f"{position_id}:approval",
            "approval",
            {"call_id": position_id, "tool": call.name, "approved": approved, "approved_by": by},
        )
        if not approved:
            return _refusal(message_id, "approval_denied"), None, False
        approval = {"call_id": position_id, "tool": call.name, "args": call.args, "approved_by": by}
    try:
        message = _execute(conn, tool, state, position_id, call)
    except Exception as exc:
        # The outcome is unknown, so the retry may run it again; the log shows both attempts.
        trail.record(
            f"{position_id}:failed", "failed", {"call_id": position_id, "tool": call.name, "error": type(exc).__name__}
        )
        raise
    key = idempotency_key(state["run_id"], _ledger_call_id(state, position_id))
    executed = {"call_id": position_id, "tool": call.name, "idempotency_key": key}
    trail.record(f"{position_id}:executed", "executed", executed)
    return message, approval, True


def _tokens(reply: dict[str, Any]) -> int:
    tokens = reply.get("tokens")  # required: a missing count would silently switch the token budget off
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 0 <= tokens <= MAX_INT:
        raise ValueError("llm reply 'tokens' must be an int from 0 to 2**64-1, taken from the adapter's usage data")
    return tokens


def make_agent_node(llm: LLM) -> Node:
    def agent_node(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        reply = llm(state["messages"])
        calls = reply.get("tool_calls") or []
        if not isinstance(calls, list):
            calls = [calls]  # refused later as one invalid call, never silently dropped
        content = reply.get("content")
        message = {
            "role": "assistant",
            "content": content if isinstance(content, str) else "",
            "tool_calls": [_stored_call(raw, index) for index, raw in enumerate(calls)],
        }
        return {
            "messages": [*state["messages"], message],
            "tokens": state["tokens"] + _tokens(reply),
            "seconds": state["seconds"] + (time.perf_counter() - started),
        }

    return agent_node


def make_tools_node(
    policy: Policy, tools: Mapping[str, Tool], conn: psycopg.Connection, definitions: Definitions | None = None
) -> Node:
    def tools_node(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        position = len(state["messages"]) - 1
        messages, history, approvals = list(state["messages"]), list(state["history"]), list(state["approvals"])
        steps, spend, halted = state["steps"], state["spend"], None
        trail = AuditTrail(conn, state["run_id"], state["tenant_id"])

        for index, raw in enumerate(state["messages"][-1]["tool_calls"]):
            message_id, position_id = _raw_id(raw, index), f"m{position}-c{index}"
            call = _parse(raw)
            name, args = (call.name, call.args) if call else (INVALID_TOOL, {})
            if halted:
                skipped = Decision(verdict=Verdict.DENY, rule="run_halted", reason="the run was halted earlier")
                _log_decision(trail, position_id, name, args, skipped)
                messages.append(_refusal(message_id, "run_halted"))
                continue
            run_state = RunState(
                steps=steps, tokens=state["tokens"], seconds=state["seconds"], history=tuple(history), spend=spend
            )
            seen = _live_hash(definitions, name) if call else None
            tool_call = ToolCall(tenant_id=state["tenant_id"], tool=name, args=args, definition_sha256=seen)
            decision = decide(policy, tool_call, run_state)
            _log_decision(trail, position_id, name, args, decision, seen)  # written before anything runs
            steps += 1
            history.append(fingerprint(name, args))

            message, approval, ran = _answer(state, conn, tools, call, decision, position_id, message_id, trail)
            if ran:
                spend += float(spend_amount(policy, tool_call))
            messages.append(message)
            if approval:
                approvals.append(approval)
            halted = decision.rule if decision.rule in HALTING_RULES else None

        return {
            "messages": messages,
            "steps": steps,
            "spend": spend,
            "history": history,
            "approvals": approvals,
            "halted": halted,
            "seconds": state["seconds"] + (time.perf_counter() - started),
        }

    return tools_node


def _after_agent(state: AgentState) -> str:
    return "tools" if state["messages"][-1]["tool_calls"] else END


def _after_tools(state: AgentState) -> str:
    return END if state["halted"] else "agent"


def build_agent(
    llm: LLM,
    policy: Policy,
    tools: Mapping[str, Tool],
    conn: psycopg.Connection,
    checkpointer: BaseCheckpointSaver,
    definitions: Definitions | None = None,
) -> CompiledStateGraph:
    """Compile the guarded agent. A checkpointer is required: approvals pause the run with interrupt()."""
    graph = StateGraph(AgentState)
    graph.add_node("agent", make_agent_node(llm))
    graph.add_node("tools", make_tools_node(policy, tools, conn, definitions))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _after_agent, ["tools", END])
    graph.add_conditional_edges("tools", _after_tools, ["agent", END])
    return graph.compile(checkpointer=checkpointer)
