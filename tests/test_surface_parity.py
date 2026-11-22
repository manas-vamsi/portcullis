"""One policy, three entry points: the same call gets the same verdict and rule from each.

Each case is sent, on fresh state, to the policy engine itself, the Claude Code hook, the HTTP
server's Gateway and the MCP proxy's Governor. If any surface disagrees with the engine, it has
drifted from the shared rules.

Where the surfaces differ on purpose, a test below says so (see the end of this file).
The server needs the Postgres container, like the other server tests.
"""

import asyncio
import json
import re
from uuid import uuid4

import pytest
from mcp import types

from portcullis import hook
from portcullis.mcp_proxy import Governor, definition_sha
from portcullis.policy import (
    ArgPattern,
    Budget,
    Policy,
    Risk,
    RunState,
    TenantPolicy,
    Threshold,
    ToolCall,
    ToolRule,
    decide,
)
from portcullis.server import Gateway, open_pool

TENANT = "parity"
TOOLS = ("lookup", "create_order", "notify", "export_all")  # export_all exists but no policy names it

POLICY = Policy(
    tenants={
        TENANT: TenantPolicy(
            budget=Budget(max_steps=20, max_tokens=10_000, max_seconds=600, max_identical_calls=5),
            tools={
                "lookup": ToolRule(risk=Risk.READ),
                "create_order": ToolRule(
                    risk=Risk.WRITE,
                    approve_above=Threshold(arg="cost", limit=500),
                    deny_if=(ArgPattern(arg="vendor", regex=r"^(?!V-[0-9]+$)", required=True),),
                ),
                "notify": ToolRule(risk=Risk.READ, ask_if=(ArgPattern(arg="message", regex=r"(?i)password\s*="),)),
            },
        )
    }
)

# (tenant, tool, args, verdict, rule). The rule of an "ask" is not checked on MCP, which hides it from the model.
CASES = [
    (TENANT, "lookup", {"trade": "plumber"}, "allow", None),
    (TENANT, "export_all", {}, "deny", "tool_not_allowed"),
    (TENANT, "create_order", {"vendor": "V-1", "cost": 100}, "allow", None),
    (TENANT, "create_order", {"vendor": "V-1", "cost": 900}, "ask", None),
    (TENANT, "create_order", {"vendor": "evil-1", "cost": 100}, "deny", "forbidden_argument"),
    (TENANT, "create_order", {"cost": 100}, "deny", "missing_argument"),
    (TENANT, "create_order", {"vendor": "V-1", "cost": "900"}, "deny", "invalid_amount"),
    (TENANT, "notify", {"message": "hi"}, "allow", None),
    (TENANT, "notify", {"message": "password=hunter2"}, "ask", None),
    ("nobody", "lookup", {}, "deny", "unknown_tenant"),
]
IDS = [f"{tool}-{verdict}-{i}" for i, (_, tool, _, verdict, _) in enumerate(CASES)]


# --- each surface, reduced to (verdict, rule) ---


def via_engine(tenant, tool, args):
    decision = decide(POLICY, ToolCall(tenant_id=tenant, tool=tool, args=args), RunState())
    return decision.verdict.value, None if decision.verdict.value == "allow" else decision.rule  # hook and MCP name no rule on allow


def via_hook(tenant, tool, args, policy_file):
    event = {"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": args, "session_id": "s"}
    code, out, err = hook.run(json.dumps(event).encode(), {"PORTCULLIS_POLICY": str(policy_file), "PORTCULLIS_TENANT": tenant})
    if code == 2:
        return "deny", re.search(r"\[(\w+)\]", err).group(1)
    if not out:
        return "allow", None
    reason = json.loads(out)["hookSpecificOutput"]["permissionDecisionReason"]
    return "ask", re.search(r"\[(\w+)\]", reason).group(1)


SERVER_VERDICT = {"allowed": "allow", "denied": "deny", "pending_approval": "ask"}


def via_server(tenant, tool, args, gateway, run_id):
    shown = gateway.decide_call(tenant, run_id, "c1", tool, args, None)
    verdict = SERVER_VERDICT[shown["status"]]
    return verdict, None if verdict == "allow" else shown["rule"]


class Downstream:
    """Advertises every tool in TOOLS and counts what reaches it."""

    def __init__(self):
        self.calls = []

    async def list_tools(self, *, cursor=None, meta=None):
        schema = {"type": "object"}
        return types.ListToolsResult(tools=[types.Tool(name=n, description=n, input_schema=schema) for n in TOOLS])

    async def call_tool(self, name, arguments=None, **kwargs):
        self.calls.append(name)
        return types.CallToolResult(content=[types.TextContent(type="text", text="ran")])


def via_mcp(tenant, tool, args):
    downstream = Downstream()
    governor = Governor(POLICY, tenant, downstream)
    result = asyncio.run(governor.call_tool(None, types.CallToolRequestParams(name=tool, arguments=args)))
    if isinstance(result, types.InputRequiredResult):
        assert downstream.calls == []
        return "ask", None
    if downstream.calls == [tool]:
        return "allow", None
    assert downstream.calls == [] and result.is_error
    text = " ".join(c.text for c in result.content)
    found = re.search(r"Refused by policy: (\w+)", text)
    return "deny", found.group(1) if found else text


# --- fixtures ---


@pytest.fixture(scope="module")
def policy_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("parity") / "policy.json"
    path.write_text(POLICY.model_dump_json(), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def pool(postgres_url):
    opened = open_pool(postgres_url)
    yield opened
    opened.close()


@pytest.fixture(scope="module")
def gateway(pool):
    return Gateway(POLICY, pool)


@pytest.fixture
def run_id(conn):
    name = f"parity-{uuid4()}"
    yield name
    for table in ("audit_log", "spend_ledger"):
        conn.execute(f"DELETE FROM {table} WHERE run_id LIKE %s", (f"%{name}%",))


# --- the checks ---


@pytest.mark.parametrize("tenant, tool, args, verdict, rule", CASES, ids=IDS)
def test_the_engine_gives_the_expected_answer(tenant, tool, args, verdict, rule):
    got = via_engine(tenant, tool, args)
    assert got[0] == verdict and (rule is None or got[1] == rule)


@pytest.mark.parametrize("tenant, tool, args, verdict, rule", CASES, ids=IDS)
def test_the_hook_agrees_with_the_engine(tenant, tool, args, verdict, rule, policy_file):
    assert via_hook(tenant, tool, args, policy_file) == via_engine(tenant, tool, args)


@pytest.mark.parametrize("tenant, tool, args, verdict, rule", CASES, ids=IDS)
def test_the_server_agrees_with_the_engine(tenant, tool, args, verdict, rule, gateway, run_id):
    assert via_server(tenant, tool, args, gateway, run_id) == via_engine(tenant, tool, args)


@pytest.mark.parametrize("tenant, tool, args, verdict, rule", CASES, ids=IDS)
def test_the_mcp_proxy_agrees_with_the_engine(tenant, tool, args, verdict, rule):
    engine = via_engine(tenant, tool, args)
    expected = ("ask", None) if engine[0] == "ask" else engine
    assert via_mcp(tenant, tool, args) == expected


# --- where the surfaces differ on purpose ---


def test_only_the_mcp_proxy_refuses_an_allowed_tool_that_the_downstream_does_not_have():
    gone = Policy(tenants={TENANT: POLICY.tenants[TENANT].model_copy(
        update={"tools": POLICY.tenants[TENANT].tools | {"vanished": ToolRule(risk=Risk.READ)}}
    )})
    # The engine, hook and server cannot see which tools exist; the proxy asks its downstream server every call.
    assert decide(gone, ToolCall(tenant_id=TENANT, tool="vanished", args={}), RunState()).verdict.value == "allow"
    result = asyncio.run(Governor(gone, TENANT, Downstream()).call_tool(None, types.CallToolRequestParams(name="vanished")))
    assert result.is_error and "tool_unavailable" in result.content[0].text


def test_a_pinned_tool_can_pass_through_the_server_and_the_proxy_but_never_through_the_hook(tmp_path, pool, run_id):
    advertised = types.Tool(name="lookup", description="lookup", input_schema={"type": "object"})
    pin = definition_sha(advertised)
    rules = POLICY.tenants[TENANT].tools | {"lookup": ToolRule(risk=Risk.READ, pinned_sha256=pin)}
    pinned = Policy(tenants={TENANT: POLICY.tenants[TENANT].model_copy(update={"tools": rules})})

    # MCP: fetches the real definition from the downstream server, so the pin matches.
    downstream = Downstream()
    asyncio.run(Governor(pinned, TENANT, downstream).call_tool(None, types.CallToolRequestParams(name="lookup", arguments={})))
    assert downstream.calls == ["lookup"]
    # Server: trusts the hash the caller sends.
    assert Gateway(pinned, pool).decide_call(TENANT, run_id, "p1", "lookup", {}, pin)["status"] == "allowed"
    # Hook: Claude Code sends no tool definition, so a pinned rule always reads as drift. Do not pin Claude Code tools.
    path = tmp_path / "pinned.json"
    path.write_text(pinned.model_dump_json(), encoding="utf-8")
    event = {"hook_event_name": "PreToolUse", "tool_name": "lookup", "tool_input": {}, "session_id": "s"}
    code, _, err = hook.run(json.dumps(event).encode(), {"PORTCULLIS_POLICY": str(path), "PORTCULLIS_TENANT": TENANT})
    assert code == 2 and "definition_drift" in err
