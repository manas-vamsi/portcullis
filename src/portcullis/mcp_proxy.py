"""A governing proxy for MCP servers: portcullis's policy on every tool call, in front of any MCP server.

    python -m portcullis.mcp_proxy pin -- <command that starts the MCP server>
    python -m portcullis.mcp_proxy serve --policy policy.json --tenant NAME [--audit] -- <command>

An MCP client (Claude Code, an agent framework) connects to this proxy instead of to the real server.
The proxy starts the real server itself and sits between them:

- Listing: only tools the policy names are shown, and each is shown with only the pinned fields (name,
  description, input schema). Titles, annotations, icons and metadata are dropped, because they could change
  unnoticed and some clients act on them. A pinned tool is shown only while its definition still hashes to
  the pin, so a description that changes after you reviewed it is hidden, not trusted.
- Calling: every call is decided by the deterministic policy first (allowlist, argument rules,
  amounts, budgets, loop detection, spend cap), against the definition the server advertises right
  now, fetched fresh for each call, so a rug pull in the middle of a session is caught on its next
  call. A refused call never reaches the real server, and the model is told only the rule code.
- Approval: a call that needs a human is turned into an MCP "input required" round trip: the client
  shows its user what is being asked and sends the answer back. The answer is tied to that exact
  call by a signed, expiring, single-use token, and the rules are checked again when it arrives. A
  client that cannot ask its user cannot approve anything: the call fails.
- Record: with --audit (and PORTCULLIS_DATABASE_URL) every decision goes to the hash-chained audit
  log before the call runs; if that write fails the call is refused.

Approval is only as strong as the client. The answer comes from the client, so a compromised client can
fabricate "the user said yes"; the token proves the call was asked about, not that a person answered. For a
human decision a client cannot fake, use the server's approvals instead.

What it does not do: it cannot tell whether a tool's result is safe (a result is passed back
unchanged, and may carry a prompt injection), whether the server does what its description says, or
whether a pinned description was already malicious when you pinned it. State lives in this process:
steps, loops, spend and time are per proxy process, which is one session. Tokens are not known here.
"""

import argparse
import asyncio
import hmac
import json
import os
import secrets
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

from mcp import types
from mcp.server.lowlevel import Server

from portcullis.audit import mcp_run
from portcullis.hook import HookError, load_policy
from portcullis.jsondata import check_json
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
    is_sensitive,
    spend_amount,
)
from portcullis.reloadable import ConfigUnavailable, Reloadable

MAX_LIST_PAGES = 20
APPROVAL_SECONDS = 300
MAX_STATE_CHARS = 200
PREVIEW_CHARS = 2000  # arguments longer than this are never shown to an approver, so they cannot be approved
MAX_ARGUMENT_CHARS = 1_000_000
DOWNSTREAM_TIMEOUT = 10.0  # seconds to list the real server's tools; a slow server must not stall decisions
APPROVER = "mcp-client-user"  # recorded as the approver: the proxy cannot know who the client's user is

Sink = Callable[[str, str, dict[str, Any]], None]


class Downstream(Protocol):
    """The real MCP server, as seen through an MCP client (or a stand-in in tests)."""

    async def list_tools(self, *, cursor: str | None = None) -> types.ListToolsResult: ...

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> types.CallToolResult: ...


def tool_definition(tool: types.Tool) -> dict[str, Any]:
    """What the model is shown about a tool, in the form that is hashed for pinning."""
    return {"name": tool.name, "description": tool.description or "", "inputSchema": tool.input_schema}


def definition_sha(tool: types.Tool) -> str:
    return definition_hash(tool_definition(tool))


async def all_tools(downstream: Downstream) -> list[types.Tool]:
    tools: list[types.Tool] = []
    cursor: str | None = None
    for _ in range(MAX_LIST_PAGES):
        page = await downstream.list_tools(cursor=cursor)
        tools += page.tools
        cursor = page.next_cursor
        if not cursor:
            return tools
    raise RuntimeError("the server lists too many pages of tools")


async def describe_tools(downstream: Downstream) -> list[dict[str, Any]]:
    """Each tool's text and its hash: read the text, and pin the hash in the policy if you are happy with it."""
    return [
        {"name": t.name, "description": t.description or "", "input_schema": t.input_schema, "sha256": definition_sha(t)}
        for t in await all_tools(downstream)
    ]


def _error(message: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=message)], is_error=True)


class AuditFailed(Exception):
    """The decision could not be recorded, so the call must not run."""


class Governor:
    """Decides every call of one session. The handlers `list_tools` and `call_tool` plug into an MCP server."""

    def __init__(
        self,
        policy: Policy | Callable[[], Policy],
        tenant: str,
        downstream: Downstream,
        *,
        sink: Sink | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._policy_of: Callable[[], Policy] = policy if callable(policy) else (lambda: policy)
        self._tenant, self._downstream, self._sink, self._clock = tenant, downstream, sink, clock
        self._lock = asyncio.Lock()  # one decision at a time, so counters and approvals cannot race
        self._secret = secrets.token_bytes(32)  # signs approval tokens; never leaves this process
        self._used: set[str] = set()
        self._started = clock()
        self._steps, self._history, self._spend, self._halted, self._asks = 0, [], 0.0, False, 0

    # ---- listing ----

    async def list_tools(self, ctx: Any, params: Any) -> types.ListToolsResult:
        try:
            policy = self._policy_of()
        except ConfigUnavailable:
            return types.ListToolsResult(tools=[], ttl_ms=0)  # nothing is shown while the policy cannot be read
        tenant = policy.tenants.get(self._tenant)
        rules = tenant.tools if tenant else {}
        shown = []
        for tool in await self._tools():
            rule = rules.get(tool.name)
            if rule is None or (rule.pinned_sha256 is not None and definition_sha(tool) != rule.pinned_sha256):
                continue
            shown.append(types.Tool(name=tool.name, description=tool.description, input_schema=tool.input_schema))
        return types.ListToolsResult(tools=shown, ttl_ms=0)  # ttl 0: never cached, so a later change is seen

    async def _tools(self) -> list[types.Tool]:
        async with asyncio.timeout(DOWNSTREAM_TIMEOUT):
            return await all_tools(self._downstream)

    # ---- calling ----

    async def call_tool(self, ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult | types.InputRequiredResult:
        name, arguments = params.name, params.arguments or {}
        async with self._lock:
            outcome = await self._decide_and_record(params, name, arguments)
        if isinstance(outcome, (types.CallToolResult, types.InputRequiredResult)):
            return outcome
        try:  # outside the lock: a slow tool must not hold up the next decision
            return await self._downstream.call_tool(name, arguments)
        except Exception as exc:
            return _error(f"The downstream server failed ({type(exc).__name__}).")

    async def _decide_and_record(
        self, params: types.CallToolRequestParams, name: str, arguments: dict[str, Any]
    ) -> types.CallToolResult | types.InputRequiredResult | None:
        """The result to return, or None if the call has been decided, recorded and may now be forwarded."""
        if self._halted:
            return _error("Refused by policy: run_halted")
        try:
            policy = self._policy_of()
        except ConfigUnavailable:
            return _error("Refused: the policy is unavailable")
        decision, call, amount = await self._decision(policy, name, arguments)
        approval: bool | None = None
        if decision.verdict is Verdict.ASK:
            blocked = self._cannot_ask(policy, call)
            if blocked:
                decision = blocked
            elif self._proof(params, name, arguments) is None:  # no valid approval yet: ask the client's user
                try:
                    await self._write(call.tool, "ask", {"tool": name, "request_sha256": fingerprint(name, arguments)})
                except AuditFailed:
                    return _error("Refused: the audit log is unavailable")
                self._asks += 1
                return self._ask(policy, call, decision)
            else:
                approval = self._said_yes(params)
        try:
            await self._record(name, arguments, decision, amount, approval)
        except AuditFailed:
            return _error("Refused: the audit log is unavailable")
        if approval is False:
            return _error("Refused: not approved")
        if decision.verdict is Verdict.DENY:
            return _error(f"Refused by policy: {decision.rule}")
        return None

    async def _decision(self, policy: Policy, name: str, arguments: dict[str, Any]) -> tuple[Decision, ToolCall, float]:
        try:
            check_json(arguments)
            too_big = len(json.dumps(arguments, separators=(",", ":"))) > MAX_ARGUMENT_CHARS
        except ValueError:
            too_big = True
        if too_big:
            call = ToolCall(tenant_id=self._tenant, tool=name, args={})
            return Decision(verdict=Verdict.DENY, rule="invalid_arguments", reason="arguments are not plain JSON data, or are too large"), call, 0.0
        try:
            tools = await self._tools()
        except Exception:  # includes a timeout: without the real definition nothing can be checked, so nothing runs
            call = ToolCall(tenant_id=self._tenant, tool=name, args={})
            return Decision(verdict=Verdict.DENY, rule="downstream_unavailable", reason="the server did not list its tools"), call, 0.0
        advertised = {t.name: t for t in tools}.get(name)
        digest = definition_sha(advertised) if advertised else None  # fetched now: a rug pull shows up on this call
        call = ToolCall(tenant_id=self._tenant, tool=name, args=arguments, definition_sha256=digest)
        state = RunState(
            steps=self._steps, seconds=self._clock() - self._started, history=tuple(self._history), spend=self._spend
        )
        decision = decide(policy, call, state)
        if advertised is None and decision.verdict is not Verdict.DENY:
            decision = Decision(verdict=Verdict.DENY, rule="tool_unavailable", reason="the server does not have this tool")
        return decision, call, spend_amount(policy, call)

    # ---- approval ----

    def _cannot_ask(self, policy: Policy, call: ToolCall) -> Decision | None:
        """Reasons not to ask a human at all: too many asks already, or arguments too long to be shown in full."""
        tenant = policy.tenants.get(self._tenant)
        if tenant is not None and self._asks >= 3 * tenant.budget.max_steps:
            return Decision(verdict=Verdict.DENY, rule="too_many_approval_requests", reason="this session has asked too often")
        shown = json.dumps(call.args, sort_keys=True, separators=(",", ":"))
        if not is_sensitive(policy, call) and len(shown) > PREVIEW_CHARS:
            # A person cannot approve what they cannot see: padding must not push the dangerous part out of the prompt.
            return Decision(verdict=Verdict.DENY, rule="approval_unshowable", reason="the arguments are too long to show a person")
        return None

    def _mac(self, nonce: str, issued: int, name: str, arguments: dict[str, Any]) -> str:
        message = f"{nonce}|{issued}|{fingerprint(name, arguments)}".encode()
        return hmac.new(self._secret, message, "sha256").hexdigest()

    def _ask(self, policy: Policy, call: ToolCall, decision: Decision) -> types.InputRequiredResult:
        nonce, issued = secrets.token_hex(8), int(self._clock())
        token = f"{nonce}.{issued}.{self._mac(nonce, issued, call.tool, call.args)}"
        if is_sensitive(policy, call):
            shown = "(arguments not shown: they matched a sensitive pattern)"
        else:
            shown = json.dumps(call.args, sort_keys=True, separators=(",", ":"))[:PREVIEW_CHARS]
        length = len(json.dumps(call.args, sort_keys=True, separators=(",", ":")))
        message = (
            f"Portcullis [{decision.rule}] needs your approval to run {call.tool}.\n{shown}\n"
            f"(request {fingerprint(call.tool, call.args)[:12]}, {length} characters of arguments)"
        )
        request = types.ElicitRequest(
            params=types.ElicitRequestFormParams(
                message=message,
                requested_schema={
                    "type": "object",
                    "properties": {"approve": {"type": "boolean", "title": "Approve this call"}},
                    "required": ["approve"],
                },
            )
        )
        return types.InputRequiredResult(input_requests={"approve": request}, request_state=token)

    def _proof(self, params: types.CallToolRequestParams, name: str, arguments: dict[str, Any]) -> str | None:
        """The approval token, if this request carries a valid one for this exact call. A valid token is used up."""
        state = params.request_state
        if not isinstance(state, str) or len(state) > MAX_STATE_CHARS or state.count(".") != 2:
            return None
        nonce, issued_text, mac = state.split(".")
        try:
            issued = int(issued_text)
        except ValueError:
            return None
        if not mac.isascii() or not hmac.compare_digest(mac, self._mac(nonce, issued, name, arguments)):
            return None  # not issued by this session for this call
        if not 0 <= self._clock() - issued <= APPROVAL_SECONDS or nonce in self._used:
            return None
        self._used.add(nonce)
        return state

    @staticmethod
    def _said_yes(params: types.CallToolRequestParams) -> bool:
        answer = (params.input_responses or {}).get("approve")
        content = getattr(answer, "content", None) or {}
        return getattr(answer, "action", None) == "accept" and content.get("approve") is True

    # ---- the record and the counters ----

    async def _write(self, tool: str, event: str, payload: dict[str, Any]) -> None:
        """One audit row, written on a thread so a slow database cannot freeze the event loop. Raises AuditFailed."""
        if self._sink is None:
            return
        try:
            await asyncio.to_thread(self._sink, f"{uuid.uuid4().hex}:{event}", event, {**payload, "tool": tool})
        except Exception as exc:
            raise AuditFailed from exc

    async def _record(self, name: str, arguments: dict[str, Any], decision: Decision, amount: float, approval: bool | None) -> None:
        """Write the decision (and a human's answer) first; only then count it. Raises AuditFailed.

        If the second row fails after the first was written, the log shows a decision the proxy then refused;
        the approval token is already used, so the call has to be asked about again. That is noise, not a bypass."""
        call_id = uuid.uuid4().hex
        request_sha = fingerprint(name, arguments)
        if self._sink is not None:
            payload = {
                "call_id": call_id, "tool": name, "verdict": decision.verdict.value, "rule": decision.rule, "reason": decision.reason, "request_sha256": request_sha,
                "amount": amount, "tokens": 0,
            }
            try:
                await asyncio.to_thread(self._sink, f"{call_id}:decision", "decision", payload)
                if approval is not None:
                    answer = {"call_id": call_id, "tool": name, "approved": approval, "approved_by": APPROVER}
                    await asyncio.to_thread(self._sink, f"{call_id}:approval", "approval", answer)
            except Exception as exc:
                raise AuditFailed from exc
        self._steps += 1
        self._history.append(request_sha)
        if decision.verdict is Verdict.ALLOW or approval:
            self._spend += amount
        if decision.rule in HALTING_RULES:
            self._halted = True


def build_server(governor: Governor) -> Server:
    return Server("portcullis-proxy", on_list_tools=governor.list_tools, on_call_tool=governor.call_tool)


def _connect(url: str) -> Any:
    import psycopg

    return psycopg.connect(url, autocommit=True, connect_timeout=5, options="-c statement_timeout=5000 -c lock_timeout=5000")


def make_sink(url: str, tenant: str) -> Sink:
    """A sink that writes to the hash-chained audit log, under this session's own run name.

    A dropped connection is replaced once; if the write still fails the caller refuses the call."""
    import psycopg

    from portcullis.audit import append, create_audit_log, require_project_database

    require_project_database(url)
    connection = {"conn": _connect(url)}
    create_audit_log(connection["conn"])
    run = mcp_run(tenant, uuid.uuid4().hex)

    def write(event_id: str, event: str, payload: dict[str, Any]) -> None:
        if connection["conn"].closed or connection["conn"].broken:  # append cannot even start on a dead connection
            connection["conn"] = _connect(url)
        try:
            append(connection["conn"], run, tenant, event_id, event, payload)
        except psycopg.OperationalError:  # it died during the write
            connection["conn"] = _connect(url)
            append(connection["conn"], run, tenant, event_id, event, payload)

    return write


def warnings_for(policy: Policy, tenant: str) -> list[str]:
    """Things about this policy that the proxy cannot honour, said at startup rather than discovered later."""
    tenant_policy = policy.tenants.get(tenant)
    if tenant_policy is None:
        return [f"the policy has no tenant {tenant!r}: every call will be refused"]
    if tenant_policy.budget.max_spend_per_day is not None:
        return ["max_spend_per_day is not enforced here (the proxy only counts one session); use the server for a daily cap"]
    return []


def format_tool(row: dict[str, Any]) -> str:
    """One tool for a person to review. The name is quoted: a hostile server must not be able to forge a line."""
    return f"{json.dumps(row['name'])}\n  sha256: {row['sha256']}\n  description: {json.dumps(row['description'])}\n"


def _environment(names: Sequence[str]) -> dict[str, str]:
    """Variables handed to the real server, by name from this process's environment (values never on a command line)."""
    missing = [n for n in names if n not in os.environ]
    if missing:
        raise ValueError(f"the environment variable {missing[0]} is not set")
    return {n: os.environ[n] for n in names}


async def serve(policy_path: Path, tenant: str, command: list[str], audit: bool, env_names: Sequence[str] = ()) -> None:  # pragma: no cover
    """Run the proxy over stdio. Covered by the process test, which cannot report coverage."""
    from mcp import Client, StdioServerParameters
    from mcp.server.stdio import stdio_server

    reloadable = Reloadable(policy_path, load_policy)
    for warning in warnings_for(reloadable.get(), tenant):
        print(f"portcullis mcp proxy: warning: {warning}", file=sys.stderr)
    policy = reloadable.get
    sink = make_sink(os.environ["PORTCULLIS_DATABASE_URL"], tenant) if audit else None
    parameters = StdioServerParameters(command=command[0], args=command[1:], env=_environment(env_names) or None)
    async with Client(parameters) as downstream:
        server = build_server(Governor(policy, tenant, downstream, sink=sink))
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())


async def pin(command: list[str]) -> None:  # pragma: no cover
    """Print each tool of the real server with its description and the hash to pin. Covered by the process test."""
    from mcp import Client, StdioServerParameters

    async with Client(StdioServerParameters(command=command[0], args=command[1:])) as downstream:
        for row in await describe_tools(downstream):
            print(format_tool(row))


def _split(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    """Everything before the first `--` is for this program; everything after is the server's command."""
    if "--" not in argv:
        raise SystemExit("give the command that starts the MCP server after --")
    cut = list(argv).index("--")
    command = list(argv[cut + 1 :])
    if not command:
        raise SystemExit("the command after -- is empty")
    return list(argv[:cut]), command


def main(argv: Sequence[str] | None = None) -> int:
    ours, command = _split(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="python -m portcullis.mcp_proxy", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("pin", help="show each tool's description and the hash to pin")
    serve_cmd = commands.add_parser("serve", help="run the proxy over stdio")
    serve_cmd.add_argument("--policy", required=True, type=Path)
    serve_cmd.add_argument("--tenant", required=True)
    serve_cmd.add_argument("--audit", action="store_true", help="write decisions to the audit log (needs PORTCULLIS_DATABASE_URL)")
    serve_cmd.add_argument("--env", action="append", default=[], metavar="NAME",
                           help="pass this environment variable to the real server (by name; repeatable)")
    args = parser.parse_args(ours)
    try:
        if args.command == "pin":
            asyncio.run(pin(command))
        else:
            if args.audit and not os.environ.get("PORTCULLIS_DATABASE_URL"):
                raise SystemExit("--audit needs PORTCULLIS_DATABASE_URL")
            asyncio.run(serve(args.policy, args.tenant, command, args.audit, args.env))
    except (HookError, ValueError) as exc:
        print(f"portcullis mcp proxy: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
