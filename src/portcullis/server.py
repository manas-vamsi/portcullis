"""HTTP server: tool-call decisions, human approvals and the audit trail, over Postgres.

An agent runtime asks before each tool call; the server decides with the policy and answers
allowed, denied or pending_approval. A person with an approver key answers pending ones. The runtime
then says when it has made an allowed call (`executed`). Every decision and answer is written to the
hash-chained audit log before it is returned.

Design points
- The tenant always comes from the API key, never from the request body (bodies with extra fields
  are refused). Run names are namespaced by tenant, so tenants cannot touch each other's runs.
- Run state (steps, loop history, spend, tokens, elapsed time) is not kept anywhere else: it is
  folded from the run's own audit chain on every request (see runstate.py), and the chain is
  verified first. A tampered chain stops the run (503) instead of producing a decision from
  corrupted state.
- Requests for one run are serialized with a Postgres advisory lock, so concurrent calls cannot
  overshoot a budget or a spend cap.
- Spend is counted when a call is allowed or approved (not when the caller reports it ran), which
  can only over-count. Two caps apply: per run (`max_spend`) and per tenant over a rolling 24 hours
  across all its runs (`max_spend_per_day`, kept in a small ledger table), so a new run does not
  reset spend. An approval re-checks both; one that no longer fits is recorded as denied with
  `blocked_by` saying why.
- A halted run (budget exceeded, loop, unknown tenant) is answered `run_halted` without writing
  more rows, so a flood of calls cannot grow the log. Starting new runs is rate limited per tenant.
- `executed`: once a runtime reports a call as made, asking about it again answers `executed`, not
  `allowed`, so a replayed answer can no longer be mistaken for a fresh permission.
- The policy and key files are reloaded when they change. A file that becomes invalid makes every
  request fail (503) until it is fixed, rather than carrying on with an older, possibly looser one.
- Fail closed: any failure is a non-2xx response. A caller must treat anything other than an
  `allowed` or `approved` answer as a refusal.

Trust and limits
- Tokens are reported by the runtime (the server never sees the model). A runtime that under-reports
  weakens the token budget; the other budgets do not depend on it. Elapsed time is wall-clock since
  the run's first call, minus time spent waiting for a human.
- For calls that wait for a human, the answer carries a bounded preview of the arguments so the
  approver can see what they are approving, unless the call matched a sensitive-argument rule (those
  arguments may be secrets and are not stored). The preview is untrusted text from the model.
- The reason in an answer is for the runtime and the audit log; do not show it to the model.
- Replays before a call is reported `executed` are still answered `allowed`: call ids must come
  from the runtime, never from model output.
- Plain HTTP on loopback by default: put TLS in front of it. The new-run limit is kept in memory,
  per process.
"""

import argparse
import json
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Path as PathParam, Query, Request
from fastapi.responses import JSONResponse
from psycopg_pool import ConnectionPool
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, field_validator

from portcullis.audit import (
    AuditConflict,
    append,
    create_audit_log,
    events,
    lock_key,
    namespaced,
    payload_data,
    require_project_database,
    verify,
)
from portcullis.hook import HookError, load_policy
from portcullis.jsondata import MAX_INT, check_json
from portcullis.keys import KeyEntry, Principal, authenticate, load_keys
from portcullis.policy import Policy, ToolCall, Verdict, decide, fingerprint, is_sensitive, spend_amount
from portcullis.reloadable import ConfigUnavailable, Reloadable
from portcullis.runstate import CallRecord, InconsistentRun, fold, halted, run_state, spent, status, view

MAX_BODY = 262_144  # bytes; larger requests are refused before they are read
MAX_NAME = 128
CALL_ID = r"^[A-Za-z0-9._:-]+$"  # it becomes part of a log event id, so no odd characters
PENDING_LIMIT = 100
PER_TENANT_REQUESTS = 3  # connections one tenant may hold at once; the pool has 10
SLOT_WAIT_SECONDS = 5.0
NEW_RUNS_PER_MINUTE = 60  # per tenant, per process
PREVIEW_CHARS = 2000
DB_OPTIONS = "-c statement_timeout=10000 -c lock_timeout=5000 -c idle_in_transaction_session_timeout=10000"

CREATE_SPEND_LEDGER = (
    """CREATE TABLE IF NOT EXISTS spend_ledger (
        tenant text NOT NULL,
        run_id text NOT NULL,
        call_id text NOT NULL,
        amount double precision NOT NULL,
        ts timestamptz NOT NULL DEFAULT now(),
        PRIMARY KEY (tenant, run_id, call_id)
    )""",
    "CREATE INDEX IF NOT EXISTS spend_ledger_window ON spend_ledger (tenant, ts)",
)


class Conflict(Exception):
    """The request clashes with what is already recorded."""


class NotFound(Exception):
    """No such call in this tenant's run."""


class Busy(Exception):
    """The tenant already has its share of requests in flight, or is starting runs too fast."""


class ChainBroken(Exception):
    """The run's audit chain does not verify, so no decision can be made from it."""


def open_pool(url: str) -> ConnectionPool:
    require_project_database(url)
    pool = ConnectionPool(
        url,
        min_size=1,
        max_size=10,
        timeout=5,
        kwargs={"autocommit": True, "connect_timeout": 5, "options": DB_OPTIONS},
        open=True,
    )
    pool.wait(timeout=10)
    with pool.connection() as conn:
        create_audit_log(conn)
        for statement in CREATE_SPEND_LEDGER:
            conn.execute(statement)
    return pool


def _preview(args: Mapping[str, Any]) -> tuple[str, bool]:
    """The arguments as compact JSON text, cut to a fixed length. For a human to read, never to parse."""
    text = json.dumps(args, sort_keys=True, separators=(",", ":"))
    return text[:PREVIEW_CHARS], len(text) > PREVIEW_CHARS


class Gateway:
    """The server's logic, apart from HTTP so it can be tested (and raced) directly."""

    def __init__(
        self,
        policy: Policy | Callable[[], Policy],
        pool: ConnectionPool,
        per_tenant: int = PER_TENANT_REQUESTS,
        wait: float = SLOT_WAIT_SECONDS,
        new_runs_per_minute: int = NEW_RUNS_PER_MINUTE,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._policy_of: Callable[[], Policy] = policy if callable(policy) else (lambda: policy)
        self._pool = pool
        self._per_tenant, self._wait = per_tenant, wait
        self._slots: dict[str, threading.BoundedSemaphore] = {}
        self._new_runs_per_minute = new_runs_per_minute
        self._new_runs: dict[str, deque[float]] = {}
        self._guard = threading.Lock()
        self._clock = clock

    @contextmanager
    def _slot(self, tenant: str):
        """At most a few requests per tenant at once, so one tenant cannot use up every pooled connection."""
        with self._guard:
            slots = self._slots.setdefault(tenant, threading.BoundedSemaphore(self._per_tenant))
        if not slots.acquire(timeout=self._wait):
            raise Busy(tenant)
        try:
            yield
        finally:
            slots.release()

    def _may_start_a_run(self, tenant: str) -> bool:
        """A sliding one-minute window of new runs per tenant. Kept in memory: it limits, it does not account."""
        now = time.monotonic()
        with self._guard:
            starts = self._new_runs.setdefault(tenant, deque())
            while starts and now - starts[0] > 60:
                starts.popleft()
            if len(starts) >= self._new_runs_per_minute:
                return False
            starts.append(now)
            return True

    def config_ok(self) -> None:
        """Raises ConfigUnavailable if the policy file is currently unusable."""
        self._policy_of()

    def ping(self) -> None:
        with self._pool.connection(timeout=3) as conn:
            conn.execute("SELECT 1")

    @contextmanager
    def _advisory(self, conn: psycopg.Connection, name: str):
        key = lock_key(name)
        conn.execute("SELECT pg_advisory_lock(%s)", (key,))
        try:
            yield
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (key,))

    @contextmanager
    def _locked_run(self, tenant: str, run: str):
        """A connection holding the run's advisory lock; one request at a time per run."""
        with self._slot(tenant), self._pool.connection() as conn, self._advisory(conn, run):
            yield conn

    @contextmanager
    def _tenant_spend(self, conn: psycopg.Connection, policy: Policy, tenant: str, amount: float):
        """The tenant's spend over the last 24 hours, read under a lock that stays held until the caller has charged
        for its own spend, so two runs cannot both pass the daily cap. Zero (and no lock) when no cap applies."""
        tenant_policy = policy.tenants.get(tenant)
        if tenant_policy is None or tenant_policy.budget.max_spend_per_day is None or not amount:
            yield 0.0
            return
        with self._advisory(conn, f"tenant-spend:{tenant}"):
            conn.execute("DELETE FROM spend_ledger WHERE tenant = %s AND ts < now() - interval '48 hours'", (tenant,))
            row = conn.execute(
                "SELECT coalesce(sum(amount), 0) FROM spend_ledger WHERE tenant = %s AND ts > now() - interval '24 hours'",
                (tenant,),
            ).fetchone()
            yield float(row[0])

    def _charge(self, conn: psycopg.Connection, tenant: str, run_id: str, call_id: str, amount: float) -> None:
        if amount:
            conn.execute(
                "INSERT INTO spend_ledger (tenant, run_id, call_id, amount) VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (tenant, run_id, call_id) DO UPDATE SET amount = greatest(spend_ledger.amount, excluded.amount)",
                (tenant, run_id, call_id, amount),
            )

    def _load(self, conn: psycopg.Connection, run: str) -> dict[str, CallRecord]:
        if not verify(conn, run).ok:
            raise ChainBroken(run)
        try:
            return fold(events(conn, run))
        except InconsistentRun as exc:
            raise ChainBroken(run) from exc

    def decide_call(
        self,
        tenant: str,
        run_id: str,
        call_id: str,
        tool: str,
        args: dict[str, Any],
        definition_sha256: str | None,
        tokens: int = 0,
    ) -> dict[str, Any]:
        run, request_sha256 = namespaced(tenant, run_id), fingerprint(tool, args)
        call = ToolCall(tenant_id=tenant, tool=tool, args=args, definition_sha256=definition_sha256)
        with self._locked_run(tenant, run) as conn:
            policy = self._policy_of()  # read after any wait for the slot and the lock, so a tightened policy applies now
            amount = spend_amount(policy, call)
            calls = self._load(conn, run)
            known = calls.get(call_id)
            if known:
                same = (known.decision["request_sha256"], known.decision.get("definition_sha256"))
                if same != (request_sha256, definition_sha256):
                    raise Conflict(f"call_id {call_id!r} was already used for a different call")
                return view(known)  # a retry: the same answer, nothing new written
            if halted(calls):
                return {"call_id": call_id, "status": "denied", "rule": "run_halted",
                        "reason": "the run was halted earlier", "halt": True}
            if not calls and not self._may_start_a_run(tenant):
                raise Busy(tenant)
            now = self._clock()
            with self._tenant_spend(conn, policy, tenant, amount) as tenant_spend:
                decision = decide(policy, call, run_state(calls, now, tenant_spend))
                payload: dict[str, Any] = {
                    "call_id": call_id,
                    "tool": tool,
                    "verdict": decision.verdict.value,
                    "rule": decision.rule,
                    "reason": decision.reason,
                    "request_sha256": request_sha256,  # a hash; the arguments may be sensitive
                    "amount": amount,
                    "tokens": tokens,
                }
                if definition_sha256 is not None:
                    payload["definition_sha256"] = definition_sha256
                if decision.verdict is Verdict.ASK and not is_sensitive(policy, call):
                    payload["args_preview"], payload["args_truncated"] = _preview(args)
                if decision.verdict is Verdict.ALLOW:
                    self._charge(conn, tenant, run_id, call_id, amount)  # before the row: a failure here over-counts
                append(conn, run, tenant, f"{call_id}:decision", "decision", payload)
            return view(CallRecord(payload, now))

    def call_status(self, tenant: str, run_id: str, call_id: str) -> dict[str, Any]:
        self.config_ok()
        run = namespaced(tenant, run_id)
        with self._locked_run(tenant, run) as conn:
            record = self._load(conn, run).get(call_id)
        if record is None:
            raise NotFound(call_id)
        return view(record)

    def mark_executed(self, tenant: str, run_id: str, call_id: str) -> dict[str, Any]:
        """The runtime says it made an allowed (or approved) call. Asking about the call afterwards answers `executed`."""
        self.config_ok()
        run = namespaced(tenant, run_id)
        with self._locked_run(tenant, run) as conn:
            record = self._load(conn, run).get(call_id)
            if record is None:
                raise NotFound(call_id)
            if record.executed is not None:
                return view(record)  # saying it twice is harmless
            if status(record) not in ("allowed", "approved"):
                raise Conflict("this call was not allowed, so it cannot have been made")
            payload = {"call_id": call_id, "tool": record.decision["tool"]}
            append(conn, run, tenant, f"{call_id}:executed", "executed", payload)
            record.executed = payload
            return view(record)

    def answer(self, tenant: str, run_id: str, call_id: str, approved: bool, by: str) -> dict[str, Any]:
        run = namespaced(tenant, run_id)
        with self._locked_run(tenant, run) as conn:
            policy = self._policy_of()
            calls = self._load(conn, run)
            record = calls.get(call_id)
            if record is None:
                raise NotFound(call_id)
            if record.decision["verdict"] != Verdict.ASK:
                raise Conflict("this call is not waiting for approval")
            if record.approval is not None:
                raise Conflict("this call was already answered")
            amount = record.decision["amount"]
            with (self._tenant_spend(conn, policy, tenant, amount) if approved else nullcontext(0.0)) as tenant_spend:
                blocked = None
                if approved:
                    blocked = "run_halted" if halted(calls) else self._cap_blocks(policy, tenant, calls, record, tenant_spend)
                final = approved and blocked is None
                payload: dict[str, Any] = {
                    "call_id": call_id, "tool": record.decision["tool"], "approved": final, "approved_by": by,
                }
                if blocked:
                    payload["blocked_by"] = blocked
                if final:
                    self._charge(conn, tenant, run_id, call_id, amount)
                append(conn, run, tenant, f"{call_id}:approval", "approval", payload)
            record.approval = payload
            return view(record)

    @staticmethod
    def _cap_blocks(
        policy: Policy, tenant: str, calls: Mapping[str, CallRecord], record: CallRecord, tenant_spend: float
    ) -> str | None:
        """An approval is a new spend, so both caps are checked again against what was spent since the ask."""
        tenant_policy = policy.tenants.get(tenant)
        if tenant_policy is None:
            return "unknown_tenant"
        amount, budget = record.decision["amount"], tenant_policy.budget
        if budget.max_spend is not None and spent(calls) + amount > budget.max_spend:
            return "spend_cap_exceeded"
        if budget.max_spend_per_day is not None and tenant_spend + amount > budget.max_spend_per_day:
            return "tenant_spend_cap_exceeded"
        return None

    def pending(self, tenant: str, limit: int = PENDING_LIMIT, offset: int = 0) -> list[dict[str, Any]]:
        """Calls of this tenant's server runs that are waiting for an answer, oldest first, one page at a time."""
        self.config_ok()
        prefix = namespaced(tenant, "")[:-2]  # '["tenant", "' : only runs made through the server
        with self._slot(tenant), self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT d.run_id, d.payload FROM audit_log d "
                "WHERE d.tenant_id = %s AND d.event = 'decision' AND starts_with(d.run_id, %s) "
                "AND d.payload LIKE '%%\"verdict\":\"ask\"%%' "
                "AND NOT EXISTS (SELECT 1 FROM audit_log a WHERE a.run_id = d.run_id "
                "AND a.event_id = left(d.event_id, length(d.event_id) - 9) || ':approval') "
                "ORDER BY d.ts, d.run_id, d.idx LIMIT %s OFFSET %s",
                (tenant, prefix, limit, offset),
            ).fetchall()
        result = []
        for run, text in rows:
            try:
                payload = json.loads(text)
                entry = {
                    "run_id": json.loads(run)[1],
                    "call_id": payload["call_id"],
                    "tool": payload["tool"],
                    "rule": payload["rule"],
                    "reason": payload["reason"],
                    "request_sha256": payload["request_sha256"],  # check any arguments you are shown against this
                }
                if "args_preview" in payload:
                    entry["args_preview"], entry["args_truncated"] = payload["args_preview"], payload["args_truncated"]
            except (KeyError, TypeError, ValueError, IndexError):
                continue  # a row that is not shaped like ours is not a call anyone can answer; skip it, do not crash
            result.append(entry)
        return result

    def audit(self, tenant: str, run_id: str) -> dict[str, Any]:
        self.config_ok()
        run = namespaced(tenant, run_id)
        with self._slot(tenant), self._pool.connection() as conn:
            result, rows = verify(conn, run), events(conn, run)
        return {
            "verified": {"ok": result.ok, "rows": result.rows, "head": result.head, "problem": result.problem},
            "events": [{**row, "payload": payload_data(row["payload"])} for row in rows],
        }


class CallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # no tenant_id or anything else smuggled in

    call_id: str = Field(min_length=1, max_length=MAX_NAME, pattern=CALL_ID)
    tool: str = Field(min_length=1, max_length=MAX_NAME)
    args: dict[str, Any] = Field(default_factory=dict)
    definition_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tokens: int = Field(default=0, ge=0, le=MAX_INT)  # what the model used for this step, from the runtime's usage data

    @field_validator("args")
    @classmethod
    def _plain_json_only(cls, args: dict[str, Any]) -> dict[str, Any]:
        check_json(args)
        return args


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: StrictBool


def create_app(
    policy: Policy | Callable[[], Policy],
    keys: Sequence[KeyEntry] | Callable[[], Sequence[KeyEntry]],
    database_url: str,
    new_runs_per_minute: int = NEW_RUNS_PER_MINUTE,
) -> FastAPI:
    """`policy` and `keys` may be values, or functions that return the current ones (see Reloadable)."""
    require_project_database(database_url)
    keys_of: Callable[[], Sequence[KeyEntry]] = keys if callable(keys) else (lambda: keys)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool = open_pool(database_url)
        app.state.gateway = Gateway(policy, pool, new_runs_per_minute=new_runs_per_minute)
        yield
        pool.close()

    # The interactive docs pages are off: they add surface and the API is small.
    app = FastAPI(title="portcullis", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def gateway() -> Gateway:
        return app.state.gateway

    def who_is(request: Request) -> Principal | None:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        return authenticate(keys_of(), token.strip()) if scheme.lower() == "bearer" else None

    def principal(request: Request) -> Principal:
        found = who_is(request)
        if found is None:  # pragma: no cover  (unreachable today: the middleware answers 401 first; kept so a new route fails closed)
            raise HTTPException(401, "missing or invalid API key", headers={"WWW-Authenticate": "Bearer"})
        return found

    def needs(role: str):
        def check(who: Principal = Depends(principal)) -> Principal:
            if who.role != role:
                raise HTTPException(403, f"this endpoint needs a key with the {role!r} role")
            return who

        return check

    def either(who: Principal = Depends(principal)) -> Principal:
        return who

    @app.middleware("http")
    async def guard(request: Request, call_next):
        try:
            # Authenticate before anything reads or validates the body, so strangers learn nothing from error text.
            if request.url.path.startswith("/v1/") and who_is(request) is None:
                return JSONResponse(
                    {"detail": "missing or invalid API key"}, status_code=401, headers={"WWW-Authenticate": "Bearer"}
                )
        except ConfigUnavailable:  # not an exception handler: errors raised in middleware bypass those
            return JSONResponse({"detail": "server configuration is unavailable"}, status_code=503)
        if request.method in ("POST", "PUT", "PATCH"):
            length = request.headers.get("content-length", "")
            if not length.isdigit():
                return JSONResponse({"detail": "Content-Length is required"}, status_code=411)
            if int(length) > MAX_BODY:
                return JSONResponse({"detail": "request body too large"}, status_code=413)
        return await call_next(request)

    @app.exception_handler(Conflict)
    @app.exception_handler(AuditConflict)
    async def _conflict(request: Request, exc: Exception):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(Busy)
    async def _busy(request: Request, exc: Busy):
        return JSONResponse(
            {"detail": "too many requests for this tenant right now"}, status_code=429, headers={"Retry-After": "1"}
        )

    @app.exception_handler(NotFound)
    async def _not_found(request: Request, exc: NotFound):
        return JSONResponse({"detail": "no such call"}, status_code=404)

    @app.exception_handler(ChainBroken)
    async def _broken(request: Request, exc: ChainBroken):
        return JSONResponse({"detail": "the audit chain for this run does not verify"}, status_code=503)

    @app.exception_handler(ConfigUnavailable)
    async def _config(request: Request, exc: ConfigUnavailable):
        return JSONResponse({"detail": "server configuration is unavailable"}, status_code=503)

    @app.exception_handler(psycopg.Error)
    async def _database(request: Request, exc: psycopg.Error):  # never echo the error: it can name hosts or users
        return JSONResponse({"detail": "database unavailable"}, status_code=503)

    def name() -> Any:  # a fresh path parameter spec each time, so none is shared
        return PathParam(min_length=1, max_length=MAX_NAME)

    def call_name() -> Any:
        return PathParam(min_length=1, max_length=MAX_NAME, pattern=CALL_ID)

    @app.get("/healthz")
    def healthz(gw: Gateway = Depends(gateway)):
        try:
            gw.ping()
        except psycopg.Error:
            raise HTTPException(503, "database unavailable") from None
        gw.config_ok()
        keys_of()
        return {"status": "ok"}

    @app.post("/v1/runs/{run_id}/calls")
    def create_call(
        body: CallRequest, run_id: str = name(), who: Principal = Depends(needs("agent")), gw: Gateway = Depends(gateway)
    ):
        return gw.decide_call(who.tenant, run_id, body.call_id, body.tool, body.args, body.definition_sha256, body.tokens)

    @app.get("/v1/runs/{run_id}/calls/{call_id}")
    def get_call(
        run_id: str = name(), call_id: str = call_name(), who: Principal = Depends(either), gw: Gateway = Depends(gateway)
    ):
        return gw.call_status(who.tenant, run_id, call_id)

    @app.post("/v1/runs/{run_id}/calls/{call_id}/executed")
    def call_executed(
        run_id: str = name(), call_id: str = call_name(), who: Principal = Depends(needs("agent")), gw: Gateway = Depends(gateway)
    ):
        return gw.mark_executed(who.tenant, run_id, call_id)

    @app.post("/v1/runs/{run_id}/calls/{call_id}/approval")
    def answer_call(
        body: ApprovalRequest,
        run_id: str = name(),
        call_id: str = call_name(),
        who: Principal = Depends(needs("approver")),
        gw: Gateway = Depends(gateway),
    ):
        return gw.answer(who.tenant, run_id, call_id, body.approved, who.name)  # the name comes from the key

    @app.get("/v1/approvals")
    def pending(
        limit: int = Query(PENDING_LIMIT, ge=1, le=PENDING_LIMIT),
        offset: int = Query(0, ge=0, le=100_000),
        who: Principal = Depends(needs("approver")),
        gw: Gateway = Depends(gateway),
    ):
        return {"pending": gw.pending(who.tenant, limit, offset)}

    @app.get("/v1/runs/{run_id}/audit")
    def run_audit(run_id: str = name(), who: Principal = Depends(either), gw: Gateway = Depends(gateway)):
        return gw.audit(who.tenant, run_id)

    return app


def _setting(env: Mapping[str, str], name: str) -> str:
    value = env.get(name)
    if not value:
        raise ValueError(f"{name} is not set")
    return value


def app_from_env(env: Mapping[str, str]) -> FastAPI:
    """Build the app from PORTCULLIS_POLICY, PORTCULLIS_KEYS and PORTCULLIS_DATABASE_URL. Nothing has a default.

    Both files are reloaded when they change; a bad file at startup is an error."""
    policy_path, keys_path, url = (_setting(env, n) for n in ("PORTCULLIS_POLICY", "PORTCULLIS_KEYS", "PORTCULLIS_DATABASE_URL"))
    try:
        policy = Reloadable(Path(policy_path), load_policy)
    except HookError as exc:
        raise ValueError(str(exc)) from exc
    return create_app(policy.get, Reloadable(Path(keys_path), load_keys).get, url)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    import uvicorn

    parser = argparse.ArgumentParser(prog="python -m portcullis.server", description="Run the portcullis server.")
    parser.add_argument("--host", default="127.0.0.1", help="default is loopback only; put TLS in front before exposing it")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        app = app_from_env(os.environ)
    except (ValueError, ValidationError) as exc:
        print(f"portcullis server: {exc}")
        return 2
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
