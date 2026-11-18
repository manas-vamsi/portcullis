"""HTTP server: tool-call decisions, human approvals and the audit trail, over Postgres.

An agent runtime asks before each tool call; the server decides with the policy and answers
allowed, denied or pending_approval. A person with an approver key answers pending ones. Every
decision and answer is written to the hash-chained audit log before it is returned.

Design points
- The tenant always comes from the API key, never from the request body (bodies with extra fields
  are refused). Run names are namespaced by tenant, so tenants cannot touch each other's runs.
- Run state (steps, loop history, spend) is not kept anywhere else: it is folded from the run's own
  audit chain on every request, and the chain is verified first. A tampered chain stops the run
  (503) instead of producing a decision from corrupted state.
- Requests for one run are serialized with a Postgres advisory lock, so concurrent calls cannot
  overshoot a budget or the spend cap.
- Spend is counted when a call is allowed or approved (not when the caller reports it ran), which
  can only over-count. An approval re-checks the cap; one that no longer fits is recorded as
  denied with `blocked_by: spend_cap_exceeded`.
- A halted run (budget exceeded, loop, unknown tenant) is answered `run_halted` without writing
  more rows, so a flood of calls cannot grow the log.
- Fail closed: any failure is a non-2xx response. A caller must treat anything other than an
  `allowed` or `approved` answer as a refusal.

Not enforced here: the token and time budgets (the server never sees tokens or model time), and
anything about what the tool then does. The reason in an answer is for the runtime and the audit
log; do not show it to the model. Plain HTTP on loopback by default: put TLS in front of it.
"""

import argparse
import hashlib
import json
import os
import threading
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
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
    namespaced,
    payload_data,
    require_project_database,
    verify,
)
from portcullis.hook import HookError, load_policy
from portcullis.jsondata import check_json
from portcullis.keys import KeyEntry, Principal, authenticate, load_keys
from portcullis.policy import HALTING_RULES, Policy, RunState, ToolCall, Verdict, decide, fingerprint, spend_amount

MAX_BODY = 262_144  # bytes; larger requests are refused before they are read
MAX_NAME = 128
CALL_ID = r"^[A-Za-z0-9._:-]+$"  # it becomes part of a log event id, so no odd characters
PENDING_LIMIT = 100
PER_TENANT_REQUESTS = 3  # connections one tenant may hold at once; the pool has 10
SLOT_WAIT_SECONDS = 5.0
DB_OPTIONS = "-c statement_timeout=10000 -c lock_timeout=5000 -c idle_in_transaction_session_timeout=10000"


class Conflict(Exception):
    """The request clashes with what is already recorded."""


class NotFound(Exception):
    """No such call in this tenant's run."""


class Busy(Exception):
    """The tenant already has its share of requests in flight."""


class ChainBroken(Exception):
    """The run's audit chain does not verify, so no decision can be made from it."""


def lock_key(run: str) -> int:
    """A 64-bit advisory lock number for a run: a sha256 prefix, so a client cannot guess or collide it."""
    return int.from_bytes(hashlib.sha256(f"run:{run}".encode()).digest()[:8], "big", signed=True)


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
    return pool


@dataclass
class CallRecord:
    decision: dict[str, Any]
    approval: dict[str, Any] | None = None


def _fold(rows: Sequence[Mapping[str, Any]]) -> dict[str, CallRecord]:
    """The run's calls in order, rebuilt from its (already verified) audit rows."""
    calls: dict[str, CallRecord] = {}
    for row in rows:
        payload = json.loads(row["payload"])
        if row["event"] == "decision":
            calls[payload["call_id"]] = CallRecord(payload)
        elif row["event"] == "approval":
            calls[payload["call_id"]].approval = payload
    return calls


def _counts_as_spent(record: CallRecord) -> bool:
    return record.decision["verdict"] == Verdict.ALLOW or bool(record.approval and record.approval["approved"])


def _spent(calls: Mapping[str, CallRecord]) -> float:
    return float(sum(r.decision["amount"] for r in calls.values() if _counts_as_spent(r)))


def _status(record: CallRecord) -> str:
    verdict = record.decision["verdict"]
    if verdict == Verdict.ALLOW:
        return "allowed"
    if verdict == Verdict.DENY:
        return "denied"
    if record.approval is None:
        return "pending_approval"
    return "approved" if record.approval["approved"] else "approval_denied"


def _view(record: CallRecord) -> dict[str, Any]:
    decision = record.decision
    view = {
        "call_id": decision["call_id"],
        "status": _status(record),
        "rule": decision["rule"],
        "reason": decision["reason"],
        "halt": decision["rule"] in HALTING_RULES,
    }
    if record.approval and record.approval.get("blocked_by"):
        view["blocked_by"] = record.approval["blocked_by"]
    return view


class Gateway:
    """The server's logic, apart from HTTP so it can be tested (and raced) directly."""

    def __init__(
        self, policy: Policy, pool: ConnectionPool, per_tenant: int = PER_TENANT_REQUESTS, wait: float = SLOT_WAIT_SECONDS
    ) -> None:
        self._policy = policy
        self._pool = pool
        self._per_tenant, self._wait = per_tenant, wait
        self._slots: dict[str, threading.BoundedSemaphore] = {}
        self._slots_guard = threading.Lock()

    @contextmanager
    def _slot(self, tenant: str):
        """At most a few requests per tenant at once, so one tenant cannot use up every pooled connection."""
        with self._slots_guard:
            slots = self._slots.setdefault(tenant, threading.BoundedSemaphore(self._per_tenant))
        if not slots.acquire(timeout=self._wait):
            raise Busy(tenant)
        try:
            yield
        finally:
            slots.release()

    def ping(self) -> None:
        with self._pool.connection(timeout=3) as conn:
            conn.execute("SELECT 1")

    @contextmanager
    def _locked_run(self, tenant: str, run: str):
        """A connection holding the run's advisory lock; one request at a time per run."""
        key = lock_key(run)
        with self._slot(tenant), self._pool.connection() as conn:
            conn.execute("SELECT pg_advisory_lock(%s)", (key,))
            try:
                yield conn
            finally:
                conn.execute("SELECT pg_advisory_unlock(%s)", (key,))

    def _load(self, conn: psycopg.Connection, run: str) -> dict[str, CallRecord]:
        if not verify(conn, run).ok:
            raise ChainBroken(run)
        return _fold(events(conn, run))

    def decide_call(
        self, tenant: str, run_id: str, call_id: str, tool: str, args: dict[str, Any], definition_sha256: str | None
    ) -> dict[str, Any]:
        run, request_sha256 = namespaced(tenant, run_id), fingerprint(tool, args)
        with self._locked_run(tenant, run) as conn:
            calls = self._load(conn, run)
            known = calls.get(call_id)
            if known:
                same = (known.decision["request_sha256"], known.decision.get("definition_sha256"))
                if same != (request_sha256, definition_sha256):
                    raise Conflict(f"call_id {call_id!r} was already used for a different call")
                return _view(known)  # a retry: the same answer, nothing new written
            if any(r.decision["rule"] in HALTING_RULES for r in calls.values()):
                return {"call_id": call_id, "status": "denied", "rule": "run_halted",
                        "reason": "the run was halted earlier", "halt": True}
            state = RunState(
                steps=len(calls), history=tuple(r.decision["request_sha256"] for r in calls.values()), spend=_spent(calls)
            )
            call = ToolCall(tenant_id=tenant, tool=tool, args=args, definition_sha256=definition_sha256)
            decision = decide(self._policy, call, state)
            payload: dict[str, Any] = {
                "call_id": call_id,
                "tool": tool,
                "verdict": decision.verdict.value,
                "rule": decision.rule,
                "reason": decision.reason,
                "request_sha256": request_sha256,  # a hash, not the arguments, which may be sensitive
                "amount": spend_amount(self._policy, call),
            }
            if definition_sha256 is not None:
                payload["definition_sha256"] = definition_sha256
            append(conn, run, tenant, f"{call_id}:decision", "decision", payload)
            return _view(CallRecord(payload))

    def call_status(self, tenant: str, run_id: str, call_id: str) -> dict[str, Any]:
        run = namespaced(tenant, run_id)
        with self._locked_run(tenant, run) as conn:
            record = self._load(conn, run).get(call_id)
        if record is None:
            raise NotFound(call_id)
        return _view(record)

    def answer(self, tenant: str, run_id: str, call_id: str, approved: bool, by: str) -> dict[str, Any]:
        run = namespaced(tenant, run_id)
        with self._locked_run(tenant, run) as conn:
            calls = self._load(conn, run)
            record = calls.get(call_id)
            if record is None:
                raise NotFound(call_id)
            if record.decision["verdict"] != Verdict.ASK:
                raise Conflict("this call is not waiting for approval")
            if record.approval is not None:
                raise Conflict("this call was already answered")
            halted = any(r.decision["rule"] in HALTING_RULES for r in calls.values())
            blocked = ("run_halted" if halted else self._cap_blocks(tenant, calls, record)) if approved else None
            payload: dict[str, Any] = {
                "call_id": call_id,
                "tool": record.decision["tool"],
                "approved": approved and blocked is None,
                "approved_by": by,
            }
            if blocked:
                payload["blocked_by"] = blocked
            append(conn, run, tenant, f"{call_id}:approval", "approval", payload)
            record.approval = payload
            return _view(record)

    def _cap_blocks(self, tenant: str, calls: Mapping[str, CallRecord], record: CallRecord) -> str | None:
        """An approval is a new spend, so the cap is checked again against what was spent since the ask."""
        tenant_policy = self._policy.tenants.get(tenant)
        if tenant_policy is None:
            return "unknown_tenant"
        cap = tenant_policy.budget.max_spend
        if cap is not None and _spent(calls) + record.decision["amount"] > cap:
            return "spend_cap_exceeded"
        return None

    def pending(self, tenant: str, limit: int = PENDING_LIMIT, offset: int = 0) -> list[dict[str, Any]]:
        """Calls of this tenant's server runs that are waiting for an answer, oldest first, one page at a time."""
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
            payload = json.loads(text)
            result.append(
                {
                    "run_id": json.loads(run)[1],
                    "call_id": payload["call_id"],
                    "tool": payload["tool"],
                    "rule": payload["rule"],
                    "reason": payload["reason"],
                    "request_sha256": payload["request_sha256"],  # the arguments are not stored; check them against this
                }
            )
        return result

    def audit(self, tenant: str, run_id: str) -> dict[str, Any]:
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

    @field_validator("args")
    @classmethod
    def _plain_json_only(cls, args: dict[str, Any]) -> dict[str, Any]:
        check_json(args)
        return args


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approved: StrictBool


def create_app(policy: Policy, keys: Sequence[KeyEntry], database_url: str) -> FastAPI:
    require_project_database(database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pool = open_pool(database_url)
        app.state.gateway = Gateway(policy, pool)
        yield
        pool.close()

    # The interactive docs pages are off: they add surface and the API is small.
    app = FastAPI(title="portcullis", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def gateway() -> Gateway:
        return app.state.gateway

    def who_is(request: Request) -> Principal | None:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        return authenticate(keys, token.strip()) if scheme.lower() == "bearer" else None

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
        # Authenticate before anything reads or validates the body, so strangers learn nothing from error text.
        if request.url.path.startswith("/v1/") and who_is(request) is None:
            return JSONResponse(
                {"detail": "missing or invalid API key"}, status_code=401, headers={"WWW-Authenticate": "Bearer"}
            )
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
            {"detail": "too many concurrent requests for this tenant"}, status_code=429, headers={"Retry-After": "1"}
        )

    @app.exception_handler(NotFound)
    async def _not_found(request: Request, exc: NotFound):
        return JSONResponse({"detail": "no such call"}, status_code=404)

    @app.exception_handler(ChainBroken)
    async def _broken(request: Request, exc: ChainBroken):
        return JSONResponse({"detail": "the audit chain for this run does not verify"}, status_code=503)

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
        return {"status": "ok"}

    @app.post("/v1/runs/{run_id}/calls")
    def create_call(
        body: CallRequest, run_id: str = name(), who: Principal = Depends(needs("agent")), gw: Gateway = Depends(gateway)
    ):
        return gw.decide_call(who.tenant, run_id, body.call_id, body.tool, body.args, body.definition_sha256)

    @app.get("/v1/runs/{run_id}/calls/{call_id}")
    def get_call(
        run_id: str = name(), call_id: str = call_name(), who: Principal = Depends(either), gw: Gateway = Depends(gateway)
    ):
        return gw.call_status(who.tenant, run_id, call_id)

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
    """Build the app from PORTCULLIS_POLICY, PORTCULLIS_KEYS and PORTCULLIS_DATABASE_URL. Nothing has a default."""
    policy_path, keys_path, url = (_setting(env, n) for n in ("PORTCULLIS_POLICY", "PORTCULLIS_KEYS", "PORTCULLIS_DATABASE_URL"))
    try:
        policy = load_policy(Path(policy_path))
    except HookError as exc:
        raise ValueError(str(exc)) from exc
    return create_app(policy, load_keys(Path(keys_path)), url)


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
