"""Tamper-evident audit log: one hash chain per run, in Postgres.

Each row stores the hash of the previous row and its own hash over every field, so editing a
row, deleting one from the middle, or reordering them breaks the chain, and verify() reports
where. Events have deterministic ids, so a repeated append (LangGraph re-runs an interrupted
node) is recorded once. Reusing an id for different content (another tenant, event or payload)
raises AuditConflict instead of being dropped.

Honest limits. This is tamper-evident, not tamper-proof:
- Deleting the last rows of a run, or a whole run, is not detected unless you keep the head
  hash somewhere else (verify() returns it) and compare.
- Someone with database write access can rewrite a whole chain and recompute every hash.
- verify() on a run that has no rows reports ok with zero rows. An empty log and a missing log
  look the same, so compare the row count or head hash against what you expect.
- Rows hold a hash of each request, not its arguments, because arguments may be sensitive.
ponytail: chains are per run, so appends serialize per run only. Add a global chain or periodic
anchors if cross-run deletion needs detecting.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import psycopg
from psycopg.pq import TransactionStatus

GENESIS = "0" * 64

CREATE_AUDIT_LOG = """
CREATE TABLE IF NOT EXISTS audit_log (
    run_id text NOT NULL,
    idx integer NOT NULL,
    event_id text NOT NULL,
    tenant_id text NOT NULL,
    event text NOT NULL,
    ts text NOT NULL,
    payload text NOT NULL,
    prev_hash text NOT NULL,
    hash text NOT NULL,
    PRIMARY KEY (run_id, idx),
    UNIQUE (run_id, event_id)
)
"""


@dataclass(frozen=True)
class Verification:
    ok: bool
    rows: int
    head: str  # hash of the last row; keep a copy elsewhere to detect truncation
    problem: str | None


class AuditConflict(Exception):
    """An event id was reused for different content. Refused, so a changed answer cannot vanish from the log."""


def require_project_database(url: str) -> None:
    """Refuse a database not named for this project, in case a URL points at someone else's."""
    if not urlparse(url).path.lstrip("/").startswith("portcullis"):
        raise ValueError("the database name must start with 'portcullis'")


def create_audit_log(conn: psycopg.Connection) -> None:
    conn.execute(CREATE_AUDIT_LOG)


def _row_hash(prev: str, idx: int, run_id: str, tenant_id: str, event_id: str, event: str, ts: str, payload: str) -> str:
    return hashlib.sha256(json.dumps([prev, idx, run_id, tenant_id, event_id, event, ts, payload]).encode()).hexdigest()


def append(conn: psycopg.Connection, run_id: str, tenant_id: str, event_id: str, event: str, payload: dict[str, Any]) -> str:
    """Add an event to the run's chain and return its hash. A repeated event_id returns the existing hash.

    `conn` must be autocommit and outside any transaction block. The payload must be JSON data.
    """
    if not conn.autocommit or conn.info.transaction_status != TransactionStatus.IDLE:
        raise ValueError("audit.append needs an autocommit connection outside a transaction block")
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))  # TypeError for non-JSON: nothing is written
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"audit:{run_id}",))  # one writer per run
        existing = conn.execute(
            "SELECT hash, tenant_id, event, payload FROM audit_log WHERE run_id = %s AND event_id = %s",
            (run_id, event_id),
        ).fetchone()
        if existing:
            if existing[1:] != (tenant_id, event, body):
                raise AuditConflict(f"event {event_id!r} in run {run_id!r} was already recorded with different content")
            return existing[0]
        last = conn.execute(
            "SELECT idx, hash FROM audit_log WHERE run_id = %s ORDER BY idx DESC LIMIT 1", (run_id,)
        ).fetchone()
        idx, prev = (last[0] + 1, last[1]) if last else (0, GENESIS)
        ts = datetime.now(timezone.utc).isoformat()
        row_hash = _row_hash(prev, idx, run_id, tenant_id, event_id, event, ts, body)
        conn.execute(
            "INSERT INTO audit_log (run_id, idx, event_id, tenant_id, event, ts, payload, prev_hash, hash) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (run_id, idx, event_id, tenant_id, event, ts, body, prev, row_hash),
        )
    return row_hash


def verify(conn: psycopg.Connection, run_id: str) -> Verification:
    rows = conn.execute(
        "SELECT idx, event_id, tenant_id, event, ts, payload, prev_hash, hash "
        "FROM audit_log WHERE run_id = %s ORDER BY idx",
        (run_id,),
    ).fetchall()
    prev = GENESIS
    for position, (idx, event_id, tenant_id, event, ts, payload, prev_hash, stored) in enumerate(rows):
        if idx != position:
            return Verification(False, len(rows), prev, f"row {position} is missing (next row is {idx})")
        if prev_hash != prev:
            return Verification(False, len(rows), prev, f"row {idx} does not follow the row before it")
        if _row_hash(prev_hash, idx, run_id, tenant_id, event_id, event, ts, payload) != stored:
            return Verification(False, len(rows), prev, f"row {idx} was changed")
        prev = stored
    return Verification(True, len(rows), prev, None)


@dataclass(frozen=True)
class AuditTrail:
    """The audit log for one run of one tenant."""

    conn: psycopg.Connection
    run_id: str
    tenant_id: str

    def record(self, event_id: str, event: str, payload: dict[str, Any]) -> None:
        append(self.conn, self.run_id, self.tenant_id, event_id, event, payload)
