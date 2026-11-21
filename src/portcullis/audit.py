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
from collections.abc import Sequence
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


def namespaced(tenant: str, run_id: str) -> str:
    """The run name the server stores in the audit log. JSON keeps ("a", "b/c") and ("a/b", "c") apart."""
    return json.dumps([tenant, run_id])


def hook_run(tenant: str, session: str) -> str:
    """The run name the Claude Code hook uses. Three parts, so it can never be mistaken for a server run
    (two parts), and the server's approval queue never lists the hook's unanswered asks."""
    return json.dumps(["claude-code-hook", tenant, session])


def mcp_run(tenant: str, session: str) -> str:
    """The run name for a session of the MCP proxy. Three parts, like the hook's, so it is never a server run."""
    return json.dumps(["mcp-proxy", tenant, session])


def lock_key(name: str) -> int:
    """A 64-bit advisory lock number for a run: a sha256 prefix, so a client cannot guess or collide it."""
    return int.from_bytes(hashlib.sha256(f"run:{name}".encode()).digest()[:8], "big", signed=True)


def require_project_database(url: str) -> None:
    """Refuse a database not named for this project, in case a URL points at someone else's.

    The name is the one the driver will actually connect to, so `.../portcullis?dbname=postgres` is refused too."""
    from psycopg.conninfo import conninfo_to_dict

    try:
        name = str(conninfo_to_dict(url).get("dbname") or "")
    except Exception:  # an unparseable URL is not a project database either
        name = ""
    if not (urlparse(url).path.lstrip("/").startswith("portcullis") and name.startswith("portcullis")):
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


ROW_COLUMNS = "idx, event_id, tenant_id, event, ts, payload, prev_hash, hash"
EVENT_FIELDS = ("idx", "event_id", "tenant_id", "event", "ts", "payload")


def fetch_rows(conn: psycopg.Connection, run_id: str, start: int = 0) -> list[tuple[Any, ...]]:
    """A run's rows from position `start` on, as (idx, event_id, tenant_id, event, ts, payload, prev_hash, hash)."""
    return conn.execute(
        f"SELECT {ROW_COLUMNS} FROM audit_log WHERE run_id = %s AND idx >= %s ORDER BY idx", (run_id, start)
    ).fetchall()


def row_dicts(rows: Sequence[tuple[Any, ...]]) -> list[dict[str, Any]]:
    """Rows as the dicts `events()` returns (without the hashes)."""
    return [dict(zip(EVENT_FIELDS, row)) for row in rows]


def events(conn: psycopg.Connection, run_id: str) -> list[dict[str, Any]]:
    """Every row of a run in order, as stored. Does not check the chain; call verify() for that."""
    return row_dicts(fetch_rows(conn, run_id))


def payload_data(text: str) -> Any:
    """A row's payload as data. A tampered row may not be JSON; it is then returned as the stored text."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def check_rows(run_id: str, rows: Sequence[tuple[Any, ...]], start_position: int = 0, prev: str = GENESIS) -> Verification:
    """Check rows that start at `start_position` and chain onto `prev` (the previous row's hash).

    The whole chain is start_position 0 and the genesis hash; to extend a chain already verified up to a
    known head, pass where the new rows begin and that head."""
    for offset, (idx, event_id, tenant_id, event, ts, payload, prev_hash, stored) in enumerate(rows):
        position = start_position + offset
        total = start_position + len(rows)
        if idx != position:
            return Verification(False, total, prev, f"row {position} is missing (next row is {idx})")
        if prev_hash != prev:
            return Verification(False, total, prev, f"row {idx} does not follow the row before it")
        if _row_hash(prev_hash, idx, run_id, tenant_id, event_id, event, ts, payload) != stored:
            return Verification(False, total, prev, f"row {idx} was changed")
        prev = stored
    return Verification(True, start_position + len(rows), prev, None)


def verify(conn: psycopg.Connection, run_id: str) -> Verification:
    return check_rows(run_id, fetch_rows(conn, run_id))


def read_verified(conn: psycopg.Connection, run_id: str) -> tuple[Verification, list[dict[str, Any]]]:
    """Verify a run and return its rows from a single fetch (calling verify() and events() reads them twice)."""
    rows = fetch_rows(conn, run_id)
    return check_rows(run_id, rows), row_dicts(rows)


# A digest of every field of a row, computed by the database so the rows need not be sent to be compared.
_ROW_DIGEST = (
    "encode(sha256(convert_to(json_build_array(idx, event_id, tenant_id, event, ts, payload, prev_hash, hash)::text, 'UTF8')), 'hex')"
)


def history_checksums(conn: psycopg.Connection, run_id: str, prefix: int) -> tuple[int, str, int, str]:
    """(rows, digest of all rows, rows before `prefix`, digest of those). Digests are '' for no rows.

    A digest covers every stored field of every row, in order. If the digest of the first `prefix` rows is the
    same as when they were last verified, those rows have not changed by a single byte, so they need not be
    verified again."""
    row = conn.execute(
        f"WITH h AS (SELECT idx, {_ROW_DIGEST} AS d FROM audit_log WHERE run_id = %s) "
        "SELECT count(*), "
        "coalesce(encode(sha256(convert_to(string_agg(d, '' ORDER BY idx), 'UTF8')), 'hex'), ''), "
        "count(*) FILTER (WHERE idx < %s), "
        "coalesce(encode(sha256(convert_to(string_agg(d, '' ORDER BY idx) FILTER (WHERE idx < %s), 'UTF8')), 'hex'), '') "
        "FROM h",
        (run_id, prefix, prefix),
    ).fetchone()
    return int(row[0]), row[1], int(row[2]), row[3]


@dataclass(frozen=True)
class AuditTrail:
    """The audit log for one run of one tenant."""

    conn: psycopg.Connection
    run_id: str
    tenant_id: str

    def record(self, event_id: str, event: str, payload: dict[str, Any]) -> None:
        append(self.conn, self.run_id, self.tenant_id, event_id, event, payload)
