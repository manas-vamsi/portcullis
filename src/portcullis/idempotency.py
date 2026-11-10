"""Idempotent tool execution: at-least-once delivery plus idempotency keys. NOT exactly-once.

Each tool call gets a deterministic key from (run_id, call_id), so a retry after a crash
carries the same key. A Postgres ledger records each call as 'started', then 'done' with
its result.

- Already 'done': the stored result is returned and the tool is not called again.
- Only 'started': the earlier attempt may or may not have reached the outside world (it may
  have crashed, or may just be slow), so the tool is called again with the same key. The
  external service must deduplicate by that key; one that ignores keys can see a duplicate.
- If two attempts both finish, the first to record its result wins and both return it.

Use a call_id that stays the same across a resume (for example the model's tool_call_id from
checkpointed state), never one made fresh on each attempt.
"""

import hashlib
import json
from collections.abc import Callable
from typing import Any

import psycopg
from psycopg.pq import TransactionStatus

from portcullis.policy import fingerprint

CREATE_LEDGER = """
CREATE TABLE IF NOT EXISTS tool_effects (
    idempotency_key text PRIMARY KEY,
    run_id text NOT NULL,
    call_id text NOT NULL,
    tool text NOT NULL,
    request_sha256 text NOT NULL,
    status text NOT NULL CHECK (status IN ('started', 'done')),
    result_json text,
    created_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
)
"""

CLAIM = """
INSERT INTO tool_effects (idempotency_key, run_id, call_id, tool, request_sha256, status)
VALUES (%s, %s, %s, %s, %s, 'started')
ON CONFLICT (idempotency_key) DO NOTHING
"""

FINISH = """
UPDATE tool_effects SET status = 'done', result_json = %s, finished_at = now()
WHERE idempotency_key = %s AND status = 'started'
"""


class KeyReuseError(Exception):
    """The same call id was used for a different tool or arguments."""


def idempotency_key(run_id: str, call_id: str) -> str:
    # JSON, not a delimiter, so ("ab", "c") and ("a", "bc") cannot collide.
    return hashlib.sha256(json.dumps([run_id, call_id]).encode()).hexdigest()


def create_ledger(conn: psycopg.Connection) -> None:
    conn.execute(CREATE_LEDGER)


def _encode(result: Any) -> str:
    """JSON text for any result.

    Kept as text, not jsonb, so nothing a tool returns (NaN, a NUL character, a Decimal) can
    make the 'done' update fail after the tool already ran. A failed update would leave the
    row 'started' and every retry would repeat the side effect.
    """
    try:
        return json.dumps(result, default=str)
    except (TypeError, ValueError):  # for example a circular reference
        return json.dumps(repr(result))


def run_once(
    conn: psycopg.Connection,
    run_id: str,
    call_id: str,
    tool: str,
    args: dict[str, Any],
    call: Callable[[str], Any],
) -> Any:
    """Call `call(key)` unless this (run_id, call_id) is already done; return its result as JSON data.

    `conn` must be autocommit and outside any transaction block, so the claim is durable before
    the tool is called. The first call and any replay return the same value: the tool's result
    after a JSON round trip (tuples become lists, Decimals become strings).
    ponytail: two workers racing on one key can both call the tool; the provider must deduplicate
    by key. A lock would mean holding a transaction open across an external call.
    """
    if not conn.autocommit or conn.info.transaction_status != TransactionStatus.IDLE:
        raise ValueError("run_once needs an autocommit connection outside a transaction block")
    key = idempotency_key(run_id, call_id)
    request = fingerprint(tool, args)

    conn.execute(CLAIM, (key, run_id, call_id, tool, request))
    status, stored_request, stored = conn.execute(
        "SELECT status, request_sha256, result_json FROM tool_effects WHERE idempotency_key = %s", (key,)
    ).fetchone()

    if stored_request != request:
        raise KeyReuseError(f"call id {call_id!r} in run {run_id!r} was already used for a different request")
    if status == "done":
        return json.loads(stored)

    stored = _encode(call(key))
    if conn.execute(FINISH, (stored, key)).rowcount == 0:
        # Another attempt finished first. Its result is the one on record, so return that.
        stored = conn.execute(
            "SELECT result_json FROM tool_effects WHERE idempotency_key = %s", (key,)
        ).fetchone()[0]
    return json.loads(stored)
