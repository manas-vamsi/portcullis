"""The hash-chained audit log."""

import json
from concurrent.futures import ThreadPoolExecutor

import psycopg
import pytest

from portcullis.audit import GENESIS, AuditConflict, append, verify


def add(conn, run_id, n, payload=None, tenant="acme"):
    return append(conn, run_id, tenant, f"e{n}", "decision", payload if payload is not None else {"n": n})


def test_an_empty_run_verifies_as_an_empty_chain(conn, run_id):
    result = verify(conn, run_id)

    assert (result.ok, result.rows, result.head, result.problem) == (True, 0, GENESIS, None)


def test_appended_rows_form_a_chain_that_verifies(conn, run_id):
    first = add(conn, run_id, 1)
    second = add(conn, run_id, 2)

    result = verify(conn, run_id)

    assert (result.ok, result.rows, result.head) == (True, 2, second)
    assert first != second
    prev_of_first = conn.execute("SELECT prev_hash FROM audit_log WHERE run_id = %s AND idx = 0", (run_id,)).fetchone()[0]
    assert prev_of_first == GENESIS


def test_the_same_event_id_is_recorded_once(conn, run_id):
    # LangGraph re-runs an interrupted node, which repeats its audit calls.
    first = add(conn, run_id, 1)
    again = add(conn, run_id, 1)

    assert again == first
    assert verify(conn, run_id).rows == 1


@pytest.mark.parametrize(
    "change",
    [{"payload": {"n": 2}}, {"event": "approval"}, {"tenant": "globex"}],
    ids=["other_payload", "other_event", "other_tenant"],
)
def test_reusing_an_event_id_for_different_content_is_refused_not_silently_deduplicated(conn, run_id, change):
    # Otherwise a second, different answer (say approved after denied) would vanish from the log.
    append(conn, run_id, "acme", "e1", "decision", {"n": 1})

    with pytest.raises(AuditConflict):
        append(
            conn,
            run_id,
            change.get("tenant", "acme"),
            "e1",
            change.get("event", "decision"),
            change.get("payload", {"n": 1}),
        )

    assert verify(conn, run_id).rows == 1


def test_known_limit_an_unknown_run_verifies_as_an_empty_chain(conn):
    assert verify(conn, "no-such-run-ever").rows == 0  # an empty log is not a missing log: compare against the expected head


@pytest.mark.parametrize(
    "tamper",
    [
        "UPDATE audit_log SET payload = '{\"n\":99}' WHERE run_id = %s AND idx = 1",
        "UPDATE audit_log SET event = 'approval' WHERE run_id = %s AND idx = 1",
        "UPDATE audit_log SET tenant_id = 'globex' WHERE run_id = %s AND idx = 1",
        "UPDATE audit_log SET ts = '2000-01-01T00:00:00+00:00' WHERE run_id = %s AND idx = 1",
        "UPDATE audit_log SET prev_hash = '" + "f" * 64 + "' WHERE run_id = %s AND idx = 1",
        "UPDATE audit_log SET hash = '" + "f" * 64 + "' WHERE run_id = %s AND idx = 1",
        "DELETE FROM audit_log WHERE run_id = %s AND idx = 1",
        "DELETE FROM audit_log WHERE run_id = %s AND idx = 0",
    ],
    ids=["payload", "event", "tenant", "timestamp", "prev_hash", "hash", "delete_middle", "delete_first"],
)
def test_any_edit_or_deletion_inside_the_chain_is_detected(conn, run_id, tamper):
    for n in range(3):
        add(conn, run_id, n)

    conn.execute(tamper, (run_id,))
    result = verify(conn, run_id)

    assert result.ok is False
    assert result.problem


def test_known_limit_deleting_the_last_row_is_not_detected_without_an_outside_copy_of_the_head(conn, run_id):
    add(conn, run_id, 1)
    head = add(conn, run_id, 2)

    conn.execute("DELETE FROM audit_log WHERE run_id = %s AND idx = 1", (run_id,))
    result = verify(conn, run_id)

    assert result.ok is True  # the chain is still consistent...
    assert result.head != head  # ...but a saved head hash exposes the truncation


def test_runs_have_separate_chains(conn, run_id):
    other = f"{run_id}-other"
    try:
        add(conn, run_id, 1)
        add(conn, other, 1)
        add(conn, other, 2)

        assert verify(conn, run_id).rows == 1
        assert verify(conn, other).rows == 2
    finally:
        conn.execute("DELETE FROM audit_log WHERE run_id = %s", (other,))


def test_a_payload_that_is_not_json_is_refused_and_writes_nothing(conn, run_id):
    with pytest.raises(TypeError):
        add(conn, run_id, 1, payload={"x": object()})

    assert verify(conn, run_id).rows == 0


def test_concurrent_writers_still_produce_one_valid_chain(postgres_url, conn, run_id):
    def work(worker: int) -> None:
        with psycopg.connect(postgres_url, autocommit=True) as c:
            for i in range(5):
                append(c, run_id, "acme", f"w{worker}-{i}", "decision", {"worker": worker})

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(work, range(4)))

    result = verify(conn, run_id)
    assert (result.ok, result.rows) == (True, 20), result.problem


def test_a_connection_that_is_not_autocommit_is_rejected(postgres_url, run_id):
    with psycopg.connect(postgres_url) as connection:
        with pytest.raises(ValueError, match="autocommit"):
            add(connection, run_id, 1)


def test_the_payload_is_stored_as_canonical_json(conn, run_id):
    add(conn, run_id, 1, payload={"b": 1, "a": [1, 2]})

    stored = conn.execute("SELECT payload FROM audit_log WHERE run_id = %s", (run_id,)).fetchone()[0]

    assert stored == '{"a":[1,2],"b":1}'
    assert json.loads(stored) == {"a": [1, 2], "b": 1}
