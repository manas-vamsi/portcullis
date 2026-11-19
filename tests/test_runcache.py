"""Reading a run from its audit log, reusing earlier work only while the stored history is byte-for-byte unchanged."""

import json
from dataclasses import replace

import pytest

import portcullis.runcache as runcache
from portcullis.audit import append, check_rows, fetch_rows, history_checksums, namespaced, read_verified, verify
from portcullis.runcache import ChainBroken, RunReader


def decision(call_id, verdict="allow", rule="allowed", amount=0):
    return {"call_id": call_id, "tool": "t", "verdict": verdict, "rule": rule, "reason": "why",
            "request_sha256": "r" + call_id, "amount": amount, "tokens": 0}


def add(conn, run, call_id, **kw):
    append(conn, run, "t", f"{call_id}:decision", "decision", decision(call_id, **kw))


def fetches(monkeypatch):
    """Record the starting index of every fetch of rows the reader makes."""
    starts, real = [], runcache.fetch_rows

    def spy(conn, run, start=0):
        starts.append(start)
        return real(conn, run, start)

    monkeypatch.setattr(runcache, "fetch_rows", spy)
    return starts


@pytest.fixture
def run(conn):
    name = namespaced("t", "cache-" + __import__("uuid").uuid4().hex)
    yield name
    conn.execute("DELETE FROM audit_log WHERE run_id = %s", (name,))


# --- the audit helpers the cache is built on ---


def test_check_rows_agrees_with_verify_and_can_start_from_a_known_head(conn, run):
    for i in range(5):
        add(conn, run, f"c{i}")
    rows = fetch_rows(conn, run)
    whole = check_rows(run, rows)
    assert (whole.ok, whole.rows, whole.head) == (True, 5, verify(conn, run).head)

    rest = check_rows(run, rows[3:], start_position=3, prev=rows[2][7])  # the last column is the row's own hash
    assert (rest.ok, rest.rows, rest.head) == (True, 5, whole.head)
    assert not check_rows(run, rows[3:], start_position=3, prev="0" * 64).ok  # not chained onto what it claims


def test_fetch_rows_can_start_part_way_down_the_chain(conn, run):
    for i in range(6):
        add(conn, run, f"c{i}")
    assert [r[0] for r in fetch_rows(conn, run, 4)] == [4, 5] and fetch_rows(conn, run, 99) == []


def test_read_verified_returns_the_verdict_and_the_rows_from_one_fetch(conn, run):
    for i in range(3):
        add(conn, run, f"c{i}")
    check, rows = read_verified(conn, run)
    assert check.ok and [json.loads(r["payload"])["call_id"] for r in rows] == ["c0", "c1", "c2"]
    assert set(rows[0]) == {"idx", "event_id", "tenant_id", "event", "ts", "payload"}


def test_the_checksum_changes_if_any_field_of_any_earlier_row_changes(conn, run):
    for i in range(4):
        add(conn, run, f"c{i}")
    before = history_checksums(conn, run, 3)
    assert before[0] == 4 and before[2] == 3 and len(before[1]) == 64 and len(before[3]) == 64
    assert history_checksums(conn, run, 3) == before  # stable

    for column, value in (("payload", "'{}'"), ("tenant_id", "'x'"), ("event", "'y'"), ("ts", "'z'"), ("event_id", "'q'")):
        conn.execute(f"UPDATE audit_log SET {column} = {value} WHERE run_id = %s AND idx = 1", (run,))
        assert history_checksums(conn, run, 3)[3] != before[3], column  # the prefix digest sees it
        conn.execute("DELETE FROM audit_log WHERE run_id = %s", (run,))
        for i in range(4):
            add(conn, run, f"c{i}")
        before = history_checksums(conn, run, 3)


def test_the_checksum_of_nothing_is_empty_not_an_error(conn, run):
    assert history_checksums(conn, run, 0) == (0, "", 0, "")


# --- the reader ---


def test_a_second_load_fetches_only_the_new_rows_and_returns_the_same_story(conn, run, monkeypatch):
    reader, starts = RunReader(), fetches(monkeypatch)
    for i in range(5):
        add(conn, run, f"c{i}")
    first = reader.load(conn, run)
    add(conn, run, "c5")
    add(conn, run, "c6", verdict="ask", rule="approval_required", amount=900)

    second = reader.load(conn, run)

    assert starts == [0, 5]  # the first load read everything; the second only what was new
    cold = RunReader().load(conn, run)
    assert list(second) == list(cold) == [f"c{i}" for i in range(7)] and list(first) == [f"c{i}" for i in range(5)]
    assert {k: v.decision for k, v in second.items()} == {k: v.decision for k, v in cold.items()}


def test_answers_and_executions_in_new_rows_attach_to_calls_already_cached(conn, run):
    reader = RunReader()
    add(conn, run, "a", verdict="ask", rule="approval_required", amount=900)
    reader.load(conn, run)
    append(conn, run, "t", "a:approval", "approval", {"call_id": "a", "tool": "t", "approved": True, "approved_by": "alice"})
    append(conn, run, "t", "a:executed", "executed", {"call_id": "a", "tool": "t"})
    record = reader.load(conn, run)["a"]
    assert record.approval["approved"] is True and record.executed is not None


def test_an_unchanged_history_is_not_read_again_at_all_beyond_the_checksum(conn, run, monkeypatch):
    reader, starts = RunReader(), fetches(monkeypatch)
    add(conn, run, "a")
    reader.load(conn, run)
    reader.load(conn, run)
    assert starts == [0, 1]  # the second fetch starts after the last known row and finds nothing


def test_editing_an_old_row_after_it_was_cached_is_caught_on_the_next_load(conn, run):
    reader = RunReader()
    for i in range(6):
        add(conn, run, f"c{i}")
    reader.load(conn, run)  # cached, and verified

    conn.execute("UPDATE audit_log SET payload = %s WHERE run_id = %s AND idx = 2", (json.dumps(decision("c2", "deny", "forbidden_argument")), run))

    with pytest.raises(ChainBroken):
        reader.load(conn, run)
    with pytest.raises(ChainBroken):  # and it stays caught: the cache was dropped, not repaired
        reader.load(conn, run)


def test_deleting_a_row_from_the_middle_is_caught_too(conn, run):
    reader = RunReader()
    for i in range(6):
        add(conn, run, f"c{i}")
    reader.load(conn, run)
    conn.execute("DELETE FROM audit_log WHERE run_id = %s AND idx = 3", (run,))
    with pytest.raises(ChainBroken):
        reader.load(conn, run)


def test_a_new_row_that_does_not_chain_onto_the_cached_head_is_caught(conn, run):
    reader = RunReader()
    for i in range(3):
        add(conn, run, f"c{i}")
    reader.load(conn, run)
    conn.execute(
        "INSERT INTO audit_log (run_id, idx, event_id, tenant_id, event, ts, payload, prev_hash, hash) "
        "VALUES (%s, 3, 'x:decision', 't', 'decision', 'now', %s, %s, %s)",
        (run, json.dumps(decision("x")), "f" * 64, "e" * 64),
    )
    with pytest.raises(ChainBroken):
        reader.load(conn, run)


def test_a_chain_that_verifies_but_tells_an_impossible_story_is_a_broken_run(conn, run):
    reader = RunReader()
    add(conn, run, "a")
    append(conn, run, "t", "ghost:approval", "approval", {"call_id": "ghost", "approved": True})
    with pytest.raises(ChainBroken):
        reader.load(conn, run)


def test_what_a_caller_does_to_the_returned_records_does_not_reach_the_cache(conn, run):
    reader = RunReader()
    add(conn, run, "a", verdict="ask", rule="approval_required", amount=900)
    first = reader.load(conn, run)
    first["a"].approval = {"call_id": "a", "approved": True}  # a caller records something that never reached the log
    first["phantom"] = first["a"]
    again = reader.load(conn, run)
    assert again["a"].approval is None and "phantom" not in again


def test_least_recently_used_runs_are_forgotten_beyond_the_capacity(conn, monkeypatch):
    reader, starts = RunReader(capacity=2), fetches(monkeypatch)
    names = [namespaced("t", "lru-" + __import__("uuid").uuid4().hex) for _ in range(3)]
    try:
        for name in names:
            add(conn, name, "a")
            reader.load(conn, name)
        starts.clear()
        reader.load(conn, names[2])  # still cached
        reader.load(conn, names[0])  # was evicted: read from the start again
        assert starts == [1, 0]
    finally:
        conn.execute("DELETE FROM audit_log WHERE run_id = ANY(%s)", (names,))


def test_rows_that_appear_between_the_checksum_and_the_fetch_are_used_but_not_cached(conn, run, monkeypatch):
    reader = RunReader()
    add(conn, run, "a")
    real = runcache.history_checksums

    def racing(connection, name, prefix):
        result = real(connection, name, prefix)
        add(connection, name, "late")  # another writer slips in after the checksum was taken
        return result

    monkeypatch.setattr(runcache, "history_checksums", racing)
    assert list(reader.load(conn, run)) == ["a", "late"]  # the verified rows are all used
    monkeypatch.setattr(runcache, "history_checksums", real)
    starts = fetches(monkeypatch)
    reader.load(conn, run)
    assert starts == [0]  # but nothing was cached on that uncertain basis: this read starts over


def test_a_reader_with_no_room_still_reads_correctly(conn, run):
    reader = RunReader(capacity=0)
    add(conn, run, "a")
    assert list(reader.load(conn, run)) == ["a"] and list(reader.load(conn, run)) == ["a"]


def test_the_cache_keeps_a_copy_of_each_record_not_the_callers_object(conn, run):
    reader = RunReader()
    add(conn, run, "a")
    record = reader.load(conn, run)["a"]
    assert replace(record) == record  # records are plain dataclasses: copies compare equal
