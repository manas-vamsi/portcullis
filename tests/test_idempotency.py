import math
from decimal import Decimal
from uuid import uuid4

import psycopg
import pytest

from portcullis.idempotency import KeyReuseError, idempotency_key, run_once


def test_key_is_deterministic_and_depends_on_run_and_call():
    assert idempotency_key("r", "c") == idempotency_key("r", "c")
    assert idempotency_key("r", "c") != idempotency_key("r", "d")
    assert idempotency_key("r", "c") != idempotency_key("s", "c")
    assert idempotency_key("ab", "c") != idempotency_key("a", "bc")  # no delimiter ambiguity


@pytest.fixture
def conn(postgres_url):
    with psycopg.connect(postgres_url, autocommit=True) as connection:
        yield connection


@pytest.fixture
def run_id(conn):
    rid = f"run-{uuid4()}"
    yield rid
    conn.execute("DELETE FROM tool_effects WHERE run_id = %s", (rid,))


class Provider:
    """Counts calls and remembers the key it was given."""

    def __init__(self, result=None, fail_first=False):
        self.keys: list[str] = []
        self._result = {"sent": 1} if result is None else result
        self._fail_first = fail_first

    def __call__(self, key: str):
        self.keys.append(key)
        if self._fail_first and len(self.keys) == 1:
            raise RuntimeError("provider down")
        return self._result


def test_runs_the_call_once_and_replays_the_stored_result(conn, run_id):
    provider = Provider()

    first = run_once(conn, run_id, "c1", "notify", {"to": "a"}, provider)
    second = run_once(conn, run_id, "c1", "notify", {"to": "a"}, provider)

    assert first == second == {"sent": 1}
    assert len(provider.keys) == 1


def test_the_provider_receives_the_idempotency_key(conn, run_id):
    provider = Provider()

    run_once(conn, run_id, "c1", "notify", {}, provider)

    assert provider.keys == [idempotency_key(run_id, "c1")]


def test_a_failed_call_is_retried_with_the_same_key(conn, run_id):
    provider = Provider(fail_first=True)

    with pytest.raises(RuntimeError):
        run_once(conn, run_id, "c1", "notify", {}, provider)
    status = conn.execute("SELECT status FROM tool_effects WHERE run_id = %s", (run_id,)).fetchone()[0]
    assert status == "started"  # the outcome is unknown, so the row is not marked done
    retried = run_once(conn, run_id, "c1", "notify", {}, provider)

    assert retried == {"sent": 1}
    assert len(provider.keys) == 2
    assert provider.keys[0] == provider.keys[1]


@pytest.mark.parametrize(
    ("tool", "args"), [("notify", {"to": "b"}), ("charge", {"to": "a"})], ids=["other_args", "other_tool"]
)
def test_reusing_a_call_id_for_a_different_request_is_refused(conn, run_id, tool, args):
    provider = Provider()
    run_once(conn, run_id, "c1", "notify", {"to": "a"}, provider)

    with pytest.raises(KeyReuseError):
        run_once(conn, run_id, "c1", tool, args, provider)

    assert len(provider.keys) == 1


def test_a_result_that_is_not_plain_json_is_stored_and_replayed_without_calling_again(conn, run_id):
    # If storing the result could fail after the tool ran, every retry would repeat the side effect.
    provider = Provider(result={"amount": Decimal("1.50"), "pair": (1, 2), "nul": "a\x00b"})

    first = run_once(conn, run_id, "c1", "charge", {}, provider)
    second = run_once(conn, run_id, "c1", "charge", {}, provider)

    assert first == second == {"amount": "1.50", "pair": [1, 2], "nul": "a\x00b"}
    assert len(provider.keys) == 1


def test_a_result_json_cannot_encode_at_all_is_stored_as_its_repr_without_repeating_the_call(conn, run_id):
    circular: list = []
    circular.append(circular)
    provider = Provider(result=circular)

    first = run_once(conn, run_id, "c1", "charge", {}, provider)
    second = run_once(conn, run_id, "c1", "charge", {}, provider)

    assert first == second == repr(circular)
    assert len(provider.keys) == 1


def test_nan_in_a_result_does_not_make_the_call_repeat(conn, run_id):
    provider = Provider(result={"x": math.nan})

    run_once(conn, run_id, "c1", "charge", {}, provider)
    replayed = run_once(conn, run_id, "c1", "charge", {}, provider)

    assert math.isnan(replayed["x"])
    assert len(provider.keys) == 1


def test_when_another_worker_finishes_first_its_result_wins(conn, run_id):
    def slow_call(key):
        # Meanwhile a retry of the same call finished and recorded its result.
        conn.execute(
            "UPDATE tool_effects SET status = 'done', result_json = %s WHERE idempotency_key = %s",
            ('"first"', key),
        )
        return "second"

    assert run_once(conn, run_id, "c1", "notify", {}, slow_call) == "first"
    assert run_once(conn, run_id, "c1", "notify", {}, Provider()) == "first"


def test_the_same_call_id_in_two_runs_is_two_calls(conn, run_id):
    provider = Provider()
    other_run = f"{run_id}-other"
    try:
        run_once(conn, run_id, "c1", "notify", {}, provider)
        run_once(conn, other_run, "c1", "notify", {}, provider)
    finally:
        conn.execute("DELETE FROM tool_effects WHERE run_id = %s", (other_run,))

    assert len(provider.keys) == 2
    assert provider.keys[0] != provider.keys[1]


def test_a_connection_that_is_not_autocommit_is_rejected(postgres_url):
    with psycopg.connect(postgres_url) as connection:  # autocommit off
        with pytest.raises(ValueError, match="autocommit"):
            run_once(connection, "r", "c", "t", {}, Provider())


def test_a_call_inside_a_transaction_block_is_rejected(conn):
    # Autocommit is on, but inside conn.transaction() the claim would not be durable yet.
    with conn.transaction():
        with pytest.raises(ValueError, match="autocommit"):
            run_once(conn, "r", "c", "t", {}, Provider())
