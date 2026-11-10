"""Kill a run mid-step, resume it in a new process, and count side effects.

Delivery is at-least-once plus idempotency keys, not exactly-once. The third scenario
pins the known limit: a provider that ignores keys can see one duplicate.
"""

import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from langgraph.checkpoint.postgres import PostgresSaver

WORKER = Path(__file__).parent / "kill_worker.py"
KILLED = 137

# name -> (where the worker is killed, how the fake provider treats the idempotency key)
SCENARIOS = {
    "keyed_provider_killed_inside_call": ("inside_call", "keyed"),
    "naive_provider_killed_after_ledger": ("after_ledger", "naive"),
    "naive_provider_killed_inside_call": ("inside_call", "naive"),
}


def run_worker(url: str, run_id: str, mode: str, crash_at: str, provider: str):
    return subprocess.run(
        [sys.executable, str(WORKER), mode],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "PORTCULLIS_DATABASE_URL": url,
            "RUN_ID": run_id,
            "CRASH_AT": crash_at,
            "PROVIDER": provider,
        },
    )


def kill_then_resume(url: str, crash_at: str, provider: str):
    run_id = f"run-{uuid4()}"
    killed = run_worker(url, run_id, "start", crash_at, provider)
    resumed = run_worker(url, run_id, "resume", "", provider)
    with psycopg.connect(url, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT effect, count(*) FROM test_effects WHERE run_id = %s GROUP BY effect", (run_id,)
        ).fetchall()
        conn.execute("DELETE FROM test_effects WHERE run_id = %s", (run_id,))
        conn.execute("DELETE FROM tool_effects WHERE run_id = %s", (run_id,))
    with PostgresSaver.from_conn_string(url) as saver:
        saver.delete_thread(run_id)
    return killed, resumed, dict(rows)


@pytest.fixture(scope="module")
def outcomes(postgres_url):
    return {name: kill_then_resume(postgres_url, *how) for name, how in SCENARIOS.items()}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_a_killed_run_is_finished_by_a_new_process(outcomes, scenario):
    killed, resumed, _ = outcomes[scenario]

    assert killed.returncode == KILLED
    assert resumed.returncode == 0, resumed.stderr


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_resume_does_not_rerun_nodes_that_already_finished(outcomes, scenario):
    _, _, effects = outcomes[scenario]

    assert effects["reserve"] == 1
    assert effects["confirm"] == 1


def test_keyed_provider_sees_one_charge_when_killed_mid_call(outcomes):
    # The retry reuses the same idempotency key, so the provider ignores the repeat.
    _, _, effects = outcomes["keyed_provider_killed_inside_call"]

    assert effects["charge"] == 1


def test_ledger_stops_a_repeat_even_for_a_provider_that_ignores_keys(outcomes):
    # Killed after the ledger recorded the result: resume replays it and never calls the provider.
    _, _, effects = outcomes["naive_provider_killed_after_ledger"]

    assert effects["charge"] == 1


def test_known_limit_a_provider_that_ignores_keys_can_see_a_duplicate(outcomes):
    # Killed after the call but before the ledger recorded it: the outcome is unknown, so it is retried.
    _, _, effects = outcomes["naive_provider_killed_inside_call"]

    assert effects["charge"] == 2
