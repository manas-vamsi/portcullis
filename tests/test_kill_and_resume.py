"""Kill a run mid-step, resume it in a new process, and count side effects."""

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


def run_worker(url: str, run_id: str, mode: str, crash: bool) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(WORKER), mode],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PORTCULLIS_DATABASE_URL": url, "RUN_ID": run_id, "CRASH": "1" if crash else "0"},
    )


@pytest.fixture(scope="module")
def killed_then_resumed(postgres_url):
    run_id = f"run-{uuid4()}"
    killed = run_worker(postgres_url, run_id, "start", crash=True)
    resumed = run_worker(postgres_url, run_id, "resume", crash=False)
    with psycopg.connect(postgres_url, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT effect, count(*) FROM test_effects WHERE run_id = %s GROUP BY effect", (run_id,)
        ).fetchall()
        conn.execute("DELETE FROM test_effects WHERE run_id = %s", (run_id,))
    with PostgresSaver.from_conn_string(postgres_url) as saver:
        saver.delete_thread(run_id)
    return killed, resumed, dict(rows)


def test_the_worker_dies_mid_run_and_a_new_process_finishes_it(killed_then_resumed):
    killed, resumed, _ = killed_then_resumed

    assert killed.returncode == KILLED
    assert resumed.returncode == 0, resumed.stderr


def test_resume_does_not_rerun_nodes_that_already_finished(killed_then_resumed):
    _, _, effects = killed_then_resumed

    assert effects["reserve"] == 1
    assert effects["confirm"] == 1


@pytest.mark.xfail(
    strict=True,
    reason="no idempotency keys yet: the node killed after its side effect runs again on resume",
)
def test_resume_never_repeats_the_side_effect_of_the_interrupted_node(killed_then_resumed):
    _, _, effects = killed_then_resumed

    assert effects["charge"] == 1
