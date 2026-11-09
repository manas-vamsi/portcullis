"""Shared fixtures.

Postgres tests run only when PORTCULLIS_DATABASE_URL is set. See .env.example, then:
    docker compose up -d --wait
    uv run --env-file .env pytest
"""

import os
from urllib.parse import urlparse

import psycopg
import pytest
from langgraph.checkpoint.postgres import PostgresSaver

CREATE_EFFECTS_TABLE = """
CREATE TABLE IF NOT EXISTS test_effects (
    run_id text NOT NULL,
    effect text NOT NULL,
    at timestamptz NOT NULL DEFAULT now()
)
"""


@pytest.fixture(scope="session")
def postgres_url() -> str:
    url = os.environ.get("PORTCULLIS_DATABASE_URL")
    if not url:
        pytest.skip("PORTCULLIS_DATABASE_URL not set (see .env.example); Postgres tests skipped")
    database = urlparse(url).path.lstrip("/")
    if not database.startswith("portcullis"):
        # Guard against pointing the tests at some other database on this machine.
        pytest.fail(f"refusing to run against database {database!r}: its name must start with 'portcullis'")
    with PostgresSaver.from_conn_string(url) as saver:
        saver.setup()
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(CREATE_EFFECTS_TABLE)
    return url
