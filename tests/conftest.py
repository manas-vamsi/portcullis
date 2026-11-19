"""Shared fixtures.

Postgres tests run only when PORTCULLIS_DATABASE_URL is set. See .env.example, then:
    docker compose up -d --wait
    uv run --env-file .env pytest
"""

import os
from urllib.parse import urlparse
from uuid import uuid4

import psycopg
import pytest
from langgraph.checkpoint.postgres import PostgresSaver

# Stand-in for an external system. idem_key lets a "keyed" fake provider deduplicate like a payment API.
TEST_EFFECTS_SETUP = (
    """CREATE TABLE IF NOT EXISTS test_effects (
        run_id text NOT NULL,
        effect text NOT NULL,
        at timestamptz NOT NULL DEFAULT now()
    )""",
    "ALTER TABLE test_effects ADD COLUMN IF NOT EXISTS idem_key text",
    "CREATE UNIQUE INDEX IF NOT EXISTS test_effects_idem_key ON test_effects (idem_key) WHERE idem_key IS NOT NULL",
)


class MaskedUrl(str):
    """The database URL as a string that never shows itself, so a failing test cannot print the password."""

    def __repr__(self) -> str:
        return "'<database url>'"


@pytest.fixture(scope="session")
def postgres_url() -> str:
    from portcullis.audit import create_audit_log
    from portcullis.idempotency import create_ledger

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
        create_ledger(conn)
        create_audit_log(conn)
        for statement in TEST_EFFECTS_SETUP:
            conn.execute(statement)
    return MaskedUrl(url)


@pytest.fixture
def conn(postgres_url):
    """An autocommit connection, as the ledger requires."""
    with psycopg.connect(postgres_url, autocommit=True) as connection:
        yield connection


@pytest.fixture
def run_id(conn):
    """A fresh run id whose ledger rows are removed afterwards."""
    rid = f"run-{uuid4()}"
    yield rid
    conn.execute("DELETE FROM tool_effects WHERE run_id = %s", (rid,))
    conn.execute("DELETE FROM audit_log WHERE run_id = %s", (rid,))
