"""Test worker for the kill-and-resume tests. Not part of the library.

Runs reserve -> charge -> confirm on a Postgres checkpointer. `charge` goes through
portcullis.idempotency.run_once; the other steps write their effect directly.

usage: kill_worker.py start|resume
env:   PORTCULLIS_DATABASE_URL, RUN_ID,
       CRASH_AT  = "" | "inside_call" (killed right after the provider call, before the ledger says done)
                      | "after_ledger" (killed after the ledger says done, before LangGraph checkpoints)
       PROVIDER  = "keyed" (deduplicates by idempotency key) | "naive" (ignores the key)
"""

import json
import os
import sys
from typing import TypedDict

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

from portcullis.idempotency import run_once

URL = os.environ["PORTCULLIS_DATABASE_URL"]
RUN_ID = os.environ["RUN_ID"]
CRASH_AT = os.environ.get("CRASH_AT", "")
PROVIDER = os.environ.get("PROVIDER", "naive")
KILLED = 137


class State(TypedDict):
    done: list[str]


def record(effect: str, key: str | None = None) -> None:
    """The fake external system. With a key and a keyed provider, a repeat of the key is ignored."""
    with psycopg.connect(URL, autocommit=True) as conn:
        if key is not None and PROVIDER == "keyed":
            conn.execute(
                "INSERT INTO test_effects (run_id, effect, idem_key) VALUES (%s, %s, %s) "
                "ON CONFLICT (idem_key) WHERE idem_key IS NOT NULL DO NOTHING",
                (RUN_ID, effect, key),
            )
        else:
            conn.execute("INSERT INTO test_effects (run_id, effect) VALUES (%s, %s)", (RUN_ID, effect))


def plain_step(name: str):
    def node(state: State):
        record(name)
        return {"done": [*state["done"], name]}

    return node


def charge(state: State):
    def call_provider(key: str):
        record("charge", key)
        if CRASH_AT == "inside_call":
            os._exit(KILLED)
        return {"charged": True}

    with psycopg.connect(URL, autocommit=True) as conn:
        run_once(conn, RUN_ID, "charge", "charge", {}, call_provider)
    if CRASH_AT == "after_ledger":
        os._exit(KILLED)
    return {"done": [*state["done"], "charge"]}


def build(checkpointer):
    graph = StateGraph(State)
    graph.add_node("reserve", plain_step("reserve"))
    graph.add_node("charge", charge)
    graph.add_node("confirm", plain_step("confirm"))
    graph.add_edge(START, "reserve")
    graph.add_edge("reserve", "charge")
    graph.add_edge("charge", "confirm")
    graph.add_edge("confirm", END)
    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    mode = sys.argv[1]
    config = {"configurable": {"thread_id": RUN_ID}}
    with PostgresSaver.from_conn_string(URL) as saver:
        result = build(saver).invoke({"done": []} if mode == "start" else None, config)
    print(json.dumps(result))
