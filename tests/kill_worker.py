"""Test worker for the kill-and-resume test. Not part of the library.

Runs reserve -> charge -> confirm on a Postgres checkpointer. Each node records its
side effect in test_effects. With CRASH=1, `charge` hard-kills the process right after
its side effect, before LangGraph can checkpoint the node (like kill -9).

usage: kill_worker.py start|resume   (env: PORTCULLIS_DATABASE_URL, RUN_ID, CRASH)
"""

import json
import os
import sys
from typing import TypedDict

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph

URL = os.environ["PORTCULLIS_DATABASE_URL"]
RUN_ID = os.environ["RUN_ID"]
CRASH = os.environ.get("CRASH") == "1"
KILLED = 137


class State(TypedDict):
    done: list[str]


def record(effect: str) -> None:
    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute("INSERT INTO test_effects (run_id, effect) VALUES (%s, %s)", (RUN_ID, effect))


def step(name: str):
    def node(state: State):
        record(name)
        if CRASH and name == "charge":
            os._exit(KILLED)
        return {"done": [*state["done"], name]}

    return node


def build(checkpointer):
    graph = StateGraph(State)
    for name in ("reserve", "charge", "confirm"):
        graph.add_node(name, step(name))
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
