"""LangGraph's Postgres checkpointer: a paused run survives a brand-new connection."""

from collections import Counter
from uuid import uuid4

from graphs import build_graph
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command


def test_paused_run_is_resumed_from_postgres_by_a_new_connection(postgres_url):
    config = {"configurable": {"thread_id": f"t-{uuid4()}"}}
    with PostgresSaver.from_conn_string(postgres_url) as saver:
        build_graph(Counter(), saver).invoke({"log": []}, config)

    calls = Counter()
    with PostgresSaver.from_conn_string(postgres_url) as saver:  # new connection, new graph object
        app = build_graph(calls, saver)
        assert app.get_state(config).next == ("gate",)
        done = app.invoke(Command(resume=True), config)

    assert done["log"] == ["before", "gate:True", "after"]
    assert calls["before"] == 0  # the finished node was not run again
