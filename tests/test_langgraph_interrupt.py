"""Pins the LangGraph interrupt/resume behaviour portcullis relies on (langgraph 1.2.x)."""

from collections import Counter

from graphs import build_graph
from langgraph.types import Command


def test_pauses_at_interrupt_then_resumes_to_completion():
    calls = Counter()
    app = build_graph(calls)
    config = {"configurable": {"thread_id": "t1"}}

    paused = app.invoke({"log": []}, config)

    assert paused["__interrupt__"][0].value == {"ask": "approve?"}
    assert app.get_state(config).next == ("gate",)
    assert calls["after"] == 0

    done = app.invoke(Command(resume=True), config)

    assert done["log"] == ["before", "gate:True", "after"]


def test_completed_nodes_are_not_rerun_but_the_interrupted_node_is():
    # Side effects placed before interrupt() inside a node run twice.
    # portcullis must put them after the approval or guard them with idempotency keys.
    calls = Counter()
    app = build_graph(calls)
    config = {"configurable": {"thread_id": "t2"}}

    app.invoke({"log": []}, config)
    app.invoke(Command(resume=True), config)

    assert calls["before"] == 1
    assert calls["gate"] == 2
    assert calls["after"] == 1
