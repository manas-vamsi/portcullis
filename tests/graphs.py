"""Small LangGraph graphs shared by tests."""

from collections import Counter
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class State(TypedDict):
    log: list[str]


def build_graph(calls: Counter, checkpointer=None):
    """before -> gate (human approval via interrupt) -> after. `calls` counts node executions."""

    def before(state: State):
        calls["before"] += 1
        return {"log": [*state["log"], "before"]}

    def gate(state: State):
        calls["gate"] += 1
        approved = interrupt({"ask": "approve?"})
        return {"log": [*state["log"], f"gate:{approved}"]}

    def after(state: State):
        calls["after"] += 1
        return {"log": [*state["log"], "after"]}

    graph = StateGraph(State)
    graph.add_node("before", before)
    graph.add_node("gate", gate)
    graph.add_node("after", after)
    graph.add_edge(START, "before")
    graph.add_edge("before", "gate")
    graph.add_edge("gate", "after")
    graph.add_edge("after", END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())
