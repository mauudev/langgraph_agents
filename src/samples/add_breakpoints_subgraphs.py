"""
This sample shows how to add breakpoints to subgraphs.
Add breakpoints to subgraphs
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.types import interrupt
from typing_extensions import TypedDict


class State(TypedDict):
    foo: str


def subgraph_node_1(state: State):
    return {"foo": state["foo"]}


subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")

subgraph = subgraph_builder.compile(interrupt_before=["subgraph_node_1"])

builder = StateGraph(State)
builder.add_node("node_1", subgraph)  # directly include subgraph as a node
builder.add_edge(START, "node_1")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}

graph.invoke({"foo": ""}, config)

# Fetch state including subgraph state.
print(graph.get_state(config, subgraphs=True).tasks[0].state)

# resume the subgraph
graph.invoke(None, config)
