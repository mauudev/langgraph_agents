"""
This sample shows how to use subgraphs.
Subgraphs
"""

import uuid
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    """The graph state."""

    state_counter: int


counter_node_in_subgraph = 0


def node_in_subgraph(state: State):
    """A node in the sub-graph."""
    global counter_node_in_subgraph
    counter_node_in_subgraph += 1  # This code will **NOT** run again!
    print(f"Entered `node_in_subgraph` a total of {counter_node_in_subgraph} times")


counter_human_node = 0


def human_node(state: State):
    global counter_human_node
    counter_human_node += 1  # This code will run again!
    print(f"Entered human_node in sub-graph a total of {counter_human_node} times")
    answer = interrupt("what is your name?")
    print(f"Got an answer of {answer}")


checkpointer = MemorySaver()

subgraph_builder = StateGraph(State)
subgraph_builder.add_node("some_node", node_in_subgraph)
subgraph_builder.add_node("human_node", human_node)
subgraph_builder.add_edge(START, "some_node")
subgraph_builder.add_edge("some_node", "human_node")
subgraph = subgraph_builder.compile(checkpointer=checkpointer)


counter_parent_node = 0


def parent_node(state: State):
    """This parent node will invoke the subgraph."""
    global counter_parent_node

    counter_parent_node += 1  # This code will run again on resuming!
    print(f"Entered `parent_node` a total of {counter_parent_node} times")

    # Please note that we're intentionally incrementing the state counter
    # in the graph state as well to demonstrate that the subgraph update
    # of the same key will not conflict with the parent graph (until
    subgraph_state = subgraph.invoke(state)
    return subgraph_state


builder = StateGraph(State)
builder.add_node("parent_node", parent_node)
builder.add_edge(START, "parent_node")

# A checkpointer must be enabled for interrupts to work!
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": uuid.uuid4(),
    }
}

for chunk in graph.stream({"state_counter": 1}, config):
    print(chunk)

print("--- Resuming ---")

for chunk in graph.stream(Command(resume="35"), config):
    print(chunk)
