from graphviz import Digraph


def visualize_graph():
    dot = Digraph(comment="State Graph")
    dot.attr(rankdir="LR")  # Left to right direction

    # Add nodes
    dot.node("START", "START", shape="circle")
    dot.node("node_a", "node_a", shape="box")
    dot.node("node_b", "node_b", shape="box")
    dot.node("END", "END", shape="circle")

    # Add edges
    dot.edge("START", "node_a")
    dot.edge("node_a", "node_b")
    dot.edge("node_b", "END")

    # Save the graph
    dot.render("state_graph", format="png", cleanup=True)


if __name__ == "__main__":
    visualize_graph()
