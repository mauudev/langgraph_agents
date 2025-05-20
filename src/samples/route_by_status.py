from typing import List, Dict, Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, END


class AgentState(BaseModel):
    messages: List[Dict[str, str]] = []
    current_input: str = ""
    tools_output: Dict[str, str] = {}
    status: str = "RUNNING"
    error_count: int = 0


def route_by_status(state: AgentState) -> Literal["process", "retry", "error", "end"]:
    """Complex routing logic"""
    if state.status == "SUCCESS":
        return "end"
    elif state.status == "ERROR":
        if state.error_count >= 3:
            return "error"
        return "retry"
    elif state.status == "NEED_TOOL":
        return "process"
    return "process"


# Build the graph structure
workflow = StateGraph(AgentState)

# Add conditional edges
workflow.add_conditional_edges(
    "check_status",
    route_by_status,
    {"process": "execute_tool", "retry": "retry_handler", "error": "error_handler", "end": END},
)
