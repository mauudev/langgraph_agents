import json
from typing import Callable, Dict, List, Optional

from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.tools.python.tool import PythonREPLTool
from langgraph.graph import StateGraph
from pydantic import BaseModel


class Tool(BaseModel):
    name: str
    description: str
    func: Callable


class AgentState(BaseModel):
    messages: List[Dict[str, str]] = []
    current_input: str = ""
    thought: str = ""
    selected_tool: Optional[str] = None
    tool_input: str = ""
    tool_output: str = ""
    final_answer: str = ""
    status: str = "STARTING"


# Define tools
tools = [
    Tool(name="python_repl", description="Used for executing python code", func=PythonREPLTool()),
    Tool(
        name="wikipedia",
        description="Used for querying Wikipedia information",
        func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    ),
]


# Think node
async def think(state: AgentState) -> AgentState:
    prompt = f"""
    Based on user input and current conversation history, think about the next action.
    User input: {state.current_input}
    Available tools: {[t.name + ': ' + t.description for t in tools]}
    Decide:
    1. Whether a tool is needed
    2. If needed, which tool to use
    3. What parameters to call the tool with
    Return in JSON format: {{"thought": "thought process", "need_tool": true/false, "tool": "tool name", "tool_input": "parameters"}}
    """
    llm = ChatOpenAI(temperature=0)
    response = await llm.ainvoke(prompt)
    result = json.loads(response.content)
    state_dict = state.model_dump()
    state_dict.update(
        {
            "thought": result["thought"],
            "selected_tool": result.get("tool"),
            "tool_input": result.get("tool_input"),
            "status": "NEED_TOOL" if result["need_tool"] else "GENERATE_RESPONSE",
        }
    )
    return AgentState(**state_dict)


# Execute tool node
async def execute_tool(state: AgentState) -> AgentState:
    tool = next((t for t in tools if t.name == state.selected_tool), None)
    if not tool:
        state_dict = state.model_dump()
        state_dict.update({"status": "ERROR", "thought": "Selected tool not found"})
        return AgentState(**state_dict)
    try:
        result = tool.func.invoke(state.tool_input)
        state_dict = state.model_dump()
        state_dict.update({"tool_output": str(result), "status": "GENERATE_RESPONSE"})
        return AgentState(**state_dict)
    except Exception as e:
        state_dict = state.model_dump()
        state_dict.update({"status": "ERROR", "thought": f"Tool execution failed: {str(e)}"})
        return AgentState(**state_dict)


# Generate final response
async def generate_response(state: AgentState) -> AgentState:
    prompt = f"""
    Generate a response to the user based on the following information:
    User input: {state.current_input}
    Thought process: {state.thought}
    Tool output: {state.tool_output}
    Please generate a clear and helpful response.
    """
    llm = ChatOpenAI(temperature=0.7)
    response = await llm.ainvoke(prompt)
    state_dict = state.model_dump()
    state_dict.update({"final_answer": response.content, "status": "SUCCESS"})
    return AgentState(**state_dict)


# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("think", think)
workflow.add_node("execute_tool", execute_tool)
workflow.add_node("generate_response", generate_response)

# Define transitions
workflow.set_entry_point("think")

# Add conditional edges
workflow.add_conditional_edges(
    "think",
    lambda state: "execute_tool" if state.status == "NEED_TOOL" else "generate_response",
    {"execute_tool": "execute_tool", "generate_response": "generate_response"},
)

workflow.add_edge("execute_tool", "generate_response")
workflow.add_edge("generate_response", "think")

# Compile
app = workflow.compile()


async def main():
    # initial_state = AgentState(current_input="What is 5 times 12 plus 8?")
    initial_state = AgentState(current_input="please execute the following code: print('Hello, world!')")
    final_state = await app.ainvoke(initial_state)
    print(final_state.final_answer)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
