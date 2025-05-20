from typing import Dict, List, Optional

from langchain.tools import BaseTool
from langchain.tools.calculator import CalculatorTool
from langchain.tools.wikipedia import WikipediaQueryRun
from langchain_core.language_models import ChatOpenAI
from pydantic import BaseModel


class Tool(BaseModel):
    name: str
    description: str
    func: callable


class AgentState(BaseModel):
    messages: List[Dict[str, str]] = []
    current_input: str = ""
    thought: str = ""
    selected_tool: Optional[str] = None
    tool_input: str = ""
    tool_output: str = ""
    final_answer: str = ""
    status: str = "STARTING"


# Define available tools
tools = [
    Tool(name="calculator", description="Used for performing mathematical calculations", func=CalculatorTool()),
    Tool(name="wikipedia", description="Used for querying Wikipedia information", func=WikipediaQueryRun()),
]


async def think(state: AgentState) -> AgentState:
    """Think about the next action"""
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
    result = json.loads(response)
    return AgentState(
        **state.dict(),
        thought=result["thought"],
        selected_tool=result.get("tool"),
        tool_input=result.get("tool_input"),
        status="NEED_TOOL" if result["need_tool"] else "GENERATE_RESPONSE",
    )


async def execute_tool(state: AgentState) -> AgentState:
    """Execute tool call"""
    tool = next((t for t in tools if t.name == state.selected_tool), None)
    if not tool:
        return AgentState(**state.dict(), status="ERROR", thought="Selected tool not found")
    try:
        result = await tool.func.ainvoke(state.tool_input)
        return AgentState(**state.dict(), tool_output=str(result), status="GENERATE_RESPONSE")
    except Exception as e:
        return AgentState(**state.dict(), status="ERROR", thought=f"Tool execution failed: {str(e)}")


async def generate_response(state: AgentState) -> AgentState:
    """Generate the final response"""
    prompt = f"""
    Generate a response to the user based on the following information:
    User input: {state.current_input}
    Thought process: {state.thought}
    Tool output: {state.tool_output}
    Please generate a clear and helpful response.
    """
    llm = ChatOpenAI(temperature=0.7)
    response = await llm.ainvoke(prompt)
    return AgentState(**state.dict(), final_answer=response, status="SUCCESS")


# Create graph structure
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("think", think)
workflow.add_node("execute_tool", execute_tool)
workflow.add_node("generate_response", generate_response)

# Add edges
workflow.add_edge("think", "execute_tool", condition=lambda s: s.status == "NEED_TOOL")
workflow.add_edge("execute_tool", "generate_response", condition=lambda s: s.status == "GENERATE_RESPONSE")
workflow.add_edge("generate_response", "think", condition=lambda s: s.status == "SUCCESS")
