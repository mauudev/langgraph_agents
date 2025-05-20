async def parallel_tools_execution(state: AgentState) -> AgentState:
    """Parallel execution of multiple tools"""
    tools = identify_required_tools(state.current_input)

    async def execute_tool(tool):
        result = await tool.ainvoke(state.current_input)
        return {tool.name: result}

    # Execute all tools in parallel
    results = await asyncio.gather(*[execute_tool(tool) for tool in tools])

    # Merge results
    tools_output = {}
    for result in results:
        tools_output.update(result)

    return AgentState(
        messages=state.messages, current_input=state.current_input, tools_output=tools_output, status="SUCCESS"
    )
