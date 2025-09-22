import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import anthropic


@dataclass
class ToolCallTracker:
    """Tracks tool calling rounds and execution details"""

    def __init__(self, max_rounds: int = 2):
        self.max_rounds = max_rounds
        self.current_round = 0
        self.tool_calls_made = []
        self.execution_times = []
        self.errors = []

    def can_make_tool_call(self) -> bool:
        """Check if another tool call round is allowed"""
        return self.current_round < self.max_rounds

    def start_round(self) -> int:
        """Start a new tool calling round"""
        self.current_round += 1
        return self.current_round

    def log_tool_call(
        self,
        tool_name: str,
        params: dict,
        execution_time: float,
        success: bool,
        error: str = None,
    ):
        """Log details of a tool call"""
        self.tool_calls_made.append(
            {
                "round": self.current_round,
                "tool_name": tool_name,
                "params": params,
                "execution_time": execution_time,
                "success": success,
                "error": error,
            }
        )

        self.execution_times.append(execution_time)

        if error:
            self.errors.append(error)

    def get_summary(self) -> dict:
        """Get summary of tool calling session"""
        return {
            "total_rounds": self.current_round,
            "total_tool_calls": len(self.tool_calls_made),
            "successful_calls": len(
                [call for call in self.tool_calls_made if call["success"]]
            ),
            "failed_calls": len(
                [call for call in self.tool_calls_made if not call["success"]]
            ),
            "total_execution_time": sum(self.execution_times),
            "errors": self.errors,
        }


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **Up to 2 sequential searches per query** - use multiple searches for complex queries requiring comparison, analysis, or comprehensive coverage
- Strategic search approach: First search for initial relevant information, second search for additional context, comparisons, or related topics
- Synthesize ALL search results into accurate, fact-based responses
- If any search yields no results, state this clearly without offering alternatives

Sequential Search Strategy:
- **Complex queries**: Use multiple searches to gather comprehensive information (e.g., "Compare X and Y" → search X, then search Y)
- **Broad topics**: Start with general search, then narrow down with specific follow-up searches
- **Multi-part questions**: Break down into separate searches for each component
- **Course comparisons**: Search each course/topic separately for detailed analysis

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search strategically, using multiple calls if beneficial for comprehensive answers
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or describe your search process

All responses must be:
1. **Comprehensive and well-informed** - Leverage all gathered information from multiple searches
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
5. **Comparative when relevant** - Highlight similarities and differences when comparing content

Provide only the direct answer to what was asked, informed by all available search results.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
        max_tool_rounds: int = 2,
    ) -> str:
        """
        Generate AI response with optional sequential tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_tool_rounds: Maximum number of sequential tool calling rounds (default: 2)

        Returns:
            Generated response as string
        """

        # Build initial system content
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize tool tracking
        tool_tracker = ToolCallTracker(max_tool_rounds)

        # Initialize conversation messages
        messages = [{"role": "user", "content": query}]

        # Handle sequential tool execution if tools are available
        if tools and tool_manager:
            return self._handle_sequential_tool_execution(
                messages, system_content, tools, tool_manager, tool_tracker
            )
        else:
            # No tools available, direct response
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }

            response = self.client.messages.create(**api_params)
            return response.content[0].text

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, **content_block.input
                )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": content_block.id,
                        "content": tool_result,
                    }
                )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _handle_sequential_tool_execution(
        self,
        messages: List[Dict[str, Any]],
        system_content: str,
        tools: List,
        tool_manager,
        tool_tracker: ToolCallTracker,
    ) -> str:
        """
        Handle sequential tool execution with up to max_tool_rounds rounds.

        Args:
            messages: Initial conversation messages
            system_content: Base system content
            tools: Available tools
            tool_manager: Manager to execute tools
            tool_tracker: Tracker for tool execution rounds

        Returns:
            Final response text after all tool executions
        """

        accumulated_context = []

        while tool_tracker.can_make_tool_call():
            # Start new round
            current_round = tool_tracker.start_round()

            # Build enriched system content with accumulated context
            enriched_system = self._build_enriched_system_content(
                system_content, accumulated_context, current_round
            )

            # Prepare API parameters
            api_params = {
                **self.base_params,
                "messages": messages.copy(),
                "system": enriched_system,
                "tools": tools,
                "tool_choice": {"type": "auto"},
            }

            # Get response from Claude
            response = self.client.messages.create(**api_params)

            # Add Claude's response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Execute tools and track results
                tool_results, context_entry = self._execute_and_track_tools(
                    response, tool_manager, tool_tracker, current_round
                )

                if tool_results:
                    # Add tool results to messages
                    messages.append({"role": "user", "content": tool_results})

                    # Add to accumulated context for next round
                    accumulated_context.append(context_entry)
                else:
                    # Tool execution failed, break out
                    break
            else:
                # Claude didn't use tools, we have our final response
                return response.content[0].text

        # Generate final synthesis response if we ended with tool results
        if messages[-1]["role"] == "user" and any(
            isinstance(content, dict) and content.get("type") == "tool_result"
            for content in messages[-1]["content"]
        ):
            final_system = self._build_synthesis_system_content(
                system_content, accumulated_context
            )

            final_params = {
                **self.base_params,
                "messages": messages,
                "system": final_system,
                # No tools for final synthesis
            }

            final_response = self.client.messages.create(**final_params)
            return final_response.content[0].text

        # Return last assistant response
        for message in reversed(messages):
            if message["role"] == "assistant":
                content = message["content"]
                if isinstance(content, list) and content:
                    # Extract text from content blocks
                    text_parts = []
                    for block in content:
                        if hasattr(block, "text"):
                            text_parts.append(block.text)
                        elif isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))

                    if text_parts:
                        return "\n".join(text_parts)

        return "I apologize, but I couldn't generate a proper response."

    def _execute_and_track_tools(
        self, response, tool_manager, tool_tracker: ToolCallTracker, current_round: int
    ):
        """
        Execute tools from a response and track the execution details.

        Args:
            response: Claude's response containing tool_use blocks
            tool_manager: Manager to execute tools
            tool_tracker: Tracker for tool execution rounds
            current_round: Current round number

        Returns:
            Tuple of (tool_results_list, context_entry_dict) or (None, None) if execution failed
        """

        tool_results = []
        tool_executions = []

        for content_block in response.content:
            if content_block.type == "tool_use":
                start_time = time.time()

                try:
                    # Execute the tool
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    execution_time = time.time() - start_time

                    # Log successful execution
                    tool_tracker.log_tool_call(
                        content_block.name, content_block.input, execution_time, True
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )

                    tool_executions.append(
                        {
                            "tool_name": content_block.name,
                            "input": content_block.input,
                            "result": tool_result,
                            "execution_time": execution_time,
                            "success": True,
                        }
                    )

                except Exception as e:
                    execution_time = time.time() - start_time
                    error_msg = f"Tool execution failed: {str(e)}"

                    # Log failed execution
                    tool_tracker.log_tool_call(
                        content_block.name,
                        content_block.input,
                        execution_time,
                        False,
                        error_msg,
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": error_msg,
                            "is_error": True,
                        }
                    )

                    tool_executions.append(
                        {
                            "tool_name": content_block.name,
                            "input": content_block.input,
                            "result": error_msg,
                            "execution_time": execution_time,
                            "success": False,
                            "error": str(e),
                        }
                    )

        # Create context entry for this round
        context_entry = {
            "round": current_round,
            "tool_executions": tool_executions,
            "total_tools": len(tool_executions),
            "successful_tools": len(
                [exec for exec in tool_executions if exec["success"]]
            ),
        }

        return tool_results if tool_results else None, context_entry

    def _build_enriched_system_content(
        self, base_content: str, accumulated_context: List[Dict], current_round: int
    ) -> str:
        """
        Build enriched system content including previous tool results and round guidance.

        Args:
            base_content: Base system prompt
            accumulated_context: List of previous tool execution contexts
            current_round: Current round number

        Returns:
            Enriched system content
        """

        # Add round-specific guidance
        round_guidance = ""
        if current_round == 1:
            round_guidance = "\n\nTool Round 1: This is your first search opportunity. You can make up to 2 total searches. Consider if you need additional information after this search to fully answer the question."
        elif current_round == 2:
            round_guidance = "\n\nTool Round 2: This is your second and final search opportunity. Use this to gather any additional information needed for a comprehensive answer."

        enriched_content = base_content + round_guidance

        # Add context from previous rounds
        if accumulated_context:
            context_summary = self._summarize_tool_context(accumulated_context)
            enriched_content += f"\n\nPrevious search results:\n{context_summary}"

        return enriched_content

    def _build_synthesis_system_content(
        self, base_content: str, accumulated_context: List[Dict]
    ) -> str:
        """
        Build system content for final synthesis phase.

        Args:
            base_content: Base system prompt
            accumulated_context: List of all tool execution contexts

        Returns:
            System content for synthesis
        """

        synthesis_guidance = """

Final Response Phase:
- Synthesize information from ALL previous searches
- Provide a comprehensive answer that leverages all gathered context
- Do not mention the search process itself
- Focus on directly answering the user's question with all available information
"""

        enriched_content = base_content + synthesis_guidance

        if accumulated_context:
            context_summary = self._summarize_tool_context(accumulated_context)
            enriched_content += (
                f"\n\nAll search results to synthesize:\n{context_summary}"
            )

        return enriched_content

    def _summarize_tool_context(self, accumulated_context: List[Dict]) -> str:
        """
        Summarize accumulated tool execution context for system prompts.

        Args:
            accumulated_context: List of tool execution contexts

        Returns:
            Formatted summary string
        """

        summaries = []

        for context_entry in accumulated_context:
            round_num = context_entry["round"]
            executions = context_entry["tool_executions"]

            round_summary = f"Round {round_num}:"

            for exec in executions:
                tool_name = exec["tool_name"]
                success = exec["success"]

                if success:
                    # Truncate result for summary
                    result = exec["result"]
                    if len(result) > 300:
                        result = result[:300] + "..."

                    round_summary += f"\n  - {tool_name}: {result}"
                else:
                    round_summary += f"\n  - {tool_name}: Failed ({exec.get('error', 'Unknown error')})"

            summaries.append(round_summary)

        return "\n\n".join(summaries)
