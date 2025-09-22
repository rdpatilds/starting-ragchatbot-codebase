import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, Mock, patch

import pytest
from ai_generator import AIGenerator, ToolCallTracker


class TestAIGenerator:
    """Test AI Generator's ability to call tools correctly"""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create mock Anthropic client"""
        with patch("ai_generator.anthropic.Anthropic") as mock:
            yield mock.return_value

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create AI generator with mocked client"""
        return AIGenerator(api_key="test_key", model="test_model")

    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager"""
        manager = Mock()
        manager.get_tool_definitions.return_value = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
        manager.execute_tool.return_value = (
            "Search results: Python is a programming language"
        )
        return manager

    def test_generate_response_without_tools(self, ai_generator, mock_anthropic_client):
        """Test generating response without tools"""
        # Configure mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="This is a response")]
        mock_response.stop_reason = "stop"
        mock_anthropic_client.messages.create.return_value = mock_response

        # Generate response
        result = ai_generator.generate_response("What is Python?")

        # Check result
        assert result == "This is a response"
        mock_anthropic_client.messages.create.assert_called_once()

    def test_generate_response_with_tool_call(
        self, ai_generator, mock_anthropic_client, mock_tool_manager
    ):
        """Test that AI generator correctly calls tools"""
        # Create mock tool use content
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.input = {"query": "Python basics"}
        mock_tool_use.id = "tool_123"

        # Configure initial response with tool use
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_use]
        mock_initial_response.stop_reason = "tool_use"

        # Configure follow-up response
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="Based on the search, Python is a programming language")
        ]

        # Set up the mock to return different responses
        mock_anthropic_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Generate response with tools
        result = ai_generator.generate_response(
            "What is Python?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Verify tool was called
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="Python basics"
        )

        # Verify final response
        assert "Based on the search" in result

    def test_handle_tool_execution(
        self, ai_generator, mock_anthropic_client, mock_tool_manager
    ):
        """Test _handle_tool_execution method"""
        # Create mock tool use
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.input = {"query": "decorators"}
        mock_tool_use.id = "tool_456"

        # Create initial response
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_use]

        # Create final response
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="Decorators are functions that modify other functions")
        ]

        mock_anthropic_client.messages.create.return_value = mock_final_response

        # Test tool execution handling
        base_params = {
            "messages": [{"role": "user", "content": "What are decorators?"}],
            "system": "Test system prompt",
        }

        result = ai_generator._handle_tool_execution(
            mock_initial_response, base_params, mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="decorators"
        )

        # Verify result
        assert "Decorators are functions" in result

    def test_conversation_history_integration(
        self, ai_generator, mock_anthropic_client
    ):
        """Test that conversation history is properly integrated"""
        # Configure mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with history")]
        mock_response.stop_reason = "stop"
        mock_anthropic_client.messages.create.return_value = mock_response

        # Generate response with history
        history = "User: Previous question\nAssistant: Previous answer"
        result = ai_generator.generate_response(
            "New question", conversation_history=history
        )

        # Verify history was included in system prompt
        call_args = mock_anthropic_client.messages.create.call_args
        assert "Previous conversation:" in call_args[1]["system"]
        assert history in call_args[1]["system"]

    def test_system_prompt_structure(self, ai_generator):
        """Test that system prompt is properly structured"""
        assert "course materials" in ai_generator.SYSTEM_PROMPT
        assert "search tool" in ai_generator.SYSTEM_PROMPT.lower()
        assert "Up to 2 sequential searches" in ai_generator.SYSTEM_PROMPT
        assert "Sequential Search Strategy" in ai_generator.SYSTEM_PROMPT

    def test_api_parameters(self, ai_generator, mock_anthropic_client):
        """Test that API parameters are correctly set"""
        # Configure mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.stop_reason = "stop"
        mock_anthropic_client.messages.create.return_value = mock_response

        # Generate response
        ai_generator.generate_response("Test query")

        # Check API call parameters
        call_args = mock_anthropic_client.messages.create.call_args
        assert call_args[1]["model"] == "test_model"
        assert call_args[1]["temperature"] == 0
        assert call_args[1]["max_tokens"] == 800

    def test_tool_choice_auto(
        self, ai_generator, mock_anthropic_client, mock_tool_manager
    ):
        """Test that tool_choice is set to auto when tools are provided"""
        # Configure mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.stop_reason = "stop"
        mock_anthropic_client.messages.create.return_value = mock_response

        # Generate response with tools and tool_manager
        ai_generator.generate_response(
            "Query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Verify tool_choice was set in the sequential tool execution path
        call_args = mock_anthropic_client.messages.create.call_args
        assert "tool_choice" in call_args[1]
        assert call_args[1]["tool_choice"]["type"] == "auto"


class TestSequentialToolCalling:
    """Test sequential tool calling functionality"""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create mock Anthropic client"""
        with patch("ai_generator.anthropic.Anthropic") as mock:
            yield mock.return_value

    @pytest.fixture
    def ai_generator(self, mock_anthropic_client):
        """Create AI generator with mocked client"""
        return AIGenerator(api_key="test_key", model="test_model")

    @pytest.fixture
    def mock_tool_manager(self):
        """Create mock tool manager"""
        manager = Mock()
        manager.get_tool_definitions.return_value = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
        # Different results for different searches
        manager.execute_tool.side_effect = [
            "First search result: Python basics information",
            "Second search result: Advanced Python topics",
        ]
        return manager

    def test_single_tool_call_backwards_compatibility(
        self, ai_generator, mock_anthropic_client, mock_tool_manager
    ):
        """Test that single tool calls still work (backwards compatibility)"""
        # Create mock tool use content
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.input = {"query": "Python basics"}
        mock_tool_use.id = "tool_123"

        # Configure initial response with tool use
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_use]
        mock_initial_response.stop_reason = "tool_use"

        # Configure follow-up response without tools
        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Python is a programming language")]
        mock_final_response.stop_reason = "stop"

        # Set up the mock to return different responses
        mock_anthropic_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Generate response with tools
        result = ai_generator.generate_response(
            "What is Python?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
            max_tool_rounds=2,
        )

        # Verify only one tool was called
        assert mock_tool_manager.execute_tool.call_count == 1
        assert "Python is a programming language" in result

    def test_sequential_tool_calls_two_rounds(
        self, ai_generator, mock_anthropic_client, mock_tool_manager
    ):
        """Test that sequential tool calls work for two rounds"""
        # Create mock tool use content for first round
        mock_tool_use_1 = Mock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "search_course_content"
        mock_tool_use_1.input = {"query": "Python basics"}
        mock_tool_use_1.id = "tool_123"

        # Create mock tool use content for second round
        mock_tool_use_2 = Mock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "search_course_content"
        mock_tool_use_2.input = {"query": "Python advanced"}
        mock_tool_use_2.id = "tool_456"

        # Configure responses
        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_use_1]
        mock_response_1.stop_reason = "tool_use"

        mock_response_2 = Mock()
        mock_response_2.content = [mock_tool_use_2]
        mock_response_2.stop_reason = "tool_use"

        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="Comprehensive Python answer covering basics and advanced topics")
        ]
        mock_final_response.stop_reason = "stop"

        mock_anthropic_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_final_response,
        ]

        # Generate response
        result = ai_generator.generate_response(
            "Tell me about Python programming from basics to advanced",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
            max_tool_rounds=2,
        )

        # Verify two tools were called
        assert mock_tool_manager.execute_tool.call_count == 2
        assert "Comprehensive Python answer" in result

    def test_max_rounds_enforcement(
        self, ai_generator, mock_anthropic_client, mock_tool_manager
    ):
        """Test that max rounds limit is enforced"""
        # Create tool use responses that would continue indefinitely
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.input = {"query": "test"}
        mock_tool_use.id = "tool_123"

        mock_tool_response = Mock()
        mock_tool_response.content = [mock_tool_use]
        mock_tool_response.stop_reason = "tool_use"

        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Final response after max rounds")]
        mock_final_response.stop_reason = "stop"

        # Mock to always return tool use for first two calls, then final response
        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_response,  # Round 1
            mock_tool_response,  # Round 2
            mock_final_response,  # Final synthesis
        ]

        # Generate response with max_tool_rounds=2
        result = ai_generator.generate_response(
            "Test query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
            max_tool_rounds=2,
        )

        # Should have called exactly 2 tools (max rounds)
        assert mock_tool_manager.execute_tool.call_count == 2
        assert "Final response after max rounds" in result

    def test_tool_execution_failure_handling(
        self, ai_generator, mock_anthropic_client, mock_tool_manager
    ):
        """Test handling of tool execution failures"""
        # Configure tool manager to fail
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution error")

        # Create mock tool use content
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.input = {"query": "test"}
        mock_tool_use.id = "tool_123"

        mock_tool_response = Mock()
        mock_tool_response.content = [mock_tool_use]
        mock_tool_response.stop_reason = "tool_use"

        # Mock final response after tool failure
        mock_final_response = Mock()
        mock_final_response.content = [
            Mock(text="I apologize, but I couldn't complete the search")
        ]
        mock_final_response.stop_reason = "stop"

        # Tool execution fails, but we should get a final response
        mock_anthropic_client.messages.create.side_effect = [
            mock_tool_response,
            mock_final_response,
        ]

        # Generate response
        result = ai_generator.generate_response(
            "Test query",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Should handle error gracefully and return the final response
        assert result is not None
        assert isinstance(result, str)
        assert "apologize" in result.lower() or "couldn't" in result.lower()

    def test_context_accumulation_between_rounds(
        self, ai_generator, mock_anthropic_client, mock_tool_manager
    ):
        """Test that context accumulates properly between rounds"""
        # Mock two tool calls with different results
        mock_tool_manager.execute_tool.side_effect = [
            "First result about Python",
            "Second result about Java",
        ]

        # Create tool use mocks
        mock_tool_use_1 = Mock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "search_course_content"
        mock_tool_use_1.input = {"query": "Python"}
        mock_tool_use_1.id = "tool_1"

        mock_tool_use_2 = Mock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "search_course_content"
        mock_tool_use_2.input = {"query": "Java"}
        mock_tool_use_2.id = "tool_2"

        # Create responses
        mock_response_1 = Mock()
        mock_response_1.content = [mock_tool_use_1]
        mock_response_1.stop_reason = "tool_use"

        mock_response_2 = Mock()
        mock_response_2.content = [mock_tool_use_2]
        mock_response_2.stop_reason = "tool_use"

        mock_final_response = Mock()
        mock_final_response.content = [Mock(text="Comparison of Python and Java")]
        mock_final_response.stop_reason = "stop"

        mock_anthropic_client.messages.create.side_effect = [
            mock_response_1,
            mock_response_2,
            mock_final_response,
        ]

        # Generate response
        result = ai_generator.generate_response(
            "Compare Python and Java",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager,
        )

        # Verify context was included in second round
        # Check that the second API call included previous search results in system prompt
        assert mock_anthropic_client.messages.create.call_count == 3

        # Get the second call (round 2) system content
        second_call_args = mock_anthropic_client.messages.create.call_args_list[1]
        system_content = second_call_args[1]["system"]

        # Should contain previous search results
        assert "Previous search results:" in system_content
        assert "Round 1:" in system_content

    def test_no_tools_direct_response(self, ai_generator, mock_anthropic_client):
        """Test direct response when Claude doesn't use tools"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Direct answer without tools")]
        mock_response.stop_reason = "stop"

        mock_anthropic_client.messages.create.return_value = mock_response

        # Create empty tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = []

        result = ai_generator.generate_response(
            "What is 2+2?", tools=[], tool_manager=mock_tool_manager
        )

        assert result == "Direct answer without tools"


class TestToolCallTracker:
    """Test the ToolCallTracker class"""

    def test_tracker_initialization(self):
        """Test tracker initializes correctly"""
        tracker = ToolCallTracker(max_rounds=2)

        assert tracker.max_rounds == 2
        assert tracker.current_round == 0
        assert tracker.tool_calls_made == []
        assert tracker.execution_times == []
        assert tracker.errors == []

    def test_can_make_tool_call(self):
        """Test tool call availability check"""
        tracker = ToolCallTracker(max_rounds=2)

        # Initially should be able to make calls
        assert tracker.can_make_tool_call() == True

        # After max rounds reached
        tracker.current_round = 2
        assert tracker.can_make_tool_call() == False

    def test_start_round(self):
        """Test starting new rounds"""
        tracker = ToolCallTracker(max_rounds=2)

        # Start first round
        round_num = tracker.start_round()
        assert round_num == 1
        assert tracker.current_round == 1

        # Start second round
        round_num = tracker.start_round()
        assert round_num == 2
        assert tracker.current_round == 2

    def test_log_tool_call(self):
        """Test logging tool calls"""
        tracker = ToolCallTracker(max_rounds=2)
        tracker.start_round()

        # Log successful call
        tracker.log_tool_call("search", {"query": "test"}, 0.5, True)

        assert len(tracker.tool_calls_made) == 1
        assert tracker.tool_calls_made[0]["tool_name"] == "search"
        assert tracker.tool_calls_made[0]["success"] == True
        assert len(tracker.execution_times) == 1

        # Log failed call
        tracker.log_tool_call("search", {"query": "test2"}, 0.3, False, "Error message")

        assert len(tracker.tool_calls_made) == 2
        assert tracker.tool_calls_made[1]["success"] == False
        assert len(tracker.errors) == 1
        assert tracker.errors[0] == "Error message"

    def test_get_summary(self):
        """Test getting execution summary"""
        tracker = ToolCallTracker(max_rounds=2)
        tracker.start_round()

        # Log some calls
        tracker.log_tool_call("search", {"query": "test1"}, 0.5, True)
        tracker.log_tool_call("search", {"query": "test2"}, 0.3, False, "Error")

        summary = tracker.get_summary()

        assert summary["total_rounds"] == 1
        assert summary["total_tool_calls"] == 2
        assert summary["successful_calls"] == 1
        assert summary["failed_calls"] == 1
        assert summary["total_execution_time"] == 0.8
        assert len(summary["errors"]) == 1
