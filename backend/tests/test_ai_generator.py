import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, MagicMock, patch
from ai_generator import AIGenerator


class TestAIGenerator:
    """Test AI Generator's ability to call tools correctly"""

    @pytest.fixture
    def mock_anthropic_client(self):
        """Create mock Anthropic client"""
        with patch('ai_generator.anthropic.Anthropic') as mock:
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
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            }
        ]
        manager.execute_tool.return_value = "Search results: Python is a programming language"
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

    def test_generate_response_with_tool_call(self, ai_generator, mock_anthropic_client, mock_tool_manager):
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
        mock_final_response.content = [Mock(text="Based on the search, Python is a programming language")]

        # Set up the mock to return different responses
        mock_anthropic_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response
        ]

        # Generate response with tools
        result = ai_generator.generate_response(
            "What is Python?",
            tools=mock_tool_manager.get_tool_definitions(),
            tool_manager=mock_tool_manager
        )

        # Verify tool was called
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="Python basics"
        )

        # Verify final response
        assert "Based on the search" in result

    def test_handle_tool_execution(self, ai_generator, mock_anthropic_client, mock_tool_manager):
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
        mock_final_response.content = [Mock(text="Decorators are functions that modify other functions")]

        mock_anthropic_client.messages.create.return_value = mock_final_response

        # Test tool execution handling
        base_params = {
            "messages": [{"role": "user", "content": "What are decorators?"}],
            "system": "Test system prompt"
        }

        result = ai_generator._handle_tool_execution(
            mock_initial_response,
            base_params,
            mock_tool_manager
        )

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="decorators"
        )

        # Verify result
        assert "Decorators are functions" in result

    def test_conversation_history_integration(self, ai_generator, mock_anthropic_client):
        """Test that conversation history is properly integrated"""
        # Configure mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with history")]
        mock_response.stop_reason = "stop"
        mock_anthropic_client.messages.create.return_value = mock_response

        # Generate response with history
        history = "User: Previous question\nAssistant: Previous answer"
        result = ai_generator.generate_response(
            "New question",
            conversation_history=history
        )

        # Verify history was included in system prompt
        call_args = mock_anthropic_client.messages.create.call_args
        assert "Previous conversation:" in call_args[1]["system"]
        assert history in call_args[1]["system"]

    def test_system_prompt_structure(self, ai_generator):
        """Test that system prompt is properly structured"""
        assert "course materials" in ai_generator.SYSTEM_PROMPT
        assert "search tool" in ai_generator.SYSTEM_PROMPT.lower()
        assert "One search per query maximum" in ai_generator.SYSTEM_PROMPT

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

    def test_tool_choice_auto(self, ai_generator, mock_anthropic_client, mock_tool_manager):
        """Test that tool_choice is set to auto when tools are provided"""
        # Configure mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="Response")]
        mock_response.stop_reason = "stop"
        mock_anthropic_client.messages.create.return_value = mock_response

        # Generate response with tools
        ai_generator.generate_response(
            "Query",
            tools=mock_tool_manager.get_tool_definitions()
        )

        # Verify tool_choice was set
        call_args = mock_anthropic_client.messages.create.call_args
        assert "tool_choice" in call_args[1]
        assert call_args[1]["tool_choice"]["type"] == "auto"