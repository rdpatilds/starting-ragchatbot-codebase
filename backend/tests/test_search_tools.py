import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, MagicMock
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test the CourseSearchTool execute method"""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store"""
        mock_store = Mock()
        return mock_store

    @pytest.fixture
    def search_tool(self, mock_vector_store):
        """Create search tool with mock vector store"""
        return CourseSearchTool(mock_vector_store)

    def test_execute_with_empty_results(self, search_tool, mock_vector_store):
        """Test execute when vector store returns empty results"""
        # Configure mock to return empty results
        empty_results = SearchResults(documents=[], metadata=[], distances=[], error=None)
        mock_vector_store.search.return_value = empty_results

        # Execute search
        result = search_tool.execute(query="Python basics")

        # Should return no content found message
        assert "No relevant content found" in result
        mock_vector_store.search.assert_called_once_with(
            query="Python basics",
            course_name=None,
            lesson_number=None
        )

    def test_execute_with_results(self, search_tool, mock_vector_store):
        """Test execute when vector store returns results"""
        # Configure mock to return results
        results = SearchResults(
            documents=["Python is a programming language", "Variables store data"],
            metadata=[
                {"course_title": "Python Basics", "lesson_number": 0},
                {"course_title": "Python Basics", "lesson_number": 1}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        mock_vector_store.search.return_value = results

        # Execute search
        result = search_tool.execute(query="Python programming")

        # Should format results properly
        assert "[Python Basics - Lesson 0]" in result
        assert "[Python Basics - Lesson 1]" in result
        assert "Python is a programming language" in result
        assert "Variables store data" in result

    def test_execute_with_course_filter(self, search_tool, mock_vector_store):
        """Test execute with course name filter"""
        # Configure mock
        results = SearchResults(
            documents=["Content from specific course"],
            metadata=[{"course_title": "Advanced Python", "lesson_number": 0}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.search.return_value = results

        # Execute search with course filter
        result = search_tool.execute(
            query="decorators",
            course_name="Advanced Python"
        )

        # Verify course filter was passed
        mock_vector_store.search.assert_called_once_with(
            query="decorators",
            course_name="Advanced Python",
            lesson_number=None
        )
        assert "Advanced Python" in result

    def test_execute_with_lesson_filter(self, search_tool, mock_vector_store):
        """Test execute with lesson number filter"""
        # Configure mock
        results = SearchResults(
            documents=["Lesson specific content"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 3}],
            distances=[0.1],
            error=None
        )
        mock_vector_store.search.return_value = results

        # Execute search with lesson filter
        result = search_tool.execute(
            query="functions",
            lesson_number=3
        )

        # Verify lesson filter was passed
        mock_vector_store.search.assert_called_once_with(
            query="functions",
            course_name=None,
            lesson_number=3
        )
        assert "Lesson 3" in result

    def test_execute_with_error(self, search_tool, mock_vector_store):
        """Test execute when vector store returns error"""
        # Configure mock to return error
        error_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        mock_vector_store.search.return_value = error_results

        # Execute search
        result = search_tool.execute(query="test query")

        # Should return error message
        assert result == "Database connection failed"

    def test_source_tracking(self, search_tool, mock_vector_store):
        """Test that sources are properly tracked"""
        # Configure mock to return results
        results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 0},
                {"course_title": "Course B", "lesson_number": 2}
            ],
            distances=[0.1, 0.2],
            error=None
        )
        mock_vector_store.search.return_value = results

        # Execute search
        search_tool.execute(query="test")

        # Check sources are tracked
        assert len(search_tool.last_sources) == 2
        assert "Course A - Lesson 0" in search_tool.last_sources
        assert "Course B - Lesson 2" in search_tool.last_sources

    def test_tool_definition(self, search_tool):
        """Test the tool definition for Anthropic"""
        definition = search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["properties"]["query"]["type"] == "string"
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]


class TestToolManager:
    """Test the ToolManager class"""

    @pytest.fixture
    def tool_manager(self):
        """Create a tool manager"""
        return ToolManager()

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool"""
        tool = Mock()
        tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "Test tool"
        }
        tool.execute.return_value = "Tool executed"
        return tool

    def test_register_tool(self, tool_manager, mock_tool):
        """Test registering a tool"""
        tool_manager.register_tool(mock_tool)

        assert "test_tool" in tool_manager.tools
        assert tool_manager.tools["test_tool"] == mock_tool

    def test_get_tool_definitions(self, tool_manager, mock_tool):
        """Test getting all tool definitions"""
        tool_manager.register_tool(mock_tool)

        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "test_tool"

    def test_execute_tool(self, tool_manager, mock_tool):
        """Test executing a tool by name"""
        tool_manager.register_tool(mock_tool)

        result = tool_manager.execute_tool("test_tool", param1="value1")

        assert result == "Tool executed"
        mock_tool.execute.assert_called_once_with(param1="value1")

    def test_execute_nonexistent_tool(self, tool_manager):
        """Test executing a tool that doesn't exist"""
        result = tool_manager.execute_tool("nonexistent_tool")

        assert "not found" in result

    def test_get_last_sources(self, tool_manager):
        """Test getting last sources from tools"""
        # Create a tool with last_sources
        tool = Mock()
        tool.get_tool_definition.return_value = {"name": "search_tool"}
        tool.last_sources = ["Source 1", "Source 2"]

        tool_manager.register_tool(tool)

        sources = tool_manager.get_last_sources()

        assert sources == ["Source 1", "Source 2"]

    def test_reset_sources(self, tool_manager):
        """Test resetting sources"""
        # Create a tool with last_sources
        tool = Mock()
        tool.get_tool_definition.return_value = {"name": "search_tool"}
        tool.last_sources = ["Source 1", "Source 2"]

        tool_manager.register_tool(tool)
        tool_manager.reset_sources()

        assert tool.last_sources == []