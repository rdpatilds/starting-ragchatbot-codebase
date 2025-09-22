"""
Shared pytest fixtures for the RAG system tests
"""
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
from fastapi.testclient import TestClient

# Mock the RAG system components to avoid real API calls and file operations
@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    from unittest.mock import Mock

    config = Mock()
    config.CHROMA_PATH = ":memory:"
    config.EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_CONVERSATION_HISTORY = 2
    config.ANTHROPIC_API_KEY = "test-key"
    config.MODEL_NAME = "claude-sonnet-4-20250514"
    return config


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing"""
    mock_store = Mock()
    mock_store.search.return_value = Mock(
        is_empty=lambda: False,
        documents=[["test document content"]],
        metadatas=[[{"course": "Test Course", "lesson": "Test Lesson", "source": "test.txt"}]],
        ids=[["test_id_1"]]
    )
    mock_store.get_course_count.return_value = 1
    mock_store.get_courses.return_value = ["Test Course"]
    return mock_store


@pytest.fixture
def mock_ai_generator():
    """Mock AI generator for testing"""
    mock_generator = Mock()
    mock_generator.generate_response.return_value = (
        "This is a test response",
        ["test.txt - Test Course: Test Lesson"]
    )
    return mock_generator


@pytest.fixture
def mock_search_tools():
    """Mock search tools for testing"""
    mock_tools = Mock()
    mock_tools.get_tools.return_value = []
    mock_tools.execute_tool.return_value = "Mock search results"
    return mock_tools


@pytest.fixture
def mock_session_manager():
    """Mock session manager for testing"""
    mock_manager = Mock()
    mock_manager.create_session.return_value = "test-session-id"
    mock_manager.add_exchange.return_value = None
    mock_manager.get_conversation_history.return_value = []
    return mock_manager


@pytest.fixture
def mock_rag_system(mock_config, mock_vector_store, mock_ai_generator, mock_search_tools, mock_session_manager):
    """Mock RAG system with all dependencies"""
    mock_rag = Mock()
    mock_rag.config = mock_config
    mock_rag.vector_store = mock_vector_store
    mock_rag.ai_generator = mock_ai_generator
    mock_rag.tool_manager = mock_search_tools
    mock_rag.session_manager = mock_session_manager

    # Mock the query method
    mock_rag.query.return_value = (
        "This is a test response to your query",
        ["test.txt - Test Course: Test Lesson"]
    )

    # Mock analytics method
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Test Course"]
    }

    return mock_rag


@pytest.fixture
def temp_docs_dir():
    """Create a temporary directory with test documents"""
    temp_dir = tempfile.mkdtemp()
    docs_dir = Path(temp_dir) / "docs"
    docs_dir.mkdir()

    # Create a test document
    test_doc = docs_dir / "test_course.txt"
    test_doc.write_text("""Course Title: Test Course
Course Link: https://example.com/test-course
Course Instructor: Test Instructor

Lesson 0: Introduction
This is the introduction lesson content.

Lesson 1: Basics
This covers the basic concepts.
""")

    yield str(docs_dir)

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_query_request():
    """Sample query request data for API testing"""
    return {
        "query": "What is the introduction lesson about?",
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_query_response():
    """Sample query response data for API testing"""
    return {
        "answer": "The introduction lesson covers basic course concepts.",
        "sources": ["test_course.txt - Test Course: Introduction"],
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_course_stats():
    """Sample course statistics for API testing"""
    return {
        "total_courses": 1,
        "course_titles": ["Test Course"]
    }


@pytest.fixture(autouse=True)
def mock_anthropic_client():
    """Mock Anthropic client to avoid real API calls"""
    with patch('anthropic.Anthropic') as mock_client:
        # Mock the client instance
        mock_instance = Mock()
        mock_client.return_value = mock_instance

        # Mock the messages.create method
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.stop_reason = "end_turn"
        mock_instance.messages.create.return_value = mock_response

        yield mock_instance


@pytest.fixture
def mock_sentence_transformers():
    """Mock SentenceTransformer to avoid downloading models"""
    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]  # Mock embedding
        mock_st.return_value = mock_model
        yield mock_model