import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, MagicMock, patch
import tempfile
import shutil
from rag_system import RAGSystem
from config import Config


class TestRAGSystem:
    """Test the complete RAG system pipeline"""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration"""
        config = Mock(spec=Config)
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.MAX_RESULTS = 0  # Test with the bug
        config.CHROMA_PATH = tempfile.mkdtemp()
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.ANTHROPIC_API_KEY = "test_key"
        config.ANTHROPIC_MODEL = "test_model"
        config.MAX_HISTORY = 2
        config.HF_TOKEN = ""
        yield config
        # Cleanup
        shutil.rmtree(config.CHROMA_PATH, ignore_errors=True)

    @pytest.fixture
    def mock_config_fixed(self):
        """Create a fixed configuration"""
        config = Mock(spec=Config)
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.MAX_RESULTS = 5  # Fixed value
        config.CHROMA_PATH = tempfile.mkdtemp()
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.ANTHROPIC_API_KEY = "test_key"
        config.ANTHROPIC_MODEL = "test_model"
        config.MAX_HISTORY = 2
        config.HF_TOKEN = ""
        yield config
        # Cleanup
        shutil.rmtree(config.CHROMA_PATH, ignore_errors=True)

    @pytest.fixture
    def rag_system_with_bug(self, mock_config):
        """Create RAG system with MAX_RESULTS=0 bug"""
        with patch('rag_system.AIGenerator') as mock_ai:
            mock_ai.return_value.generate_response.return_value = "Query failed: No results found"
            return RAGSystem(mock_config)

    @pytest.fixture
    def rag_system_fixed(self, mock_config_fixed):
        """Create RAG system with fixed configuration"""
        with patch('rag_system.AIGenerator') as mock_ai:
            mock_ai.return_value.generate_response.return_value = "Python is a programming language"
            return RAGSystem(mock_config_fixed)

    def test_rag_system_initialization(self, mock_config):
        """Test that RAG system initializes all components"""
        with patch('rag_system.AIGenerator'):
            rag_system = RAGSystem(mock_config)

            assert rag_system.document_processor is not None
            assert rag_system.vector_store is not None
            assert rag_system.ai_generator is not None
            assert rag_system.session_manager is not None
            assert rag_system.tool_manager is not None
            assert rag_system.search_tool is not None

    def test_vector_store_max_results_bug(self, mock_config):
        """Test that vector store is initialized with MAX_RESULTS from config"""
        with patch('rag_system.AIGenerator'):
            rag_system = RAGSystem(mock_config)

            # Vector store should have max_results=0 (the bug)
            assert rag_system.vector_store.max_results == 0

    def test_vector_store_max_results_fixed(self, mock_config_fixed):
        """Test that vector store works with fixed MAX_RESULTS"""
        with patch('rag_system.AIGenerator'):
            rag_system = RAGSystem(mock_config_fixed)

            # Vector store should have max_results=5
            assert rag_system.vector_store.max_results == 5

    def test_query_with_zero_max_results(self, rag_system_with_bug):
        """Test query behavior with MAX_RESULTS=0 (demonstrates the bug)"""
        # Mock the AI generator to simulate actual behavior
        with patch.object(rag_system_with_bug.ai_generator, 'generate_response') as mock_gen:
            # When no results are found, typical response
            mock_gen.return_value = "I couldn't find any relevant information in the course materials."

            response, sources = rag_system_with_bug.query("What is Python?")

            # With MAX_RESULTS=0, we expect no useful response
            assert "couldn't find" in response.lower() or "no" in response.lower()
            assert sources == [] or len(sources) == 0

    def test_query_with_fixed_max_results(self, rag_system_fixed):
        """Test query behavior with fixed MAX_RESULTS"""
        # This should work properly
        response, sources = rag_system_fixed.query("What is Python?")

        # Should return a meaningful response
        assert "Python" in response
        assert "programming" in response.lower()

    def test_tool_registration(self, mock_config):
        """Test that search tool is properly registered"""
        with patch('rag_system.AIGenerator'):
            rag_system = RAGSystem(mock_config)

            # Check tool is registered
            tool_defs = rag_system.tool_manager.get_tool_definitions()
            assert len(tool_defs) > 0
            assert any(tool["name"] == "search_course_content" for tool in tool_defs)

    def test_session_management(self, rag_system_fixed):
        """Test session management in queries"""
        # First query without session
        response1, sources1 = rag_system_fixed.query("First question")

        # Second query with new session
        session_id = rag_system_fixed.session_manager.create_session()
        response2, sources2 = rag_system_fixed.query("Second question", session_id)

        # Session should have the exchange
        history = rag_system_fixed.session_manager.get_conversation_history(session_id)
        assert history is not None

    def test_add_course_document_error_handling(self, rag_system_fixed):
        """Test error handling when adding course documents"""
        # Try to add non-existent document
        course, chunks = rag_system_fixed.add_course_document("/nonexistent/file.txt")

        assert course is None
        assert chunks == 0

    def test_get_course_analytics(self, rag_system_fixed):
        """Test getting course analytics"""
        analytics = rag_system_fixed.get_course_analytics()

        assert "total_courses" in analytics
        assert "course_titles" in analytics
        assert isinstance(analytics["total_courses"], int)
        assert isinstance(analytics["course_titles"], list)

    @patch('rag_system.os.path.exists')
    @patch('rag_system.os.listdir')
    def test_add_course_folder(self, mock_listdir, mock_exists, rag_system_fixed):
        """Test adding course folder"""
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.pdf"]

        # Mock document processor
        with patch.object(rag_system_fixed.document_processor, 'process_course_document') as mock_process:
            # Return None to simulate processing without actual file reading
            mock_process.return_value = (None, [])

            courses, chunks = rag_system_fixed.add_course_folder("/test/folder")

            # Should attempt to process files
            assert mock_process.call_count == 2

    def test_query_integration(self, mock_config_fixed):
        """Integration test for the full query pipeline"""
        with patch('rag_system.AIGenerator') as mock_ai_class:
            # Set up mock AI generator
            mock_ai = mock_ai_class.return_value
            mock_ai.generate_response.return_value = "Test response about Python"

            rag_system = RAGSystem(mock_config_fixed)

            # Execute query
            response, sources = rag_system.query("Tell me about Python")

            # Verify AI generator was called with tools
            mock_ai.generate_response.assert_called_once()
            call_args = mock_ai.generate_response.call_args

            # Check that tools were provided
            assert "tools" in call_args[1]
            assert "tool_manager" in call_args[1]

            # Check response
            assert response == "Test response about Python"