import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
import shutil
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk
from config import config


class TestVectorStore:
    """Test vector store functionality including the MAX_RESULTS bug"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary directory for test database"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def vector_store_zero_results(self, temp_db_path):
        """Create vector store with MAX_RESULTS=0 (reproduces bug)"""
        return VectorStore(temp_db_path, config.EMBEDDING_MODEL, max_results=0)

    @pytest.fixture
    def vector_store_normal(self, temp_db_path):
        """Create vector store with normal MAX_RESULTS"""
        return VectorStore(temp_db_path, config.EMBEDDING_MODEL, max_results=5)

    @pytest.fixture
    def sample_course(self):
        """Create sample course for testing"""
        lessons = [
            Lesson(
                lesson_number=0,
                title="Introduction to Python",
                content="Python is a programming language",
                lesson_link="https://example.com/lesson0"
            ),
            Lesson(
                lesson_number=1,
                title="Variables and Types",
                content="Variables store data. Types include int, float, string",
                lesson_link="https://example.com/lesson1"
            )
        ]
        return Course(
            title="Python Basics",
            course_link="https://example.com/course",
            instructor="Test Instructor",
            lessons=lessons
        )

    @pytest.fixture
    def sample_chunks(self, sample_course):
        """Create sample chunks for testing"""
        return [
            CourseChunk(
                course_title=sample_course.title,
                lesson_number=0,
                content="Python is a high-level programming language",
                chunk_index=0
            ),
            CourseChunk(
                course_title=sample_course.title,
                lesson_number=1,
                content="Variables in Python can store different types of data",
                chunk_index=1
            )
        ]

    def test_search_with_zero_max_results(self, vector_store_zero_results, sample_course, sample_chunks):
        """Test that MAX_RESULTS=0 returns no results (demonstrates bug)"""
        # Add test data
        vector_store_zero_results.add_course_metadata(sample_course)
        vector_store_zero_results.add_course_content(sample_chunks)

        # Search should return empty results due to max_results=0
        results = vector_store_zero_results.search("Python programming")

        assert results.is_empty(), "Search with MAX_RESULTS=0 should return empty results"
        assert len(results.documents) == 0, f"Expected 0 documents, got {len(results.documents)}"

    def test_search_with_normal_max_results(self, vector_store_normal, sample_course, sample_chunks):
        """Test that normal MAX_RESULTS returns results"""
        # Add test data
        vector_store_normal.add_course_metadata(sample_course)
        vector_store_normal.add_course_content(sample_chunks)

        # Search should return results
        results = vector_store_normal.search("Python programming")

        assert not results.is_empty(), "Search with normal MAX_RESULTS should return results"
        assert len(results.documents) > 0, f"Expected results, got {len(results.documents)} documents"

    def test_course_name_resolution(self, vector_store_normal, sample_course):
        """Test course name resolution works"""
        vector_store_normal.add_course_metadata(sample_course)

        # Test exact match
        resolved = vector_store_normal._resolve_course_name("Python Basics")
        assert resolved == "Python Basics", f"Expected 'Python Basics', got '{resolved}'"

        # Test partial match
        resolved_partial = vector_store_normal._resolve_course_name("Python")
        assert resolved_partial == "Python Basics", f"Expected 'Python Basics', got '{resolved_partial}'"

    def test_lesson_filtering(self, vector_store_normal, sample_course, sample_chunks):
        """Test filtering by lesson number"""
        vector_store_normal.add_course_metadata(sample_course)
        vector_store_normal.add_course_content(sample_chunks)

        # Search in specific lesson
        results = vector_store_normal.search("data", lesson_number=1)

        if not results.is_empty():
            # Check all results are from lesson 1
            for metadata in results.metadata:
                assert metadata.get('lesson_number') == 1, "Results should only be from lesson 1"

    def test_course_filtering(self, vector_store_normal, sample_course, sample_chunks):
        """Test filtering by course name"""
        vector_store_normal.add_course_metadata(sample_course)
        vector_store_normal.add_course_content(sample_chunks)

        # Search in specific course
        results = vector_store_normal.search("programming", course_name="Python Basics")

        if not results.is_empty():
            # Check all results are from the specified course
            for metadata in results.metadata:
                assert metadata.get('course_title') == "Python Basics", "Results should only be from Python Basics"

    def test_empty_search_response(self, vector_store_normal):
        """Test handling of searches with no results"""
        # Search in empty database
        results = vector_store_normal.search("nonexistent content")

        assert results.is_empty(), "Search in empty database should return empty results"
        assert results.error is None or "No relevant content found" in str(results.error)

    def test_get_course_count(self, vector_store_normal, sample_course):
        """Test course counting functionality"""
        # Initially should be 0
        count = vector_store_normal.get_course_count()
        assert count == 0, f"Initial count should be 0, got {count}"

        # Add a course
        vector_store_normal.add_course_metadata(sample_course)

        # Should be 1
        count = vector_store_normal.get_course_count()
        assert count == 1, f"Count after adding course should be 1, got {count}"

    def test_get_course_metadata(self, vector_store_normal, sample_course):
        """Test retrieving course metadata"""
        vector_store_normal.add_course_metadata(sample_course)

        metadata = vector_store_normal.get_all_courses_metadata()

        assert len(metadata) == 1, f"Expected 1 course metadata, got {len(metadata)}"
        assert metadata[0]['title'] == "Python Basics"
        assert metadata[0]['instructor'] == "Test Instructor"
        assert 'lessons' in metadata[0], "Metadata should include lessons"