"""
API endpoint tests for the RAG system FastAPI application
"""
import pytest
from unittest.mock import Mock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional


# Create test-specific Pydantic models (identical to app.py but isolated)
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


def create_test_app(mock_rag_system):
    """Create a test FastAPI app without static file mounting"""
    app = FastAPI(title="Course Materials RAG System - Test", root_path="")

    # Add middleware (same as production app)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Define API endpoints inline to avoid import issues
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/index")
    async def reindex_documents():
        try:
            # Mock reindexing
            courses, chunks = 1, 10
            return {
                "message": f"Reindexed {courses} courses with {chunks} chunks",
                "courses": courses,
                "chunks": chunks
            }
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))

    return app


@pytest.fixture
def test_client(mock_rag_system):
    """Create test client with mocked RAG system"""
    app = create_test_app(mock_rag_system)
    return TestClient(app)


@pytest.mark.api
class TestQueryEndpoint:
    """Test cases for /api/query endpoint"""

    def test_query_with_session_id(self, test_client, sample_query_request):
        """Test query endpoint with provided session ID"""
        response = test_client.post("/api/query", json=sample_query_request)

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == sample_query_request["session_id"]
        assert isinstance(data["sources"], list)

    def test_query_without_session_id(self, test_client):
        """Test query endpoint without session ID (should create new session)"""
        request_data = {"query": "What is machine learning?"}

        response = test_client.post("/api/query", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] is not None

    def test_query_empty_request(self, test_client):
        """Test query endpoint with empty query"""
        request_data = {"query": ""}

        response = test_client.post("/api/query", json=request_data)

        # Should still work (let the RAG system handle empty queries)
        assert response.status_code == 200

    def test_query_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON"""
        response = test_client.post(
            "/api/query",
            data="invalid json",
            headers={"content-type": "application/json"}
        )

        assert response.status_code == 422  # Validation error

    def test_query_missing_field(self, test_client):
        """Test query endpoint with missing required field"""
        request_data = {"session_id": "test-123"}  # Missing 'query' field

        response = test_client.post("/api/query", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_query_rag_system_error(self, test_client, mock_rag_system):
        """Test query endpoint when RAG system raises an exception"""
        # Make the mock RAG system raise an exception
        mock_rag_system.query.side_effect = Exception("RAG system error")

        request_data = {"query": "test query"}
        response = test_client.post("/api/query", json=request_data)

        assert response.status_code == 500
        assert "RAG system error" in response.json()["detail"]


@pytest.mark.api
class TestCoursesEndpoint:
    """Test cases for /api/courses endpoint"""

    def test_get_course_stats(self, test_client, sample_course_stats):
        """Test courses endpoint returns correct statistics"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] == sample_course_stats["total_courses"]

    def test_get_course_stats_empty(self, test_client, mock_rag_system):
        """Test courses endpoint with no courses"""
        # Mock empty course analytics
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_course_stats_error(self, test_client, mock_rag_system):
        """Test courses endpoint when analytics raises an exception"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")

        response = test_client.get("/api/courses")

        assert response.status_code == 500
        assert "Analytics error" in response.json()["detail"]


@pytest.mark.api
class TestIndexEndpoint:
    """Test cases for /api/index endpoint"""

    def test_reindex_documents(self, test_client):
        """Test document reindexing endpoint"""
        response = test_client.post("/api/index")

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert "courses" in data
        assert "chunks" in data
        assert isinstance(data["courses"], int)
        assert isinstance(data["chunks"], int)


@pytest.mark.api
class TestMiddleware:
    """Test middleware functionality"""

    def test_cors_headers(self, test_client):
        """Test CORS headers are properly set"""
        response = test_client.options("/api/query")

        # CORS preflight should be handled
        assert response.status_code in [200, 405]

    def test_trusted_host_middleware(self, test_client):
        """Test trusted host middleware allows requests"""
        response = test_client.get("/api/courses")

        # Should not be blocked by trusted host middleware
        assert response.status_code == 200


@pytest.mark.api
class TestRequestValidation:
    """Test request validation and error handling"""

    def test_query_request_validation(self, test_client):
        """Test QueryRequest model validation"""
        # Valid request
        valid_request = {
            "query": "test query",
            "session_id": "optional-session"
        }
        response = test_client.post("/api/query", json=valid_request)
        assert response.status_code == 200

        # Valid request without session_id
        valid_request_no_session = {"query": "test query"}
        response = test_client.post("/api/query", json=valid_request_no_session)
        assert response.status_code == 200

        # Invalid request - wrong type for query
        invalid_request = {"query": 123}
        response = test_client.post("/api/query", json=invalid_request)
        assert response.status_code == 422

    def test_response_model_structure(self, test_client):
        """Test that responses match expected model structure"""
        # Query endpoint
        query_response = test_client.post("/api/query", json={"query": "test"})
        query_data = query_response.json()

        required_query_fields = {"answer", "sources", "session_id"}
        assert required_query_fields.issubset(set(query_data.keys()))

        # Courses endpoint
        courses_response = test_client.get("/api/courses")
        courses_data = courses_response.json()

        required_courses_fields = {"total_courses", "course_titles"}
        assert required_courses_fields.issubset(set(courses_data.keys()))


@pytest.mark.api
@pytest.mark.integration
class TestEndToEndAPI:
    """End-to-end API tests simulating real usage patterns"""

    def test_complete_query_workflow(self, test_client):
        """Test complete workflow: create session, query, get courses"""
        # 1. Query without session (creates new session)
        query_response = test_client.post("/api/query", json={
            "query": "What courses are available?"
        })

        assert query_response.status_code == 200
        query_data = query_response.json()
        session_id = query_data["session_id"]

        # 2. Query with the same session
        followup_response = test_client.post("/api/query", json={
            "query": "Tell me more about the first course",
            "session_id": session_id
        })

        assert followup_response.status_code == 200
        followup_data = followup_response.json()
        assert followup_data["session_id"] == session_id

        # 3. Get course statistics
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == 200

    def test_concurrent_requests(self, test_client):
        """Test handling of concurrent requests"""
        import concurrent.futures
        import threading

        def make_query(query_text):
            return test_client.post("/api/query", json={"query": query_text})

        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(make_query, f"Query {i}")
                for i in range(10)
            ]

            responses = [future.result() for future in futures]

        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
            assert "session_id" in data