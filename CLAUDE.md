# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) chatbot system that enables users to query course materials and receive intelligent, context-aware responses using ChromaDB for vector storage and Anthropic's Claude for AI generation.

## Essential Commands

### Running the Application

```bash
# Quick start with script (recommended)
./run.sh

# Manual start on custom port (e.g., 8010 if 8000 is in use)
cd backend
uv run uvicorn app:app --reload --port 8010

# Alternative: using main.py
uv run python main.py
```

### Dependency Management

```bash
# Install all dependencies (uses uv package manager)
uv sync

# The project requires Python 3.12+ (specified in pyproject.toml)
# uv will handle Python version automatically
```

### Code Quality and Formatting

```bash
# Format code and run all quality checks (modifies files)
./format.sh

# Run quality checks only (read-only, no file modifications)
./check.sh

# Individual tools (run from project root):
uv run black backend/          # Format code with black
uv run isort backend/          # Sort imports
uv run flake8 backend/         # Lint code
uv run mypy backend/           # Type checking
uv run pytest backend/tests/   # Run tests
```

### Environment Setup

Create a `.env` file with:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Architecture Overview

### Backend Structure (`/backend/`)

The backend follows a modular architecture with clear separation of concerns:

1. **Entry Point (`app.py`)**: FastAPI server handling HTTP endpoints
   - `/api/query` - Main query endpoint
   - `/api/courses` - Course statistics
   - `/api/index` - Re-index documents
   - Serves frontend static files

2. **Core RAG Pipeline (`rag_system.py`)**: 
   - Orchestrates the entire RAG workflow
   - Manages tool-based search integration with Claude
   - Coordinates between document processor, vector store, and AI generator

3. **Document Processing (`document_processor.py`)**:
   - Parses course documents expecting format: Course Title, Link, Instructor, then Lessons
   - Implements smart chunking with overlap (800 chars chunks, 100 chars overlap)
   - Preserves lesson context in chunks

4. **Vector Storage (`vector_store.py`)**:
   - Uses ChromaDB with two collections: `course_catalog` and `course_content`
   - Implements semantic search using sentence-transformers embeddings
   - Handles course name resolution and filtered searches

5. **AI Integration (`ai_generator.py`)**:
   - Manages Claude API interactions with tool use
   - Implements two-phase generation: tool use → final response
   - Handles conversation history integration

6. **Search Tools (`search_tools.py`)**:
   - Provides search tool interface for Claude
   - Formats search results with course/lesson context
   - Tracks sources for citation

7. **Session Management (`session_manager.py`)**:
   - Maintains conversation history per session
   - Configurable history length (default: 2 exchanges)

8. **Configuration (`config.py`)**:
   - Centralized settings management
   - Key parameters: CHUNK_SIZE=800, CHUNK_OVERLAP=100, MAX_RESULTS=5

### Frontend Structure (`/frontend/`)

- **index.html**: Layout with sidebar (course stats, suggested questions) and main chat area
- **script.js**: Handles user interactions, API calls, message display with markdown support
- **style.css**: Dark theme styling with animations

### Data Flow

1. User query → Frontend `sendMessage()` → POST `/api/query`
2. Backend creates/retrieves session → Calls `rag_system.query()`
3. RAG system prepares prompt → AI Generator with search tools
4. Claude decides to search → Executes search tool → Vector store query
5. ChromaDB returns relevant chunks → AI generates informed response
6. Response with sources → Frontend displays with markdown formatting

## Key Technical Details

- **Vector Database**: ChromaDB persisted in `./chroma_db/`
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **Claude Model**: `claude-sonnet-4-20250514`
- **Document Format**: Text files in `/docs/` with structured course/lesson format
- **Chunking Strategy**: Sentence-based with configurable overlap for context preservation

## Common Development Tasks

### Adding New Course Documents
Place `.txt` files in `/docs/` folder following the expected format:
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: [title]
[content]

Lesson 1: [title]
[content]
```

### Re-indexing Documents
```bash
# API endpoint to rebuild vector database
curl -X POST http://localhost:8010/api/index
```

### Modifying Configuration
Edit `backend/config.py` for:
- Chunk sizes and overlap
- Max search results
- Conversation history length
- Model selection

### Port Configuration
Default port is 8000. To change:
1. Edit `run.sh`: Change `--port 8000` to desired port
2. Or run manually with: `uv run uvicorn app:app --reload --port [PORT]`

## Important Notes

- The application automatically indexes documents on startup
- ChromaDB data persists between restarts in `backend/chroma_db/`
- Frontend uses relative API paths, works with any port
- Session management is in-memory (resets on server restart)
- Test suite available in `backend/tests/` directory
- always use uv to run the server do not use pip directly
- make sure to use uv for all dependency management