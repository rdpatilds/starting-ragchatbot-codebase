#!/bin/bash

# Code formatting and quality checks script
# This script runs all code quality tools for the RAG chatbot project

set -e

echo "🔧 Running code quality checks..."

# Run black formatter
echo "📝 Formatting code with black..."
uv run black backend/

# Sort imports with isort
echo "📋 Sorting imports with isort..."
uv run isort backend/

# Run flake8 linter
echo "🔍 Running flake8 linter..."
uv run flake8 backend/

# Run mypy type checker
echo "🏷️  Running mypy type checker..."
uv run mypy backend/

echo "✅ All quality checks completed!"