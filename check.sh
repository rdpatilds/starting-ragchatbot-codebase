#!/bin/bash

# Code quality check script (read-only checks)
# This script runs quality checks without modifying files

set -e

echo "🔍 Running code quality checks (read-only)..."

# Check code formatting with black (dry-run)
echo "📝 Checking code formatting with black..."
uv run black --check --diff backend/

# Check import sorting with isort (dry-run)
echo "📋 Checking import sorting with isort..."
uv run isort --check-only --diff backend/

# Run flake8 linter
echo "🔍 Running flake8 linter..."
uv run flake8 backend/

# Run mypy type checker
echo "🏷️  Running mypy type checker..."
uv run mypy backend/

# Run tests
echo "🧪 Running tests..."
cd backend && uv run pytest tests/ -v

echo "✅ All quality checks passed!"