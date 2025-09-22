import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from config import Config, config


class TestConfig:
    """Test configuration values and environment loading"""

    def test_max_results_not_zero(self):
        """Test that MAX_RESULTS is not set to 0 (this is the bug)"""
        # This test will FAIL with current config where MAX_RESULTS = 0
        assert (
            config.MAX_RESULTS > 0
        ), f"MAX_RESULTS is {config.MAX_RESULTS}, should be > 0"

    def test_max_results_reasonable_range(self):
        """Test that MAX_RESULTS is in a reasonable range"""
        assert (
            1 <= config.MAX_RESULTS <= 20
        ), f"MAX_RESULTS ({config.MAX_RESULTS}) should be between 1 and 20"

    def test_chunk_size_valid(self):
        """Test that chunk size is valid"""
        assert config.CHUNK_SIZE > 0, "CHUNK_SIZE must be positive"
        assert config.CHUNK_SIZE >= 100, "CHUNK_SIZE too small for meaningful content"

    def test_chunk_overlap_valid(self):
        """Test that chunk overlap is valid"""
        assert config.CHUNK_OVERLAP >= 0, "CHUNK_OVERLAP must be non-negative"
        assert (
            config.CHUNK_OVERLAP < config.CHUNK_SIZE
        ), "CHUNK_OVERLAP must be less than CHUNK_SIZE"

    def test_anthropic_api_key_exists(self):
        """Test that Anthropic API key is configured"""
        assert config.ANTHROPIC_API_KEY != "", "ANTHROPIC_API_KEY is not set"

    def test_model_configuration(self):
        """Test model configurations are valid"""
        assert config.ANTHROPIC_MODEL != "", "ANTHROPIC_MODEL is not set"
        assert config.EMBEDDING_MODEL != "", "EMBEDDING_MODEL is not set"

    def test_max_history_valid(self):
        """Test conversation history setting"""
        assert config.MAX_HISTORY >= 0, "MAX_HISTORY must be non-negative"
        assert (
            config.MAX_HISTORY <= 10
        ), "MAX_HISTORY too large, may cause context issues"

    def test_chroma_path_configured(self):
        """Test ChromaDB path is configured"""
        assert config.CHROMA_PATH != "", "CHROMA_PATH is not set"
        assert isinstance(config.CHROMA_PATH, str), "CHROMA_PATH must be a string"
