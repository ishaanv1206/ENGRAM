"""
Unit tests for main application entry point.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.main import setup_logging, validate_paths
from src.config import SystemConfig, Neo4jConfig, MainLLMConfig, SLMConfig, StorageConfig, ConfigurationError


class TestMainApplication:
    """Test cases for main application functions."""
    
    def test_setup_logging_creates_logs_directory(self):
        """Test that setup_logging creates logs directory."""
        config = SystemConfig(
            neo4j=Neo4jConfig("bolt://localhost:7687", "neo4j", "password"),
            main_llm=MainLLMConfig("./models/main.gguf", 4096, -1),
            slm=SLMConfig("./models/slm.gguf", 2048, -1),
            storage=StorageConfig("./data/pinned.json", "./data/sessions/", "./data/archive/"),
            log_level="INFO",
            max_conversation_turns=10,
            memory_budget_tokens=500,
            cache_size_tier1=100,
            gradio_host="127.0.0.1",
            gradio_port=7860,
            gradio_share=False
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory for test
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                setup_logging(config)
                
                # Check that logs directory was created
                logs_dir = Path("logs")
                assert logs_dir.exists()
                assert logs_dir.is_dir()
                
            finally:
                os.chdir(original_cwd)
    
    def test_validate_paths_missing_model_file_raises_error(self):
        """Test that validate_paths raises error for missing model files."""
        config = SystemConfig(
            neo4j=Neo4jConfig("bolt://localhost:7687", "neo4j", "password"),
            main_llm=MainLLMConfig("./nonexistent/main.gguf", 4096, -1),
            slm=SLMConfig("./models/slm.gguf", 2048, -1),
            storage=StorageConfig("./data/pinned.json", "./data/sessions/", "./data/archive/"),
            log_level="INFO",
            max_conversation_turns=10,
            memory_budget_tokens=500,
            cache_size_tier1=100,
            gradio_host="127.0.0.1",
            gradio_port=7860,
            gradio_share=False
        )
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_paths(config)
        
        assert "Main LLM model file not found" in str(exc_info.value)
    
    def test_validate_paths_creates_storage_directories(self):
        """Test that validate_paths creates storage directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create temporary model files
            main_llm_path = Path(temp_dir) / "main.gguf"
            slm_path = Path(temp_dir) / "slm.gguf"
            main_llm_path.touch()
            slm_path.touch()
            
            config = SystemConfig(
                neo4j=Neo4jConfig("bolt://localhost:7687", "neo4j", "password"),
                main_llm=MainLLMConfig(str(main_llm_path), 4096, -1),
                slm=SLMConfig(str(slm_path), 2048, -1),
                storage=StorageConfig(
                    str(Path(temp_dir) / "pinned.json"),
                    str(Path(temp_dir) / "sessions"),
                    str(Path(temp_dir) / "archive")
                ),
                log_level="INFO",
                max_conversation_turns=10,
                memory_budget_tokens=500,
                cache_size_tier1=100,
                gradio_host="127.0.0.1",
                gradio_port=7860,
                gradio_share=False
            )
            
            # Should not raise any errors
            validate_paths(config)
            
            # Check that directories were created
            assert Path(temp_dir, "sessions").exists()
            assert Path(temp_dir, "archive").exists()