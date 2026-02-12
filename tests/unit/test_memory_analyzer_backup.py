"""
Unit tests for configuration management.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from src.config import ConfigManager, ConfigurationError, SystemConfig


class TestConfigManager:
    """Test cases for ConfigManager."""
    
    def test_load_config_with_valid_env_file(self):
        """Test loading configuration from a valid .env file."""
        # Create a temporary .env file with all required variables
        env_content = """NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=test_password
MAIN_LLM_MODEL_PATH=./models/main.gguf
STT_ENGINE=whisper
WHISPER_MODEL_PATH=./models/whisper.bin
SLM_MODEL_PATH=./models/slm.gguf
PINNED_MEMORY_PATH=./data/pinned.json
SESSION_STATE_PATH=./data/sessions/
ARCHIVE_PATH=./data/archive/"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            env_file_path = f.name
        
        try:
            # Clear environment to avoid interference
            env_vars_to_clear = [
                'NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD',
                'MAIN_LLM_MODEL_PATH', 'MAIN_LLM_N_CTX', 'MAIN_LLM_N_GPU_LAYERS',
                'STT_ENGINE', 'WHISPER_MODEL_PATH', 'VOSK_MODEL_PATH',
                'SLM_MODEL_PATH', 'SLM_N_CTX', 'SLM_N_GPU_LAYERS',
                'PINNED_MEMORY_PATH', 'SESSION_STATE_PATH', 'ARCHIVE_PATH',
                'LOG_LEVEL', 'MAX_CONVERSATION_TURNS', 'MEMORY_BUDGET_TOKENS',
                'CACHE_SIZE_TIER1', 'GRADIO_HOST', 'GRADIO_PORT', 'GRADIO_SHARE'
            ]
            
            with patch.dict(os.environ, {}, clear=False):
                # Remove specific env vars that might interfere
                for var in env_vars_to_clear:
                    os.environ.pop(var, None)
                
                config = ConfigManager.load(env_file_path)
                
                # Verify Neo4j config
                assert config.neo4j.uri == "bolt://localhost:7687"
                assert config.neo4j.username == "neo4j"
                assert config.neo4j.password == "test_password"
                
                # Verify Main LLM config
                assert config.main_llm.model_path == "./models/main.gguf"
                assert config.main_llm.n_ctx == 4096  # default
                assert config.main_llm.n_gpu_layers == -1  # default
                
                # Verify STT config
                assert config.stt.engine == "whisper"
                assert config.stt.whisper_model_path == "./models/whisper.bin"
                assert config.stt.vosk_model_path is None
                
                # Verify SLM config
                assert config.slm.model_path == "./models/slm.gguf"
                assert config.slm.n_ctx == 2048  # default
                
                # Verify storage config
                assert config.storage.pinned_memory_path == "./data/pinned.json"
                assert config.storage.session_state_path == "./data/sessions/"
                assert config.storage.archive_path == "./data/archive/"
                
                # Verify defaults
                assert config.log_level == "INFO"
                assert config.max_conversation_turns == 10
                assert config.memory_budget_tokens == 500
                assert config.cache_size_tier1 == 100
                assert config.gradio_host == "127.0.0.1"
                assert config.gradio_port == 7860
                assert config.gradio_share is False
            
        finally:
            os.unlink(env_file_path)
    
    def test_missing_required_variable_raises_error(self):
        """Test that missing required variables raise ConfigurationError."""
        # Create env file missing NEO4J_PASSWORD
        env_content = """NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
MAIN_LLM_MODEL_PATH=./models/main.gguf
STT_ENGINE=whisper
WHISPER_MODEL_PATH=./models/whisper.bin
SLM_MODEL_PATH=./models/slm.gguf
PINNED_MEMORY_PATH=./data/pinned.json
SESSION_STATE_PATH=./data/sessions/
ARCHIVE_PATH=./data/archive/"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            env_file_path = f.name
        
        try:
            # Clear environment to avoid interference
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ConfigurationError) as exc_info:
                    ConfigManager.load(env_file_path)
                
                assert "NEO4J_PASSWORD" in str(exc_info.value)
            
        finally:
            os.unlink(env_file_path)
    
    def test_invalid_stt_engine_raises_error(self):
        """Test that invalid STT engine raises ConfigurationError."""
        env_content = """NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=test_password
MAIN_LLM_MODEL_PATH=./models/main.gguf
STT_ENGINE=invalid_engine
SLM_MODEL_PATH=./models/slm.gguf
PINNED_MEMORY_PATH=./data/pinned.json
SESSION_STATE_PATH=./data/sessions/
ARCHIVE_PATH=./data/archive/"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            env_file_path = f.name
        
        try:
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ConfigurationError) as exc_info:
                    ConfigManager.load(env_file_path)
                
                assert "STT_ENGINE must be 'whisper' or 'vosk'" in str(exc_info.value)
            
        finally:
            os.unlink(env_file_path)
    
    def test_missing_stt_model_path_raises_error(self):
        """Test that missing STT model path for selected engine raises error."""
        env_content = """NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=test_password
MAIN_LLM_MODEL_PATH=./models/main.gguf
STT_ENGINE=whisper
SLM_MODEL_PATH=./models/slm.gguf
PINNED_MEMORY_PATH=./data/pinned.json
SESSION_STATE_PATH=./data/sessions/
ARCHIVE_PATH=./data/archive/"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            env_file_path = f.name
        
        try:
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ConfigurationError) as exc_info:
                    ConfigManager.load(env_file_path)
                
                assert "WHISPER_MODEL_PATH is required" in str(exc_info.value)
            
        finally:
            os.unlink(env_file_path)
    
    def test_environment_variable_override(self):
        """Test that environment variables override .env file values."""
        env_content = """NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=file_password
MAIN_LLM_MODEL_PATH=./models/main.gguf
STT_ENGINE=whisper
WHISPER_MODEL_PATH=./models/whisper.bin
SLM_MODEL_PATH=./models/slm.gguf
PINNED_MEMORY_PATH=./data/pinned.json
SESSION_STATE_PATH=./data/sessions/
ARCHIVE_PATH=./data/archive/"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            env_file_path = f.name
        
        try:
            # Set environment variable that should override .env file
            # First load with env var set
            with patch.dict(os.environ, {'NEO4J_PASSWORD': 'env_password'}, clear=True):
                # Load other required vars from the file content
                os.environ.update({
                    'NEO4J_URI': 'bolt://localhost:7687',
                    'NEO4J_USERNAME': 'neo4j',
                    'MAIN_LLM_MODEL_PATH': './models/main.gguf',
                    'STT_ENGINE': 'whisper',
                    'WHISPER_MODEL_PATH': './models/whisper.bin',
                    'SLM_MODEL_PATH': './models/slm.gguf',
                    'PINNED_MEMORY_PATH': './data/pinned.json',
                    'SESSION_STATE_PATH': './data/sessions/',
                    'ARCHIVE_PATH': './data/archive/'
                })
                config = ConfigManager.load(env_file_path)
                assert config.neo4j.password == "env_password"  # Should use env var, not file
            
        finally:
            os.unlink(env_file_path)
    
    def test_invalid_integer_value_raises_error(self):
        """Test that invalid integer values raise ConfigurationError."""
        env_content = """NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=test_password
MAIN_LLM_MODEL_PATH=./models/main.gguf
MAIN_LLM_N_CTX=not_an_integer
STT_ENGINE=whisper
WHISPER_MODEL_PATH=./models/whisper.bin
SLM_MODEL_PATH=./models/slm.gguf
PINNED_MEMORY_PATH=./data/pinned.json
SESSION_STATE_PATH=./data/sessions/
ARCHIVE_PATH=./data/archive/"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            env_file_path = f.name
        
        try:
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ConfigurationError) as exc_info:
                    ConfigManager.load(env_file_path)
                
                assert "must be an integer" in str(exc_info.value)
            
        finally:
            os.unlink(env_file_path)
    
    def test_boolean_parsing(self):
        """Test boolean environment variable parsing."""
        env_content = """NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=test_password
MAIN_LLM_MODEL_PATH=./models/main.gguf
STT_ENGINE=whisper
WHISPER_MODEL_PATH=./models/whisper.bin
SLM_MODEL_PATH=./models/slm.gguf
PINNED_MEMORY_PATH=./data/pinned.json
SESSION_STATE_PATH=./data/sessions/
ARCHIVE_PATH=./data/archive/
GRADIO_SHARE=true"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            env_file_path = f.name
        
        try:
            with patch.dict(os.environ, {}, clear=True):
                config = ConfigManager.load(env_file_path)
                assert config.gradio_share is True
            
        finally:
            os.unlink(env_file_path)