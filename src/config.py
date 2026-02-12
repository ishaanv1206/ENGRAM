"""
Configuration management for the Cognitive Memory Controller.

This module provides configuration loading and validation using environment variables
and .env files. It defines the SystemConfig dataclass and ConfigManager for secure
credential handling and validation.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing required values."""
    pass


@dataclass
class GraphConfig:
    """Graph database configuration (NetworkX)."""
    storage_path: str = "data/memory_graph.json"


@dataclass
class MainLLMConfig:
    """Main Language Model configuration for response generation."""
    model_path: str
    n_ctx: int
    n_gpu_layers: int


@dataclass
class SLMConfig:
    """Small Language Model configuration for memory analysis."""
    model_path: str
    n_ctx: int
    n_gpu_layers: int
    lora_path: Optional[str] = None  # Path to LoRA adapter GGUF file


@dataclass
class EmbeddingConfig:
    """Embedding model configuration for semantic search."""
    model_path: str
    n_ctx: int
    n_gpu_layers: int


@dataclass
class StorageConfig:
    """Storage paths configuration."""
    pinned_memory_path: str
    session_state_path: str
    archive_path: str


@dataclass
class SystemConfig:
    """Complete system configuration containing all subsystem configs."""
    graph: GraphConfig
    main_llm: MainLLMConfig
    slm: SLMConfig
    embedding: EmbeddingConfig
    storage: StorageConfig
    log_level: str
    max_conversation_turns: int
    memory_budget_tokens: int
    cache_size_tier1: int


class ConfigManager:
    """Manages configuration loading and validation."""
    
    @staticmethod
    def load(env_file: Optional[str] = None) -> SystemConfig:
        """
        Load configuration from environment variables and .env file.
        
        Args:
            env_file: Optional path to .env file. If None, looks for .env in current directory.
            
        Returns:
            SystemConfig: Validated configuration object.
            
        Raises:
            ConfigurationError: If required configuration values are missing or invalid.
        """
        # Load .env file if it exists (but don't override existing env vars)
        if env_file:
            env_path = Path(env_file)
            if env_path.exists():
                load_dotenv(env_path, override=False)
        else:
            env_path = Path('.env')
            if env_path.exists():
                load_dotenv(env_path, override=False)
        
        try:
            # Graph Configuration
            graph_config = GraphConfig(
                storage_path=os.getenv('GRAPH_STORAGE_PATH', 'data/memory_graph_master.json')
            )
            
            # Main LLM Configuration
            main_llm_config = MainLLMConfig(
                model_path=ConfigManager._get_required_env('MAIN_LLM_MODEL_PATH'),
                n_ctx=ConfigManager._get_int_env('MAIN_LLM_N_CTX', 4096),
                n_gpu_layers=ConfigManager._get_int_env('MAIN_LLM_N_GPU_LAYERS', -1)
            )
            
            # SLM Configuration
            slm_config = SLMConfig(
                model_path=ConfigManager._get_required_env('SLM_MODEL_PATH'),
                n_ctx=ConfigManager._get_int_env('SLM_N_CTX', 2048),
                n_gpu_layers=ConfigManager._get_int_env('SLM_N_GPU_LAYERS', -1),
                lora_path=os.getenv('SLM_LORA_PATH'),  # Optional LoRA adapter
            )
            
            # Embedding Configuration
            embedding_config = EmbeddingConfig(
                model_path=ConfigManager._get_required_env('EMBEDDING_MODEL_PATH'),
                n_ctx=ConfigManager._get_int_env('EMBEDDING_N_CTX', 512),
                n_gpu_layers=ConfigManager._get_int_env('EMBEDDING_N_GPU_LAYERS', -1)
            )
            
            # Storage Configuration
            storage_config = StorageConfig(
                pinned_memory_path=ConfigManager._get_required_env('PINNED_MEMORY_PATH'),
                session_state_path=ConfigManager._get_required_env('SESSION_STATE_PATH'),
                archive_path=ConfigManager._get_required_env('ARCHIVE_PATH')
            )
            
            # System Configuration
            log_level = os.getenv('LOG_LEVEL', 'INFO')
            if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                raise ConfigurationError(f"LOG_LEVEL must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL, got: {log_level}")
            
            return SystemConfig(
                graph=graph_config,
                main_llm=main_llm_config,
                slm=slm_config,
                embedding=embedding_config,
                storage=storage_config,
                log_level=log_level,
                max_conversation_turns=ConfigManager._get_int_env('MAX_CONVERSATION_TURNS', 10),
                memory_budget_tokens=ConfigManager._get_int_env('MEMORY_BUDGET_TOKENS', 500),
                cache_size_tier1=ConfigManager._get_int_env('CACHE_SIZE_TIER1', 100)
            )
            
        except KeyError as e:
            raise ConfigurationError(f"Missing required environment variable: {e}")
        except ValueError as e:
            raise ConfigurationError(f"Invalid configuration value: {e}")
    
    @staticmethod
    def _get_required_env(key: str) -> str:
        """
        Get required environment variable.
        
        Args:
            key: Environment variable name.
            
        Returns:
            str: Environment variable value.
            
        Raises:
            ConfigurationError: If environment variable is missing or empty.
        """
        value = os.getenv(key)
        if not value:
            raise ConfigurationError(f"Required environment variable '{key}' is missing or empty")
        return value
    
    @staticmethod
    def _get_int_env(key: str, default: int) -> int:
        """
        Get integer environment variable with default.
        
        Args:
            key: Environment variable name.
            default: Default value if not set.
            
        Returns:
            int: Environment variable value as integer.
            
        Raises:
            ConfigurationError: If value cannot be converted to integer.
        """
        value = os.getenv(key)
        if value is None:
            return default
        
        try:
            return int(value)
        except ValueError:
            raise ConfigurationError(f"Environment variable '{key}' must be an integer, got: {value}")
    
    @staticmethod
    def _get_bool_env(key: str, default: bool) -> bool:
        """
        Get boolean environment variable with default.
        
        Args:
            key: Environment variable name.
            default: Default value if not set.
            
        Returns:
            bool: Environment variable value as boolean.
        """
        value = os.getenv(key)
        if value is None:
            return default
        
        return value.lower() in ('true', '1', 'yes', 'on')