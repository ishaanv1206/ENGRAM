"""
Main application entry point for the Cognitive Memory Controller.

This module provides the main entry point for starting the Cognitive Memory Controller
application. It handles configuration loading, component initialization, error handling,
and launches the terminal-based chat interface.

The application follows this startup sequence:
1. Load and validate configuration from .env file
2. Set up logging based on configuration
3. Initialize the CognitivePipeline with all components
4. Start background tasks (reflection loop, decay manager)
5. Launch the terminal chat interface
6. Handle graceful shutdown on exit

All configuration errors are handled with clear error messages to help users
identify and fix configuration issues.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Optional

from .config import ConfigManager, ConfigurationError, SystemConfig
from .pipeline import CognitivePipeline
from .terminal_ui import run_terminal_interface


# Global pipeline instance for cleanup
pipeline: Optional[CognitivePipeline] = None


def setup_logging(config: SystemConfig) -> None:
    """
    Set up logging configuration based on system config.
    
    Args:
        config: System configuration containing log level.
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Set up file handler only (no console output to keep UI clean)
    file_handler = logging.FileHandler(logs_dir / "cognitive_memory.log", mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        handlers=[file_handler]
    )
    
    # Set specific logger levels to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at level: {config.log_level}")


def validate_paths(config: SystemConfig) -> None:
    """
    Validate that required paths exist or can be created.
    
    Args:
        config: System configuration to validate.
        
    Raises:
        ConfigurationError: If paths are invalid or cannot be created.
    """
    logger = logging.getLogger(__name__)
    
    # Check model files exist
    main_llm_path = Path(config.main_llm.model_path)
    if not main_llm_path.exists():
        raise ConfigurationError(
            f"Main LLM model file not found: {config.main_llm.model_path}\n"
            f"Please download the model file or update MAIN_LLM_MODEL_PATH in .env"
        )
    
    slm_path = Path(config.slm.model_path)
    if not slm_path.exists():
        raise ConfigurationError(
            f"SLM model file not found: {config.slm.model_path}\n"
            f"Please download the model file or update SLM_MODEL_PATH in .env"
        )
    
    # Create storage directories if they don't exist
    storage_paths = [
        Path(config.storage.pinned_memory_path).parent,
        Path(config.storage.session_state_path),
        Path(config.storage.archive_path)
    ]
    
    for path in storage_paths:
        try:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Storage directory ready: {path}")
        except Exception as e:
            raise ConfigurationError(f"Cannot create storage directory {path}: {e}")


async def initialize_pipeline(config: SystemConfig) -> CognitivePipeline:
    """
    Initialize the Cognitive Pipeline with all components.
    
    Args:
        config: System configuration.
        
    Returns:
        CognitivePipeline: Initialized pipeline instance.
        
    Raises:
        Exception: If pipeline initialization fails.
    """
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing Cognitive Pipeline...")
        
        # Create pipeline instance
        pipeline = CognitivePipeline(config)
        
        # Start background tasks
        await pipeline.start_background_tasks()
        
        logger.info("Cognitive Pipeline initialized successfully")
        return pipeline
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        raise


async def shutdown_handler(signum: int, frame) -> None:
    """
    Handle graceful shutdown on SIGINT/SIGTERM.
    
    Args:
        signum: Signal number.
        frame: Current stack frame.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    
    global pipeline
    if pipeline:
        try:
            await pipeline.stop_background_tasks()
            logger.info("Pipeline shutdown complete")
        except Exception as e:
            logger.error(f"Error during pipeline shutdown: {e}", exc_info=True)
    
    sys.exit(0)


def main() -> None:
    """
    Main application entry point.
    
    This function orchestrates the entire application startup:
    1. Load configuration
    2. Set up logging
    3. Validate paths and dependencies
    4. Initialize pipeline
    5. Launch terminal interface
    6. Handle errors gracefully
    """
    global pipeline
    
    try:
        # Step 1: Load configuration
        try:
            config = ConfigManager.load()
        except ConfigurationError as e:
            print(f"❌ Configuration Error: {e}")
            sys.exit(1)
        
        # Step 2: Set up logging
        setup_logging(config)
        logger = logging.getLogger(__name__)
        logger.info("=== Cognitive Memory Controller Starting ===")
        
        # Step 3: Validate paths and dependencies
        try:
            validate_paths(config)
            logger.info("Path validation complete")
        except ConfigurationError as e:
            logger.error(f"Path validation failed: {e}")
            print(f"❌ Path Error: {e}")
            sys.exit(1)
        
        # Step 4: Initialize pipeline (async)
        logger.info("Initializing Cognitive Pipeline...")
        
        async def async_main():
            global pipeline
            
            try:
                # Initialize pipeline
                pipeline = await initialize_pipeline(config)
                logger.info("Pipeline initialized successfully")
                
                # Step 5: Launch terminal interface
                # Minimal startup message
                print("\n🧠 Cognitive Memory Controller Ready")
                
                # Set up signal handlers for graceful shutdown
                def signal_handler(signum, frame):
                    logger.info("Shutting down...")
                    raise KeyboardInterrupt
                
                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)
                
                # Run the terminal interface (this blocks until exit)
                await run_terminal_interface(pipeline)
                
                # Cleanup
                logger.info("Terminal interface closed, cleaning up...")
                if pipeline:
                    await pipeline.stop_background_tasks()
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                if pipeline:
                    await pipeline.stop_background_tasks()
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                print(f"❌ Unexpected Error: {e}")
                if pipeline:
                    await pipeline.stop_background_tasks()
                sys.exit(1)
        
        # Run the async main function
        asyncio.run(async_main())
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fatal Error: {e}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()