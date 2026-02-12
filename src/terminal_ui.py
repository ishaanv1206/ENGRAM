"""
Terminal-based chat interface for the Cognitive Memory Controller.

This module provides a simple command-line interface for interacting
with the Cognitive Memory Controller using text input/output.
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime
from pathlib import Path

from .pipeline import CognitivePipeline
from .models import ConversationContext, create_conversation_context

logger = logging.getLogger(__name__)


class TerminalUI:
    """
    Terminal-based chat interface for the Cognitive Memory Controller.
    
    Provides a simple REPL (Read-Eval-Print Loop) for text-based conversations.
    """
    
    def __init__(self, pipeline: CognitivePipeline, debug_mode: bool = False):
        """
        Initialize the Terminal UI with a cognitive pipeline.
        
        Args:
            pipeline: The CognitivePipeline instance to use for processing.
            debug_mode: If True, show memory classification info after each message.
        """
        self.pipeline = pipeline
        self.context: Optional[ConversationContext] = None
        self.running = False
        self.debug_mode = debug_mode
        
        # Initialize conversation context
        self._reset_conversation()
        
        # Setup session logger
        self._setup_session_logger()
        
        logger.info("TerminalUI initialized")

    def _setup_session_logger(self):
        """Setup a dedicated logger for this session's conversation history."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"conversation_{timestamp}.txt"
        
        self.session_logger = logging.getLogger(f"session_{timestamp}")
        self.session_logger.setLevel(logging.INFO)
        
        # File handler only
        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.session_logger.addHandler(handler)
        self.session_logger.propagate = False # Don't bubble up to root logger
        
        print(f"\n📝 Conversation logging to: {log_file}")

    async def _process_message(self, message: str) -> str:
        """
        Process a user message through the pipeline.
        
        Args:
            message: User input text.
            
        Returns:
            Assistant's response.
        """
        try:
            # Pipeline now returns (response, metadata)
            response, metadata = await self.pipeline.process_turn(message.strip(), self.context)
            
            # Extract metadata for logging
            extraction_meta = metadata.get('extraction', {})
            retrieval_meta = metadata.get('retrieval', {})
            
            # Log to file with decision details
            self._log_turn(message.strip(), response, extraction_meta, retrieval_meta)
            
            return response
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return "I apologize, but I encountered an error processing your message. Please try again."

    def _log_turn(self, user_text: str, assistant_text: str, extraction=None, retrieval=None):
        """Log a full turn to the session file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.session_logger.info(f"[{timestamp}] USER: {user_text}")
        self.session_logger.info(f"[{timestamp}] ASSISTANT: {assistant_text}")
        
        if extraction:
            cat = extraction.get('category', 'UNKNOWN')
            conf = extraction.get('confidence', 0.0)
            content = extraction.get('content', 'N/A')
            self.session_logger.info(f"[MEMORY] Category: {cat} | Confidence: {conf:.2f} | Content: {content}")
            
        if retrieval:
            intent = retrieval.get('intent', 'UNKNOWN')
            total = retrieval.get('total_memories', 0)
            queried = retrieval.get('graph_queried', False)
            self.session_logger.info(f"[DECISION] Intent: {intent} | Graph Queried: {queried} | Total Memories: {total}")
        
        self.session_logger.info("-" * 40)
        
    def _reset_conversation(self) -> None:
        """Reset the conversation context to start fresh."""
        self.context = create_conversation_context()
        logger.info(f"New conversation started with session_id: {self.context.session_id}")
    
    def _print_help(self) -> None:
        """Print help information."""
        print("\n📖 Available Commands:")
        print("  /help                     - Show this help message")
        print("  /clear                    - Clear conversation history")
        print("  /status                   - Show current session information")
        print("  /pinned                   - Show all pinned memories")
        print("  /graph                    - Generate interactive knowledge graph")
        print("  /quit                     - Exit the application")
        print()
    
    def _print_status(self) -> None:
        """Print current session status."""
        print(f"\n📊 Session Status:")
        print(f"  Session ID: {self.context.session_id}")
        print(f"  Turn Count: {self.context.turn_count}")
        print()
    
    def _print_pinned_memories(self) -> None:
        """Print all pinned memories."""
        pinned = self.pipeline.pinned_mgr.get_all()
        print("\n📌 Pinned Memories:")
        for category, content in pinned.items():
            print(f"  {category.upper()}: {content}")
        print()

    async def run(self) -> None:
        """
        Run the terminal chat interface.
        """
        self.running = True
        print("\n🧠 Cognitive Memory Controller Ready")
        print("Type /help for commands. Type /quit to exit.\n")
        
        try:
            while self.running:
                # Get user input
                try:
                    user_input = (await asyncio.to_thread(input, "User: ")).strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    command = user_input.lower().split()[0]
                    if command in ['/quit', '/exit']:
                        print("Goodbye!")
                        break
                    elif command == '/help':
                        self._print_help()
                        continue
                    elif command == '/clear':
                        self._reset_conversation()
                        print("Conversation cleared.\n")
                        continue
                    elif command == '/status':
                        self._print_status()
                        continue
                    elif command == '/pinned':
                        self._print_pinned_memories()
                        continue
                    elif command == '/graph':
                        print("Generating graph visualization...")
                        path = await self.pipeline.graph_engine.visualize_network("main")
                        if path:
                            print(f"✅ Graph saved to: {path}\n")
                        else:
                            print("❌ Failed to generate graph. Check logs (requires 'pyvis').\n")
                        continue
                    elif command == '/debug':
                        self.debug_mode = not self.debug_mode
                        print(f"Debug mode: {self.debug_mode}")
                        continue
                    else:
                        print(f"Unknown command: {command}")
                        continue
                
                # Process message
                try:
                    response = await self._process_message(user_input)
                    print(f"Assistant: {response}\n")

                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    print("Error processing message. Check logs.\n")
        
        finally:
            self.running = False
            logger.info("Terminal UI stopped")

    def stop(self) -> None:
        self.running = False


async def run_terminal_interface(pipeline: CognitivePipeline) -> None:
    """
    Run the terminal-based chat interface.
    
    Args:
        pipeline: Initialized CognitivePipeline instance.
    """
    ui = TerminalUI(pipeline)
    await ui.run()
