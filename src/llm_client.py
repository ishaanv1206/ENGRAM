"""
Local LLM Client for response generation using llama.cpp.

This module provides the LocalLLMClient class that uses llama.cpp to generate
responses with memory context injection. It supports both synchronous and
streaming response generation, handles model loading errors, and limits
conversation history to the last N turns.
"""

import logging
from typing import List, Optional, AsyncIterator
from llama_cpp import Llama
from src.models import Message, ConversationContext
from src.config import MainLLMConfig


logger = logging.getLogger(__name__)


class LocalLLMClient:
    """
    Local LLM client using llama.cpp for response generation.
    
    This client loads a local LLM model (Llama 3.2 3B uncensored) and generates
    responses with memory context injection. It limits conversation history to
    the last N turns to prevent context overflow.
    """
    
    def __init__(self, config: MainLLMConfig, max_history_turns: int = 10):
        """
        Initialize the Local LLM client.
        
        Args:
            config: Main LLM configuration containing model path and parameters.
            max_history_turns: Maximum number of conversation turns to include in context.
        
        Raises:
            RuntimeError: If model loading fails after retries.
        """
        self.config = config
        self.max_history_turns = max_history_turns
        self.llm: Optional[Llama] = None
        
        # Load model with retries
        self._load_model_with_retries()
    
    def _load_model_with_retries(self, max_retries: int = 3) -> None:
        """
        Load the LLM model with retry logic.
        
        Args:
            max_retries: Maximum number of retry attempts.
            
        Raises:
            RuntimeError: If model loading fails after all retries.
        """
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading Main LLM model from {self.config.model_path} (attempt {attempt + 1}/{max_retries})")
                
                self.llm = Llama(
                    model_path=self.config.model_path,
                    n_ctx=self.config.n_ctx,
                    n_gpu_layers=self.config.n_gpu_layers,
                    verbose=False,
                    flash_attn=True  # Enable Flash Attention
                )
                
                logger.info("Main LLM model loaded successfully")
                return
                
            except Exception as e:
                logger.error(f"Failed to load Main LLM model (attempt {attempt + 1}/{max_retries}): {e}")
                
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed to load Main LLM model after {max_retries} attempts. "
                        f"Please check the model path: {self.config.model_path}"
                    ) from e
    
    def _format_system_message(self, memory_context: str) -> str:
        """
        Format the system message with memory context.
        
        Args:
            memory_context: Formatted memory context from Memory Influence Layer.
            
        Returns:
            str: Formatted system message.
        """
        return f"""{memory_context}

You are a helpful AI assistant. Use the provided memory context to inform your responses, 
but respond naturally without explicitly mentioning the memory system. Be concise and direct."""
    
    def _get_recent_history(self, conversation_history: List[Message]) -> List[Message]:
        """
        Get the most recent N turns from conversation history.
        
        Args:
            conversation_history: Full conversation history.
            
        Returns:
            List[Message]: Last N turns (up to max_history_turns * 2 messages).
        """
        # Each turn has 2 messages (user + assistant), so we take last N*2 messages
        max_messages = self.max_history_turns * 2
        return conversation_history[-max_messages:] if len(conversation_history) > max_messages else conversation_history
    
    def _build_messages(
        self,
        query: str,
        memory_context: str,
        conversation_history: List[Message]
    ) -> List[dict]:
        """
        Build the messages list for the LLM.
        
        Args:
            query: Current user query.
            memory_context: Formatted memory context.
            conversation_history: Full conversation history.
            
        Returns:
            List[dict]: Messages formatted for llama.cpp.
        """
        messages = []
        
        # Add system message with memory context
        system_message = self._format_system_message(memory_context)
        messages.append({"role": "system", "content": system_message})
        
        # Add recent conversation history (limited to last N turns)
        recent_history = self._get_recent_history(conversation_history)
        for msg in recent_history:
            # Ensure message is properly formatted as dict
            if isinstance(msg, dict):
                messages.append({"role": msg.get("role", "user"), "content": str(msg.get("content", ""))})
            else:
                # msg is a Message object
                messages.append({"role": str(msg.role), "content": str(msg.content)})
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    async def generate(
        self,
        query: str,
        memory_context: str,
        conversation_history: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate a response with memory context injection.
        
        Args:
            query: User query to respond to.
            memory_context: Formatted memory context from Memory Influence Layer.
            conversation_history: Full conversation history.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            
        Returns:
            str: Generated response.
            
        Raises:
            RuntimeError: If generation fails after retries.
        """
        if not self.llm:
            raise RuntimeError("LLM model not loaded")
        
        messages = self._build_messages(query, memory_context, conversation_history)
        
        try:
            logger.debug(f"Generating response for query: {query[:50]}...")
            
            import asyncio
            loop = asyncio.get_running_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            )
            
            content = response['choices'][0]['message']['content']
            logger.debug(f"Generated response: {content[:50]}...")
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            
            # Retry once with reduced context
            try:
                logger.info("Retrying with reduced context...")
                
                # Reduce history to last 5 turns
                reduced_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
                messages = self._build_messages(query, memory_context, reduced_history)
                
                response = self.llm.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens // 2  # Reduce max tokens
                )
                
                content = response['choices'][0]['message']['content']
                logger.info("Successfully generated response with reduced context")
                
                return content
                
            except Exception as retry_error:
                logger.error(f"Failed to generate response after retry: {retry_error}")
                raise RuntimeError(
                    f"Failed to generate response: {e}. Retry also failed: {retry_error}"
                ) from e
    
    def generate_stream(
        self,
        query: str,
        memory_context: str,
        conversation_history: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response with memory context injection.
        
        Args:
            query: User query to respond to.
            memory_context: Formatted memory context from Memory Influence Layer.
            conversation_history: Full conversation history.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            
        Yields:
            str: Response tokens as they are generated.
            
        Raises:
            RuntimeError: If generation fails.
        """
        if not self.llm:
            raise RuntimeError("LLM model not loaded")
        
        messages = self._build_messages(query, memory_context, conversation_history)
        
        try:
            logger.debug(f"Generating streaming response for query: {query[:50]}...")
            
            stream = self.llm.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        yield delta['content']
            
            logger.debug("Streaming response completed")
            
        except Exception as e:
            logger.error(f"Failed to generate streaming response: {e}")
            raise RuntimeError(f"Failed to generate streaming response: {e}") from e
    
    def __del__(self):
        """Clean up resources when the client is destroyed."""
        if self.llm:
            logger.info("Cleaning up Main LLM model")
            del self.llm
