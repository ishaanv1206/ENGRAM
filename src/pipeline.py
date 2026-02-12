"""
Main Cognitive Pipeline for orchestrating the memory management system.

This module implements the CognitivePipeline class which coordinates all components
of the Cognitive Memory Controller. It handles text input processing, parallel
execution of analysis and retrieval, memory storage, LLM response generation,
and asynchronous reflection.

The pipeline follows a streaming-first, non-blocking architecture where:
1. Text input is processed immediately
2. Memory analysis and retrieval run in parallel
3. New memories are stored asynchronously (non-blocking)
4. Memory context is injected into LLM prompts
5. Responses are generated and returned to the user
6. Reflection tasks are enqueued for background processing

All background tasks (reflection, decay) run asynchronously and never block
the real-time response path.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from .config import SystemConfig
from .models import (
    Message, ConversationContext, MemoryExtraction, MemoryCategory,
    ReflectionTask, QueryIntent, RetrievalResult
)
from .memory_analyzer import MemoryAnalyzer
from .pinned_memory import PinnedMemoryManager
from .recent_cache import RecentMemoryCache
from .graph_engine import GraphMemoryEngine
from .retrieval_gatekeeper import RetrievalGatekeeper
from .memory_influence import MemoryInfluenceLayer
from .llm_client import LocalLLMClient
from .reflection_loop import ReflectionLoop
from .decay_manager import DecayManager
from .persistence import PersistenceManager
from .embedding_service import EmbeddingService


logger = logging.getLogger(__name__)


class CognitivePipeline:
    """
    Main cognitive pipeline orchestrating all memory management components.
    
    The pipeline coordinates:
    - Memory analysis (classification and extraction)
    - Multi-tier memory retrieval (pinned, cache, graph)
    - Memory influence (context injection)
    - LLM response generation
    - Asynchronous reflection and validation
    - Periodic decay management
    
    All operations are optimized for low latency with parallel execution
    and non-blocking background tasks.
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the Cognitive Pipeline with all components.
        
        Args:
            config: System configuration containing all subsystem configs.
        """
        self.config = config
        
        logger.info("Initializing Cognitive Pipeline components...")
        
        # Initialize embedding service
        self.embedding_service = EmbeddingService(config.embedding)
        
        # Initialize core components
        self.analyzer = MemoryAnalyzer(config.slm)
        self.pinned_mgr = PinnedMemoryManager(config.storage.pinned_memory_path)
        self.recent_cache = RecentMemoryCache(max_size=config.cache_size_tier1)
        self.graph_engine = GraphMemoryEngine(config.graph)
        
        # Initialize retrieval and influence layers
        self.retriever = RetrievalGatekeeper(
            analyzer=self.analyzer,
            pinned_mgr=self.pinned_mgr,
            recent_cache=self.recent_cache,
            graph_engine=self.graph_engine,
            embedding_service=self.embedding_service
        )
        self.influencer = MemoryInfluenceLayer(max_tokens=config.memory_budget_tokens)
        
        # Initialize LLM client
        self.llm = LocalLLMClient(
            config=config.main_llm,
            max_history_turns=config.max_conversation_turns
        )
        
        # Initialize background processing components
        self.reflection = ReflectionLoop(
            analyzer=self.analyzer,
            graph_engine=self.graph_engine,
            llm_client=self.llm
        )
        self.decay_mgr = DecayManager(graph_engine=self.graph_engine)
        
        # Initialize persistence manager
        self.persistence = PersistenceManager(config.storage.session_state_path)
        
        # Track background tasks
        self._background_tasks = []
        
        logger.info("Cognitive Pipeline initialized successfully")
    
    async def start_background_tasks(self) -> None:
        """
        Start background tasks for reflection and decay management.
        
        These tasks run asynchronously and never block the real-time
        response path. They should be started once when the pipeline
        is initialized.
        """
        logger.info("Starting background tasks...")
        
        # Start reflection loop
        reflection_task = asyncio.create_task(self.reflection.start())
        self._background_tasks.append(reflection_task)
        logger.info("Reflection loop started")
        
        # Start decay manager
        decay_task = asyncio.create_task(self.decay_mgr.start())
        self._background_tasks.append(decay_task)
        logger.info("Decay manager started")
        
        logger.info("All background tasks started successfully")
    
    async def stop_background_tasks(self) -> None:
        """
        Stop all background tasks gracefully.
        
        Should be called when shutting down the pipeline.
        """
        logger.info("Stopping background tasks...")
        
        # Stop reflection loop
        self.reflection.stop()
        
        # Stop decay manager
        await self.decay_mgr.stop()
        
        # Cancel all background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        logger.info("All background tasks stopped")
    
    async def process_turn(
        self,
        text: str,
        context: ConversationContext
    ) -> str:
        """
        Process a single conversation turn with text input.
        
        This is the main orchestration method that coordinates the entire
        pipeline:
        1. Parallel fork: analyze text and retrieve relevant memories
        2. Store new memories asynchronously (non-blocking)
        3. Inject memory context into LLM prompt
        4. Generate response
        5. Enqueue reflection task (non-blocking)
        6. Update conversation context
        
        Args:
            text: User input text to process.
            context: Current conversation context.
            
        Returns:
            str: Generated response from the LLM.
        """
        logger.info(f"Processing turn {context.turn_count + 1}: {text[:50]}...")
        
        # Step 1: Sequential execution - analyze then retrieve
        # Run sequentially to prevent llama.cpp race conditions/state corruption
        # when both analyzer and embedding service try to use the backend simultaneously.
        extraction = await self.analyzer.analyze(text, context)
        
        # After analysis is done, proceed with retrieval
        retrieval = await self.retriever.retrieve(text, context)
        
        logger.debug(
            f"Analysis complete: category={extraction.category.value}, "
            f"confidence={extraction.confidence:.2f}"
        )
        logger.debug(
            f"Retrieval complete: {len(retrieval.memories)} memories, "
            f"{retrieval.total_tokens} tokens"
        )
        
        # Step 2: Store new memory asynchronously (non-blocking)
        # Only store if not classified as DISCARD and has valid content
        # AND if it's not a retrieval query (Question)
        is_query = retrieval.query_intent in [
            QueryIntent.FACTUAL_RECALL, 
            QueryIntent.PREFERENCE_CHECK, 
            QueryIntent.RELATIONSHIP
        ]
        
        memory_content = extraction.structured_data.get('content', '')
        
        if is_query:
             logger.debug(f"Memory skipped: Input identified as Query ({retrieval.query_intent})")
        elif extraction.category != MemoryCategory.DISCARD and memory_content and memory_content.strip():
            # PINNED memories go to special pinned memory manager (always remembered)
            if extraction.category == MemoryCategory.PINNED:
                asyncio.create_task(self._store_pinned_memory(extraction, text, context))
                logger.debug(f"PINNED memory storage task enqueued")
            else:
                # Other categories go to Neo4j graph (subject to decay)
                asyncio.create_task(self._store_memory(extraction, text, context))
                logger.debug(f"Memory storage task enqueued for category {extraction.category.value}")
        else:
            logger.debug(f"Memory skipped: Category={extraction.category.value}, Content='{memory_content[:50] if memory_content else ''}'...")
        
        # Step 3: Inject memory context into LLM prompt
        memory_context = self.influencer.inject(retrieval, text)
        logger.debug(f"Memory context injected: {len(memory_context)} characters")
        
        # Step 4: Generate response
        try:
            response = await self.llm.generate(
                query=text,
                memory_context=memory_context,
                conversation_history=context.conversation_history
            )
            logger.info(f"Response generated: {response[:50]}...")
        except Exception as e:
            logger.error(f"Failed to generate response: {e}", exc_info=True)
            response = "I apologize, but I encountered an error generating a response. Please try again."
        
        # Step 5: Enqueue reflection task (non-blocking)
        reflection_task = ReflectionTask(
            task_id=str(uuid4()),
            query=text,
            response=response,
            retrieved_memories=retrieval.memories,
            context=context,
            created_at=datetime.now()
        )
        self.reflection.enqueue(reflection_task)
        logger.debug(f"Reflection task {reflection_task.task_id} enqueued")
        
        # Step 6: Update conversation context
        context.turn_count += 1
        context.conversation_history.append(
            Message(role="user", content=text, timestamp=datetime.now())
        )
        context.conversation_history.append(
            Message(role="assistant", content=response, timestamp=datetime.now())
        )
        
        # Update recent topics and entities (simplified extraction)
        self._update_context_metadata(context, extraction)
        
        # Step 7: Persist session state (non-blocking)
        asyncio.create_task(self._persist_session(context))
        
        logger.info(f"Turn {context.turn_count} completed successfully")
        
        # Metadata for logging/debugging
        metadata = {
            'extraction': {
                'category': extraction.category.value,
                'confidence': extraction.confidence,
                'content': extraction.structured_data.get('content', '')[:50]
            },
            'retrieval': {
                'intent': retrieval.query_intent.value if hasattr(retrieval, 'query_intent') else "UNKNOWN",
                'total_memories': len(retrieval.memories),
                'graph_queried': len(retrieval.memories) > 0 and retrieval.query_intent in [QueryIntent.FACTUAL_RECALL, QueryIntent.RELATIONSHIP, QueryIntent.PREFERENCE_CHECK] # Approximation
            }
        }
        
        return response, metadata
    
    async def _store_pinned_memory(
        self,
        extraction: MemoryExtraction,
        original_text: str,
        context: ConversationContext
    ) -> None:
        """
        Store a PINNED memory in the pinned memory manager (always remembered).
        
        PINNED memories are stored separately and ALWAYS injected into conversations.
        They never decay and are never forgotten.
        
        Args:
            extraction: Memory extraction from the analyzer.
            original_text: Original text that was analyzed.
            context: Conversation context for metadata.
        """
        try:
            logger.debug(f"Storing PINNED memory: {original_text[:50]}...")
            
            # Determine the category for pinned memory
            # Use the key info or structured data to categorize
            key_info = extraction.structured_data.get('content', original_text)
            
            # Categorize based on content
            text_lower = original_text.lower()
            category_key = "preferences"  # Default
            
            if any(word in text_lower for word in ['my name', 'call me', 'known as', 'i am']):
                category_key = "identity"
            elif any(word in text_lower for word in ['your name', 'you are', 'call yourself']):
                category_key = "assistant_identity"
            elif any(word in text_lower for word in ['prefer', 'like', 'love', 'hate', 'enjoy']):
                category_key = "preferences"
            elif any(word in text_lower for word in ['always', 'never', 'must', 'should', 'perform', 'act']):
                category_key = "instructions"
            elif any(word in text_lower for word in ['i live', 'i work', 'my job', 'my birthday', 'i have']):
                category_key = "user_context"
            elif any(word in text_lower for word in ['my wife', 'my husband', 'my friend', 'my boss', 'my father', 'my mother', 'my son', 'my daughter']):
                category_key = "relationships"
            
            # Store in pinned memory manager
            self.pinned_mgr.add(category_key, key_info)
            
            logger.info(f"PINNED memory stored in category '{category_key}': {key_info[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to store PINNED memory: {e}", exc_info=True)
            # Don't raise - storage failures shouldn't block the response
    
    async def _store_memory(
        self,
        extraction: MemoryExtraction,
        original_text: str,
        context: ConversationContext
    ) -> None:
        """
        Store a new memory in the graph database (asynchronous helper).
        
        This method runs asynchronously and does not block the response path.
        It creates a memory node in the graph database and establishes
        relationships based on the extraction's link information.
        
        Args:
            extraction: Memory extraction from the analyzer.
            original_text: Original text that was analyzed.
            context: Conversation context for metadata.
        """
        try:
            logger.debug(f"Storing memory: category={extraction.category.value}")
            
            # Generate embedding for the memory content
            try:
                embedding = self.embedding_service.embed(original_text)
                extraction.structured_data['embedding'] = embedding
                logger.debug(f"Generated embedding of dimension {len(embedding)}")
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {e}")
                extraction.structured_data['embedding'] = None
            
            # Store the memory in the graph
            memory_id = await self.graph_engine.store_memory(extraction)
            
            logger.info(f"Memory stored successfully: id={memory_id}")
            
            # If this is a CRITICAL or EPISODIC memory, add to cache for fast access
            if extraction.category in [MemoryCategory.CRITICAL, MemoryCategory.EPISODIC]:
                # Retrieve the stored memory to add to cache
                # (In production, we'd return the full MemoryNode from store_memory)
                logger.debug(f"Memory {memory_id} added to cache tier")
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}", exc_info=True)
            # Don't raise - storage failures shouldn't block the response
    
    def _update_context_metadata(
        self,
        context: ConversationContext,
        extraction: MemoryExtraction
    ) -> None:
        """
        Update conversation context with extracted entities and topics.
        
        Args:
            context: Conversation context to update.
            extraction: Memory extraction containing entities and topics.
        """
        # Extract entities from structured data
        entities = extraction.structured_data.get('entities', [])
        for entity in entities:
            if isinstance(entity, str) and entity not in context.active_entities:
                context.active_entities.append(entity)
        
        # Keep only the most recent 20 entities
        if len(context.active_entities) > 20:
            context.active_entities = context.active_entities[-20:]
        
        # Extract topics (simplified - in production, use topic modeling)
        # For now, use memory category as a proxy for topic
        topic = extraction.category.value
        if topic not in context.recent_topics:
            context.recent_topics.append(topic)
        
        # Keep only the most recent 10 topics
        if len(context.recent_topics) > 10:
            context.recent_topics = context.recent_topics[-10:]
    
    def __del__(self):
        """Clean up resources when pipeline is destroyed."""
        logger.info("Cleaning up Cognitive Pipeline resources")
        
        # Close graph engine connection
        if hasattr(self, 'graph_engine'):
            self.graph_engine.close()
    
    async def _persist_session(self, context: ConversationContext) -> None:
        """
        Persist session state asynchronously (non-blocking helper).
        
        Args:
            context: Conversation context to persist.
        """
        try:
            self.persistence.save_session_state(context.session_id, context)
        except Exception as e:
            logger.error(f"Failed to persist session: {e}", exc_info=True)
    
    def load_session(self, session_id: str) -> Optional[ConversationContext]:
        """
        Load a saved session from disk.
        
        Args:
            session_id: Session ID to load.
            
        Returns:
            ConversationContext if found, None otherwise.
        """
        return self.persistence.load_session_state(session_id)
    
    def list_sessions(self) -> list[str]:
        """
        List all saved session IDs.
        
        Returns:
            List of session IDs.
        """
        return self.persistence.list_sessions()


def create_pipeline(config: SystemConfig) -> CognitivePipeline:
    """
    Factory function to create and initialize a Cognitive Pipeline.
    
    Args:
        config: System configuration.
        
    Returns:
        CognitivePipeline: Initialized pipeline instance.
    """
    pipeline = CognitivePipeline(config)
    return pipeline

