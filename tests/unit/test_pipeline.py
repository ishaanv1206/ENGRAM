"""
Unit tests for the Cognitive Pipeline.

Tests the main pipeline orchestration, component initialization,
and turn processing logic.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.pipeline import CognitivePipeline, create_pipeline
from src.config import (
    SystemConfig, Neo4jConfig, MainLLMConfig, STTConfig,
    SLMConfig, StorageConfig
)
from src.models import (
    ConversationContext, MemoryExtraction, MemoryCategory,
    DecayPolicy, RetrievalResult, Message
)


@pytest.fixture
def mock_config():
    """Create a mock system configuration for testing."""
    return SystemConfig(
        neo4j=Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password"
        ),
        main_llm=MainLLMConfig(
            model_path="test_model.gguf",
            n_ctx=4096,
            n_gpu_layers=-1
        ),
        stt=STTConfig(
            engine="whisper",
            whisper_model_path="test_whisper.bin",
            vosk_model_path=None
        ),
        slm=SLMConfig(
            model_path="test_slm.gguf",
            n_ctx=2048,
            n_gpu_layers=-1
        ),
        storage=StorageConfig(
            pinned_memory_path="./data/test_pinned.json",
            session_state_path="./data/test_session.json",
            archive_path="./data/test_archive"
        ),
        log_level="INFO",
        max_conversation_turns=10,
        memory_budget_tokens=500,
        cache_size_tier1=100,
        gradio_host="127.0.0.1",
        gradio_port=7860,
        gradio_share=False
    )


@pytest.fixture
def mock_conversation_context():
    """Create a mock conversation context for testing."""
    return ConversationContext(
        session_id="test-session-123",
        turn_count=0,
        recent_topics=[],
        active_entities=[],
        conversation_history=[]
    )


@pytest.fixture
def mock_memory_extraction():
    """Create a mock memory extraction for testing."""
    return MemoryExtraction(
        category=MemoryCategory.EPISODIC,
        structured_data={
            'preferences': [],
            'facts': ['User mentioned testing'],
            'entities': ['testing'],
            'relationships': [],
            'commitments': [],
            'constraints': []
        },
        confidence=0.8,
        stability=0.7,
        decay_policy=DecayPolicy.MEDIUM,
        links=[],
        timestamp=datetime.now()
    )


@pytest.fixture
def mock_retrieval_result():
    """Create a mock retrieval result for testing."""
    return RetrievalResult(
        pinned={'language': 'English', 'style': 'Professional'},
        memories=[],
        total_tokens=50,
        retrieval_time_ms=25.0,
        query_intent=None
    )


class TestCognitivePipeline:
    """Test suite for CognitivePipeline class."""
    
    @patch('src.pipeline.MemoryAnalyzer')
    @patch('src.pipeline.PinnedMemoryManager')
    @patch('src.pipeline.RecentMemoryCache')
    @patch('src.pipeline.GraphMemoryEngine')
    @patch('src.pipeline.RetrievalGatekeeper')
    @patch('src.pipeline.MemoryInfluenceLayer')
    @patch('src.pipeline.LocalLLMClient')
    @patch('src.pipeline.ReflectionLoop')
    @patch('src.pipeline.DecayManager')
    def test_pipeline_initialization(
        self,
        mock_decay,
        mock_reflection,
        mock_llm,
        mock_influence,
        mock_retriever,
        mock_graph,
        mock_cache,
        mock_pinned,
        mock_analyzer,
        mock_config
    ):
        """Test that pipeline initializes all components correctly."""
        # Create pipeline
        pipeline = CognitivePipeline(mock_config)
        
        # Verify all components were initialized
        assert pipeline.analyzer is not None
        assert pipeline.pinned_mgr is not None
        assert pipeline.recent_cache is not None
        assert pipeline.graph_engine is not None
        assert pipeline.retriever is not None
        assert pipeline.influencer is not None
        assert pipeline.llm is not None
        assert pipeline.reflection is not None
        assert pipeline.decay_mgr is not None
        
        # Verify configuration was stored
        assert pipeline.config == mock_config
    
    @patch('src.pipeline.MemoryAnalyzer')
    @patch('src.pipeline.PinnedMemoryManager')
    @patch('src.pipeline.RecentMemoryCache')
    @patch('src.pipeline.GraphMemoryEngine')
    @patch('src.pipeline.RetrievalGatekeeper')
    @patch('src.pipeline.MemoryInfluenceLayer')
    @patch('src.pipeline.LocalLLMClient')
    @patch('src.pipeline.ReflectionLoop')
    @patch('src.pipeline.DecayManager')
    @pytest.mark.asyncio
    async def test_start_background_tasks(
        self,
        mock_decay,
        mock_reflection,
        mock_llm,
        mock_influence,
        mock_retriever,
        mock_graph,
        mock_cache,
        mock_pinned,
        mock_analyzer,
        mock_config
    ):
        """Test that background tasks are started correctly."""
        # Create pipeline
        pipeline = CognitivePipeline(mock_config)
        
        # Mock the start methods
        pipeline.reflection.start = AsyncMock()
        pipeline.decay_mgr.start = AsyncMock()
        
        # Start background tasks
        await pipeline.start_background_tasks()
        
        # Verify tasks were created
        assert len(pipeline._background_tasks) == 2
    
    @patch('src.pipeline.MemoryAnalyzer')
    @patch('src.pipeline.PinnedMemoryManager')
    @patch('src.pipeline.RecentMemoryCache')
    @patch('src.pipeline.GraphMemoryEngine')
    @patch('src.pipeline.RetrievalGatekeeper')
    @patch('src.pipeline.MemoryInfluenceLayer')
    @patch('src.pipeline.LocalLLMClient')
    @patch('src.pipeline.ReflectionLoop')
    @patch('src.pipeline.DecayManager')
    @pytest.mark.asyncio
    async def test_process_turn_basic_flow(
        self,
        mock_decay,
        mock_reflection,
        mock_llm,
        mock_influence,
        mock_retriever,
        mock_graph,
        mock_cache,
        mock_pinned,
        mock_analyzer,
        mock_config,
        mock_conversation_context,
        mock_memory_extraction,
        mock_retrieval_result
    ):
        """Test basic turn processing flow."""
        # Create pipeline
        pipeline = CognitivePipeline(mock_config)
        
        # Mock component methods
        pipeline.analyzer.analyze = AsyncMock(return_value=mock_memory_extraction)
        pipeline.retriever.retrieve = AsyncMock(return_value=mock_retrieval_result)
        pipeline.influencer.inject = Mock(return_value="<memory_context>Test context</memory_context>")
        pipeline.llm.generate = Mock(return_value="Test response")
        pipeline.reflection.enqueue = Mock()
        pipeline.graph_engine.store_memory = AsyncMock(return_value="memory-123")
        
        # Process a turn
        text = "Hello, this is a test"
        response = await pipeline.process_turn(text, mock_conversation_context)
        
        # Verify response was generated
        assert response == "Test response"
        
        # Verify analyzer was called
        pipeline.analyzer.analyze.assert_called_once_with(text, mock_conversation_context)
        
        # Verify retriever was called
        pipeline.retriever.retrieve.assert_called_once_with(text, mock_conversation_context)
        
        # Verify influencer was called
        pipeline.influencer.inject.assert_called_once()
        
        # Verify LLM was called
        pipeline.llm.generate.assert_called_once()
        
        # Verify reflection was enqueued
        pipeline.reflection.enqueue.assert_called_once()
        
        # Verify context was updated
        assert mock_conversation_context.turn_count == 1
        assert len(mock_conversation_context.conversation_history) == 2
        assert mock_conversation_context.conversation_history[0].role == "user"
        assert mock_conversation_context.conversation_history[0].content == text
        assert mock_conversation_context.conversation_history[1].role == "assistant"
        assert mock_conversation_context.conversation_history[1].content == response
    
    @patch('src.pipeline.MemoryAnalyzer')
    @patch('src.pipeline.PinnedMemoryManager')
    @patch('src.pipeline.RecentMemoryCache')
    @patch('src.pipeline.GraphMemoryEngine')
    @patch('src.pipeline.RetrievalGatekeeper')
    @patch('src.pipeline.MemoryInfluenceLayer')
    @patch('src.pipeline.LocalLLMClient')
    @patch('src.pipeline.ReflectionLoop')
    @patch('src.pipeline.DecayManager')
    @pytest.mark.asyncio
    async def test_process_turn_discard_category(
        self,
        mock_decay,
        mock_reflection,
        mock_llm,
        mock_influence,
        mock_retriever,
        mock_graph,
        mock_cache,
        mock_pinned,
        mock_analyzer,
        mock_config,
        mock_conversation_context,
        mock_retrieval_result
    ):
        """Test that DISCARD category memories are not stored."""
        # Create pipeline
        pipeline = CognitivePipeline(mock_config)
        
        # Create DISCARD extraction
        discard_extraction = MemoryExtraction(
            category=MemoryCategory.DISCARD,
            structured_data={},
            confidence=0.1,
            stability=0.1,
            decay_policy=DecayPolicy.FAST,
            links=[],
            timestamp=datetime.now()
        )
        
        # Mock component methods
        pipeline.analyzer.analyze = AsyncMock(return_value=discard_extraction)
        pipeline.retriever.retrieve = AsyncMock(return_value=mock_retrieval_result)
        pipeline.influencer.inject = Mock(return_value="<memory_context>Test context</memory_context>")
        pipeline.llm.generate = Mock(return_value="Test response")
        pipeline.reflection.enqueue = Mock()
        pipeline.graph_engine.store_memory = AsyncMock(return_value="memory-123")
        
        # Process a turn
        text = "Hello"
        response = await pipeline.process_turn(text, mock_conversation_context)
        
        # Verify response was generated
        assert response == "Test response"
        
        # Give async tasks time to run (if any were created)
        await asyncio.sleep(0.1)
        
        # Verify store_memory was NOT called (DISCARD category)
        pipeline.graph_engine.store_memory.assert_not_called()
    
    @patch('src.pipeline.MemoryAnalyzer')
    @patch('src.pipeline.PinnedMemoryManager')
    @patch('src.pipeline.RecentMemoryCache')
    @patch('src.pipeline.GraphMemoryEngine')
    @patch('src.pipeline.RetrievalGatekeeper')
    @patch('src.pipeline.MemoryInfluenceLayer')
    @patch('src.pipeline.LocalLLMClient')
    @patch('src.pipeline.ReflectionLoop')
    @patch('src.pipeline.DecayManager')
    @pytest.mark.asyncio
    async def test_process_turn_parallel_execution(
        self,
        mock_decay,
        mock_reflection,
        mock_llm,
        mock_influence,
        mock_retriever,
        mock_graph,
        mock_cache,
        mock_pinned,
        mock_analyzer,
        mock_config,
        mock_conversation_context,
        mock_memory_extraction,
        mock_retrieval_result
    ):
        """Test that analysis and retrieval run in parallel."""
        # Create pipeline
        pipeline = CognitivePipeline(mock_config)
        
        # Track call order
        call_order = []
        
        async def mock_analyze(*args, **kwargs):
            call_order.append('analyze_start')
            await asyncio.sleep(0.05)  # Simulate work
            call_order.append('analyze_end')
            return mock_memory_extraction
        
        async def mock_retrieve(*args, **kwargs):
            call_order.append('retrieve_start')
            await asyncio.sleep(0.05)  # Simulate work
            call_order.append('retrieve_end')
            return mock_retrieval_result
        
        # Mock component methods
        pipeline.analyzer.analyze = mock_analyze
        pipeline.retriever.retrieve = mock_retrieve
        pipeline.influencer.inject = Mock(return_value="<memory_context>Test context</memory_context>")
        pipeline.llm.generate = Mock(return_value="Test response")
        pipeline.reflection.enqueue = Mock()
        pipeline.graph_engine.store_memory = AsyncMock(return_value="memory-123")
        
        # Process a turn
        text = "Test parallel execution"
        response = await pipeline.process_turn(text, mock_conversation_context)
        
        # Verify both operations started before either finished (parallel execution)
        assert 'analyze_start' in call_order
        assert 'retrieve_start' in call_order
        
        # Both should start before either finishes
        analyze_start_idx = call_order.index('analyze_start')
        retrieve_start_idx = call_order.index('retrieve_start')
        analyze_end_idx = call_order.index('analyze_end')
        retrieve_end_idx = call_order.index('retrieve_end')
        
        # Both should start before either finishes (indicating parallel execution)
        assert min(analyze_start_idx, retrieve_start_idx) < min(analyze_end_idx, retrieve_end_idx)
    
    @patch('src.pipeline.MemoryAnalyzer')
    @patch('src.pipeline.PinnedMemoryManager')
    @patch('src.pipeline.RecentMemoryCache')
    @patch('src.pipeline.GraphMemoryEngine')
    @patch('src.pipeline.RetrievalGatekeeper')
    @patch('src.pipeline.MemoryInfluenceLayer')
    @patch('src.pipeline.LocalLLMClient')
    @patch('src.pipeline.ReflectionLoop')
    @patch('src.pipeline.DecayManager')
    @pytest.mark.asyncio
    async def test_process_turn_llm_error_handling(
        self,
        mock_decay,
        mock_reflection,
        mock_llm,
        mock_influence,
        mock_retriever,
        mock_graph,
        mock_cache,
        mock_pinned,
        mock_analyzer,
        mock_config,
        mock_conversation_context,
        mock_memory_extraction,
        mock_retrieval_result
    ):
        """Test that LLM errors are handled gracefully."""
        # Create pipeline
        pipeline = CognitivePipeline(mock_config)
        
        # Mock component methods
        pipeline.analyzer.analyze = AsyncMock(return_value=mock_memory_extraction)
        pipeline.retriever.retrieve = AsyncMock(return_value=mock_retrieval_result)
        pipeline.influencer.inject = Mock(return_value="<memory_context>Test context</memory_context>")
        pipeline.llm.generate = Mock(side_effect=RuntimeError("LLM error"))
        pipeline.reflection.enqueue = Mock()
        pipeline.graph_engine.store_memory = AsyncMock(return_value="memory-123")
        
        # Process a turn
        text = "Test error handling"
        response = await pipeline.process_turn(text, mock_conversation_context)
        
        # Verify error response was returned
        assert "error" in response.lower() or "apologize" in response.lower()
        
        # Verify context was still updated
        assert mock_conversation_context.turn_count == 1
    
    @patch('src.pipeline.MemoryAnalyzer')
    @patch('src.pipeline.PinnedMemoryManager')
    @patch('src.pipeline.RecentMemoryCache')
    @patch('src.pipeline.GraphMemoryEngine')
    @patch('src.pipeline.RetrievalGatekeeper')
    @patch('src.pipeline.MemoryInfluenceLayer')
    @patch('src.pipeline.LocalLLMClient')
    @patch('src.pipeline.ReflectionLoop')
    @patch('src.pipeline.DecayManager')
    def test_create_pipeline_factory(
        self,
        mock_decay,
        mock_reflection,
        mock_llm,
        mock_influence,
        mock_retriever,
        mock_graph,
        mock_cache,
        mock_pinned,
        mock_analyzer,
        mock_config
    ):
        """Test the create_pipeline factory function."""
        pipeline = create_pipeline(mock_config)
        
        assert isinstance(pipeline, CognitivePipeline)
        assert pipeline.config == mock_config


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
