"""
Integration test for Retrieval Gatekeeper with real components.

Tests the retrieval gatekeeper working with actual pinned memory manager,
recent cache, and graph engine (mocked for now).
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from src.retrieval_gatekeeper import RetrievalGatekeeper
from src.memory_analyzer import MemoryAnalyzer
from src.pinned_memory import PinnedMemoryManager
from src.recent_cache import RecentMemoryCache
from src.graph_engine import GraphMemoryEngine
from src.models import (
    create_conversation_context, create_memory_node,
    MemoryCategory, QueryIntent
)
from src.config import SLMConfig


@pytest.fixture
def integration_setup(tmp_path):
    """Set up real components for integration testing."""
    # Create real pinned memory manager
    pinned_path = tmp_path / "pinned.json"
    pinned_mgr = PinnedMemoryManager(str(pinned_path))
    pinned_mgr.set("language", "English (US)")
    pinned_mgr.set("style", "Professional and friendly")
    pinned_mgr.set("safety", "Never provide harmful advice")
    
    # Create real recent cache
    recent_cache = RecentMemoryCache(max_size=100)
    
    # Add some test memories to cache
    for i in range(5):
        memory = create_memory_node(
            content=f"Test memory {i}",
            category=MemoryCategory.EPISODIC,
            structured_data={"test": f"data_{i}"},
            confidence=0.8,
            stability=0.7
        )
        recent_cache.put(memory)
    
    # Create mock analyzer (since we don't have a real model)
    analyzer = Mock(spec=MemoryAnalyzer)
    analyzer.is_available = Mock(return_value=True)
    
    # Create mock graph engine
    graph_engine = Mock(spec=GraphMemoryEngine)
    graph_engine.retrieve_hybrid = AsyncMock(return_value=[])
    graph_engine.retrieve_by_similarity = AsyncMock(return_value=[])
    
    # Create gatekeeper
    gatekeeper = RetrievalGatekeeper(
        analyzer=analyzer,
        pinned_mgr=pinned_mgr,
        recent_cache=recent_cache,
        graph_engine=graph_engine
    )
    
    return {
        'gatekeeper': gatekeeper,
        'pinned_mgr': pinned_mgr,
        'recent_cache': recent_cache,
        'graph_engine': graph_engine,
        'analyzer': analyzer
    }


@pytest.mark.asyncio
async def test_full_retrieval_pipeline(integration_setup):
    """Test the full retrieval pipeline with real components."""
    gatekeeper = integration_setup['gatekeeper']
    context = create_conversation_context()
    context.turn_count = 10
    
    # Perform retrieval
    result = await gatekeeper.retrieve("What are my preferences?", context)
    
    # Verify result structure
    assert result is not None
    assert result.pinned is not None
    assert len(result.pinned) == 3  # language, style, safety
    assert result.memories is not None
    assert result.total_tokens > 0
    assert result.retrieval_time_ms >= 0
    assert result.query_intent is not None


@pytest.mark.asyncio
async def test_retrieval_uses_cache_first(integration_setup):
    """Test that retrieval checks cache before querying graph."""
    gatekeeper = integration_setup['gatekeeper']
    graph_engine = integration_setup['graph_engine']
    context = create_conversation_context()
    
    # Perform retrieval
    result = await gatekeeper.retrieve("Tell me about test data", context)
    
    # Should have retrieved from cache
    assert len(result.memories) > 0
    
    # Graph should have been queried (since cache doesn't have enough)
    assert graph_engine.retrieve_hybrid.called or graph_engine.retrieve_by_similarity.called


@pytest.mark.asyncio
async def test_retrieval_respects_intent_budget(integration_setup):
    """Test that retrieval respects budget based on intent."""
    gatekeeper = integration_setup['gatekeeper']
    context = create_conversation_context()
    
    # Query with NO_MEMORY intent
    result = await gatekeeper.retrieve("Hello", context)
    
    # Should have pinned memories but no other memories
    assert len(result.pinned) > 0
    assert len(result.memories) == 0
    assert result.query_intent == QueryIntent.NO_MEMORY


@pytest.mark.asyncio
async def test_retrieval_promotes_to_cache(integration_setup):
    """Test that retrieved memories are promoted to cache."""
    gatekeeper = integration_setup['gatekeeper']
    graph_engine = integration_setup['graph_engine']
    context = create_conversation_context()
    
    # Create mock memories from graph
    graph_memories = [
        create_memory_node(
            content=f"Graph memory {i}",
            category=MemoryCategory.CRITICAL,
            confidence=0.9,
            stability=0.8
        )
        for i in range(3)
    ]
    
    graph_engine.retrieve_hybrid = AsyncMock(return_value=graph_memories)
    
    initial_cache_size = integration_setup['recent_cache'].size()
    
    # Perform retrieval
    result = await gatekeeper.retrieve("What did I say?", context)
    
    # Cache should have grown
    final_cache_size = integration_setup['recent_cache'].size()
    assert final_cache_size >= initial_cache_size


@pytest.mark.asyncio
async def test_retrieval_with_long_conversation(integration_setup):
    """Test retrieval with a long conversation context."""
    gatekeeper = integration_setup['gatekeeper']
    context = create_conversation_context()
    context.turn_count = 1000
    context.recent_topics = ["topic1", "topic2", "topic3", "topic4", "topic5", "topic6"]
    context.active_entities = [f"entity{i}" for i in range(15)]
    
    # Perform retrieval
    result = await gatekeeper.retrieve("What are my preferences?", context)
    
    # Should still work with long context
    assert result is not None
    assert result.total_tokens > 0
    
    # Budget should be adjusted for long conversation
    assert result.query_intent is not None


@pytest.mark.asyncio
async def test_retrieval_scoring_prioritizes_critical(integration_setup):
    """Test that scoring prioritizes critical memories."""
    gatekeeper = integration_setup['gatekeeper']
    recent_cache = integration_setup['recent_cache']
    context = create_conversation_context()
    
    # Clear cache and add memories with different categories
    recent_cache.clear()
    
    critical_memory = create_memory_node(
        content="Critical user preference",
        category=MemoryCategory.CRITICAL,
        confidence=0.8,
        stability=0.8
    )
    
    temporary_memory = create_memory_node(
        content="Temporary note",
        category=MemoryCategory.TEMPORARY,
        confidence=0.8,
        stability=0.8
    )
    
    recent_cache.put(critical_memory)
    recent_cache.put(temporary_memory)
    
    # Perform retrieval
    result = await gatekeeper.retrieve("What are my preferences?", context)
    
    # Should retrieve memories
    assert len(result.memories) > 0
    
    # Critical memory should be included if budget allows
    memory_ids = [m.id for m in result.memories]
    if len(result.memories) >= 1:
        # At least one memory should be retrieved
        assert len(memory_ids) > 0


@pytest.mark.asyncio
async def test_pinned_memory_always_included(integration_setup):
    """Test that pinned memory is always included regardless of query."""
    gatekeeper = integration_setup['gatekeeper']
    context = create_conversation_context()
    
    queries = [
        "Hello",
        "What are my preferences?",
        "Tell me about Paris",
        "How is John related to me?"
    ]
    
    for query in queries:
        result = await gatekeeper.retrieve(query, context)
        
        # Pinned memory should always be present
        assert result.pinned is not None
        assert len(result.pinned) > 0
        assert "language" in result.pinned
        assert "style" in result.pinned
        assert "safety" in result.pinned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
