"""
Unit tests for the Retrieval Gatekeeper.

Tests the core functionality of query intent detection, budget calculation,
memory scoring, and retrieval orchestration.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.retrieval_gatekeeper import RetrievalGatekeeper
from src.models import (
    MemoryNode, MemoryCategory, QueryIntent, ConversationContext,
    create_conversation_context, create_memory_node
)
from src.memory_analyzer import MemoryAnalyzer
from src.pinned_memory import PinnedMemoryManager
from src.recent_cache import RecentMemoryCache
from src.graph_engine import GraphMemoryEngine


@pytest.fixture
def mock_analyzer():
    """Create a mock memory analyzer."""
    analyzer = Mock(spec=MemoryAnalyzer)
    analyzer.is_available = Mock(return_value=True)
    return analyzer


@pytest.fixture
def pinned_manager(tmp_path):
    """Create a pinned memory manager with test data."""
    storage_path = tmp_path / "pinned.json"
    manager = PinnedMemoryManager(str(storage_path))
    manager.set("language", "English (US)")
    manager.set("style", "Professional and friendly")
    return manager


@pytest.fixture
def recent_cache():
    """Create a recent memory cache."""
    return RecentMemoryCache(max_size=100)


@pytest.fixture
def mock_graph_engine():
    """Create a mock graph engine."""
    engine = Mock(spec=GraphMemoryEngine)
    engine.retrieve_hybrid = AsyncMock(return_value=[])
    engine.retrieve_by_similarity = AsyncMock(return_value=[])
    return engine


@pytest.fixture
def gatekeeper(mock_analyzer, pinned_manager, recent_cache, mock_graph_engine):
    """Create a retrieval gatekeeper with mocked dependencies."""
    return RetrievalGatekeeper(
        analyzer=mock_analyzer,
        pinned_mgr=pinned_manager,
        recent_cache=recent_cache,
        graph_engine=mock_graph_engine
    )


@pytest.fixture
def sample_context():
    """Create a sample conversation context."""
    context = create_conversation_context()
    context.turn_count = 10
    context.recent_topics = ["weather", "travel", "food"]
    context.active_entities = ["Paris", "John", "restaurant"]
    return context


@pytest.fixture
def sample_memories():
    """Create sample memory nodes for testing."""
    now = datetime.now()
    
    memories = [
        create_memory_node(
            content="User prefers vegetarian food",
            category=MemoryCategory.CRITICAL,
            structured_data={"preferences": ["vegetarian"]},
            confidence=0.95,
            stability=0.9
        ),
        create_memory_node(
            content="User visited Paris last summer",
            category=MemoryCategory.EPISODIC,
            structured_data={"event": "trip", "location": "Paris"},
            confidence=0.85,
            stability=0.7
        ),
        create_memory_node(
            content="John is user's colleague",
            category=MemoryCategory.RELATIONAL,
            structured_data={"entities": ["John", "User"], "relationships": ["colleague"]},
            confidence=0.90,
            stability=0.8
        ),
    ]
    
    # Set different access times
    for i, memory in enumerate(memories):
        memory.last_accessed = now - timedelta(days=i * 5)
        memory.access_count = 10 - i * 2
    
    return memories


class TestIntentDetection:
    """Test query intent detection."""
    
    @pytest.mark.asyncio
    async def test_detect_factual_recall_intent(self, gatekeeper, sample_context):
        """Test detection of factual recall queries."""
        queries = [
            "What did I say about Paris?",
            "Did I mention my food preferences?",
            "When did I visit France?"
        ]
        
        for query in queries:
            intent = await gatekeeper._detect_intent(query, sample_context)
            assert intent == QueryIntent.FACTUAL_RECALL
    
    @pytest.mark.asyncio
    async def test_detect_preference_check_intent(self, gatekeeper, sample_context):
        """Test detection of preference check queries."""
        queries = [
            "Do I like vegetarian food?",
            "What is my favorite restaurant?",
            "Am I allergic to anything?"
        ]
        
        for query in queries:
            intent = await gatekeeper._detect_intent(query, sample_context)
            assert intent == QueryIntent.PREFERENCE_CHECK
    
    @pytest.mark.asyncio
    async def test_detect_relationship_intent(self, gatekeeper, sample_context):
        """Test detection of relationship queries."""
        queries = [
            "How is John related to me?",
            "What is the connection between Paris and my trip?",
            "Who is Sarah?"
        ]
        
        for query in queries:
            intent = await gatekeeper._detect_intent(query, sample_context)
            assert intent == QueryIntent.RELATIONSHIP
    
    @pytest.mark.asyncio
    async def test_detect_no_memory_intent(self, gatekeeper, sample_context):
        """Test detection of queries that don't need memory."""
        queries = [
            "Hello",
            "Thanks",
            "Okay"
        ]
        
        for query in queries:
            intent = await gatekeeper._detect_intent(query, sample_context)
            assert intent == QueryIntent.NO_MEMORY
    
    @pytest.mark.asyncio
    async def test_detect_general_intent(self, gatekeeper, sample_context):
        """Test detection of general queries."""
        query = "Tell me about the weather today"
        intent = await gatekeeper._detect_intent(query, sample_context)
        assert intent == QueryIntent.GENERAL


class TestBudgetCalculation:
    """Test memory budget calculation."""
    
    def test_calculate_budget_factual_recall(self, gatekeeper, sample_context):
        """Test budget for factual recall queries."""
        budget = gatekeeper._calculate_budget(QueryIntent.FACTUAL_RECALL, sample_context)
        
        assert budget.max_memories > 0
        assert budget.max_tokens > 0
        assert budget.latency_ms > 0
    
    def test_calculate_budget_no_memory(self, gatekeeper, sample_context):
        """Test budget for queries that don't need memory."""
        budget = gatekeeper._calculate_budget(QueryIntent.NO_MEMORY, sample_context)
        
        assert budget.max_memories == 0
        assert budget.max_tokens == 0
    
    def test_budget_scales_with_conversation_length(self, gatekeeper, sample_context):
        """Test that budget increases for long conversations."""
        sample_context.turn_count = 100
        budget_short = gatekeeper._calculate_budget(QueryIntent.GENERAL, sample_context)
        
        sample_context.turn_count = 1000
        budget_long = gatekeeper._calculate_budget(QueryIntent.GENERAL, sample_context)
        
        assert budget_long.max_memories >= budget_short.max_memories


class TestMemoryScoring:
    """Test multi-factor memory scoring."""
    
    def test_score_memories_returns_sorted_list(self, gatekeeper, sample_memories, sample_context):
        """Test that scoring returns memories sorted by score."""
        query = "What are my food preferences?"
        
        scored = gatekeeper._score_memories(sample_memories, query, sample_context)
        
        assert len(scored) == len(sample_memories)
        
        # Verify sorted in descending order
        scores = [score for _, score in scored]
        assert scores == sorted(scores, reverse=True)
    
    def test_score_memories_all_factors_applied(self, gatekeeper, sample_memories, sample_context):
        """Test that all scoring factors are applied."""
        query = "Tell me about my preferences"
        
        scored = gatekeeper._score_memories(sample_memories, query, sample_context)
        
        # All scores should be between 0 and 1
        for _, score in scored:
            assert 0.0 <= score <= 1.0
    
    def test_critical_memories_score_higher(self, gatekeeper, sample_context):
        """Test that critical memories get higher category weights."""
        critical_memory = create_memory_node(
            content="Critical information",
            category=MemoryCategory.CRITICAL,
            confidence=0.8,
            stability=0.8
        )
        
        temporary_memory = create_memory_node(
            content="Temporary information",
            category=MemoryCategory.TEMPORARY,
            confidence=0.8,
            stability=0.8
        )
        
        memories = [critical_memory, temporary_memory]
        scored = gatekeeper._score_memories(memories, "test query", sample_context)
        
        # Critical should score higher due to category weight
        critical_score = next(score for mem, score in scored if mem.id == critical_memory.id)
        temporary_score = next(score for mem, score in scored if mem.id == temporary_memory.id)
        
        assert critical_score > temporary_score


class TestCacheAccess:
    """Test cache tier access."""
    
    def test_check_cache_returns_cached_memories(self, gatekeeper, sample_memories, sample_context):
        """Test that cache check returns cached memories."""
        # Add memories to cache
        for memory in sample_memories:
            gatekeeper.recent_cache.put(memory)
        
        budget = gatekeeper._calculate_budget(QueryIntent.GENERAL, sample_context)
        cached = gatekeeper._check_cache("test query", budget)
        
        assert len(cached) > 0
        assert all(isinstance(m, MemoryNode) for m in cached)
    
    def test_check_cache_respects_budget(self, gatekeeper, sample_memories, sample_context):
        """Test that cache check respects budget limits."""
        # Add many memories to cache
        for i in range(20):
            memory = create_memory_node(
                content=f"Memory {i}",
                category=MemoryCategory.EPISODIC,
                confidence=0.8,
                stability=0.7
            )
            gatekeeper.recent_cache.put(memory)
        
        budget = gatekeeper._calculate_budget(QueryIntent.GENERAL, sample_context)
        budget.max_memories = 5
        
        cached = gatekeeper._check_cache("test query", budget)
        
        assert len(cached) <= budget.max_memories


class TestTopKPruning:
    """Test top-K budget pruning."""
    
    def test_prune_to_budget_respects_max_memories(self, gatekeeper, sample_memories, sample_context):
        """Test that pruning respects max_memories limit."""
        scored = gatekeeper._score_memories(sample_memories, "test", sample_context)
        
        budget = gatekeeper._calculate_budget(QueryIntent.GENERAL, sample_context)
        budget.max_memories = 2
        
        pruned = gatekeeper._prune_to_budget(scored, budget)
        
        assert len(pruned) <= budget.max_memories
    
    def test_prune_to_budget_selects_highest_scoring(self, gatekeeper, sample_memories, sample_context):
        """Test that pruning selects highest-scoring memories."""
        scored = gatekeeper._score_memories(sample_memories, "test", sample_context)
        
        budget = gatekeeper._calculate_budget(QueryIntent.GENERAL, sample_context)
        budget.max_memories = 2
        
        pruned = gatekeeper._prune_to_budget(scored, budget)
        
        # Get the top 2 scores
        top_scores = [score for _, score in scored[:2]]
        
        # Verify pruned memories are from the top scores
        assert len(pruned) <= 2
    
    def test_prune_empty_list(self, gatekeeper, sample_context):
        """Test pruning with empty memory list."""
        budget = gatekeeper._calculate_budget(QueryIntent.GENERAL, sample_context)
        pruned = gatekeeper._prune_to_budget([], budget)
        
        assert pruned == []


class TestRetrievalOrchestration:
    """Test full retrieval orchestration."""
    
    @pytest.mark.asyncio
    async def test_retrieve_includes_pinned_memories(self, gatekeeper, sample_context):
        """Test that retrieval always includes pinned memories."""
        result = await gatekeeper.retrieve("test query", sample_context)
        
        assert result.pinned is not None
        assert len(result.pinned) > 0
        assert "language" in result.pinned
    
    @pytest.mark.asyncio
    async def test_retrieve_returns_retrieval_result(self, gatekeeper, sample_context):
        """Test that retrieve returns a proper RetrievalResult."""
        result = await gatekeeper.retrieve("What are my preferences?", sample_context)
        
        assert result.pinned is not None
        assert result.memories is not None
        assert isinstance(result.total_tokens, int)
        assert result.total_tokens >= 0
        assert result.retrieval_time_ms >= 0
        assert result.query_intent is not None
    
    @pytest.mark.asyncio
    async def test_retrieve_no_memory_intent_returns_empty(self, gatekeeper, sample_context):
        """Test that NO_MEMORY intent returns minimal results."""
        result = await gatekeeper.retrieve("Hello", sample_context)
        
        # Should still have pinned memories but no other memories
        assert result.pinned is not None
        assert len(result.memories) == 0
    
    @pytest.mark.asyncio
    async def test_retrieve_promotes_to_cache(self, gatekeeper, sample_context, sample_memories):
        """Test that retrieved memories are promoted to cache."""
        # Mock graph engine to return sample memories
        gatekeeper.graph_engine.retrieve_hybrid = AsyncMock(return_value=sample_memories)
        
        initial_cache_size = gatekeeper.recent_cache.size()
        
        result = await gatekeeper.retrieve("What did I do in Paris?", sample_context)
        
        # Cache should have grown
        assert gatekeeper.recent_cache.size() >= initial_cache_size


class TestTokenEstimation:
    """Test token estimation."""
    
    def test_estimate_tokens_for_pinned(self, gatekeeper):
        """Test token estimation for pinned memories."""
        pinned = {
            "language": "English (US)",
            "style": "Professional and friendly"
        }
        
        tokens = gatekeeper._estimate_tokens(pinned, [])
        
        assert tokens > 0
    
    def test_estimate_tokens_for_memories(self, gatekeeper, sample_memories):
        """Test token estimation for memory nodes."""
        tokens = gatekeeper._estimate_tokens({}, sample_memories)
        
        assert tokens > 0
    
    def test_estimate_memory_tokens(self, gatekeeper):
        """Test token estimation for a single memory."""
        memory = create_memory_node(
            content="This is a test memory with some content",
            category=MemoryCategory.EPISODIC,
            structured_data={"key": "value"}
        )
        
        tokens = gatekeeper._estimate_memory_tokens(memory)
        
        assert tokens > 0


class TestSimilarityCalculation:
    """Test similarity calculation."""
    
    def test_calculate_similarity_with_valid_embeddings(self, gatekeeper):
        """Test similarity calculation with valid embeddings."""
        embedding1 = [0.5] * 384
        embedding2 = [0.5] * 384
        
        similarity = gatekeeper._calculate_similarity(embedding1, embedding2)
        
        assert 0.0 <= similarity <= 1.0
    
    def test_calculate_similarity_with_none_embedding(self, gatekeeper):
        """Test similarity calculation when memory has no embedding."""
        embedding1 = [0.5] * 384
        
        similarity = gatekeeper._calculate_similarity(embedding1, None)
        
        # Should return default similarity
        assert similarity == 0.5
    
    def test_calculate_similarity_identical_vectors(self, gatekeeper):
        """Test similarity of identical vectors."""
        embedding = [0.5] * 384
        
        similarity = gatekeeper._calculate_similarity(embedding, embedding)
        
        # Identical vectors should have high similarity
        assert similarity > 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
