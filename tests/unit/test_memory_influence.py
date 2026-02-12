"""
Unit tests for the Memory Influence Layer.

Tests memory formatting, deduplication, token budget enforcement,
and priority ordering.
"""

import pytest
from datetime import datetime
from src.memory_influence import MemoryInfluenceLayer
from src.models import (
    MemoryNode, MemoryCategory, RetrievalResult,
    create_memory_node
)


class TestMemoryInfluenceLayer:
    """Test suite for MemoryInfluenceLayer class."""
    
    @pytest.fixture
    def influence_layer(self):
        """Create a MemoryInfluenceLayer instance."""
        return MemoryInfluenceLayer(max_tokens=500)
    
    @pytest.fixture
    def sample_pinned_memories(self):
        """Create sample pinned memories."""
        return {
            'language': 'English (US)',
            'style': 'Professional but friendly',
            'safety': 'Never provide medical/legal advice',
            'timezone': 'America/Los_Angeles',
            'persona': 'Helpful AI assistant'
        }
    
    @pytest.fixture
    def sample_critical_memory(self):
        """Create a sample critical memory node."""
        return create_memory_node(
            content="User is vegetarian and allergic to peanuts",
            category=MemoryCategory.CRITICAL,
            structured_data={
                'preferences': ['vegetarian'],
                'constraints': ['no peanuts', 'allergy: peanuts']
            },
            confidence=0.95,
            stability=0.9
        )
    
    @pytest.fixture
    def sample_episodic_memory(self):
        """Create a sample episodic memory node."""
        return create_memory_node(
            content="User mentioned planning a trip to Japan in March",
            category=MemoryCategory.EPISODIC,
            structured_data={
                'event': 'trip planning',
                'location': 'Japan',
                'timeframe': 'March 2024'
            },
            confidence=0.85,
            stability=0.6
        )
    
    @pytest.fixture
    def sample_relational_memory(self):
        """Create a sample relational memory node."""
        return create_memory_node(
            content="Sarah is the user's sister who lives in Seattle",
            category=MemoryCategory.RELATIONAL,
            structured_data={
                'entities': ['Sarah', 'User'],
                'relationships': [
                    "Sarah is User's sister",
                    "Sarah lives in Seattle"
                ]
            },
            confidence=0.90,
            stability=0.8
        )
    
    def test_inject_with_pinned_only(self, influence_layer, sample_pinned_memories):
        """Test injection with only pinned memories."""
        retrieval = RetrievalResult(
            pinned=sample_pinned_memories,
            memories=[],
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        assert '<system_constraints>' in result
        assert 'English (US)' in result
        assert 'Professional but friendly' in result
        assert 'Never provide medical/legal advice' in result
        assert 'America/Los_Angeles' in result
        assert 'Helpful AI assistant' in result
    
    def test_inject_with_critical_memory(self, influence_layer, sample_critical_memory):
        """Test injection with critical memories."""
        retrieval = RetrievalResult(
            pinned={},
            memories=[sample_critical_memory],
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        assert '<critical_memory>' in result
        assert 'User prefers: vegetarian' in result
        assert 'Constraint: no peanuts' in result
        assert 'Constraint: allergy: peanuts' in result
    
    def test_inject_with_episodic_memory(self, influence_layer, sample_episodic_memory):
        """Test injection with episodic memories."""
        retrieval = RetrievalResult(
            pinned={},
            memories=[sample_episodic_memory],
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        assert '<relevant_context>' in result
        assert 'User mentioned planning a trip to Japan in March' in result
        # Check that timestamp is included
        timestamp = sample_episodic_memory.created_at.strftime("%Y-%m-%d")
        assert timestamp in result
    
    def test_inject_with_relational_memory(self, influence_layer, sample_relational_memory):
        """Test injection with relational memories."""
        retrieval = RetrievalResult(
            pinned={},
            memories=[sample_relational_memory],
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        assert '<entity_relationships>' in result
        assert "Sarah is User's sister" in result
        assert "Sarah lives in Seattle" in result
    
    def test_inject_with_all_categories(
        self,
        influence_layer,
        sample_pinned_memories,
        sample_critical_memory,
        sample_episodic_memory,
        sample_relational_memory
    ):
        """Test injection with all memory categories."""
        retrieval = RetrievalResult(
            pinned=sample_pinned_memories,
            memories=[
                sample_critical_memory,
                sample_episodic_memory,
                sample_relational_memory
            ],
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        # Check all sections are present
        assert '<system_constraints>' in result
        assert '<critical_memory>' in result
        assert '<relevant_context>' in result
        assert '<entity_relationships>' in result
        
        # Verify pinned comes first (priority)
        pinned_pos = result.index('<system_constraints>')
        critical_pos = result.index('<critical_memory>')
        assert pinned_pos < critical_pos
    
    def test_no_duplicate_information(self, influence_layer):
        """Test that duplicate information is not included."""
        # Create two critical memories with overlapping preferences
        mem1 = create_memory_node(
            content="User prefers vegetarian food",
            category=MemoryCategory.CRITICAL,
            structured_data={
                'preferences': ['vegetarian', 'organic']
            }
        )
        
        mem2 = create_memory_node(
            content="User likes vegetarian and gluten-free options",
            category=MemoryCategory.CRITICAL,
            structured_data={
                'preferences': ['vegetarian', 'gluten-free']
            }
        )
        
        retrieval = RetrievalResult(
            pinned={},
            memories=[mem1, mem2],
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        # Count occurrences of "vegetarian"
        vegetarian_count = result.count('User prefers: vegetarian')
        assert vegetarian_count == 1, "Duplicate preference should appear only once"
        
        # Both unique preferences should be present
        assert 'User prefers: organic' in result
        assert 'User prefers: gluten-free' in result
    
    def test_truncate_to_budget(self, influence_layer):
        """Test that output is truncated to token budget."""
        # Create many memories to exceed budget
        memories = []
        for i in range(50):
            mem = create_memory_node(
                content=f"This is a long episodic memory entry number {i} with lots of text to fill up the token budget",
                category=MemoryCategory.EPISODIC,
                structured_data={}
            )
            memories.append(mem)
        
        retrieval = RetrievalResult(
            pinned={},
            memories=memories,
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        # Check that result is within budget
        estimated_tokens = influence_layer.estimate_tokens(result)
        assert estimated_tokens <= influence_layer.max_tokens, \
            f"Output exceeds token budget: {estimated_tokens} > {influence_layer.max_tokens}"
    
    def test_truncate_preserves_structure(self, influence_layer):
        """Test that truncation preserves section structure when possible."""
        # Create memories that will exceed budget
        critical_memories = [
            create_memory_node(
                content=f"Critical fact {i}",
                category=MemoryCategory.CRITICAL,
                structured_data={'facts': [f"Important fact number {i} with details"]}
            )
            for i in range(20)
        ]
        
        episodic_memories = [
            create_memory_node(
                content=f"Episodic event {i} with lots of contextual information",
                category=MemoryCategory.EPISODIC,
                structured_data={}
            )
            for i in range(30)
        ]
        
        retrieval = RetrievalResult(
            pinned={},
            memories=critical_memories + episodic_memories,
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        # Check that sections are properly closed (no broken XML-like tags)
        if '<critical_memory>' in result:
            assert '</critical_memory>' in result
        if '<relevant_context>' in result:
            assert '</relevant_context>' in result
    
    def test_empty_retrieval(self, influence_layer):
        """Test injection with empty retrieval result."""
        retrieval = RetrievalResult(
            pinned={},
            memories=[],
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        # Should return empty string or minimal output
        assert len(result) == 0 or result.strip() == ""
    
    def test_format_pinned_with_defaults(self, influence_layer):
        """Test pinned formatting with missing fields uses defaults."""
        partial_pinned = {
            'language': 'Spanish'
        }
        
        result = influence_layer._format_pinned(partial_pinned)
        
        assert 'Spanish' in result
        assert 'Professional' in result  # Default style
        assert 'Standard safety guidelines' in result  # Default safety
        assert 'UTC' in result  # Default timezone
        assert 'AI Assistant' in result  # Default persona
    
    def test_estimate_tokens(self, influence_layer):
        """Test token estimation."""
        text = "This is a test string with approximately twenty tokens in it for testing purposes."
        
        estimated = influence_layer.estimate_tokens(text)
        
        # Should be roughly len(text) / 4
        expected = len(text) // 4
        assert abs(estimated - expected) <= 1
    
    def test_critical_memory_with_facts(self, influence_layer):
        """Test critical memory formatting includes facts."""
        mem = create_memory_node(
            content="User information",
            category=MemoryCategory.CRITICAL,
            structured_data={
                'facts': ['Lives in San Francisco', 'Works as a software engineer']
            }
        )
        
        retrieval = RetrievalResult(
            pinned={},
            memories=[mem],
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        assert 'Fact: Lives in San Francisco' in result
        assert 'Fact: Works as a software engineer' in result
    
    def test_episodic_memory_deduplication(self, influence_layer):
        """Test that duplicate episodic content is not repeated."""
        mem1 = create_memory_node(
            content="User mentioned liking pizza",
            category=MemoryCategory.EPISODIC,
            structured_data={}
        )
        
        mem2 = create_memory_node(
            content="User mentioned liking pizza",
            category=MemoryCategory.EPISODIC,
            structured_data={}
        )
        
        retrieval = RetrievalResult(
            pinned={},
            memories=[mem1, mem2],
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        # Should only appear once
        count = result.count("User mentioned liking pizza")
        assert count == 1
    
    def test_relational_memory_without_structured_data(self, influence_layer):
        """Test relational memory formatting when structured_data lacks relationships."""
        mem = create_memory_node(
            content="Alice knows Bob from college",
            category=MemoryCategory.RELATIONAL,
            structured_data={}  # No relationships field
        )
        
        retrieval = RetrievalResult(
            pinned={},
            memories=[mem],
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        # Should fall back to raw content
        assert 'Alice knows Bob from college' in result
    
    def test_mixed_categories_ordering(self, influence_layer):
        """Test that memory categories appear in correct order."""
        pinned = {'language': 'English'}
        
        critical = create_memory_node(
            content="Critical info",
            category=MemoryCategory.CRITICAL,
            structured_data={'facts': ['Important fact']}
        )
        
        episodic = create_memory_node(
            content="Recent event",
            category=MemoryCategory.EPISODIC,
            structured_data={}
        )
        
        relational = create_memory_node(
            content="Person relationship",
            category=MemoryCategory.RELATIONAL,
            structured_data={'relationships': ['A knows B']}
        )
        
        retrieval = RetrievalResult(
            pinned=pinned,
            memories=[episodic, relational, critical],  # Intentionally out of order
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        # Find positions of each section
        pinned_pos = result.find('<system_constraints>')
        critical_pos = result.find('<critical_memory>')
        episodic_pos = result.find('<relevant_context>')
        relational_pos = result.find('<entity_relationships>')
        
        # Verify order: pinned < critical < episodic < relational
        assert pinned_pos < critical_pos
        assert critical_pos < episodic_pos
        assert episodic_pos < relational_pos
