"""
Integration tests for Memory Influence Layer.

Tests the complete flow of memory injection with realistic scenarios.
"""

import pytest
from datetime import datetime
from src.memory_influence import MemoryInfluenceLayer
from src.models import (
    MemoryNode, MemoryCategory, RetrievalResult,
    create_memory_node
)


class TestMemoryInfluenceIntegration:
    """Integration tests for realistic memory injection scenarios."""
    
    def test_realistic_conversation_scenario(self):
        """Test a realistic conversation with multiple memory types."""
        # Setup
        influence_layer = MemoryInfluenceLayer(max_tokens=500)
        
        # Pinned memories (system configuration)
        pinned = {
            'language': 'English (US)',
            'style': 'Friendly and concise',
            'safety': 'Never provide medical advice',
            'timezone': 'America/New_York',
            'persona': 'Helpful assistant named Alex'
        }
        
        # Critical memories (user preferences and constraints)
        critical_mem1 = create_memory_node(
            content="User is vegan and has nut allergy",
            category=MemoryCategory.CRITICAL,
            structured_data={
                'preferences': ['vegan'],
                'constraints': ['no nuts', 'nut allergy']
            },
            confidence=0.98
        )
        
        critical_mem2 = create_memory_node(
            content="User prefers morning meetings",
            category=MemoryCategory.CRITICAL,
            structured_data={
                'preferences': ['morning meetings'],
                'commitments': ['Schedule meetings before noon']
            },
            confidence=0.92
        )
        
        # Episodic memories (recent events)
        episodic_mem1 = create_memory_node(
            content="User mentioned planning a vacation to Hawaii next month",
            category=MemoryCategory.EPISODIC,
            structured_data={
                'event': 'vacation planning',
                'location': 'Hawaii',
                'timeframe': 'next month'
            },
            confidence=0.88
        )
        
        episodic_mem2 = create_memory_node(
            content="User is working on a Python project for data analysis",
            category=MemoryCategory.EPISODIC,
            structured_data={
                'event': 'project work',
                'technology': 'Python',
                'domain': 'data analysis'
            },
            confidence=0.85
        )
        
        # Relational memories (entity relationships)
        relational_mem = create_memory_node(
            content="Sarah is user's colleague who works in marketing",
            category=MemoryCategory.RELATIONAL,
            structured_data={
                'entities': ['Sarah', 'User'],
                'relationships': [
                    "Sarah is User's colleague",
                    "Sarah works in marketing"
                ]
            },
            confidence=0.90
        )
        
        # Create retrieval result
        retrieval = RetrievalResult(
            pinned=pinned,
            memories=[
                critical_mem1,
                critical_mem2,
                episodic_mem1,
                episodic_mem2,
                relational_mem
            ],
            total_tokens=0
        )
        
        # Execute injection
        result = influence_layer.inject(retrieval, "What should I know about the user?")
        
        # Verify structure
        assert '<system_constraints>' in result
        assert '<critical_memory>' in result
        assert '<relevant_context>' in result
        assert '<entity_relationships>' in result
        
        # Verify content
        assert 'vegan' in result
        assert 'nut allergy' in result
        assert 'morning meetings' in result
        assert 'Hawaii' in result
        assert 'Python project' in result
        assert 'Sarah' in result
        assert 'marketing' in result
        
        # Verify token budget
        estimated_tokens = influence_layer.estimate_tokens(result)
        assert estimated_tokens <= 500, f"Exceeded token budget: {estimated_tokens}"
        
        # Verify priority ordering (pinned first)
        pinned_pos = result.index('<system_constraints>')
        critical_pos = result.index('<critical_memory>')
        assert pinned_pos < critical_pos
    
    def test_large_memory_set_truncation(self):
        """Test that large memory sets are properly truncated."""
        influence_layer = MemoryInfluenceLayer(max_tokens=500)
        
        # Create a large set of memories
        memories = []
        for i in range(100):
            mem = create_memory_node(
                content=f"User mentioned interest in topic {i} with detailed information about preferences and history",
                category=MemoryCategory.EPISODIC,
                structured_data={'topic': f'topic_{i}'}
            )
            memories.append(mem)
        
        retrieval = RetrievalResult(
            pinned={'language': 'English'},
            memories=memories,
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        # Verify truncation
        estimated_tokens = influence_layer.estimate_tokens(result)
        assert estimated_tokens <= 500
        
        # Verify structure is maintained
        assert '<system_constraints>' in result
        assert '<relevant_context>' in result
        if '<relevant_context>' in result:
            assert '</relevant_context>' in result
    
    def test_empty_memories_with_pinned(self):
        """Test injection with only pinned memories and no retrieved memories."""
        influence_layer = MemoryInfluenceLayer(max_tokens=500)
        
        pinned = {
            'language': 'French',
            'style': 'Formal',
            'safety': 'Standard guidelines',
            'timezone': 'Europe/Paris',
            'persona': 'Professional assistant'
        }
        
        retrieval = RetrievalResult(
            pinned=pinned,
            memories=[],
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        # Should only have pinned section
        assert '<system_constraints>' in result
        assert 'French' in result
        assert 'Formal' in result
        
        # Should not have other sections
        assert '<critical_memory>' not in result
        assert '<relevant_context>' not in result
        assert '<entity_relationships>' not in result
    
    def test_mixed_structured_and_unstructured_data(self):
        """Test handling of memories with varying structured data completeness."""
        influence_layer = MemoryInfluenceLayer(max_tokens=500)
        
        # Memory with full structured data
        mem1 = create_memory_node(
            content="User prefers dark mode",
            category=MemoryCategory.CRITICAL,
            structured_data={
                'preferences': ['dark mode', 'high contrast'],
                'constraints': ['avoid bright colors']
            }
        )
        
        # Memory with partial structured data
        mem2 = create_memory_node(
            content="User likes coffee",
            category=MemoryCategory.CRITICAL,
            structured_data={'preferences': ['coffee']}
        )
        
        # Memory with no structured data
        mem3 = create_memory_node(
            content="User mentioned enjoying hiking",
            category=MemoryCategory.EPISODIC,
            structured_data={}
        )
        
        retrieval = RetrievalResult(
            pinned={},
            memories=[mem1, mem2, mem3],
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        # All content should be present
        assert 'dark mode' in result
        assert 'coffee' in result
        assert 'hiking' in result
    
    def test_deduplication_across_categories(self):
        """Test that deduplication works within each category."""
        influence_layer = MemoryInfluenceLayer(max_tokens=500)
        
        # Two critical memories with same preference
        mem1 = create_memory_node(
            content="User is vegetarian",
            category=MemoryCategory.CRITICAL,
            structured_data={'preferences': ['vegetarian', 'organic']}
        )
        
        mem2 = create_memory_node(
            content="User prefers vegetarian options",
            category=MemoryCategory.CRITICAL,
            structured_data={'preferences': ['vegetarian', 'local produce']}
        )
        
        # Two episodic memories with same content
        mem3 = create_memory_node(
            content="User went to the gym",
            category=MemoryCategory.EPISODIC,
            structured_data={}
        )
        
        mem4 = create_memory_node(
            content="User went to the gym",
            category=MemoryCategory.EPISODIC,
            structured_data={}
        )
        
        retrieval = RetrievalResult(
            pinned={},
            memories=[mem1, mem2, mem3, mem4],
            total_tokens=0
        )
        
        result = influence_layer.inject(retrieval, "test query")
        
        # Vegetarian should appear only once
        vegetarian_count = result.count('User prefers: vegetarian')
        assert vegetarian_count == 1
        
        # Gym should appear only once
        gym_count = result.count('User went to the gym')
        assert gym_count == 1
        
        # Unique items should still be present
        assert 'organic' in result
        assert 'local produce' in result
