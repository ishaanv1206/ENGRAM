"""
Unit tests for the Memory Analyzer module.
"""

import pytest
import pytest_asyncio
from datetime import datetime
from unittest.mock import Mock, patch

from src.memory_analyzer import MemoryAnalyzer, create_test_analyzer, analyze_text_simple
from src.models import MemoryCategory, DecayPolicy, ConversationContext, MemoryExtraction
from src.config import SLMConfig


class TestMemoryAnalyzer:
    """Test cases for MemoryAnalyzer class."""
    
    def test_init_with_real_model(self):
        """Test MemoryAnalyzer initialization with real model."""
        config = SLMConfig(
            model_path="data/models/Llama-3.2-1B-Instruct-Q6_K_L.gguf",
            n_ctx=2048,
            n_gpu_layers=0
        )
        
        analyzer = MemoryAnalyzer(config)
        assert analyzer.llm is not None
        assert analyzer.is_available()
        assert analyzer.config.model_path == "data/models/Llama-3.2-1B-Instruct-Q6_K_L.gguf"
    
    def test_get_decay_policy(self):
        """Test decay policy mapping for different memory categories."""
        analyzer = create_test_analyzer()
        
        assert analyzer._get_decay_policy(MemoryCategory.PINNED) == DecayPolicy.NO_DECAY
        assert analyzer._get_decay_policy(MemoryCategory.CRITICAL) == DecayPolicy.VERY_SLOW
        assert analyzer._get_decay_policy(MemoryCategory.EPISODIC) == DecayPolicy.MEDIUM
        assert analyzer._get_decay_policy(MemoryCategory.RELATIONAL) == DecayPolicy.MEDIUM
        assert analyzer._get_decay_policy(MemoryCategory.TEMPORARY) == DecayPolicy.FAST
        assert analyzer._get_decay_policy(MemoryCategory.DISCARD) == DecayPolicy.FAST
    
    def test_analyze_text_simple(self):
        """Test analyze_text_simple utility function."""
        result = analyze_text_simple("This is a test fact", "episodic")
        
        assert isinstance(result, MemoryExtraction)
        assert result.category == MemoryCategory.EPISODIC
        assert result.confidence == 0.8
        assert result.stability == 0.7
        assert "This is a test fact" in result.structured_data['facts']
        assert result.decay_policy == DecayPolicy.MEDIUM
    
    def test_create_fallback_extraction(self):
        """Test fallback extraction creation."""
        analyzer = create_test_analyzer()
        
        result = analyzer._create_fallback_extraction("test text")
        
        assert result.category == MemoryCategory.DISCARD
        assert result.confidence == 0.1
        assert result.stability == 0.1
        assert result.decay_policy == DecayPolicy.FAST
        assert len(result.links) == 0
        
        # Check structured data has all required fields
        expected_fields = ['preferences', 'facts', 'entities', 'relationships', 'commitments', 'constraints']
        for field in expected_fields:
            assert field in result.structured_data
            assert isinstance(result.structured_data[field], list)
    
    @pytest.mark.asyncio
    async def test_analyze_with_real_model(self):
        """Test analyze method with real model."""
        config = SLMConfig(
            model_path="data/models/Llama-3.2-1B-Instruct-Q6_K_L.gguf",
            n_ctx=2048,
            n_gpu_layers=0
        )
        
        analyzer = MemoryAnalyzer(config)
        
        context = ConversationContext(
            session_id="test_session",
            turn_count=1,
            recent_topics=["programming"],
            active_entities=["user"]
        )
        
        result = await analyzer.analyze("I love Python programming", context)
        
        assert isinstance(result, MemoryExtraction)
        assert result.category in [cat for cat in MemoryCategory]
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.stability <= 1.0
        assert result.decay_policy in [pol for pol in DecayPolicy]
    
    def test_create_classification_prompt(self):
        """Test classification prompt creation."""
        analyzer = create_test_analyzer()
        
        context = ConversationContext(
            session_id="test_session",
            turn_count=5,
            recent_topics=["programming", "python", "testing"],
            active_entities=["user", "system", "code"]
        )
        
        prompt = analyzer._create_classification_prompt("I prefer Python over Java", context)
        
        assert "Turn number: 5" in prompt
        assert "python, testing" in prompt
        assert "system, code" in prompt
        assert "I prefer Python over Java" in prompt
        assert "PINNED:" in prompt
        assert "CRITICAL:" in prompt
        assert "EPISODIC:" in prompt
        assert "Analysis:" in prompt
