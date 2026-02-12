import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime

from src.models import (
    MemoryNode, MemoryCategory, ConversationContext, 
    ReflectionTask, ValidationResult
)
from src.reflection_loop import ReflectionLoop
from src.graph_engine import GraphMemoryEngine
from src.llm_client import LocalLLMClient

class TestMemoryConsolidation:
    
    @pytest.fixture
    def mock_components(self):
        analyzer = Mock()
        graph_engine = AsyncMock(spec=GraphMemoryEngine)
        llm_client = AsyncMock(spec=LocalLLMClient)
        
        return analyzer, graph_engine, llm_client

    @pytest.mark.asyncio
    async def test_consolidation_logic(self, mock_components):
        analyzer, graph_engine, llm_client = mock_components
        
        # Setup memories
        memories = [
            MemoryNode(
                id="m1", category=MemoryCategory.CRITICAL, 
                content="Project Phoenix uses Python.", 
                confidence=0.9, stability=0.8, created_at=datetime.now(), 
                last_accessed=datetime.now(), access_count=1, decay_rate=0.0, 
                structured_data={"entities": ["project phoenix"]}, embedding=[]
            ),
            MemoryNode(
                id="m2", category=MemoryCategory.CRITICAL, 
                content="Phoenix Project is a Python app.", 
                confidence=0.8, stability=0.7, created_at=datetime.now(), 
                last_accessed=datetime.now(), access_count=1, decay_rate=0.0, 
                structured_data={"entities": ["project phoenix"]}, embedding=[]
            ),
            MemoryNode(
                id="m3", category=MemoryCategory.CRITICAL, 
                content="The backend of Phoenix is Python.", 
                confidence=0.85, stability=0.75, created_at=datetime.now(), 
                last_accessed=datetime.now(), access_count=1, decay_rate=0.0, 
                structured_data={"entities": ["project phoenix"]}, embedding=[]
            )
        ]
        
        # Setup mocks
        graph_engine.get_memories_by_entity.return_value = memories
        graph_engine.merge_memories.return_value = True
        
        # Mock LLM response
        llm_client.generate.return_value = """
        MERGE_IDS: [m1, m2, m3]
        MERGED_CONTENT: Project Phoenix is a Python application with a Python backend.
        """
        
        # Initialize loop
        loop = ReflectionLoop(analyzer, graph_engine, llm_client)
        
        # Create context with active entity
        context = ConversationContext(session_id="test_session", turn_count=0)
        context.active_entities = ["project phoenix"]
        
        # Run consolidation directly
        await loop._consolidate_memories(context)
        
        # Verify calls
        graph_engine.get_memories_by_entity.assert_awaited_with("project phoenix")
        
        # Verify LLM call
        assert llm_client.generate.called
        call_args = llm_client.generate.call_args[0][0]
        assert "Project Phoenix uses Python" in call_args
        assert "Phoenix Project is a Python app" in call_args
        
        # Verify Merge call
        # m1 should be primary (first in list), m2 and m3 secondary
        graph_engine.merge_memories.assert_awaited_with(
            primary_id="m1",
            secondary_ids=["m2", "m3"],
            new_content="Project Phoenix is a Python application with a Python backend."
        )

    @pytest.mark.asyncio
    async def test_no_merge_needed(self, mock_components):
        analyzer, graph_engine, llm_client = mock_components
        
        # Memories that shouldn't be merged
        memories = [
            MemoryNode(id="m1", category=MemoryCategory.CRITICAL, content="Project Phoenix is cool.", confidence=1.0, stability=1.0, created_at=datetime.now(), last_accessed=datetime.now(), access_count=1, decay_rate=0, structured_data={}, embedding=[]),
            MemoryNode(id="m2", category=MemoryCategory.CRITICAL, content="Project Phoenix deadlines are tight.", confidence=1.0, stability=1.0, created_at=datetime.now(), last_accessed=datetime.now(), access_count=1, decay_rate=0, structured_data={}, embedding=[])
        ]
        
        graph_engine.get_memories_by_entity.return_value = memories
        llm_client.generate.return_value = "NO_MERGE"
        
        loop = ReflectionLoop(analyzer, graph_engine, llm_client)
        context = ConversationContext(session_id="test_session", turn_count=0)
        context.active_entities = ["project phoenix"]
        
        await loop._consolidate_memories(context)
        
        # Verify no merge called
        assert not graph_engine.merge_memories.called
