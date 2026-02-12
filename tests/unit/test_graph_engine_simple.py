"""
Simple integration tests for Graph Memory Engine.
Tests actual Neo4j connection and basic operations.
"""

import pytest
import asyncio
from datetime import datetime

from src.graph_engine import GraphMemoryEngine
from src.models import MemoryExtraction, MemoryCategory, DecayPolicy, MemoryLink, LinkType
from src.config import ConfigManager


class TestGraphMemoryEngineReal:
    """Real integration tests for GraphMemoryEngine."""
    
    @pytest.fixture
    def config(self):
        """Load real configuration from .env file."""
        return ConfigManager.load()
    
    @pytest.fixture
    def engine(self, config):
        """Create GraphMemoryEngine with real Neo4j connection."""
        engine = GraphMemoryEngine(config.neo4j)
        yield engine
        engine.close()
    
    def test_connection_and_schema_creation(self, engine):
        """Test that we can connect to Neo4j and create schema."""
        # If we get here without exception, connection worked
        assert engine.driver is not None
        
        # Test basic connectivity by running a simple query
        with engine.driver.session() as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            assert record["test"] == 1
    
    def test_store_simple_memory(self, engine):
        """Test storing a simple memory in Neo4j."""
        # Create a simple memory extraction
        extraction = MemoryExtraction(
            category=MemoryCategory.EPISODIC,
            structured_data={
                "content": "Test memory for integration test",
                "facts": ["This is a test fact"]
            },
            confidence=0.9,
            stability=0.8,
            decay_policy=DecayPolicy.MEDIUM,
            links=[],
            timestamp=datetime.now()
        )
        
        # Store the memory
        memory_id = asyncio.run(engine.store_memory(extraction))
        
        # Verify it was stored
        assert memory_id is not None
        assert len(memory_id) > 0
        
        # Clean up - delete the test memory
        with engine.driver.session() as session:
            session.run("MATCH (m:Memory {id: $id}) DELETE m", id=memory_id)
    
    def test_retrieve_by_confidence(self, engine):
        """Test retrieving memories by confidence (fallback method)."""
        # First store a test memory
        extraction = MemoryExtraction(
            category=MemoryCategory.CRITICAL,
            structured_data={
                "content": "High confidence test memory",
                "preferences": ["testing"]
            },
            confidence=0.95,
            stability=0.9,
            decay_policy=DecayPolicy.VERY_SLOW,
            links=[],
            timestamp=datetime.now()
        )
        
        memory_id = asyncio.run(engine.store_memory(extraction))
        
        try:
            # Try to retrieve it
            memories = asyncio.run(engine.retrieve_by_similarity([0.1, 0.2, 0.3], limit=5))
            
            # Should get some results (may include our test memory or others)
            assert isinstance(memories, list)
            
        finally:
            # Clean up
            with engine.driver.session() as session:
                session.run("MATCH (m:Memory {id: $id}) DELETE m", id=memory_id)
    
    def test_update_access_metrics(self, engine):
        """Test updating access metrics for a memory."""
        # Store a test memory
        extraction = MemoryExtraction(
            category=MemoryCategory.EPISODIC,
            structured_data={"content": "Access metrics test"},
            confidence=0.8,
            stability=0.7,
            decay_policy=DecayPolicy.MEDIUM,
            links=[],
            timestamp=datetime.now()
        )
        
        memory_id = asyncio.run(engine.store_memory(extraction))
        
        try:
            # Update access metrics
            asyncio.run(engine.update_access_metrics(memory_id))
            
            # Verify the update worked by checking the memory
            with engine.driver.session() as session:
                result = session.run(
                    "MATCH (m:Memory {id: $id}) RETURN m.access_count as count",
                    id=memory_id
                )
                record = result.single()
                if record:
                    assert record["count"] >= 1  # Should have been incremented
                    
        finally:
            # Clean up
            with engine.driver.session() as session:
                session.run("MATCH (m:Memory {id: $id}) DELETE m", id=memory_id)
    
    def test_get_by_category(self, engine):
        """Test retrieving memories by category."""
        # Store a test memory with specific category
        extraction = MemoryExtraction(
            category=MemoryCategory.TEMPORARY,
            structured_data={"content": "Category test memory"},
            confidence=0.7,
            stability=0.6,
            decay_policy=DecayPolicy.FAST,
            links=[],
            timestamp=datetime.now()
        )
        
        memory_id = asyncio.run(engine.store_memory(extraction))
        
        try:
            # Retrieve by category
            memories = asyncio.run(engine.get_by_category(MemoryCategory.TEMPORARY, limit=10))
            
            # Should get a list (may be empty or contain our memory)
            assert isinstance(memories, list)
            
            # If we got results, verify they're the right category
            for memory in memories:
                assert memory.category == MemoryCategory.TEMPORARY
                
        finally:
            # Clean up
            with engine.driver.session() as session:
                session.run("MATCH (m:Memory {id: $id}) DELETE m", id=memory_id)
    
    def test_apply_decay(self, engine):
        """Test applying decay to memory confidence."""
        # Store a test memory
        extraction = MemoryExtraction(
            category=MemoryCategory.EPISODIC,
            structured_data={"content": "Decay test memory"},
            confidence=0.8,
            stability=0.7,
            decay_policy=DecayPolicy.MEDIUM,
            links=[],
            timestamp=datetime.now()
        )
        
        memory_id = asyncio.run(engine.store_memory(extraction))
        
        try:
            # Get initial confidence
            with engine.driver.session() as session:
                result = session.run(
                    "MATCH (m:Memory {id: $id}) RETURN m.confidence as confidence",
                    id=memory_id
                )
                initial_confidence = result.single()["confidence"]
            
            # Apply decay
            asyncio.run(engine.apply_decay([memory_id]))
            
            # Check if confidence was reduced
            with engine.driver.session() as session:
                result = session.run(
                    "MATCH (m:Memory {id: $id}) RETURN m.confidence as confidence",
                    id=memory_id
                )
                new_confidence = result.single()["confidence"]
                
            # Confidence should be reduced (or at least not increased)
            assert new_confidence <= initial_confidence
                
        finally:
            # Clean up
            with engine.driver.session() as session:
                session.run("MATCH (m:Memory {id: $id}) DELETE m", id=memory_id)
    
    def test_archive_low_confidence(self, engine):
        """Test archiving low-confidence memories."""
        # Store a low-confidence memory
        extraction = MemoryExtraction(
            category=MemoryCategory.TEMPORARY,
            structured_data={"content": "Low confidence memory"},
            confidence=0.05,  # Very low confidence
            stability=0.1,
            decay_policy=DecayPolicy.FAST,
            links=[],
            timestamp=datetime.now()
        )
        
        memory_id = asyncio.run(engine.store_memory(extraction))
        
        try:
            # Archive low-confidence memories
            archived_count = asyncio.run(engine.archive_low_confidence(threshold=0.1))
            
            # Should return a number (may be 0 or more)
            assert isinstance(archived_count, int)
            assert archived_count >= 0
            
            # Check if our memory was archived
            with engine.driver.session() as session:
                result = session.run(
                    "MATCH (m:Memory {id: $id}) RETURN m:Archived as is_archived",
                    id=memory_id
                )
                record = result.single()
                if record:
                    # If confidence was 0.05, it should be archived (threshold 0.1)
                    assert record["is_archived"] == True
                
        finally:
            # Clean up (including archived memories)
            with engine.driver.session() as session:
                session.run("MATCH (m:Memory {id: $id}) DELETE m", id=memory_id)
    
    def test_context_manager(self, config):
        """Test using GraphMemoryEngine as context manager."""
        # Test that context manager works properly
        with GraphMemoryEngine(config.neo4j) as engine:
            assert engine.driver is not None
            
            # Test basic operation
            with engine.driver.session() as session:
                result = session.run("RETURN 'context_test' as test")
                record = result.single()
                assert record["test"] == "context_test"
        
        # After context manager, connection should be closed
        # (We can't easily test this without accessing private members)