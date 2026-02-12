"""
Integration tests for Graph Memory Engine.

These tests connect to a real Neo4j database to verify functionality.
"""

import json
import pytest
import asyncio
from datetime import datetime
from typing import List
from unittest.mock import patch, Mock

from src.graph_engine import GraphMemoryEngine, GraphMemoryEngineError
from src.models import (
    MemoryExtraction, MemoryCategory, LinkType, MemoryLink, 
    DecayPolicy, MemoryNode
)
from src.config import ConfigManager


class TestGraphMemoryEngine:
    """Test cases for GraphMemoryEngine."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock Neo4j configuration."""
        return Neo4jConfig(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="test_password"
        )
    
    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = Mock()
        session = Mock()
        # Properly mock the context manager
        session_context = Mock()
        session_context.__enter__ = Mock(return_value=session)
        session_context.__exit__ = Mock(return_value=None)
        driver.session.return_value = session_context
        return driver, session
    
    def test_initialization_success(self, mock_config):
        """Test successful initialization with mocked Neo4j."""
        with patch('src.graph_engine.GraphDatabase') as mock_graphdb:
            mock_driver = Mock()
            mock_session = Mock()
            # Properly mock the context manager
            session_context = Mock()
            session_context.__enter__ = Mock(return_value=mock_session)
            session_context.__exit__ = Mock(return_value=None)
            mock_driver.session.return_value = session_context
            mock_session.run.return_value = None
            mock_graphdb.driver.return_value = mock_driver
            
            engine = GraphMemoryEngine(mock_config)
            
            # Verify connection was attempted
            mock_graphdb.driver.assert_called_once_with(
                mock_config.uri,
                auth=(mock_config.username, mock_config.password)
            )
            
            # Verify schema creation was attempted
            assert mock_session.run.call_count >= 5  # Multiple schema creation calls
    
    def test_initialization_connection_failure(self, mock_config):
        """Test initialization failure when Neo4j is unavailable."""
        with patch('src.graph_engine.GraphDatabase') as mock_graphdb:
            from neo4j.exceptions import ServiceUnavailable
            mock_graphdb.driver.side_effect = ServiceUnavailable("Connection failed")
            
            with pytest.raises(GraphMemoryEngineError) as exc_info:
                GraphMemoryEngine(mock_config)
            
            assert "Neo4j connection failed" in str(exc_info.value)
    
    @patch('src.graph_engine.GraphDatabase')
    def test_store_memory_success(self, mock_graphdb, mock_config):
        """Test successful memory storage."""
        # Setup mocks
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        
        # Mock the result of the CREATE query
        mock_result = Mock()
        mock_record = Mock()
        mock_record.__getitem__.return_value = "test-memory-id"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        
        mock_graphdb.driver.return_value = mock_driver
        
        engine = GraphMemoryEngine(mock_config)
        
        # Create test memory extraction
        extraction = MemoryExtraction(
            category=MemoryCategory.EPISODIC,
            structured_data={
                "content": "Test memory content",
                "preferences": ["coffee"],
                "facts": ["user likes morning meetings"]
            },
            confidence=0.9,
            stability=0.8,
            decay_policy=DecayPolicy.MEDIUM,
            links=[],
            timestamp=datetime.now()
        )
        
        # Test store_memory (need to run async)
        import asyncio
        memory_id = asyncio.run(engine.store_memory(extraction))
        
        # Verify the memory was stored
        assert memory_id == "test-memory-id"
        
        # Verify CREATE query was called
        create_calls = [call for call in mock_session.run.call_args_list 
                       if 'CREATE (m:Memory' in str(call)]
        assert len(create_calls) >= 1
    
    @patch('src.graph_engine.GraphDatabase')
    def test_store_memory_with_links(self, mock_graphdb, mock_config):
        """Test memory storage with relationship links."""
        # Setup mocks
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        
        # Mock the result of the CREATE query
        mock_result = Mock()
        mock_record = Mock()
        mock_record.__getitem__.return_value = "test-memory-id"
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        
        mock_graphdb.driver.return_value = mock_driver
        
        engine = GraphMemoryEngine(mock_config)
        
        # Create memory extraction with links
        memory_link = MemoryLink(
            target_id="existing-memory-id",
            link_type=LinkType.STRENGTHEN,
            weight=0.8,
            created_at=datetime.now()
        )
        
        extraction = MemoryExtraction(
            category=MemoryCategory.CRITICAL,
            structured_data={"content": "Test memory with links"},
            confidence=0.95,
            stability=0.9,
            decay_policy=DecayPolicy.VERY_SLOW,
            links=[memory_link],
            timestamp=datetime.now()
        )
        
        # Test store_memory
        import asyncio
        memory_id = asyncio.run(engine.store_memory(extraction))
        
        # Verify both memory creation and link creation were called
        assert memory_id == "test-memory-id"
        
        # Should have calls for both CREATE memory and CREATE relationship
        all_calls = [str(call) for call in mock_session.run.call_args_list]
        create_memory_calls = [call for call in all_calls if 'CREATE (m:Memory' in call]
        create_link_calls = [call for call in all_calls if 'STRENGTHENS' in call]
        
        assert len(create_memory_calls) >= 1
        assert len(create_link_calls) >= 1
    
    @patch('src.graph_engine.GraphDatabase')
    def test_retrieve_by_similarity_with_vector_index(self, mock_graphdb, mock_config):
        """Test similarity retrieval using vector index."""
        # Setup mocks
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        
        # Mock vector search result
        mock_node = {
            'id': 'test-id',
            'category': 'episodic',
            'content': 'Test content',
            'structured_data': '{"facts": ["test fact"]}',
            'confidence': 0.9,
            'stability': 0.8,
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 5,
            'decay_rate': 0.02,
            'embedding': None
        }
        
        mock_record = Mock()
        mock_record.__getitem__.return_value = mock_node
        mock_session.run.return_value = [mock_record]
        
        mock_graphdb.driver.return_value = mock_driver
        
        engine = GraphMemoryEngine(mock_config)
        
        # Test similarity retrieval
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        import asyncio
        memories = asyncio.run(engine.retrieve_by_similarity(query_embedding, limit=10))
        
        # Verify results
        assert len(memories) == 1
        assert memories[0].id == 'test-id'
        assert memories[0].category == MemoryCategory.EPISODIC
        assert memories[0].confidence == 0.9
        
        # Verify vector search query was called
        vector_calls = [call for call in mock_session.run.call_args_list 
                       if 'db.index.vector.queryNodes' in str(call)]
        assert len(vector_calls) >= 1
    
    @patch('src.graph_engine.GraphDatabase')
    def test_retrieve_by_similarity_fallback(self, mock_graphdb, mock_config):
        """Test similarity retrieval fallback when vector search fails."""
        # Setup mocks
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        
        # Mock vector search failure, then successful fallback
        mock_session.run.side_effect = [
            Exception("Vector index not available"),  # First call fails
            []  # Second call (fallback) succeeds with empty result
        ]
        
        mock_graphdb.driver.return_value = mock_driver
        
        engine = GraphMemoryEngine(mock_config)
        
        # Test similarity retrieval
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        import asyncio
        memories = asyncio.run(engine.retrieve_by_similarity(query_embedding, limit=10))
        
        # Should return empty list but not crash
        assert memories == []
        
        # Should have attempted both vector search and fallback
        assert mock_session.run.call_count >= 2
    
    @patch('src.graph_engine.GraphDatabase')
    def test_retrieve_by_graph_walk(self, mock_graphdb, mock_config):
        """Test graph traversal retrieval."""
        # Setup mocks
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        
        # Mock graph walk result
        mock_node = {
            'id': 'related-id',
            'category': 'critical',
            'content': 'Related content',
            'structured_data': '{}',
            'confidence': 0.85,
            'stability': 0.9,
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 3,
            'decay_rate': 0.001,
            'embedding': None
        }
        
        mock_record = Mock()
        mock_record.__getitem__.return_value = mock_node
        mock_session.run.return_value = [mock_record]
        
        mock_graphdb.driver.return_value = mock_driver
        
        engine = GraphMemoryEngine(mock_config)
        
        # Test graph walk
        start_nodes = ["start-id-1", "start-id-2"]
        import asyncio
        memories = asyncio.run(engine.retrieve_by_graph_walk(start_nodes, max_depth=3, min_weight=0.1))
        
        # Verify results
        assert len(memories) == 1
        assert memories[0].id == 'related-id'
        assert memories[0].category == MemoryCategory.CRITICAL
        
        # Verify graph traversal query was called
        graph_calls = [call for call in mock_session.run.call_args_list 
                      if 'MATCH path =' in str(call)]
        assert len(graph_calls) >= 1
    
    @patch('src.graph_engine.GraphDatabase')
    def test_update_access_metrics(self, mock_graphdb, mock_config):
        """Test updating access metrics for a memory."""
        # Setup mocks
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        mock_session.run.return_value = None
        
        mock_graphdb.driver.return_value = mock_driver
        
        engine = GraphMemoryEngine(mock_config)
        
        # Test access metrics update
        import asyncio
        asyncio.run(engine.update_access_metrics("test-memory-id"))
        
        # Verify UPDATE query was called
        update_calls = [call for call in mock_session.run.call_args_list 
                       if 'SET m.last_accessed' in str(call)]
        assert len(update_calls) >= 1
    
    @patch('src.graph_engine.GraphDatabase')
    def test_apply_decay(self, mock_graphdb, mock_config):
        """Test applying decay to memory confidence scores."""
        # Setup mocks
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        mock_session.run.return_value = None
        
        mock_graphdb.driver.return_value = mock_driver
        
        engine = GraphMemoryEngine(mock_config)
        
        # Test decay application
        memory_ids = ["id1", "id2", "id3"]
        import asyncio
        asyncio.run(engine.apply_decay(memory_ids))
        
        # Verify decay query was called
        decay_calls = [call for call in mock_session.run.call_args_list 
                      if 'SET m.confidence' in str(call)]
        assert len(decay_calls) >= 1
    
    @patch('src.graph_engine.GraphDatabase')
    def test_archive_low_confidence(self, mock_graphdb, mock_config):
        """Test archiving low-confidence memories."""
        # Setup mocks
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        
        # Mock count query result
        mock_count_record = Mock()
        mock_count_record.__getitem__.return_value = 5
        mock_count_result = Mock()
        mock_count_result.single.return_value = mock_count_record
        
        # Mock archive query result
        mock_archive_result = Mock()
        
        mock_session.run.side_effect = [mock_count_result, mock_archive_result]
        
        mock_graphdb.driver.return_value = mock_driver
        
        engine = GraphMemoryEngine(mock_config)
        
        # Test archival
        import asyncio
        archived_count = asyncio.run(engine.archive_low_confidence(threshold=0.1))
        
        # Verify correct count returned
        assert archived_count == 5
        
        # Verify both count and archive queries were called
        assert mock_session.run.call_count >= 2
    
    @patch('src.graph_engine.GraphDatabase')
    def test_get_by_category(self, mock_graphdb, mock_config):
        """Test retrieving memories by category."""
        # Setup mocks
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        
        # Mock category query result
        mock_node = {
            'id': 'category-id',
            'category': 'critical',
            'content': 'Category content',
            'structured_data': '{}',
            'confidence': 0.9,
            'stability': 0.85,
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'access_count': 2,
            'decay_rate': 0.001,
            'embedding': None
        }
        
        mock_record = Mock()
        mock_record.__getitem__.return_value = mock_node
        mock_session.run.return_value = [mock_record]
        
        mock_graphdb.driver.return_value = mock_driver
        
        engine = GraphMemoryEngine(mock_config)
        
        # Test category retrieval
        import asyncio
        memories = asyncio.run(engine.get_by_category(MemoryCategory.CRITICAL, limit=50))
        
        # Verify results
        assert len(memories) == 1
        assert memories[0].id == 'category-id'
        assert memories[0].category == MemoryCategory.CRITICAL
        
        # Verify category query was called
        category_calls = [call for call in mock_session.run.call_args_list 
                         if 'WHERE m.category' in str(call)]
        assert len(category_calls) >= 1
    
    def test_get_relationship_type_conversion(self, mock_config):
        """Test LinkType to Neo4j relationship type conversion."""
        with patch('src.graph_engine.GraphDatabase'):
            engine = GraphMemoryEngine(mock_config)
            
            assert engine._get_relationship_type(LinkType.STRENGTHEN) == "STRENGTHENS"
            assert engine._get_relationship_type(LinkType.REPLACE) == "REPLACES"
            assert engine._get_relationship_type(LinkType.CONTRADICT) == "CONTRADICTS"
    
    def test_get_initial_decay_rate(self, mock_config):
        """Test decay policy to rate conversion."""
        with patch('src.graph_engine.GraphDatabase'):
            engine = GraphMemoryEngine(mock_config)
            
            assert engine._get_initial_decay_rate(DecayPolicy.NO_DECAY) == 0.0
            assert engine._get_initial_decay_rate(DecayPolicy.VERY_SLOW) == 0.001
            assert engine._get_initial_decay_rate(DecayPolicy.MEDIUM) == 0.02
            assert engine._get_initial_decay_rate(DecayPolicy.FAST) == 0.20
    
    def test_node_to_memory_conversion(self, mock_config):
        """Test conversion from Neo4j node to MemoryNode object."""
        with patch('src.graph_engine.GraphDatabase'):
            engine = GraphMemoryEngine(mock_config)
            
            # Create mock Neo4j node
            now = datetime.now()
            mock_node = {
                'id': 'test-id',
                'category': 'episodic',
                'content': 'Test content',
                'structured_data': '{"facts": ["test fact"], "preferences": ["coffee"]}',
                'confidence': 0.9,
                'stability': 0.8,
                'created_at': now.isoformat(),
                'last_accessed': now.isoformat(),
                'access_count': 5,
                'decay_rate': 0.02,
                'embedding': [0.1, 0.2, 0.3]
            }
            
            memory = engine._node_to_memory(mock_node)
            
            assert memory.id == 'test-id'
            assert memory.category == MemoryCategory.EPISODIC
            assert memory.content == 'Test content'
            assert memory.structured_data == {"facts": ["test fact"], "preferences": ["coffee"]}
            assert memory.confidence == 0.9
            assert memory.stability == 0.8
            assert memory.access_count == 5
            assert memory.decay_rate == 0.02
            assert memory.embedding == [0.1, 0.2, 0.3]
    
    def test_node_to_memory_with_invalid_json(self, mock_config):
        """Test node conversion with invalid JSON in structured_data."""
        with patch('src.graph_engine.GraphDatabase'):
            engine = GraphMemoryEngine(mock_config)
            
            # Create mock node with invalid JSON
            now = datetime.now()
            mock_node = {
                'id': 'test-id',
                'category': 'episodic',
                'content': 'Test content',
                'structured_data': 'invalid json {',
                'confidence': 0.9,
                'stability': 0.8,
                'created_at': now.isoformat(),
                'last_accessed': now.isoformat(),
                'access_count': 5,
                'decay_rate': 0.02,
                'embedding': None
            }
            
            memory = engine._node_to_memory(mock_node)
            
            # Should handle gracefully with empty dict
            assert memory.structured_data == {}
            assert memory.id == 'test-id'
    
    @patch('src.graph_engine.GraphDatabase')
    def test_context_manager_usage(self, mock_graphdb, mock_config):
        """Test using GraphMemoryEngine as context manager."""
        mock_driver = Mock()
        mock_session = Mock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None
        mock_session.run.return_value = None
        
        mock_graphdb.driver.return_value = mock_driver
        
        # Test context manager
        with GraphMemoryEngine(mock_config) as engine:
            assert engine is not None
        
        # Verify close was called
        mock_driver.close.assert_called_once()


class TestGraphMemoryEngineIntegration:
    """Integration tests that require a running Neo4j instance."""
    
    def test_real_neo4j_connection(self):
        """Test connection to real Neo4j instance."""
        # Load config from environment
        from src.config import ConfigManager
        import os
        
        # Load configuration from .env file
        config = ConfigManager.load()
        neo4j_config = config.neo4j
        
        try:
            with GraphMemoryEngine(neo4j_config) as engine:
                # Test basic connection
                assert engine.driver is not None
                
                # Test schema creation (should not raise errors)
                # Schema is created in __init__, so if we get here it worked
                
                # Test a simple query to verify connection works
                with engine.driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    record = result.single()
                    assert record["test"] == 1
                
        except Exception as e:
            pytest.fail(f"Neo4j connection failed: {e}")
    
    def test_real_neo4j_schema_creation(self):
        """Test that schema creation works with real Neo4j."""
        from src.config import ConfigManager
        
        config = ConfigManager.load()
        neo4j_config = config.neo4j
        
        try:
            with GraphMemoryEngine(neo4j_config) as engine:
                # Verify indexes were created by checking they exist
                with engine.driver.session() as session:
                    # Check for memory_id_unique constraint
                    result = session.run("SHOW CONSTRAINTS")
                    constraints = [record["name"] for record in result if "memory_id_unique" in record["name"]]
                    assert len(constraints) > 0, "memory_id_unique constraint not found"
                    
                    # Check for category index
                    result = session.run("SHOW INDEXES")
                    indexes = [record["name"] for record in result if "memory_category_idx" in record["name"]]
                    assert len(indexes) > 0, "memory_category_idx index not found"
                
        except Exception as e:
            pytest.fail(f"Schema verification failed: {e}")