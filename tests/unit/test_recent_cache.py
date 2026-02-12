"""
Unit tests for Recent Memory Cache (Tier 1).
"""

import pytest
from datetime import datetime
from src.recent_cache import RecentMemoryCache
from src.models import MemoryNode, MemoryCategory, create_memory_node


class TestRecentMemoryCache:
    """Test cases for RecentMemoryCache."""
    
    def test_initialization(self):
        """Test cache initialization with default and custom sizes."""
        # Default size
        cache = RecentMemoryCache()
        assert cache.max_size == 100
        assert cache.size() == 0
        assert not cache.is_full()
        
        # Custom size
        cache = RecentMemoryCache(max_size=50)
        assert cache.max_size == 50
        assert cache.size() == 0
    
    def test_put_and_get_single_memory(self):
        """Test basic put and get operations."""
        cache = RecentMemoryCache(max_size=5)
        
        # Create a test memory
        memory = create_memory_node(
            content="Test memory content",
            category=MemoryCategory.EPISODIC,
            confidence=0.9
        )
        
        # Put memory in cache
        cache.put(memory)
        assert cache.size() == 1
        
        # Get memory from cache
        retrieved = cache.get(memory.id)
        assert retrieved is not None
        assert retrieved.id == memory.id
        assert retrieved.content == "Test memory content"
        assert retrieved.confidence == 0.9
    
    def test_get_nonexistent_memory(self):
        """Test getting a memory that doesn't exist."""
        cache = RecentMemoryCache()
        
        result = cache.get("nonexistent_id")
        assert result is None
    
    def test_lru_promotion_on_get(self):
        """Test that get() promotes memory to most recent."""
        cache = RecentMemoryCache(max_size=3)
        
        # Add three memories
        memory1 = create_memory_node("Memory 1", MemoryCategory.EPISODIC)
        memory2 = create_memory_node("Memory 2", MemoryCategory.EPISODIC)
        memory3 = create_memory_node("Memory 3", MemoryCategory.EPISODIC)
        
        cache.put(memory1)
        cache.put(memory2)
        cache.put(memory3)
        
        # Order should be: memory1 (oldest), memory2, memory3 (newest)
        ids = cache.get_all_ids()
        assert ids == [memory1.id, memory2.id, memory3.id]
        
        # Access memory1 - should promote it to most recent
        retrieved = cache.get(memory1.id)
        assert retrieved.id == memory1.id
        
        # Order should now be: memory2 (oldest), memory3, memory1 (newest)
        ids = cache.get_all_ids()
        assert ids == [memory2.id, memory3.id, memory1.id]
    
    def test_lru_eviction_when_full(self):
        """Test LRU eviction when cache reaches max capacity."""
        cache = RecentMemoryCache(max_size=3)
        
        # Fill cache to capacity
        memory1 = create_memory_node("Memory 1", MemoryCategory.EPISODIC)
        memory2 = create_memory_node("Memory 2", MemoryCategory.EPISODIC)
        memory3 = create_memory_node("Memory 3", MemoryCategory.EPISODIC)
        
        cache.put(memory1)
        cache.put(memory2)
        cache.put(memory3)
        
        assert cache.size() == 3
        assert cache.is_full()
        
        # Add fourth memory - should evict memory1 (oldest)
        memory4 = create_memory_node("Memory 4", MemoryCategory.EPISODIC)
        cache.put(memory4)
        
        assert cache.size() == 3  # Still at max capacity
        assert cache.get(memory1.id) is None  # memory1 should be evicted
        assert cache.get(memory2.id) is not None  # memory2 should still exist
        assert cache.get(memory3.id) is not None  # memory3 should still exist
        assert cache.get(memory4.id) is not None  # memory4 should be present
    
    def test_update_existing_memory(self):
        """Test updating an existing memory in cache."""
        cache = RecentMemoryCache(max_size=3)
        
        # Add initial memory
        memory = create_memory_node("Original content", MemoryCategory.EPISODIC)
        cache.put(memory)
        
        # Add other memories
        memory2 = create_memory_node("Memory 2", MemoryCategory.EPISODIC)
        memory3 = create_memory_node("Memory 3", MemoryCategory.EPISODIC)
        cache.put(memory2)
        cache.put(memory3)
        
        # Order: memory, memory2, memory3
        ids = cache.get_all_ids()
        assert ids == [memory.id, memory2.id, memory3.id]
        
        # Update the first memory (should promote to most recent)
        updated_memory = create_memory_node("Updated content", MemoryCategory.EPISODIC)
        updated_memory.id = memory.id  # Same ID
        cache.put(updated_memory)
        
        # Should still have 3 items, but order changed
        assert cache.size() == 3
        ids = cache.get_all_ids()
        assert ids == [memory2.id, memory3.id, memory.id]  # memory promoted to end
        
        # Verify content was updated
        retrieved = cache.get(memory.id)
        assert retrieved.content == "Updated content"
    
    def test_clear_cache(self):
        """Test clearing all entries from cache."""
        cache = RecentMemoryCache()
        
        # Add some memories
        for i in range(5):
            memory = create_memory_node(f"Memory {i}", MemoryCategory.EPISODIC)
            cache.put(memory)
        
        assert cache.size() == 5
        
        # Clear cache
        cache.clear()
        
        assert cache.size() == 0
        assert not cache.is_full()
        assert cache.get_all_ids() == []
    
    def test_contains_method(self):
        """Test contains() method without promotion."""
        cache = RecentMemoryCache(max_size=3)
        
        memory1 = create_memory_node("Memory 1", MemoryCategory.EPISODIC)
        memory2 = create_memory_node("Memory 2", MemoryCategory.EPISODIC)
        
        cache.put(memory1)
        cache.put(memory2)
        
        # Test contains
        assert cache.contains(memory1.id)
        assert cache.contains(memory2.id)
        assert not cache.contains("nonexistent_id")
        
        # Verify order didn't change (no promotion)
        ids = cache.get_all_ids()
        assert ids == [memory1.id, memory2.id]
    
    def test_peek_lru_and_mru(self):
        """Test peek methods for least and most recently used."""
        cache = RecentMemoryCache()
        
        # Empty cache
        assert cache.peek_lru() is None
        assert cache.peek_mru() is None
        
        # Add memories
        memory1 = create_memory_node("Memory 1", MemoryCategory.EPISODIC)
        memory2 = create_memory_node("Memory 2", MemoryCategory.EPISODIC)
        memory3 = create_memory_node("Memory 3", MemoryCategory.EPISODIC)
        
        cache.put(memory1)
        cache.put(memory2)
        cache.put(memory3)
        
        # Check LRU and MRU
        lru = cache.peek_lru()
        mru = cache.peek_mru()
        
        assert lru is not None
        assert lru.id == memory1.id  # First added = least recent
        assert mru is not None
        assert mru.id == memory3.id  # Last added = most recent
        
        # Verify order didn't change (no promotion)
        ids = cache.get_all_ids()
        assert ids == [memory1.id, memory2.id, memory3.id]
    
    def test_cache_with_different_memory_categories(self):
        """Test cache behavior with different memory categories."""
        cache = RecentMemoryCache(max_size=5)
        
        # Add memories of different categories
        critical = create_memory_node("Critical info", MemoryCategory.CRITICAL)
        episodic = create_memory_node("Episodic info", MemoryCategory.EPISODIC)
        relational = create_memory_node("Relational info", MemoryCategory.RELATIONAL)
        temporary = create_memory_node("Temporary info", MemoryCategory.TEMPORARY)
        
        cache.put(critical)
        cache.put(episodic)
        cache.put(relational)
        cache.put(temporary)
        
        assert cache.size() == 4
        
        # All should be retrievable regardless of category
        assert cache.get(critical.id) is not None
        assert cache.get(episodic.id) is not None
        assert cache.get(relational.id) is not None
        assert cache.get(temporary.id) is not None
    
    def test_large_cache_operations(self):
        """Test cache operations with larger number of entries."""
        cache = RecentMemoryCache(max_size=100)
        
        memories = []
        # Add exactly 100 memories
        for i in range(100):
            memory = create_memory_node(f"Memory {i}", MemoryCategory.EPISODIC)
            memories.append(memory)
            cache.put(memory)
        
        assert cache.size() == 100
        assert cache.is_full()
        
        # All memories should be retrievable
        for memory in memories:
            assert cache.get(memory.id) is not None
        
        # Add one more - should evict the first one
        extra_memory = create_memory_node("Extra memory", MemoryCategory.EPISODIC)
        cache.put(extra_memory)
        
        assert cache.size() == 100  # Still at max
        assert cache.get(memories[0].id) is None  # First memory evicted
        assert cache.get(extra_memory.id) is not None  # New memory present
    
    def test_memory_node_with_all_fields(self):
        """Test cache with fully populated MemoryNode objects."""
        cache = RecentMemoryCache()
        
        # Create memory with all fields populated
        now = datetime.now()
        memory = MemoryNode(
            id="test_id_123",
            category=MemoryCategory.CRITICAL,
            content="Detailed memory content",
            structured_data={"preferences": ["coffee"], "facts": ["user likes morning meetings"]},
            confidence=0.95,
            stability=0.8,
            created_at=now,
            last_accessed=now,
            access_count=5,
            decay_rate=0.01,
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        
        cache.put(memory)
        retrieved = cache.get("test_id_123")
        
        assert retrieved is not None
        assert retrieved.id == "test_id_123"
        assert retrieved.category == MemoryCategory.CRITICAL
        assert retrieved.content == "Detailed memory content"
        assert retrieved.structured_data == {"preferences": ["coffee"], "facts": ["user likes morning meetings"]}
        assert retrieved.confidence == 0.95
        assert retrieved.stability == 0.8
        assert retrieved.access_count == 5
        assert retrieved.decay_rate == 0.01
        assert retrieved.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]