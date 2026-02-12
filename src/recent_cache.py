"""
Recent Memory Cache (Tier 1) - LRU cache for frequently accessed memories.

This module implements a Least Recently Used (LRU) cache for storing the 100 most
recently accessed episodic memory entries. It provides fast O(1) access with
automatic eviction of least recently used entries when the cache is full.

Requirements addressed:
- 10.2: Tier 1 cache containing the 100 most recently accessed Episodic_Memory entries
- 10.6: LRU eviction when cache tiers are full
"""

from collections import OrderedDict
from typing import Optional
from .models import MemoryNode


class RecentMemoryCache:
    """
    LRU cache for recently accessed memory nodes.
    
    Maintains up to 100 memory entries with automatic eviction of least recently
    used entries when the cache is full. Provides O(1) access and promotion.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize the recent memory cache.
        
        Args:
            max_size: Maximum number of entries to store (default: 100)
        """
        self.cache: OrderedDict[str, MemoryNode] = OrderedDict()
        self.max_size = max_size
    
    def get(self, memory_id: str) -> Optional[MemoryNode]:
        """
        Get a memory node and promote it to most recent.
        
        Args:
            memory_id: Unique identifier of the memory node
            
        Returns:
            MemoryNode if found, None otherwise
            
        Time Complexity: O(1)
        """
        if memory_id in self.cache:
            # Move to end (most recent) and return
            self.cache.move_to_end(memory_id)
            return self.cache[memory_id]
        return None
    
    def put(self, memory: MemoryNode) -> None:
        """
        Add or update a memory node, evicting LRU entry if full.
        
        Args:
            memory: MemoryNode to store in cache
            
        Time Complexity: O(1)
        """
        memory_id = memory.id
        
        if memory_id in self.cache:
            # Update existing entry and move to end
            self.cache.move_to_end(memory_id)
        else:
            # Check if we need to evict before adding
            if len(self.cache) >= self.max_size:
                # Remove least recently used (first item)
                self.cache.popitem(last=False)
        
        # Add/update the memory (will be at the end = most recent)
        self.cache[memory_id] = memory
    
    def clear(self) -> None:
        """
        Clear all entries from the cache.
        
        Time Complexity: O(1)
        """
        self.cache.clear()
    
    def size(self) -> int:
        """
        Get the current number of entries in the cache.
        
        Returns:
            Number of entries currently stored
        """
        return len(self.cache)
    
    def is_full(self) -> bool:
        """
        Check if the cache is at maximum capacity.
        
        Returns:
            True if cache is full, False otherwise
        """
        return len(self.cache) >= self.max_size
    
    def contains(self, memory_id: str) -> bool:
        """
        Check if a memory ID exists in the cache without promoting it.
        
        Args:
            memory_id: Unique identifier to check
            
        Returns:
            True if memory exists in cache, False otherwise
        """
        return memory_id in self.cache
    
    def get_all_ids(self) -> list[str]:
        """
        Get all memory IDs in the cache, ordered from least to most recent.
        
        Returns:
            List of memory IDs in LRU order
        """
        return list(self.cache.keys())
    
    def peek_lru(self) -> Optional[MemoryNode]:
        """
        Get the least recently used memory without removing or promoting it.
        
        Returns:
            Least recently used MemoryNode, or None if cache is empty
        """
        if not self.cache:
            return None
        # First item is least recently used
        first_key = next(iter(self.cache))
        return self.cache[first_key]
    
    def peek_mru(self) -> Optional[MemoryNode]:
        """
        Get the most recently used memory without promoting it.
        
        Returns:
            Most recently used MemoryNode, or None if cache is empty
        """
        if not self.cache:
            return None
        # Last item is most recently used
        last_key = next(reversed(self.cache))
        return self.cache[last_key]