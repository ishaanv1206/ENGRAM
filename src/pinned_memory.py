"""
Pinned Memory Manager for ultra-critical persistent instructions.

This module implements the Tier 0 cache for pinned memories that never decay
and provide O(1) access to critical system constraints like language, style,
safety rules, timezone, and persona information.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
import threading
import asyncio
from datetime import datetime


class MemoryBudgetExceeded(Exception):
    """Raised when pinned memory exceeds the 2KB size limit."""
    pass


class PinnedMemoryManager:
    """
    Ultra-fast in-memory cache for critical persistent instructions.
    
    Provides O(1) access to pinned memories with a 2KB size limit.
    Supports category replacement and persistent storage.
    """
    
    def __init__(self, storage_path: str):
        """
        Initialize the pinned memory manager.
        
        Args:
            storage_path: Path to the JSON file for persistent storage
        """
        self.storage_path = Path(storage_path)
        self.cache: Dict[str, str] = {}
        self.max_size_bytes = 2048  # 2KB limit
        self._lock = threading.RLock()  # Thread-safe operations
        
        # Ensure storage directory exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing data on initialization
        self.load()
    
    def get(self, category: str) -> Optional[str]:
        """
        Get pinned memory for a specific category with O(1) access.
        
        Args:
            category: The memory category (e.g., 'language', 'style', 'safety')
            
        Returns:
            The content for the category, or None if not found
        """
        with self._lock:
            return self.cache.get(category)
    
    def set(self, category: str, content: str) -> None:
        """
        Set pinned memory for a category, replacing any existing entry.
        
        Args:
            category: The memory category
            content: The content to store
            
        Raises:
            MemoryBudgetExceeded: If the total size would exceed 2KB
        """
        with self._lock:
            # Calculate size with the new entry
            temp_cache = self.cache.copy()
            temp_cache[category] = content
            
            if self._calculate_size(temp_cache) > self.max_size_bytes:
                raise MemoryBudgetExceeded(
                    f"Pinned memory exceeds {self.max_size_bytes} byte limit. "
                    f"Current size would be {self._calculate_size(temp_cache)} bytes."
                )
            
            # Update cache and persist
            self.cache[category] = content
            self._persist()

    def add(self, category: str, content: str) -> None:
        """
        Add content to a pinned memory category, appending if it exists.
        
        Args:
            category: The memory category
            content: The content to add
            
        Raises:
            MemoryBudgetExceeded: If the total size would exceed 2KB
        """
        with self._lock:
            current_content = self.cache.get(category)
            if current_content:
                # Append with a separator (e.g., newline)
                new_content = f"{current_content}\n{content}"
            else:
                new_content = content
            
            self.set(category, new_content)
    
    def get_all(self) -> Dict[str, str]:
        """
        Return all pinned memories as a dictionary copy.
        
        Returns:
            Dictionary mapping categories to their content
        """
        with self._lock:
            return self.cache.copy()
    
    def remove(self, category: str) -> bool:
        """
        Remove a pinned memory category.
        
        Args:
            category: The category to remove
            
        Returns:
            True if the category was removed, False if it didn't exist
        """
        with self._lock:
            if category in self.cache:
                del self.cache[category]
                self._persist()
                return True
            return False
    
    def clear(self) -> None:
        """Clear all pinned memories."""
        with self._lock:
            self.cache.clear()
            self._persist()
    
    def get_size_bytes(self) -> int:
        """
        Get the current total size of pinned memories in bytes.
        
        Returns:
            Total size in bytes
        """
        with self._lock:
            return self._calculate_size(self.cache)
    
    def get_available_bytes(self) -> int:
        """
        Get the remaining available space in bytes.
        
        Returns:
            Available space in bytes
        """
        return self.max_size_bytes - self.get_size_bytes()
    
    def load(self) -> None:
        """
        Load pinned memories from persistent storage.
        
        Creates an empty cache if the storage file doesn't exist.
        """
        with self._lock:
            try:
                if self.storage_path.exists():
                    with open(self.storage_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Validate the loaded data
                    if isinstance(data, dict):
                        # Ensure all values are strings
                        self.cache = {k: str(v) for k, v in data.items()}
                        
                        # Check size limit
                        if self._calculate_size(self.cache) > self.max_size_bytes:
                            raise MemoryBudgetExceeded(
                                f"Loaded pinned memory exceeds {self.max_size_bytes} byte limit"
                            )
                    else:
                        self.cache = {}
                else:
                    # Initialize with empty cache
                    self.cache = {}
                    
            except (json.JSONDecodeError, IOError, MemoryBudgetExceeded) as e:
                # If loading fails, start with empty cache and log the error
                self.cache = {}
                print(f"Warning: Failed to load pinned memory from {self.storage_path}: {e}")
    
    def _persist(self) -> None:
        """
        Persist the current cache to disk storage.
        
        This method is called automatically after any modification.
        """
        try:
            # Create a backup of the current file if it exists
            if self.storage_path.exists():
                backup_path = self.storage_path.with_suffix('.bak')
                self.storage_path.rename(backup_path)
            
            # Write the new data
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
            
            # Remove backup on successful write
            backup_path = self.storage_path.with_suffix('.bak')
            if backup_path.exists():
                backup_path.unlink()
                
        except IOError as e:
            # Restore backup if write failed
            backup_path = self.storage_path.with_suffix('.bak')
            if backup_path.exists():
                backup_path.rename(self.storage_path)
            raise IOError(f"Failed to persist pinned memory: {e}")
    
    def _calculate_size(self, cache_dict: Optional[Dict[str, str]] = None) -> int:
        """
        Calculate the total size of the cache in bytes.
        
        Args:
            cache_dict: Dictionary to calculate size for (defaults to self.cache)
            
        Returns:
            Total size in bytes (UTF-8 encoding)
        """
        if cache_dict is None:
            cache_dict = self.cache
            
        total_size = 0
        for key, value in cache_dict.items():
            # Calculate UTF-8 byte size for both key and value
            total_size += len(key.encode('utf-8'))
            total_size += len(value.encode('utf-8'))
        
        return total_size
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the pinned memory manager.
        
        Returns:
            Dictionary with statistics including size, count, and categories
        """
        with self._lock:
            return {
                'total_categories': len(self.cache),
                'size_bytes': self.get_size_bytes(),
                'available_bytes': self.get_available_bytes(),
                'max_size_bytes': self.max_size_bytes,
                'utilization_percent': (self.get_size_bytes() / self.max_size_bytes) * 100,
                'categories': list(self.cache.keys()),
                'storage_path': str(self.storage_path),
                'last_updated': datetime.now().isoformat()
            }
    
    def __len__(self) -> int:
        """Return the number of pinned memory categories."""
        return len(self.cache)
    
    def __contains__(self, category: str) -> bool:
        """Check if a category exists in pinned memory."""
        return category in self.cache
    
    def __repr__(self) -> str:
        """String representation of the pinned memory manager."""
        return (f"PinnedMemoryManager(categories={len(self.cache)}, "
                f"size={self.get_size_bytes()}/{self.max_size_bytes} bytes)")


# Utility functions for common pinned memory operations

def create_default_pinned_memory() -> Dict[str, str]:
    """
    Create a dictionary with default pinned memory entries.
    
    Returns:
        Dictionary with default categories and content
    """
    return {
        'language': 'English (US)',
        'style': 'Professional but friendly, concise responses',
        'safety': 'Never provide medical/legal advice, refuse harmful requests',
        'timezone': 'UTC',
        'persona': 'Helpful AI assistant'
    }


def initialize_pinned_memory_manager(storage_path: str, 
                                   use_defaults: bool = True) -> PinnedMemoryManager:
    """
    Initialize a pinned memory manager with optional default values.
    
    Args:
        storage_path: Path to the storage file
        use_defaults: Whether to populate with default values if empty
        
    Returns:
        Initialized PinnedMemoryManager instance
    """
    manager = PinnedMemoryManager(storage_path)
    
    # If empty and defaults requested, populate with default values
    if use_defaults and len(manager) == 0:
        defaults = create_default_pinned_memory()
        for category, content in defaults.items():
            try:
                manager.set(category, content)
            except MemoryBudgetExceeded:
                # If defaults don't fit, add what we can
                break
    
    return manager