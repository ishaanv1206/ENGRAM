"""
Unit tests for Pinned Memory Manager.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from src.pinned_memory import (
    PinnedMemoryManager, MemoryBudgetExceeded, 
    create_default_pinned_memory, initialize_pinned_memory_manager
)


class TestPinnedMemoryManager:
    """Test cases for PinnedMemoryManager."""
    
    def test_initialization_with_new_file(self):
        """Test initialization with a new storage file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            
            manager = PinnedMemoryManager(storage_path)
            
            assert manager.storage_path == Path(storage_path)
            assert manager.max_size_bytes == 2048
            assert len(manager) == 0
            assert manager.get_size_bytes() == 0
            assert manager.get_available_bytes() == 2048
    
    def test_initialization_with_existing_file(self):
        """Test initialization with existing storage file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            
            # Create existing file with data
            existing_data = {
                "language": "English",
                "style": "Professional"
            }
            with open(storage_path, 'w') as f:
                json.dump(existing_data, f)
            
            manager = PinnedMemoryManager(storage_path)
            
            assert len(manager) == 2
            assert manager.get("language") == "English"
            assert manager.get("style") == "Professional"
    
    def test_get_and_set_operations(self):
        """Test basic get and set operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            manager = PinnedMemoryManager(storage_path)
            
            # Test get on empty cache
            assert manager.get("language") is None
            
            # Test set and get
            manager.set("language", "English (US)")
            assert manager.get("language") == "English (US)"
            
            # Test category replacement
            manager.set("language", "Spanish")
            assert manager.get("language") == "Spanish"
            assert len(manager) == 1
    
    def test_size_limit_enforcement(self):
        """Test that 2KB size limit is enforced."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            manager = PinnedMemoryManager(storage_path)
            
            # Create content that exceeds 2KB
            large_content = "x" * 2100  # Larger than 2KB
            
            with pytest.raises(MemoryBudgetExceeded) as exc_info:
                manager.set("large_category", large_content)
            
            assert "exceeds 2048 byte limit" in str(exc_info.value)
            assert len(manager) == 0  # Should not have stored anything
    
    def test_size_calculation_with_unicode(self):
        """Test size calculation with Unicode characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            manager = PinnedMemoryManager(storage_path)
            
            # Unicode characters take more bytes in UTF-8
            unicode_content = "Hello 世界 🌍"  # Mix of ASCII, Chinese, and emoji
            manager.set("unicode", unicode_content)
            
            # Verify size calculation includes UTF-8 encoding
            expected_size = len("unicode".encode('utf-8')) + len(unicode_content.encode('utf-8'))
            assert manager.get_size_bytes() == expected_size
    
    def test_get_all_operation(self):
        """Test get_all returns copy of all memories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            manager = PinnedMemoryManager(storage_path)
            
            # Add multiple entries
            manager.set("language", "English")
            manager.set("style", "Professional")
            manager.set("timezone", "UTC")
            
            all_memories = manager.get_all()
            
            assert len(all_memories) == 3
            assert all_memories["language"] == "English"
            assert all_memories["style"] == "Professional"
            assert all_memories["timezone"] == "UTC"
            
            # Verify it's a copy (modifying returned dict doesn't affect manager)
            all_memories["new_key"] = "new_value"
            assert "new_key" not in manager
    
    def test_remove_operation(self):
        """Test removing pinned memory categories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            manager = PinnedMemoryManager(storage_path)
            
            # Add entries
            manager.set("language", "English")
            manager.set("style", "Professional")
            
            # Remove existing entry
            result = manager.remove("language")
            assert result is True
            assert manager.get("language") is None
            assert len(manager) == 1
            
            # Remove non-existent entry
            result = manager.remove("nonexistent")
            assert result is False
            assert len(manager) == 1
    
    def test_clear_operation(self):
        """Test clearing all pinned memories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            manager = PinnedMemoryManager(storage_path)
            
            # Add entries
            manager.set("language", "English")
            manager.set("style", "Professional")
            manager.set("timezone", "UTC")
            
            assert len(manager) == 3
            
            # Clear all
            manager.clear()
            
            assert len(manager) == 0
            assert manager.get_size_bytes() == 0
            assert manager.get_all() == {}
    
    def test_persistence_across_instances(self):
        """Test that data persists across manager instances."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            
            # Create first instance and add data
            manager1 = PinnedMemoryManager(storage_path)
            manager1.set("language", "English")
            manager1.set("style", "Professional")
            
            # Create second instance - should load existing data
            manager2 = PinnedMemoryManager(storage_path)
            
            assert len(manager2) == 2
            assert manager2.get("language") == "English"
            assert manager2.get("style") == "Professional"
    
    def test_contains_and_len_operations(self):
        """Test __contains__ and __len__ magic methods."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            manager = PinnedMemoryManager(storage_path)
            
            # Test empty manager
            assert len(manager) == 0
            assert "language" not in manager
            
            # Add entries
            manager.set("language", "English")
            manager.set("style", "Professional")
            
            # Test with entries
            assert len(manager) == 2
            assert "language" in manager
            assert "style" in manager
            assert "nonexistent" not in manager
    
    def test_get_stats_operation(self):
        """Test statistics reporting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            manager = PinnedMemoryManager(storage_path)
            
            # Add some data
            manager.set("language", "English")
            manager.set("style", "Professional")
            
            stats = manager.get_stats()
            
            assert stats['total_categories'] == 2
            assert stats['size_bytes'] > 0
            assert stats['available_bytes'] < 2048
            assert stats['max_size_bytes'] == 2048
            assert stats['utilization_percent'] > 0
            assert "language" in stats['categories']
            assert "style" in stats['categories']
            assert stats['storage_path'] == str(Path(storage_path))
            assert 'last_updated' in stats
    
    def test_repr_method(self):
        """Test string representation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            manager = PinnedMemoryManager(storage_path)
            
            manager.set("language", "English")
            
            repr_str = repr(manager)
            assert "PinnedMemoryManager" in repr_str
            assert "categories=1" in repr_str
            assert "2048 bytes" in repr_str
    
    def test_corrupted_file_handling(self):
        """Test handling of corrupted storage files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            
            # Create corrupted JSON file
            with open(storage_path, 'w') as f:
                f.write("invalid json content {")
            
            # Should handle gracefully and start with empty cache
            manager = PinnedMemoryManager(storage_path)
            
            assert len(manager) == 0
            assert manager.get_size_bytes() == 0
    
    def test_non_dict_file_handling(self):
        """Test handling of files with non-dict content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            
            # Create file with list instead of dict
            with open(storage_path, 'w') as f:
                json.dump(["not", "a", "dict"], f)
            
            # Should handle gracefully and start with empty cache
            manager = PinnedMemoryManager(storage_path)
            
            assert len(manager) == 0
            assert manager.get_size_bytes() == 0


class TestUtilityFunctions:
    """Test utility functions for pinned memory."""
    
    def test_create_default_pinned_memory(self):
        """Test creation of default pinned memory entries."""
        defaults = create_default_pinned_memory()
        
        assert isinstance(defaults, dict)
        assert "language" in defaults
        assert "style" in defaults
        assert "safety" in defaults
        assert "timezone" in defaults
        assert "persona" in defaults
        
        # Verify reasonable default values
        assert defaults["language"] == "English (US)"
        assert "UTC" in defaults["timezone"]
        assert "ai assistant" in defaults["persona"].lower()
    
    def test_initialize_pinned_memory_manager_with_defaults(self):
        """Test initialization with default values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            
            manager = initialize_pinned_memory_manager(storage_path, use_defaults=True)
            
            # Should have default entries
            assert len(manager) > 0
            assert manager.get("language") is not None
            assert manager.get("style") is not None
            assert manager.get("safety") is not None
    
    def test_initialize_pinned_memory_manager_without_defaults(self):
        """Test initialization without default values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            
            manager = initialize_pinned_memory_manager(storage_path, use_defaults=False)
            
            # Should be empty
            assert len(manager) == 0
    
    def test_initialize_with_existing_data(self):
        """Test initialization when data already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = os.path.join(temp_dir, "pinned.json")
            
            # Create existing data
            existing_data = {"custom": "value"}
            with open(storage_path, 'w') as f:
                json.dump(existing_data, f)
            
            manager = initialize_pinned_memory_manager(storage_path, use_defaults=True)
            
            # Should keep existing data, not add defaults
            assert len(manager) == 1
            assert manager.get("custom") == "value"
            assert manager.get("language") is None  # No defaults added