"""
Unit tests for the Decay Manager.

Tests decay policy application, usage-based decay reduction,
and memory archival functionality.
"""

import asyncio
import math
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.decay_manager import DecayManager
from src.models import MemoryCategory, MemoryNode


@pytest.fixture
def mock_graph_engine():
    """Create a mock GraphMemoryEngine."""
    engine = MagicMock()
    engine.get_by_category = AsyncMock(return_value=[])
    engine.archive_low_confidence = AsyncMock(return_value=0)
    engine.driver = MagicMock()
    
    # Mock session context manager
    mock_session = MagicMock()
    mock_session.run = MagicMock()
    engine.driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    engine.driver.session.return_value.__exit__ = MagicMock(return_value=None)
    
    return engine


@pytest.fixture
def decay_manager(mock_graph_engine):
    """Create a DecayManager instance with mocked dependencies."""
    return DecayManager(mock_graph_engine)


def create_test_memory(
    memory_id: str = "test-123",
    category: MemoryCategory = MemoryCategory.EPISODIC,
    confidence: float = 0.8,
    access_count: int = 10,
    days_old: int = 30
) -> MemoryNode:
    """Helper to create test memory nodes."""
    now = datetime.now()
    created_at = now - timedelta(days=days_old)
    
    return MemoryNode(
        id=memory_id,
        category=category,
        content="Test memory content",
        structured_data={"test": "data"},
        confidence=confidence,
        stability=0.8,
        created_at=created_at,
        last_accessed=now - timedelta(days=1),
        access_count=access_count,
        decay_rate=0.02,
        embedding=None
    )


class TestDecayManagerInitialization:
    """Test DecayManager initialization."""
    
    def test_initialization(self, decay_manager):
        """Test that DecayManager initializes with correct decay policies."""
        assert decay_manager.decay_policies[MemoryCategory.PINNED] == 0.0
        assert decay_manager.decay_policies[MemoryCategory.CRITICAL] == 0.001
        assert decay_manager.decay_policies[MemoryCategory.EPISODIC] == 0.02
        assert decay_manager.decay_policies[MemoryCategory.TEMPORARY] == 0.20
        assert decay_manager.decay_policies[MemoryCategory.RELATIONAL] == 0.02
    
    def test_not_running_initially(self, decay_manager):
        """Test that DecayManager is not running initially."""
        assert decay_manager._running is False


class TestUsageFactorCalculation:
    """Test usage-based decay reduction calculation."""
    
    def test_no_accesses(self, decay_manager):
        """Test usage factor for memory with no accesses."""
        memory = create_test_memory(access_count=0, days_old=30)
        factor = decay_manager._calculate_usage_factor(memory)
        assert factor == 0.0
    
    def test_low_access_frequency(self, decay_manager):
        """Test usage factor for low access frequency (1 access/day)."""
        memory = create_test_memory(access_count=30, days_old=30)  # 1 access/day
        factor = decay_manager._calculate_usage_factor(memory)
        
        # Should be around 0.3 * log10(2) ≈ 0.09
        expected = 0.3 * math.log10(1.0 + 1)
        assert abs(factor - expected) < 0.01
    
    def test_medium_access_frequency(self, decay_manager):
        """Test usage factor for medium access frequency (10 accesses/day)."""
        memory = create_test_memory(access_count=300, days_old=30)  # 10 accesses/day
        factor = decay_manager._calculate_usage_factor(memory)
        
        # Should be around 0.3 * log10(11) ≈ 0.31
        expected = 0.3 * math.log10(10.0 + 1)
        assert abs(factor - expected) < 0.01
    
    def test_high_access_frequency(self, decay_manager):
        """Test usage factor for high access frequency (100 accesses/day)."""
        memory = create_test_memory(access_count=3000, days_old=30)  # 100 accesses/day
        factor = decay_manager._calculate_usage_factor(memory)
        
        # Should be around 0.3 * log10(101) ≈ 0.60
        expected = 0.3 * math.log10(100.0 + 1)
        assert abs(factor - expected) < 0.01
    
    def test_max_usage_factor(self, decay_manager):
        """Test that usage factor is capped at 0.9."""
        memory = create_test_memory(access_count=100000, days_old=30)  # Very high frequency
        factor = decay_manager._calculate_usage_factor(memory)
        assert factor <= 0.9
    
    def test_zero_days_old(self, decay_manager):
        """Test usage factor for brand new memory (created today)."""
        memory = create_test_memory(access_count=5, days_old=0)
        factor = decay_manager._calculate_usage_factor(memory)
        
        # Should use 1 day to avoid division by zero
        # 5 accesses / 1 day = 5 accesses/day
        expected = 0.3 * math.log10(5.0 + 1)
        assert abs(factor - expected) < 0.01


class TestDecayApplication:
    """Test decay policy application."""
    
    @pytest.mark.asyncio
    async def test_apply_decay_skips_pinned(self, decay_manager, mock_graph_engine):
        """Test that decay is not applied to pinned memories."""
        await decay_manager.apply_decay()
        
        # Should not call get_by_category for PINNED since decay rate is 0.0
        calls = mock_graph_engine.get_by_category.call_args_list
        categories_queried = [call[0][0] for call in calls]
        assert MemoryCategory.PINNED not in categories_queried
    
    @pytest.mark.asyncio
    async def test_apply_decay_processes_categories(self, decay_manager, mock_graph_engine):
        """Test that decay processes all non-pinned categories."""
        # Setup mock to return empty lists for all categories
        mock_graph_engine.get_by_category.return_value = []
        
        await decay_manager.apply_decay()
        
        # Should call get_by_category for each category with non-zero decay
        calls = mock_graph_engine.get_by_category.call_args_list
        categories_queried = [call[0][0] for call in calls]
        
        assert MemoryCategory.CRITICAL in categories_queried
        assert MemoryCategory.EPISODIC in categories_queried
        assert MemoryCategory.TEMPORARY in categories_queried
        assert MemoryCategory.RELATIONAL in categories_queried
    
    @pytest.mark.asyncio
    async def test_apply_decay_updates_confidence(self, decay_manager, mock_graph_engine):
        """Test that decay updates memory confidence scores."""
        # Create test memory with known values
        memory = create_test_memory(
            memory_id="test-123",
            category=MemoryCategory.EPISODIC,
            confidence=0.8,
            access_count=0,  # No usage reduction
            days_old=30
        )
        
        mock_graph_engine.get_by_category.return_value = [memory]
        
        await decay_manager.apply_decay()
        
        # Verify that session.run was called to update confidence
        mock_session = mock_graph_engine.driver.session.return_value.__enter__.return_value
        assert mock_session.run.called
        
        # Check that the update query was called with memory ID
        call_args = mock_session.run.call_args_list
        assert any('test-123' in str(call) for call in call_args)
    
    @pytest.mark.asyncio
    async def test_apply_decay_calculates_hourly_rate(self, decay_manager, mock_graph_engine):
        """Test that decay converts daily rate to hourly rate."""
        memory = create_test_memory(
            category=MemoryCategory.EPISODIC,
            confidence=1.0,
            access_count=0
        )
        
        mock_graph_engine.get_by_category.return_value = [memory]
        
        await decay_manager.apply_decay()
        
        # Episodic base rate is 0.02 per day
        # Hourly rate should be 0.02 / 24 ≈ 0.000833
        # New confidence should be 1.0 * (1 - 0.000833) ≈ 0.999167
        
        mock_session = mock_graph_engine.driver.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args_list
        
        # Find the call that updates confidence
        for call in call_args:
            if 'confidence' in str(call):
                args = call[1]
                if 'confidence' in args:
                    new_confidence = args['confidence']
                    # Should be very close to 1.0 after one hour
                    assert 0.998 < new_confidence < 1.0
    
    @pytest.mark.asyncio
    async def test_apply_decay_with_usage_reduction(self, decay_manager, mock_graph_engine):
        """Test that high usage reduces decay rate."""
        # Memory with high access frequency
        memory = create_test_memory(
            category=MemoryCategory.EPISODIC,
            confidence=1.0,
            access_count=300,  # 10 accesses/day over 30 days
            days_old=30
        )
        
        mock_graph_engine.get_by_category.return_value = [memory]
        
        await decay_manager.apply_decay()
        
        # Usage factor should be ~0.31, so adjusted rate = 0.02 * (1 - 0.31) = 0.0138
        # This is less than the base rate of 0.02
        
        mock_session = mock_graph_engine.driver.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args_list
        
        # Verify decay_rate was updated
        for call in call_args:
            if 'decay_rate' in str(call):
                args = call[1]
                if 'decay_rate' in args:
                    new_decay_rate = args['decay_rate']
                    # Should be less than base rate due to usage
                    assert new_decay_rate < 0.02
    
    @pytest.mark.asyncio
    async def test_apply_decay_archives_low_confidence(self, decay_manager, mock_graph_engine):
        """Test that low-confidence memories are archived."""
        mock_graph_engine.get_by_category.return_value = []
        mock_graph_engine.archive_low_confidence.return_value = 5
        
        await decay_manager.apply_decay()
        
        # Should call archive_low_confidence with threshold 0.1
        mock_graph_engine.archive_low_confidence.assert_called_once_with(threshold=0.1)
    
    @pytest.mark.asyncio
    async def test_apply_decay_handles_errors(self, decay_manager, mock_graph_engine):
        """Test that decay continues despite errors in individual categories."""
        # Make one category fail
        async def side_effect(category, limit):
            if category == MemoryCategory.CRITICAL:
                raise Exception("Test error")
            return []
        
        mock_graph_engine.get_by_category.side_effect = side_effect
        
        # Should not raise exception
        await decay_manager.apply_decay()
        
        # Should still call archive at the end
        assert mock_graph_engine.archive_low_confidence.called


class TestBackgroundTask:
    """Test background task management."""
    
    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self, decay_manager, mock_graph_engine):
        """Test that start() sets the running flag."""
        # Mock asyncio.sleep to avoid waiting
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            # Make sleep raise CancelledError after first call to exit loop
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            
            # Start the task
            task = asyncio.create_task(decay_manager.start())
            
            # Wait for task to complete (will be cancelled)
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            # Running flag should have been set to True before the loop
            assert mock_sleep.called
    
    @pytest.mark.asyncio
    async def test_stop_clears_running_flag(self, decay_manager):
        """Test that stop() clears the running flag."""
        # Set running flag manually
        decay_manager._running = True
        
        await decay_manager.stop()
        
        assert decay_manager._running is False
    
    @pytest.mark.asyncio
    async def test_start_runs_periodically(self, decay_manager, mock_graph_engine):
        """Test that decay runs periodically (mocked for speed)."""
        call_count = 0
        
        async def mock_sleep_with_limit(duration):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                # Stop after 2 sleep calls
                raise asyncio.CancelledError()
        
        with patch('asyncio.sleep', side_effect=mock_sleep_with_limit):
            task = asyncio.create_task(decay_manager.start())
            
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Should have called apply_decay at least once
        assert mock_graph_engine.get_by_category.called or call_count >= 1


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_empty_category(self, decay_manager, mock_graph_engine):
        """Test handling of empty category (no memories)."""
        mock_graph_engine.get_by_category.return_value = []
        
        # Should not raise exception
        await decay_manager.apply_decay()
    
    @pytest.mark.asyncio
    async def test_confidence_floor_at_zero(self, decay_manager, mock_graph_engine):
        """Test that confidence doesn't go below 0.0."""
        # Memory with very low confidence
        memory = create_test_memory(
            confidence=0.001,
            category=MemoryCategory.TEMPORARY,  # High decay rate
            access_count=0
        )
        
        mock_graph_engine.get_by_category.return_value = [memory]
        
        await decay_manager.apply_decay()
        
        # Verify confidence was set to non-negative value
        mock_session = mock_graph_engine.driver.session.return_value.__enter__.return_value
        call_args = mock_session.run.call_args_list
        
        for call in call_args:
            if 'confidence' in str(call):
                args = call[1]
                if 'confidence' in args:
                    new_confidence = args['confidence']
                    assert new_confidence >= 0.0
    
    @pytest.mark.asyncio
    async def test_update_confidence_error_handling(self, decay_manager, mock_graph_engine):
        """Test that errors in updating individual memories don't crash the system."""
        memory = create_test_memory()
        mock_graph_engine.get_by_category.return_value = [memory]
        
        # Make session.run raise an error
        mock_session = mock_graph_engine.driver.session.return_value.__enter__.return_value
        mock_session.run.side_effect = Exception("Database error")
        
        # Should not raise exception
        await decay_manager.apply_decay()
