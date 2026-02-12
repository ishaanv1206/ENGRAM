"""
Decay Manager for automated memory lifecycle management.

This module implements the DecayManager class which applies category-specific
decay policies to memories, adjusts decay rates based on usage patterns,
and archives low-confidence memories.
"""

import asyncio
import logging
import math
from datetime import datetime
from typing import Dict

from .graph_engine import GraphMemoryEngine
from .models import MemoryCategory, MemoryNode


logger = logging.getLogger(__name__)


class DecayManager:
    """
    Manages memory decay and lifecycle policies.
    
    Applies category-specific decay rates to memories, adjusts decay based on
    usage frequency, and archives memories that fall below confidence thresholds.
    Runs as a periodic background task.
    """
    
    def __init__(self, graph_engine: GraphMemoryEngine):
        """
        Initialize the Decay Manager.
        
        Args:
            graph_engine: GraphMemoryEngine instance for memory operations.
        """
        self.graph_engine = graph_engine
        
        # Define decay policies for each category (per day rates)
        self.decay_policies: Dict[MemoryCategory, float] = {
            MemoryCategory.PINNED: 0.0,        # No decay
            MemoryCategory.CRITICAL: 0.005,    # 0.5% per day
            MemoryCategory.EPISODIC: 0.05,     # 5% per day
            MemoryCategory.TEMPORARY: 0.28,    # 28% per day
            MemoryCategory.RELATIONAL: 0.05,   # 5% per day (same as episodic)
        }
        
        self._running = False
        self._task = None
        
        logger.info("Decay Manager initialized with category-specific policies")
    
    async def start(self) -> None:
        """
        Start periodic decay application.
        
        Runs every hour to apply decay policies to all active memories.
        This is a long-running background task that should be started
        with asyncio.create_task().
        """
        self._running = True
        logger.info("Starting Decay Manager background task (runs every hour)")
        
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self.apply_decay()
            except asyncio.CancelledError:
                logger.info("Decay Manager task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in decay manager loop: {e}", exc_info=True)
                # Continue running despite errors
    
    async def stop(self) -> None:
        """Stop the periodic decay application."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Decay Manager stopped")
    
    async def apply_decay(self) -> None:
        """
        Apply decay to all active memories based on their category policies.
        
        For each category with a non-zero decay rate:
        1. Retrieve all memories in that category
        2. Calculate usage-adjusted decay rate for each memory
        3. Update confidence scores
        4. Archive memories that fall below threshold (0.1)
        """
        logger.info("Applying decay policies to all active memories")
        
        total_processed = 0
        total_archived = 0
        
        for category, base_rate in self.decay_policies.items():
            if base_rate == 0.0:
                # Skip categories with no decay (e.g., PINNED)
                continue
            
            try:
                # Get all memories of this category
                memories = await self.graph_engine.get_by_category(category, limit=1000)
                
                if not memories:
                    continue
                
                logger.debug(f"Processing {len(memories)} memories in category {category.value}")
                
                # Process each memory
                memories_to_update = []
                for memory in memories:
                    # Calculate usage-based decay reduction
                    usage_factor = self._calculate_usage_factor(memory)
                    
                    # Adjust decay rate based on usage
                    adjusted_rate = base_rate * (1.0 - usage_factor)
                    
                    # Convert daily rate to hourly rate
                    hourly_rate = adjusted_rate / 24.0
                    
                    # Calculate new confidence
                    new_confidence = memory.confidence * (1.0 - hourly_rate)
                    new_confidence = max(0.0, new_confidence)  # Ensure non-negative
                    
                    # Update the memory's decay rate and confidence
                    await self._update_memory_confidence(memory.id, new_confidence, adjusted_rate)
                    
                    total_processed += 1
                    
                    # Check if memory should be archived
                    if new_confidence < 0.1:
                        logger.debug(f"Memory {memory.id} confidence dropped to {new_confidence:.3f}, will be archived")
                
            except Exception as e:
                logger.error(f"Error processing decay for category {category.value}: {e}", exc_info=True)
        
        # Archive all low-confidence memories
        try:
            archived_count = await self.graph_engine.archive_low_confidence(threshold=0.1)
            total_archived = archived_count
            logger.info(f"Decay cycle complete: processed {total_processed} memories, archived {total_archived}")
        except Exception as e:
            logger.error(f"Error archiving low-confidence memories: {e}", exc_info=True)
    
    def _calculate_usage_factor(self, memory: MemoryNode) -> float:
        """
        Calculate usage-based decay reduction factor.
        
        More frequently accessed memories decay slower. The usage factor
        represents the percentage reduction in decay rate (0.0 to 0.9).
        
        Args:
            memory: MemoryNode to calculate usage factor for.
            
        Returns:
            float: Usage factor between 0.0 and 0.9 (90% max reduction).
        """
        # Calculate days since creation
        days_since_creation = (datetime.now() - memory.created_at).days
        if days_since_creation == 0:
            days_since_creation = 1  # Avoid division by zero
        
        # Calculate average accesses per day
        accesses_per_day = memory.access_count / days_since_creation
        
        # Logarithmic scaling: 
        # - 1 access/day = ~0.3 reduction (30% slower decay)
        # - 10 accesses/day = ~0.6 reduction (60% slower decay)
        # - 100+ accesses/day = ~0.9 reduction (90% slower decay, max)
        if accesses_per_day > 0:
            factor = min(0.3 * math.log10(accesses_per_day + 1), 0.9)
        else:
            factor = 0.0
        
        return factor
    
    async def _update_memory_confidence(self, memory_id: str, new_confidence: float, 
                                       new_decay_rate: float) -> None:
        """
        Update a memory's confidence score and decay rate in the database.
        
        Args:
            memory_id: ID of the memory to update.
            new_confidence: New confidence score (0.0 to 1.0).
            new_decay_rate: New adjusted decay rate.
        """
        try:
            # Use the graph engine's update method (NetworkX compatible)
            await self.graph_engine.update_memory_confidence(memory_id, new_confidence, new_decay_rate)
        except Exception as e:
            logger.error(f"Failed to update confidence for memory {memory_id}: {e}")
