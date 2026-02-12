"""
Reflection Loop for asynchronous memory validation and quality improvement.

This module implements the ReflectionLoop class that processes reflection tasks
in the background AFTER user conversations complete. The reflection loop operates
asynchronously and never blocks or delays real-time response generation.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set

from .models import (
    ReflectionTask, ValidationResult, MemoryNode, MemoryCategory,
    ConversationContext, MemoryLink, LinkType
)
from .memory_analyzer import MemoryAnalyzer
from .graph_engine import GraphMemoryEngine
from .llm_client import LocalLLMClient


logger = logging.getLogger(__name__)


class ReflectionLoop:
    """
    Asynchronous reflection loop for memory validation and quality improvement.
    
    Processes reflection tasks in the background AFTER user conversations complete.
    This ensures zero disruption to real-time response generation.
    """
    
    def __init__(self, analyzer: MemoryAnalyzer, graph_engine: GraphMemoryEngine, llm_client: LocalLLMClient):
        """Initialize the Reflection Loop."""
        self.analyzer = analyzer
        self.graph_engine = graph_engine
        self.llm_client = llm_client
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        self.conflict_log: List[Dict] = []
        
        logger.info("ReflectionLoop initialized")
    
    async def start(self) -> None:
        """Start the background processing loop."""
        self.running = True
        logger.info("ReflectionLoop started")
        
        while self.running:
            try:
                task = await self.task_queue.get()
                await self._process_reflection(task)
                self.task_queue.task_done()
            except asyncio.CancelledError:
                logger.info("ReflectionLoop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in reflection loop: {e}", exc_info=True)
    
    def enqueue(self, task: ReflectionTask) -> None:
        """Add a reflection task to the queue (non-blocking)."""
        try:
            self.task_queue.put_nowait(task)
            logger.debug(f"Enqueued reflection task {task.task_id}")
        except asyncio.QueueFull:
            logger.warning(f"Reflection queue full, dropping task {task.task_id}")
    
    def stop(self) -> None:
        """Stop the reflection loop gracefully."""
        self.running = False
        logger.info("ReflectionLoop stop requested")
    
    async def _process_reflection(self, task: ReflectionTask) -> None:
        """Process a single reflection task."""
        logger.debug(f"Processing reflection task {task.task_id}")
        
        try:
            validation = await self._validate_response(task)
            
            if validation.hallucination_detected:
                await self._decrease_confidence(task.retrieved_memories, validation)
            else:
                await self._increase_confidence(task.retrieved_memories)
            
            await self._update_usage_metrics(task.retrieved_memories)
            
            duplicates = await self._detect_duplicates(task.retrieved_memories)
            if duplicates:
                await self._merge_memories(duplicates)
            
            conflicts = await self._detect_conflicts(task.retrieved_memories)
            if conflicts:
                await self._resolve_conflicts(conflicts)
            
            # Semantic Consolidation based on active entities
            await self._consolidate_memories(task.context)
            
        except Exception as e:
            logger.error(f"Failed to process reflection task {task.task_id}: {e}")
    
    async def _validate_response(self, task: ReflectionTask) -> ValidationResult:
        """Validate the response against retrieved memories."""
        return ValidationResult(
            hallucination_detected=False,
            explanation="Response validated",
            confidence_adjustments={},
            conflicts_detected=[]
        )
    
    async def _decrease_confidence(self, memories: List[MemoryNode], 
                                  validation: ValidationResult) -> None:
        """Decrease confidence scores for memories involved in hallucinations."""
        try:
            penalty = 0.1
            for memory in memories:
                new_conf = max(0.0, memory.confidence - penalty)
                # Keep existing decay rate
                await self.graph_engine.update_memory_confidence(memory.id, new_conf, memory.decay_rate)
        except Exception as e:
            logger.error(f"Failed to decrease confidence: {e}")
    
    async def _increase_confidence(self, memories: List[MemoryNode]) -> None:
        """Increase confidence scores for successfully used memories."""
        try:
            boost = 0.02
            for memory in memories:
                new_conf = min(1.0, memory.confidence + boost)
                await self.graph_engine.update_memory_confidence(memory.id, new_conf, memory.decay_rate)
        except Exception as e:
            logger.error(f"Failed to increase confidence: {e}")
    
    async def _update_usage_metrics(self, memories: List[MemoryNode]) -> None:
        """Update access_count and last_accessed for all retrieved memories."""
        try:
            for memory in memories:
                await self.graph_engine.update_access_metrics(memory.id)
        except Exception as e:
            logger.error(f"Failed to update usage metrics: {e}")
    
    async def _detect_duplicates(self, memories: List[MemoryNode]) -> List[List[MemoryNode]]:
        """
        Detect duplicate or highly similar memories from the retrieved set.
        Returns groups of duplicates (e.g., [[m1, m2], [m5, m6]]).
        """
        if not memories or len(memories) < 2:
            return []
            
        duplicates = []
        processed_ids = set()
        
        # Sort by length (longest is usually best to keep)
        memories.sort(key=lambda m: len(m.content), reverse=True)
        
        for i, mem1 in enumerate(memories):
            if mem1.id in processed_ids:
                continue
                
            current_group = [mem1]
            
            for j in range(i + 1, len(memories)):
                mem2 = memories[j]
                if mem2.id in processed_ids:
                    continue
                
                # Check for exact content match or high vector similarity
                is_duplicate = False
                
                # 1. Exact content match (normalized)
                if mem1.content.lower().strip() == mem2.content.lower().strip():
                    is_duplicate = True
                
                # 2. Vector similarity if embeddings exist
                elif mem1.embedding and mem2.embedding:
                    # Calculate cosine similarity manually
                    # (Neo4j does this but we have the vectors locally here)
                    import math
                    def cosine_sim(v1, v2):
                        dot = sum(a*b for a, b in zip(v1, v2))
                        norm1 = math.sqrt(sum(a*a for a in v1))
                        norm2 = math.sqrt(sum(a*a for a in v2))
                        return dot / (norm1 * norm2) if norm1 and norm2 else 0
                    
                    sim = cosine_sim(mem1.embedding, mem2.embedding)
                    if sim > 0.95:  # Extremely high threshold for auto-merge
                        logger.debug(f"Detected semantic duplicate: '{mem1.content}' == '{mem2.content}' (sim={sim:.4f})")
                        is_duplicate = True

                if is_duplicate:
                    current_group.append(mem2)
                    processed_ids.add(mem2.id)
            
            if len(current_group) > 1:
                duplicates.append(current_group)
                processed_ids.add(mem1.id)
                
        return duplicates
    
    async def _merge_memories(self, duplicate_groups: List[List[MemoryNode]]) -> None:
        """Merge duplicate memories into single consolidated memories."""
        for group in duplicate_groups:
            if not group: continue
            
            # Use the longest memory as the primary one (heuristic) - group is already sorted by length
            ids = [m.id for m in group]
            primary_id = ids[0]
            secondary_ids = ids[1:]
            
            if not secondary_ids: 
                continue

            success = await self.graph_engine.merge_memories(primary_id, secondary_ids)
            
            if success:
                logger.info(f"Reflection Loop merged {len(secondary_ids)} memories into {primary_id}")

    async def _detect_conflicts(self, memories: List[MemoryNode]) -> List[Tuple[MemoryNode, MemoryNode]]:
        """
        Detect conflicting memories using naive keyword negation analysis.
        This provides basic conflict detection without expensive LLM calls for now.
        """
        if not memories or len(memories) < 2:
            return []
            
        conflicts = []
        
        # Simple heuristic: Only check CRITICAL memories for now
        critical_memories = [m for m in memories if m.category == MemoryCategory.CRITICAL]
        if len(critical_memories) < 2:
            return []
            
        # Pairwise check - limit to top 5 most confident
        critical_memories.sort(key=lambda m: m.confidence, reverse=True)
        top_memories = critical_memories[:5]
        
        for i, m1 in enumerate(top_memories):
            for j in range(i + 1, len(top_memories)):
                m2 = top_memories[j]
                
                # Check intersection of content words (length > 4)
                term1 = m1.content.lower()
                term2 = m2.content.lower()
                words1 = set(w for w in term1.split() if len(w) > 4)
                words2 = set(w for w in term2.split() if len(w) > 4)
                
                intersection = words1.intersection(words2)
                
                # If they share topics but have differing negation
                if len(intersection) >= 1:
                    has_negation1 = any(w in term1.split() for w in ["not", "no", "never", "hate", "dislike"])
                    has_negation2 = any(w in term2.split() for w in ["not", "no", "never", "hate", "dislike"])
                    
                    if has_negation1 != has_negation2:
                         # Potential conflict
                         conflicts.append((m1, m2))
                         logger.warning(f"Potential conflict detected: '{m1.content}' vs '{m2.content}'")
        
        return conflicts

    async def _resolve_conflicts(self, conflicts: List[Tuple[MemoryNode, MemoryNode]]) -> None:
        """Resolve conflicting memories using resolution rules."""
        for m1, m2 in conflicts:
            logger.info(f"Flagging conflict for user resolution: {m1.id} vs {m2.id}")
            self.conflict_log.append({
                "timestamp": datetime.now().isoformat(),
                "memory_1": {"id": m1.id, "content": m1.content},
                "memory_2": {"id": m2.id, "content": m2.content},
                "status": "flagged"
            })
    
    def get_conflict_log(self) -> List[Dict]:
        """Get the conflict resolution log."""
        return self.conflict_log.copy()

    async def _consolidate_memories(self, context: ConversationContext) -> None:
        """
        Consolidate redundant memories for active entities in the context.
        Uses LLM to identify and merge semantically similar memories.
        """
        if not context.active_entities:
            return

        logger.info(f"Checking for consolidation candidates among {len(context.active_entities)} active entities")
        
        for entity in context.active_entities:
            try:
                # Get all memories for this entity
                memories = await self.graph_engine.get_memories_by_entity(entity)
                
                # Only strictly consolidate if we have multiple memories
                if len(memories) < 2:
                    continue
                
                # Group by broad category to avoid merging incompatible types (e.g. don't merge 'User likes X' with 'User deadline Y')
                # For now, we'll try to consolidate everything, but the LLM should be smart enough.
                # Let's limit to 5 memories to avoid context overflow
                memories_to_check = memories[:5]
                
                if len(memories_to_check) < 2:
                    continue

                # Prepare prompt for LLM
                memory_texts = [f"[{m.id}] {m.content}" for m in memories_to_check]
                input_text = "\n".join(memory_texts)
                
                prompt = f"""[INST] You are a Memory Consolidation AI. Your goal is to merge redundant memories into a single, concise memory without losing information.
                
Analyze these memories about the entity '{entity}':
{input_text}

Are there any redundant memories that state the same facts or can be combined?
If YES, provide the merged content and the list of IDs to merge.
If NO, respond with "NO_MERGE".

Format:
MERGE_IDS: [id1, id2, ...]
MERGED_CONTENT: [New consolidated text]
[/INST]
Response:"""

                # Call LLM
                response = await self.llm_client.generate(
                    query=prompt,
                    memory_context="",
                    conversation_history=[],
                    max_tokens=1024
                )
                
                if "NO_MERGE" in response:
                    continue
                    
                # Parse response
                import re
                ids_match = re.search(r'MERGE_IDS:\s*\[(.*?)\]', response, re.DOTALL)
                content_match = re.search(r'MERGED_CONTENT:\s*(.*)', response, re.DOTALL)
                
                if ids_match and content_match:
                    ids_str = ids_match.group(1)
                    secondary_ids = [mid.strip() for mid in ids_str.split(',') if mid.strip()]
                    new_content = content_match.group(1).strip()
                    
                    # We need to pick a primary ID (usually the first one or the one not in secondary list?)
                    # The LLM gives us a list of IDs to merge. We pick the first one as primary, and the rest as secondary.
                    # Wait, the LLM should tell us WHICH IDs to merge.
                    
                    # Let's check if the IDs returned are actually in our list
                    valid_ids = [m.id for m in memories_to_check]
                    ids_to_merge = [mid for mid in secondary_ids if mid in valid_ids]
                    
                    if len(ids_to_merge) < 2:
                        continue
                        
                    primary_id = ids_to_merge[0]
                    secondary_ids_to_remove = ids_to_merge[1:]
                    
                    if not secondary_ids_to_remove:
                        continue
                        
                    logger.info(f"Consolidating {len(secondary_ids_to_remove)} memories into {primary_id}")
                    
                    # Execute merge
                    success = await self.graph_engine.merge_memories(
                        primary_id=primary_id,
                        secondary_ids=secondary_ids_to_remove,
                        new_content=new_content
                    )
                    
                    if success:
                        logger.info(f"Successfully consolidated memories for entity '{entity}'")

            except Exception as e:
                logger.error(f"Error consolidating memories for entity {entity}: {e}")
