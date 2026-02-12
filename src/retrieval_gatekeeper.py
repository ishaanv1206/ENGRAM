"""
Retrieval Gatekeeper for intelligent memory retrieval.

This module implements the RetrievalGatekeeper class that decides when, how much,
and which memory to retrieve based on query intent, context, and budget constraints.
It orchestrates multi-tier cache access, graph queries, and multi-factor scoring
to provide relevant memories within latency constraints.
"""

import logging
import math
import time
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

from .models import (
    MemoryNode, QueryIntent, MemoryBudget, RetrievalResult,
    ConversationContext, MemoryCategory, create_memory_budget
)
from .memory_analyzer import MemoryAnalyzer
from .pinned_memory import PinnedMemoryManager
from .recent_cache import RecentMemoryCache
from .graph_engine import GraphMemoryEngine


logger = logging.getLogger(__name__)


class RetrievalGatekeeper:
    """
    Intelligent memory retrieval orchestrator.
    
    Decides when, how much, and which memory to retrieve based on query analysis,
    context, and budget constraints. Coordinates access across multiple cache tiers
    and applies multi-factor scoring for relevance ranking.
    """
    
    def __init__(
        self,
        analyzer: MemoryAnalyzer,
        pinned_mgr: PinnedMemoryManager,
        recent_cache: RecentMemoryCache,
        graph_engine: GraphMemoryEngine,
        embedding_service: 'EmbeddingService'
    ):
        """
        Initialize the Retrieval Gatekeeper.
        
        Args:
            analyzer: Memory analyzer for query intent detection
            pinned_mgr: Pinned memory manager (Tier 0)
            recent_cache: Recent memory cache (Tier 1)
            graph_engine: Graph memory engine (Tier 2)
            embedding_service: Embedding service for semantic search
        """
        self.analyzer = analyzer
        self.pinned_mgr = pinned_mgr
        self.recent_cache = recent_cache
        self.graph_engine = graph_engine
        self.embedding_service = embedding_service
    
    async def retrieve(
        self,
        query: str,
        context: ConversationContext
    ) -> RetrievalResult:
        """
        Main retrieval orchestration method.
        
        Coordinates the full retrieval pipeline:
        1. Detect query intent and estimate memory demand
        2. Calculate memory budget based on intent and context
        3. Retrieve from pinned memory (Tier 0)
        4. Check recent cache (Tier 1)
        5. Query graph if needed (Tier 2)
        6. Score and rank all candidates
        7. Apply top-K pruning to stay within budget
        
        Args:
            query: User query text
            context: Current conversation context
            
        Returns:
            RetrievalResult: Retrieved memories with metadata
        """
        start_time = time.time()
        
        logger.debug(f"Starting retrieval for query: {query[:100]}...")
        
        # Step 1: Detect query intent and estimate memory demand
        intent = await self._detect_intent(query, context)
        logger.debug(f"Detected query intent: {intent}")
        
        # Step 2: Calculate memory budget based on intent and context
        budget = self._calculate_budget(intent, context)
        logger.debug(f"Memory budget: {budget.max_memories} memories, {budget.max_tokens} tokens")
        print(f"DEBUG: Intent={intent}, Budget={budget.max_memories}")
        
        # Step 3: Always include pinned memories (Tier 0)
        pinned = self.pinned_mgr.get_all()
        logger.debug(f"Retrieved {len(pinned)} pinned memory categories")
        
        # Step 4: Check recent cache (Tier 1)
        cached = self._check_cache(query, budget)
        logger.debug(f"Found {len(cached)} memories in cache")
        
        # Step 5: Query graph if needed (Tier 2)
        # FORCE graph query for explicit retrieval intents, or if cache is insufficient
        explicit_retrieval_intents = {
            QueryIntent.FACTUAL_RECALL, 
            QueryIntent.RELATIONSHIP, 
            QueryIntent.PREFERENCE_CHECK
        }
        
        # Check Cache Quality - If best cached memory is weak, force graph
        # This prevents getting stuck with "okay" but not "great" memories in cache
        cache_quality_low = False
        if cached:
            # Quick score check for cache
            scored_cache = self._score_memories(cached, query, context)
            if scored_cache:
                best_score = scored_cache[0][1]
                logger.debug(f"Best cache score: {best_score:.4f}")
                if best_score < 0.7:
                    cache_quality_low = True
                    logger.info(f"Cache quality low ({best_score:.4f} < 0.7), forcing graph search")
        
        should_query_graph = (intent in explicit_retrieval_intents) or \
                             (len(cached) < budget.max_memories and intent != QueryIntent.NO_MEMORY) or \
                             cache_quality_low
        
        # Log decision logic
        logger.debug(f"Checking graph? Intent={intent}, Cached={len(cached)}, Budget={budget.max_memories}, QualityLow={cache_quality_low} -> {should_query_graph}")
        
        if should_query_graph:
            logger.debug("Querying Graph based on intent or cache miss")
            graph_results = await self._query_graph(query, budget, context)
            logger.debug(f"Retrieved {len(graph_results)} memories from graph")
        else:
            logger.debug("Skipping Graph (Intent is General/Phatic or Cache is sufficient)")
            graph_results = []
        
        # Step 6: Score and rank all candidates
        candidates = cached + graph_results
        scored = self._score_memories(candidates, query, context)
        logger.debug(f"Scored {len(scored)} candidate memories")
        
        # Step 7: Apply top-K pruning to stay within budget
        selected = self._prune_to_budget(scored, budget)
        logger.debug(f"Selected {len(selected)} memories after pruning")
        
        # Calculate total tokens
        total_tokens = self._estimate_tokens(pinned, selected)
        
        # Calculate retrieval time
        retrieval_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Retrieval completed in {retrieval_time_ms:.2f}ms: "
            f"{len(selected)} memories, {total_tokens} tokens"
        )
        
        return RetrievalResult(
            pinned=pinned,
            memories=selected,
            total_tokens=total_tokens,
            retrieval_time_ms=retrieval_time_ms,
            query_intent=intent
        )
    
    async def _detect_intent(
        self,
        query: str,
        context: ConversationContext
    ) -> QueryIntent:
        """
        Detect query intent using SLM classification.
        
        Uses the Small Language Model to classify query intent into one of five categories.
        Falls back to pattern matching only if SLM is unavailable.
        
        Args:
            query: User query text
            context: Conversation context
            
        Returns:
            QueryIntent: Detected intent category
        """
        # Use SLM classification if available
        if self.analyzer.is_available():
            try:
                return await self._detect_intent_with_slm(query, context)
            except Exception as e:
                logger.warning(f"SLM intent detection failed, using pattern fallback: {e}")
        
        # Fallback to pattern-based detection only if SLM unavailable
        logger.info("SLM not available, using pattern-based intent detection")
        return self._detect_intent_with_patterns(query)
    
    async def _detect_intent_with_slm(
        self,
        query: str,
        context: ConversationContext
    ) -> QueryIntent:
        """
        Use SLM to classify query intent.
        
        Args:
            query: User query text
            context: Conversation context
            
        Returns:
            QueryIntent: Detected intent category
        """
        prompt = f"""Classify this query into ONE category:

Query: "{query}"

Categories:
1. FACTUAL_RECALL - asking about past information ("What did I say?", "When did I?")
2. PREFERENCE_CHECK - asking about preferences ("Do I like?", "Am I?")
3. RELATIONSHIP - asking about relationships ("How is X related to Y?")
4. NO_MEMORY - greetings/acknowledgments ("Hello", "Thanks", "OK")
5. GENERAL - general conversation

Answer with just the category name:"""
        
        # Use the analyzer's LLM for classification
        import asyncio
        loop = asyncio.get_running_loop()
        
        t0 = time.time()
        response = await loop.run_in_executor(
            None,
            lambda: self.analyzer.llm(
                prompt,
                max_tokens=30,
                temperature=0.0,  # Deterministic
                stop=["\n"],
                echo=False
            )
        )
        t1 = time.time()
        logger.debug(f"SLM intent detection took {(t1-t0)*1000:.2f}ms")
        
        # Parse response
        intent_text = response['choices'][0]['text'].strip().upper()
        logger.debug(f"SLM intent response: '{intent_text}' for query: '{query}'")
        
        # Map to QueryIntent enum with fuzzy matching
        if "FACTUAL" in intent_text or "RECALL" in intent_text:
            return QueryIntent.FACTUAL_RECALL
        elif "PREFERENCE" in intent_text or "CHECK" in intent_text:
            return QueryIntent.PREFERENCE_CHECK
        elif "RELATIONSHIP" in intent_text or "RELATION" in intent_text:
            return QueryIntent.RELATIONSHIP
        elif "NO_MEMORY" in intent_text or "NO MEMORY" in intent_text:
            return QueryIntent.NO_MEMORY
        elif "GENERAL" in intent_text:
            return QueryIntent.GENERAL
        
        # If SLM response is unclear, fall back to pattern matching
        logger.warning(f"SLM gave unclear intent '{intent_text}', using pattern fallback")
        return self._detect_intent_with_patterns(query)
    
    def _detect_intent_with_patterns(self, query: str) -> QueryIntent:
        """
        Pattern-based intent detection fallback.
        
        Args:
            query: User query text
            
        Returns:
            QueryIntent: Detected intent category
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Pattern-based intent detection for common cases
        factual_patterns = [
            "what did i", "did i say", "did i mention", "what was",
            "when did", "where did", "who did", "recall", "remember",
            "when is", "what is", "who is", "where is", # Added for present/future facts
            "who do", "what do" # Added for active queries
        ]
        
        preference_patterns = [
            "do i like", "do i prefer", "am i", "my preference",
            "my favorite", "i like", "i prefer", "who am i", "my name"
        ]
        
        relationship_patterns = [
            "how is", "related to", "connection between", "relationship",
            "who is", "what is the relationship"
        ]
        
        # Single-word patterns that should match whole words only
        no_memory_words = {
            "hello", "hi", "hey", "thanks", "ok", "okay",
            "yes", "no", "sure", "alright", "bye", "goodbye"
        }
        
        # Multi-word patterns for no_memory
        no_memory_phrases = ["thank you"]
        
        # Check patterns
        if any(pattern in query_lower for pattern in factual_patterns):
            return QueryIntent.FACTUAL_RECALL
        
        if any(pattern in query_lower for pattern in preference_patterns):
            return QueryIntent.PREFERENCE_CHECK
        
        if any(pattern in query_lower for pattern in relationship_patterns):
            return QueryIntent.RELATIONSHIP
        
        # Check NO_MEMORY: must be short query with exact word match
        if len(query.split()) < 5:
            # Check if any no_memory word matches exactly
            if query_words & no_memory_words:
                return QueryIntent.NO_MEMORY
            # Check multi-word phrases
            if any(phrase in query_lower for phrase in no_memory_phrases):
                return QueryIntent.NO_MEMORY
        
        # Default to GENERAL for queries that don't match patterns
        return QueryIntent.GENERAL
    
    def _calculate_budget(
        self,
        intent: QueryIntent,
        context: ConversationContext
    ) -> MemoryBudget:
        """
        Calculate retrieval budget based on query complexity and context.
        
        Args:
            intent: Detected query intent
            context: Conversation context
            
        Returns:
            MemoryBudget: Calculated budget constraints
        """
        # Use the utility function from models
        budget = create_memory_budget(intent, context.turn_count)
        
        # Additional adjustments based on context
        if len(context.active_entities) > 10:
            # More entities = potentially more relevant memories
            budget.max_memories = int(budget.max_memories * 1.1)
        
        if len(context.recent_topics) > 5:
            # More topics = broader context needed
            budget.max_tokens = int(budget.max_tokens * 1.1)
        
        return budget
    
    def _check_cache(
        self,
        query: str,
        budget: MemoryBudget
    ) -> List[MemoryNode]:
        """
        Check Tier 1 cache for relevant memories.
        
        Args:
            query: User query text
            budget: Memory budget constraints
            
        Returns:
            List[MemoryNode]: Cached memories (may be empty)
        """
        cached_memories = []
        
        # Get all cached memory IDs
        cached_ids = self.recent_cache.get_all_ids()
        
        # Retrieve each cached memory
        for memory_id in cached_ids:
            memory = self.recent_cache.get(memory_id)
            if memory and memory.confidence > 0.1:
                cached_memories.append(memory)
            
            # Stop if we have enough from cache
            if len(cached_memories) >= budget.max_memories:
                break
        
        return cached_memories
    
    async def _query_graph(
        self,
        query: str,
        budget: MemoryBudget,
        context: ConversationContext
    ) -> List[MemoryNode]:
        """
        Query graph database for relevant memories.
        
        Args:
            query: User query text
            budget: Memory budget constraints
            context: Conversation context
            
        Returns:
            List[MemoryNode]: Memories from graph database
        """
        try:
            # Step 5: Query graph if needed (Tier 2)
            # Try text-based retrieval first (works without embeddings)
            memories = await self.graph_engine.retrieve_by_text_match(
                query=query,
                limit=budget.max_memories * 2  # Get more candidates for scoring
            )
            
            logger.debug(f"Text-based retrieval found {len(memories)} memories")
            
            # If text match returns nothing, try embedding-based retrieval
            if not memories:
                query_embedding = self._get_query_embedding(query)
                
                # Try hybrid retrieval
                memories = await self.graph_engine.retrieve_hybrid(
                    query=query,
                    query_embedding=query_embedding,
                    limit=budget.max_memories * 2
                )
                
                # If hybrid fails, try similarity search
                if not memories:
                    memories = await self.graph_engine.retrieve_by_similarity(
                        query_embedding=query_embedding,
                        limit=budget.max_memories * 2
                    )

            # --- DEPTH-2 TRAVERSAL ---
            # For the top 3 most relevant memories, fetch their related memories (Depth 1 from them = Depth 2 from query)
            if memories:
                related_candidates = []
                # Sort by relevance if available, otherwise just take first few
                top_memories = memories[:3] 
                
                for mem in top_memories:
                    related = await self.graph_engine.retrieve_related(mem.id, depth=1, limit=3)
                    if related:
                        print(f"DEBUG: Found {len(related)} related memories for {mem.id}")
                        related_candidates.extend(related)
                
                # Deduplicate and merge
                existing_ids = {m.id for m in memories}
                for rm in related_candidates:
                    if rm.id not in existing_ids:
                        memories.append(rm)
                        existing_ids.add(rm.id)

            # Promote retrieved memories to cache
            for memory in memories[:budget.max_memories]:
                self.recent_cache.put(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return []
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """
        Get embedding vector for query using the embedding service.
        
        Args:
            query: Query text
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            if self.embedding_service and self.embedding_service.is_available():
                return self.embedding_service.embed(query)
            else:
                logger.warning("Embedding service not available, using zero vector")
                return [0.0] * 1024  # mxbai-embed-large-v1 dimension
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return [0.0] * 1024
    
    def _score_memories(
        self,
        candidates: List[MemoryNode],
        query: str,
        context: ConversationContext
    ) -> List[Tuple[MemoryNode, float]]:
        """
        Apply multi-factor scoring algorithm to candidate memories.
        
        Scoring factors:
        - Semantic similarity (35%)
        - Recency (15%)
        - Confidence (20%)
        - Access frequency (10%)
        - Stability (10%)
        - Category priority (10%)
        
        Args:
            candidates: List of candidate memory nodes
            query: User query text
            context: Conversation context
            
        Returns:
            List[Tuple[MemoryNode, float]]: Scored memories (memory, score)
        """
        if not candidates:
            return []
        
        scored = []
        query_embedding = self._get_query_embedding(query)
        current_time = datetime.now()
        
        for memory in candidates:
            # Factor 1: Semantic similarity (0.0 to 1.0)
            similarity = self._calculate_similarity(query_embedding, memory.embedding)
            
            # Factor 2: Recency (exponential decay)
            age_days = (current_time - memory.last_accessed).days
            recency = math.exp(-age_days / 30.0)  # 30-day half-life
            
            # Factor 3: Confidence (0.0 to 1.0)
            confidence = memory.confidence
            
            # Factor 4: Access frequency (normalized)
            frequency = min(memory.access_count / 100.0, 1.0)
            
            # Factor 5: Stability (0.0 to 1.0)
            stability = memory.stability
            
            # Factor 6: Category priority
            category_weight = self._get_category_weight(memory.category)
            
            # Weighted combination (weights sum to 1.0)
            score = (
                similarity * 0.35 +
                recency * 0.15 +
                confidence * 0.20 +
                frequency * 0.10 +
                stability * 0.10 +
                category_weight * 0.10
            )
            
            scored.append((memory, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored
    
    def _calculate_similarity(
        self,
        query_embedding: List[float],
        memory_embedding: Optional[List[float]]
    ) -> float:
        """
        Calculate cosine similarity between query and memory embeddings.
        
        Args:
            query_embedding: Query embedding vector
            memory_embedding: Memory embedding vector (may be None)
            
        Returns:
            float: Similarity score (0.0 to 1.0)
        """
        if memory_embedding is None or len(memory_embedding) == 0:
            # Default similarity for memories without embeddings
            return 0.5
        
        # Ensure same dimensions
        min_len = min(len(query_embedding), len(memory_embedding))
        query_vec = query_embedding[:min_len]
        memory_vec = memory_embedding[:min_len]
        
        # Calculate cosine similarity
        dot_product = sum(q * m for q, m in zip(query_vec, memory_vec))
        query_norm = math.sqrt(sum(q * q for q in query_vec))
        memory_norm = math.sqrt(sum(m * m for m in memory_vec))
        
        if query_norm == 0 or memory_norm == 0:
            return 0.0
        
        similarity = dot_product / (query_norm * memory_norm)
        
        # Normalize to [0, 1] range (cosine similarity is in [-1, 1])
        return (similarity + 1.0) / 2.0
    
    def _get_category_weight(self, category: MemoryCategory) -> float:
        """
        Get priority weight for memory category.
        
        Args:
            category: Memory category
            
        Returns:
            float: Category weight (0.0 to 1.0)
        """
        category_weights = {
            MemoryCategory.PINNED: 1.0,
            MemoryCategory.CRITICAL: 1.0,
            MemoryCategory.RELATIONAL: 0.9,
            MemoryCategory.EPISODIC: 0.8,
            MemoryCategory.TEMPORARY: 0.5,
            MemoryCategory.DISCARD: 0.0,
        }
        return category_weights.get(category, 0.5)
    
    def _prune_to_budget(
        self,
        scored_memories: List[Tuple[MemoryNode, float]],
        budget: MemoryBudget
    ) -> List[MemoryNode]:
        """
        Apply top-K pruning to stay within memory budget.
        
        Args:
            scored_memories: List of (memory, score) tuples sorted by score
            budget: Memory budget constraints
            
        Returns:
            List[MemoryNode]: Top-K memories within budget
        """
        if not scored_memories:
            return []
        
        # Take top K memories
        selected = []
        total_tokens = 0
        
        for memory, score in scored_memories:
            # Estimate tokens for this memory
            memory_tokens = self._estimate_memory_tokens(memory)
            
            # Check if adding this memory would exceed budget
            if len(selected) >= budget.max_memories:
                break
            
            if total_tokens + memory_tokens > budget.max_tokens:
                # Try to fit smaller memories
                continue
            
            selected.append(memory)
            total_tokens += memory_tokens
        
        return selected
    
    def _estimate_tokens(
        self,
        pinned: Dict[str, str],
        memories: List[MemoryNode]
    ) -> int:
        """
        Estimate total token count for pinned and selected memories.
        
        Args:
            pinned: Pinned memory dictionary
            memories: List of selected memory nodes
            
        Returns:
            int: Estimated total token count
        """
        total_tokens = 0
        
        # Estimate pinned memory tokens
        for content in pinned.values():
            total_tokens += len(content.split()) * 1.3  # Rough token estimate
        
        # Estimate memory node tokens
        for memory in memories:
            total_tokens += self._estimate_memory_tokens(memory)
        
        return int(total_tokens)
    
    def _estimate_memory_tokens(self, memory: MemoryNode) -> int:
        """
        Estimate token count for a single memory node.
        
        Args:
            memory: Memory node
            
        Returns:
            int: Estimated token count
        """
        # Rough estimate: 1 token ≈ 0.75 words
        content_tokens = len(memory.content.split()) * 1.3
        
        # Add tokens for structured data (excluding embedding which is metadata)
        data_copy = memory.structured_data.copy()
        if 'embedding' in data_copy:
            del data_copy['embedding']
            
        structured_str = str(data_copy)
        structured_tokens = len(structured_str.split()) * 1.3
        
        return int(content_tokens + structured_tokens)
