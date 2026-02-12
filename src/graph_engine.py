"""
Graph Memory Engine for NetworkX-based memory storage and retrieval.

This module implements the GraphMemoryEngine class that handles all interactions
with the NetworkX in-memory graph for storing, retrieving, and managing memory nodes
and their relationships. It supports vector similarity search (via numpy), graph traversal,
hybrid retrieval, and JSON persistence.
"""

import json
import logging
import math
import os
import networkx as nx
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
import uuid

from .models import (
    MemoryNode, MemoryExtraction, MemoryCategory, LinkType, 
    MemoryLink, ErrorResponse
)
from .config import GraphConfig


logger = logging.getLogger(__name__)


class GraphMemoryEngineError(Exception):
    """Base exception for Graph Memory Engine operations."""
    pass


_open = open

class GraphMemoryEngine:
    """
    NetworkX-based graph memory storage and retrieval engine.
    
    Handles memory nodes, relationships, vector similarity search (numpy),
    and JSON persistence.
    """
    
    def __init__(self, config: GraphConfig):
        """
        Initialize the Graph Memory Engine.
        
        Args:
            config: Graph configuration containing storage path.
        """
        self.config = config
        self.graph = nx.DiGraph()
        self.vector_index: Dict[str, np.ndarray] = {}  # ID -> Embedding Mapping
        self.bm25_index = None
        self.bm25_corpus: List[str] = []
        self.bm25_ids: List[str] = []
        
        # Load graph if exists
        self.load_graph()
        
    def close(self) -> None:
        """Close connection (noop for NetworkX, but saves graph)."""
        try:
            self.save_graph()
            logger.info("Graph Memory Engine closed")
        except Exception:
            # Ignore errors during shutdown (e.g. 'open' not defined)
            pass

    def load_graph(self) -> None:
        """Load graph from JSON storage."""
        if os.path.exists(self.config.storage_path):
            try:
                with _open(self.config.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.graph = nx.node_link_graph(data)
                    
                    # Rebuild vector index and convert attributes back to objects
                    self.vector_index = {}
                    for node_id, attrs in self.graph.nodes(data=True):
                        # Convert datetime strings back to datetime objects
                        if 'created_at' in attrs and isinstance(attrs['created_at'], str):
                            attrs['created_at'] = datetime.fromisoformat(attrs['created_at'])
                        if 'last_accessed' in attrs and isinstance(attrs['last_accessed'], str):
                            attrs['last_accessed'] = datetime.fromisoformat(attrs['last_accessed'])
                        
                        # Load embedding into vector index
                        if 'embedding' in attrs and attrs['embedding']:
                            self.vector_index[node_id] = np.array(attrs['embedding'], dtype=np.float32)
                            
                    self._rebuild_bm25_index()
                            
                logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            except Exception as e:
                logger.error(f"Failed to load graph: {e}")
                self.graph = nx.DiGraph()
        else:
            logger.info("No existing graph found, starting fresh.")
            self.graph = nx.DiGraph()

    def save_graph(self) -> None:
        """Save graph to JSON storage."""
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(self.config.storage_path), exist_ok=True)
            
            # Prepare data for serialization
            # NetworkX can't serialize datetime or numpy arrays directly
            graph_data = nx.node_link_data(self.graph)
            
            # Convert non-serializable objects in nodes
            for node in graph_data['nodes']:
                for k, v in node.items():
                    if isinstance(v, datetime):
                        node[k] = v.isoformat()
                    # Embedding is already a list in the node attributes (we kept it there for persistence)
            
            with _open(self.config.storage_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2)
            
            logger.debug("Graph saved successfully")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    def _rebuild_bm25_index(self):
        """Rebuild BM25 index from current graph content."""
        if BM25Okapi is None:
            logger.warning("rank_bm25 not installed, skipping BM25 index build")
            return

        self.bm25_ids = []
        self.bm25_corpus = []
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("label") == "Memory":
                content = attrs.get("content", "").lower()
                self.bm25_corpus.append(content.split()) # Simple whitespace tokenization
                self.bm25_ids.append(node_id)
        
        if self.bm25_corpus:
            self.bm25_index = BM25Okapi(self.bm25_corpus)
            logger.debug(f"Rebuilt BM25 index with {len(self.bm25_ids)} documents")
        else:
            self.bm25_index = None

    async def store_memory(self, extraction: MemoryExtraction) -> str:
        """
        Store a memory extraction in the graph.
        
        Args:
            extraction: MemoryExtraction object.
            
        Returns:
            str: ID of the stored memory node.
        """
        return await self.store_or_update_memory(extraction)

    async def store_or_update_memory(self, extraction: MemoryExtraction) -> str:
        """
        Store a new memory or update an existing one if a duplicate is detected.
        
        Args:
            extraction: MemoryExtraction object.
            
        Returns:
            str: ID of the stored or updated memory node.
        """
        print("DEBUG: Entering store_or_update_memory...")
        try:
            # 1. Check for duplicates/updates
            if extraction.structured_data.get('embedding'):
                similar_memories = await self.retrieve_by_similarity(
                    extraction.structured_data['embedding'], 
                    limit=1, 
                    min_score=0.90 # High threshold for duplication
                )
                
                if similar_memories:
                    existing = similar_memories[0]
                    # Check if category matches to be sure
                    if existing.category == extraction.category:
                         print(f"DEBUG: Found existing memory {existing.id} with similarity > 0.90. Updating content.")
                         
                         # UPDATE logic
                         attributes = self.graph.nodes[existing.id]
                         attributes['content'] = extraction.structured_data.get('content', '')
                         attributes['structured_data'] = json.dumps(extraction.structured_data)
                         attributes['confidence'] = extraction.confidence
                         attributes['last_accessed'] = datetime.now()
                         attributes['embedding'] = extraction.structured_data.get('embedding')
                         
                         # Update Vector Index
                         self.vector_index[existing.id] = np.array(extraction.structured_data['embedding'], dtype=np.float32)
                         
                         self.save_graph()
                         return existing.id
            
            # 2. Create NEW MemoryNode
            memory_id = str(uuid.uuid4())
            memory = MemoryNode(
                id=memory_id,
                category=extraction.category,
                content=extraction.structured_data.get('content', ''),
                structured_data=extraction.structured_data,
                confidence=extraction.confidence,
                stability=extraction.stability,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                embedding=extraction.structured_data.get('embedding')
            )
            
            # Add node to NetworkX
            attributes = {
                "label": "Memory",
                "category": memory.category.value,
                "content": memory.content,
                "confidence": memory.confidence,
                "created_at": memory.created_at,
                "last_accessed": memory.last_accessed,
                "access_count": memory.access_count,
                "decay_rate": memory.decay_rate,
                "structured_data": json.dumps(memory.structured_data), # Store as string for JSON
                "embedding": memory.embedding # Store list for JSON
            }
            
            self.graph.add_node(memory.id, **attributes)
            
            # Update Vector Index
            if memory.embedding:
                self.vector_index[memory.id] = np.array(memory.embedding, dtype=np.float32)
                print(f"DEBUG: Added memory {memory.id} to vector index. Total: {len(self.vector_index)}")
            else:
                print(f"DEBUG: Memory {memory.id} has NO EMBEDDING")
            
            # Auto-save Master + Sub-graphs
            self.save_graph()
            self.save_specialized_graphs()
            
            # Update BM25 (Naive full rebuild for now to ensure consistency)
            # Optimization: Append to corpus and re-init might be faster than loop, but BM25Okapi takes full corpus
            self._rebuild_bm25_index()
            
            return memory.id
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            raise GraphMemoryEngineError(f"Failed to add memory: {e}")

    async def retrieve_related(self, memory_id: str, depth: int = 1, limit: int = 5) -> List[MemoryNode]:
        """
        Retrieve memories related to a specific memory ID by traversing the graph.
        
        Args:
            memory_id: Starting memory ID.
            depth: Traversal depth (default 1).
            limit: Max memories to return.
            
        Returns:
            List[MemoryNode]: Related memories.
        """
        if not self.graph.has_node(memory_id):
            return []
            
        related_ids = set()
        
        # BFS traversal
        # We want successors (outgoing) and predecessors (incoming) usually?
        # NetworkX neighbors() gives successors in DiGraph. 
        # For semantic relations, we might want undirected traversal or checking both.
        # Let's use undirected view for traversal to find "connected" items.
        
        undirected_view = self.graph.to_undirected()
        
        try:
            # Get nodes within depth
            subgraph_nodes = nx.single_source_shortest_path_length(undirected_view, memory_id, cutoff=depth)
            
            for node_id, dist in subgraph_nodes.items():
                if node_id == memory_id:
                    continue # Skip self
                
                attrs = self.graph.nodes[node_id]
                if attrs.get('label') == 'Memory':
                    related_ids.add(node_id)
                    if len(related_ids) >= limit:
                        break
                        
        except Exception as e:
            logger.error(f"Traversal failed: {e}")
            
        results = []
        for mid in related_ids:
            if self.graph.has_node(mid):
                results.append(self._node_to_memory(mid, self.graph.nodes[mid]))
                
        return results

    async def retrieve_by_text_match(self, query: str, limit: int = 10) -> List[MemoryNode]:
        """Simple text containment search."""
        matches = []
        query_lower = query.lower()
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("label") == "Memory":
                content = attrs.get("content", "").lower()
                if query_lower in content:
                    matches.append(self._node_to_memory(node_id, attrs))
                    
        return matches[:limit]

    async def retrieve_by_similarity(self, query_embedding: List[float], limit: int = 10, min_score: float = 0.0) -> List[MemoryNode]:
        """Vector similarity search using numpy."""
        # print(f"DEBUG: retrieve_by_similarity. Index size: {len(self.vector_index)}") # Noisy log
        if not self.vector_index:
            return []
            
        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []
            
        scores = []
        for node_id, m_vec in self.vector_index.items():
            m_norm = np.linalg.norm(m_vec)
            if m_norm == 0:
                similarity = 0.0
            else:
                similarity = np.dot(q_vec, m_vec) / (q_norm * m_norm)
            
            if similarity >= min_score:
                scores.append((node_id, similarity))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for node_id, score in scores[:limit]:
            if self.graph.has_node(node_id):
                attrs = self.graph.nodes[node_id]
                memory = self._node_to_memory(node_id, attrs)
                # Helper: Inject relevance score
                if hasattr(memory, "model_extra") and memory.model_config.get('extra') == 'allow':
                     memory.relevance = score
                else:
                    setattr(memory, 'relevance', score)
                results.append(memory)
                
        return results

    async def retrieve_hybrid(self, query: str, query_embedding: List[float], limit: int = 20) -> List[MemoryNode]:
        """
        Combine BM25 text match and vector similarity search.
        Formula: Final Score = (0.7 * User Vector Score) + (0.3 * User BM25 Score)
            (Note: normalization needed as Vector is -1..1 (usually 0..1 for embedding) and BM25 is unbound/positive)
            
        Simple approach: Rank Fusion.
        1. Get Top N Vector Results
        2. Get Top N BM25 Results
        3. Combine unique results, calculating weighted score on intersections.
        """
        if not self.graph:
            return []

        # 1. Vector Search
        vector_candidates = await self.retrieve_by_similarity(query_embedding, limit=limit, min_score=0.3)
        vec_scores = {m.id: getattr(m, 'relevance', 0.0) for m in vector_candidates}
        
        # 2. BM25 Search
        bm25_candidates = []
        bm25_scores_map = {}
        
        if self.bm25_index and BM25Okapi:
            tokenized_query = query.lower().split()
            # Get scores for all docs
            doc_scores = self.bm25_index.get_scores(tokenized_query)
            
            # Map back to IDs
            # efficient way?
            # We have self.bm25_ids aligned with corpus
            top_n_indices = np.argsort(doc_scores)[::-1][:limit]
            
            for idx in top_n_indices:
                if doc_scores[idx] > 0:
                   mid = self.bm25_ids[idx]
                   bm25_scores_map[mid] = doc_scores[idx]
                   # Retrieve node
                   if self.graph.has_node(mid):
                       bm25_candidates.append(self._node_to_memory(mid, self.graph.nodes[mid]))
        
        # 3. Normalize scores
        # Vector is Cosine (approx 0 to 1). BM25 is 0 to infinity.
        # Max-normalize BM25 scores based on current specific query results
        if bm25_scores_map:
            max_bm25 = max(bm25_scores_map.values())
            for k in bm25_scores_map:
                bm25_scores_map[k] = bm25_scores_map[k] / max_bm25 if max_bm25 > 0 else 0
                
        # 4. Combine
        # Union of keys
        all_ids = set(vec_scores.keys()) | set(bm25_scores_map.keys())
        final_scored = []
        
        for mid in all_ids:
             v_score = vec_scores.get(mid, 0.0)
             b_score = bm25_scores_map.get(mid, 0.0)
             
             # Weighted Sum
             final_score = (0.7 * v_score) + (0.3 * b_score)
             final_scored.append((mid, final_score))
             
        # Sort
        final_scored.sort(key=lambda x: x[1], reverse=True)
        
        # Return Objects
        results = []
        for mid, score in final_scored[:limit]:
            if self.graph.has_node(mid):
                 attrs = self.graph.nodes[mid]
                 mem = self._node_to_memory(mid, attrs)
                 # Inject combined score
                 if hasattr(mem, "model_extra") and mem.model_config.get('extra') == 'allow':
                     mem.relevance = score
                 else:
                     setattr(mem, 'relevance', score)
                 results.append(mem)
                 
        return results
        
    async def get_episodic_graph(self) -> nx.DiGraph:
        """Return subgraph of EPISODIC memories."""
        nodes = [n for n, d in self.graph.nodes(data=True) if d.get('category') == 'episodic']
        return self.graph.subgraph(nodes).copy()

    def _node_to_memory(self, node_id: str, attrs: Dict) -> MemoryNode:
        """Convert graph node attributes back to MemoryNode object."""
        return MemoryNode(
            id=node_id,
            category=MemoryCategory(attrs.get("category", "discard")),
            content=attrs.get("content", ""),
            structured_data=json.loads(attrs.get("structured_data", "{}")),
            confidence=attrs.get("confidence", 0.5),
            stability=attrs.get("stability", 1.0), # Default for now
            created_at=attrs.get("created_at") if isinstance(attrs.get("created_at"), datetime) else datetime.now(),
            last_accessed=attrs.get("last_accessed") if isinstance(attrs.get("last_accessed"), datetime) else datetime.now(),
            access_count=attrs.get("access_count", 0),
            decay_rate=attrs.get("decay_rate", 0.0),
            embedding=attrs.get("embedding", [])
        )

    # --- Sub-Graph Export Methods for User Requirement ---
    def save_specialized_graphs(self):
        """Save specialized sub-graphs as requested."""
        try:
            # 1. Episodic Graph (Events)
            episodic_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('category') == 'episodic']
            episodic_graph = self.graph.subgraph(episodic_nodes).copy()
            self._save_subgraph(episodic_graph, "memory_graph_episodic.json")
            
            # 2. Knowledge Graph (Entities + Relations + Pinned/Critical facts)
            # Filter for semantic categories
            knowledge_cats = {'pinned', 'critical', 'relational'}
            knowledge_nodes = [
                n for n, d in self.graph.nodes(data=True) 
                if d.get('category') in knowledge_cats or d.get('label') == 'Entity'
            ]
            knowledge_graph = self.graph.subgraph(knowledge_nodes).copy()
            self._save_subgraph(knowledge_graph, "memory_graph_knowledge.json")
            
            logger.debug("Specialized graphs exported.")
        except Exception as e:
            logger.error(f"Failed to export specialized graphs: {e}")

    def _save_subgraph(self, subgraph: nx.DiGraph, filename: str):
        """Helper to save a subgraph to JSON."""
        dir_path = os.path.dirname(self.config.storage_path)
        path = os.path.join(dir_path, filename)
        
        data = nx.node_link_data(subgraph)
        # Serialize datetime
        for node in data['nodes']:
            for k, v in node.items():
                if isinstance(v, datetime):
                    node[k] = v.isoformat()
        
        with _open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    # --- Decay Management Methods ---

    async def get_by_category(self, category: MemoryCategory, limit: int = 1000) -> List[MemoryNode]:
        """
        Retrieve memories by category.
        
        Args:
            category: MemoryCategory to filter by.
            limit: Maximum number of memories to return.
            
        Returns:
            List of MemoryNode objects.
        """
        results = []
        target_cat = category.value if isinstance(category, MemoryCategory) else category
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("label") == "Memory" and attrs.get("category") == target_cat:
                results.append(self._node_to_memory(node_id, attrs))
                if len(results) >= limit:
                    break
        return results

    async def update_memory_confidence(self, memory_id: str, confidence: float, decay_rate: float) -> None:
        """
        Update confidence and decay rate for a memory node.
        
        Args:
            memory_id: ID of the memory to update.
            confidence: New confidence score.
            decay_rate: New decay rate.
        """
        if self.graph.has_node(memory_id):
            self.graph.nodes[memory_id]['confidence'] = confidence
            self.graph.nodes[memory_id]['decay_rate'] = decay_rate
            # Optimized: Don't save on every single update if batch processing, 
            # but for safety we save here. In a real batch loop, we might want a bulk update method.
            # For now, let's defer saving to the caller or rely on periodic saves? 
            # Actually, to be safe, let's save.
            self.save_graph()

    async def update_access_metrics(self, memory_id: str) -> None:
        """
        Update last accessed timestamp and access count for a memory node.
        
        Args:
            memory_id: ID of the memory to update.
        """
        if self.graph.has_node(memory_id):
            attrs = self.graph.nodes[memory_id]
            attrs['last_accessed'] = datetime.now()
            attrs['access_count'] = attrs.get('access_count', 0) + 1
            # Batch saving optimization could happen here, but for safety we save.
            # Ideally reflection loop could trigger a save after batch updates.
            # For now, let's just save to be consistent.
            self.save_graph()

    async def merge_memories(self, primary_id: str, secondary_ids: List[str], new_content: str = None, new_embedding: List[float] = None) -> bool:
        """
        Merge secondary memories into a primary memory node.
        
        Args:
            primary_id: The ID of the memory to keep and update.
            secondary_ids: List of memory IDs to merge into the primary and delete.
            new_content: Optional new content for the primary memory (e.g. consolidated summary).
            new_embedding: Optional new embedding for the primary memory.
            
        Returns:
            bool: True if merge was successful, False otherwise.
        """
        if not self.graph.has_node(primary_id):
            logger.error(f"Merge failed: Primary memory {primary_id} not found")
            return False
            
        try:
            # 1. Update primary memory content if provided
            if new_content:
                self.graph.nodes[primary_id]['content'] = new_content
                # Update structured_data content as well
                try:
                    struct_data = json.loads(self.graph.nodes[primary_id].get('structured_data', '{}'))
                    struct_data['content'] = new_content
                    self.graph.nodes[primary_id]['structured_data'] = json.dumps(struct_data)
                except Exception:
                    pass
            
            # 2. Update primary memory embedding if provided
            if new_embedding:
                self.graph.nodes[primary_id]['embedding'] = new_embedding
                self.vector_index[primary_id] = np.array(new_embedding, dtype=np.float32)
            
            # 3. Transfer outgoing edges from secondary nodes to primary
            for sec_id in secondary_ids:
                if not self.graph.has_node(sec_id):
                    continue
                    
                # Get all successors (outgoing edges)
                for successor in self.graph.successors(sec_id):
                    if successor == primary_id: continue # Don't link to self
                    
                    # Add edge to primary if it doesn't exist
                    if not self.graph.has_edge(primary_id, successor):
                        # Copy edge data
                        edge_data = self.graph.get_edge_data(sec_id, successor)
                        self.graph.add_edge(primary_id, successor, **edge_data)
                
                # Get all predecessors (incoming edges)
                for predecessor in self.graph.predecessors(sec_id):
                    if predecessor == primary_id: continue
                    
                    if not self.graph.has_edge(predecessor, primary_id):
                        edge_data = self.graph.get_edge_data(predecessor, sec_id)
                        self.graph.add_edge(predecessor, primary_id, **edge_data)
                        
                # 4. Remove secondary node
                self.graph.remove_node(sec_id)
                if sec_id in self.vector_index:
                    del self.vector_index[sec_id]
                    
            logger.info(f"Merged {len(secondary_ids)} memories into {primary_id}")
            self.save_graph()
            self._rebuild_bm25_index()
            return True
            
        except Exception as e:
            logger.error(f"Failed to merge memories: {e}", exc_info=True)
            return False

    async def get_memories_by_entity(self, entity_name: str) -> List[MemoryNode]:
        """
        Retrieve memories associated with a specific entity.
        
        Args:
            entity_name: Name of the entity to search for.
            
        Returns:
            List[MemoryNode]: List of related memories.
        """
        results = []
        entity_lower = entity_name.lower().strip()
        
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("label") == "Memory":
                # Check 1: Does structured_data have this entity?
                try:
                    struct_data = json.loads(attrs.get("structured_data", "{}"))
                    entities = [e.lower() for e in struct_data.get("entities", [])]
                    if entity_lower in entities:
                        results.append(self._node_to_memory(node_id, attrs))
                        continue
                except Exception:
                    pass
                
                # Check 2: Simple text containment in content (fallback)
                content = attrs.get("content", "").lower()
                if entity_lower in content:
                    results.append(self._node_to_memory(node_id, attrs))
        
        return results

    async def archive_low_confidence(self, threshold: float = 0.1) -> int:
        """
        Archive memories with confidence below threshold.
        
        Moves them from the active graph to 'memory_archive.jsonl'.
        
        Args:
            threshold: Confidence threshold below which to archive.
            
        Returns:
            Number of archived memories.
        """
        to_archive = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("label") == "Memory":
                conf = attrs.get("confidence", 1.0)
                if conf < threshold:
                    to_archive.append(node_id)
        
        if not to_archive:
            return 0
            
        archive_path = os.path.join(os.path.dirname(self.config.storage_path), "memory_archive.jsonl")
        count = 0
        
        try:
            with _open(archive_path, 'a', encoding='utf-8') as f:
                for node_id in to_archive:
                    attrs = self.graph.nodes[node_id]
                    # Create a storable dict
                    record = {
                        "id": node_id,
                        "archived_at": datetime.now().isoformat(),
                        **attrs
                    }
                    # Serialize datetime in attrs if necessary (though _node_to_memory handles it)
                    # But here we are dumping raw attrs.
                    # Convert objects to strings for JSON
                    serializable_record = {}
                    for k, v in record.items():
                        if isinstance(v, datetime):
                            serializable_record[k] = v.isoformat()
                        elif isinstance(v, (set, tuple)): # Should not happen in loaded graph but good practice
                             serializable_record[k] = list(v)
                        else:
                            serializable_record[k] = v
                            
                    f.write(json.dumps(serializable_record) + "\n")
                    
                    # Remove from graph and vector index
                    self.graph.remove_node(node_id)
                    if node_id in self.vector_index:
                        del self.vector_index[node_id]
                    count += 1
            
            self.save_graph()
            self._rebuild_bm25_index()    
            logger.info(f"Archived {count} low-confidence memories")
            return count
            
        except Exception as e:
            logger.error(f"Failed to archive memories: {e}")
            return 0

    # --- Visualization Methods ---

    async def visualize_network(self, graph_type: str = 'main') -> Optional[str]:
        """
        Generate an interactive HTML visualization of the memory graph.
        
        Args:
            graph_type: Type of graph to visualize ('main', 'episodic', 'knowledge').
                        'main' includes clustering by color.
            
        Returns:
            str: Path to the generated HTML file, or None if failed.
        """
        try:
            # Lazy import to avoid hard dependency if not used
            try:
                from pyvis.network import Network
            except ImportError:
                logger.error("pyvis not installed. Install with: pip install pyvis")
                return None
                
            # Initialize Network
            net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white")
            net.force_atlas_2based()
            
            # Select Subgraph based on type
            if graph_type == 'episodic':
                target_graph = await self.get_episodic_graph()
                filename = "memory_graph_episodic.html"
            elif graph_type == 'knowledge':
                # Re-create knowledge subgraph logic
                knowledge_cats = {'pinned', 'critical', 'relational'}
                nodes = [n for n, d in self.graph.nodes(data=True) 
                        if d.get('category') in knowledge_cats or d.get('label') == 'Entity']
                target_graph = self.graph.subgraph(nodes).copy()
                filename = "memory_graph_knowledge.html"
            else:
                # Main graph (default)
                target_graph = self.graph
                filename = "memory_graph_main.html"
            
            if target_graph.number_of_nodes() == 0:
                logger.warning(f"Graph type '{graph_type}' is empty.")
                return None

            # Add nodes with visual attributes
            for node_id, attrs in target_graph.nodes(data=True):
                label = attrs.get('label', 'Unknown')
                
                # Default visual props
                color = '#97c2fc' # Default blue
                title = ""
                
                if label == 'Memory':
                    category = attrs.get('category', 'discard')
                    content = attrs.get('content', '')
                    title = f"Type: Memory\nCategory: {category}\nContent: {content[:100]}..."
                    
                    # Color coding by category
                    if category == 'pinned':
                        color = '#32CD32' # Lime Green
                    elif category == 'critical':
                        color = '#FF4500' # Orange Red
                    elif category == 'episodic':
                        color = '#1E90FF' # Dodger Blue
                    elif category == 'relational':
                        color = '#9370DB' # Medium Purple
                    elif category == 'temporary':
                        color = '#808080' # Gray
                    else:
                        color = '#A9A9A9' # Dark Gray
                        
                    net.add_node(node_id, label=category.upper(), title=title, color=color, size=20)
                    
                elif label == 'Entity':
                    name = attrs.get('name', node_id)
                    cat = attrs.get('category', 'Entity')
                    title = f"Type: Entity\nName: {name}\nCategory: {cat}"
                    color = '#FFD700' # Gold for Entities
                    
                    net.add_node(node_id, label=name, title=title, color=color, size=25, shape='star')
            
            # Add edges
            for source, target, data in target_graph.edges(data=True):
                relation = data.get('relation', 'related')
                net.add_edge(source, target, title=relation, label=relation)
                
            # Save output
            output_dir = os.path.dirname(self.config.storage_path)
            output_path = os.path.join(output_dir, filename)
            
            # Pyvis save needs write access
            net.save_graph(output_path)
            logger.info(f"Graph visualization saved to {output_path}")
            return os.path.abspath(output_path)
            
        except Exception as e:
            logger.error(f"Failed to verify/generate graph: {e}", exc_info=True)
            return None