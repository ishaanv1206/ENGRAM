"""
Core data models and enums for the Cognitive Memory Controller.

This module defines all the data structures used throughout the system,
including memory categories, query intents, and various dataclasses for
memory management, conversation handling, and system operations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import uuid


# Enums

class MemoryCategory(Enum):
    """Categories for memory classification with different decay policies."""
    PINNED = "pinned"          # Ultra-critical persistent instructions
    CRITICAL = "critical"      # Extremely important, very slow decay
    EPISODIC = "episodic"      # Medium priority, usage-influenced decay
    RELATIONAL = "relational"  # Entity relationships
    TEMPORARY = "temporary"    # Short-term, fast decay
    DISCARD = "discard"        # Not worth storing


class LinkType(Enum):
    """Types of relationships between memories."""
    STRENGTHEN = "strengthen"  # New memory reinforces existing memory
    REPLACE = "replace"        # New memory supersedes old memory
    CONTRADICT = "contradict"  # Memories are in conflict


class QueryIntent(Enum):
    """Classification of user query intentions for retrieval optimization."""
    FACTUAL_RECALL = "factual_recall"      # "What did I say about X?"
    PREFERENCE_CHECK = "preference_check"  # "Do I like Y?"
    RELATIONSHIP = "relationship"          # "How is A related to B?"
    GENERAL = "general"                    # General conversation
    NO_MEMORY = "no_memory"               # Doesn't need memory


class DecayPolicy(Enum):
    """Decay policies for different memory categories."""
    NO_DECAY = "no_decay"        # Pinned memories (0.0% per day)
    VERY_SLOW = "very_slow"      # Critical memories (0.1% per day)
    MEDIUM = "medium"            # Episodic memories (2% per day)
    FAST = "fast"                # Temporary memories (20% per day)


# Core Message and Conversation Models

@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ConversationContext:
    """Context information about the current conversation state."""
    session_id: str
    turn_count: int
    recent_topics: List[str] = field(default_factory=list)
    active_entities: List[str] = field(default_factory=list)
    conversation_history: List[Message] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)


# Memory Models

@dataclass
class MemoryLink:
    """A link between memories indicating their relationship."""
    target_id: str
    link_type: LinkType
    weight: float  # 0.0 to 1.0, strength of the relationship
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryExtraction:
    """Structured memory extracted from conversation text."""
    category: MemoryCategory
    structured_data: Dict[str, Any]  # Extracted entities, facts, preferences, etc.
    confidence: float  # 0.0 to 1.0
    stability: float   # How stable/permanent this memory should be (0.0 to 1.0)
    decay_policy: DecayPolicy
    links: List[MemoryLink] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    extraction_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class MemoryNode:
    """A stored memory node in the graph database."""
    id: str
    category: MemoryCategory
    content: str  # Raw text content
    structured_data: Dict[str, Any]  # JSON-encoded structured data
    confidence: float  # 0.0 to 1.0
    stability: float   # 0.0 to 1.0
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    decay_rate: float = 0.0  # Current decay rate (adjusted by usage)
    embedding: Optional[List[float]] = None  # Vector embedding for semantic search
    
    def __post_init__(self):
        """Ensure datetime objects are properly initialized."""
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.last_accessed, str):
            self.last_accessed = datetime.fromisoformat(self.last_accessed)


# Retrieval Models

@dataclass
class MemoryBudget:
    """Budget constraints for memory retrieval operations."""
    max_memories: int      # Maximum number of memory nodes
    max_tokens: int        # Maximum token overhead
    latency_ms: int        # Time budget for retrieval in milliseconds


@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation."""
    pinned: Dict[str, str]  # Pinned memories (category -> content)
    memories: List[MemoryNode]  # Retrieved memory nodes
    total_tokens: int  # Estimated token count
    retrieval_time_ms: float = 0.0  # Actual retrieval time
    query_intent: Optional[QueryIntent] = None


# Reflection and Validation Models

@dataclass
class ReflectionTask:
    """A task for the reflection loop to process asynchronously."""
    task_id: str
    query: str
    response: str
    retrieved_memories: List[MemoryNode]
    context: ConversationContext
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Generate task ID if not provided."""
        if not self.task_id:
            self.task_id = str(uuid.uuid4())


@dataclass
class ValidationResult:
    """Result of validating a response against retrieved memories."""
    hallucination_detected: bool
    explanation: str = ""
    confidence_adjustments: Dict[str, float] = field(default_factory=dict)  # memory_id -> adjustment
    conflicts_detected: List[str] = field(default_factory=list)  # List of memory IDs in conflict


# Error Handling Models

@dataclass
class ErrorResponse:
    """Standardized error response structure."""
    error_type: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    error_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    recoverable: bool = True


# Utility Functions

def create_memory_node(
    content: str,
    category: MemoryCategory,
    structured_data: Optional[Dict[str, Any]] = None,
    confidence: float = 1.0,
    stability: float = 1.0
) -> MemoryNode:
    """Create a new MemoryNode with default values."""
    now = datetime.now()
    return MemoryNode(
        id=str(uuid.uuid4()),
        category=category,
        content=content,
        structured_data=structured_data or {},
        confidence=confidence,
        stability=stability,
        created_at=now,
        last_accessed=now,
        access_count=0,
        decay_rate=0.0,
        embedding=None
    )


def create_conversation_context(session_id: Optional[str] = None) -> ConversationContext:
    """Create a new ConversationContext with default values."""
    return ConversationContext(
        session_id=session_id or str(uuid.uuid4()),
        turn_count=0,
        recent_topics=[],
        active_entities=[],
        conversation_history=[],
        started_at=datetime.now(),
        last_activity=datetime.now()
    )


def create_memory_budget(
    intent: QueryIntent,
    turn_count: int = 0
) -> MemoryBudget:
    """Create a MemoryBudget based on query intent and conversation state."""
    base_budgets = {
        QueryIntent.FACTUAL_RECALL: MemoryBudget(max_memories=10, max_tokens=500, latency_ms=100),
        QueryIntent.PREFERENCE_CHECK: MemoryBudget(max_memories=5, max_tokens=300, latency_ms=80),
        QueryIntent.RELATIONSHIP: MemoryBudget(max_memories=15, max_tokens=600, latency_ms=120),
        QueryIntent.GENERAL: MemoryBudget(max_memories=8, max_tokens=400, latency_ms=100),
        QueryIntent.NO_MEMORY: MemoryBudget(max_memories=0, max_tokens=0, latency_ms=0),
    }
    
    budget = base_budgets[intent]
    
    # Adjust based on conversation length
    if turn_count > 500:
        budget.max_memories = int(budget.max_memories * 1.2)
        budget.max_tokens = int(budget.max_tokens * 1.2)
    
    return budget