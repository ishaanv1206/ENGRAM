# Memory System Architecture

## Memory Categories & Lifecycle

### 📌 PINNED (Always Remembered)
**Storage**: `data/pinned_memory.json` (in-memory, persistent)
**Decay Rate**: 0% (NEVER decays)
**Retrieval**: ALWAYS included in every conversation
**Examples**:
- "My name is Ishaan but call me Jumbo"
- "I prefer Python over Java"
- "Always be formal with me"
- "I'm in EST timezone"

**How it works**:
1. SLM classifies as PINNED
2. Stored in `PinnedMemoryManager` (separate from Neo4j)
3. Saved to `pinned_memory.json` on disk
4. Loaded on startup
5. Injected into EVERY conversation automatically

### 🔴 CRITICAL (Very Slow Decay)
**Storage**: Neo4j graph database
**Decay Rate**: 0.5% per day (0.005)
**Retrieval**: Retrieved when semantically relevant
**Examples**:
- "I will submit the report by Friday"
- "My birthday is June 5th"
- "I work at Google"

**Lifecycle**:
- Day 1: confidence = 1.0
- Day 100: confidence ≈ 0.90
- Day 1000: confidence ≈ 0.37
- Archived when confidence < 0.1

### 📝 EPISODIC (Medium Decay)
**Storage**: Neo4j graph database
**Decay Rate**: 5% per day (0.05)
**Retrieval**: Retrieved when semantically relevant
**Examples**:
- "Yesterday I went to the park"
- "I just finished reading a book"
- "Last week I met John"

**Lifecycle**:
- Day 1: confidence = 1.0
- Day 10: confidence ≈ 0.82
- Day 50: confidence ≈ 0.36
- Day 100: confidence ≈ 0.13
- Archived when confidence < 0.1

### ⏰ TEMPORARY (Fast Decay)
**Storage**: Neo4j graph database
**Decay Rate**: 28% per day (0.28)
**Retrieval**: Retrieved when semantically relevant (rarely)
**Examples**:
- "I'm working on chapter 3"
- "It's raining today"
- "I'm feeling tired"

**Lifecycle**:
- Day 1: confidence = 1.0
- Day 3: confidence ≈ 0.51
- Day 7: confidence ≈ 0.21
- Day 10: confidence ≈ 0.11
- Archived when confidence < 0.1

### 🔗 RELATIONAL (Medium Decay)
**Storage**: Neo4j graph database
**Decay Rate**: 5% per day (0.05)
**Retrieval**: Retrieved when entities are mentioned
**Examples**:
- "John is my brother"
- "Python is faster than Ruby"
- "Sarah works with me"

## Decay System

### How Decay Works

1. **Hourly Background Task**:
   - Runs every hour automatically
   - Processes all memories in Neo4j
   - Skips PINNED memories (0% decay)

2. **Usage-Based Adjustment**:
   - Frequently accessed memories decay slower
   - 1 access/day = 30% slower decay
   - 10 accesses/day = 60% slower decay
   - 100+ accesses/day = 90% slower decay (max)

3. **Confidence Update**:
   ```
   new_confidence = old_confidence * (1 - hourly_decay_rate)
   ```

4. **Archival**:
   - When confidence < 0.1, memory is archived
   - Archived memories moved to `data/archive/`
   - Can be restored if needed

### Example: EPISODIC Memory Decay

```
Day 0:  confidence = 1.00 (just created)
Day 1:  confidence = 0.98 (2% decay)
Day 7:  confidence = 0.87
Day 30: confidence = 0.55
Day 60: confidence = 0.30
Day 90: confidence = 0.16
Day 100: confidence = 0.13
Day 115: confidence < 0.1 → ARCHIVED
```

## Retrieval System

### Retrieval Priority

1. **PINNED memories** (Tier 0):
   - ALWAYS included
   - No search needed
   - From `pinned_memory.json`

2. **Recent Cache** (Tier 1):
   - Last 100 accessed memories
   - Fast in-memory lookup
   - LRU eviction

3. **Graph Database** (Tier 2):
   - Semantic search using embeddings
   - Text-based keyword matching
   - Entity-based relationships

### Retrieval Methods

**1. Text-Based Matching** (Fast, no embeddings needed):
```
Query: "What's my name?"
Keywords: ["name"]
Matches: Memories containing "name"
```

**2. Semantic Search** (Slower, more accurate):
```
Query: "What's my nickname?"
Embedding: [0.23, -0.45, 0.67, ...]
Matches: Memories with similar embeddings
Result: "Call me Jumbo" (even though "nickname" not in text)
```

**3. Entity-Based** (Graph traversal):
```
Query mentions "Python"
→ Find Entity node "Python"
→ Traverse MENTIONS relationships
→ Return all memories about Python
```

## Storage Locations

```
data/
├── pinned_memory.json          # PINNED memories (always remembered)
├── sessions/                   # Conversation sessions
│   └── {session-id}.json
├── archive/                    # Archived low-confidence memories
│   └── {date}-archived.json
└── models/                     # LLM and embedding models
    ├── Llama-3.2-3B-Instruct-uncensored-Q6_K_L.gguf
    ├── Llama-3.2-1B-Instruct-Q6_K_L.gguf
    └── mxbai-embed-large-v1_fp32.gguf
```

**Neo4j Database**:
- All non-PINNED memories
- Entity nodes
- Relationships between memories
- Embeddings for semantic search

## Key Differences from "Store Everything"

### ❌ What We DON'T Do:
- Store everything forever
- Retrieve all memories every time
- Use only keyword matching
- Treat all memories equally

### ✅ What We DO:
- Classify memories by importance
- Apply category-specific decay
- Retrieve only relevant memories
- Prioritize PINNED memories
- Archive old/unused memories
- Use semantic + text + entity search

## Configuration

**Decay Rates** (in `src/decay_manager.py`):
```python
decay_policies = {
    MemoryCategory.PINNED: 0.0,      # Never
    MemoryCategory.CRITICAL: 0.005,  # 0.5% per day
    MemoryCategory.EPISODIC: 0.05,   # 5% per day
    MemoryCategory.TEMPORARY: 0.28,  # 28% per day
}
```

**Decay Frequency**: Every 1 hour
**Archive Threshold**: confidence < 0.1
**Cache Size**: 100 memories (LRU)
**Memory Budget**: 500 tokens max per retrieval

## Summary

Your system is **NOT** just storing everything and doing semantic search. It's a sophisticated memory management system that:

1. **Classifies** memories by importance
2. **Stores** PINNED memories separately (always remembered)
3. **Decays** other memories over time
4. **Archives** old/unused memories
5. **Retrieves** only relevant memories using multiple strategies
6. **Prioritizes** important memories over temporary ones

This mimics human memory: you always remember your name, but forget what you had for lunch last Tuesday!
