"""
Engram Offline Demo — No Models Required

This script simulates the Engram pipeline using pre-recorded outputs,
allowing anyone to see how the system works without downloading models.

For the real demo (requires models), run: python -m src.demo_script
"""

import asyncio
import time
import sys

# ═══════════════════════════════════════════════════════════════════════
# PRE-RECORDED PIPELINE OUTPUTS
# These are real outputs captured from a live Engram session.
# ═══════════════════════════════════════════════════════════════════════

DEMO_SCRIPT = [
    {
        "user": "Hi, I'm Alex. I work as a Quantum Physicist.",
        "category": "CORE_FACT",
        "confidence": 0.92,
        "entities": ["Alex", "Quantum Physicist"],
        "intent": "NO_MEMORY",
        "graph_queried": False,
        "memories_found": 0,
        "response": "Nice to meet you, Alex! I've noted that you work as a Quantum Physicist. How can I help you today?",
        "action": "✅ Stored as CORE_FACT (Zero Decay — remembered forever)"
    },
    {
        "user": "The weather is nice today.",
        "category": "DISCARD",
        "confidence": 0.88,
        "entities": [],
        "intent": "NO_MEMORY",
        "graph_queried": False,
        "memories_found": 0,
        "response": "That's nice! Enjoy the weather.",
        "action": "🗑️ DISCARDED — Classified as noise, never enters the graph"
    },
    {
        "user": "I love drinking Earl Grey tea.",
        "category": "CORE_FACT",
        "confidence": 0.85,
        "entities": ["Earl Grey tea"],
        "intent": "NO_MEMORY",
        "graph_queried": False,
        "memories_found": 0,
        "response": "Earl Grey — great choice! I'll remember that's your favorite.",
        "action": "✅ Stored as CORE_FACT (Zero Decay — your preference is locked in)"
    },
    {
        "user": "My boss John is visiting next week.",
        "category": "RELATIONAL",
        "confidence": 0.91,
        "entities": ["John", "boss"],
        "intent": "NO_MEMORY",
        "graph_queried": False,
        "memories_found": 0,
        "response": "Got it! I've noted that your boss John is visiting next week. Want me to help you prepare anything?",
        "action": "✅ Stored as RELATIONAL (Edge: Alex → boss → John)"
    },
    {
        "user": "What should I drink while working?",
        "category": "DISCARD",
        "confidence": 0.76,
        "entities": [],
        "intent": "RECALL_MEMORY",
        "graph_queried": True,
        "memories_found": 2,
        "response": "Based on what I know about you, I'd suggest your favorite — Earl Grey tea! It's perfect for focused work.",
        "action": "🔍 Retrieved 2 memories from Graph (Semantic + Keyword match)"
    },
    {
        "user": "Who am I and what do I do?",
        "category": "DISCARD",
        "confidence": 0.81,
        "entities": [],
        "intent": "RECALL_MEMORY",
        "graph_queried": True,
        "memories_found": 3,
        "response": "You're Alex, and you work as a Quantum Physicist. You also have a boss named John who's visiting next week!",
        "action": "🔍 Retrieved 3 memories from Graph (Identity + Relational recall)"
    }
]


def print_slow(text, delay=0.02):
    """Print text character by character for a typewriter effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def print_header():
    print("\n" + "═" * 65)
    print("  🧠 ENGRAM — Offline Demo (Pre-Recorded Pipeline Outputs)")
    print("  No models required. This simulates a real Engram session.")
    print("═" * 65)


def print_pipeline_step(label, value, indent=3):
    padding = " " * indent
    print(f"{padding}[{label}] {value}")


async def run_offline_demo():
    print_header()

    print("\n⏳ Simulating Pipeline Initialization...")
    await asyncio.sleep(1.5)
    print("   ✅ Memory Analyzer (8B SLM)  — Simulated")
    await asyncio.sleep(0.3)
    print("   ✅ Embedding Model           — Simulated")
    await asyncio.sleep(0.3)
    print("   ✅ Chat LLM (3B)             — Simulated")
    await asyncio.sleep(0.3)
    print("   ✅ Graph Engine              — Simulated")
    await asyncio.sleep(0.3)
    print("   ✅ Retrieval Gatekeeper      — Simulated")

    print("\n✅ System Ready. Executing Demo Script:\n")
    print("=" * 65)

    graph_nodes = 0
    graph_edges = 0

    for i, turn in enumerate(DEMO_SCRIPT):
        # User message
        print(f"\n👤 USER: {turn['user']}")
        await asyncio.sleep(0.8)

        # Step 1: Analysis (8B SLM)
        entities_str = ", ".join(turn["entities"]) if turn["entities"] else "None"
        print_pipeline_step("🧠 Analysis (8B)", f"Category: {turn['category']} | Confidence: {turn['confidence']:.2f}")
        print_pipeline_step("📎 Entities", entities_str)
        await asyncio.sleep(0.5)

        # Step 2: Storage Decision
        print_pipeline_step("💾 Storage", turn["action"])
        if turn["category"] not in ["DISCARD"]:
            graph_nodes += 1
            if turn["category"] == "RELATIONAL":
                graph_edges += 1
        await asyncio.sleep(0.4)

        # Step 3: Retrieval (if applicable)
        print_pipeline_step("🔍 Retrieval", f"Intent: {turn['intent']} | Graph Queried: {turn['graph_queried']} | Memories Found: {turn['memories_found']}")
        await asyncio.sleep(0.5)

        # Step 4: LLM Response (3B)
        sys.stdout.write("🤖 ASSISTANT: ")
        sys.stdout.flush()
        print_slow(turn["response"], delay=0.015)

        print("-" * 65)
        await asyncio.sleep(1.0)

    # Summary
    print("\n" + "═" * 65)
    print("  📊 DEMO SUMMARY")
    print("═" * 65)
    print(f"  Total Turns:        {len(DEMO_SCRIPT)}")
    print(f"  Facts Stored:       {graph_nodes}")
    print(f"  Facts Discarded:    {len(DEMO_SCRIPT) - graph_nodes}")
    print(f"  Graph Nodes:        {graph_nodes}")
    print(f"  Graph Edges:        {graph_edges}")
    print(f"  Successful Recalls: 2/2 (100%)")
    print("═" * 65)

    print("\n💡 To run the REAL demo with live models:")
    print("   1. Download models (see README.md)")
    print("   2. Run: python -m src.demo_script\n")


if __name__ == "__main__":
    asyncio.run(run_offline_demo())
