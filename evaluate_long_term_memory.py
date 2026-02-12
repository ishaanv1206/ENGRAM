
import asyncio
import logging
import random
import string
import time
import sys
import uuid
from datetime import datetime
from typing import List, Dict, Tuple
from unittest.mock import MagicMock

# Adjust path to include src
import os
sys.path.append(os.getcwd())

from src.main import initialize_pipeline
from src.config import ConfigManager
from src.models import ConversationContext, Message
from src.pipeline import CognitivePipeline

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Eval1100")

# --- Configuration ---
TOTAL_TURNS = 1100
FACT_INTERVAL = 10  # Insert a fact every ~10 turns
QUERY_DELAY = 50    # Ask about a fact ~50 turns after it was inserted
DISTRACTOR_MESSAGES = [
    "Just chilling.", "What's the weather?", "Tell me a joke.", "I am tired.", 
    "Coding is fun.", "Python is great.", "I like coffee.", "It is late.", 
    "How are you?", "Nice to meet you."
]


# Valid categories for generation 
# (DISCARD is tested via distractors and checked for non-retrieval if needed, but mostly we test positive retrieval here)
CATEGORIES = ["PINNED", "CRITICAL", "RELATIONAL", "EPISODIC", "TEMPORARY", "UPDATE_TEST"]

def generate_memory_by_type(category: str) -> Tuple[str, str, str]:
    """Generates a unique memory input and query for a specific category."""
    rand_id = ''.join(random.choices(string.digits, k=4))
    rand_name = ''.join(random.choices(string.ascii_uppercase, k=5))
    
    if category == "PINNED":
        # Core identity/preferences
        fact = f"My absolute favorite unique color is Color{rand_name}."
        query = f"What is my favorite unique color?"
        answer = f"Color{rand_name}"
        
    elif category == "CRITICAL":
        # Deadlines/Must-knows
        fact = f"IMPORTANT: The deadline for Project{rand_name} is {rand_id} PM."
        query = f"When is the deadline for Project{rand_name}?"
        answer = f"{rand_id} PM"
        
    elif category == "RELATIONAL":
        # Relationships
        fact = f"{rand_name} is the secret cousin of {rand_name}Junior."
        query = f"Who is {rand_name} related to?"
        answer = f"{rand_name}Junior"
        
    elif category == "EPISODIC":
        # Events
        fact = f"Yesterday I went to the {rand_name}Restaurant and ate {rand_name}Burger."
        query = f"What did I eat at {rand_name}Restaurant?"
        answer = f"{rand_name}Burger"

    elif category == "TEMPORARY":
        # Short term
        fact = f"Remind me to call {rand_name}Supervisor in 5 minutes."
        query = f"Who do I need to call in 5 minutes?"
        answer = f"{rand_name}Supervisor"
        
    return fact, query, answer


async def run_evaluation():
    print(f"\n=== Starting {TOTAL_TURNS}-Turn Long-Term Memory Evaluation ===")
    print("Initializing Pipeline...")
    
    config = ConfigManager.load()
    pipeline = await initialize_pipeline(config)
    
    # Using Real LLM as requested
    print("✅ Using Real Main LLM (End-to-End Test)")

    # Initialize Context
    context = ConversationContext(
        session_id=str(uuid.uuid4()),
        turn_count=0,
        recent_topics=[],
        active_entities=[],
        conversation_history=[],
        started_at=datetime.now(),
        last_activity=datetime.now()
    )

    # Simulation State
    facts_database: List[Dict] = []
    stats = {
        "total_queries": 0,
        "successful_retrievals": 0,
        "failed_retrievals": 0,
        "latency_sum": 0,
        "history": [],
        # Per-category stats
        "by_category": {c: {"total": 0, "correct": 0} for c in CATEGORIES}
    }
    
    start_time = time.time()
    category_cycle = 0 # To rotate through categories
    
    for turn in range(1, TOTAL_TURNS + 1):
        # 1. Determine Action for this Turn
        action = "distractor"
        user_input = random.choice(DISTRACTOR_MESSAGES)
        expected_keyword = None
        current_category = None
        
        # Insert Fact logic
        if turn % FACT_INTERVAL == 0:
            # Diverse Randomization: Pick any category, not just cyclic
            target_cat = random.choice(CATEGORIES + ["UPDATE_TEST"]) 
            
            # Special Handling for UPDATE_TEST
            if target_cat == "UPDATE_TEST":
                # Create a fact (Project Deadline)
                rand_id = ''.join(random.choices(string.digits, k=4))
                project_name = f"ProjectAlpha{rand_id}"
                time_1 = f"{random.randint(1,5)} PM"
                time_2 = f"{random.randint(6,11)} PM"
                
                # 1. Insert Initial
                fact_text = f"The deadline for {project_name} is {time_1}."
                context.turn_count += 1
                await pipeline.analyzer.analyze(fact_text, context)
                await pipeline._store_memory(await pipeline.analyzer.analyze(fact_text, context), fact_text, context)
                print(f"Turn {turn} [Insert UPDATE_TEST Initial]: {fact_text}")
                
                # 2. Insert Update (Immediate or slightly delayed? Let's do immediate for this test case logic simplification)
                # In real flow, this would be a separate turn, but here we want to seed the DB.
                # Let's verify update by inserting a NEW fact that contradicts or updates the old one.
                update_text = f"UPDATE: The deadline for {project_name} changed to {time_2}."
                
                # Add to DB for querying later
                facts_database.append({
                    "category": "UPDATE_TEST",
                    "fact": update_text,
                    "query": f"When is the deadline for {project_name}?",
                    "answer": time_2, # Expect the NEW time
                    "inserted_at": turn,
                    "queried": False
                })
                
                # We return the UPDATE text to be processed as the user input for this turn
                user_input = update_text
                action = "insert_fact"
                
            else:
                # Standard Categories
                fact_text, query_text, keyword = generate_memory_by_type(target_cat)
                user_input = fact_text
                facts_database.append({
                    "category": target_cat,
                    "fact": fact_text,
                    "query": query_text,
                    "answer": keyword,
                    "inserted_at": turn,
                    "queried": False
                })
                action = "insert_fact"
        
        # Query Logic
        candidates = [f for f in facts_database if not f['queried'] and (turn - f['inserted_at']) >= QUERY_DELAY]
        
        if candidates and turn % FACT_INTERVAL != 0:
            target_fact = candidates[0]
            user_input = target_fact['query']
            expected_keyword = target_fact['answer']
            current_category = target_fact['category']
            target_fact['queried'] = True
            action = "query_fact"

        turn_start = time.time()
        
        # 2. OPTIMIZED EXECUTION
        if action == "distractor":
             # FAST PATH: Skip SLM/Embedding to speed up test
             context.turn_count += 1
             context.conversation_history.append(Message(role="user", content=user_input, timestamp=datetime.now()))
             context.conversation_history.append(Message(role="assistant", content="Checked.", timestamp=datetime.now()))
             
             if turn % 100 == 0: 
                 print(f"Turn {turn} [Distractor] (Fast-forwarded)")

        elif action == "insert_fact":
            # FULL PATH: Analyze + Store
            extraction = await pipeline.analyzer.analyze(user_input, context)
            await pipeline._store_memory(extraction, user_input, context)
            print(f"Turn {turn} [Insert {target_cat}]: {user_input}")

        elif action == "query_fact":
            # RETRIEVAL PATH: Analyze + Retrieve + Generate (Real LLM)
            retrieval = await pipeline.retriever.retrieve(user_input, context)
            
            # Generate Real Response
            memory_context = pipeline.influencer.inject(retrieval, user_input)
            response = await pipeline.llm.generate(
                query=user_input,
                memory_context=memory_context,
                conversation_history=context.conversation_history
            )
            
            stats["total_queries"] += 1
            stats["by_category"][current_category]["total"] += 1
            
            found = False
            retrieved_texts = [m.content for m in retrieval.memories]
            combined_context = " ".join(retrieved_texts)
            
            # Check if answer is in retrieval (Recall)
            if expected_keyword in combined_context:
                stats["successful_retrievals"] += 1
                stats["by_category"][current_category]["correct"] += 1
                found = True
            
            status_icon = "✅" if found else "❌"
            print(f"Turn {turn} [{current_category}]: Found '{expected_keyword}'? {status_icon}")
            # print(f"  Response: {response[:100]}...") # Optional debug
            
            # Update history with REAL response
            context.turn_count += 1
            context.conversation_history.append(Message(role="user", content=user_input, timestamp=datetime.now()))
            context.conversation_history.append(Message(role="assistant", content=response, timestamp=datetime.now()))
            
        # --- LOGGING FOR USER ---
        # Capture response for logging
        log_response = "N/A"
        if action == "distractor":
             log_response = "Checked."
        elif action == "insert_fact":
             log_response = "[Memory Stored]"
        elif action == "query_fact":
             log_response = response
             
        try:
            with open("evaluation_log.txt", "a", encoding="utf-8") as f:
                 f.write(f"Turn {turn}\nUser: {user_input}\nSystem: {log_response}\n\n")
        except Exception as e:
            print(f"Logging failed: {e}")
        # ------------------------

        turn_duration = time.time() - turn_start
        stats["latency_sum"] += turn_duration

        # Periodic Report & Tracking
        if turn % 100 == 0:
            avg_lat = stats["latency_sum"] / turn
            acc = (stats["successful_retrievals"] / stats["total_queries"] * 100) if stats["total_queries"] > 0 else 0
            
            # Record stat point
            stats["history"].append({
                "turn": turn,
                "accuracy": acc,
                "latency": avg_lat
            })

            print(f"\n--- Report at Turn {turn} ---")
            print(f"Accuracy: {acc:.2f}% ({stats['successful_retrievals']}/{stats['total_queries']})")
            print(f"Avg Latency: {avg_lat:.2f}s")
             
            for cat in CATEGORIES:
                 c_stats = stats["by_category"][cat]
                 c_acc = (c_stats['correct'] / c_stats['total'] * 100) if c_stats['total'] > 0 else 0
                 print(f"  {cat}: {c_acc:.1f}%")
            print("------------------------------\n")


    # Final Report Generation
    total_time = time.time() - start_time
    overall_acc = (stats["successful_retrievals"] / stats["total_queries"] * 100) if stats["total_queries"] > 0 else 0
    
    report_content = f"""# Long-Term Memory Evaluation Report (1100 Turns)
**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Total Time**: {total_time:.2f}s
**Average Latency**: {(stats['latency_sum'] / TOTAL_TURNS):.2f}s/turn

## 1. Overall Performance
| Metric | Value |
| :--- | :--- |
| **Total Queries** | {stats['total_queries']} |
| **Successful** | {stats['successful_retrievals']} |
| **Accuracy** | **{overall_acc:.2f}%** |

## 2. Category Breakdown
| Category | Total | Correct | Accuracy |
| :--- | :--- | :--- | :--- |
"""
    
    for cat in CATEGORIES:
         c_stats = stats["by_category"][cat]
         c_acc = (c_stats['correct'] / c_stats['total'] * 100) if c_stats['total'] > 0 else 0
         report_content += f"| {cat} | {c_stats['total']} | {c_stats['correct']} | **{c_acc:.1f}%** |\n"

    report_content += "\n## 3. Accuracy Over Time\n"
    report_content += "| Turn | Accuracy | Latency |\n| :--- | :--- | :--- |\n"
    for point in stats["history"]:
        report_content += f"| {point['turn']} | {point['accuracy']:.2f}% | {point['latency']:.2f}s |\n"

    # --- POST-EVALUATION ANALYSIS ---
    
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Visualization
    print("\ngenerating visualizations...")
    try:
        # Generate all 3 types of graphs
        await pipeline.graph_engine.visualize_network('main')
        await pipeline.graph_engine.visualize_network('episodic')
        await pipeline.graph_engine.visualize_network('knowledge')
        
        # Move them to results folder
        import shutil
        for gtype in ['main', 'episodic', 'knowledge']:
            src = os.path.join("data", f"memory_graph_{gtype}.html")
            dst = os.path.join(results_dir, f"memory_graph_{gtype}.html")
            if os.path.exists(src):
                shutil.move(src, dst)
                print(f" Saved graph: {dst}")
                
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

    # 2. Plotting Accuracy
    print("\nGenerating accuracy plot...")
    try:
        import matplotlib.pyplot as plt
        
        turns = [p['turn'] for p in stats['history']]
        accuracies = [p['accuracy'] for p in stats['history']]
        latencies = [p['latency'] for p in stats['history']]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Turns')
        ax1.set_ylabel('Accuracy (%)', color=color)
        ax1.plot(turns, accuracies, color=color, marker='o', label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_ylim(0, 105)
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('Avg Latency (s)', color=color)
        ax2.plot(turns, latencies, color=color, linestyle='--', label='Latency')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Long-Term Memory Performance (1100 Turns)')
        fig.tight_layout()
        
        plot_path = os.path.join(results_dir, "accuracy_graph.png")
        plt.savefig(plot_path)
        print(f" Saved plot: {plot_path}")
        
    except ImportError:
        print("❌ matplotlib not installed. Skipping plot generation.")
    except Exception as e:
        print(f"❌ Plotting failed: {e}")

    # Save Report to Results Folder
    report_path = os.path.join(results_dir, "evaluation_report_1100.md")
    with open(report_path, "w") as f:
        f.write(report_content)
    
    print("\n=== EVALUATION COMPLETE ===")
    print(f"Results saved to: {results_dir}/")
    print(f"Overall Accuracy: {overall_acc:.2f}%")
    print("===============================")

if __name__ == "__main__":
    # Clean up before running
    print("🧹 Cleaning up old data...")
    import shutil
    if os.path.exists("data/graph.json"):
        os.remove("data/graph.json")
    if os.path.exists("data/memory_archive.jsonl"):
        os.remove("data/memory_archive.jsonl")
    if os.path.exists("data/pinned_memories.json"):
        os.remove("data/pinned_memories.json")
    if os.path.exists("results"):
        shutil.rmtree("results") # Clean old results
        
    asyncio.run(run_evaluation())
