import asyncio
import logging
from src.main import initialize_pipeline
from src.config import ConfigManager
from src.models import create_conversation_context

# Configure logging to show only INFO (clean output)
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("src.demo")
logger.setLevel(logging.INFO)

async def run_demo():
    print("\n🚀 Starting Neurohacks Cognitive Memory Demo...\n")
    print("Initializing Pipeline... please wait.")
    
    try:
        config = ConfigManager.load()
        pipeline = await initialize_pipeline(config)
        context = create_conversation_context()
        
        # Define a script that showcases:
        # 1. Fact Storage (Episodic)
        # 2. Chit-Chat (Discard)
        # 3. Recall (Retrieval)
        
        conversation = [
            "Hi, I'm Alex. I work as a Quantum Physicist.",
            "The weather is nice today.",  # Should be DISCARDed
            "I love drinking Earl Grey tea.",
            "What should I drink while working?",
            "Who am I and what do I do?"
        ]
        
        print("\n✅ System Ready. Executing Demo Script:\n")
        print("="*60)
        
        for i, text in enumerate(conversation):
            print(f"\n👤 USER: {text}")
            
            # Step 1: Analyze (Show what's happening behind scenes)
            extraction = await pipeline.analyzer.analyze(text, context)
            print(f"   [🧠 Analysis] Category: {extraction.category.value.upper()} | Confidence: {extraction.confidence:.2f}")
            
            # Step 2: Process & Answer
            response, metadata = await pipeline.process_turn(text, context)
            
            # Show Retrieval Decision
            retrieval = metadata.get('retrieval', {})
            intent = retrieval.get('intent', 'UNKNOWN')
            print(f"   [🔍 Retrieval] Intent: {intent} | Graph Checked: {retrieval.get('graph_queried')}")
            
            print(f"🤖 ASSISTANT: {response}")
            print("-" * 60)
            
            # Small pause for readability
            await asyncio.sleep(2)
            
        print("\nDemo Complete. Memory persisted to `data/memory_graph.json`.")
        
    except Exception as e:
        print(f"\n❌ Demo Failed: {e}")

if __name__ == "__main__":
    asyncio.run(run_demo())
