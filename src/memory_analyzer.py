"""
Memory Analyzer for the Cognitive Memory Controller.

This module implements the MemoryAnalyzer class that uses a Small Language Model (SLM)
via llama.cpp to classify conversation text and extract structured memory information.
The analyzer categorizes text into memory types and extracts structured data including
preferences, facts, constraints, commitments, and relationships.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

try:
    from llama_cpp import Llama
except ImportError:
    # Fallback for testing or when llama-cpp-python is not available
    Llama = None

from .models import (
    MemoryExtraction, MemoryCategory, LinkType, DecayPolicy, 
    MemoryLink, ConversationContext
)
from .config import SLMConfig


logger = logging.getLogger(__name__)


class MemoryAnalyzer:
    """
    Analyzes conversation text using a Small Language Model to classify and extract
    structured memory information.
    
    The analyzer uses llama.cpp with a small model (Llama 3.2 1B) for fast classification
    and structured data extraction. It categorizes text into memory types and extracts
    entities, facts, preferences, relationships, and other structured information.
    """
    
    def __init__(self, config: SLMConfig):
        """
        Initialize the Memory Analyzer with SLM configuration.
        
        Args:
            config: SLM configuration containing model path and parameters.
            
        Raises:
            RuntimeError: If llama-cpp-python is not available or model fails to load.
        """
        self.config = config
        self.llm: Optional[Llama] = None
        
        if Llama is None:
            logger.warning("llama-cpp-python not available, MemoryAnalyzer will not function")
            return
            
        try:
            logger.info(f"Loading SLM model from {config.model_path}")
            lora_msg = f" with LoRA adapter {config.lora_path}" if config.lora_path else ""
            logger.info(f"Loading SLM model from {config.model_path}{lora_msg}")
            self.llm = Llama(
                model_path=config.model_path,
                n_ctx=config.n_ctx,
                n_gpu_layers=config.n_gpu_layers,
                lora_path=config.lora_path,
                verbose=False,
                seed=-1,
                n_threads=None,
                flash_attn=True  # Enable Flash Attention for speed
            )
            logger.info("SLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SLM model: {e}")
            raise RuntimeError(f"Failed to initialize SLM model: {e}")
    
    async def analyze(self, text: str, context: ConversationContext) -> MemoryExtraction:
        """
        Analyze conversation text and extract structured memory information.
        
        Args:
            text: The conversation text to analyze.
            context: Current conversation context for better classification.
            
        Returns:
            MemoryExtraction: Structured memory extraction with category, data, and metadata.
            
        Raises:
            RuntimeError: If SLM is not available or analysis fails.
        """
        if self.llm is None:
            raise RuntimeError("SLM model not available")
        
        logger.debug(f"Analyzing text: {text[:100]}...")
        
        # Create the classification prompt
        prompt = self._create_classification_prompt(text, context)
        
        try:
            # Generate response from SLM in a separate thread to avoid blocking event loop
            import asyncio
            loop = asyncio.get_running_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.llm(
                    prompt,
                    max_tokens=256,
                    temperature=0.2,
                    top_p=0.9,
                    stop=["\n\n", "Text:", "Analyze"],
                    echo=False
                )
            )
            
            # Extract the generated text
            generated_text = response['choices'][0]['text'].strip()
            logger.info(f"SLM raw output: '{generated_text[:500]}'")
            
            # Parse the natural language response into structured JSON
            extraction = self._parse_slm_response(generated_text, text)
            
            # Create relationships based on entities and context
            extraction.links = self._create_entity_relationships(extraction, context)
            
            logger.debug(f"Extracted memory: category={extraction.category.value}, confidence={extraction.confidence:.2f}, links={len(extraction.links)}")
            return extraction
            
        except Exception as e:
            logger.error(f"Failed to analyze text: {e}")
            return self._create_fallback_extraction(text)
    
    def _create_classification_prompt(self, text: str, context: ConversationContext) -> str:
        """
        Create the prompt template for memory classification and extraction.
        
        Args:
            text: The text to analyze.
            context: Conversation context for better classification.
            
        Returns:
            str: Formatted prompt for the SLM.
        """
        # Format recent topics and entities for context
        recent_topics_str = ", ".join(context.recent_topics[-5:]) if context.recent_topics else "None"
        
        prompt = f"""[INST] You are a Memory Extraction AI. Your goal is to identify and extract important information from user messages.
        
Categories:
- PINNED: Core identity, names, strong preferences (e.g. "My name is John", "I hate cilantro") - NEVER FORGET THESE.
- CRITICAL: Deadlines, commitments, important facts (e.g. "Meeting at 2pm", "I moved to Berlin")
- EPISODIC: Life events, experiences (e.g. "I went hiking", "I saw a movie")
- RELATIONAL: Facts about other people/pets (e.g. "My wife is Sarah")
- TEMPORARY: Short-term planning (e.g. "Remind me in 5 mins")
- DISCARD: Greetings, chit-chat, vague acknowledgments (e.g. "Hi", "Thanks", "Cool")

Format your response exactly like this:
Category: [CATEGORY]
Confidence: [0.0-1.0]
Key info: [Concise summary of the memory]
Entities: [List of people/places/things, even if lowercase. Convert to Title Case.]
Triples: [Subject, Predicate, Object]

Examples:

Input: "My name is Sarah"
Category: PINNED
Confidence: 1.0
Key info: User's name is Sarah
Triples: [User, HAS_NAME, Sarah]

Input: "I have a meeting on Friday at 2pm"
Category: CRITICAL
Confidence: 0.95
Key info: Meeting scheduled for Friday 2pm
Triples: [User, HAS_MEETING, Friday 2pm]

Input: "Hi there"
Category: DISCARD
Confidence: 0.90
Key info: Greeting
Triples: []

Input: "{text}"
[/INST]
Response:"""
        
        return prompt
    
    def _parse_slm_response(self, response_text: str, original_text: str) -> MemoryExtraction:
        """
        Parse the SLM natural language response and create structured JSON.
        
        Args:
            response_text: The raw response from the SLM.
            original_text: The original text being analyzed.
            
        Returns:
            MemoryExtraction: Parsed memory extraction.
        """
        try:
            # Parse the natural language response
            response_lower = response_text.lower().strip()
            original_lower = original_text.lower()
            
            # Extract category from SLM response
            # Try structured format first (Category: XXX), then simple format (just the category name)
            category = None
            confidence = 0.5  # Default
            
            # Method 1: Look for "Category:" prefix (structured format)
            if "category:" in response_lower:
                category_line = response_text.split("Category:")[-1].split("\n")[0].strip().lower()
                if "pinned" in category_line:
                    category = MemoryCategory.PINNED
                elif "critical" in category_line:
                    category = MemoryCategory.CRITICAL
                elif "episodic" in category_line:
                    category = MemoryCategory.EPISODIC
                elif "relational" in category_line:
                    category = MemoryCategory.RELATIONAL
                elif "temporary" in category_line:
                    category = MemoryCategory.TEMPORARY
                elif "discard" in category_line:
                    category = MemoryCategory.DISCARD
            
            # Method 2: Simple category detection (fine-tuned model output)
            # Check if response is just a category name or starts with one
            if category is None:
                # Clean response and check first word/line
                first_word = response_lower.split()[0] if response_lower.split() else ""
                first_line = response_lower.split("\n")[0].strip()
                
                # Check both first word and first line
                for check_text in [first_word, first_line]:
                    if "pinned" in check_text:
                        category = MemoryCategory.PINNED
                        confidence = 0.85
                        break
                    elif "critical" in check_text:
                        category = MemoryCategory.CRITICAL
                        confidence = 0.85
                        break
                    elif "episodic" in check_text:
                        category = MemoryCategory.EPISODIC
                        confidence = 0.85
                        break
                    elif "relational" in check_text:
                        category = MemoryCategory.RELATIONAL
                        confidence = 0.85
                        break
                    elif "temporary" in check_text:
                        category = MemoryCategory.TEMPORARY
                        confidence = 0.85
                        break
                    elif "discard" in check_text:
                        category = MemoryCategory.DISCARD
                        confidence = 0.85
                        break
            
            # Final fallback
            if category is None:
                category = MemoryCategory.DISCARD
            
            # Extract confidence (only if structured format has it, otherwise keep default from Method 2)
            if "confidence:" in response_lower:
                try:
                    conf_line = response_text.split("Confidence:")[-1].split("\n")[0].strip()
                    # Extract first number found
                    import re
                    numbers = re.findall(r'0?\.\d+|[01]\.?\d*', conf_line)
                    if numbers:
                        confidence = max(0.0, min(1.0, float(numbers[0])))
                except (ValueError, IndexError):
                    pass  # Keep existing confidence
            
            # Extract key information
            key_info = ""
            if "key info:" in response_lower:
                key_info = response_text.split("Key info:")[-1].split("\n")[0].strip()
            
            # Extract entities (SLM extracted)
            entities = []
            if "entities:" in response_lower:
                try:
                    entities_line = response_text.split("Entities:")[-1].split("\n")[0].strip()
                    # Remove brackets if present
                    entities_line = entities_line.replace("[", "").replace("]", "")
                    # Split by comma
                    raw_entities = [e.strip() for e in entities_line.split(",") if e.strip()]
                    # Title Case normalization handled later, but good to have here
                    entities = raw_entities
                except Exception as e:
                    logger.warning(f"Failed to parse entities: {e}")

            # Extract triples
            triples = []
            if "triples:" in response_lower:
                try:
                    triples_text = response_text.split("Triples:")[-1].strip()
                    import re
                    # Robust regex for [S, P, O]
                    matches = re.findall(r'\[(.*?)\]', triples_text)
                    for match in matches:
                        parts = match.split(',')
                        if len(parts) >= 3:
                            s, p, o = [x.strip() for x in parts[:3]]
                            if s and p and o:
                                triples.append((s, p, o))
                except Exception as e:
                    logger.warning(f"Failed to parse triples: {e}")

            # Build structured data based on category and extracted info
            structured_data = self._build_structured_data(
                category, original_text, key_info, response_text, entities, triples
            )
            structured_data['triples'] = triples
            
            # Set stability based on category
            stability = {
                MemoryCategory.PINNED: 1.0,
                MemoryCategory.CRITICAL: 0.9,
                MemoryCategory.EPISODIC: 0.6,
                MemoryCategory.RELATIONAL: 0.7,
                MemoryCategory.TEMPORARY: 0.3,
                MemoryCategory.DISCARD: 0.1
            }.get(category, 0.5)
            
            # Determine decay policy
            decay_policy = self._get_decay_policy(category)
            
            logger.info(f"SLM classification: category={category.value}, confidence={confidence:.2f}")
            
            return MemoryExtraction(
                category=category,
                structured_data=structured_data,
                confidence=confidence,
                stability=stability,
                decay_policy=decay_policy,
                links=[],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse SLM response: {e}")
            return self._create_fallback_extraction(original_text)
    
    def _build_structured_data(
        self, 
        category: MemoryCategory, 
        original_text: str, 
        key_info: str,
        full_response: str,
        slm_entities: List[str] = None,
        slm_triples: List[Tuple[str, str, str]] = None
    ) -> Dict[str, Any]:
        """
        Build structured data dictionary based on category and extracted information.
        
        Args:
            category: The memory category.
            original_text: The original text being analyzed.
            key_info: Key information extracted from SLM response.
            full_response: Full SLM response for additional parsing.
            slm_entities: Entities extracted by SLM (optional).
            slm_triples: Triples extracted by SLM (optional).
            
        Returns:
            Dict with structured data fields.
        """
        # Initialize all fields INCLUDING content
        structured_data = {
            'content': original_text,  # Store the original text
            'preferences': [],
            'facts': [],
            'entities': [],
            'relationships': [],
            'commitments': [],
            'constraints': []
        }
        
        if slm_entities:
            # Use SLM extracted entities if available
            structured_data['entities'] = list(set(slm_entities))
        else:
            # Failover 1: Extract from Triples (Subject and Object are usually entities)
            # This captures 'jacky' from [jacky, LOVES, chicken] even if Regex fails
            triple_entities = []
            triples = slm_triples or []
            for s, p, o in triples:
                if len(s) > 1: triple_entities.append(s)
                if len(o) > 1: triple_entities.append(o) # Object might be a concept/entity
            
            if triple_entities:
                 structured_data['entities'] = list(set(triple_entities))
            else:
                # Failover 2: Fallback to Regex extraction
                import re
                entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', original_text)
                structured_data['entities'] = list(set(entities))[:5]
        
        # Category-specific extraction
        if category == MemoryCategory.PINNED:
            # Look for preferences and constraints
            if any(word in original_text.lower() for word in ['prefer', 'like', 'want', 'always', 'never']):
                structured_data['preferences'].append(key_info or original_text[:100])
            if any(word in original_text.lower() for word in ['must', 'should', 'cannot', 'dont']):
                structured_data['constraints'].append(key_info or original_text[:100])
        
        elif category == MemoryCategory.CRITICAL:
            # Look for facts and commitments
            if any(word in original_text.lower() for word in ['will', 'promise', 'commit', 'deadline']):
                structured_data['commitments'].append(key_info or original_text[:100])
            else:
                structured_data['facts'].append(key_info or original_text[:100])
        
        elif category == MemoryCategory.EPISODIC:
            # Store as facts
            structured_data['facts'].append(key_info or original_text[:100])
        
        elif category == MemoryCategory.RELATIONAL:
            # Look for relationships
            if any(word in original_text.lower() for word in ['is', 'are', 'was', 'were', 'related', 'connected']):
                structured_data['relationships'].append(key_info or original_text[:100])
        
        return structured_data
    
    def _get_decay_policy(self, category: MemoryCategory) -> DecayPolicy:
        """
        Get the appropriate decay policy for a memory category.
        
        Args:
            category: The memory category.
            
        Returns:
            DecayPolicy: The corresponding decay policy.
        """
        decay_mapping = {
            MemoryCategory.PINNED: DecayPolicy.NO_DECAY,
            MemoryCategory.CRITICAL: DecayPolicy.VERY_SLOW,
            MemoryCategory.EPISODIC: DecayPolicy.MEDIUM,
            MemoryCategory.RELATIONAL: DecayPolicy.MEDIUM,
            MemoryCategory.TEMPORARY: DecayPolicy.FAST,
            MemoryCategory.DISCARD: DecayPolicy.FAST,  # Won't be stored anyway
        }
        return decay_mapping.get(category, DecayPolicy.MEDIUM)
    
    def _create_fallback_extraction(self, text: str) -> MemoryExtraction:
        """
        Create a fallback memory extraction when SLM analysis fails.
        
        Args:
            text: The original text being analyzed.
            
        Returns:
            MemoryExtraction: Fallback extraction with DISCARD category.
        """
        logger.info("Creating fallback extraction for failed analysis")
        
        return MemoryExtraction(
            category=MemoryCategory.DISCARD,
            structured_data={
                'content': text,
                'preferences': [],
                'facts': [],
                'entities': [],
                'relationships': [],
                'commitments': [],
                'constraints': []
            },
            confidence=0.1,  # Low confidence for fallback
            stability=0.1,
            decay_policy=DecayPolicy.FAST,
            links=[],
            timestamp=datetime.now()
        )
    
    def _create_entity_relationships(
        self, 
        extraction: MemoryExtraction, 
        context: ConversationContext
    ) -> List[MemoryLink]:
        """
        Create relationship links based on entities and context.
        
        This creates connections between the new memory and existing memories
        that share entities or topics.
        
        Args:
            extraction: The memory extraction with entities.
            context: Conversation context with active entities.
            
        Returns:
            List of MemoryLink objects.
        """
        links = []
        
        # Extract entities from this memory
        new_entities = set(extraction.structured_data.get('entities', []))
        
        # Find overlapping entities with context
        context_entities = set(context.active_entities)
        shared_entities = new_entities.intersection(context_entities)
        
        if shared_entities:
            logger.debug(f"Found {len(shared_entities)} shared entities: {shared_entities}")
            
            # For now, we'll create links in the graph_engine when we have memory IDs
            # This method prepares the data for relationship creation
            # The actual linking happens in graph_engine after storage
        
        # Detect relationship patterns in text
        text_lower = extraction.structured_data.get('content', '').lower()
        
        # Name/nickname relationships
        if 'call me' in text_lower or 'known as' in text_lower:
            # This creates an ALSO_KNOWN_AS relationship
            # Will be handled by graph_engine
            logger.debug("Detected name/nickname relationship pattern")
        
        # Preference relationships  
        if any(word in text_lower for word in ['prefer', 'like', 'love', 'hate']):
            # This creates a PREFERS/DISLIKES relationship
            logger.debug("Detected preference relationship pattern")
        
        return links
    
    def is_available(self) -> bool:
        """
        Check if the SLM is available and ready for analysis.
        
        Returns:
            bool: True if SLM is loaded and ready, False otherwise.
        """
        return self.llm is not None


# Utility functions for testing and development

def create_test_analyzer() -> MemoryAnalyzer:
    """
    Create a test memory analyzer with mock configuration.
    Used for testing when actual SLM model is not available.
    
    Returns:
        MemoryAnalyzer: Test analyzer instance.
    """
    from .config import SLMConfig
    
    config = SLMConfig(
        model_path="test_model.gguf",
        n_ctx=2048,
        n_gpu_layers=0
    )
    
    # Create analyzer without trying to load the model
    analyzer = object.__new__(MemoryAnalyzer)
    analyzer.config = config
    analyzer.llm = None
    
    return analyzer


def analyze_text_simple(text: str, category: str = "episodic") -> MemoryExtraction:
    """
    Simple text analysis for testing without SLM.
    
    Args:
        text: Text to analyze.
        category: Memory category to assign.
        
    Returns:
        MemoryExtraction: Simple extraction for testing.
    """
    try:
        mem_category = MemoryCategory(category.lower())
    except ValueError:
        mem_category = MemoryCategory.EPISODIC
    
    # Simple keyword extraction for testing
    structured_data = {
        'preferences': [],
        'facts': [text] if len(text) > 10 else [],
        'entities': [],
        'relationships': [],
        'commitments': [],
        'constraints': []
    }
    
    analyzer = create_test_analyzer()
    return MemoryExtraction(
        category=mem_category,
        structured_data=structured_data,
        confidence=0.8,
        stability=0.7,
        decay_policy=analyzer._get_decay_policy(mem_category),
        links=[],
        timestamp=datetime.now()
    )