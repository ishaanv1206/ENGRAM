"""
Integration tests for the Local LLM Client.

Tests the LocalLLMClient with real model loading and generation.
These tests require the actual model files to be present.
"""

import pytest
import os
from datetime import datetime
from src.llm_client import LocalLLMClient
from src.models import Message
from src.config import MainLLMConfig


@pytest.fixture
def real_llm_config():
    """Create a real Main LLM configuration from environment."""
    model_path = os.getenv('MAIN_LLM_MODEL_PATH', './data/models/Llama-3.2-3B-Instruct-uncensored-Q6_K_L.gguf')
    
    # Skip if model doesn't exist
    if not os.path.exists(model_path):
        pytest.skip(f"Model file not found: {model_path}")
    
    return MainLLMConfig(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1
    )


@pytest.fixture
def sample_conversation():
    """Create a sample conversation history."""
    return [
        Message(role="user", content="My name is Alice", timestamp=datetime.now()),
        Message(role="assistant", content="Nice to meet you, Alice!", timestamp=datetime.now()),
        Message(role="user", content="I like pizza", timestamp=datetime.now()),
        Message(role="assistant", content="Pizza is delicious!", timestamp=datetime.now()),
    ]


class TestLocalLLMClientIntegration:
    """Integration test suite for LocalLLMClient with real models."""
    
    def test_model_loading(self, real_llm_config):
        """Test that the model loads successfully."""
        client = LocalLLMClient(real_llm_config, max_history_turns=10)
        
        assert client.llm is not None
        assert client.config == real_llm_config
    
    def test_generate_response(self, real_llm_config, sample_conversation):
        """Test generating a response with real model."""
        client = LocalLLMClient(real_llm_config, max_history_turns=10)
        
        memory_context = """<system_constraints>
Language: English
Style: Friendly and concise
</system_constraints>"""
        
        response = client.generate(
            query="What's my name?",
            memory_context=memory_context,
            conversation_history=sample_conversation,
            temperature=0.7,
            max_tokens=100
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"\nGenerated response: {response}")
    
    def test_conversation_history_limit(self, real_llm_config):
        """Test that conversation history is properly limited."""
        client = LocalLLMClient(real_llm_config, max_history_turns=2)
        
        # Create 10 turns
        long_history = []
        for i in range(10):
            long_history.append(Message(role="user", content=f"Message {i}", timestamp=datetime.now()))
            long_history.append(Message(role="assistant", content=f"Response {i}", timestamp=datetime.now()))
        
        memory_context = "<system_constraints>Language: English</system_constraints>"
        
        response = client.generate(
            query="Final question",
            memory_context=memory_context,
            conversation_history=long_history,
            temperature=0.7,
            max_tokens=50
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"\nResponse with limited history: {response}")
    
    def test_generate_with_memory_context(self, real_llm_config):
        """Test that memory context influences response."""
        client = LocalLLMClient(real_llm_config, max_history_turns=10)
        
        memory_context = """<system_constraints>
Language: English
Style: Professional
</system_constraints>

<critical_memory>
- User prefers: vegetarian food
- Constraint: no meat products
</critical_memory>"""
        
        conversation = [
            Message(role="user", content="I'm planning dinner", timestamp=datetime.now()),
            Message(role="assistant", content="That sounds nice!", timestamp=datetime.now()),
        ]
        
        response = client.generate(
            query="What should I cook?",
            memory_context=memory_context,
            conversation_history=conversation,
            temperature=0.7,
            max_tokens=100
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"\nResponse with memory context: {response}")
    
    @pytest.mark.skip(reason="Streaming test - run manually if needed")
    def test_generate_stream(self, real_llm_config, sample_conversation):
        """Test streaming response generation."""
        client = LocalLLMClient(real_llm_config, max_history_turns=10)
        
        memory_context = "<system_constraints>Language: English</system_constraints>"
        
        chunks = []
        for chunk in client.generate_stream(
            query="Tell me a short joke",
            memory_context=memory_context,
            conversation_history=sample_conversation,
            temperature=0.7,
            max_tokens=100
        ):
            chunks.append(chunk)
            print(chunk, end='', flush=True)
        
        full_response = ''.join(chunks)
        assert len(full_response) > 0
        print(f"\n\nFull streamed response: {full_response}")
