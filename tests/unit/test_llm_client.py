"""
Unit tests for the Local LLM Client.

Tests the LocalLLMClient class including model loading, response generation,
streaming, error handling, and conversation history limiting.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from src.llm_client import LocalLLMClient
from src.models import Message, ConversationContext
from src.config import MainLLMConfig


@pytest.fixture
def mock_llm_config():
    """Create a mock Main LLM configuration."""
    return MainLLMConfig(
        model_path="./data/models/test-model.gguf",
        n_ctx=4096,
        n_gpu_layers=-1
    )


@pytest.fixture
def sample_messages():
    """Create sample conversation messages."""
    return [
        Message(role="user", content="Hello", timestamp=datetime.now()),
        Message(role="assistant", content="Hi there!", timestamp=datetime.now()),
        Message(role="user", content="How are you?", timestamp=datetime.now()),
        Message(role="assistant", content="I'm doing well, thanks!", timestamp=datetime.now()),
    ]


@pytest.fixture
def mock_llama():
    """Create a mock Llama instance."""
    mock = Mock()
    mock.create_chat_completion = Mock(return_value={
        'choices': [{
            'message': {
                'content': 'This is a test response.'
            }
        }]
    })
    return mock


class TestLocalLLMClient:
    """Test suite for LocalLLMClient."""
    
    def test_initialization_success(self, mock_llm_config):
        """Test successful client initialization."""
        with patch('src.llm_client.Llama') as mock_llama_class:
            mock_llama_class.return_value = Mock()
            
            client = LocalLLMClient(mock_llm_config, max_history_turns=10)
            
            assert client.config == mock_llm_config
            assert client.max_history_turns == 10
            assert client.llm is not None
            mock_llama_class.assert_called_once_with(
                model_path=mock_llm_config.model_path,
                n_ctx=mock_llm_config.n_ctx,
                n_gpu_layers=mock_llm_config.n_gpu_layers,
                verbose=False
            )
    
    def test_initialization_with_retries(self, mock_llm_config):
        """Test model loading with retries on failure."""
        with patch('src.llm_client.Llama') as mock_llama_class:
            # Fail twice, succeed on third attempt
            mock_llama_class.side_effect = [
                Exception("Load failed"),
                Exception("Load failed again"),
                Mock()
            ]
            
            client = LocalLLMClient(mock_llm_config)
            
            assert client.llm is not None
            assert mock_llama_class.call_count == 3
    
    def test_initialization_failure_after_retries(self, mock_llm_config):
        """Test that initialization fails after max retries."""
        with patch('src.llm_client.Llama') as mock_llama_class:
            mock_llama_class.side_effect = Exception("Persistent load failure")
            
            with pytest.raises(RuntimeError) as exc_info:
                LocalLLMClient(mock_llm_config)
            
            assert "Failed to load Main LLM model after 3 attempts" in str(exc_info.value)
            assert mock_llama_class.call_count == 3
    
    def test_format_system_message(self, mock_llm_config):
        """Test system message formatting with memory context."""
        with patch('src.llm_client.Llama') as mock_llama_class:
            mock_llama_class.return_value = Mock()
            client = LocalLLMClient(mock_llm_config)
            
            memory_context = "<system_constraints>Language: English</system_constraints>"
            system_msg = client._format_system_message(memory_context)
            
            assert memory_context in system_msg
            assert "helpful AI assistant" in system_msg
            assert "memory context" in system_msg
    
    def test_get_recent_history_within_limit(self, mock_llm_config, sample_messages):
        """Test getting recent history when within limit."""
        with patch('src.llm_client.Llama') as mock_llama_class:
            mock_llama_class.return_value = Mock()
            client = LocalLLMClient(mock_llm_config, max_history_turns=10)
            
            recent = client._get_recent_history(sample_messages)
            
            assert len(recent) == len(sample_messages)
            assert recent == sample_messages
    
    def test_get_recent_history_exceeds_limit(self, mock_llm_config):
        """Test getting recent history when exceeding limit."""
        with patch('src.llm_client.Llama') as mock_llama_class:
            mock_llama_class.return_value = Mock()
            client = LocalLLMClient(mock_llm_config, max_history_turns=2)
            
            # Create 10 turns (20 messages)
            messages = []
            for i in range(10):
                messages.append(Message(role="user", content=f"Query {i}", timestamp=datetime.now()))
                messages.append(Message(role="assistant", content=f"Response {i}", timestamp=datetime.now()))
            
            recent = client._get_recent_history(messages)
            
            # Should only get last 2 turns (4 messages)
            assert len(recent) == 4
            assert recent[0].content == "Query 8"
            assert recent[-1].content == "Response 9"
    
    def test_build_messages(self, mock_llm_config, sample_messages):
        """Test building messages list for LLM."""
        with patch('src.llm_client.Llama') as mock_llama_class:
            mock_llama_class.return_value = Mock()
            client = LocalLLMClient(mock_llm_config, max_history_turns=10)
            
            query = "What's the weather?"
            memory_context = "<system_constraints>Language: English</system_constraints>"
            
            messages = client._build_messages(query, memory_context, sample_messages)
            
            # Should have: system + history + current query
            assert len(messages) == 1 + len(sample_messages) + 1
            assert messages[0]['role'] == 'system'
            assert memory_context in messages[0]['content']
            assert messages[-1]['role'] == 'user'
            assert messages[-1]['content'] == query
    
    def test_generate_success(self, mock_llm_config, sample_messages):
        """Test successful response generation."""
        with patch('src.llm_client.Llama') as mock_llama_class:
            mock_llm = Mock()
            mock_llm.create_chat_completion = Mock(return_value={
                'choices': [{
                    'message': {
                        'content': 'Generated response text'
                    }
                }]
            })
            mock_llama_class.return_value = mock_llm
            
            client = LocalLLMClient(mock_llm_config)
            
            response = client.generate(
                query="Test query",
                memory_context="<memory>Test context</memory>",
                conversation_history=sample_messages
            )
            
            assert response == 'Generated response text'
            mock_llm.create_chat_completion.assert_called_once()
    
    def test_generate_with_retry_on_failure(self, mock_llm_config, sample_messages):
        """Test response generation with retry on failure."""
        with patch('src.llm_client.Llama') as mock_llama_class:
            mock_llm = Mock()
            # Fail first, succeed on retry
            mock_llm.create_chat_completion = Mock(side_effect=[
                Exception("Generation failed"),
                {
                    'choices': [{
                        'message': {
                            'content': 'Retry response'
                        }
                    }]
                }
            ])
            mock_llama_class.return_value = mock_llm
            
            client = LocalLLMClient(mock_llm_config)
            
            response = client.generate(
                query="Test query",
                memory_context="<memory>Test context</memory>",
                conversation_history=sample_messages
            )
            
            assert response == 'Retry response'
            assert mock_llm.create_chat_completion.call_count == 2
    
    def test_generate_failure_after_retry(self, mock_llm_config, sample_messages):
        """Test that generation fails after retry."""
        with patch('src.llm_client.Llama') as mock_llama_class:
            mock_llm = Mock()
            mock_llm.create_chat_completion = Mock(side_effect=Exception("Persistent failure"))
            mock_llama_class.return_value = mock_llm
            
            client = LocalLLMClient(mock_llm_config)
            
            with pytest.raises(RuntimeError) as exc_info:
                client.generate(
                    query="Test query",
                    memory_context="<memory>Test context</memory>",
                    conversation_history=sample_messages
                )
            
            assert "Failed to generate response" in str(exc_info.value)
            assert mock_llm.create_chat_completion.call_count == 2
    
    def test_generate_without_loaded_model(self, mock_llm_config):
        """Test that generation fails if model not loaded."""
        with patch('src.llm_client.Llama') as mock_llama_class:
            mock_llama_class.return_value = Mock()
            client = LocalLLMClient(mock_llm_config)
            client.llm = None  # Simulate unloaded model
            
            with pytest.raises(RuntimeError) as exc_info:
                client.generate(
                    query="Test query",
                    memory_context="<memory>Test context</memory>",
                    conversation_history=[]
                )
            
            assert "LLM model not loaded" in str(exc_info.value)
    
    def test_generate_stream_success(self, mock_llm_config, sample_messages):
        """Test successful streaming response generation."""
        with patch('src.llm_client.Llama') as mock_llama_class:
            mock_llm = Mock()
            # Simulate streaming chunks
            mock_llm.create_chat_completion = Mock(return_value=[
                {'choices': [{'delta': {'content': 'Hello'}}]},
                {'choices': [{'delta': {'content': ' world'}}]},
                {'choices': [{'delta': {'content': '!'}}]},
            ])
            mock_llama_class.return_value = mock_llm
            
            client = LocalLLMClient(mock_llm_config)
            
            chunks = list(client.generate_stream(
                query="Test query",
                memory_context="<memory>Test context</memory>",
                conversation_history=sample_messages
            ))
            
            assert chunks == ['Hello', ' world', '!']
            mock_llm.create_chat_completion.assert_called_once()
    
    def test_generate_stream_without_loaded_model(self, mock_llm_config):
        """Test that streaming fails if model not loaded."""
        with patch('src.llm_client.Llama') as mock_llama_class:
            mock_llama_class.return_value = Mock()
            client = LocalLLMClient(mock_llm_config)
            client.llm = None  # Simulate unloaded model
            
            with pytest.raises(RuntimeError) as exc_info:
                list(client.generate_stream(
                    query="Test query",
                    memory_context="<memory>Test context</memory>",
                    conversation_history=[]
                ))
            
            assert "LLM model not loaded" in str(exc_info.value)
    
    def test_conversation_history_limit_enforced(self, mock_llm_config):
        """Test that conversation history is limited to max_history_turns."""
        with patch('src.llm_client.Llama') as mock_llama_class:
            mock_llm = Mock()
            mock_llm.create_chat_completion = Mock(return_value={
                'choices': [{
                    'message': {
                        'content': 'Response'
                    }
                }]
            })
            mock_llama_class.return_value = mock_llm
            
            client = LocalLLMClient(mock_llm_config, max_history_turns=3)
            
            # Create 10 turns (20 messages)
            messages = []
            for i in range(10):
                messages.append(Message(role="user", content=f"Query {i}", timestamp=datetime.now()))
                messages.append(Message(role="assistant", content=f"Response {i}", timestamp=datetime.now()))
            
            client.generate(
                query="Final query",
                memory_context="<memory>Test</memory>",
                conversation_history=messages
            )
            
            # Check that only last 3 turns (6 messages) + system + current query were used
            call_args = mock_llm.create_chat_completion.call_args
            messages_arg = call_args[1]['messages']
            
            # Should be: 1 system + 6 history + 1 current = 8 total
            assert len(messages_arg) == 8
            assert messages_arg[0]['role'] == 'system'
            assert messages_arg[-1]['role'] == 'user'
            assert messages_arg[-1]['content'] == 'Final query'
