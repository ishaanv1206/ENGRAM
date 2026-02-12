"""
Integration tests for the Gradio Web Interface.

Tests the GradioUI integration with the cognitive pipeline components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from src.gradio_ui import GradioUI
from src.models import ConversationContext, create_conversation_context


class TestGradioUIIntegration:
    """Integration test cases for the GradioUI class."""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock CognitivePipeline that simulates real behavior."""
        pipeline = Mock()
        
        async def mock_process_turn(text: str, context: ConversationContext) -> str:
            """Mock process_turn that simulates memory-aware responses."""
            # Simulate processing delay
            await asyncio.sleep(0.01)
            
            # Update context as the real pipeline would
            context.turn_count += 1
            
            # Generate contextual responses based on input
            if "hello" in text.lower():
                return "Hello! I'm ready to help you. I'll remember our conversation as we talk."
            elif "remember" in text.lower():
                return "Yes, I have a memory system that stores and retrieves relevant information from our conversations."
            elif "name" in text.lower():
                return "I'm an AI assistant with cognitive memory capabilities. What would you like me to remember about you?"
            else:
                return f"I understand you said: '{text}'. I'll remember this for future reference."
        
        pipeline.process_turn = AsyncMock(side_effect=mock_process_turn)
        return pipeline
    
    @pytest.fixture
    def gradio_ui(self, mock_pipeline):
        """Create a GradioUI instance with mock pipeline."""
        return GradioUI(mock_pipeline)
    
    @pytest.mark.asyncio
    async def test_conversation_flow(self, gradio_ui, mock_pipeline):
        """Test a complete conversation flow through the UI."""
        history = []
        
        # First message
        message1 = "Hello, my name is Alice"
        empty_msg, history = await gradio_ui.handle_text(message1, history)
        
        assert empty_msg == ""
        assert len(history) == 1
        assert history[0][0] == message1
        assert "Hello" in history[0][1]
        assert gradio_ui.context.turn_count == 1
        
        # Second message
        message2 = "Can you remember my name?"
        empty_msg, history = await gradio_ui.handle_text(message2, history)
        
        assert len(history) == 2
        assert history[1][0] == message2
        assert gradio_ui.context.turn_count == 2
        
        # Verify pipeline was called correctly
        assert mock_pipeline.process_turn.call_count == 2
    
    @pytest.mark.asyncio
    async def test_context_persistence(self, gradio_ui):
        """Test that conversation context persists across messages."""
        initial_session_id = gradio_ui.context.session_id
        initial_turn_count = gradio_ui.context.turn_count
        
        # Send a message
        await gradio_ui.handle_text("Test message", [])
        
        # Verify context was updated but session persists
        assert gradio_ui.context.session_id == initial_session_id
        assert gradio_ui.context.turn_count == initial_turn_count + 1
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, gradio_ui, mock_pipeline):
        """Test that UI recovers gracefully from pipeline errors."""
        # Make pipeline raise an exception
        mock_pipeline.process_turn.side_effect = Exception("Pipeline error")
        
        history = []
        message = "This will cause an error"
        
        empty_msg, updated_history = await gradio_ui.handle_text(message, history)
        
        # Verify error was handled gracefully
        assert empty_msg == ""
        assert len(updated_history) == 1
        assert updated_history[0][0] == message
        assert "error" in updated_history[0][1].lower()
        assert not gradio_ui.processing  # Processing flag should be reset
    
    def test_conversation_reset(self, gradio_ui):
        """Test conversation reset functionality."""
        # Simulate some conversation state
        gradio_ui.context.turn_count = 5
        original_session_id = gradio_ui.context.session_id
        
        # Reset conversation
        history, status = gradio_ui.clear_conversation()
        
        # Verify reset
        assert history == []
        assert "cleared" in status.lower()
        assert gradio_ui.context.turn_count == 0
        assert gradio_ui.context.session_id != original_session_id
    
    def test_status_indicators(self, gradio_ui):
        """Test status indicator functionality."""
        # Test ready state
        gradio_ui.processing = False
        status = gradio_ui.get_status_message()
        assert "Ready" in status or "✅" in status
        
        # Test processing state
        gradio_ui.processing = True
        status = gradio_ui.get_status_message()
        assert "Processing" in status or "🔄" in status
    
    @pytest.mark.asyncio
    async def test_concurrent_message_handling(self, gradio_ui):
        """Test that concurrent messages are handled properly."""
        # Start processing a message
        gradio_ui.processing = True
        
        # Try to send another message while processing
        history = []
        empty_msg, updated_history = await gradio_ui.handle_text("Second message", history)
        
        # Should get a wait message
        assert len(updated_history) == 1
        assert "wait" in updated_history[0][0].lower()
    
    def test_ui_component_interface(self, gradio_ui):
        """Test that UI has the required interface methods."""
        # Verify all required methods exist
        required_methods = ['handle_text', 'display_response', 'launch', 'clear_conversation']
        
        for method_name in required_methods:
            assert hasattr(gradio_ui, method_name)
            assert callable(getattr(gradio_ui, method_name))
        
        # Verify launch method signature
        import inspect
        launch_sig = inspect.signature(gradio_ui.launch)
        expected_params = ['share', 'server_name', 'server_port']
        
        for param in expected_params:
            assert param in launch_sig.parameters