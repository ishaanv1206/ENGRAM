"""
Unit tests for the Gradio Web Interface.

Tests the GradioUI class functionality including initialization,
message handling, and conversation management.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.gradio_ui import GradioUI
from src.models import ConversationContext, create_conversation_context


class TestGradioUI:
    """Test cases for the GradioUI class."""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock CognitivePipeline for testing."""
        pipeline = Mock()
        pipeline.process_turn = AsyncMock(return_value="Test response")
        return pipeline
    
    @pytest.fixture
    def gradio_ui(self, mock_pipeline):
        """Create a GradioUI instance for testing."""
        return GradioUI(mock_pipeline)
    
    def test_initialization(self, mock_pipeline):
        """Test GradioUI initialization."""
        ui = GradioUI(mock_pipeline)
        
        assert ui.pipeline == mock_pipeline
        assert ui.context is not None
        assert isinstance(ui.context, ConversationContext)
        assert ui.context.turn_count == 0
        assert not ui.processing
    
    def test_reset_conversation(self, gradio_ui):
        """Test conversation reset functionality."""
        # Modify context to simulate an ongoing conversation
        gradio_ui.context.turn_count = 5
        old_session_id = gradio_ui.context.session_id
        
        # Reset conversation
        gradio_ui._reset_conversation()
        
        # Verify reset
        assert gradio_ui.context.turn_count == 0
        assert gradio_ui.context.session_id != old_session_id
        assert len(gradio_ui.context.conversation_history) == 0
    
    @pytest.mark.asyncio
    async def test_handle_text_success(self, gradio_ui, mock_pipeline):
        """Test successful text message handling."""
        message = "Hello, how are you?"
        history = []
        
        # Mock pipeline response
        mock_pipeline.process_turn.return_value = "I'm doing well, thank you!"
        
        # Handle the message
        empty_msg, updated_history = await gradio_ui.handle_text(message, history)
        
        # Verify results
        assert empty_msg == ""
        assert len(updated_history) == 1
        assert updated_history[0] == [message, "I'm doing well, thank you!"]
        
        # Verify pipeline was called correctly
        mock_pipeline.process_turn.assert_called_once_with(message, gradio_ui.context)
    
    @pytest.mark.asyncio
    async def test_handle_text_empty_message(self, gradio_ui):
        """Test handling of empty or whitespace-only messages."""
        history = [["Previous", "Response"]]
        
        # Test empty message
        empty_msg, updated_history = await gradio_ui.handle_text("", history)
        assert empty_msg == ""
        assert updated_history == history
        
        # Test whitespace-only message
        empty_msg, updated_history = await gradio_ui.handle_text("   ", history)
        assert empty_msg == ""
        assert updated_history == history
    
    @pytest.mark.asyncio
    async def test_handle_text_while_processing(self, gradio_ui):
        """Test handling messages while already processing."""
        gradio_ui.processing = True
        message = "Test message"
        history = []
        
        empty_msg, updated_history = await gradio_ui.handle_text(message, history)
        
        # Should return wait message
        assert empty_msg == ""
        assert len(updated_history) == 1
        assert "Please wait" in updated_history[0][0]
    
    @pytest.mark.asyncio
    async def test_handle_text_pipeline_error(self, gradio_ui, mock_pipeline):
        """Test handling of pipeline errors."""
        message = "Test message"
        history = []
        
        # Mock pipeline to raise an exception
        mock_pipeline.process_turn.side_effect = Exception("Pipeline error")
        
        empty_msg, updated_history = await gradio_ui.handle_text(message, history)
        
        # Verify error handling
        assert empty_msg == ""
        assert len(updated_history) == 1
        assert updated_history[0][0] == message
        assert "error" in updated_history[0][1].lower()
        assert not gradio_ui.processing  # Should reset processing flag
    
    def test_display_response(self, gradio_ui):
        """Test response display formatting."""
        response = "Test response"
        formatted = gradio_ui.display_response(response)
        assert formatted == response
    
    def test_get_status_message(self, gradio_ui):
        """Test status message generation."""
        # Test ready state
        gradio_ui.processing = False
        status = gradio_ui.get_status_message()
        assert "Ready" in status
        
        # Test processing state
        gradio_ui.processing = True
        status = gradio_ui.get_status_message()
        assert "Processing" in status
    
    def test_clear_conversation(self, gradio_ui):
        """Test conversation clearing functionality."""
        # Set up some conversation state
        gradio_ui.context.turn_count = 3
        old_session_id = gradio_ui.context.session_id
        
        # Clear conversation
        history, status = gradio_ui.clear_conversation()
        
        # Verify clearing
        assert history == []
        assert "cleared" in status.lower()
        assert gradio_ui.context.turn_count == 0
        assert gradio_ui.context.session_id != old_session_id
    
    def test_launch_parameters(self, gradio_ui):
        """Test that launch method accepts correct parameters."""
        # Just test that the method exists and accepts the expected parameters
        # without actually launching the interface
        import inspect
        
        launch_method = getattr(gradio_ui, 'launch')
        sig = inspect.signature(launch_method)
        
        # Verify expected parameters exist
        expected_params = ['share', 'server_name', 'server_port']
        for param in expected_params:
            assert param in sig.parameters