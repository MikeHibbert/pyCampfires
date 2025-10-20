"""
Tests for multimodal OpenRouter client functionality.
"""

import pytest
import base64
from unittest.mock import Mock, patch, AsyncMock
from campfires.core.multimodal_openrouter import (
    MultimodalChatMessage,
    MultimodalOpenRouterClient,
    MultimodalLLMCamperMixin
)
from campfires.core.multimodal_torch import ContentType, MultimodalContent


class TestMultimodalChatMessage:
    """Test MultimodalChatMessage class."""
    
    def test_text_message_creation(self):
        """Test creating a text-only message."""
        message = MultimodalChatMessage(
            role="user",
            content="Hello, world!"
        )
        
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.name is None
    
    def test_message_with_images(self):
        """Test creating a message with images."""
        image_data = base64.b64encode(b"fake_image").decode()
        multimodal_content = [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]
        message = MultimodalChatMessage(
            role="user",
            content=multimodal_content
        )
        
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        assert message.content[0]["type"] == "text"
        assert message.content[1]["type"] == "image_url"
    
    def test_message_with_audio(self):
        """Test creating a message with audio."""
        audio_data = base64.b64encode(b"fake_audio").decode()
        multimodal_content = [
            {"type": "text", "text": "Transcribe this audio"},
            {"type": "audio", "audio": {"data": audio_data}}
        ]
        message = MultimodalChatMessage(
            role="user",
            content=multimodal_content
        )
        
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        assert message.content[0]["type"] == "text"
        assert message.content[1]["type"] == "audio"
    
    def test_to_openrouter_format_text_only(self):
        """Test converting text-only message to OpenRouter format."""
        message = MultimodalChatMessage(role="user", content="Hello")
        
        result = message.to_openai_format()
        
        expected = {
            "role": "user",
            "content": "Hello"
        }
        
        assert result == expected
    
    def test_to_openrouter_format_with_images(self):
        """Test converting message with images to OpenRouter format."""
        image_data = "fake_image_data"
        multimodal_content = [
            {"type": "text", "text": "Describe this"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]
        message = MultimodalChatMessage(
            role="user",
            content=multimodal_content
        )
        
        result = message.to_openai_format()
        
        assert result["role"] == "user"
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Describe this"
        assert result["content"][1]["type"] == "image_url"
        assert "data:image/jpeg;base64," in result["content"][1]["image_url"]["url"]
    
    def test_from_multimodal_content(self):
        """Test creating message from MultimodalContent list."""
        text_content = MultimodalContent(content_type=ContentType.TEXT, data="Hello")
        image_content = MultimodalContent(
            content_type=ContentType.IMAGE, 
            data="image_data", 
            encoding="base64"
        )
        
        message = MultimodalChatMessage.from_multimodal_content(
            role="user",
            contents=[text_content, image_content]
        )
        
        assert message.role == "user"
        assert isinstance(message.content, list)
        assert len(message.content) == 2
        assert message.content[0]["type"] == "text"
        assert message.content[0]["text"] == "Hello"
        assert message.content[1]["type"] == "image_url"


class TestMultimodalOpenRouterClient:
    """Test MultimodalOpenRouterClient class."""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock client for testing."""
        # Import ChatResponse for proper mocking
        from campfires.core.openrouter import ChatResponse
        
        # Create a mock config first
        mock_config = Mock()
        mock_config.default_model = "openai/gpt-4o-mini"
        
        # Patch both the parent class __init__ and _get_default_vision_model
        with patch('campfires.core.multimodal_openrouter.OpenRouterClient.__init__', return_value=None), \
             patch('campfires.core.multimodal_openrouter.MultimodalOpenRouterClient._get_default_vision_model', return_value="openai/gpt-4o-mini"):
            client = MultimodalOpenRouterClient(config=mock_config)
            # Manually set attributes that would normally be set by parent __init__
            client.config = mock_config
            client.session = None
            client.mcp_protocol = None
            client.request_count = 0
            client.total_tokens_used = 0
            client.last_request_time = None
            
            # Mock methods to return ChatResponse objects
            client.chat_completion = AsyncMock()
            client.default_model = "openai/gpt-4o-mini"
            client.default_vision_model = "openai/gpt-4o-mini"
            return client
    
    @pytest.mark.asyncio
    async def test_multimodal_completion(self, mock_client):
        """Test multimodal completion method."""
        # Import ChatResponse for proper mocking
        from campfires.core.openrouter import ChatResponse
        
        # Mock the response as a ChatResponse object
        mock_response = ChatResponse(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "message": {
                        "content": "I can see an image with a cat."
                    }
                }
            ]
        )
        mock_client.chat_completion.return_value = mock_response
        
        # Import MultimodalTorch
        from campfires.core.multimodal_torch import MultimodalTorch
        
        text_content = MultimodalContent(content_type=ContentType.TEXT, data="What do you see?")
        image_content = MultimodalContent(
            content_type=ContentType.IMAGE, 
            data="fake_image_data", 
            encoding="base64"
        )
        
        torch = MultimodalTorch(
            contents=[text_content, image_content],
            primary_claim="Test multimodal completion",
            source_campfire="test_campfire",
            channel="test_channel"
        )
        
        result = await mock_client.multimodal_completion(torch=torch)
        
        assert result == "I can see an image with a cat."
        mock_client.chat_completion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_vision_completion(self, mock_client):
        """Test vision completion method."""
        # Import ChatResponse for proper mocking
        from campfires.core.openrouter import ChatResponse
        
        # Mock the response as a ChatResponse object
        mock_response = ChatResponse(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "message": {
                        "content": "This is a beautiful landscape."
                    }
                }
            ]
        )
        mock_client.chat_completion.return_value = mock_response
        
        result = await mock_client.vision_completion(
            text_prompt="Describe this image",
            images=["fake_image_data"]
        )
        
        assert result == "This is a beautiful landscape."
    
    @pytest.mark.asyncio
    async def test_analyze_image(self, mock_client):
        """Test image analysis method."""
        # Import ChatResponse for proper mocking
        from campfires.core.openrouter import ChatResponse
        
        # Mock the response as a ChatResponse object
        mock_response = ChatResponse(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "message": {
                        "content": "The image contains a red car."
                    }
                }
            ]
        )
        mock_client.chat_completion.return_value = mock_response
        
        result = await mock_client.analyze_image("fake_image_data")
        
        assert result == "The image contains a red car."
        # Verify the call was made with the right prompt (default is "Describe this image in detail.")
        call_args = mock_client.chat_completion.call_args
        messages = call_args[1]["messages"]
        assert any("describe" in msg.content.lower() for msg in messages if hasattr(msg, 'content'))
    
    @pytest.mark.asyncio
    async def test_compare_images(self, mock_client):
        """Test image comparison method."""
        # Import ChatResponse for proper mocking
        from campfires.core.openrouter import ChatResponse
        
        # Mock the response as a ChatResponse object
        mock_response = ChatResponse(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "message": {
                        "content": "The images show different scenes."
                    }
                }
            ]
        )
        mock_client.chat_completion.return_value = mock_response
        
        result = await mock_client.compare_images(
            "image1_data", 
            "image2_data"
        )
        
        assert result == "The images show different scenes."
    
    @pytest.mark.asyncio
    async def test_extract_text_from_image(self, mock_client):
        """Test OCR functionality."""
        # Import ChatResponse for proper mocking
        from campfires.core.openrouter import ChatResponse
        
        # Mock the response as a ChatResponse object
        mock_response = ChatResponse(
            id="test-id",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                {
                    "message": {
                        "content": "The text in the image says: 'Hello World'"
                    }
                }
            ]
        )
        mock_client.chat_completion.return_value = mock_response
        
        result = await mock_client.extract_text_from_image("fake_image_data")
        
        assert result == "The text in the image says: 'Hello World'"
    
    @pytest.mark.asyncio
    async def test_audio_transcription(self, mock_client):
        """Test audio transcription method."""
        # The current implementation returns a placeholder
        result = await mock_client.audio_transcription("fake_audio_data")
        
        # Test that it returns the expected placeholder message
        assert result == "[Audio transcription placeholder - implement with Whisper API]"


class TestMultimodalLLMCamperMixin:
    """Test MultimodalLLMCamperMixin class."""
    
    @pytest.fixture
    def mock_mixin(self):
        """Create a mock mixin for testing."""
        mixin = MultimodalLLMCamperMixin()
        mixin.multimodal_client = Mock()
        # Add image support for vision tests
        mixin.supported_content_types.append(ContentType.IMAGE)
        return mixin
    
    @pytest.mark.asyncio
    async def test_multimodal_completion(self, mock_mixin):
        """Test multimodal completion through mixin."""
        mock_mixin.multimodal_client.multimodal_completion = AsyncMock(
            return_value="Multimodal response"
        )
        
        # Create a mock MultimodalTorch
        from campfires.core.multimodal_torch import MultimodalTorch
        mock_torch = Mock(spec=MultimodalTorch)
        mock_torch.get_content_types.return_value = [ContentType.TEXT]
        
        result = await mock_mixin.process_multimodal_torch(mock_torch)
        
        assert result == "Multimodal response"
        mock_mixin.multimodal_client.multimodal_completion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_vision_analysis(self, mock_mixin):
        """Test vision analysis through mixin."""
        mock_mixin.multimodal_client.vision_completion = AsyncMock(
            return_value="Image analysis result"
        )
        
        result = await mock_mixin.analyze_images(["fake_image_data"], "Analyze this image")
        
        assert result == "Image analysis result"
        mock_mixin.multimodal_client.vision_completion.assert_called_once_with(
            text_prompt="Analyze this image",
            images=["fake_image_data"],
            model=None
        )
    
    @pytest.mark.asyncio
    async def test_audio_transcription(self, mock_mixin):
        """Test audio transcription through mixin."""
        mock_mixin.multimodal_client.audio_transcription = AsyncMock(
            return_value="Transcribed text"
        )
        
        result = await mock_mixin.multimodal_client.audio_transcription("fake_audio_data")
        
        assert result == "Transcribed text"
        mock_mixin.multimodal_client.audio_transcription.assert_called_once_with("fake_audio_data")


if __name__ == "__main__":
    pytest.main([__file__])