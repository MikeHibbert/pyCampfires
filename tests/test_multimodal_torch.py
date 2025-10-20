"""
Tests for multimodal torch functionality.
"""

import pytest
import base64
from datetime import datetime
from unittest.mock import Mock, patch
from campfires.core.multimodal_torch import (
    ContentType,
    MultimodalContent,
    MultimodalTorch
)


class TestContentType:
    """Test ContentType enum."""
    
    def test_content_type_values(self):
        """Test that all expected content types are available."""
        assert ContentType.TEXT == "text"
        assert ContentType.IMAGE == "image"
        assert ContentType.AUDIO == "audio"
        assert ContentType.VIDEO == "video"
        assert ContentType.DOCUMENT == "document"


class TestMultimodalContent:
    """Test MultimodalContent class."""
    
    def test_text_content_creation(self):
        """Test creating text content."""
        content = MultimodalContent(
            content_type=ContentType.TEXT,
            data="Hello, world!",
            metadata={"language": "en"}
        )
        
        assert content.content_type == ContentType.TEXT
        assert content.data == "Hello, world!"
        assert content.metadata["language"] == "en"
        assert content.mime_type is None
        assert content.encoding is None
    
    def test_image_content_creation(self):
        """Test creating image content."""
        image_data = base64.b64encode(b"fake_image_data").decode()
        content = MultimodalContent(
            content_type=ContentType.IMAGE,
            data=image_data,
            mime_type="image/jpeg",
            encoding="base64",
            metadata={"width": 800, "height": 600}
        )
        
        assert content.content_type == ContentType.IMAGE
        assert content.data == image_data
        assert content.mime_type == "image/jpeg"
        assert content.encoding == "base64"
        assert content.metadata["width"] == 800
    
    def test_audio_content_creation(self):
        """Test creating audio content."""
        audio_data = base64.b64encode(b"fake_audio_data").decode()
        content = MultimodalContent(
            content_type=ContentType.AUDIO,
            data=audio_data,
            mime_type="audio/mp3",
            encoding="base64",
            metadata={"duration": 120.5, "bitrate": 128}
        )
        
        assert content.content_type == ContentType.AUDIO
        assert content.mime_type == "audio/mp3"
        assert content.metadata["duration"] == 120.5
    
    def test_is_text(self):
        """Test is_text method."""
        text_content = MultimodalContent(ContentType.TEXT, "Hello")
        image_content = MultimodalContent(ContentType.IMAGE, "data")
        
        assert text_content.is_text() is True
        assert image_content.is_text() is False
    
    def test_is_media(self):
        """Test is_media method."""
        text_content = MultimodalContent(ContentType.TEXT, "Hello")
        image_content = MultimodalContent(ContentType.IMAGE, "data")
        audio_content = MultimodalContent(ContentType.AUDIO, "data")
        video_content = MultimodalContent(ContentType.VIDEO, "data")
        
        assert text_content.is_media() is False
        assert image_content.is_media() is True
        assert audio_content.is_media() is True
        assert video_content.is_media() is True
    
    def test_get_size(self):
        """Test get_size method."""
        content = MultimodalContent(ContentType.TEXT, "Hello, world!")
        assert content.get_size() == 13
        
        # Test with base64 encoded data
        encoded_data = base64.b64encode(b"Hello").decode()
        content_encoded = MultimodalContent(
            ContentType.IMAGE, 
            encoded_data, 
            encoding="base64"
        )
        assert content_encoded.get_size() == 5  # Original data size
    
    def test_to_dict(self):
        """Test to_dict method."""
        content = MultimodalContent(
            content_type=ContentType.IMAGE,
            data="image_data",
            mime_type="image/jpeg",
            encoding="base64",
            metadata={"width": 800}
        )
        
        result = content.to_dict()
        expected = {
            "content_type": "image",
            "data": "image_data",
            "mime_type": "image/jpeg",
            "encoding": "base64",
            "metadata": {"width": 800}
        }
        
        assert result == expected
    
    def test_from_dict(self):
        """Test from_dict class method."""
        data = {
            "content_type": "audio",
            "data": "audio_data",
            "mime_type": "audio/mp3",
            "encoding": "base64",
            "metadata": {"duration": 120}
        }
        
        content = MultimodalContent.from_dict(data)
        
        assert content.content_type == ContentType.AUDIO
        assert content.data == "audio_data"
        assert content.mime_type == "audio/mp3"
        assert content.encoding == "base64"
        assert content.metadata["duration"] == 120


class TestMultimodalTorch:
    """Test MultimodalTorch class."""
    
    def test_single_content_creation(self):
        """Test creating torch with single content."""
        content = MultimodalContent(ContentType.TEXT, "Hello, world!")
        torch = MultimodalTorch(
            contents=[content],
            claim="Test message",
            confidence=0.9
        )
        
        assert len(torch.contents) == 1
        assert torch.contents[0].content_type == ContentType.TEXT
        assert torch.claim == "Test message"
        assert torch.confidence == 0.9
    
    def test_multiple_content_creation(self):
        """Test creating torch with multiple contents."""
        text_content = MultimodalContent(ContentType.TEXT, "Description")
        image_content = MultimodalContent(ContentType.IMAGE, "image_data")
        
        torch = MultimodalTorch(
            contents=[text_content, image_content],
            claim="Multimodal message"
        )
        
        assert len(torch.contents) == 2
        assert torch.contents[0].content_type == ContentType.TEXT
        assert torch.contents[1].content_type == ContentType.IMAGE
    
    def test_get_content_by_type(self):
        """Test getting content by type."""
        text_content = MultimodalContent(ContentType.TEXT, "Description")
        image_content = MultimodalContent(ContentType.IMAGE, "image_data")
        
        torch = MultimodalTorch(
            contents=[text_content, image_content],
            claim="Test"
        )
        
        text_results = torch.get_content_by_type(ContentType.TEXT)
        image_results = torch.get_content_by_type(ContentType.IMAGE)
        audio_results = torch.get_content_by_type(ContentType.AUDIO)
        
        assert len(text_results) == 1
        assert len(image_results) == 1
        assert len(audio_results) == 0
        assert text_results[0].data == "Description"
    
    def test_has_content_type(self):
        """Test checking if torch has specific content type."""
        text_content = MultimodalContent(ContentType.TEXT, "Description")
        torch = MultimodalTorch(contents=[text_content], claim="Test")
        
        assert torch.has_content_type(ContentType.TEXT) is True
        assert torch.has_content_type(ContentType.IMAGE) is False
    
    def test_get_primary_text(self):
        """Test getting primary text content."""
        text_content = MultimodalContent(ContentType.TEXT, "Primary text")
        image_content = MultimodalContent(ContentType.IMAGE, "image_data")
        
        torch = MultimodalTorch(
            contents=[image_content, text_content],
            claim="Test"
        )
        
        assert torch.get_primary_text() == "Primary text"
        
        # Test torch with no text content
        torch_no_text = MultimodalTorch(
            contents=[image_content],
            claim="Test"
        )
        assert torch_no_text.get_primary_text() is None
    
    def test_get_total_size(self):
        """Test calculating total size of all contents."""
        text_content = MultimodalContent(ContentType.TEXT, "Hello")  # 5 bytes
        image_content = MultimodalContent(ContentType.IMAGE, "12345")  # 5 bytes
        
        torch = MultimodalTorch(
            contents=[text_content, image_content],
            claim="Test"
        )
        
        assert torch.get_total_size() == 10
    
    def test_to_mcp_message(self):
        """Test converting to MCP message format."""
        text_content = MultimodalContent(ContentType.TEXT, "Hello")
        torch = MultimodalTorch(
            contents=[text_content],
            claim="Test message",
            source_campfire="test_campfire"
        )
        
        mcp_message = torch.to_mcp_message()
        
        assert mcp_message["role"] == "user"
        assert len(mcp_message["content"]) == 1
        assert mcp_message["content"][0]["type"] == "text"
        assert mcp_message["content"][0]["text"] == "Hello"
    
    def test_from_mcp_message(self):
        """Test creating torch from MCP message."""
        mcp_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello, world!"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,fake_data"
                    }
                }
            ]
        }
        
        torch = MultimodalTorch.from_mcp_message(mcp_message)
        
        assert len(torch.contents) == 2
        assert torch.contents[0].content_type == ContentType.TEXT
        assert torch.contents[0].data == "Hello, world!"
        assert torch.contents[1].content_type == ContentType.IMAGE
        assert torch.contents[1].data == "fake_data"
        assert torch.contents[1].encoding == "base64"
    
    def test_to_legacy_torch(self):
        """Test converting to legacy Torch format."""
        text_content = MultimodalContent(ContentType.TEXT, "Hello")
        torch = MultimodalTorch(
            contents=[text_content],
            claim="Test message",
            confidence=0.8,
            source_campfire="test_campfire"
        )
        
        legacy_torch = torch.to_legacy_torch()
        
        # Should be a dict with Torch-compatible fields
        assert isinstance(legacy_torch, dict)
        assert legacy_torch["claim"] == "Test message"
        assert legacy_torch["confidence"] == 0.8
        assert legacy_torch["source_campfire"] == "test_campfire"
    
    def test_from_legacy_torch(self):
        """Test creating multimodal torch from legacy torch."""
        legacy_data = {
            "claim": "Legacy message",
            "confidence": 0.7,
            "metadata": {"key": "value"},
            "source_campfire": "legacy_campfire"
        }
        
        torch = MultimodalTorch.from_legacy_torch(legacy_data)
        
        assert torch.claim == "Legacy message"
        assert torch.confidence == 0.7
        assert torch.metadata["key"] == "value"
        assert torch.source_campfire == "legacy_campfire"
        assert len(torch.contents) == 1
        assert torch.contents[0].content_type == ContentType.TEXT
        assert torch.contents[0].data == "Legacy message"


if __name__ == "__main__":
    pytest.main([__file__])