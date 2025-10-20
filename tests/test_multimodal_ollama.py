"""
Tests for multimodal Ollama client functionality.
"""

import pytest
import base64
from unittest.mock import Mock, patch, AsyncMock
from campfires.core.multimodal_ollama import (
    MultimodalOllamaConfig,
    MultimodalOllamaClient,
    OllamaMultimodalCamper
)
from campfires.core.multimodal_torch import ContentType, MultimodalContent


class TestMultimodalOllamaConfig:
    """Test MultimodalOllamaConfig class."""
    
    def test_default_config(self):
        """Test creating config with default values."""
        config = MultimodalOllamaConfig()
        
        assert config.base_url == "http://localhost:11434"
        assert config.vision_model == "llava"
        assert config.text_model == "llama2"
        assert config.timeout == 30
        assert config.max_image_size == 10 * 1024 * 1024  # 10MB
        assert config.supported_formats == ["jpg", "jpeg", "png", "gif", "bmp", "webp"]
    
    def test_custom_config(self):
        """Test creating config with custom values."""
        config = MultimodalOllamaConfig(
            base_url="http://custom:8080",
            vision_model="custom-llava",
            text_model="mistral",
            timeout=60,
            max_image_size=20 * 1024 * 1024,
            supported_formats=["jpg", "png"]
        )
        
        assert config.base_url == "http://custom:8080"
        assert config.vision_model == "custom-llava"
        assert config.text_model == "mistral"
        assert config.timeout == 60
        assert config.max_image_size == 20 * 1024 * 1024
        assert config.supported_formats == ["jpg", "png"]


class TestMultimodalOllamaClient:
    """Test MultimodalOllamaClient class."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
        return MultimodalOllamaConfig(
            vision_model="llava",
            text_model="llama2"
        )
    
    @pytest.fixture
    def client(self, config):
        """Create test client."""
        return MultimodalOllamaClient(config)
    
    @pytest.fixture
    def sample_image_data(self):
        """Create sample base64 image data."""
        return base64.b64encode(b"fake_image_data").decode()
    
    def test_validate_image_valid(self, client, sample_image_data):
        """Test image validation with valid image."""
        result = client._validate_image(sample_image_data, "test.jpg")
        assert result is True
    
    def test_validate_image_too_large(self, client):
        """Test image validation with oversized image."""
        # Create data larger than max_image_size
        large_data = base64.b64encode(b"x" * (15 * 1024 * 1024)).decode()
        
        with pytest.raises(ValueError) as exc_info:
            client._validate_image(large_data, "large.jpg")
        
        assert "exceeds maximum size" in str(exc_info.value)
    
    def test_validate_image_unsupported_format(self, client, sample_image_data):
        """Test image validation with unsupported format."""
        with pytest.raises(ValueError) as exc_info:
            client._validate_image(sample_image_data, "test.tiff")
        
        assert "Unsupported image format" in str(exc_info.value)
    
    def test_encode_image(self, client, sample_image_data):
        """Test image encoding."""
        encoded = client._encode_image(sample_image_data)
        assert encoded == sample_image_data
    
    @pytest.mark.asyncio
    async def test_analyze_image_success(self, client, sample_image_data):
        """Test successful image analysis."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "content": "This image shows a beautiful landscape."
            },
            "done": True
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await client.analyze_image(
                sample_image_data,
                "test.jpg",
                "Describe this image"
            )
            
            assert result == "This image shows a beautiful landscape."
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_describe_image(self, client, sample_image_data):
        """Test image description."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "content": "A detailed description of the image."
            },
            "done": True
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await client.describe_image(sample_image_data, "test.jpg")
            
            assert result == "A detailed description of the image."
            # Verify the prompt includes description request
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            assert "describe" in payload['messages'][0]['content'].lower()
    
    @pytest.mark.asyncio
    async def test_extract_text_from_image(self, client, sample_image_data):
        """Test text extraction from image."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "content": "Extracted text: Hello World"
            },
            "done": True
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await client.extract_text_from_image(sample_image_data, "test.jpg")
            
            assert result == "Extracted text: Hello World"
            # Verify the prompt includes OCR request
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            assert "text" in payload['messages'][0]['content'].lower()
    
    @pytest.mark.asyncio
    async def test_identify_objects(self, client, sample_image_data):
        """Test object identification in image."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "content": "Objects found: car, tree, building"
            },
            "done": True
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await client.identify_objects(sample_image_data, "test.jpg")
            
            assert result == "Objects found: car, tree, building"
            # Verify the prompt includes object identification request
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            assert "objects" in payload['messages'][0]['content'].lower()
    
    @pytest.mark.asyncio
    async def test_compare_images(self, client, sample_image_data):
        """Test image comparison."""
        image2_data = base64.b64encode(b"fake_image_data_2").decode()
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "content": "The images are similar in composition but different in color."
            },
            "done": True
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await client.compare_images(
                sample_image_data, "test1.jpg",
                image2_data, "test2.jpg"
            )
            
            assert result == "The images are similar in composition but different in color."
            # Verify the prompt includes comparison request
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            assert "compare" in payload['messages'][0]['content'].lower()
    
    @pytest.mark.asyncio
    async def test_get_multimodal_stats(self, client):
        """Test getting multimodal statistics."""
        stats = await client.get_multimodal_stats()
        
        assert "vision_model" in stats
        assert "text_model" in stats
        assert "supported_formats" in stats
        assert "max_image_size" in stats
        assert stats["vision_model"] == "llava"
        assert stats["text_model"] == "llama2"
    
    @pytest.mark.asyncio
    async def test_analyze_image_error_handling(self, client, sample_image_data):
        """Test error handling in image analysis."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = Exception("API error")
            
            with pytest.raises(Exception) as exc_info:
                await client.analyze_image(
                    sample_image_data,
                    "test.jpg",
                    "Describe this image"
                )
            
            assert "API error" in str(exc_info.value)


class TestOllamaMultimodalCamper:
    """Test OllamaMultimodalCamper class."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
        return {
            'name': 'test_camper',
            'ollama_base_url': 'http://localhost:11434',
            'ollama_vision_model': 'llava',
            'ollama_text_model': 'llama2'
        }
    
    @pytest.fixture
    def camper(self, config):
        """Create test camper."""
        return OllamaMultimodalCamper(config)
    
    @pytest.fixture
    def sample_multimodal_content(self):
        """Create sample multimodal content."""
        image_data = base64.b64encode(b"fake_image").decode()
        return MultimodalContent(
            content_type=ContentType.IMAGE,
            data=image_data,
            metadata={"filename": "test.jpg"}
        )
    
    def test_camper_initialization(self, camper):
        """Test camper initialization."""
        assert camper.name == "test_camper"
        assert hasattr(camper, 'multimodal_client')
        assert camper.multimodal_client.config.vision_model == "llava"
        assert camper.multimodal_client.config.text_model == "llama2"
    
    @pytest.mark.asyncio
    async def test_process_multimodal_content(self, camper, sample_multimodal_content):
        """Test processing multimodal content."""
        with patch.object(camper.multimodal_client, 'analyze_image') as mock_analyze:
            mock_analyze.return_value = "Image analysis result"
            
            result = await camper.process_multimodal_content(
                sample_multimodal_content,
                "Analyze this image"
            )
            
            assert result == "Image analysis result"
            mock_analyze.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_describe_image_content(self, camper, sample_multimodal_content):
        """Test describing image content."""
        with patch.object(camper.multimodal_client, 'describe_image') as mock_describe:
            mock_describe.return_value = "Image description"
            
            result = await camper.describe_image_content(sample_multimodal_content)
            
            assert result == "Image description"
            mock_describe.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extract_text_content(self, camper, sample_multimodal_content):
        """Test extracting text from content."""
        with patch.object(camper.multimodal_client, 'extract_text_from_image') as mock_extract:
            mock_extract.return_value = "Extracted text"
            
            result = await camper.extract_text_content(sample_multimodal_content)
            
            assert result == "Extracted text"
            mock_extract.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_identify_objects_content(self, camper, sample_multimodal_content):
        """Test identifying objects in content."""
        with patch.object(camper.multimodal_client, 'identify_objects') as mock_identify:
            mock_identify.return_value = "Objects: car, tree"
            
            result = await camper.identify_objects_content(sample_multimodal_content)
            
            assert result == "Objects: car, tree"
            mock_identify.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_compare_content(self, camper, sample_multimodal_content):
        """Test comparing content."""
        image2_data = base64.b64encode(b"fake_image_2").decode()
        content2 = MultimodalContent(
            content_type=ContentType.IMAGE,
            data=image2_data,
            metadata={"filename": "test2.jpg"}
        )
        
        with patch.object(camper.multimodal_client, 'compare_images') as mock_compare:
            mock_compare.return_value = "Comparison result"
            
            result = await camper.compare_content(sample_multimodal_content, content2)
            
            assert result == "Comparison result"
            mock_compare.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_capabilities(self, camper):
        """Test getting camper capabilities."""
        capabilities = await camper.get_capabilities()
        
        expected_capabilities = [
            "image_analysis",
            "image_description", 
            "text_extraction",
            "object_identification",
            "image_comparison"
        ]
        
        for capability in expected_capabilities:
            assert capability in capabilities
    
    @pytest.mark.asyncio
    async def test_get_stats(self, camper):
        """Test getting camper statistics."""
        with patch.object(camper.multimodal_client, 'get_multimodal_stats') as mock_stats:
            mock_stats.return_value = {
                "vision_model": "llava",
                "text_model": "llama2"
            }
            
            stats = await camper.get_stats()
            
            assert "vision_model" in stats
            assert "text_model" in stats
            assert stats["vision_model"] == "llava"
            assert stats["text_model"] == "llama2"


if __name__ == "__main__":
    pytest.main([__file__])