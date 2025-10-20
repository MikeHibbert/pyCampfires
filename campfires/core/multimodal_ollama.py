"""
Multimodal Ollama integration for Campfires.

This module provides multimodal capabilities for Ollama, including vision models
and image processing functionality.
"""

import base64
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import asyncio

from .ollama import OllamaClient, OllamaConfig

logger = logging.getLogger(__name__)


@dataclass
class MultimodalOllamaConfig(OllamaConfig):
    """Configuration for multimodal Ollama client."""
    
    # Vision model settings
    vision_model: str = "llava"
    image_quality: str = "auto"  # auto, low, high
    max_image_size: int = 1024 * 1024  # 1MB default
    
    # Supported image formats
    supported_formats: List[str] = field(default_factory=lambda: [
        'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'
    ])
    
    def __post_init__(self):
        """Validate multimodal configuration."""
        super().__post_init__()
        
        if self.max_image_size <= 0:
            raise ValueError("max_image_size must be positive")


class MultimodalOllamaClient:
    """Multimodal client for Ollama with vision capabilities."""
    
    def __init__(self, config: MultimodalOllamaConfig):
        """Initialize multimodal Ollama client."""
        self.config = config
        self.client = OllamaClient(config)
        self.multimodal_stats = {
            'images_processed': 0,
            'vision_requests': 0,
            'encoding_errors': 0,
            'validation_errors': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    def _validate_image(self, image_data: bytes, filename: str = "") -> bool:
        """Validate image data and format."""
        try:
            # Check file size
            if len(image_data) > self.config.max_image_size:
                logger.warning(f"Image too large: {len(image_data)} bytes")
                return False
            
            # Check if data looks like an image (basic check)
            if len(image_data) < 10:
                logger.warning("Image data too small")
                return False
            
            # Check file extension if provided
            if filename:
                ext = filename.lower().split('.')[-1]
                if ext not in self.config.supported_formats:
                    logger.warning(f"Unsupported format: {ext}")
                    return False
            
            # Basic image header validation
            image_headers = {
                b'\xff\xd8\xff': 'jpg',
                b'\x89PNG\r\n\x1a\n': 'png',
                b'GIF87a': 'gif',
                b'GIF89a': 'gif',
                b'BM': 'bmp',
                b'RIFF': 'webp'
            }
            
            for header, format_name in image_headers.items():
                if image_data.startswith(header):
                    return True
            
            # Check for SVG (XML-based format)
            if image_data.startswith(b'<svg') or b'<svg' in image_data[:100]:
                return True
            
            # If no header matches but filename suggests image, allow it
            if filename and any(filename.lower().endswith(f'.{fmt}') for fmt in self.config.supported_formats):
                return True
            
            logger.warning("Image format not recognized")
            return False
            
        except Exception as e:
            self.multimodal_stats['validation_errors'] += 1
            logger.error(f"Image validation failed: {e}")
            return False
    
    def _encode_image(self, image_data: bytes) -> str:
        """Encode image data to base64."""
        try:
            encoded = base64.b64encode(image_data).decode('utf-8')
            self.multimodal_stats['images_processed'] += 1
            return encoded
            
        except Exception as e:
            self.multimodal_stats['encoding_errors'] += 1
            logger.error(f"Image encoding failed: {e}")
            raise
    
    async def analyze_image(self, image_data: bytes, prompt: str, model: Optional[str] = None) -> str:
        """Analyze an image with a custom prompt using vision model."""
        if not self._validate_image(image_data):
            raise ValueError("Invalid image data")
        
        vision_model = model or self.config.vision_model
        encoded_image = self._encode_image(image_data)
        
        # Prepare multimodal message
        messages = [
            {
                "role": "user",
                "content": prompt,
                "images": [encoded_image]
            }
        ]
        
        try:
            self.multimodal_stats['vision_requests'] += 1
            response = await self.client.chat(messages, vision_model)
            return response
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            raise
    
    async def describe_image(self, image_data: bytes, model: Optional[str] = None) -> str:
        """Generate a detailed description of an image."""
        prompt = "Describe this image in detail. Include information about objects, people, colors, composition, and any text visible in the image."
        return await self.analyze_image(image_data, prompt, model)
    
    async def extract_text_from_image(self, image_data: bytes, model: Optional[str] = None) -> str:
        """Extract text content from an image (OCR functionality)."""
        prompt = "Extract and transcribe all text visible in this image. If no text is present, respond with 'No text found'."
        return await self.analyze_image(image_data, prompt, model)
    
    async def identify_objects(self, image_data: bytes, model: Optional[str] = None) -> str:
        """Identify and list objects in an image."""
        prompt = "List all objects, people, and items visible in this image. Provide a structured list with confidence levels if possible."
        return await self.analyze_image(image_data, prompt, model)
    
    async def compare_images(self, image1_data: bytes, image2_data: bytes, model: Optional[str] = None) -> str:
        """Compare two images and describe differences."""
        if not self._validate_image(image1_data) or not self._validate_image(image2_data):
            raise ValueError("Invalid image data")
        
        vision_model = model or self.config.vision_model
        encoded_image1 = self._encode_image(image1_data)
        encoded_image2 = self._encode_image(image2_data)
        
        messages = [
            {
                "role": "user",
                "content": "Compare these two images and describe the differences, similarities, and any notable changes between them.",
                "images": [encoded_image1, encoded_image2]
            }
        ]
        
        try:
            self.multimodal_stats['vision_requests'] += 1
            response = await self.client.chat(messages, vision_model)
            return response
            
        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
            raise
    
    async def check_vision_model_available(self, model_name: Optional[str] = None) -> bool:
        """Check if a vision model is available in Ollama."""
        model = model_name or self.config.vision_model
        return await self.client.check_model_exists(model)
    
    async def pull_vision_model(self, model_name: Optional[str] = None) -> bool:
        """Pull/download a vision model to Ollama."""
        model = model_name or self.config.vision_model
        return await self.client.pull_model(model)
    
    async def list_vision_models(self) -> List[str]:
        """List available vision models in Ollama."""
        try:
            all_models = await self.client.list_models()
            
            # Common vision model patterns
            vision_patterns = ['llava', 'bakllava', 'moondream', 'vision']
            
            vision_models = []
            for model in all_models:
                model_name = model.get('name', '').lower()
                if any(pattern in model_name for pattern in vision_patterns):
                    vision_models.append(model.get('name', ''))
            
            return vision_models
            
        except Exception as e:
            logger.error(f"Failed to list vision models: {e}")
            return []
    
    async def get_vision_model_info(self, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get information about a vision model."""
        model = model_name or self.config.vision_model
        return await self.client.get_model_info(model)
    
    def get_multimodal_stats(self) -> Dict[str, Any]:
        """Get multimodal client statistics."""
        return {
            **self.multimodal_stats,
            'ollama_stats': self.client.get_stats(),
            'config': {
                'vision_model': self.config.vision_model,
                'max_image_size': self.config.max_image_size,
                'supported_formats': self.config.supported_formats
            }
        }
    
    async def close(self):
        """Close the client."""
        await self.client.close()


class OllamaMultimodalCamper:
    """Specialized camper using Ollama's multimodal capabilities."""
    
    def __init__(self, party_box, config: Dict[str, Any], role_requirement=None):
        """Initialize Ollama multimodal camper."""
        self.party_box = party_box
        self.config = config
        self.role_requirement = role_requirement
        
        # Initialize Ollama multimodal client
        ollama_config = MultimodalOllamaConfig(
            base_url=config.get('ollama_base_url', 'http://localhost:11434'),
            model=config.get('model', 'llama3.2'),
            vision_model=config.get('vision_model', 'llava'),
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', None)
        )
        
        self.ollama_client = MultimodalOllamaClient(ollama_config)
        self.stats = {
            'images_analyzed': 0,
            'text_extracted': 0,
            'objects_identified': 0,
            'comparisons_made': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.ollama_client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.ollama_client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def analyze_image(self, image_data: bytes, prompt: str) -> str:
        """Analyze an image with custom prompt."""
        self.stats['images_analyzed'] += 1
        return await self.ollama_client.analyze_image(image_data, prompt)
    
    async def describe_image(self, image_data: bytes) -> str:
        """Generate detailed image description."""
        self.stats['images_analyzed'] += 1
        return await self.ollama_client.describe_image(image_data)
    
    async def extract_text(self, image_data: bytes) -> str:
        """Extract text from image."""
        self.stats['text_extracted'] += 1
        return await self.ollama_client.extract_text_from_image(image_data)
    
    async def identify_objects(self, image_data: bytes) -> str:
        """Identify objects in image."""
        self.stats['objects_identified'] += 1
        return await self.ollama_client.identify_objects(image_data)
    
    async def compare_images(self, image1_data: bytes, image2_data: bytes) -> str:
        """Compare two images."""
        self.stats['comparisons_made'] += 1
        return await self.ollama_client.compare_images(image1_data, image2_data)
    
    async def process(self, torch):
        """Process a multimodal torch using Ollama vision capabilities."""
        try:
            # Extract image content from torch
            image_contents = [content for content in torch.contents if content.content_type.name == 'IMAGE']
            
            if not image_contents:
                return "No images found in the torch to analyze."
            
            results = []
            for content in image_contents:
                if hasattr(content, 'data') and content.data:
                    # Analyze the image
                    description = await self.describe_image(content.data)
                    results.append(f"Image analysis: {description}")
            
            return "\n\n".join(results)
            
        except Exception as e:
            logger.error(f"Torch processing failed: {e}")
            return f"Error processing torch: {str(e)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Get camper statistics."""
        return {
            **self.stats,
            'multimodal_stats': self.ollama_client.get_multimodal_stats()
        }