"""
Multimodal OpenRouter client for vision and audio processing.
"""

import base64
import logging
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from .openrouter import OpenRouterClient, OpenRouterConfig, ChatMessage, ChatResponse
from .multimodal_torch import MultimodalTorch, MultimodalContent, ContentType

logger = logging.getLogger(__name__)


class MultimodalChatMessage(BaseModel):
    """
    Extended chat message that supports multimodal content.
    """
    
    role: str = Field(..., description="Role: system, user, or assistant")
    content: Union[str, List[Dict[str, Any]]] = Field(..., description="Message content - string or multimodal content array")
    name: Optional[str] = Field(None, description="Optional name for the message")
    
    @classmethod
    def from_multimodal_content(cls, role: str, contents: List[MultimodalContent], name: Optional[str] = None) -> "MultimodalChatMessage":
        """
        Create a multimodal chat message from MultimodalContent objects.
        
        Args:
            role: Message role (system, user, assistant)
            contents: List of MultimodalContent objects
            name: Optional name for the message
            
        Returns:
            MultimodalChatMessage instance
        """
        if len(contents) == 1 and contents[0].content_type == ContentType.TEXT:
            # Simple text message
            return cls(role=role, content=contents[0].data, name=name)
        
        # Multimodal content array
        content_array = []
        for content in contents:
            if content.content_type == ContentType.TEXT:
                content_array.append({
                    "type": "text",
                    "text": content.data
                })
            elif content.content_type == ContentType.IMAGE:
                # Handle image content
                image_data = content.get_data_as_string()
                if not image_data.startswith("data:"):
                    # Add data URL prefix if not present
                    mime_type = content.mime_type or "image/jpeg"
                    image_data = f"data:{mime_type};base64,{image_data}"
                
                content_array.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_data,
                        "detail": content.metadata.get("detail", "auto")
                    }
                })
            elif content.content_type == ContentType.AUDIO:
                # For audio, we'll include it as text description for now
                # In the future, this could be enhanced with audio API support
                content_array.append({
                    "type": "text",
                    "text": f"[Audio content: {content.mime_type}, {content.metadata.get('duration', 'unknown duration')}]"
                })
        
        return cls(role=role, content=content_array, name=name)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API format."""
        message = {"role": self.role, "content": self.content}
        if self.name:
            message["name"] = self.name
        return message


class MultimodalOpenRouterClient(OpenRouterClient):
    """
    Extended OpenRouter client with multimodal capabilities.
    """
    
    # Vision-capable models available on OpenRouter
    VISION_MODELS = [
        "google/gemini-2.0-flash-exp:free",  # Free vision model
        "openai/gpt-4o-mini",               # Working vision model
        "anthropic/claude-3-haiku",         # Working vision model
        "meta-llama/llama-3.2-90b-vision-instruct",
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-opus",
        "google/gemini-pro-vision",
        "google/gemini-2.0-flash-exp"
    ]
    
    # Audio-capable models (for transcription)
    AUDIO_MODELS = [
        "openai/whisper-1"
    ]
    
    def __init__(self, config: OpenRouterConfig = None, mcp_protocol=None):
        super().__init__(config, mcp_protocol)
        self.default_vision_model = self._get_default_vision_model()
    
    def _get_default_vision_model(self) -> str:
        """Get the default vision model, preferring the configured default if it supports vision."""
        if self.config.default_model in self.VISION_MODELS:
            return self.config.default_model
        return "openai/gpt-4o-mini"
    
    def is_vision_model(self, model: str) -> bool:
        """Check if a model supports vision capabilities."""
        return model in self.VISION_MODELS
    
    def is_audio_model(self, model: str) -> bool:
        """Check if a model supports audio processing."""
        return model in self.AUDIO_MODELS
    
    async def multimodal_completion(
        self,
        torch: MultimodalTorch,
        model: str = None,
        system_prompt: str = None,
        **kwargs
    ) -> str:
        """
        Process a multimodal torch and return a text response.
        
        Args:
            torch: MultimodalTorch containing multimodal content
            model: Model to use (defaults to vision model if images present)
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the API call
            
        Returns:
            Text response from the model
        """
        # Determine appropriate model
        if model is None:
            if torch.has_content_type(ContentType.IMAGE):
                model = self.default_vision_model
            else:
                model = self.config.default_model
        
        # Validate model capabilities
        if torch.has_content_type(ContentType.IMAGE) and not self.is_vision_model(model):
            raise ValueError(f"Model {model} does not support vision. Use a vision-capable model for image content.")
        
        # Build messages
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append(MultimodalChatMessage(role="system", content=system_prompt))
        
        # Add user message with multimodal content
        user_message = MultimodalChatMessage.from_multimodal_content("user", torch.contents)
        messages.append(user_message)
        
        # Convert to OpenAI format
        openai_messages = [msg.to_openai_format() for msg in messages]
        
        # Make API call
        response = await self.chat_completion(
            messages=[ChatMessage(role=msg["role"], content=str(msg["content"])) for msg in openai_messages],
            model=model,
            **kwargs
        )
        
        if response.choices:
            return response.choices[0]["message"]["content"]
        else:
            raise ValueError("No response received from the model")
    
    async def vision_completion(
        self,
        text_prompt: str,
        images: List[Union[str, bytes, MultimodalContent]],
        model: str = None,
        system_prompt: str = None,
        **kwargs
    ) -> str:
        """
        Process text and images together.
        
        Args:
            text_prompt: Text prompt describing the task
            images: List of images (as base64 strings, bytes, or MultimodalContent)
            model: Vision model to use
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Text response from the vision model
        """
        if model is None:
            model = self.default_vision_model
        
        if not self.is_vision_model(model):
            raise ValueError(f"Model {model} does not support vision capabilities")
        
        # Create multimodal torch
        torch = MultimodalTorch(
            contents=[],
            primary_claim=text_prompt,
            source_campfire="vision_client",
            channel="vision"
        )
        
        # Add text content
        torch.add_text(text_prompt)
        
        # Add image content
        for i, image in enumerate(images):
            if isinstance(image, MultimodalContent):
                torch.add_content(image)
            elif isinstance(image, (str, bytes)):
                torch.add_image(
                    image_data=image,
                    metadata={"index": i, "source": "vision_completion"}
                )
        
        return await self.multimodal_completion(torch, model=model, system_prompt=system_prompt, **kwargs)
    
    async def analyze_image(
        self,
        image: Union[str, bytes, MultimodalContent],
        prompt: str = "Describe this image in detail.",
        model: str = None,
        **kwargs
    ) -> str:
        """
        Analyze a single image with a text prompt.
        
        Args:
            image: Image to analyze
            prompt: Analysis prompt
            model: Vision model to use
            **kwargs: Additional parameters
            
        Returns:
            Analysis result
        """
        return await self.vision_completion(
            text_prompt=prompt,
            images=[image],
            model=model,
            **kwargs
        )
    
    async def compare_images(
        self,
        images: List[Union[str, bytes, MultimodalContent]],
        prompt: str = "Compare these images and describe the differences.",
        model: str = None,
        **kwargs
    ) -> str:
        """
        Compare multiple images.
        
        Args:
            images: List of images to compare
            prompt: Comparison prompt
            model: Vision model to use
            **kwargs: Additional parameters
            
        Returns:
            Comparison result
        """
        if len(images) < 2:
            raise ValueError("At least 2 images are required for comparison")
        
        return await self.vision_completion(
            text_prompt=prompt,
            images=images,
            model=model,
            **kwargs
        )
    
    async def extract_text_from_image(
        self,
        image: Union[str, bytes, MultimodalContent],
        model: str = None,
        **kwargs
    ) -> str:
        """
        Extract text from an image (OCR).
        
        Args:
            image: Image containing text
            model: Vision model to use
            **kwargs: Additional parameters
            
        Returns:
            Extracted text
        """
        prompt = ("Extract all text from this image. Return only the text content, "
                 "maintaining the original formatting and structure as much as possible.")
        
        return await self.analyze_image(
            image=image,
            prompt=prompt,
            model=model,
            **kwargs
        )
    
    async def audio_transcription(
        self,
        audio: Union[str, bytes, MultimodalContent],
        model: str = "openai/whisper-1",
        language: str = None,
        **kwargs
    ) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio content to transcribe
            model: Audio model to use
            language: Optional language hint
            **kwargs: Additional parameters
            
        Returns:
            Transcribed text
        """
        if not self.is_audio_model(model):
            raise ValueError(f"Model {model} does not support audio transcription")
        
        # For now, return a placeholder since audio transcription requires special handling
        # This would need to be implemented with the actual Whisper API endpoint
        logger.warning("Audio transcription not yet fully implemented - returning placeholder")
        return "[Audio transcription placeholder - implement with Whisper API]"
    
    def get_supported_models(self) -> Dict[str, List[str]]:
        """
        Get lists of supported models by capability.
        
        Returns:
            Dictionary with model lists by capability
        """
        return {
            "vision": self.VISION_MODELS,
            "audio": self.AUDIO_MODELS,
            "text": [self.config.default_model]  # All models support text
        }
    
    def get_model_capabilities(self, model: str) -> List[str]:
        """
        Get capabilities of a specific model.
        
        Args:
            model: Model name
            
        Returns:
            List of capabilities (text, vision, audio)
        """
        capabilities = ["text"]  # All models support text
        
        if self.is_vision_model(model):
            capabilities.append("vision")
        
        if self.is_audio_model(model):
            capabilities.append("audio")
        
        return capabilities


class MultimodalLLMCamperMixin:
    """
    Mixin class that provides multimodal LLM capabilities to campers.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.multimodal_client: Optional[MultimodalOpenRouterClient] = None
        self.supported_content_types = [ContentType.TEXT]  # Override in subclasses
    
    def setup_multimodal_llm(self, config: OpenRouterConfig = None, mcp_protocol=None) -> None:
        """
        Set up the multimodal LLM client.
        
        Args:
            config: OpenRouter configuration
            mcp_protocol: Optional MCP protocol instance
        """
        self.multimodal_client = MultimodalOpenRouterClient(config, mcp_protocol)
        
        # Update supported content types based on default model capabilities
        if self.multimodal_client.is_vision_model(self.multimodal_client.config.default_model):
            self.supported_content_types.extend([ContentType.IMAGE])
    
    async def process_multimodal_torch(
        self,
        torch: MultimodalTorch,
        model: str = None,
        system_prompt: str = None,
        **kwargs
    ) -> str:
        """
        Process a multimodal torch and return a response.
        
        Args:
            torch: MultimodalTorch to process
            model: Optional model override
            system_prompt: Optional system prompt
            **kwargs: Additional parameters
            
        Returns:
            Text response
        """
        if not self.multimodal_client:
            raise ValueError("Multimodal LLM client not initialized. Call setup_multimodal_llm() first.")
        
        # Validate content types
        torch_content_types = torch.get_content_types()
        unsupported_types = [ct for ct in torch_content_types if ct not in self.supported_content_types]
        
        if unsupported_types:
            raise ValueError(f"Unsupported content types: {[ct.value for ct in unsupported_types]}")
        
        return await self.multimodal_client.multimodal_completion(
            torch=torch,
            model=model,
            system_prompt=system_prompt,
            **kwargs
        )
    
    async def analyze_images(
        self,
        images: List[Union[str, bytes, MultimodalContent]],
        prompt: str,
        model: str = None,
        **kwargs
    ) -> str:
        """
        Analyze images with a text prompt.
        
        Args:
            images: List of images to analyze
            prompt: Analysis prompt
            model: Optional model override
            **kwargs: Additional parameters
            
        Returns:
            Analysis result
        """
        if not self.multimodal_client:
            raise ValueError("Multimodal LLM client not initialized.")
        
        if ContentType.IMAGE not in self.supported_content_types:
            raise ValueError("This camper does not support image processing.")
        
        return await self.multimodal_client.vision_completion(
            text_prompt=prompt,
            images=images,
            model=model,
            **kwargs
        )
    
    def supports_content_type(self, content_type: ContentType) -> bool:
        """Check if this camper supports a specific content type."""
        return content_type in self.supported_content_types
    
    def get_supported_content_types(self) -> List[ContentType]:
        """Get list of supported content types."""
        return self.supported_content_types.copy()