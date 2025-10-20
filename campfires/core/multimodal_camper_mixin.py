"""
Multimodal Camper Mixin for handling different input types.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
from .multimodal_torch import MultimodalTorch, MultimodalContent, ContentType
from .multimodal_openrouter import MultimodalOpenRouterClient, MultimodalLLMCamperMixin
from .openrouter import OpenRouterConfig

logger = logging.getLogger(__name__)


class MultimodalCamperMixin(MultimodalLLMCamperMixin):
    """
    Enhanced mixin class that provides comprehensive multimodal capabilities to campers.
    
    This mixin extends the basic LLM capabilities with:
    - Content type validation and routing
    - Specialized processing methods for different media types
    - Content preprocessing and postprocessing hooks
    - Batch processing capabilities
    - Error handling and fallback mechanisms
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Content type handlers - can be overridden by subclasses
        self.content_handlers: Dict[ContentType, Callable] = {
            ContentType.TEXT: self._handle_text_content,
            ContentType.IMAGE: self._handle_image_content,
            ContentType.AUDIO: self._handle_audio_content,
            ContentType.VIDEO: self._handle_video_content,
            ContentType.DOCUMENT: self._handle_document_content
        }
        
        # Content preprocessors - applied before processing
        self.content_preprocessors: Dict[ContentType, List[Callable]] = {
            ContentType.TEXT: [],
            ContentType.IMAGE: [],
            ContentType.AUDIO: [],
            ContentType.VIDEO: [],
            ContentType.DOCUMENT: []
        }
        
        # Content postprocessors - applied after processing
        self.content_postprocessors: List[Callable] = []
        
        # Processing options
        self.processing_options = {
            "max_image_size": (1024, 1024),  # Max image dimensions
            "supported_image_formats": ["jpg", "jpeg", "png", "gif", "webp"],
            "supported_audio_formats": ["mp3", "wav", "m4a", "ogg"],
            "supported_document_formats": ["pdf", "txt", "md", "docx"],
            "batch_size": 5,  # Max items to process in a batch
            "enable_fallback": True,  # Enable fallback to text-only processing
            "cache_results": True  # Cache processing results
        }
        
        # Result cache
        self._result_cache: Dict[str, Any] = {}
    
    def setup_multimodal_processing(
        self,
        config: OpenRouterConfig = None,
        mcp_protocol=None,
        supported_types: List[ContentType] = None,
        processing_options: Dict[str, Any] = None
    ) -> None:
        """
        Set up multimodal processing capabilities.
        
        Args:
            config: OpenRouter configuration
            mcp_protocol: Optional MCP protocol instance
            supported_types: List of content types this camper supports
            processing_options: Custom processing options
        """
        # Initialize the multimodal LLM client
        self.setup_multimodal_llm(config, mcp_protocol)
        
        # Update supported content types
        if supported_types:
            self.supported_content_types = supported_types
        
        # Update processing options
        if processing_options:
            self.processing_options.update(processing_options)
        
        logger.info(f"Multimodal processing initialized with types: {[ct.value for ct in self.supported_content_types]}")
    
    async def process_torch(
        self,
        torch: MultimodalTorch,
        processing_mode: str = "auto",
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Process a multimodal torch with intelligent routing.
        
        Args:
            torch: MultimodalTorch to process
            processing_mode: Processing mode (auto, sequential, parallel, specialized)
            **kwargs: Additional processing parameters
            
        Returns:
            Processing result (string or structured data)
        """
        try:
            # Validate torch content
            self._validate_torch_content(torch)
            
            # Apply preprocessing
            processed_torch = await self._preprocess_torch(torch)
            
            # Route to appropriate processing method
            if processing_mode == "auto":
                result = await self._auto_process_torch(processed_torch, **kwargs)
            elif processing_mode == "sequential":
                result = await self._sequential_process_torch(processed_torch, **kwargs)
            elif processing_mode == "parallel":
                result = await self._parallel_process_torch(processed_torch, **kwargs)
            elif processing_mode == "specialized":
                result = await self._specialized_process_torch(processed_torch, **kwargs)
            else:
                raise ValueError(f"Unknown processing mode: {processing_mode}")
            
            # Apply postprocessing
            final_result = await self._postprocess_result(result, torch)
            
            # Cache result if enabled
            if self.processing_options.get("cache_results", True):
                cache_key = self._generate_cache_key(torch, processing_mode, kwargs)
                self._result_cache[cache_key] = final_result
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error processing torch: {e}")
            
            # Try fallback processing if enabled
            if self.processing_options.get("enable_fallback", True):
                return await self._fallback_process_torch(torch, **kwargs)
            else:
                raise
    
    async def process_content_batch(
        self,
        contents: List[MultimodalContent],
        prompt: str,
        batch_size: int = None,
        **kwargs
    ) -> List[str]:
        """
        Process multiple content items in batches.
        
        Args:
            contents: List of MultimodalContent to process
            prompt: Processing prompt
            batch_size: Batch size (defaults to configured value)
            **kwargs: Additional parameters
            
        Returns:
            List of processing results
        """
        if batch_size is None:
            batch_size = self.processing_options.get("batch_size", 5)
        
        results = []
        
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i + batch_size]
            
            # Create torch for this batch
            torch = MultimodalTorch(
                contents=batch,
                primary_claim=prompt,
                source_campfire=getattr(self, 'name', 'multimodal_camper'),
                channel="batch_processing"
            )
            
            # Process batch
            batch_result = await self.process_torch(torch, **kwargs)
            results.append(batch_result)
        
        return results
    
    async def analyze_content_by_type(
        self,
        content: MultimodalContent,
        analysis_type: str = "general",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze content with type-specific methods.
        
        Args:
            content: Content to analyze
            analysis_type: Type of analysis (general, detailed, technical, creative)
            **kwargs: Additional parameters
            
        Returns:
            Analysis results
        """
        content_type = content.content_type
        
        if content_type not in self.supported_content_types:
            raise ValueError(f"Content type {content_type.value} not supported")
        
        # Get appropriate handler
        handler = self.content_handlers.get(content_type)
        if not handler:
            raise ValueError(f"No handler available for content type {content_type.value}")
        
        # Perform type-specific analysis
        return await handler(content, analysis_type, **kwargs)
    
    def add_content_preprocessor(
        self,
        content_type: ContentType,
        preprocessor: Callable
    ) -> None:
        """
        Add a preprocessor for a specific content type.
        
        Args:
            content_type: Content type to preprocess
            preprocessor: Preprocessor function
        """
        if content_type not in self.content_preprocessors:
            self.content_preprocessors[content_type] = []
        
        self.content_preprocessors[content_type].append(preprocessor)
    
    def add_content_postprocessor(self, postprocessor: Callable) -> None:
        """
        Add a postprocessor for all results.
        
        Args:
            postprocessor: Postprocessor function
        """
        self.content_postprocessors.append(postprocessor)
    
    def register_content_handler(
        self,
        content_type: ContentType,
        handler: Callable
    ) -> None:
        """
        Register a custom handler for a content type.
        
        Args:
            content_type: Content type to handle
            handler: Handler function
        """
        self.content_handlers[content_type] = handler
        
        # Add to supported types if not already present
        if content_type not in self.supported_content_types:
            self.supported_content_types.append(content_type)
    
    # Private methods
    
    def _validate_torch_content(self, torch: MultimodalTorch) -> None:
        """Validate that torch content is supported."""
        torch_content_types = torch.get_content_types()
        unsupported_types = [ct for ct in torch_content_types if ct not in self.supported_content_types]
        
        if unsupported_types:
            raise ValueError(f"Unsupported content types: {[ct.value for ct in unsupported_types]}")
    
    async def _preprocess_torch(self, torch: MultimodalTorch) -> MultimodalTorch:
        """Apply preprocessing to torch content."""
        processed_contents = []
        
        for content in torch.contents:
            processed_content = content
            
            # Apply content-type specific preprocessors
            preprocessors = self.content_preprocessors.get(content.content_type, [])
            for preprocessor in preprocessors:
                processed_content = await preprocessor(processed_content)
            
            processed_contents.append(processed_content)
        
        # Create new torch with processed content
        processed_torch = MultimodalTorch(
            contents=processed_contents,
            primary_claim=torch.primary_claim,
            source_campfire=torch.source_campfire,
            channel=torch.channel,
            metadata=torch.metadata.copy()
        )
        
        return processed_torch
    
    async def _postprocess_result(self, result: Any, original_torch: MultimodalTorch) -> Any:
        """Apply postprocessing to results."""
        processed_result = result
        
        for postprocessor in self.content_postprocessors:
            processed_result = await postprocessor(processed_result, original_torch)
        
        return processed_result
    
    async def _auto_process_torch(self, torch: MultimodalTorch, **kwargs) -> str:
        """Automatically determine the best processing approach."""
        content_types = torch.get_content_types()
        
        # Simple text-only processing
        if content_types == [ContentType.TEXT]:
            return await self.process_multimodal_torch(torch, **kwargs)
        
        # Image-heavy processing
        if ContentType.IMAGE in content_types and len([c for c in torch.contents if c.content_type == ContentType.IMAGE]) > 2:
            return await self._image_focused_processing(torch, **kwargs)
        
        # Mixed content processing
        return await self.process_multimodal_torch(torch, **kwargs)
    
    async def _sequential_process_torch(self, torch: MultimodalTorch, **kwargs) -> Dict[str, Any]:
        """Process torch content sequentially by type."""
        results = {}
        
        for content_type in torch.get_content_types():
            type_contents = [c for c in torch.contents if c.content_type == content_type]
            
            if type_contents:
                type_torch = MultimodalTorch(
                    contents=type_contents,
                    primary_claim=f"Process {content_type.value} content: {torch.primary_claim}",
                    source_campfire=torch.source_campfire,
                    channel=f"{torch.channel}_{content_type.value}"
                )
                
                results[content_type.value] = await self.process_multimodal_torch(type_torch, **kwargs)
        
        return results
    
    async def _parallel_process_torch(self, torch: MultimodalTorch, **kwargs) -> Dict[str, Any]:
        """Process torch content in parallel (placeholder for future async implementation)."""
        # For now, fall back to sequential processing
        # In the future, this could use asyncio.gather for true parallel processing
        return await self._sequential_process_torch(torch, **kwargs)
    
    async def _specialized_process_torch(self, torch: MultimodalTorch, **kwargs) -> Dict[str, Any]:
        """Use specialized handlers for each content type."""
        results = {}
        
        for content in torch.contents:
            handler = self.content_handlers.get(content.content_type)
            if handler:
                results[f"{content.content_type.value}_{content.metadata.get('index', 0)}"] = await handler(
                    content, "specialized", **kwargs
                )
        
        return results
    
    async def _fallback_process_torch(self, torch: MultimodalTorch, **kwargs) -> str:
        """Fallback processing using text-only content."""
        logger.warning("Using fallback text-only processing")
        
        # Extract text content only
        text_contents = [c for c in torch.contents if c.content_type == ContentType.TEXT]
        
        if not text_contents:
            # Create a description of non-text content
            content_descriptions = []
            for content in torch.contents:
                desc = f"[{content.content_type.value.upper()} content"
                if content.metadata:
                    desc += f": {content.metadata}"
                desc += "]"
                content_descriptions.append(desc)
            
            text_contents = [MultimodalContent.create_text(" ".join(content_descriptions))]
        
        fallback_torch = MultimodalTorch(
            contents=text_contents,
            primary_claim=torch.primary_claim,
            source_campfire=torch.source_campfire,
            channel=f"{torch.channel}_fallback"
        )
        
        return await self.process_multimodal_torch(fallback_torch, **kwargs)
    
    async def _image_focused_processing(self, torch: MultimodalTorch, **kwargs) -> str:
        """Specialized processing for image-heavy content."""
        # Use vision-specific prompts and processing
        system_prompt = ("You are an expert at analyzing images. Provide detailed, "
                        "accurate descriptions and insights about visual content.")
        
        return await self.process_multimodal_torch(
            torch,
            system_prompt=system_prompt,
            **kwargs
        )
    
    def _generate_cache_key(self, torch: MultimodalTorch, processing_mode: str, kwargs: Dict) -> str:
        """Generate a cache key for the processing request."""
        # Simple hash-based cache key
        import hashlib
        
        key_data = f"{torch.torch_id}_{processing_mode}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    # Default content handlers
    
    async def _handle_text_content(self, content: MultimodalContent, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """Handle text content analysis."""
        text_data = content.get_data_as_string()
        
        analysis_prompts = {
            "general": "Analyze this text and provide a summary.",
            "detailed": "Provide a detailed analysis of this text including themes, tone, and key points.",
            "technical": "Analyze this text for technical content, terminology, and complexity.",
            "creative": "Analyze this text for creative elements, style, and literary devices."
        }
        
        prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])
        
        torch = MultimodalTorch(
            contents=[content],
            primary_claim=f"{prompt}\n\nText: {text_data}",
            source_campfire=getattr(self, 'name', 'text_analyzer'),
            channel="text_analysis"
        )
        
        result = await self.process_multimodal_torch(torch, **kwargs)
        
        return {
            "content_type": "text",
            "analysis_type": analysis_type,
            "result": result,
            "metadata": content.metadata
        }
    
    async def _handle_image_content(self, content: MultimodalContent, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """Handle image content analysis."""
        analysis_prompts = {
            "general": "Describe this image in detail.",
            "detailed": "Provide a comprehensive analysis of this image including objects, composition, colors, and mood.",
            "technical": "Analyze the technical aspects of this image including quality, format, and visual elements.",
            "creative": "Analyze the artistic and creative elements of this image."
        }
        
        prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])
        
        torch = MultimodalTorch(
            contents=[content],
            primary_claim=prompt,
            source_campfire=getattr(self, 'name', 'image_analyzer'),
            channel="image_analysis"
        )
        
        result = await self.process_multimodal_torch(torch, **kwargs)
        
        return {
            "content_type": "image",
            "analysis_type": analysis_type,
            "result": result,
            "metadata": content.metadata
        }
    
    async def _handle_audio_content(self, content: MultimodalContent, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """Handle audio content analysis."""
        # For now, return metadata analysis
        # In the future, this could include transcription and audio analysis
        
        return {
            "content_type": "audio",
            "analysis_type": analysis_type,
            "result": f"Audio content analysis not yet fully implemented. Metadata: {content.metadata}",
            "metadata": content.metadata
        }
    
    async def _handle_video_content(self, content: MultimodalContent, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """Handle video content analysis."""
        # Placeholder for video analysis
        return {
            "content_type": "video",
            "analysis_type": analysis_type,
            "result": f"Video content analysis not yet implemented. Metadata: {content.metadata}",
            "metadata": content.metadata
        }
    
    async def _handle_document_content(self, content: MultimodalContent, analysis_type: str, **kwargs) -> Dict[str, Any]:
        """Handle document content analysis."""
        # Extract text from document and analyze
        text_data = content.get_data_as_string()
        
        analysis_prompts = {
            "general": "Analyze this document and provide a summary.",
            "detailed": "Provide a detailed analysis of this document including structure, content, and key information.",
            "technical": "Analyze the technical aspects and information in this document.",
            "creative": "Analyze the writing style and creative elements in this document."
        }
        
        prompt = analysis_prompts.get(analysis_type, analysis_prompts["general"])
        
        torch = MultimodalTorch(
            contents=[MultimodalContent.create_text(text_data)],
            primary_claim=f"{prompt}\n\nDocument content: {text_data}",
            source_campfire=getattr(self, 'name', 'document_analyzer'),
            channel="document_analysis"
        )
        
        result = await self.process_multimodal_torch(torch, **kwargs)
        
        return {
            "content_type": "document",
            "analysis_type": analysis_type,
            "result": result,
            "metadata": content.metadata
        }