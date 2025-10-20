# Multimodal Input Support Requirements for Campfires

## Overview

This document outlines the requirements and implementation plan for adding comprehensive multimodal input support to the Campfires framework. Currently, Campfires supports text-based LLM interactions but lacks native support for processing images, audio, video, and other media types alongside text.

## Current State Analysis

### What Campfires Currently Has
- **Party Box System**: Storage system for assets (images, audio, documents) with file type categorization
- **Text-based LLM Integration**: Robust text prompt processing through OpenRouter
- **Binary Data Handling**: Basic infrastructure for storing and retrieving binary files in `campfires/party_box/`
- **File Organization**: Automatic categorization by file type (images, audio, documents, other)
- **Enhanced Orchestration**: Advanced task orchestration with detailed execution tracking

### What's Missing for Multimodal Input
- **Multimodal Message Format**: No support for mixed text/image/audio messages
- **Vision Model Integration**: No integration with vision-capable models (GPT-4V, Claude 3, etc.)
- **Audio Processing**: No audio transcription or analysis capabilities
- **Multimodal Prompt Engineering**: No templates for multimodal prompts
- **Content Type Handling**: No unified system for processing different media types

## Implementation Requirements

### 1. Multimodal Message Format (Priority: HIGH)

**Objective**: Create a unified message format that supports multiple content types in a single interaction.

**Requirements**:
- Design `MultimodalContent` class to represent different content types
- Create `MultimodalTorch` class extending current `Torch` functionality
- Support content types: TEXT, IMAGE, AUDIO, VIDEO, DOCUMENT
- Enable asset references to Party Box storage
- Maintain backward compatibility with existing `Torch` class

**Files to Create/Modify**:
- `campfires/core/multimodal_torch.py` (new)
- `campfires/core/torch.py` (modify for compatibility)
- `campfires/core/__init__.py` (add exports)

**Implementation Details**:
```python
from enum import Enum
from dataclasses import dataclass
from typing import Union, List, Optional, Dict, Any

class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"

@dataclass
class MultimodalContent:
    content_type: ContentType
    data: Union[str, bytes]  # Text or binary data
    metadata: Optional[Dict[str, Any]] = None
    asset_hash: Optional[str] = None  # Reference to Party Box asset
    mime_type: Optional[str] = None

@dataclass
class MultimodalTorch:
    contents: List[MultimodalContent]
    primary_claim: str  # Main text description
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
```

### 2. Vision Model Integration (Priority: HIGH)

**Objective**: Integrate vision-capable models for image analysis and understanding.

**Requirements**:
- Extend OpenRouter client to support vision models
- Support models: GPT-4V, Claude 3 Opus/Sonnet, Google Gemini Pro Vision
- Handle image encoding (base64) for API calls
- Support multiple images in single request
- Implement proper error handling for vision API calls

**Files to Create/Modify**:
- `campfires/core/multimodal_openrouter.py` (new)
- `campfires/core/openrouter.py` (extend existing)
- `campfires/mcp/multimodal_protocol.py` (new)

**Implementation Details**:
```python
class MultimodalOpenRouterClient(OpenRouterClient):
    VISION_MODELS = [
        "openai/gpt-4-vision-preview",
        "anthropic/claude-3-opus",
        "anthropic/claude-3-sonnet",
        "google/gemini-pro-vision"
    ]
    
    async def vision_completion(
        self, 
        text_prompt: str,
        images: List[Union[str, bytes]],
        model: str = "openai/gpt-4-vision-preview",
        **kwargs
    ) -> str:
        # Implementation for vision API calls
```

### 3. MultimodalCamperMixin (Priority: HIGH)

**Objective**: Create a mixin class that enables campers to process multimodal inputs.

**Requirements**:
- Extend existing `LLMCamperMixin` functionality
- Support automatic content type detection and routing
- Provide methods for processing different media types
- Enable seamless integration with existing camper classes
- Support fallback to text-only processing when needed

**Files to Create/Modify**:
- `campfires/core/multimodal_camper_mixin.py` (new)
- `campfires/core/__init__.py` (add exports)

**Implementation Details**:
```python
class MultimodalCamperMixin(LLMCamperMixin):
    def __init__(self):
        super().__init__()
        self.supported_content_types = [ContentType.TEXT]
        self.vision_models = ["openai/gpt-4-vision-preview"]
        self.audio_models = ["openai/whisper-1"]
    
    async def process_multimodal_torch(self, torch: MultimodalTorch) -> MultimodalTorch:
        # Main processing method for multimodal content
    
    async def _process_vision_content(self, text: str, images: List[MultimodalContent]) -> str:
        # Vision processing implementation
    
    async def _process_audio_content(self, audio_contents: List[MultimodalContent]) -> str:
        # Audio processing implementation
```

### 4. Audio Processing (Priority: MEDIUM)

**Objective**: Add audio transcription and analysis capabilities.

**Requirements**:
- Integrate with Whisper API for transcription
- Support common audio formats (MP3, WAV, OGG, M4A, FLAC)
- Provide audio analysis beyond transcription (sentiment, speaker detection)
- Handle large audio files with chunking
- Support real-time audio processing

**Files to Create/Modify**:
- `campfires/core/audio_processor.py` (new)
- `campfires/utils/audio_utils.py` (new)

**Dependencies to Add**:
- `openai-whisper` or OpenAI API for transcription
- `pydub` for audio format handling
- `librosa` for audio analysis (optional)

### 5. Enhanced Party Box (Priority: MEDIUM)

**Objective**: Enhance Party Box system with multimodal metadata extraction and better asset management.

**Requirements**:
- Extract metadata from images (dimensions, format, EXIF data)
- Extract metadata from audio files (duration, bitrate, format)
- Extract metadata from video files (duration, resolution, codec)
- Implement content-based deduplication
- Add search capabilities by content type and metadata
- Support thumbnail generation for images and videos

**Files to Create/Modify**:
- `campfires/party_box/multimodal_driver.py` (new)
- `campfires/party_box/metadata_extractor.py` (new)
- `campfires/party_box/local_driver.py` (extend existing)

**Dependencies to Add**:
- `Pillow` for image processing
- `mutagen` for audio metadata
- `opencv-python` for video processing (optional)
- `ffmpeg-python` for media processing (optional)

### 6. Multimodal Prompt Templates (Priority: MEDIUM)

**Objective**: Develop specialized prompt templates and engineering patterns for multimodal interactions.

**Requirements**:
- Create templates for vision analysis tasks
- Create templates for audio analysis tasks
- Create templates for multi-modal reasoning
- Support dynamic prompt generation based on content types
- Provide best practices and examples

**Files to Create/Modify**:
- `campfires/utils/multimodal_template_loader.py` (new)
- `campfires/utils/template_loader.py` (extend existing)
- `templates/multimodal/` (new directory with template files)

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
1. **Multimodal Message Format** - Core data structures
2. **Basic Vision Integration** - Simple image + text processing
3. **MultimodalCamperMixin** - Basic multimodal camper support

### Phase 2: Core Features (Weeks 3-4)
4. **Audio Processing** - Transcription and basic analysis
5. **Enhanced Party Box** - Metadata extraction and improved storage
6. **Advanced Vision Features** - Multiple images, complex analysis

### Phase 3: Polish and Templates (Weeks 5-6)
7. **Multimodal Prompt Templates** - Specialized templates and patterns
8. **Documentation and Examples** - Comprehensive guides and demos
9. **Testing and Optimization** - Performance tuning and edge cases

## Technical Dependencies

### New Python Packages Required
```
# Vision and Image Processing
Pillow>=10.0.0
opencv-python>=4.8.0  # Optional, for advanced image processing

# Audio Processing
openai-whisper>=20231117  # Or use OpenAI API
pydub>=0.25.1
mutagen>=1.47.0  # For audio metadata

# Video Processing (Optional)
ffmpeg-python>=0.2.0

# Additional utilities
python-magic>=0.4.27  # For MIME type detection
```

### API Requirements
- **OpenAI API**: For GPT-4V and Whisper (if not using local Whisper)
- **Anthropic API**: For Claude 3 vision capabilities
- **Google AI API**: For Gemini Pro Vision (optional)

## File Structure Changes

```
campfires/
├── core/
│   ├── multimodal_torch.py          # New: Multimodal message format
│   ├── multimodal_openrouter.py     # New: Vision model integration
│   ├── multimodal_camper_mixin.py   # New: Multimodal camper capabilities
│   ├── audio_processor.py           # New: Audio processing
│   └── openrouter.py                # Modified: Extended for multimodal
├── party_box/
│   ├── multimodal_driver.py         # New: Enhanced storage driver
│   ├── metadata_extractor.py        # New: Media metadata extraction
│   └── local_driver.py              # Modified: Enhanced with metadata
├── utils/
│   ├── multimodal_template_loader.py # New: Multimodal prompt templates
│   ├── audio_utils.py               # New: Audio processing utilities
│   └── template_loader.py           # Modified: Extended for multimodal
├── mcp/
│   └── multimodal_protocol.py       # New: Multimodal MCP support
└── templates/                       # New: Template directory
    └── multimodal/
        ├── vision_analysis.j2
        ├── audio_analysis.j2
        └── multimodal_reasoning.j2
```

## Usage Examples

### Basic Vision Analysis
```python
from campfires import Campfire, MultimodalCamper, MultimodalTorch, ContentType

class VisionAnalyzer(MultimodalCamper):
    def __init__(self, name: str):
        super().__init__(name)
        self.supported_content_types = [ContentType.TEXT, ContentType.IMAGE]

# Usage
torch = MultimodalTorch(
    contents=[
        MultimodalContent(ContentType.TEXT, "Analyze this product for marketing insights"),
        MultimodalContent(ContentType.IMAGE, asset_hash="product_image_hash")
    ],
    primary_claim="Analyze this product image for marketing insights"
)

result = await campfire.send_multimodal_torch(torch)
```

### Audio Transcription and Analysis
```python
class AudioAnalyzer(MultimodalCamper):
    def __init__(self, name: str):
        super().__init__(name)
        self.supported_content_types = [ContentType.TEXT, ContentType.AUDIO]

# Usage
torch = MultimodalTorch(
    contents=[
        MultimodalContent(ContentType.TEXT, "Summarize this meeting recording"),
        MultimodalContent(ContentType.AUDIO, asset_hash="meeting_audio_hash")
    ],
    primary_claim="Summarize this meeting recording"
)
```

## Testing Strategy

### Unit Tests
- Test multimodal message format serialization/deserialization
- Test vision model integration with mock API responses
- Test audio processing with sample files
- Test Party Box metadata extraction

### Integration Tests
- Test end-to-end multimodal workflows
- Test error handling and fallback scenarios
- Test performance with large media files
- Test compatibility with existing Campfires features

### Demo Applications
- **Multimodal Document Analyzer**: Process documents with text, images, and charts
- **Meeting Assistant**: Transcribe audio and analyze presentation slides
- **Product Review System**: Analyze product images and customer feedback
- **Content Moderation**: Analyze text, images, and audio for policy compliance

## Performance Considerations

### Optimization Strategies
- **Lazy Loading**: Load media content only when needed
- **Caching**: Cache transcriptions and analysis results
- **Compression**: Compress media files in Party Box storage
- **Streaming**: Support streaming for large audio/video files
- **Parallel Processing**: Process multiple media types concurrently

### Resource Management
- **Memory Management**: Handle large media files without memory issues
- **API Rate Limiting**: Respect API limits for vision and audio models
- **Storage Optimization**: Implement efficient storage strategies for media
- **Bandwidth Optimization**: Minimize data transfer for remote storage

## Security Considerations

### Data Privacy
- **Secure Storage**: Encrypt sensitive media files in Party Box
- **API Security**: Secure handling of API keys for vision/audio services
- **Data Retention**: Implement policies for media file retention
- **Access Control**: Control access to multimodal content

### Content Safety
- **Content Filtering**: Implement safety filters for uploaded media
- **Malware Detection**: Scan uploaded files for security threats
- **Format Validation**: Validate media file formats and headers
- **Size Limits**: Implement reasonable file size limits

## Migration Strategy

### Backward Compatibility
- Existing `Torch` class remains fully functional
- Existing campers work without modification
- Gradual migration path for existing applications
- Clear deprecation timeline for old patterns

### Migration Steps
1. **Install Dependencies**: Add new required packages
2. **Update Imports**: Import new multimodal classes
3. **Extend Campers**: Add `MultimodalCamperMixin` to existing campers
4. **Test Integration**: Verify existing functionality still works
5. **Add Multimodal Features**: Gradually add multimodal capabilities

## Success Metrics

### Functional Metrics
- Support for at least 3 content types (text, image, audio)
- Integration with at least 2 vision models
- Audio transcription accuracy > 95% for clear recordings
- Backward compatibility with all existing features

### Performance Metrics
- Image processing time < 10 seconds for typical images
- Audio transcription time < 2x audio duration
- Memory usage increase < 50% for multimodal vs text-only processing
- API response time < 30 seconds for complex multimodal requests

### Developer Experience Metrics
- Clear documentation with working examples
- Easy migration path from existing code
- Comprehensive error messages and debugging support
- Active community adoption and feedback

## Future Enhancements

### Advanced Features
- **Video Processing**: Full video analysis and understanding
- **Real-time Processing**: Live audio/video stream processing
- **Multi-language Support**: Audio transcription in multiple languages
- **Advanced Vision**: OCR, object detection, scene understanding
- **Cross-modal Reasoning**: Advanced reasoning across multiple content types

### Integration Opportunities
- **Web Interface**: Browser-based multimodal input
- **Mobile Support**: Mobile app integration for camera/microphone
- **Cloud Storage**: Integration with cloud storage providers
- **Streaming Platforms**: Integration with video/audio streaming services

---

## Implementation Notes

This requirements document serves as a comprehensive guide for implementing multimodal input support in Campfires. The implementation should be done incrementally, with each phase building upon the previous one. Regular testing and community feedback should guide the development process.

The goal is to make Campfires the most comprehensive and user-friendly framework for multimodal AI applications, while maintaining its core strengths in orchestration and LLM integration.