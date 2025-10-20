"""
Campfires Framework

A Python framework for orchestrating multimodal Large Language Models (LLMs) 
and tools to achieve emergent, task-driven behavior.
"""

__version__ = "0.4.0"
__author__ = "Campfires Team"

# Core components
from .core.campfire import Campfire
from .core.camper import Camper
from .core.torch import Torch
from .core.state_manager import StateManager
from .core.openrouter import (
    OpenRouterClient, 
    OpenRouterConfig, 
    ChatMessage, 
    ChatRequest, 
    ChatResponse,
    LLMCamperMixin
)

# Multimodal components
from .core.multimodal_torch import (
    ContentType,
    MultimodalContent,
    MultimodalTorch
)
from .core.multimodal_openrouter import (
    MultimodalOpenRouterClient,
    MultimodalLLMCamperMixin
)
from .core.multimodal_camper_mixin import MultimodalCamperMixin
from .core.multimodal_prompts import (
    PromptType,
    PromptTemplate,
    MultimodalPromptLibrary,
    PromptEngineeringPatterns,
    MultimodalPromptBuilder,
    get_prompt_for_content_types
)

# MCP (Model Context Protocol) components
from .mcp.protocol import MCPProtocol, ChannelManager
from .mcp.transport import AsyncQueueTransport

# Party Box storage components
from .party_box.box_driver import BoxDriver
from .party_box.local_driver import LocalDriver
from .party_box.multimodal_local_driver import MultimodalLocalDriver, MultimodalAssetManager
from .party_box.metadata_extractor import (
    ContentMetadata,
    ImageMetadata,
    AudioMetadata,
    VideoMetadata,
    DocumentMetadata,
    MetadataExtractor
)

# Zeitgeist components
from .zeitgeist.zeitgeist_engine import ZeitgeistEngine
from .zeitgeist.opinion_analyzer import OpinionAnalyzer
from .zeitgeist.role_query_generator import RoleQueryGenerator
from .zeitgeist.config import ZeitgeistConfig

# Audio processing components
from .core.audio_processor import AudioProcessor
from .core.audio_utils import (
    AudioFormatDetector,
    AudioValidator,
    AudioConverter
)

# Utility functions
from .utils.hash_utils import generate_torch_id, generate_asset_id
from .utils.template_loader import TemplateLoader, render_template

__all__ = [
    # Core framework
    "Campfire",
    "Camper", 
    "Torch",
    "StateManager",
    
    # LLM integration
    "OpenRouterClient",
    "OpenRouterConfig",
    "ChatMessage",
    "ChatRequest", 
    "ChatResponse",
    "LLMCamperMixin",
    
    # Multimodal components
    "ContentType",
    "MultimodalContent",
    "MultimodalTorch",
    "MultimodalOpenRouterClient",
    "MultimodalLLMCamperMixin",
    "MultimodalCamperMixin",
    "PromptType",
    "PromptTemplate",
    "MultimodalPromptLibrary",
    "PromptEngineeringPatterns",
    "MultimodalPromptBuilder",
    "get_prompt_for_content_types",
    
    # MCP protocol
    "MCPProtocol",
    "ChannelManager",
    "AsyncQueueTransport",
    
    # Storage
    "BoxDriver",
    "LocalDriver",
    "MultimodalLocalDriver",
    "MultimodalAssetManager",
    "ContentMetadata",
    "ImageMetadata",
    "AudioMetadata",
    "VideoMetadata",
    "DocumentMetadata",
    "MetadataExtractor",
    
    # Zeitgeist
    "ZeitgeistEngine",
    "OpinionAnalyzer", 
    "RoleQueryGenerator",
    "ZeitgeistConfig",
    
    # Audio processing
    "AudioProcessor",
    "AudioFormatDetector",
    "AudioValidator",
    "AudioConverter",
    
    # Utilities
    "generate_torch_id",
    "generate_asset_id",
    "TemplateLoader",
    "render_template",
]