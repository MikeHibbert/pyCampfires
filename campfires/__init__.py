"""
Campfires Framework

A Python framework for orchestrating multimodal Large Language Models (LLMs) 
and tools to achieve emergent, task-driven behavior.
"""

__version__ = "0.2.0"
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

# MCP (Model Context Protocol) components
from .mcp.protocol import MCPProtocol, ChannelManager
from .mcp.transport import AsyncQueueTransport

# Party Box storage components
from .party_box.box_driver import BoxDriver
from .party_box.local_driver import LocalDriver

# Zeitgeist components
from .zeitgeist.zeitgeist_engine import ZeitgeistEngine
from .zeitgeist.opinion_analyzer import OpinionAnalyzer
from .zeitgeist.role_query_generator import RoleQueryGenerator
from .zeitgeist.config import ZeitgeistConfig

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
    
    # MCP protocol
    "MCPProtocol",
    "ChannelManager",
    "AsyncQueueTransport",
    
    # Storage
    "BoxDriver",
    "LocalDriver",
    
    # Zeitgeist
    "ZeitgeistEngine",
    "OpinionAnalyzer", 
    "RoleQueryGenerator",
    "ZeitgeistConfig",
    
    # Utilities
    "generate_torch_id",
    "generate_asset_id",
    "TemplateLoader",
    "render_template",
]