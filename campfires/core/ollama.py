"""
Ollama integration for Campfires.

This module provides integration with Ollama for running local LLM models,
including support for chat completions and model management.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import aiohttp
import time

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""
    
    # Connection settings
    base_url: str = "http://localhost:11434"
    timeout: int = 30
    
    # Model settings
    model: str = "gemma3"
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    
    # Advanced settings
    stream: bool = False
    keep_alive: str = "5m"
    
    # Custom options for model
    options: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.base_url.startswith(('http://', 'https://')):
            raise ValueError("base_url must start with http:// or https://")
        
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature must be between 0 and 2")
        
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, config: OllamaConfig):
        """Initialize Ollama client with configuration."""
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats = {
            'requests_made': 0,
            'total_tokens': 0,
            'errors': 0,
            'models_loaded': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure HTTP session is created."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Ollama API."""
        await self._ensure_session()
        
        url = f"{self.config.base_url}/api/{endpoint}"
        
        try:
            self.stats['requests_made'] += 1
            
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
                    
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Ollama request failed: {e}")
            raise
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models in Ollama."""
        try:
            await self._ensure_session()
            url = f"{self.config.base_url}/api/tags"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('models', [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to list models: {error_text}")
                    
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Failed to list models: {e}")
            raise
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull/download a model to Ollama."""
        try:
            data = {"name": model_name}
            await self._make_request("pull", data)
            self.stats['models_loaded'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    async def generate(self, prompt: str, model: Optional[str] = None) -> str:
        """Generate text completion using Ollama."""
        model_name = model or self.config.model
        
        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repeat_penalty": self.config.repeat_penalty,
                **self.config.options
            }
        }
        
        if self.config.max_tokens:
            data["options"]["num_predict"] = self.config.max_tokens
        
        try:
            response = await self._make_request("generate", data)
            
            # Update stats
            if 'eval_count' in response:
                self.stats['total_tokens'] += response['eval_count']
            
            return response.get('response', '')
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    async def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        """Chat completion using Ollama."""
        model_name = model or self.config.model
        
        data = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repeat_penalty": self.config.repeat_penalty,
                **self.config.options
            }
        }
        
        if self.config.max_tokens:
            data["options"]["num_predict"] = self.config.max_tokens
        
        try:
            response = await self._make_request("chat", data)
            
            # Update stats
            if 'eval_count' in response:
                self.stats['total_tokens'] += response['eval_count']
            
            message = response.get('message', {})
            return message.get('content', '')
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise
    
    async def check_model_exists(self, model_name: str) -> bool:
        """Check if a model exists in Ollama."""
        try:
            models = await self.list_models()
            model_names = [model.get('name', '') for model in models]
            return model_name in model_names
            
        except Exception as e:
            logger.error(f"Failed to check model existence: {e}")
            return False
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model."""
        try:
            data = {"name": model_name}
            return await self._make_request("show", data)
            
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return {
            **self.stats,
            'config': {
                'base_url': self.config.base_url,
                'model': self.config.model,
                'temperature': self.config.temperature
            }
        }


class OllamaMCPClient:
    """MCP-compatible client for Ollama integration."""
    
    def __init__(self, config: OllamaConfig):
        """Initialize MCP client."""
        self.config = config
        self.client = OllamaClient(config)
        self.mcp_stats = {
            'mcp_requests': 0,
            'mcp_errors': 0,
            'tools_used': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def process_mcp_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process MCP-formatted request."""
        try:
            self.mcp_stats['mcp_requests'] += 1
            
            method = request.get('method', '')
            params = request.get('params', {})
            
            if method == 'completion/complete':
                return await self._handle_completion(params)
            elif method == 'tools/list':
                return await self._handle_tools_list()
            elif method == 'tools/call':
                return await self._handle_tool_call(params)
            else:
                raise ValueError(f"Unsupported MCP method: {method}")
                
        except Exception as e:
            self.mcp_stats['mcp_errors'] += 1
            logger.error(f"MCP request failed: {e}")
            raise
    
    async def _handle_completion(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle completion request."""
        prompt = params.get('prompt', '')
        model = params.get('model')
        
        response = await self.client.generate(prompt, model)
        
        return {
            'completion': {
                'text': response,
                'model': model or self.config.model,
                'finish_reason': 'stop'
            }
        }
    
    async def _handle_tools_list(self) -> Dict[str, Any]:
        """Handle tools list request."""
        return {
            'tools': [
                {
                    'name': 'ollama_generate',
                    'description': 'Generate text using Ollama model',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'prompt': {'type': 'string'},
                            'model': {'type': 'string', 'optional': True}
                        },
                        'required': ['prompt']
                    }
                },
                {
                    'name': 'ollama_chat',
                    'description': 'Chat completion using Ollama model',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {
                            'messages': {'type': 'array'},
                            'model': {'type': 'string', 'optional': True}
                        },
                        'required': ['messages']
                    }
                },
                {
                    'name': 'ollama_list_models',
                    'description': 'List available Ollama models',
                    'inputSchema': {
                        'type': 'object',
                        'properties': {}
                    }
                }
            ]
        }
    
    async def _handle_tool_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call request."""
        self.mcp_stats['tools_used'] += 1
        
        tool_name = params.get('name', '')
        arguments = params.get('arguments', {})
        
        if tool_name == 'ollama_generate':
            result = await self.client.generate(
                arguments.get('prompt', ''),
                arguments.get('model')
            )
            return {'content': [{'type': 'text', 'text': result}]}
            
        elif tool_name == 'ollama_chat':
            result = await self.client.chat(
                arguments.get('messages', []),
                arguments.get('model')
            )
            return {'content': [{'type': 'text', 'text': result}]}
            
        elif tool_name == 'ollama_list_models':
            models = await self.client.list_models()
            model_list = [model.get('name', '') for model in models]
            return {'content': [{'type': 'text', 'text': json.dumps(model_list, indent=2)}]}
            
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def get_mcp_stats(self) -> Dict[str, Any]:
        """Get MCP client statistics."""
        return {
            **self.mcp_stats,
            'ollama_stats': self.client.get_stats()
        }