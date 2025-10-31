"""
Ollama integration for Campfires.

This module provides integration with Ollama for running local LLM models,
including support for chat completions and model management.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from ..mcp.protocol import MCPProtocol
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
    model: str = "llama2"
    temperature: float = 0.7
    max_tokens: Optional[int] = 1000
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
        """Initialize the Ollama client."""
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats = {
            'requests_made': 0,
            'total_tokens': 0,
            'errors': 0,
            'models_loaded': 0
        }
        self.mcp_stats = {
            'messages_sent': 0,
            'mcp_errors': 0
        }

    async def start_session(self):
        """Starts the aiohttp client session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(base_url=self.config.base_url, timeout=aiohttp.ClientTimeout(total=self.config.timeout))
            logger.debug("Ollama aiohttp client session started.")

    async def close_session(self):
        """Closes the aiohttp client session."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("Ollama aiohttp client session closed.")
            self.session = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_session()

    async def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Ollama API."""
        if self.session is None or self.session.closed:
            await self.start_session()

        url = f"{self.config.base_url}/api/{endpoint}"

        try:
            self.stats['requests_made'] += 1

            async with self.session.post(url, json=data) as response:
                # Use raise_for_status to align with test mocks
                if hasattr(response, 'raise_for_status'):
                    response.raise_for_status()
                result = response.json()
                if asyncio.iscoroutine(result):
                    result = await result
                return result

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Ollama request failed: {e}")
            raise

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models in Ollama."""
        try:
            if self.session is None or self.session.closed:
                await self.start_session()
            url = f"{self.config.base_url}/api/tags"

            async with self.session.get(url) as response:
                if hasattr(response, 'raise_for_status'):
                    response.raise_for_status()
                result = response.json()
                if asyncio.iscoroutine(result):
                    result = await result
                return result.get('models', [])

        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Failed to list models: {e}")
            raise

    async def pull_model(self, model_name: str) -> Dict[str, Any]:
        """Pull/download a model to Ollama."""
        try:
            data = {"name": model_name}
            result = await self._make_request("pull", data)
            self.stats['models_loaded'] += 1
            return result

        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False

    async def generate(self, prompt: str, model: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Generate text completion using Ollama."""
        model_name = model or self.config.model

        data = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repeat_penalty": self.config.repeat_penalty,
                **self.config.options
            }
        }

        # Respect per-call override for max_tokens, else use config default
        effective_max = max_tokens if max_tokens is not None else self.config.max_tokens
        if effective_max:
            data["options"]["num_predict"] = effective_max

        try:
            response = await self._make_request("generate", data)

            # Update stats
            if 'eval_count' in response:
                self.stats['total_tokens'] += response['eval_count']

            return response.get('response', '')

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    async def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        """Chat completion using Ollama."""
        model_name = model or self.config.model

        data = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repeat_penalty": self.config.repeat_penalty,
                **self.config.options
            }
        }

        effective_max = max_tokens if max_tokens is not None else self.config.max_tokens
        if effective_max:
            data["options"]["num_predict"] = effective_max

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
    
    def __init__(self, config: OllamaConfig, mcp_protocol: Optional[MCPProtocol] = None):
        """Initialize MCP client."""
        self.config = config
        self.client = OllamaClient(config)
        self.mcp_protocol = mcp_protocol
        self.mcp_stats = {
            'mcp_requests': 0,
            'mcp_errors': 0,
            'tools_used': 0,
            'messages_sent': 0
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.client.start_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.close_session()

    async def start_session(self):
        await self.client.start_session()

    async def close_session(self):
        await self.client.close_session()

    async def send_mcp_message(self, message: str) -> str:
        """
        Send an MCP message via the protocol and update stats.
        """
        if self.mcp_protocol:
            self.mcp_stats['messages_sent'] += 1
            response = await self.mcp_protocol.send_message(message)
            return response
        else:
            raise RuntimeError("MCP protocol not configured for OllamaMCPClient")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    def get_available_tools(self) -> Dict[str, Dict[str, Any]]:
        """Get available tools for the Ollama MCP client (synchronous for tests)."""
        return {
            'ollama_generate': {
                'name': 'ollama_generate',
                'description': 'Generate text using Ollama model',
                'parameters': ['prompt', 'temperature', 'model']
            },
            'ollama_chat': {
                'name': 'ollama_chat',
                'description': 'Chat completion using Ollama model',
                'parameters': ['messages', 'model']
            },
            'ollama_list_models': {
                'name': 'ollama_list_models',
                'description': 'List available Ollama models',
                'parameters': []
            },
            'ollama_pull_model': {
                'name': 'ollama_pull_model',
                'description': 'Pull a model from Ollama registry',
                'parameters': ['model']
            }
        }

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a simple MCP-like request for tests."""
        try:
            self.mcp_stats['mcp_requests'] += 1
            tool = request.get('tool')
            params = request.get('parameters', {})

            if tool == 'ollama_generate':
                prompt = params.get('prompt', '')
                temperature = params.get('temperature')
                result = await self.client.generate(prompt, temperature=temperature)
                return {'success': True, 'response': result}

            elif tool == 'ollama_chat':
                messages = params.get('messages', [])
                result = await self.client.chat(messages)
                return {'success': True, 'response': result}

            elif tool == 'ollama_list_models':
                models = await self.client.list_models()
                return {'success': True, 'models': models}

            elif tool == 'ollama_pull_model':
                model = params.get('model', '')
                result = await self.client.pull_model(model)
                return {'success': True, 'status': result.get('status', 'success')}

            else:
                return {'success': False, 'error': f"Unknown tool: {tool}"}
        except Exception as e:
            self.mcp_stats['mcp_errors'] += 1
            return {'success': False, 'error': str(e)}

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