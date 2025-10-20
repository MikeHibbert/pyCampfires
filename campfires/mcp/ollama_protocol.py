#!/usr/bin/env python3
"""
Ollama MCP Protocol implementation.
Extends the base MCPProtocol with Ollama LLM processing capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

from .protocol import MCPProtocol, Transport
from ..core.ollama import OllamaClient, OllamaConfig

logger = logging.getLogger(__name__)


class OllamaMCPProtocol(MCPProtocol):
    """
    MCP Protocol implementation with Ollama LLM processing.
    """
    
    def __init__(self, transport: Optional[Transport] = None, ollama_config: Optional[OllamaConfig] = None):
        """
        Initialize the Ollama MCP protocol.
        
        Args:
            transport: Transport layer for message delivery
            ollama_config: Ollama configuration
        """
        super().__init__(transport)
        self.ollama_config = ollama_config
        self.ollama_client: Optional[OllamaClient] = None
        
        if self.ollama_config:
            # Create client without MCP protocol to avoid circular dependency
            # This client will make direct API calls, not MCP calls
            self.ollama_client = OllamaClient(self.ollama_config)
    
    async def start(self) -> None:
        """Start the MCP protocol and Ollama client."""
        await super().start()
        
        if self.ollama_client:
            logger.info("Ollama client ready for MCP operations")
            
            # Check if Ollama server is accessible
            try:
                models = await self.ollama_client.list_models()
                logger.info(f"Ollama server accessible with {len(models)} models available")
            except Exception as e:
                logger.warning(f"Ollama server check failed: {e}")
    
    async def stop(self) -> None:
        """Stop the MCP protocol and Ollama client."""
        if self.ollama_client:
            logger.info("Ollama client session closed")
        
        await super().stop()
    
    async def _process_llm_request(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        """
        Process an LLM request through Ollama.
        
        Args:
            prompt: The prompt to process
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Temperature setting
            
        Returns:
            LLM response text
        """
        if not self.ollama_client:
            raise RuntimeError("Ollama client not configured")
        
        try:
            logger.debug(f"Processing LLM request with model: {model}")
            
            # Use the Ollama client to get completion
            response = await self.ollama_client.generate(
                model=model,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            )
            
            logger.debug(f"LLM response received: {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"Error processing LLM request: {e}")
            raise RuntimeError(f"LLM processing failed: {e}") from e
    
    async def _process_chat_request(self, messages: List[Dict], model: str, **kwargs) -> Dict:
        """
        Process a chat request through Ollama.
        
        Args:
            messages: Chat messages
            model: Model to use
            **kwargs: Additional parameters
            
        Returns:
            Chat response data
        """
        if not self.ollama_client:
            raise RuntimeError("Ollama client not configured")
        
        try:
            logger.debug(f"Processing chat request with model: {model}")
            
            # Extract parameters
            max_tokens = kwargs.get('max_tokens', 1000)
            temperature = kwargs.get('temperature', 0.7)
            
            # Use the Ollama client to get chat completion
            response = await self.ollama_client.chat(
                model=model,
                messages=messages,
                options={
                    "num_predict": max_tokens,
                    "temperature": temperature
                }
            )
            
            # Format response to match expected structure
            formatted_response = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 0,  # Ollama doesn't provide token counts
                    "completion_tokens": 0,
                    "total_tokens": 0
                },
                "model": model
            }
            
            logger.debug(f"Chat response received: {len(response)} characters")
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error processing chat request: {e}")
            raise RuntimeError(f"Chat processing failed: {e}") from e
    
    def set_ollama_config(self, config: OllamaConfig) -> None:
        """
        Update the Ollama configuration.
        
        Args:
            config: New Ollama configuration
        """
        self.ollama_config = config
        self.ollama_client = OllamaClient(config)
        logger.info("Ollama configuration updated")
    
    async def get_available_models(self) -> List[str]:
        """
        Get list of available models from Ollama.
        
        Returns:
            List of available model names
        """
        if not self.ollama_client:
            raise RuntimeError("Ollama client not configured")
        
        try:
            models = await self.ollama_client.list_models()
            return [model['name'] for model in models]
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        if not self.ollama_client:
            raise RuntimeError("Ollama client not configured")
        
        try:
            await self.ollama_client.pull_model(model_name)
            logger.info(f"Successfully pulled model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False