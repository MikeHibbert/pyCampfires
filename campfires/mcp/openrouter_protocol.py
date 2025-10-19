#!/usr/bin/env python3
"""
OpenRouter MCP Protocol implementation.
Extends the base MCPProtocol with OpenRouter LLM processing capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

from .protocol import MCPProtocol, Transport
from ..core.openrouter import OpenRouterClient, OpenRouterConfig, ChatMessage, ChatResponse

logger = logging.getLogger(__name__)


class OpenRouterMCPProtocol(MCPProtocol):
    """
    MCP Protocol implementation with OpenRouter LLM processing.
    """
    
    def __init__(self, transport: Optional[Transport] = None, openrouter_config: Optional[OpenRouterConfig] = None):
        """
        Initialize the OpenRouter MCP protocol.
        
        Args:
            transport: Transport layer for message delivery
            openrouter_config: OpenRouter configuration
        """
        super().__init__(transport)
        self.openrouter_config = openrouter_config
        self.openrouter_client: Optional[OpenRouterClient] = None
        
        if self.openrouter_config:
            # Create client without MCP protocol to avoid circular dependency
            # This client will make direct API calls, not MCP calls
            self.openrouter_client = OpenRouterClient(self.openrouter_config)
    
    async def start(self) -> None:
        """Start the MCP protocol and OpenRouter client."""
        await super().start()
        
        if self.openrouter_client:
            logger.info("OpenRouter client ready for MCP operations")
    
    async def stop(self) -> None:
        """Stop the MCP protocol and OpenRouter client."""
        if self.openrouter_client:
            logger.info("OpenRouter client session closed")
        
        await super().stop()
    
    async def _process_llm_request(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        """
        Process an LLM request through OpenRouter.
        
        Args:
            prompt: The prompt to process
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Temperature setting
            
        Returns:
            LLM response text
        """
        if not self.openrouter_client:
            raise RuntimeError("OpenRouter client not configured")
        
        try:
            logger.debug(f"Processing LLM request with model: {model}")
            
            # Use the OpenRouter client to get completion
            response = await self.openrouter_client.simple_completion(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            logger.debug(f"LLM response received: {len(response)} characters")
            return response
            
        except Exception as e:
            logger.error(f"Error processing LLM request: {e}")
            raise RuntimeError(f"LLM processing failed: {e}") from e
    
    async def _process_chat_request(self, messages: List[Dict], model: str, **kwargs) -> Dict:
        """
        Process a chat request through OpenRouter.
        
        Args:
            messages: Chat messages
            model: Model to use
            **kwargs: Additional parameters
            
        Returns:
            Chat response data
        """
        if not self.openrouter_client:
            raise RuntimeError("OpenRouter client not configured")
        
        try:
            logger.debug(f"Processing chat request with model: {model}")
            
            # Convert message dicts to ChatMessage objects
            chat_messages = []
            for msg in messages:
                chat_messages.append(ChatMessage(
                    role=msg.get('role', 'user'),
                    content=msg.get('content', ''),
                    name=msg.get('name')
                ))
            
            # Extract additional parameters
            max_tokens = kwargs.get('max_tokens')
            temperature = kwargs.get('temperature')
            top_p = kwargs.get('top_p')
            frequency_penalty = kwargs.get('frequency_penalty')
            presence_penalty = kwargs.get('presence_penalty')
            stop = kwargs.get('stop')
            
            # Use the OpenRouter client to get chat completion
            response: ChatResponse = await self.openrouter_client.chat_completion(
                messages=chat_messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop
            )
            
            # Convert response to dict format
            response_dict = {
                'id': response.id,
                'object': response.object,
                'created': response.created,
                'model': response.model,
                'choices': response.choices,
                'usage': response.usage
            }
            
            logger.debug(f"Chat response received: {response.id}")
            return response_dict
            
        except Exception as e:
            logger.error(f"Error processing chat request: {e}")
            raise RuntimeError(f"Chat processing failed: {e}") from e
    
    def set_openrouter_config(self, config: OpenRouterConfig) -> None:
        """
        Set or update the OpenRouter configuration.
        
        Args:
            config: OpenRouter configuration
        """
        self.openrouter_config = config
        self.openrouter_client = OpenRouterClient(config)
        logger.info("OpenRouter configuration updated")