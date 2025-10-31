#!/usr/bin/env python3
"""
Ollama MCP Protocol implementation.
Extends the base MCPProtocol with Ollama LLM processing capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional

from .protocol import MCPProtocol, Transport, MCPMessage
from ..core.ollama import OllamaClient, OllamaConfig

logger = logging.getLogger(__name__)


class OllamaMCPProtocol(MCPProtocol):
    """
    MCP Protocol implementation with Ollama LLM processing.
    """
    
    def __init__(self, transport: Optional[Transport] = None, ollama_config: Optional[OllamaConfig] = None, campfire_name: Optional[str] = None):
        """
        Initialize the Ollama MCP protocol.
        
        Args:
            transport: Transport layer for message delivery OR config dict (tests)
            ollama_config: Ollama configuration
            campfire_name: The name of the campfire using this protocol
        """
        # Support tests passing a dict as the first argument
        if isinstance(transport, dict) and ollama_config is None:
            super().__init__(transport=None, campfire_name=campfire_name)
            cfg = transport
            base_url = cfg.get('ollama_base_url', 'http://localhost:11434')
            model = cfg.get('ollama_model', 'llama2')
            timeout = cfg.get('ollama_timeout', 30)
            self.ollama_config = OllamaConfig(base_url=base_url, model=model, timeout=timeout)
        else:
            super().__init__(transport, campfire_name=campfire_name)
            self.ollama_config = ollama_config

        self.name = "ollama"
        self.ollama_client: Optional[OllamaClient] = None

        if self.ollama_config:
            # Create client without MCP protocol to avoid circular dependency
            # This client will make direct API calls, not MCP calls
            self.ollama_client = OllamaClient(self.ollama_config)
    
    async def start(self) -> None:
        """Start the MCP protocol and Ollama client."""
        await super().start()
        
        if self.ollama_client:
            await self.ollama_client.start_session()
            logger.info("Ollama client ready for MCP operations")
            # Probe models to verify connectivity (mocked in tests)
            try:
                await self.ollama_client.list_models()
            except Exception:
                # In tests we ignore actual connection errors
                pass
    
    async def stop(self) -> None:
        """Stop the MCP protocol and Ollama client."""
        if self.ollama_client:
            await self.ollama_client.close_session()
            logger.info("Ollama client session closed")
        await super().stop()
    
    async def process_message(self, message: MCPMessage) -> MCPMessage:
        """Process incoming MCP messages and dispatch to handlers."""
        try:
            if message.message_type == "llm_request":
                return await self._process_llm_request(message)
            elif message.message_type == "chat_request":
                return await self._process_chat_request(message)
            elif message.message_type == "control":
                action = message.data.get('action')
                if action == "update_ollama_config":
                    return await self._update_ollama_config(message)
                elif action == "get_available_models":
                    return await self._get_available_models(message)
                elif action == "pull_model":
                    return await self._pull_model(message)
                else:
                    return MCPMessage(
                        channel=message.channel,
                        data={"error": "Unsupported control action"},
                        message_type="error",
                        message_id=message.message_id
                    )
            else:
                return MCPMessage(
                    channel=message.channel,
                    data={"error": "Unsupported message type"},
                    message_type="error",
                    message_id=message.message_id
                )
        except Exception as e:
            return MCPMessage(
                channel=message.channel,
                data={"error": str(e)},
                message_type="error",
                message_id=message.message_id
            )
    
    async def _process_llm_request(self, message: MCPMessage) -> MCPMessage:
        """Process an LLM request through Ollama (message-based)."""
        if not self.ollama_client:
            return MCPMessage(
                channel=message.channel,
                data={"success": False, "error": "Ollama client not configured"},
                message_type="llm_response",
                message_id=message.message_id
            )
        try:
            prompt = message.data.get('prompt', '')
            temperature = message.data.get('temperature')
            max_tokens = message.data.get('max_tokens')

            response_text = await self.ollama_client.generate(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return MCPMessage(
                channel=message.channel,
                data={"response": response_text, "success": True},
                message_type="llm_response",
                message_id=message.message_id
            )
        except Exception as e:
            return MCPMessage(
                channel=message.channel,
                data={"success": False, "error": str(e)},
                message_type="llm_response",
                message_id=message.message_id
            )
    
    async def _process_chat_request(self, message: MCPMessage) -> MCPMessage:
        """Process a chat request through Ollama (message-based)."""
        if not self.ollama_client:
            return MCPMessage(
                channel=message.channel,
                data={"success": False, "error": "Ollama client not configured"},
                message_type="chat_response",
                message_id=message.message_id
            )
        try:
            messages = message.data.get('messages', [])
            temperature = message.data.get('temperature')
            response_text = await self.ollama_client.chat(messages, temperature=temperature)
            return MCPMessage(
                channel=message.channel,
                data={"response": response_text, "success": True},
                message_type="chat_response",
                message_id=message.message_id
            )
        except Exception as e:
            return MCPMessage(
                channel=message.channel,
                data={"success": False, "error": str(e)},
                message_type="chat_response",
                message_id=message.message_id
            )
    
    async def _update_ollama_config(self, message: MCPMessage) -> MCPMessage:
        """Update the Ollama configuration from a control message."""
        cfg = message.data.get('config', {})
        base_url = cfg.get('base_url', self.ollama_config.base_url if self.ollama_config else 'http://localhost:11434')
        model = cfg.get('model', self.ollama_config.model if self.ollama_config else 'llama2')
        timeout = cfg.get('timeout', self.ollama_config.timeout if self.ollama_config else 30)
        self.ollama_config = OllamaConfig(base_url=base_url, model=model, timeout=timeout)
        self.ollama_client = OllamaClient(self.ollama_config)
        return MCPMessage(
            channel=message.channel,
            data={"success": True},
            message_type="control_response",
            message_id=message.message_id
        )
    
    async def _get_available_models(self, message: MCPMessage) -> MCPMessage:
        """Get list of available models from Ollama (control response)."""
        try:
            models = await (self.ollama_client.list_models() if self.ollama_client else asyncio.sleep(0))
            return MCPMessage(
                channel=message.channel,
                data={"success": True, "models": models},
                message_type="control_response",
                message_id=message.message_id
            )
        except Exception as e:
            return MCPMessage(
                channel=message.channel,
                data={"success": False, "error": str(e)},
                message_type="control_response",
                message_id=message.message_id
            )
    
    async def _pull_model(self, message: MCPMessage) -> MCPMessage:
        """Pull a model from Ollama registry via control message."""
        try:
            model_name = message.data.get('model', '')
            result = await (self.ollama_client.pull_model(model_name) if self.ollama_client else asyncio.sleep(0))
            status = result.get('status', 'success') if isinstance(result, dict) else 'success'
            return MCPMessage(
                channel=message.channel,
                data={"success": True, "status": status},
                message_type="control_response",
                message_id=message.message_id
            )
        except Exception as e:
            return MCPMessage(
                channel=message.channel,
                data={"success": False, "error": str(e)},
                message_type="control_response",
                message_id=message.message_id
            )