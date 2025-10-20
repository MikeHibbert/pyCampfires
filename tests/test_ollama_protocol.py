"""
Tests for Ollama MCP protocol functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from campfires.mcp.ollama_protocol import OllamaMCPProtocol
from campfires.core.ollama import OllamaConfig
from campfires.mcp.protocol import MCPMessage


class TestOllamaMCPProtocol:
    """Test OllamaMCPProtocol class."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
        return {
            'ollama_base_url': 'http://localhost:11434',
            'ollama_model': 'llama2',
            'ollama_timeout': 30
        }
    
    @pytest.fixture
    def protocol(self, config):
        """Create test protocol."""
        return OllamaMCPProtocol(config)
    
    def test_protocol_initialization(self, protocol):
        """Test protocol initialization."""
        assert protocol.name == "ollama"
        assert hasattr(protocol, 'ollama_client')
        assert protocol.ollama_client.config.base_url == "http://localhost:11434"
        assert protocol.ollama_client.config.model == "llama2"
    
    @pytest.mark.asyncio
    async def test_start_protocol(self, protocol):
        """Test starting the protocol."""
        with patch.object(protocol.ollama_client, 'list_models') as mock_list:
            mock_list.return_value = [{"name": "llama2"}]
            
            await protocol.start()
            
            assert protocol.is_running is True
            mock_list.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_protocol(self, protocol):
        """Test stopping the protocol."""
        protocol.is_running = True
        
        await protocol.stop()
        
        assert protocol.is_running is False
    
    @pytest.mark.asyncio
    async def test_process_llm_request(self, protocol):
        """Test processing LLM request."""
        message = MCPMessage(
            message_type="llm_request",
            data={
                "prompt": "Hello, world!",
                "temperature": 0.7,
                "max_tokens": 100
            }
        )
        
        with patch.object(protocol.ollama_client, 'generate') as mock_generate:
            mock_generate.return_value = "Hello! How can I help you?"
            
            response = await protocol._process_llm_request(message)
            
            assert response.message_type == "llm_response"
            assert response.data["response"] == "Hello! How can I help you?"
            assert response.data["success"] is True
            mock_generate.assert_called_once_with(
                "Hello, world!",
                temperature=0.7,
                max_tokens=100
            )
    
    @pytest.mark.asyncio
    async def test_process_chat_request(self, protocol):
        """Test processing chat request."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        message = MCPMessage(
            message_type="chat_request",
            data={
                "messages": messages,
                "temperature": 0.5
            }
        )
        
        with patch.object(protocol.ollama_client, 'chat') as mock_chat:
            mock_chat.return_value = "I'm doing well, thank you!"
            
            response = await protocol._process_chat_request(message)
            
            assert response.message_type == "chat_response"
            assert response.data["response"] == "I'm doing well, thank you!"
            assert response.data["success"] is True
            mock_chat.assert_called_once_with(
                messages,
                temperature=0.5
            )
    
    @pytest.mark.asyncio
    async def test_update_ollama_config(self, protocol):
        """Test updating Ollama configuration."""
        new_config = {
            'base_url': 'http://new-host:8080',
            'model': 'mistral',
            'timeout': 60
        }
        
        message = MCPMessage(
            message_type="control",
            data={
                "action": "update_ollama_config",
                "config": new_config
            }
        )
        
        response = await protocol._update_ollama_config(message)
        
        assert response.message_type == "control_response"
        assert response.data["success"] is True
        assert protocol.ollama_client.config.base_url == "http://new-host:8080"
        assert protocol.ollama_client.config.model == "mistral"
        assert protocol.ollama_client.config.timeout == 60
    
    @pytest.mark.asyncio
    async def test_get_available_models(self, protocol):
        """Test getting available models."""
        models = [
            {"name": "llama2", "size": 3800000000},
            {"name": "mistral", "size": 4100000000}
        ]
        
        message = MCPMessage(
            message_type="control",
            data={"action": "get_available_models"}
        )
        
        with patch.object(protocol.ollama_client, 'list_models') as mock_list:
            mock_list.return_value = models
            
            response = await protocol._get_available_models(message)
            
            assert response.message_type == "control_response"
            assert response.data["success"] is True
            assert response.data["models"] == models
            mock_list.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pull_model(self, protocol):
        """Test pulling a model."""
        message = MCPMessage(
            message_type="control",
            data={
                "action": "pull_model",
                "model": "mistral"
            }
        )
        
        with patch.object(protocol.ollama_client, 'pull_model') as mock_pull:
            mock_pull.return_value = {"status": "success"}
            
            response = await protocol._pull_model(message)
            
            assert response.message_type == "control_response"
            assert response.data["success"] is True
            assert response.data["status"] == "success"
            mock_pull.assert_called_once_with("mistral")
    
    @pytest.mark.asyncio
    async def test_process_message_llm_request(self, protocol):
        """Test processing message with LLM request type."""
        message = MCPMessage(
            message_type="llm_request",
            data={"prompt": "Test prompt"}
        )
        
        with patch.object(protocol, '_process_llm_request') as mock_process:
            mock_response = MCPMessage(
                message_type="llm_response",
                data={"response": "Test response"}
            )
            mock_process.return_value = mock_response
            
            response = await protocol.process_message(message)
            
            assert response == mock_response
            mock_process.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_process_message_chat_request(self, protocol):
        """Test processing message with chat request type."""
        message = MCPMessage(
            message_type="chat_request",
            data={"messages": []}
        )
        
        with patch.object(protocol, '_process_chat_request') as mock_process:
            mock_response = MCPMessage(
                message_type="chat_response",
                data={"response": "Chat response"}
            )
            mock_process.return_value = mock_response
            
            response = await protocol.process_message(message)
            
            assert response == mock_response
            mock_process.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_process_message_control_update_config(self, protocol):
        """Test processing control message for config update."""
        message = MCPMessage(
            message_type="control",
            data={
                "action": "update_ollama_config",
                "config": {"model": "new_model"}
            }
        )
        
        with patch.object(protocol, '_update_ollama_config') as mock_update:
            mock_response = MCPMessage(
                message_type="control_response",
                data={"success": True}
            )
            mock_update.return_value = mock_response
            
            response = await protocol.process_message(message)
            
            assert response == mock_response
            mock_update.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_process_message_control_get_models(self, protocol):
        """Test processing control message for getting models."""
        message = MCPMessage(
            message_type="control",
            data={"action": "get_available_models"}
        )
        
        with patch.object(protocol, '_get_available_models') as mock_get:
            mock_response = MCPMessage(
                message_type="control_response",
                data={"models": []}
            )
            mock_get.return_value = mock_response
            
            response = await protocol.process_message(message)
            
            assert response == mock_response
            mock_get.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_process_message_control_pull_model(self, protocol):
        """Test processing control message for pulling model."""
        message = MCPMessage(
            message_type="control",
            data={
                "action": "pull_model",
                "model": "mistral"
            }
        )
        
        with patch.object(protocol, '_pull_model') as mock_pull:
            mock_response = MCPMessage(
                message_type="control_response",
                data={"status": "success"}
            )
            mock_pull.return_value = mock_response
            
            response = await protocol.process_message(message)
            
            assert response == mock_response
            mock_pull.assert_called_once_with(message)
    
    @pytest.mark.asyncio
    async def test_process_message_unsupported_type(self, protocol):
        """Test processing message with unsupported type."""
        message = MCPMessage(
            message_type="unsupported",
            data={}
        )
        
        response = await protocol.process_message(message)
        
        assert response.message_type == "error"
        assert "Unsupported message type" in response.data["error"]
    
    @pytest.mark.asyncio
    async def test_process_message_unsupported_control_action(self, protocol):
        """Test processing control message with unsupported action."""
        message = MCPMessage(
            message_type="control",
            data={"action": "unsupported_action"}
        )
        
        response = await protocol.process_message(message)
        
        assert response.message_type == "error"
        assert "Unsupported control action" in response.data["error"]
    
    @pytest.mark.asyncio
    async def test_error_handling_llm_request(self, protocol):
        """Test error handling in LLM request processing."""
        message = MCPMessage(
            message_type="llm_request",
            data={"prompt": "Test prompt"}
        )
        
        with patch.object(protocol.ollama_client, 'generate') as mock_generate:
            mock_generate.side_effect = Exception("API error")
            
            response = await protocol._process_llm_request(message)
            
            assert response.message_type == "llm_response"
            assert response.data["success"] is False
            assert "API error" in response.data["error"]
    
    @pytest.mark.asyncio
    async def test_error_handling_chat_request(self, protocol):
        """Test error handling in chat request processing."""
        message = MCPMessage(
            message_type="chat_request",
            data={"messages": []}
        )
        
        with patch.object(protocol.ollama_client, 'chat') as mock_chat:
            mock_chat.side_effect = Exception("Chat error")
            
            response = await protocol._process_chat_request(message)
            
            assert response.message_type == "chat_response"
            assert response.data["success"] is False
            assert "Chat error" in response.data["error"]


if __name__ == "__main__":
    pytest.main([__file__])