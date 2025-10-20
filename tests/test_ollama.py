"""
Tests for Ollama client functionality.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from campfires.core.ollama import OllamaConfig, OllamaClient, OllamaMCPClient


class TestOllamaConfig:
    """Test OllamaConfig class."""
    
    def test_default_config(self):
        """Test creating config with default values."""
        config = OllamaConfig()
        
        assert config.base_url == "http://localhost:11434"
        assert config.model == "llama2"
        assert config.timeout == 30
        assert config.stream is False
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
    
    def test_custom_config(self):
        """Test creating config with custom values."""
        config = OllamaConfig(
            base_url="http://custom:8080",
            model="mistral",
            timeout=60,
            stream=True,
            temperature=0.5,
            max_tokens=2000
        )
        
        assert config.base_url == "http://custom:8080"
        assert config.model == "mistral"
        assert config.timeout == 60
        assert config.stream is True
        assert config.temperature == 0.5
        assert config.max_tokens == 2000


class TestOllamaClient:
    """Test OllamaClient class."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
        return OllamaConfig(
            base_url="http://localhost:11434",
            model="llama2"
        )
    
    @pytest.fixture
    def client(self, config):
        """Create test client."""
        return OllamaClient(config)
    
    @pytest.mark.asyncio
    async def test_generate_success(self, client):
        """Test successful text generation."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "This is a test response",
            "done": True
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await client.generate("Test prompt")
            
            assert result == "This is a test response"
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_with_options(self, client):
        """Test generation with custom options."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "response": "Custom response",
            "done": True
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await client.generate(
                "Test prompt",
                temperature=0.5,
                max_tokens=500
            )
            
            assert result == "Custom response"
            # Verify the request payload includes custom options
            call_args = mock_post.call_args
            payload = call_args[1]['json']
            assert payload['options']['temperature'] == 0.5
            assert payload['options']['num_predict'] == 500
    
    @pytest.mark.asyncio
    async def test_chat_success(self, client):
        """Test successful chat completion."""
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you?"
            },
            "done": True
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await client.chat(messages)
            
            assert result == "Hello! How can I help you?"
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_models_success(self, client):
        """Test successful model listing."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama2", "size": 3800000000},
                {"name": "mistral", "size": 4100000000}
            ]
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await client.list_models()
            
            assert len(result) == 2
            assert result[0]["name"] == "llama2"
            assert result[1]["name"] == "mistral"
    
    @pytest.mark.asyncio
    async def test_pull_model_success(self, client):
        """Test successful model pulling."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "status": "success"
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.return_value.__aenter__.return_value = mock_response
            
            result = await client.pull_model("mistral")
            
            assert result["status"] == "success"
            mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_error_handling(self, client):
        """Test error handling in generate method."""
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = Exception("Connection error")
            
            with pytest.raises(Exception) as exc_info:
                await client.generate("Test prompt")
            
            assert "Connection error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_chat_error_handling(self, client):
        """Test error handling in chat method."""
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_post.side_effect = Exception("API error")
            
            with pytest.raises(Exception) as exc_info:
                await client.chat(messages)
            
            assert "API error" in str(exc_info.value)


class TestOllamaMCPClient:
    """Test OllamaMCPClient class."""
    
    @pytest.fixture
    def config(self):
        """Create test config."""
        return OllamaConfig(model="llama2")
    
    @pytest.fixture
    def mcp_client(self, config):
        """Create test MCP client."""
        return OllamaMCPClient(config)
    
    def test_get_available_tools(self, mcp_client):
        """Test getting available tools."""
        tools = mcp_client.get_available_tools()
        
        expected_tools = [
            "ollama_generate",
            "ollama_chat", 
            "ollama_list_models",
            "ollama_pull_model"
        ]
        
        for tool in expected_tools:
            assert tool in tools
        
        # Check tool structure
        generate_tool = tools["ollama_generate"]
        assert "name" in generate_tool
        assert "description" in generate_tool
        assert "parameters" in generate_tool
    
    @pytest.mark.asyncio
    async def test_process_request_generate(self, mcp_client):
        """Test processing generate request."""
        request = {
            "tool": "ollama_generate",
            "parameters": {
                "prompt": "Test prompt",
                "temperature": 0.5
            }
        }
        
        with patch.object(mcp_client.client, 'generate') as mock_generate:
            mock_generate.return_value = "Generated response"
            
            result = await mcp_client.process_request(request)
            
            assert result["success"] is True
            assert result["response"] == "Generated response"
            mock_generate.assert_called_once_with(
                "Test prompt",
                temperature=0.5
            )
    
    @pytest.mark.asyncio
    async def test_process_request_chat(self, mcp_client):
        """Test processing chat request."""
        messages = [{"role": "user", "content": "Hello"}]
        request = {
            "tool": "ollama_chat",
            "parameters": {
                "messages": messages
            }
        }
        
        with patch.object(mcp_client.client, 'chat') as mock_chat:
            mock_chat.return_value = "Chat response"
            
            result = await mcp_client.process_request(request)
            
            assert result["success"] is True
            assert result["response"] == "Chat response"
            mock_chat.assert_called_once_with(messages)
    
    @pytest.mark.asyncio
    async def test_process_request_list_models(self, mcp_client):
        """Test processing list models request."""
        request = {
            "tool": "ollama_list_models",
            "parameters": {}
        }
        
        models = [{"name": "llama2"}, {"name": "mistral"}]
        
        with patch.object(mcp_client.client, 'list_models') as mock_list:
            mock_list.return_value = models
            
            result = await mcp_client.process_request(request)
            
            assert result["success"] is True
            assert result["models"] == models
    
    @pytest.mark.asyncio
    async def test_process_request_pull_model(self, mcp_client):
        """Test processing pull model request."""
        request = {
            "tool": "ollama_pull_model",
            "parameters": {
                "model": "mistral"
            }
        }
        
        with patch.object(mcp_client.client, 'pull_model') as mock_pull:
            mock_pull.return_value = {"status": "success"}
            
            result = await mcp_client.process_request(request)
            
            assert result["success"] is True
            assert result["status"] == "success"
            mock_pull.assert_called_once_with("mistral")
    
    @pytest.mark.asyncio
    async def test_process_request_invalid_tool(self, mcp_client):
        """Test processing request with invalid tool."""
        request = {
            "tool": "invalid_tool",
            "parameters": {}
        }
        
        result = await mcp_client.process_request(request)
        
        assert result["success"] is False
        assert "Unknown tool" in result["error"]
    
    @pytest.mark.asyncio
    async def test_process_request_error_handling(self, mcp_client):
        """Test error handling in process_request."""
        request = {
            "tool": "ollama_generate",
            "parameters": {
                "prompt": "Test prompt"
            }
        }
        
        with patch.object(mcp_client.client, 'generate') as mock_generate:
            mock_generate.side_effect = Exception("API error")
            
            result = await mcp_client.process_request(request)
            
            assert result["success"] is False
            assert "API error" in result["error"]


if __name__ == "__main__":
    pytest.main([__file__])