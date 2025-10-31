#!/usr/bin/env python3
"""
Comprehensive Ollama Integration Demo

This demo showcases the Ollama integration capabilities of the Campfires framework,
including:
- Basic text generation
- Chat conversations
- Multimodal image analysis
- Model management
- MCP protocol integration

Prerequisites:
- Ollama server running on localhost:11434
- Models downloaded: llama2, llava (for vision)
"""

import sys
import asyncio
import base64
import json
from pathlib import Path
from campfires.party_box.local_driver import LocalDriver
from campfires.party_box.multimodal_local_driver import MultimodalLocalDriver

# Add the parent directory to the path so we can import campfires
sys.path.append(str(Path(__file__).parent.parent))

from campfires.core.ollama import OllamaConfig, OllamaClient, OllamaMCPClient
from campfires.core.multimodal_ollama import (
    MultimodalOllamaConfig, 
    MultimodalOllamaClient,
    OllamaMultimodalCamper
)
from campfires.mcp.ollama_protocol import OllamaMCPProtocol
from campfires.mcp.protocol import MCPMessage
from campfires.core.multimodal_torch import ContentType, MultimodalContent, Torch, MultimodalTorch


class OllamaDemo:
    """Comprehensive Ollama demonstration class."""
    
    def __init__(self):
        """Initialize the demo with Ollama configurations."""
        self.ollama_config = OllamaConfig(
            base_url="http://localhost:11434",
            model="gemma3",
            temperature=0.7,
            max_tokens=500
        )
        
        self.multimodal_config = MultimodalOllamaConfig(
            base_url="http://localhost:11434",
            vision_model="llava",
            model="gemma3"
        )
        
        self.ollama_client = OllamaClient(self.ollama_config)
        self.multimodal_client = MultimodalOllamaClient(self.multimodal_config)
        self.mcp_client = OllamaMCPClient(self.ollama_config)
        
        # MCP Protocol for advanced integration
        mcp_config = {
            'ollama_base_url': 'http://localhost:11434',
            'ollama_model': 'gemma3',
            'ollama_timeout': 30
        }
        mcp_ollama_config = OllamaConfig(
            base_url=mcp_config['ollama_base_url'],
            model=mcp_config['ollama_model'],
            timeout=mcp_config['ollama_timeout']
        )
        self.mcp_protocol = OllamaMCPProtocol(ollama_config=mcp_ollama_config)
    
    def print_section(self, title: str):
        """Print a formatted section header."""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    
    def print_subsection(self, title: str):
        """Print a formatted subsection header."""
        print(f"\n{'-'*40}")
        print(f"  {title}")
        print(f"{'-'*40}")
    
    async def demo_basic_text_generation(self):
        """Demonstrate basic text generation with Ollama."""
        self.print_section("Basic Text Generation")
        
        prompts = [
            "Explain quantum computing in simple terms.",
            "Write a short poem about artificial intelligence.",
            "What are the benefits of using local LLMs?"
        ]
        
        for i, prompt in enumerate(prompts, 1):
            self.print_subsection(f"Generation {i}")
            print(f"Prompt: {prompt}")
            
            try:
                response = await self.ollama_client.generate(prompt)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error: {e}")
    
    async def demo_chat_conversation(self):
        """Demonstrate chat conversation capabilities."""
        self.print_section("Chat Conversation")
        
        conversation = [
            {"role": "user", "content": "Hello! I'm interested in learning about machine learning."},
            {"role": "assistant", "content": "Hello! I'd be happy to help you learn about machine learning. What specific aspect would you like to explore?"},
            {"role": "user", "content": "Can you explain the difference between supervised and unsupervised learning?"}
        ]
        
        print("Conversation history:")
        for msg in conversation:
            print(f"{msg['role'].title()}: {msg['content']}")
        
        try:
            response = await self.ollama_client.chat(conversation)
            print(f"\nAssistant: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    async def demo_model_management(self):
        """Demonstrate model management capabilities."""
        self.print_section("Model Management")
        
        try:
            # List available models
            self.print_subsection("Available Models")
            models = await self.ollama_client.list_models()
            
            if models:
                for model in models:
                    name = model.get('name', 'Unknown')
                    size = model.get('size', 0)
                    size_mb = size / (1024 * 1024) if size else 0
                    print(f"- {name} ({size_mb:.1f} MB)")
            else:
                print("No models found. Please ensure Ollama is running and models are installed.")
            
            # Demonstrate model pulling (commented out to avoid long download)
            # self.print_subsection("Model Pulling")
            # print("Pulling model 'mistral' (this may take a while)...")
            # result = await self.ollama_client.pull_model("mistral")
            # print(f"Pull result: {result}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    def create_sample_image(self):
        """Create a sample base64 encoded PNG image for multimodal testing."""
        # A very small 1x1 transparent PNG image (base64 encoded)
        # This is a minimal valid PNG.
        return base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
    
    async def demo_multimodal_capabilities(self):
        """Demonstrate multimodal image analysis capabilities."""
        self.print_section("Multimodal Image Analysis")
        
        # Create sample image data
        image_data = self.create_sample_image()
        
        try:
            # Image description
            self.print_subsection("Image Description")
            description = await self.multimodal_client.describe_image(image_data, "sample.svg")
            print(f"Description: {description}")
            
            # Object identification
            self.print_subsection("Object Identification")
            objects = await self.multimodal_client.identify_objects(image_data, "sample.svg")
            print(f"Objects: {objects}")
            
            # Custom analysis
            self.print_subsection("Custom Analysis")
            analysis = await self.multimodal_client.analyze_image(
                image_data, 
                "sample.svg",
                "What colors are used in this image and what might it represent?"
            )
            print(f"Analysis: {analysis}")
            
            # Get multimodal statistics
            self.print_subsection("Multimodal Statistics")
            stats = await self.multimodal_client.get_multimodal_stats()
            print("Configuration:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"Error in multimodal demo: {e}")
            print("Note: Multimodal features require the 'llava' model to be installed.")
    
    async def demo_mcp_integration(self):
        """Demonstrate MCP (Model Context Protocol) integration."""
        self.print_section("MCP Protocol Integration")
        
        try:
            # Start MCP protocol
            await self.mcp_protocol.start()
            
            # Test LLM request through MCP
            self.print_subsection("MCP LLM Request")
            llm_message = MCPMessage(
                channel="ollama",
                message_type="llm_request",
                data={
                    "prompt": "Explain the benefits of using MCP for LLM integration.",
                    "model": "gemma3",
                    "temperature": 0.6,
                    "max_tokens": 200
                }
            )
            
            response = await self.mcp_protocol.process_message(llm_message)
            print(f"MCP Response: {response.data.get('response', 'No response')}")
            
            # Test chat request through MCP
            self.print_subsection("MCP Chat Request")
            chat_message = MCPMessage(
                channel="ollama",
                message_type="chat_request",
                data={
                    "messages": [
                        {"role": "user", "content": "What is the Model Context Protocol?"}
                    ],
                    "model": "gemma3",
                    "temperature": 0.5
                }
            )
            
            chat_response = await self.mcp_protocol.process_message(chat_message)
            print(f"MCP Chat Response: {chat_response.data.get('response', 'No response')}")
            
            # Test control operations
            self.print_subsection("MCP Control Operations")
            
            # Get available models
            models_message = MCPMessage(
                channel="ollama",
                message_type="control",
                data={"action": "get_available_models"}
            )
            
            models_response = await self.mcp_protocol.process_message(models_message)
            if models_response.data.get('success'):
                models = models_response.data.get('models', [])
                print(f"Available models via MCP: {len(models)} models")
                for model in models[:3]:  # Show first 3
                    print(f"  - {model.get('name', 'Unknown')}")
            
            # Stop MCP protocol
            await self.mcp_protocol.stop()
            
        except Exception as e:
            print(f"Error in MCP demo: {e}")
    
    async def demo_mcp_client_tools(self):
        """Demonstrate MCP client tools functionality."""
        self.print_section("MCP Client Tools")
        
        try:
            # Get available tools
            self.print_subsection("Available Tools")
            tools = await self.mcp_client.get_available_tools()

            print("Available MCP tools:")
            for tool_info in tools:
                print(f"  - {tool_info.get('name', 'Unknown')}: {tool_info.get('description', 'No description')}")
            
            # Test generate tool
            self.print_subsection("Generate Tool")
            generate_request = {
                "method": "tools/call",
                "params": {
                    "name": "ollama_generate",
                    "arguments": {
                        "prompt": "Write a haiku about programming.",
                        "temperature": 0.8
                    }
                }
            }
            
            result = await self.mcp_client.process_mcp_request(generate_request)
            if result.get('completion'):
                print(f"Generated text: {result.get('completion').get('text')}")
            else:
                print(f"Error: {result.get('error')}")
            
            # Test list models tool
            self.print_subsection("List Models Tool")
            list_request = {
                "method": "tools/call",
                "params": {
                    "name": "ollama_list_models",
                    "arguments": {}
                }
            }
            
            result = await self.mcp_client.process_mcp_request(list_request)
            if result.get('content'):
                models = json.loads(result.get('content')[0].get('text', '[]'))
                print(f"Found {len(models)} models via MCP client")
            else:
                print(f"Error: {result.get('error')}")
            
        except Exception as e:
            print(f"Error in MCP client demo: {e}")
    
    async def demo_multimodal_camper(self):
        """Demonstrate the OllamaMultimodalCamper integration."""
        self.print_section("Multimodal Camper Integration")
        
        try:
            # Create camper configuration
            camper_config = {
                'name': 'ollama_multimodal_camper',
                'ollama_base_url': 'http://localhost:11434',
                'ollama_vision_model': 'llava',
                'ollama_text_model': 'gemma3'
            }
            
            # Initialize camper
            party_box = MultimodalLocalDriver("demos/party_box")
            camper = OllamaMultimodalCamper(party_box, camper_config)
            
            # Create sample multimodal content
            image_data = self.create_sample_image()
            content = MultimodalContent(
                content_type=ContentType.IMAGE,
                data=image_data,
                metadata={"filename": "sample.png", "format": "png"},
                mime_type="image/png",
                encoding="base64"
            )
            multimodal_torch_obj = MultimodalTorch(
                contents=[content],
                primary_claim="Image analysis request",
                source_campfire="ollama_demo",
                channel="multimodal_analysis"
            )
            
            # Test camper capabilities
            self.print_subsection("Camper Capabilities")
            capabilities = await camper.get_capabilities()
            print("Camper capabilities:")
            for capability in capabilities:
                print(f"  - {capability}")
            
            # Test content processing
            self.print_subsection("Content Processing")
            result = await camper.process(
                multimodal_torch_obj
            )
            print(f"Processing result: {result}")
            
            # Test specific operations
            self.print_subsection("Specific Operations")
            
            # Description
            description = await camper.describe_image(multimodal_torch_obj.contents[0].data)
            print(f"Description: {description}")
            
            # Object identification
            objects = await camper.identify_objects(multimodal_torch_obj.contents[0].data)
            print(f"Objects: {objects}")
            
            # Get camper statistics
            self.print_subsection("Camper Statistics")
            stats = camper.get_stats()
            print("Camper statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
        except Exception as e:
            print(f"Error in multimodal camper demo: {e}")
    
    async def run_full_demo(self):
        """Run the complete Ollama demonstration."""
        print("Campfires Ollama Integration Demo")
        print("=" * 60)
        print("This demo showcases Ollama integration capabilities.")
        print("Make sure Ollama is running on localhost:11434 with models installed.")
        print("Recommended models: gemma3 (text), llava (vision)")
        
        try:
            # Check if Ollama is accessible
            models = await self.ollama_client.list_models()
            if not models:
                print("\n⚠️  Warning: No models found. Please install Ollama models first.")
                print("   Run: ollama pull gemma3")
                print("   Run: ollama pull llava")
                return
            
            print(f"Found {len(models)} Ollama models. Starting demo...")
            
            # Run all demo sections
            await self.demo_basic_text_generation()
            await self.demo_chat_conversation()
            await self.demo_model_management()
            await self.demo_multimodal_capabilities()
            await self.demo_mcp_integration()
            await self.demo_mcp_client_tools()
            await self.demo_multimodal_camper()
            
            self.print_section("Demo Complete!")
            print("All Ollama integration features demonstrated successfully!")
            print("\nKey capabilities showcased:")
            print("  ✓ Text generation and chat")
            print("  ✓ Model management")
            print("  ✓ Multimodal image analysis")
            print("  ✓ MCP protocol integration")
            print("  ✓ Camper framework integration")
            
        except Exception as e:
            print(f"\nDemo failed: {e}")
            print("\nTroubleshooting:")
            print("  1. Ensure Ollama is running: ollama serve")
            print("  2. Install required models: ollama pull gemma3 && ollama pull llava")
            print("  3. Check Ollama is accessible at http://localhost:11434")


async def main():
    """Main demo function."""
    demo = OllamaDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    asyncio.run(main())