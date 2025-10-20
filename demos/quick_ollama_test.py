#!/usr/bin/env python3
"""
Quick Ollama Test

A simple script to test basic Ollama functionality.
This is useful for verifying that Ollama integration is working correctly.
"""

import sys
import asyncio
from pathlib import Path

# Add the parent directory to the path so we can import campfires
sys.path.append(str(Path(__file__).parent.parent))

from campfires.core.ollama import OllamaConfig, OllamaClient


async def test_ollama_connection():
    """Test basic Ollama connection and functionality."""
    print("🔥 Quick Ollama Test")
    print("=" * 40)
    
    # Create Ollama configuration
    config = OllamaConfig(
        base_url="http://localhost:11434",
        model="gemma3",
        temperature=0.7
    )
    
    # Create client
    client = OllamaClient(config)
    
    try:
        # Test 1: List available models
        print("1. Testing model listing...")
        models = await client.list_models()
        
        if models:
            print(f"✅ Found {len(models)} models:")
            for model in models:
                name = model.get('name', 'Unknown')
                print(f"   - {name}")
        else:
            print("❌ No models found. Please install models first:")
            print("   Run: ollama pull gemma3")
            return False
        
        # Test 2: Simple text generation
        print("\n2. Testing text generation...")
        prompt = "Hello! Please respond with a brief greeting."
        
        response = await client.generate(prompt)
        print(f"✅ Generation successful!")
        print(f"   Prompt: {prompt}")
        print(f"   Response: {response[:100]}{'...' if len(response) > 100 else ''}")
        
        # Test 3: Chat functionality
        print("\n3. Testing chat functionality...")
        messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        chat_response = await client.chat(messages)
        print(f"✅ Chat successful!")
        print(f"   Question: What is 2+2?")
        print(f"   Answer: {chat_response[:100]}{'...' if len(chat_response) > 100 else ''}")
        
        print("\n🎉 All tests passed! Ollama integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure Ollama is running: ollama serve")
        print("  2. Install a model: ollama pull gemma3")
        print("  3. Check Ollama is accessible at http://localhost:11434")
        return False


async def test_factory_integration():
    """Test Ollama integration through the factory system."""
    print("\n" + "=" * 40)
    print("Testing Factory Integration")
    print("=" * 40)
    
    try:
        from campfires.core.factory import DynamicCamper
        from campfires.core.orchestration import RoleRequirement
        from campfires.party_box.local_driver import LocalDriver
        
        # Create a local party box for testing
        party_box = LocalDriver("./test_party_box")
        
        # Create configuration for Ollama
        config = {
            'name': 'test_ollama_camper',
            'llm_provider': 'ollama',
            'ollama_base_url': 'http://localhost:11434',
            'ollama_model': 'gemma3:latest'
        }
        
        # Create a simple role requirement for testing
        role_requirement = RoleRequirement(
            role_name="Test Analyst",
            expertise_areas=["testing", "analysis"],
            required_capabilities=["text_processing", "problem_solving"],
            personality_traits=["analytical", "thorough"],
            context_sources=["documents", "reports"]
        )
        
        print("Creating DynamicCamper with Ollama configuration...")
        camper = DynamicCamper(party_box, config, role_requirement)
        
        print("✅ DynamicCamper created successfully with Ollama!")
        print(f"   Provider: {config['llm_provider']}")
        print(f"   Model: {config['ollama_model']}")
        print(f"   Base URL: {config['ollama_base_url']}")
        print(f"   Role: {role_requirement.role_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Factory integration test failed: {e}")
        return False


async def main():
    """Run all quick tests."""
    print("Starting Ollama integration tests...\n")
    
    # Test basic connection
    connection_ok = await test_ollama_connection()
    
    if connection_ok:
        # Test factory integration
        factory_ok = await test_factory_integration()
        
        if factory_ok:
            print("\n🎉 All integration tests passed!")
            print("Ollama is ready to use with Campfires!")
        else:
            print("\n⚠️  Basic connection works, but factory integration failed.")
    else:
        print("\n❌ Basic connection failed. Please fix Ollama setup first.")


if __name__ == "__main__":
    asyncio.run(main())