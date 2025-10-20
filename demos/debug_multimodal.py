#!/usr/bin/env python3
"""
Debug script to test multimodal Ollama integration
"""
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from campfires.core.ollama import OllamaClient, OllamaConfig
from campfires.core.multimodal_ollama import MultimodalOllamaClient, MultimodalOllamaConfig

async def test_ollama_connection():
    """Test basic Ollama connection and model availability."""
    print("🔍 Testing Ollama Connection...")
    
    try:
        # Test basic Ollama connection
        ollama_config = OllamaConfig(
            base_url="http://localhost:11434",
            model="gemma3:latest",
            temperature=0.7,
            max_tokens=100
        )
        
        ollama_client = OllamaClient(ollama_config)
        
        # Test simple text generation
        print("📝 Testing text generation...")
        response = await ollama_client.generate("Hello, this is a test. Please respond briefly.")
        print(f"✅ Text generation successful: {response[:100]}...")
        
        # Test multimodal client
        print("🖼️ Testing multimodal client...")
        multimodal_config = MultimodalOllamaConfig(
            base_url="http://localhost:11434",
            model="gemma3:latest",
            vision_model="llava:latest",
            temperature=0.7,
            max_tokens=100
        )
        
        multimodal_client = MultimodalOllamaClient(multimodal_config)
        
        # Create a simple test image (SVG)
        test_svg = '''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="40" fill="blue"/>
            <text x="50" y="55" text-anchor="middle" fill="white">Test</text>
        </svg>'''
        
        image_data = test_svg.encode('utf-8')
        
        print("👁️ Testing image analysis...")
        analysis = await multimodal_client.analyze_image(image_data, "Describe this image in detail.")
        print(f"✅ Image analysis successful: {analysis[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🚀 Starting Ollama Debug Test")
    print("=" * 50)
    
    success = await test_ollama_connection()
    
    if success:
        print("\n✅ All tests passed! Ollama integration is working.")
    else:
        print("\n❌ Tests failed. Check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())