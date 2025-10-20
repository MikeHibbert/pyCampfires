#!/usr/bin/env python3
"""Debug script to test Ollama connection and identify issues."""

import asyncio
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from campfires.core.ollama import OllamaClient, OllamaConfig
from campfires.core.multimodal_ollama import MultimodalOllamaClient, MultimodalOllamaConfig

async def debug_ollama():
    """Debug Ollama connection step by step."""
    print("🔍 Debugging Ollama connection...")
    
    try:
        # Step 1: Test basic configuration
        print("\n1️⃣ Creating Ollama configuration...")
        ollama_config = OllamaConfig(
            base_url="http://localhost:11434",
            model="gemma3",
            temperature=0.7,
            max_tokens=500
        )
        print(f"✅ Configuration created: {ollama_config.base_url}")
        
        # Step 2: Test client creation
        print("\n2️⃣ Creating Ollama client...")
        ollama_client = OllamaClient(ollama_config)
        print("✅ Client created successfully")
        
        # Step 3: Test model listing
        print("\n3️⃣ Testing model listing...")
        models = await ollama_client.list_models()
        print(f"✅ Models retrieved: {len(models) if models else 0}")
        if models:
            for model in models:
                print(f"   - {model.get('name', 'Unknown')}")
        
        # Step 4: Test text generation
        print("\n4️⃣ Testing text generation...")
        text_prompt = "Hello, how are you?"
        text_response = await ollama_client.generate(text_prompt)
        print(f"✅ Text generation successful!")
        print(f"   Response: {text_response[:100]}...")
        
        # Step 5: Test multimodal configuration
        print("\n5️⃣ Creating multimodal configuration...")
        multimodal_config = MultimodalOllamaConfig(
            base_url="http://localhost:11434",
            model="gemma3",
            vision_model="llava",
            temperature=0.7,
            max_tokens=500
        )
        print("✅ Multimodal configuration created")
        
        # Step 6: Test multimodal client
        print("\n6️⃣ Creating multimodal client...")
        multimodal_client = MultimodalOllamaClient(multimodal_config)
        print("✅ Multimodal client created successfully")
        
        print("\n🎉 All Ollama tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error occurred: {type(e).__name__}: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(debug_ollama())
    if result:
        print("\n✅ Ollama is working correctly!")
    else:
        print("\n❌ Ollama has issues that need to be resolved.")