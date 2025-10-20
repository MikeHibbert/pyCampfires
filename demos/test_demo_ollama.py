#!/usr/bin/env python3
"""Test script to mimic the exact demo context and identify Ollama issues."""

import asyncio
import sys
import os

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from campfires.core.multimodal_torch import MultimodalTorch
from multimodal_demo import MultimodalDemo

async def test_demo_ollama():
    """Test the demo_ollama_integration method in the exact same context."""
    print("🔍 Testing demo_ollama_integration in demo context...")
    
    try:
        # Create demo instance
        demo = MultimodalDemo()
        
        # Load sample assets like in the demo
        sample_assets = {
            'image': 'multimodal_demo_assets/sample_image.svg',
            'audio': 'multimodal_demo_assets/sample_audio.wav',
            'document': 'multimodal_demo_assets/sample_document.md'
        }
        
        # Load content first
        contents = []
        for content_type, file_path in sample_assets.items():
            if os.path.exists(file_path):
                print(f"✅ Found {content_type}: {file_path}")
            else:
                print(f"⚠️ File not found: {file_path}")
        
        # Create torch instance like in the demo with proper initialization
        torch = MultimodalTorch(
            torch_id="test_torch_001",
            primary_claim="Testing Ollama integration in demo context",
            source_campfire="test_demo",
            channel="test_channel",
            contents=contents,
            metadata={
                "demo_name": "Ollama Test",
                "created_by": "Test Script",
                "content_count": len(contents)
            }
        )
        
        # Load content into torch
        for content_type, file_path in sample_assets.items():
            if os.path.exists(file_path):
                torch.load_content(file_path)
                print(f"✅ Loaded {content_type}: {file_path}")
        
        # Initialize demo results like in the demo
        demo.demo_results = {
            "ollama_models": []
        }
        
        # Call the demo_ollama_integration method
        print("\n🔥 Calling demo_ollama_integration...")
        result = await demo.demo_ollama_integration(torch)
        
        print(f"✅ demo_ollama_integration completed successfully!")
        print(f"   Result: {result}")
        print(f"   Ollama models in results: {len(demo.demo_results['ollama_models'])}")
        
        for model in demo.demo_results['ollama_models']:
            print(f"   - {model['name']}: {model['status']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error in demo_ollama_integration: {type(e).__name__}: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_demo_ollama())
    if result:
        print("\n✅ Demo Ollama integration is working correctly!")
    else:
        print("\n❌ Demo Ollama integration has issues.")