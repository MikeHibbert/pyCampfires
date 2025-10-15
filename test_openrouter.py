#!/usr/bin/env python3
"""
Test script to verify OpenRouter integration.
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from campfires.core.openrouter import OpenRouterClient, OpenRouterConfig

async def test_openrouter():
    """Test OpenRouter API integration."""
    print("🧪 Testing OpenRouter Integration")
    print("=" * 40)
    
    # Check if API key is loaded
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    # Create config and client
    config = OpenRouterConfig(
        default_model="openai/gpt-oss-20b:free"
    )
    
    client = OpenRouterClient(config)
    
    try:
        print("🔄 Testing simple completion...")
        
        async with client:
            response = await client.simple_completion(
                prompt="Analyze this text for crisis indicators: 'I'm feeling really hopeless today'",
                max_tokens=100
            )
            
        print(f"✅ Response received: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ OpenRouter test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_openrouter())
    if success:
        print("\n🎉 OpenRouter integration working!")
    else:
        print("\n💥 OpenRouter integration failed!")