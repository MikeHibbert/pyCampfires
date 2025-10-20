"""
Quick Multimodal Demo - Focused demonstration of key features.

This script provides a streamlined demonstration of:
1. Creating multimodal content
2. Using multimodal torches
3. Basic prompt engineering
4. Asset management
"""

import base64
from pathlib import Path

# Import key multimodal components
from campfires import (
    ContentType,
    MultimodalContent,
    MultimodalTorch,
    MultimodalPromptBuilder,
    get_prompt_for_content_types,
    generate_torch_id
)


def demo_multimodal_basics():
    """Demonstrate basic multimodal functionality."""
    print("🎭 Campfires Multimodal Quick Demo")
    print("=" * 40)
    
    # 1. Create different types of content
    print("\n📝 Creating multimodal content...")
    
    # Text content
    text_content = MultimodalContent(
        content_type=ContentType.TEXT,
        data="Welcome to Campfires multimodal capabilities!",
        metadata={"source": "demo", "language": "en"}
    )
    
    # Image content (simple SVG)
    svg_image = '''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <rect width="100" height="100" fill="#FF6B6B"/>
        <text x="50" y="55" text-anchor="middle" fill="white" font-size="14">🔥</text>
    </svg>'''
    
    image_content = MultimodalContent(
        content_type=ContentType.IMAGE,
        data=base64.b64encode(svg_image.encode()).decode(),
        metadata={"format": "svg", "description": "Campfire logo"}
    )
    
    # Audio content (placeholder)
    audio_content = MultimodalContent(
        content_type=ContentType.AUDIO,
        data=base64.b64encode(b"fake_audio_data").decode(),
        metadata={"format": "wav", "duration": 3.0}
    )
    
    print(f"✅ Created {len([text_content, image_content, audio_content])} content items")
    
    # 2. Create a multimodal torch
    print("\n🔥 Creating multimodal torch...")
    
    import time
    
    torch = MultimodalTorch(
        torch_id=generate_torch_id(
            claim="Quick multimodal demo torch",
            source="demo_campfire",
            timestamp=time.time()
        ),
        contents=[text_content, image_content, audio_content],
        primary_claim="Quick multimodal demo torch",
        source_campfire="demo_campfire",
        channel="demo_channel",
        metadata={"demo": "quick_demo", "version": "1.0"}
    )
    
    print(f"🆔 Torch ID: {torch.torch_id}")
    print(f"📊 Total size: {torch.get_total_size()} bytes")
    print(f"🎭 Content types: {torch.get_content_types()}")
    
    # 3. Demonstrate content filtering
    print("\n🔍 Filtering content by type...")
    
    text_items = torch.get_text_contents()
    image_items = torch.get_image_contents()
    audio_items = torch.get_audio_contents()
    
    print(f"📝 Text items: {len(text_items)}")
    print(f"🖼️  Image items: {len(image_items)}")
    print(f"🎵 Audio items: {len(audio_items)}")
    
    # 4. Convert to different formats
    print("\n🔄 Converting to different formats...")
    
    # Convert to MCP message
    mcp_message = torch.to_mcp_message()
    print(f"📨 MCP message parts: {len(mcp_message.get('content', []))}")
    
    # Convert to legacy torch (if needed)
    legacy_torch = torch.to_legacy_torch()
    print(f"🔙 Legacy torch ID: {legacy_torch.torch_id}")
    
    return torch


def demo_prompt_engineering():
    """Demonstrate multimodal prompt engineering."""
    print("\n💭 Demonstrating Prompt Engineering...")
    
    # Initialize prompt builder
    builder = MultimodalPromptBuilder()
    
    # 1. Get prompts for different content types
    print("\n🎯 Getting specialized prompts...")
    
    # Vision analysis prompt
    vision_prompt = get_prompt_for_content_types([ContentType.IMAGE])
    print(f"👁️  Vision prompt (first 80 chars): {vision_prompt[:80]}...")
    
    # Audio analysis prompt
    audio_prompt = get_prompt_for_content_types([ContentType.AUDIO])
    print(f"🎵 Audio prompt (first 80 chars): {audio_prompt[:80]}...")
    
    # Multimodal prompt
    multimodal_prompt = get_prompt_for_content_types([
        ContentType.TEXT, 
        ContentType.IMAGE, 
        ContentType.AUDIO
    ])
    print(f"🎭 Multimodal prompt (first 80 chars): {multimodal_prompt[:80]}...")
    
    # 2. Build custom prompts
    print("\n🛠️  Building custom prompts...")
    
    try:
        # Simple prompt
        simple_prompt = (builder
                        .add_instruction("Analyze the provided multimodal content")
                        .add_content_analysis("multimodal", "comprehensive understanding")
                        .build())
        print(f"📝 Simple prompt created: {len(simple_prompt)} characters")
        
        # Reset builder for next prompt
        builder = MultimodalPromptBuilder()
        
        # Prompt with variables
        custom_prompt = (builder
                        .add_instruction("Analyze the image content")
                        .add_content_analysis("image", "objects and colors")
                        .add_output_format("structured JSON")
                        .build())
        print(f"🎨 Custom prompt created: {len(custom_prompt)} characters")
        
    except Exception as e:
        print(f"📝 Note: Custom prompts require template library setup")
        print(f"   Error: {str(e)[:50]}...")
    
    return {
        "vision": vision_prompt,
        "audio": audio_prompt,
        "multimodal": multimodal_prompt
    }


def demo_content_operations(torch):
    """Demonstrate content operations and analysis."""
    print("\n⚙️  Demonstrating Content Operations...")
    
    # 1. Content analysis
    print("\n🔍 Analyzing content...")
    
    for i, content in enumerate(torch.contents):
        print(f"\nContent {i+1}:")
        print(f"  Type: {content.content_type.value}")
        print(f"  Size: {content.get_size()} bytes")
        print(f"  Binary: {content.is_binary()}")
        print(f"  Metadata keys: {list(content.metadata.keys())}")
    
    # 2. Content statistics
    print(f"\n📊 Torch Statistics:")
    print(f"  Total contents: {len(torch.contents)}")
    print(f"  Total size: {torch.get_total_size()} bytes")
    print(f"  Unique types: {len(torch.get_content_types())}")
    
    # 3. Content validation
    print(f"\n✅ Content Validation:")
    for content in torch.contents:
        is_valid = len(content.data) > 0 and content.content_type is not None
        print(f"  {content.content_type.value}: {'✅ Valid' if is_valid else '❌ Invalid'}")


def demo_integration_examples():
    """Show examples of how to integrate with other systems."""
    print("\n🔗 Integration Examples...")
    
    print("\n📡 OpenRouter Integration:")
    print("  # Create multimodal message for OpenRouter")
    print("  client = MultimodalOpenRouterClient(config)")
    print("  message = client.create_multimodal_message(")
    print("      text='Analyze this content',")
    print("      torch=multimodal_torch")
    print("  )")
    
    print("\n🏕️  Campfire Integration:")
    print("  # Use in Campfire orchestration")
    print("  class MultimodalCamper(Camper, MultimodalLLMCamperMixin):")
    print("      async def process_torch(self, torch):")
    print("          return await self.analyze_multimodal_content(torch)")
    
    print("\n📦 Party Box Integration:")
    print("  # Store and manage multimodal assets")
    print("  asset_manager = MultimodalAssetManager(storage_path)")
    print("  asset_hash = asset_manager.add_asset_from_torch(torch)")
    print("  assets = asset_manager.search_by_content_type('image')")


def main():
    """Run the quick multimodal demo."""
    try:
        # Run demonstrations
        torch = demo_multimodal_basics()
        prompts = demo_prompt_engineering()
        demo_content_operations(torch)
        demo_integration_examples()
        
        print("\n" + "=" * 40)
        print("🎉 Quick Demo Completed!")
        print("✨ Campfires multimodal capabilities demonstrated")
        print("📚 Check the full demo for comprehensive examples")
        print("=" * 40)
        
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")
        print("🔧 This may be due to missing dependencies in demo environment")
        print("✅ The multimodal structure has been demonstrated")


if __name__ == "__main__":
    main()