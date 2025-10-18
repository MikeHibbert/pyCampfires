#!/usr/bin/env python3
"""
Zeitgeist Demo - Showcasing internet knowledge and opinion mining for campers

This demo shows how campers can use the Zeitgeist feature to search the internet
for current information, opinions, and trends relevant to their roles.
"""

import asyncio
import logging
from typing import Dict, Any

from campfires import (
    Campfire, 
    Camper, 
    LLMCamperMixin, 
    ZeitgeistEngine, 
    ZeitgeistConfig,
    OpenRouterConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZeitgeistCamper(LLMCamperMixin, Camper):
    """A camper that can use Zeitgeist to gather internet knowledge."""
    
    def __init__(self, name: str, role: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.set_role(role)
        self.enable_zeitgeist()
        
    async def research_topic(self, topic: str) -> Dict[str, Any]:
        """Research a topic using Zeitgeist and provide insights."""
        logger.info(f"{self.name} ({self.get_role()}) researching: {topic}")
        
        # Get zeitgeist information
        zeitgeist_info = await self.get_zeitgeist(topic)
        
        # Get role-specific opinions
        role_opinions = await self.get_role_opinions(topic)
        
        # Get trending tools/methods
        trending_tools = await self.get_trending_tools(topic)
        
        # Get expert perspectives
        expert_perspectives = await self.get_expert_perspectives(topic)
        
        # Compile research results
        research = {
            'topic': topic,
            'role': self.get_role(),
            'zeitgeist': zeitgeist_info,
            'role_opinions': role_opinions,
            'trending_tools': trending_tools,
            'expert_perspectives': expert_perspectives,
            'timestamp': zeitgeist_info.get('timestamp') if zeitgeist_info else None
        }
        
        return research
    
    async def share_insights(self, research: Dict[str, Any]) -> str:
        """Share research insights in a conversational format."""
        role = research['role']
        topic = research['topic']
        
        insights = f"🔍 **{self.name}** ({role}) shares insights on **{topic}**:\n\n"
        
        # Add zeitgeist summary
        if research.get('zeitgeist'):
            zeitgeist = research['zeitgeist']
            insights += f"📊 **Current Zeitgeist:**\n"
            insights += f"- {zeitgeist.get('summary', 'No summary available')}\n\n"
        
        # Add role-specific opinions
        if research.get('role_opinions'):
            opinions = research['role_opinions']
            insights += f"💭 **{role.title()} Perspective:**\n"
            for opinion in opinions.get('opinions', [])[:3]:  # Top 3 opinions
                insights += f"- {opinion.get('text', '')}\n"
            insights += "\n"
        
        # Add trending tools
        if research.get('trending_tools'):
            tools = research['trending_tools']
            insights += f"🔧 **Trending Tools/Methods:**\n"
            for tool in tools.get('tools', [])[:3]:  # Top 3 tools
                insights += f"- {tool.get('name', '')}: {tool.get('description', '')}\n"
            insights += "\n"
        
        # Add expert perspectives
        if research.get('expert_perspectives'):
            experts = research['expert_perspectives']
            insights += f"🎓 **Expert Insights:**\n"
            for expert in experts.get('perspectives', [])[:2]:  # Top 2 expert views
                insights += f"- {expert.get('summary', '')}\n"
        
        return insights


async def run_zeitgeist_demo():
    """Run the Zeitgeist demonstration."""
    print("🔥 Campfires Zeitgeist Demo - Internet Knowledge for Campers 🔥\n")
    
    # Create Zeitgeist configuration
    config = ZeitgeistConfig(
        max_search_results=8,
        search_timeout=30,
        enable_caching=True,
        log_searches=True
    )
    
    # Create campfire
    campfire = Campfire("Zeitgeist Research Campfire")
    
    # Create campers with different roles
    campers = [
        ZeitgeistCamper("Dr. Sarah", "academic", campfire=campfire),
        ZeitgeistCamper("Alex", "developer", campfire=campfire),
        ZeitgeistCamper("Maya", "journalist", campfire=campfire),
        ZeitgeistCamper("Jordan", "analyst", campfire=campfire)
    ]
    
    # Research topics
    topics = [
        "artificial intelligence ethics",
        "remote work productivity",
        "sustainable technology",
        "cybersecurity trends"
    ]
    
    print("🔍 Campers are researching current topics using Zeitgeist...\n")
    
    # Each camper researches a topic
    for i, camper in enumerate(campers):
        topic = topics[i % len(topics)]
        
        try:
            print(f"📚 {camper.name} is researching '{topic}'...")
            research = await camper.research_topic(topic)
            insights = await camper.share_insights(research)
            print(insights)
            print("-" * 80)
            
        except Exception as e:
            print(f"❌ Error during research by {camper.name}: {e}")
            print("-" * 80)
    
    print("\n🎯 Demo completed! Campers have gathered current internet knowledge.")
    print("💡 This shows how Zeitgeist helps campers stay informed about their domains.")


async def run_simple_zeitgeist_test():
    """Run a simple test of Zeitgeist functionality."""
    print("🧪 Simple Zeitgeist Test\n")
    
    try:
        # Create a simple camper
        camper = ZeitgeistCamper("TestCamper", "developer")
        
        # Test basic zeitgeist functionality
        print("Testing basic zeitgeist search...")
        result = await camper.get_zeitgeist("Python programming")
        
        if result:
            print(f"✅ Zeitgeist search successful!")
            print(f"📊 Summary: {result.get('summary', 'No summary')}")
            print(f"🔍 Found {len(result.get('search_results', []))} results")
        else:
            print("⚠️ No results returned from zeitgeist search")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.exception("Zeitgeist test error")


if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Full Zeitgeist Demo (requires internet)")
    print("2. Simple Test")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(run_zeitgeist_demo())
    elif choice == "2":
        asyncio.run(run_simple_zeitgeist_test())
    else:
        print("Invalid choice. Running simple test...")
        asyncio.run(run_simple_zeitgeist_test())