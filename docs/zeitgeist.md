# Zeitgeist - Internet Knowledge for Campers

The Zeitgeist module enables campers to search the internet for current information, opinions, and beliefs relevant to their roles. This feature helps campers stay informed and make better decisions around the campfire by accessing real-time knowledge from the web.

## Overview

Zeitgeist provides campers with the ability to:
- Search the internet for role-specific information
- Analyze current opinions and trends
- Gather expert perspectives
- Discover trending tools and methods
- Stay updated with the latest developments in their domain

## Core Components

### ZeitgeistEngine
The main engine that coordinates web searches and opinion analysis.

```python
from campfires import ZeitgeistEngine, ZeitgeistConfig

# Create with default configuration
engine = ZeitgeistEngine()

# Create with custom configuration
config = ZeitgeistConfig(max_search_results=15, cache_ttl=7200)
engine = ZeitgeistEngine(config)
```

### OpinionAnalyzer
Analyzes search results to extract opinions, trends, and beliefs.

```python
from campfires import OpinionAnalyzer

analyzer = OpinionAnalyzer()
analysis = await analyzer.analyze_zeitgeist(search_results, "AI ethics")
```

### RoleQueryGenerator
Generates role-specific search queries for different types of campers.

```python
from campfires import RoleQueryGenerator

generator = RoleQueryGenerator()
queries = generator.generate_role_queries("machine learning", "developer")
```

### ZeitgeistConfig
Configuration settings for the Zeitgeist functionality.

```python
from campfires import ZeitgeistConfig

config = ZeitgeistConfig(
    max_search_results=10,
    search_timeout=30,
    cache_ttl=3600,
    enable_caching=True,
    filter_adult_content=True,
    log_searches=True
)
```

## Using Zeitgeist with Campers

### Basic Setup

```python
from campfires import Camper, LLMCamperMixin

class ResearchCamper(LLMCamperMixin, Camper):
    def __init__(self, name: str, role: str, **kwargs):
        super().__init__(name=name, **kwargs)
        self.set_role(role)
        self.enable_zeitgeist()
```

### Searching for Information

```python
# Create a camper
camper = ResearchCamper("Dr. Smith", "academic")

# Get general zeitgeist information
zeitgeist_info = await camper.get_zeitgeist("artificial intelligence")

# Get role-specific opinions
opinions = await camper.get_role_opinions("machine learning ethics")

# Get trending tools
tools = await camper.get_trending_tools("data science")

# Get expert perspectives
experts = await camper.get_expert_perspectives("quantum computing")
```

### Available Methods

#### `get_zeitgeist(topic: str, context: str = None)`
Get general zeitgeist information about a topic.

**Returns:**
```python
{
    'summary': 'Brief summary of current zeitgeist',
    'search_results': [...],
    'timestamp': '2024-01-15T10:30:00Z',
    'query_used': 'artificial intelligence trends current'
}
```

#### `get_role_opinions(topic: str)`
Get opinions specific to the camper's role.

**Returns:**
```python
{
    'opinions': [
        {
            'text': 'Opinion text...',
            'confidence': 0.85,
            'source': 'example.com',
            'sentiment': 'positive'
        }
    ],
    'role': 'academic',
    'topic': 'AI ethics'
}
```

#### `get_trending_tools(topic: str)`
Discover trending tools and methods related to a topic.

**Returns:**
```python
{
    'tools': [
        {
            'name': 'Tool Name',
            'description': 'Tool description...',
            'popularity_score': 0.9,
            'source': 'github.com'
        }
    ],
    'topic': 'data science'
}
```

#### `get_expert_perspectives(topic: str)`
Get expert perspectives and professional insights.

**Returns:**
```python
{
    'perspectives': [
        {
            'summary': 'Expert insight summary...',
            'expert_type': 'industry_professional',
            'confidence': 0.8,
            'source': 'expert-site.com'
        }
    ],
    'topic': 'quantum computing'
}
```

## Configuration Options

### Environment Variables

You can configure Zeitgeist using environment variables:

```bash
# Search settings
export ZEITGEIST_MAX_RESULTS=15
export ZEITGEIST_TIMEOUT=45
export ZEITGEIST_CACHE_TTL=7200

# Rate limiting
export ZEITGEIST_MAX_SEARCHES_PER_MIN=15
export ZEITGEIST_MAX_SEARCHES_PER_HOUR=150

# Content filtering
export ZEITGEIST_FILTER_ADULT=true
export ZEITGEIST_FILTER_SPAM=true
export ZEITGEIST_MIN_CONFIDENCE=0.7

# Caching
export ZEITGEIST_ENABLE_CACHE=true
export ZEITGEIST_CACHE_DIR=/path/to/cache

# Logging
export ZEITGEIST_LOG_SEARCHES=true
export ZEITGEIST_LOG_LEVEL=INFO
```

### Programmatic Configuration

```python
from campfires import ZeitgeistConfig

config = ZeitgeistConfig(
    max_search_results=20,
    search_timeout=60,
    cache_ttl=7200,
    max_searches_per_minute=20,
    max_searches_per_hour=200,
    min_confidence_threshold=0.7,
    filter_adult_content=True,
    filter_spam=True,
    enable_caching=True,
    cache_directory="/custom/cache/path",
    log_searches=True,
    log_level="DEBUG"
)

# Use with ZeitgeistEngine
engine = ZeitgeistEngine(config)

# Or set globally for campers
camper = Camper("TestCamper")
camper.enable_zeitgeist(config)
```

## Role-Specific Query Templates

Zeitgeist uses different query templates based on camper roles:

```python
role_templates = {
    'expert': '{topic} expert analysis professional opinion',
    'academic': '{topic} research academic study findings',
    'journalist': '{topic} news reporting current events',
    'analyst': '{topic} market analysis trends data',
    'developer': '{topic} development best practices tools',
    'designer': '{topic} design trends user experience',
    'manager': '{topic} management strategy leadership',
}
```

You can customize these templates in the configuration:

```python
custom_templates = {
    'scientist': '{topic} scientific research breakthrough discovery',
    'artist': '{topic} artistic trends creative expression',
}

config = ZeitgeistConfig(role_query_templates=custom_templates)
```

## Example: Multi-Role Research Session

```python
import asyncio
from campfires import Campfire, ZeitgeistConfig

async def research_session():
    # Create campfire
    campfire = Campfire("Research Campfire")
    
    # Create campers with different roles
    campers = [
        ResearchCamper("Dr. Sarah", "academic", campfire=campfire),
        ResearchCamper("Alex", "developer", campfire=campfire),
        ResearchCamper("Maya", "journalist", campfire=campfire)
    ]
    
    topic = "sustainable technology"
    
    # Each camper researches from their perspective
    for camper in campers:
        print(f"\n{camper.name} ({camper.get_role()}) researching {topic}:")
        
        # Get role-specific insights
        opinions = await camper.get_role_opinions(topic)
        tools = await camper.get_trending_tools(topic)
        
        print(f"Found {len(opinions.get('opinions', []))} opinions")
        print(f"Found {len(tools.get('tools', []))} trending tools")

# Run the session
asyncio.run(research_session())
```

## Best Practices

### 1. Role Assignment
Always assign appropriate roles to campers for better search results:

```python
camper.set_role("academic")  # For research-focused searches
camper.set_role("developer")  # For technical/programming searches
camper.set_role("journalist")  # For news and current events
```

### 2. Context Provision
Provide context to improve search relevance:

```python
camper.add_conversation_context("We're discussing AI ethics in healthcare")
zeitgeist_info = await camper.get_zeitgeist("machine learning bias")
```

### 3. Caching
Enable caching to improve performance and reduce API calls:

```python
config = ZeitgeistConfig(enable_caching=True, cache_ttl=3600)
```

### 4. Rate Limiting
Be mindful of rate limits to avoid being blocked:

```python
config = ZeitgeistConfig(
    max_searches_per_minute=10,
    max_searches_per_hour=100
)
```

### 5. Error Handling
Always handle potential errors:

```python
try:
    zeitgeist_info = await camper.get_zeitgeist("topic")
    if zeitgeist_info:
        # Process results
        pass
    else:
        print("No results found")
except Exception as e:
    print(f"Search failed: {e}")
```

## Dependencies

Zeitgeist requires the following additional packages:

```bash
pip install duckduckgo-search>=3.9.0
pip install beautifulsoup4>=4.12.0
pip install requests>=2.31.0
```

These are automatically included when you install Campfires with Zeitgeist support.

## Limitations

- Search results depend on internet connectivity
- Rate limiting may apply based on search provider
- Content quality varies by source
- Some topics may have limited or biased information
- Real-time information may have slight delays

## Privacy and Ethics

- Zeitgeist respects search provider terms of service
- No personal information is stored in search queries
- Adult content filtering is enabled by default
- Search logs can be disabled for privacy
- Results should be verified from authoritative sources

## Troubleshooting

### Common Issues

1. **No search results**: Check internet connectivity and search provider availability
2. **Rate limiting**: Reduce search frequency or increase delays between searches
3. **Import errors**: Ensure all dependencies are installed
4. **Cache issues**: Clear cache directory or disable caching temporarily

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

config = ZeitgeistConfig(log_level="DEBUG", log_searches=True)
```

This will provide detailed information about search queries, results, and any errors encountered.