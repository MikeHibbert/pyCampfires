"""
ZeitgeistEngine - Core web search and knowledge gathering engine
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None
    logger.warning("duckduckgo-search not installed. Web search functionality will be limited.")

from .opinion_analyzer import OpinionAnalyzer
from .role_query_generator import RoleQueryGenerator
from .config import ZeitgeistConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class ZeitgeistEngine:
    """
    Core engine for gathering zeitgeist information from the internet.
    Coordinates web searches and opinion analysis.
    """
    
    def __init__(self, config: Optional[ZeitgeistConfig] = None):
        """
        Initialize the ZeitgeistEngine.
        
        Args:
            config: ZeitgeistConfig instance, uses DEFAULT_CONFIG if None
        """
        self.config = config or DEFAULT_CONFIG
        self.search_cache: Dict[str, Dict[str, Any]] = {}
        
        # Initialize search configuration
        self.search_provider = "duckduckgo"  # Default search provider
        self.cache = {}  # Search result cache
        self.cache_duration = timedelta(hours=1)  # Cache duration
        self.max_results = 10  # Maximum search results per query
        
        # Initialize components
        self.opinion_analyzer = OpinionAnalyzer(self.config)
        self.query_generator = RoleQueryGenerator(self.config)
        
    async def gather_zeitgeist(self, 
                              role: str, 
                              context: str = "",
                              specific_topics: List[str] = None) -> Dict[str, Any]:
        """
        Gather current zeitgeist information for a specific role.
        
        Args:
            role: The camper's role (e.g., "data_scientist", "journalist", "teacher")
            context: Additional context about the current discussion/task
            specific_topics: Specific topics to focus the search on
            
        Returns:
            Dictionary containing zeitgeist information including:
            - current_opinions: Current popular opinions in the field
            - trending_topics: What's trending in this role's domain
            - expert_perspectives: Perspectives from recognized experts
            - controversies: Current debates and controversies
            - tools_and_methods: Popular tools and methodologies
        """
        try:
            # Generate role-specific search queries
            queries = self.query_generator.generate_queries(role, context, specific_topics)
            
            # Perform searches
            search_results = []
            for query in queries:
                results = await self._search_web(query)
                search_results.extend(results)
            
            # Analyze opinions and extract zeitgeist
            zeitgeist = await self.opinion_analyzer.analyze_zeitgeist(
                search_results, role, context
            )
            
            # Add metadata
            zeitgeist['gathered_at'] = datetime.now().isoformat()
            zeitgeist['role'] = role
            zeitgeist['context'] = context
            zeitgeist['source_count'] = len(search_results)
            
            return zeitgeist
            
        except Exception as e:
            logger.error(f"Error gathering zeitgeist for role {role}: {e}")
            return self._empty_zeitgeist(role, str(e))
    
    async def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a web search using the configured provider.
        
        Args:
            query: The search query
            
        Returns:
            List of search results with title, url, snippet, and date
        """
        # Check cache first
        cache_key = f"{self.search_provider}:{query}"
        if cache_key in self.cache:
            cached_result, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                return cached_result
        
        try:
            if self.search_provider == "duckduckgo":
                results = await self._search_duckduckgo(query)
            else:
                logger.warning(f"Unsupported search provider: {self.search_provider}")
                results = []
            
            # Cache the results
            self.cache[cache_key] = (results, datetime.now())
            return results
            
        except Exception as e:
            logger.error(f"Error searching for '{query}': {e}")
            return []
    
    async def _search_duckduckgo(self, query: str) -> List[Dict[str, Any]]:
        """
        Search using DuckDuckGo.
        """
        if DDGS is None:
            logger.warning("DuckDuckGo search not available. Returning empty results.")
            return []
        
        try:
            # Perform the search using DuckDuckGo
            with DDGS() as ddgs:
                results = []
                search_results = ddgs.text(
                    keywords=query,
                    max_results=self.max_results,
                    safesearch='moderate'
                )
                
                for result in search_results:
                    # Extract domain from URL for source
                    url = result.get('href', '')
                    source = url.split('/')[2] if len(url.split('/')) > 2 else 'unknown'
                    
                    results.append({
                        'title': result.get('title', ''),
                        'snippet': result.get('body', ''),
                        'url': url,
                        'date': datetime.now().isoformat(),
                        'source': source
                    })
                
                logger.info(f"Found {len(results)} results for query: {query}")
                return results
                
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    def _empty_zeitgeist(self, role: str, error: str = "") -> Dict[str, Any]:
        """
        Return an empty zeitgeist structure when search fails.
        """
        return {
            'current_opinions': [],
            'trending_topics': [],
            'expert_perspectives': [],
            'controversies': [],
            'tools_and_methods': [],
            'gathered_at': datetime.now().isoformat(),
            'role': role,
            'context': '',
            'source_count': 0,
            'error': error
        }
    
    async def get_zeitgeist(self, 
                           topic: str,
                           role: str = "general", 
                           context: str = "",
                           search_types: List[str] = None) -> Dict[str, Any]:
        """
        Get zeitgeist information for a specific topic and role.
        This method provides the interface expected by the Camper class.
        
        Args:
            topic: The topic to search for
            role: The camper's role (defaults to "general")
            context: Additional context for the search
            search_types: Types of searches to perform (currently ignored)
            
        Returns:
            Dictionary containing zeitgeist information
        """
        # Use the topic as specific topics for the search
        specific_topics = [topic] if topic else None
        return await self.gather_zeitgeist(role, context, specific_topics)

    async def get_role_opinions(self, topic: str, role: str = "general") -> Dict[str, Any]:
        """
        Get role-specific opinions about a topic.
        
        Args:
            topic: The topic to get opinions about
            role: The role to get opinions for
            
        Returns:
            Dictionary containing role-specific opinions
        """
        context = f"opinions perspectives views {role}"
        return await self.gather_zeitgeist(role, context, [topic])
    
    async def get_trending_tools(self, topic: str, role: str = "general") -> Dict[str, Any]:
        """
        Get trending tools and methods for a topic and role.
        
        Args:
            topic: The topic to get trending tools for
            role: The role context for tools
            
        Returns:
            Dictionary containing trending tools information
        """
        context = f"tools methods solutions software {role}"
        return await self.gather_zeitgeist(role, context, [f"{topic} tools"])

    async def get_role_insights(self, role: str) -> Dict[str, Any]:
        """
        Get general insights about a role without specific context.
        
        Args:
            role: The role to get insights for
            
        Returns:
            General zeitgeist information for the role
        """
        return await self.gather_zeitgeist(role, "general role insights")
    
    async def refresh_cache(self):
        """
        Clear the search cache to force fresh results.
        """
        self.cache.clear()
        logger.info("Zeitgeist cache cleared")