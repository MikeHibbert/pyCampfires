"""
Zeitgeist Configuration - Settings and configuration for the Zeitgeist module
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import os


@dataclass
class ZeitgeistConfig:
    """Configuration settings for Zeitgeist functionality."""
    
    # Search settings
    max_search_results: int = 10
    search_timeout: int = 30  # seconds
    cache_ttl: int = 3600  # 1 hour in seconds
    
    # Rate limiting
    max_searches_per_minute: int = 10
    max_searches_per_hour: int = 100
    
    # Opinion analysis settings
    min_confidence_threshold: float = 0.6
    max_opinion_length: int = 500
    
    # Role-specific settings
    role_query_templates: Dict[str, str] = field(default_factory=lambda: {
        'default': '{topic} opinions trends current',
        'expert': '{topic} expert analysis professional opinion',
        'academic': '{topic} research academic study findings',
        'journalist': '{topic} news reporting current events',
        'analyst': '{topic} market analysis trends data',
        'developer': '{topic} development best practices tools',
        'designer': '{topic} design trends user experience',
        'manager': '{topic} management strategy leadership',
    })
    
    # Search engine preferences
    preferred_search_engines: list = field(default_factory=lambda: ['duckduckgo'])
    
    # Content filtering
    filter_adult_content: bool = True
    filter_spam: bool = True
    min_content_quality_score: float = 0.5
    
    # Caching
    enable_caching: bool = True
    cache_directory: Optional[str] = None
    
    # Logging
    log_searches: bool = True
    log_level: str = 'INFO'
    
    def __post_init__(self):
        """Post-initialization setup."""
        if self.cache_directory is None:
            self.cache_directory = os.path.join(os.getcwd(), '.zeitgeist_cache')
    
    @classmethod
    def from_env(cls) -> 'ZeitgeistConfig':
        """Create configuration from environment variables."""
        return cls(
            max_search_results=int(os.getenv('ZEITGEIST_MAX_RESULTS', '10')),
            search_timeout=int(os.getenv('ZEITGEIST_TIMEOUT', '30')),
            cache_ttl=int(os.getenv('ZEITGEIST_CACHE_TTL', '3600')),
            max_searches_per_minute=int(os.getenv('ZEITGEIST_MAX_SEARCHES_PER_MIN', '10')),
            max_searches_per_hour=int(os.getenv('ZEITGEIST_MAX_SEARCHES_PER_HOUR', '100')),
            min_confidence_threshold=float(os.getenv('ZEITGEIST_MIN_CONFIDENCE', '0.6')),
            filter_adult_content=os.getenv('ZEITGEIST_FILTER_ADULT', 'true').lower() == 'true',
            filter_spam=os.getenv('ZEITGEIST_FILTER_SPAM', 'true').lower() == 'true',
            enable_caching=os.getenv('ZEITGEIST_ENABLE_CACHE', 'true').lower() == 'true',
            cache_directory=os.getenv('ZEITGEIST_CACHE_DIR'),
            log_searches=os.getenv('ZEITGEIST_LOG_SEARCHES', 'true').lower() == 'true',
            log_level=os.getenv('ZEITGEIST_LOG_LEVEL', 'INFO'),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'max_search_results': self.max_search_results,
            'search_timeout': self.search_timeout,
            'cache_ttl': self.cache_ttl,
            'max_searches_per_minute': self.max_searches_per_minute,
            'max_searches_per_hour': self.max_searches_per_hour,
            'min_confidence_threshold': self.min_confidence_threshold,
            'max_opinion_length': self.max_opinion_length,
            'role_query_templates': self.role_query_templates,
            'preferred_search_engines': self.preferred_search_engines,
            'filter_adult_content': self.filter_adult_content,
            'filter_spam': self.filter_spam,
            'min_content_quality_score': self.min_content_quality_score,
            'enable_caching': self.enable_caching,
            'cache_directory': self.cache_directory,
            'log_searches': self.log_searches,
            'log_level': self.log_level,
        }


# Default configuration instance
DEFAULT_CONFIG = ZeitgeistConfig()