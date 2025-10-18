"""
RoleQueryGenerator - Generates role-specific search queries for different camper types
"""

import random
from typing import List, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class RoleQueryGenerator:
    """
    Generates contextually relevant search queries based on camper roles,
    current topics, and specific interests.
    """
    
    def __init__(self, config=None):
        """
        Initialize the query generator with role-specific templates.
        
        Args:
            config: ZeitgeistConfig instance for configuration settings
        """
        self.config = config
        self.role_templates = {
            'developer': {
                'general': [
                    "latest {topic} development trends 2024",
                    "best practices {topic} programming",
                    "popular {topic} frameworks tools",
                    "{topic} developer community opinions",
                    "emerging {topic} technologies"
                ],
                'tools': [
                    "best {topic} development tools",
                    "popular {topic} libraries frameworks",
                    "{topic} IDE recommendations",
                    "trending {topic} packages"
                ],
                'opinions': [
                    "developer opinions on {topic}",
                    "{topic} programming debates",
                    "what developers think about {topic}",
                    "{topic} community discussions"
                ]
            },
            'designer': {
                'general': [
                    "latest {topic} design trends 2024",
                    "current {topic} design principles",
                    "popular {topic} design tools",
                    "{topic} design community opinions",
                    "emerging {topic} design patterns"
                ],
                'tools': [
                    "best {topic} design software",
                    "popular {topic} design tools",
                    "{topic} design platforms",
                    "trending {topic} design apps"
                ],
                'opinions': [
                    "designer opinions on {topic}",
                    "{topic} design debates",
                    "what designers think about {topic}",
                    "{topic} design community discussions"
                ]
            },
            'manager': {
                'general': [
                    "latest {topic} management trends",
                    "current {topic} leadership practices",
                    "{topic} management strategies",
                    "{topic} team management opinions",
                    "emerging {topic} management methods"
                ],
                'tools': [
                    "best {topic} management tools",
                    "popular {topic} project management",
                    "{topic} productivity software",
                    "trending {topic} management platforms"
                ],
                'opinions': [
                    "manager opinions on {topic}",
                    "{topic} management debates",
                    "what managers think about {topic}",
                    "{topic} leadership discussions"
                ]
            },
            'analyst': {
                'general': [
                    "latest {topic} analysis trends",
                    "current {topic} analytical methods",
                    "{topic} data analysis practices",
                    "{topic} analyst community opinions",
                    "emerging {topic} analytics"
                ],
                'tools': [
                    "best {topic} analysis tools",
                    "popular {topic} analytics software",
                    "{topic} data visualization tools",
                    "trending {topic} analysis platforms"
                ],
                'opinions': [
                    "analyst opinions on {topic}",
                    "{topic} analysis debates",
                    "what analysts think about {topic}",
                    "{topic} analytics discussions"
                ]
            },
            'marketer': {
                'general': [
                    "latest {topic} marketing trends 2024",
                    "current {topic} marketing strategies",
                    "{topic} marketing best practices",
                    "{topic} marketing community opinions",
                    "emerging {topic} marketing tactics"
                ],
                'tools': [
                    "best {topic} marketing tools",
                    "popular {topic} marketing platforms",
                    "{topic} advertising software",
                    "trending {topic} marketing apps"
                ],
                'opinions': [
                    "marketer opinions on {topic}",
                    "{topic} marketing debates",
                    "what marketers think about {topic}",
                    "{topic} marketing discussions"
                ]
            },
            'researcher': {
                'general': [
                    "latest {topic} research findings",
                    "current {topic} research trends",
                    "{topic} academic studies",
                    "{topic} research community opinions",
                    "emerging {topic} research"
                ],
                'tools': [
                    "best {topic} research tools",
                    "popular {topic} research methods",
                    "{topic} analysis software",
                    "trending {topic} research platforms"
                ],
                'opinions': [
                    "researcher opinions on {topic}",
                    "{topic} research debates",
                    "what researchers think about {topic}",
                    "{topic} academic discussions"
                ]
            },
            'entrepreneur': {
                'general': [
                    "latest {topic} startup trends",
                    "current {topic} business opportunities",
                    "{topic} entrepreneurship insights",
                    "{topic} startup community opinions",
                    "emerging {topic} business models"
                ],
                'tools': [
                    "best {topic} business tools",
                    "popular {topic} startup platforms",
                    "{topic} entrepreneurship software",
                    "trending {topic} business apps"
                ],
                'opinions': [
                    "entrepreneur opinions on {topic}",
                    "{topic} startup debates",
                    "what entrepreneurs think about {topic}",
                    "{topic} business discussions"
                ]
            }
        }
        
        # Generic templates for unknown roles
        self.generic_templates = {
            'general': [
                "latest {topic} trends 2024",
                "current {topic} opinions",
                "popular {topic} discussions",
                "{topic} community thoughts",
                "emerging {topic} ideas"
            ],
            'tools': [
                "best {topic} tools",
                "popular {topic} software",
                "{topic} platforms",
                "trending {topic} apps"
            ],
            'opinions': [
                "opinions on {topic}",
                "{topic} debates",
                "what people think about {topic}",
                "{topic} discussions"
            ]
        }
        
        # Time-based modifiers
        self.time_modifiers = [
            "2024", "recent", "latest", "current", "new", "today", "this year"
        ]
        
        # Quality indicators
        self.quality_indicators = [
            "expert", "professional", "industry", "best", "top", "leading"
        ]
    
    def generate_queries(self, 
                        role: str, 
                        topic: str, 
                        context: str = "", 
                        query_types: List[str] = None,
                        max_queries: int = 5) -> List[str]:
        """
        Generate role-specific search queries.
        
        Args:
            role: The camper's role (e.g., 'developer', 'designer')
            topic: The main topic to search for
            context: Additional context for the search
            query_types: Types of queries to generate ('general', 'tools', 'opinions')
            max_queries: Maximum number of queries to generate
            
        Returns:
            List of generated search queries
        """
        if query_types is None:
            query_types = ['general', 'tools', 'opinions']
        
        queries = []
        
        # Get role-specific templates or fall back to generic
        role_lower = role.lower()
        templates = self.role_templates.get(role_lower, self.generic_templates)
        
        # Generate queries for each type
        for query_type in query_types:
            type_templates = templates.get(query_type, self.generic_templates.get(query_type, []))
            
            # Select random templates from this type
            selected_templates = random.sample(
                type_templates, 
                min(2, len(type_templates))
            )
            
            for template in selected_templates:
                query = template.format(topic=topic)
                
                # Add context if provided
                if context:
                    query = f"{query} {context}"
                
                queries.append(query)
        
        # Add some enhanced queries with quality indicators
        if len(queries) < max_queries:
            enhanced_queries = self._generate_enhanced_queries(role, topic, context)
            queries.extend(enhanced_queries)
        
        # Shuffle and limit
        random.shuffle(queries)
        return queries[:max_queries]
    
    def generate_contextual_queries(self, 
                                   role: str, 
                                   topic: str, 
                                   conversation_context: List[str],
                                   max_queries: int = 3) -> List[str]:
        """
        Generate queries based on conversation context.
        
        Args:
            role: The camper's role
            topic: Main topic
            conversation_context: Recent conversation topics/keywords
            max_queries: Maximum queries to generate
            
        Returns:
            List of contextual search queries
        """
        queries = []
        
        # Extract key terms from conversation context
        context_keywords = self._extract_keywords(conversation_context)
        
        # Generate queries that combine topic with context
        for keyword in context_keywords[:2]:
            query = f"{topic} {keyword} {role} perspective"
            queries.append(query)
        
        # Add a general query
        general_query = f"current {role} opinions on {topic}"
        queries.append(general_query)
        
        return queries[:max_queries]
    
    def generate_trending_queries(self, 
                                 role: str, 
                                 topic: str,
                                 max_queries: int = 3) -> List[str]:
        """
        Generate queries focused on trending information.
        
        Args:
            role: The camper's role
            topic: Main topic
            max_queries: Maximum queries to generate
            
        Returns:
            List of trending-focused queries
        """
        trending_templates = [
            "trending {topic} {role} discussions",
            "viral {topic} {role} content",
            "popular {topic} {role} opinions 2024",
            "hot {topic} debates {role}",
            "emerging {topic} trends {role}"
        ]
        
        queries = []
        for template in trending_templates[:max_queries]:
            query = template.format(topic=topic, role=role)
            queries.append(query)
        
        return queries
    
    def generate_expert_queries(self, 
                               role: str, 
                               topic: str,
                               max_queries: int = 3) -> List[str]:
        """
        Generate queries focused on expert opinions.
        
        Args:
            role: The camper's role
            topic: Main topic
            max_queries: Maximum queries to generate
            
        Returns:
            List of expert-focused queries
        """
        expert_templates = [
            "expert {role} opinion on {topic}",
            "leading {role} thoughts on {topic}",
            "professional {role} analysis {topic}",
            "industry {role} perspective {topic}",
            "{role} thought leaders {topic}"
        ]
        
        queries = []
        for template in expert_templates[:max_queries]:
            query = template.format(topic=topic, role=role)
            queries.append(query)
        
        return queries
    
    def _generate_enhanced_queries(self, role: str, topic: str, context: str) -> List[str]:
        """Generate enhanced queries with quality indicators."""
        enhanced = []
        
        # Add time-based queries
        time_modifier = random.choice(self.time_modifiers)
        enhanced.append(f"{time_modifier} {topic} {role} insights")
        
        # Add quality-focused queries
        quality_indicator = random.choice(self.quality_indicators)
        enhanced.append(f"{quality_indicator} {role} {topic} recommendations")
        
        return enhanced
    
    def _extract_keywords(self, conversation_context: List[str]) -> List[str]:
        """Extract relevant keywords from conversation context."""
        if not conversation_context:
            return []
        
        # Simple keyword extraction - in a real implementation,
        # this could use NLP techniques
        all_text = " ".join(conversation_context).lower()
        words = all_text.split()
        
        # Filter for meaningful words (length > 3, not common words)
        common_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
            'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use', 'way', 'will'
        }
        
        keywords = []
        for word in words:
            if len(word) > 3 and word not in common_words:
                keywords.append(word)
        
        # Return unique keywords, limited to most recent/relevant
        return list(dict.fromkeys(keywords))[:5]
    
    def get_role_specific_modifiers(self, role: str) -> List[str]:
        """Get role-specific search modifiers."""
        modifiers = {
            'developer': ['programming', 'coding', 'software', 'technical'],
            'designer': ['design', 'visual', 'creative', 'aesthetic'],
            'manager': ['management', 'leadership', 'strategy', 'team'],
            'analyst': ['analysis', 'data', 'metrics', 'insights'],
            'marketer': ['marketing', 'promotion', 'audience', 'campaign'],
            'researcher': ['research', 'study', 'academic', 'scientific'],
            'entrepreneur': ['business', 'startup', 'venture', 'innovation']
        }
        
        return modifiers.get(role.lower(), ['professional', 'industry', 'expert'])