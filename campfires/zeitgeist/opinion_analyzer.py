"""
OpinionAnalyzer - Analyzes search results to extract opinions, trends, and beliefs
"""

import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


class OpinionAnalyzer:
    """
    Analyzes search results to extract opinions, trends, and beliefs.
    """
    
    def __init__(self, config=None):
        """
        Initialize the OpinionAnalyzer.
        
        Args:
            config: ZeitgeistConfig instance for configuration settings
        """
        self.config = config
        
        self.opinion_keywords = [
            'believe', 'think', 'opinion', 'view', 'perspective', 'stance',
            'position', 'argue', 'claim', 'suggest', 'recommend', 'prefer'
        ]
        
        self.trend_keywords = [
            'trending', 'popular', 'emerging', 'growing', 'increasing',
            'rising', 'hot', 'viral', 'latest', 'new', 'current'
        ]
        
        self.controversy_keywords = [
            'controversial', 'debate', 'dispute', 'disagree', 'conflict',
            'criticism', 'oppose', 'against', 'controversy', 'divided'
        ]
        
        self.expert_indicators = [
            'expert', 'professor', 'researcher', 'scientist', 'analyst', 'specialist',
            'authority', 'leader', 'pioneer', 'founder', 'CEO', 'director', 'PhD'
        ]
        
        self.tool_indicators = [
            'tool', 'software', 'platform', 'framework', 'library', 'application',
            'system', 'method', 'technique', 'approach', 'strategy', 'methodology'
        ]
    
    async def analyze_zeitgeist(self, 
                               search_results: List[Dict[str, Any]], 
                               role: str, 
                               context: str) -> Dict[str, Any]:
        """
        Analyze search results to extract zeitgeist information.
        
        Args:
            search_results: List of search result dictionaries
            role: The role this analysis is for
            context: Additional context for the analysis
            
        Returns:
            Zeitgeist analysis containing opinions, trends, etc.
        """
        if not search_results:
            return self._empty_analysis()
        
        try:
            # Extract text content from all results
            all_text = self._extract_text_content(search_results)
            
            # Analyze different aspects
            current_opinions = self._extract_opinions(all_text, search_results)
            trending_topics = self._extract_trending_topics(all_text, search_results)
            expert_perspectives = self._extract_expert_perspectives(search_results)
            controversies = self._extract_controversies(all_text, search_results)
            tools_and_methods = self._extract_tools_and_methods(all_text, search_results)
            
            return {
                'current_opinions': current_opinions,
                'trending_topics': trending_topics,
                'expert_perspectives': expert_perspectives,
                'controversies': controversies,
                'tools_and_methods': tools_and_methods,
                'sentiment_summary': self._analyze_overall_sentiment(all_text),
                'key_themes': self._extract_key_themes(all_text),
                'confidence_score': self._calculate_confidence(search_results)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing zeitgeist: {e}")
            return self._empty_analysis()
    
    def _extract_text_content(self, search_results: List[Dict[str, Any]]) -> str:
        """Extract and combine text content from search results."""
        text_parts = []
        for result in search_results:
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            text_parts.append(f"{title} {snippet}")
        
        return ' '.join(text_parts).lower()
    
    def _extract_opinions(self, text: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract current opinions from the text."""
        opinions = []
        
        # Look for opinion-indicating phrases
        opinion_patterns = [
            r'people think (?:that )?([^.!?]+)',
            r'experts believe (?:that )?([^.!?]+)',
            r'consensus is (?:that )?([^.!?]+)',
            r'general opinion (?:is )?(?:that )?([^.!?]+)',
            r'most (?:people|experts) (?:agree|believe) (?:that )?([^.!?]+)'
        ]
        
        for pattern in opinion_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                sentiment = self._analyze_sentiment(match)
                opinions.append({
                    'opinion': match.strip(),
                    'sentiment': sentiment,
                    'confidence': 0.7,
                    'source_type': 'general'
                })
        
        # Analyze sentiment of titles and snippets
        for result in results[:5]:  # Top 5 results
            title = result.get('title', '')
            snippet = result.get('snippet', '')
            
            if title:
                sentiment = self._analyze_sentiment(title)
                if sentiment != 'neutral':
                    opinions.append({
                        'opinion': title,
                        'sentiment': sentiment,
                        'confidence': 0.6,
                        'source_type': 'headline',
                        'url': result.get('url', '')
                    })
        
        return opinions[:10]  # Return top 10 opinions
    
    def _extract_trending_topics(self, text: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract trending topics and themes."""
        trending = []
        
        # Look for trending indicators
        for keyword in self.opinion_keywords['trending']:
            pattern = rf'{keyword}\s+([^.!?]+)'
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                trending.append({
                    'topic': match.strip(),
                    'trend_indicator': keyword,
                    'confidence': 0.8
                })
        
        # Extract frequently mentioned terms
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        word_freq = Counter(words)
        
        # Filter out common words and get top terms
        common_words = {'that', 'this', 'with', 'from', 'they', 'have', 'been', 'will', 'more', 'than'}
        for word, freq in word_freq.most_common(10):
            if word.lower() not in common_words and freq > 2:
                trending.append({
                    'topic': word,
                    'trend_indicator': 'frequency',
                    'confidence': min(0.9, freq / 10),
                    'mention_count': freq
                })
        
        return trending[:8]  # Return top 8 trending topics
    
    def _extract_expert_perspectives(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract perspectives that appear to come from experts."""
        expert_perspectives = []
        
        for result in results:
            title = result.get('title', '').lower()
            snippet = result.get('snippet', '').lower()
            url = result.get('url', '').lower()
            
            # Check if this looks like expert content
            expert_score = 0
            expert_type = 'unknown'
            
            for indicator in self.expert_indicators:
                if indicator in title or indicator in snippet or indicator in url:
                    expert_score += 1
                    expert_type = indicator
            
            # Check for academic or professional domains
            if any(domain in url for domain in ['.edu', '.org', 'research', 'institute']):
                expert_score += 2
                expert_type = 'academic'
            
            if expert_score > 0:
                expert_perspectives.append({
                    'perspective': result.get('title', ''),
                    'summary': result.get('snippet', ''),
                    'expert_type': expert_type,
                    'confidence': min(0.9, expert_score / 3),
                    'url': result.get('url', ''),
                    'source': result.get('source', '')
                })
        
        return expert_perspectives[:5]  # Return top 5 expert perspectives
    
    def _extract_controversies(self, text: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract current controversies and debates."""
        controversies = []
        
        for keyword in self.opinion_keywords['controversial']:
            pattern = rf'{keyword}[^.!?]*([^.!?]+)'
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                controversies.append({
                    'controversy': match.strip(),
                    'type': keyword,
                    'confidence': 0.7
                })
        
        # Look for opposing viewpoints
        opposition_patterns = [
            r'however[^.!?]*([^.!?]+)',
            r'but[^.!?]*([^.!?]+)',
            r'critics (?:argue|say|claim)[^.!?]*([^.!?]+)',
            r'opponents[^.!?]*([^.!?]+)'
        ]
        
        for pattern in opposition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                controversies.append({
                    'controversy': match.strip(),
                    'type': 'opposition',
                    'confidence': 0.6
                })
        
        return controversies[:5]  # Return top 5 controversies
    
    def _extract_tools_and_methods(self, text: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract popular tools and methodologies."""
        tools = []
        
        for indicator in self.tool_indicators:
            pattern = rf'(?:popular|best|top|leading|new)\s+{indicator}[^.!?]*([^.!?]+)'
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                tools.append({
                    'tool_or_method': match.strip(),
                    'category': indicator,
                    'confidence': 0.7
                })
        
        # Look for specific tool mentions
        tool_patterns = [
            r'using\s+([A-Z][a-zA-Z]+)',
            r'with\s+([A-Z][a-zA-Z]+)',
            r'([A-Z][a-zA-Z]+)\s+(?:tool|software|platform)'
        ]
        
        for pattern in tool_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 2:  # Filter out very short matches
                    tools.append({
                        'tool_or_method': match,
                        'category': 'software',
                        'confidence': 0.6
                    })
        
        return tools[:6]  # Return top 6 tools/methods
    
    def _analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of a text snippet."""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.opinion_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.opinion_keywords['negative'] if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def _analyze_overall_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze overall sentiment of all text."""
        sentiment = self._analyze_sentiment(text)
        
        positive_count = sum(1 for word in self.opinion_keywords['positive'] if word in text.lower())
        negative_count = sum(1 for word in self.opinion_keywords['negative'] if word in text.lower())
        total_sentiment_words = positive_count + negative_count
        
        return {
            'overall_sentiment': sentiment,
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'sentiment_strength': total_sentiment_words / max(1, len(text.split()) / 100)
        }
    
    def _extract_key_themes(self, text: str) -> List[str]:
        """Extract key themes from the text."""
        # Simple keyword extraction based on frequency
        words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())
        word_freq = Counter(words)
        
        # Filter common words
        common_words = {
            'about', 'would', 'there', 'could', 'should', 'which', 'their', 'other',
            'after', 'first', 'never', 'these', 'think', 'where', 'being', 'every'
        }
        
        themes = []
        for word, freq in word_freq.most_common(15):
            if word not in common_words and freq > 1:
                themes.append(word)
        
        return themes[:10]
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on result quality."""
        if not results:
            return 0.0
        
        # Base confidence on number of results and their diversity
        result_count = len(results)
        unique_sources = len(set(result.get('source', 'unknown') for result in results))
        
        confidence = min(1.0, (result_count / 10) * 0.7 + (unique_sources / result_count) * 0.3)
        return round(confidence, 2)
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            'current_opinions': [],
            'trending_topics': [],
            'expert_perspectives': [],
            'controversies': [],
            'tools_and_methods': [],
            'sentiment_summary': {
                'overall_sentiment': 'neutral',
                'positive_indicators': 0,
                'negative_indicators': 0,
                'sentiment_strength': 0
            },
            'key_themes': [],
            'confidence_score': 0.0
        }