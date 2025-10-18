"""
Zeitgeist Module - Internet knowledge and opinion mining for campers

This module enables campers to search the internet for current information,
opinions, and beliefs relevant to their roles, helping them stay informed
and make better decisions around the campfire.
"""

from .zeitgeist_engine import ZeitgeistEngine
from .opinion_analyzer import OpinionAnalyzer
from .role_query_generator import RoleQueryGenerator
from .config import ZeitgeistConfig, DEFAULT_CONFIG

__all__ = ['ZeitgeistEngine', 'OpinionAnalyzer', 'RoleQueryGenerator', 'ZeitgeistConfig', 'DEFAULT_CONFIG']